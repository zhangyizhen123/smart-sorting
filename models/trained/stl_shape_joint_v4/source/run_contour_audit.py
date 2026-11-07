"""Four fixed inference runs; no training, model replacement, or label edits."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from contour_quality import regularize_masks, pairwise_geometry
from evaluate_focused import score_records, match_instances, digest, masks_from_truth
from prepare_real_training import verify
from shape_refiner import read_polygons

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'vision/configs/contour_audit_v1.json'


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8')
    temp.replace(path)


def metric_record(row, predictions, masks, truths, width, height, truth_classes):
    return {'image':row['image'],'domain':row['domain'],'category':row.get('category'),
            'truth_classes':truth_classes,'predictions':predictions,
            **pairwise_geometry(truths,masks,width,height)}


def geometry_summary(records):
    matched_boundary, center_pixels, center_normalized, mask_overlap = [],[],[],[]
    truth_count=0
    for record in records:
        n,p=len(record['truth_classes']),len(record['predictions']);truth_count+=n
        ious=np.asarray(record['ious'],float).reshape(n,p)
        for i,j in match_instances(ious,.5):
            matched_boundary.append(record['boundary_ious'][i][j])
            mask_overlap.append(ious[i,j])
            center=record['center_errors_pixels'][i][j]
            if center is not None:
                center_pixels.append(center);center_normalized.append(center/record['image_diagonal'])
    mean=lambda xs:float(np.mean(xs)) if xs else None
    return {'matched_instances':len(matched_boundary),'truth_instances':truth_count,
            'unmatched_truth':truth_count-len(matched_boundary),
            'matched_mean_mask_iou':mean(mask_overlap),'matched_mean_boundary_iou':mean(matched_boundary),
            'boundary_iou_with_unmatched_zero':float(sum(matched_boundary)/truth_count) if truth_count else None,
            'matched_mean_center_error_pixels':mean(center_pixels),
            'matched_p95_center_error_pixels':float(np.percentile(center_pixels,95)) if center_pixels else None,
            'matched_mean_center_error_normalized':mean(center_normalized)}


def group_summary(records, thresholds):
    result={'images':len(records),'raw':{},'regularized':{}}
    for mode in ['raw','regularized']:
        values=[r[mode] for r in records]
        for cutoff in thresholds:
            score=score_records(values,cutoff)
            result[mode][str(cutoff)]={k:score[k] for k in [
                'instance_count','prediction_count','clean_correct_instances','strict_instance_score',
                'missed_instances','spurious_detections','duplicate_detections','whole_scene_accuracy','per_class']}
        result[mode]['geometry']=geometry_summary(values)
    result['geometry_statuses']=dict(Counter(p['geometry_status'] for r in records for p in r['geometry']))
    result['fallback_reasons']=dict(Counter(p['geometry_reason'] for r in records for p in r['geometry'] if p['geometry_status']=='fallback'))
    result['raw_vertices']=sum(p['raw_vertices'] for r in records for p in r['geometry'])
    result['regularized_vertices']=sum(p['regularized_vertices'] for r in records for p in r['geometry'])
    deltas=[]
    for r in records:
        n,p=len(r['raw']['truth_classes']),len(r['raw']['predictions'])
        for i,j in match_instances(np.asarray(r['raw']['ious']).reshape(n,p),.5):
            deltas.append(r['regularized']['boundary_ious'][i][j]-r['raw']['boundary_ious'][i][j])
    result['paired_boundary_comparison_on_raw_matches']={
        'count':len(deltas),'improved':sum(d>1e-9 for d in deltas),'worsened':sum(d< -1e-9 for d in deltas),
        'unchanged':sum(abs(d)<=1e-9 for d in deltas),'mean_delta':float(np.mean(deltas)) if deltas else None}
    times=[r['inference_seconds'] for r in records]
    result['median_inference_seconds']=float(np.median(times)) if times else None
    result['mean_geometry_seconds']=float(np.mean([r['geometry_seconds'] for r in records])) if records else None
    return result


def main():
    started=time.perf_counter()
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));dataset=ROOT/cfg['dataset'];out=ROOT/cfg['output']
    lock=verify(dataset)
    rows=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['domain']=='real' or r['split']=='val']
    if Counter(r['domain'] for r in rows)!=Counter(real=46,legacy=108,focused=90):
        raise ValueError('Expected exactly 46 real training and 198 synthetic validation images')
    frozen={'config_sha256':digest(CONFIG),'manifest_sha256':lock['manifest_sha256'],
            'models':{name:digest(ROOT/path) for name,path in cfg['models'].items()},
            'sources':{name:digest(ROOT/'vision/scripts'/name) for name in ['contour_quality.py','run_contour_audit.py','evaluate_focused.py']}}
    frozen_path=out/'freeze.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text(encoding='utf-8'))!=frozen:raise ValueError('Frozen inputs/code changed; create a new experiment version')
    else:write_json(frozen_path,frozen)
    review=json.loads((ROOT/'dataset/real/annotations/review_v2/annotations.json').read_text(encoding='utf-8'))
    names={'real_'+r['image_id']:r['name'] for r in review['images']}
    occlusion_keys={key for key,name in names.items() if name=='IMG_20261102_002244..jpg'}
    torch.set_num_threads(2)
    summaries={};completed=0
    try:
        for model_name,model_path in cfg['models'].items():
            model=YOLO(str(ROOT/model_path))
            for size in cfg['sizes']:
                variant=f'{model_name}_{size}';records=[]
                settings={**cfg['settings'],'imgsz':size}
                for index,row in enumerate(rows):
                    target=out/'records'/variant/(row['key']+'.json')
                    if target.exists():
                        record=json.loads(target.read_text(encoding='utf-8'))
                        if record['image_sha256']!=row['image_sha256'] or record['settings']!=settings:
                            raise ValueError('Resume record mismatch')
                        for asset,sha in record['mask_hashes'].items():
                            if digest(out/asset)!=sha:raise ValueError('Saved mask changed: '+asset)
                    else:
                        with Image.open(dataset/row['image']) as source:im=source.convert('RGB')
                        truth=read_polygons(dataset/row['label'],im.width,im.height)
                        truth_classes=[c for c,_ in truth]
                        if set(truth_classes)!=set(row['classes']):raise ValueError('Truth class coverage differs from manifest')
                        truth_masks=masks_from_truth(truth,im.width,im.height)
                        t=time.perf_counter();result=model.predict(im,verbose=False,save=False,**settings)[0]
                        inference_seconds=time.perf_counter()-t;t=time.perf_counter()
                        predictions=[];raw=[]
                        if result.boxes is not None and len(result.boxes):
                            if result.masks is None or len(result.masks)!=len(result.boxes):raise ValueError('Mask/box count mismatch')
                            if tuple(result.masks.data.shape[-2:])!=(im.height,im.width):raise ValueError('Masks must use original-image coordinates')
                            for j,mask in enumerate(result.masks.data.cpu().numpy()):
                                predictions.append({'class_id':int(result.boxes.cls[j]),'confidence':float(result.boxes.conf[j]),
                                                    'box':result.boxes.xyxy[j].cpu().tolist()})
                                raw.append((mask>0).astype(np.uint8))
                        regularized,geometry=regularize_masks(raw,cfg['regularization'])
                        raw_record=metric_record(row,predictions,raw,truth_masks,im.width,im.height,truth_classes)
                        reg_record=metric_record(row,predictions,regularized,truth_masks,im.width,im.height,truth_classes)
                        geometry_seconds=time.perf_counter()-t
                        files={}
                        for mode,masks in [('raw',raw),('regularized',regularized)]:
                            directory=out/'masks'/variant/row['key']/mode;directory.mkdir(parents=True,exist_ok=True)
                            for j,mask in enumerate(masks):
                                path=directory/f'{j+1:02d}.png';Image.fromarray(mask*255).save(path)
                                rel=path.relative_to(out).as_posix();files[rel]=digest(path)
                        record={'key':row['key'],'name':names.get(row['key'],row['key']),
                            'image':row['image'],'image_sha256':row['image_sha256'],'domain':row['domain'],
                            'category':row.get('category'),'confirmed_occlusion_example':row['key'] in occlusion_keys,
                            'width':im.width,'height':im.height,'settings':settings,
                            'inference_seconds':inference_seconds,'geometry_seconds':geometry_seconds,
                            'raw':raw_record,'regularized':reg_record,'geometry':geometry,'mask_hashes':files,
                            'truth_polygons':[[c,p.tolist()] for c,p in truth]}
                        write_json(target,record)
                    records.append(record);completed+=1
                    if (index+1)%20==0 or index==len(rows)-1:
                        progress={'stage':'evaluating','variant':variant,'variant_images':index+1,'variant_total':len(rows),
                                  'completed_image_runs':completed,'total_image_runs':len(rows)*4,'elapsed_seconds':round(time.perf_counter()-started,1)}
                        write_json(out/'status.json',progress);print(json.dumps(progress),flush=True)
                groups={'all':records,**{d:[r for r in records if r['domain']==d] for d in ['real','legacy','focused']}}
                groups.update({key:[r for r in records if r['domain']=='real' and r['category']==cat]
                               for key,cat in [('empty_tray','empty_tray'),('white_cube','white_cube')]})
                groups['multi_object']=[r for r in records if r['domain']=='real' and r['category'] in {'multi_object','tray_multi'}]
                groups['occlusion_example']=[r for r in records if r['confirmed_occlusion_example']]
                summaries[variant]={key:group_summary(values,cfg['mask_match_thresholds']) for key,values in groups.items()}
                if summaries[variant]['empty_tray']['images']!=11 or summaries[variant]['white_cube']['images']!=12:
                    raise ValueError('Expected 11 empty tray and 12 white cube cases')
                write_json(out/f'{variant}_summary.json',summaries[variant])
        for name,path in cfg['models'].items():
            if digest(ROOT/path)!=frozen['models'][name]:raise ValueError('Model modified during audit')
        summary={'variants':summaries,'image_count':len(rows),'image_runs':completed,
                 'real_training_images':46,'synthetic_validation_images':198,'independent_real_test_images':0,
                 'no_training':True,'automatic_deployment':False,'cloud_spend':0,
                 'boundary_band_definition':'Inner mask band, 3x3 erosion, radius=round(0.005*image_diagonal), min 1; padded image border',
                 'geometric_matching':'Class-blind maximum-cardinality one-to-one at mask IoU >=0.5; unmatched truth reported separately',
                 'occlusion_group':'Confirmed example IMG_20261102_002244..jpg; overlaps multi_object group, not all occlusion scenes',
                 'elapsed_seconds_this_invocation':round(time.perf_counter()-started,1),'completed_at':datetime.now(timezone.utc).isoformat()}
        write_json(out/'summary.json',summary);write_json(out/'status.json',{'stage':'complete','image_runs':completed})
        print(json.dumps({'stage':'complete','image_runs':completed,'seconds':summary['elapsed_seconds_this_invocation']}),flush=True)
    except Exception as exc:
        write_json(out/'status.json',{'stage':'failed','error':str(exc),'completed_image_runs':completed})
        raise


if __name__=='__main__':main()
