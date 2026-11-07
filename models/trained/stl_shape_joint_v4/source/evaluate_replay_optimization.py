"""Evaluate unchanged deployed pipelines and candidates on frozen larger validation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from contour_quality import pairwise_geometry
from evaluate_focused import score_records, masks_from_truth, digest
from shape_refiner import PairwiseRefiner,read_polygons
from run_contour_audit import geometry_summary
from prepare_replay_optimization import verify

ROOT=Path(__file__).resolve().parents[2]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8');temporary.replace(path)


def resolve_spec(spec):
    if 'pipeline' in spec:
        path=ROOT/spec['pipeline'];p=json.loads(path.read_text(encoding='utf-8'))
        model=path.parent/p['model'];settings=p['settings'];refiner=path.parent/p['refiner'] if p.get('refiner') else None
        if digest(model)!=p['model_sha256']:raise ValueError('Deployed model hash changed')
        if refiner:
            for name,key in [('refiner.pt','refiner_sha256'),('metadata.json','refiner_metadata_sha256')]:
                if digest(refiner/name)!=p[key]:raise ValueError('Deployed refiner hash changed')
        provenance={'pipeline_sha256':digest(path)}
    else:
        model=ROOT/spec['model'];settings=spec['settings'];refiner=ROOT/spec['refiner'] if spec.get('refiner') else None;provenance={}
    provenance.update(model_sha256=digest(model),settings=settings,role=spec['role'])
    if refiner:
        provenance.update(refiner_sha256=digest(refiner/'refiner.pt'),refiner_metadata_sha256=digest(refiner/'metadata.json'))
    return model,settings,refiner,provenance


def summarize(records):
    scores={str(threshold):score_records(records,threshold) for threshold in [.5,.75,.9]}
    for score in scores.values():score.pop('errors',None)
    return {'images':len(records),'scores':scores,'geometry':geometry_summary(records),
            'false_positives_per_image':scores['0.5']['spurious_detections']/len(records) if records else None,
            'median_inference_seconds':float(np.median([r['inference_seconds'] for r in records])) if records else None}


def evaluate(name,spec,cfg):
    dataset=ROOT/cfg['dataset'];lock=verify(dataset)
    out=ROOT/cfg['output']/'evaluation'/name
    rows=[json.loads(line) for line in (dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split']=='val' or (r['domain']=='real' and r.get('repetition')==0)]
    if Counter(r['domain'] for r in rows)!=Counter(legacy=108,focused=900,negative=100,real=46):raise ValueError('Unexpected evaluation composition')
    model_path,settings,refiner_path,provenance=resolve_spec(spec)
    provenance.update(dataset_manifest_sha256=lock['manifest_sha256'],evaluator_sha256=digest(Path(__file__)),
                      geometry_sha256=digest(ROOT/'vision/scripts/contour_quality.py'),scoring_sha256=digest(ROOT/'vision/scripts/evaluate_focused.py'))
    freeze=out/'freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text(encoding='utf-8'))!=provenance:raise ValueError('Evaluation changed; use a new run name')
    else:write_json(freeze,provenance)
    model=YOLO(str(model_path));refiner=PairwiseRefiner(refiner_path) if refiner_path else None
    records=[];started=time.perf_counter()
    for index,row in enumerate(rows):
        path=out/'records'/(row['key']+'.json')
        if path.exists():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['image_sha256']!=row['image_sha256'] or record['label_sha256']!=row['label_sha256']:raise ValueError('Record input changed')
            for asset,value in record['mask_hashes'].items():
                if digest(out/asset)!=value:raise ValueError('Saved mask changed')
        else:
            with Image.open(dataset/row['image']) as source:image=source.convert('RGB')
            truth=read_polygons(dataset/row['label'],image.width,image.height)
            truth_classes=[c for c,_ in truth]
            if truth_classes!=row['classes']:raise ValueError('Per-instance label sequence differs from frozen data')
            truth_masks=masks_from_truth(truth,image.width,image.height)
            tick=time.perf_counter();result=model.predict(image,verbose=False,save=False,**settings)[0]
            masks=[];predictions=[]
            if result.boxes is not None and len(result.boxes):
                if result.masks is None or len(result.masks)!=len(result.boxes):raise ValueError('Mask count mismatch')
                if tuple(result.masks.data.shape[-2:])!=(image.height,image.width):raise ValueError('Mask coordinates mismatch')
                polygons=result.masks.xy
                for j,mask in enumerate(result.masks.data.cpu().numpy()):
                    base=int(result.boxes.cls[j]);confidence=float(result.boxes.conf[j])
                    cls=refiner.refine(np.asarray(image),polygons[j],base,confidence) if refiner else base
                    predictions.append({'class_id':int(cls),'base_class_id':base,'confidence':confidence,
                                        'box':result.boxes.xyxy[j].cpu().tolist()})
                    masks.append((mask>0).astype(np.uint8))
            elapsed=time.perf_counter()-tick
            files={};directory=out/'masks'/row['key'];directory.mkdir(parents=True,exist_ok=True)
            for j,mask in enumerate(masks):
                mask_file=directory/f'{j+1:02d}.png';Image.fromarray(mask*255).save(mask_file)
                files[mask_file.relative_to(out).as_posix()]=digest(mask_file)
            record={'key':row['key'],'image':row['image'],'image_sha256':row['image_sha256'],'label_sha256':row['label_sha256'],
                    'domain':row['domain'],'category':row.get('category'),'mode':row.get('mode'),
                    'width':image.width,'height':image.height,'truth_classes':truth_classes,'predictions':predictions,
                    'inference_seconds':elapsed,'mask_hashes':files,
                    **pairwise_geometry(truth_masks,masks,image.width,image.height)}
            write_json(path,record)
        records.append(record)
        if (index+1)%50==0 or index+1==len(rows):
            status={'stage':'evaluating','name':name,'images':index+1,'total':len(rows),'elapsed_seconds':round(time.perf_counter()-started,1)}
            write_json(out/'status.json',status);print(json.dumps(status),flush=True)
    groups={'synthetic_validation':[r for r in records if r['domain']!='real'],
            'positive_validation':[r for r in records if r['domain'] in {'legacy','focused'}],
            **{domain:[r for r in records if r['domain']==domain] for domain in ['legacy','focused','negative','real']},
            'real_empty':[r for r in records if r['category']=='empty_tray'],
            'real_white':[r for r in records if r['category']=='white_cube']}
    summary={'name':name,'provenance':provenance,'image_count':len(rows),'validation_images':1108,'real_train_fit_images':46,
             'real_independent_test_images':0,'test_used_for_selection':False,'groups':{key:summarize(values) for key,values in groups.items()}}
    for key in ['negative','real_empty']:
        group=groups[key]
        summary['groups'][key]['empty_false_detection_images']=sum(bool(r['predictions']) for r in group)
        summary['groups'][key]['empty_false_detection_rate']=sum(bool(r['predictions']) for r in group)/len(group)
    if digest(model_path)!=provenance['model_sha256']:raise ValueError('Evaluated model changed during run')
    write_json(out/'summary.json',summary);write_json(out/'status.json',{'stage':'complete','images':len(rows)})
    print(json.dumps({'name':name,'complete':True,'strict':summary['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score']}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default='all');p.add_argument('--name');p.add_argument('--model');p.add_argument('--imgsz',type=int,default=640)
    args=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads((ROOT/'vision/configs/replay_optimization_v3.json').read_text(encoding='utf-8'))
    if args.model:
        if not args.name or not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('A safe distinct candidate name is required')
        spec={'model':args.model,'role':'candidate','settings':{'imgsz':args.imgsz,'conf':.25,'iou':.7,'device':'cpu','retina_masks':True,'agnostic_nms':True}}
        evaluate(args.name,spec,cfg)
    else:
        baselines=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text(encoding='utf-8'))
        selected=baselines if args.baseline=='all' else {args.baseline:baselines[args.baseline]}
        for name,spec in selected.items():evaluate(name,spec,cfg)


if __name__=='__main__':main()
