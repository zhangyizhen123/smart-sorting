"""Lock a development-passing candidate, then evaluate a never-selected-on test set."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from evaluate_replay_optimization import ROOT,write_json,digest,resolve_spec,summarize
from prepare_unseen_shape_test import verify as verify_test
from evaluate_focused import score_records,match_instances,masks_from_truth
from contour_quality import pairwise_geometry
from shape_refiner import PairwiseRefiner,read_polygons

RUN=ROOT/'runs/segment/replay_optimization_v3'
TEST=ROOT/'dataset/unseen_shape_test_v3'


def lock_candidate():
    state=json.loads((RUN/'status.json').read_text(encoding='utf-8'))
    watcher=json.loads((RUN/'watcher_status.json').read_text(encoding='utf-8'))
    if state['stage']!='training_complete' or watcher['stage']!='decision_needed':raise ValueError('Complete the training and candidate comparisons before locking')
    comparison=json.loads((RUN/'development_comparison.json').read_text(encoding='utf-8'))
    selected=comparison['selected_for_sealed_test']
    if not selected or not comparison['candidates'][selected]['passes_development_gate']:raise ValueError('No development candidate passes all joint gates')
    test_lock=verify_test(TEST)
    report=RUN/'evaluation'/selected/'summary.json';summary=json.loads(report.read_text(encoding='utf-8'))
    raw_name=selected.removesuffix('_refined');source=RUN/'candidates'/raw_name/'model.pt'
    if digest(source)!=summary['provenance']['model_sha256']:raise ValueError('Selected checkpoint hash mismatch')
    lock_file=RUN/'candidate_lock.json'
    identity={'candidate':selected,'model_sha256':digest(source),'development_comparison_sha256':digest(RUN/'development_comparison.json'),
              'candidate_summary_sha256':digest(report),'test_manifest_sha256':test_lock['manifest_sha256']}
    if lock_file.exists():
        current=json.loads(lock_file.read_text(encoding='utf-8'))
        if any(current[key]!=value for key,value in identity.items()):raise ValueError('Cannot change a locked candidate after test exposure')
        return current
    bundle=RUN/'sealed_candidate';bundle.mkdir(exist_ok=True)
    shutil.copy2(source,bundle/'best.pt')
    pipeline={'version':1,'model':'best.pt','model_sha256':identity['model_sha256'],
              'settings':summary['provenance']['settings'],'evaluation':{'stage':'locked_before_sealed_test','not_deployed':True}}
    if selected.endswith('_refined'):
        refiner=ROOT/'models/trained/stl_shape_focus_v2/refiner';target=bundle/'refiner';target.mkdir(exist_ok=True)
        for filename in ['refiner.pt','metadata.json']:shutil.copy2(refiner/filename,target/filename)
        pipeline.update(refiner='refiner',refiner_sha256=digest(target/'refiner.pt'),refiner_metadata_sha256=digest(target/'metadata.json'))
    write_json(bundle/'pipeline.json',pipeline)
    lock={**identity,'pipeline':(bundle/'pipeline.json').relative_to(ROOT).as_posix(),'pipeline_sha256':digest(bundle/'pipeline.json'),
          'locked_at':datetime.now(timezone.utc).isoformat(),'training_and_selection_data':'replay_optimization_v3 validation only',
          'test_used_for_selection':False,'independent_real_test_images':0,'automatic_deployment':False}
    write_json(lock_file,lock)
    for file in ['compare_optimization_candidates.py','evaluate_replay_optimization.py','sealed_optimization_test.py','contour_quality.py']:
        destination=bundle/'source'/file;destination.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'vision/scripts'/file,destination)
    return lock


def evaluate_one(name,spec,test_rows):
    out=RUN/'sealed_test'/name
    model_path,settings,refiner_path,provenance=resolve_spec(spec)
    provenance.update(test_manifest_sha256=digest(TEST/'manifest.jsonl'),candidate_lock_sha256=digest(RUN/'candidate_lock.json'),
                      evaluator_sha256=digest(Path(__file__)))
    if (out/'freeze.json').exists():
        if json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=provenance:raise ValueError('Sealed-test evaluation changed')
    else:write_json(out/'freeze.json',provenance)
    model=YOLO(str(model_path));refiner=PairwiseRefiner(refiner_path) if refiner_path else None;records=[]
    for index,row in enumerate(test_rows):
        target=out/'records'/(row['key']+'.json')
        if target.exists():
            record=json.loads(target.read_text(encoding='utf-8'))
            for asset,value in record['mask_hashes'].items():
                if digest(out/asset)!=value:raise ValueError('Saved holdout mask changed')
        else:
            with Image.open(TEST/row['image']) as image:im=image.convert('RGB')
            truth=read_polygons(TEST/row['label'],im.width,im.height);truth_masks=masks_from_truth(truth,im.width,im.height)
            tick=time.perf_counter();result=model.predict(im,verbose=False,save=False,**settings)[0]
            masks=[];predictions=[]
            if result.boxes is not None and len(result.boxes):
                if result.masks is None or len(result.masks)!=len(result.boxes):raise ValueError('Mask/box count mismatch')
                if tuple(result.masks.data.shape[-2:])!=(im.height,im.width):raise ValueError('Unexpected mask coordinates')
                polygons=result.masks.xy
                for j,mask in enumerate(result.masks.data.cpu().numpy()):
                    base=int(result.boxes.cls[j]);confidence=float(result.boxes.conf[j])
                    cls=refiner.refine(np.asarray(im),polygons[j],base,confidence) if refiner else base
                    predictions.append({'class_id':int(cls),'base_class_id':base,'confidence':confidence,'box':result.boxes.xyxy[j].cpu().tolist()})
                    masks.append((mask>0).astype(np.uint8))
            elapsed=time.perf_counter()-tick;hashes={}
            for j,mask in enumerate(masks):
                path=out/'masks'/row['key']/f'{j+1:02d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask*255).save(path);hashes[path.relative_to(out).as_posix()]=digest(path)
            record={'key':row['key'],'image':row['image'],'domain':row['domain'],'truth_classes':[cls for cls,_ in truth],
                    'predictions':predictions,'inference_seconds':elapsed,'mask_hashes':hashes,
                    **pairwise_geometry(truth_masks,masks,im.width,im.height)}
            write_json(target,record)
        records.append(record)
        if (index+1)%100==0 or index+1==len(test_rows):
            write_json(out/'status.json',{'stage':'evaluating','images':index+1,'total':len(test_rows)})
            print(json.dumps({'sealed_test':name,'images':index+1,'total':len(test_rows)}),flush=True)
    summary={'name':name,'provenance':provenance,'images':len(records),'test_used_for_selection':False,
             'groups':{'synthetic_test':summarize(records),**{d:summarize([r for r in records if r['domain']==d]) for d in ['legacy','focused','negative']}}}
    negatives=[r for r in records if r['domain']=='negative']
    summary['groups']['negative']['empty_false_detection_rate']=sum(bool(r['predictions']) for r in negatives)/len(negatives)
    write_json(out/'summary.json',summary);write_json(out/'status.json',{'stage':'complete','images':len(records)})
    return summary


def image_features(record):
    scored=score_records([record],.5)
    n,p=len(record['truth_classes']),len(record['predictions'])
    pairs=match_instances(np.asarray(record['ious']).reshape(n,p),.5)
    boundaries={i:record['boundary_ious'][i][j] for i,j in pairs}
    overlaps={i:record['ious'][i][j] for i,j in pairs}
    return np.array([scored['clean_correct_instances'],n,scored['spurious_detections'],sum(boundaries.values())],float),boundaries,overlaps


def gains(candidate,baseline,paired):
    c,b,p=candidate.sum(axis=0),baseline.sum(axis=0),paired.sum(axis=0)
    return np.array([c[0]/(c[1]+c[2])-b[0]/(b[1]+b[2]),(b[2]-c[2])/len(candidate),
                     c[3]/c[1]-b[3]/b[1],p[0]/p[2] if p[2] else -1.],float)


def bootstrap_gains(candidate,baseline,paired,strata,replicates=5000,seed=20261219):
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        indices=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata])
        samples.append(gains(candidate[indices],baseline[indices],paired[indices]))
    values=np.asarray(samples)
    # Two old baselines x four improvement metrics: family-wise adjustment.
    tail=.05/(2*4*2)
    return np.quantile(values,[tail,1-tail],axis=0).T.tolist()


def compare_test():
    directory=RUN/'sealed_test'
    summaries={name:json.loads((directory/name/'summary.json').read_text(encoding='utf-8')) for name in ['v1_deployed','v2_deployed','candidate']}
    read=lambda name:{p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (directory/name/'records').glob('*.json')}
    candidate=read('candidate');keys=sorted(candidate)
    if len(keys)!=2900:raise ValueError('Incomplete sealed test')
    features={key:image_features(candidate[key]) for key in keys};a=np.stack([features[key][0] for key in keys])
    strata=[np.array([i for i,key in enumerate(keys) if candidate[key]['domain']==domain]) for domain in ['legacy','focused','negative']]
    comparisons={}
    for name in ['v1_deployed','v2_deployed']:
        old=read(name)
        if set(old)!=set(candidate):raise ValueError('Models were not tested on identical scenes')
        b=[];paired=[]
        for key in keys:
            bf,bb,bi=image_features(old[key]);af,ab,ai=features[key]
            if old[key]['truth_classes']!=candidate[key]['truth_classes']:raise ValueError('Truth changed between models')
            common=set(bb)&set(ab)
            b.append(bf);paired.append([sum(ab[i]-bb[i] for i in common),sum(ai[i]-bi[i] for i in common),len(common)])
        b=np.stack(b);paired=np.asarray(paired,float)
        point=gains(a,b,paired);intervals=bootstrap_gains(a,b,paired,strata)
        comparisons[name]={'gains':dict(zip(['strict_score','false_positives_per_image_reduction','coverage_boundary_iou','common_match_boundary_iou'],point.tolist())),
                           'simultaneous_adjusted_intervals':intervals,'all_improvement_lower_bounds_positive':all(low>0 for low,high in intervals),
                           'common_match_mask_iou_delta':float(paired[:,1].sum()/paired[:,2].sum()) if paired[:,2].sum() else None}
    candidate_summary=summaries['candidate']['groups']
    style_guard=all(candidate_summary[domain]['scores']['0.5']['strict_instance_score']>=max(summaries[old]['groups'][domain]['scores']['0.5']['strict_instance_score'] for old in ['v1_deployed','v2_deployed']) for domain in ['legacy','focused'])
    negative_guard=candidate_summary['negative']['empty_false_detection_rate']<=min(summaries[old]['groups']['negative']['empty_false_detection_rate'] for old in ['v1_deployed','v2_deployed'])
    passed=style_guard and negative_guard and all(value['all_improvement_lower_bounds_positive'] and value['common_match_mask_iou_delta'] is not None and value['common_match_mask_iou_delta']>=0 for value in comparisons.values())
    report={'scope':'2900 unseen synthetic scenes; not independent real-camera validation','images':2900,
            'comparisons':comparisons,'both_styles_do_not_regress':style_guard,'negative_alarm_does_not_regress':negative_guard,
            'passes_joint_test_gate':bool(passed),'bootstrap_replicates':5000,'bootstrap_unit':'paired scene; stratified by domain',
            'interval_adjustment':'Bonferroni family of 8 two-sided comparisons, nominal family alpha 0.05',
            'independent_real_test_images':0,'automatically_deployed':False}
    write_json(RUN/'sealed_test_result.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['lock','evaluate','compare']);args=parser.parse_args()
    torch.set_num_threads(2)
    if args.action=='lock':print(json.dumps(lock_candidate(),ensure_ascii=False,indent=2));return
    lock=json.loads((RUN/'candidate_lock.json').read_text(encoding='utf-8'));test_lock=verify_test(TEST)
    if test_lock['manifest_sha256']!=lock['test_manifest_sha256'] or digest(ROOT/lock['pipeline'])!=lock['pipeline_sha256']:raise ValueError('Locked candidate or test changed')
    if args.action=='evaluate':
        usage=TEST/'usage.json';value={'candidate_lock_sha256':digest(RUN/'candidate_lock.json'),'purpose':'final comparison after development selection'}
        if usage.exists() and json.loads(usage.read_text(encoding='utf-8'))!=value:raise ValueError('This test has already been exposed to a different selection')
        write_json(usage,value)
        rows=[json.loads(line) for line in (TEST/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        specs=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text(encoding='utf-8'))
        for name in ['v1_deployed','v2_deployed']:evaluate_one(name,specs[name],rows)
        evaluate_one('candidate',{'pipeline':lock['pipeline'],'role':'locked_candidate'},rows)
    print(json.dumps(compare_test(),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
