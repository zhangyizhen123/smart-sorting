"""Lock one second-round recipe, then evaluate 5800 entirely new held-out scenes."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import ultralytics.utils.torch_utils as torch_utils

from bundle_development_recipe import verified_copy
from contour_quality import pairwise_geometry
from evaluate_focused import masks_from_truth
from evaluate_replay_optimization import digest,resolve_spec,summarize,write_json
from pipeline_v5 import ProtectedPipeline
from prepare_joint_round_data import verify
from shape_refiner import read_polygons
import sealed_optimization_test as baseline_evaluator
from sealed_optimization_test import image_features,gains

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/joint_round_v4'
CONFIG=ROOT/'vision/configs/joint_round_v4_test.json'
TEST=ROOT/'dataset/joint_round_v4_test'
N_IMAGES=5800
SOURCE_FILES=['sealed_joint_round_test.py','sealed_optimization_test.py','evaluate_replay_optimization.py',
              'evaluate_focused.py','contour_quality.py','run_contour_audit.py','pipeline_v2.py','pipeline_v3.py',
              'pipeline_v5.py','image_guided_masks.py','shape_verifier.py','foreground_gate.py','consensus_shapes.py',
              'consensus_evidence.py','protected_foreground.py','specialist_shapes.py','shape_refiner.py','mask_display.py',
              'prepare_joint_round_data.py','bundle_development_recipe.py','select_joint_round_candidate.py','bundle_joint_round_candidate.py']


def validate_lock():
    lock=json.loads((RUN/'candidate_lock.json').read_text(encoding='utf-8'))
    if not lock['selection_closed'] or digest(ROOT/lock['pipeline'])!=lock['pipeline_sha256']:
        raise ValueError('Locked selection/pipeline changed')
    if verify(CONFIG)['manifest_sha256']!=lock['test_manifest_sha256'] or digest(CONFIG)!=lock['test_config_sha256']:
        raise ValueError('Held-out dataset changed')
    if digest(RUN/'sealed_candidate/selection_snapshot.json')!=lock['selection_sha256']:
        raise ValueError('Selection snapshot changed')
    specs_path=ROOT/'vision/configs/optimization_baselines_v3.json'
    if digest(specs_path)!=lock['baseline_specs_sha256']: raise ValueError('Original baseline settings changed')
    specs=json.loads(specs_path.read_text())
    if {name:resolve_spec(specs[name])[3] for name in ['v1_deployed','v2_deployed']}!=lock['baseline_identities']:
        raise ValueError('Original baseline weights changed')
    for name,expected in lock['source_sha256'].items():
        if digest(ROOT/'vision/scripts'/name)!=expected or digest(RUN/'sealed_candidate/source'/name)!=expected:
            raise ValueError('Frozen inference/evaluation code changed: '+name)
    return lock


def lock_candidate():
    if (RUN/'candidate_lock.json').exists(): return validate_lock()
    if (TEST/'usage.json').exists(): raise ValueError('Cannot change selection after test exposure')
    selection=json.loads((RUN/'selection.json').read_text());selected=selection['selected']
    if not selected or not selected['eligible_before_geometry'] or not all(v['passed'] for v in selected['geometry'].values()):
        raise ValueError('Original and expanded development gates must pass')
    if not all(all(v['gates'].values()) for v in selected['scores'].values()): raise ValueError('A development safeguard failed')
    source=RUN/'runtime_candidate';audit=json.loads((source/'runtime_check.json').read_text())
    if not audit['passed'] or audit['pipeline_sha256']!=digest(source/'pipeline.json') or audit['selection_sha256']!=digest(RUN/'selection.json'):
        raise ValueError('A matching native runtime check is required')
    if audit['torch_runtime_threads']!=7 or any(audit[k]!=0 for k in ['box_tolerance_pixels','confidence_tolerance','mask_pixel_tolerance']) or len(audit['cases'])<14:
        raise ValueError('Exact native runtime coverage is required')
    test=verify(CONFIG)
    if test['images']!=N_IMAGES or test['role']!='sealed_test': raise ValueError('Wrong held-out dataset')
    ProtectedPipeline(source/'pipeline.json')
    target=RUN/'sealed_candidate'
    for path in source.rglob('*'):
        if path.is_file(): verified_copy(path,target/path.relative_to(source),digest(path))
    verified_copy(RUN/'selection.json',target/'selection_snapshot.json',digest(RUN/'selection.json'))
    hashes={}
    for name in SOURCE_FILES:
        path=ROOT/'vision/scripts'/name;hashes[name]=digest(path);verified_copy(path,target/'source'/name,hashes[name])
    specs_path=ROOT/'vision/configs/optimization_baselines_v3.json';specs=json.loads(specs_path.read_text())
    lock=dict(candidate='protected_consensus_v4',selection_closed=True,selection_sha256=digest(RUN/'selection.json'),
              pipeline=(target/'pipeline.json').relative_to(ROOT).as_posix(),pipeline_sha256=digest(target/'pipeline.json'),
              test_manifest_sha256=test['manifest_sha256'],test_config_sha256=digest(CONFIG),source_sha256=hashes,
              baseline_specs_sha256=digest(specs_path),baseline_identities={n:resolve_spec(specs[n])[3] for n in ['v1_deployed','v2_deployed']},
              previous_failed_test_result_sha256=digest(ROOT/'runs/segment/replay_optimization_v3/sealed_test_result.json'),
              locked_at=datetime.now(timezone.utc).isoformat(),test_used_for_selection=False,independent_real_test_images=0,
              bootstrap_replicates=5000,per_round_family_alpha=.025,automatic_deployment=False)
    write_json(RUN/'candidate_lock.json',lock);return lock


def evaluate_candidate(lock,rows):
    pipeline=ProtectedPipeline(ROOT/lock['pipeline']);out=RUN/'sealed_test/candidate'
    identity=dict(pipeline_sha256=lock['pipeline_sha256'],candidate_lock_sha256=digest(RUN/'candidate_lock.json'),
                  test_manifest_sha256=lock['test_manifest_sha256'],evaluator_sha256=digest(Path(__file__)))
    if (out/'freeze.json').exists() and json.loads((out/'freeze.json').read_text())!=identity:
        raise ValueError('Cached candidate belongs to different evaluation')
    write_json(out/'freeze.json',identity);records=[]
    for index,row in enumerate(rows):
        target=out/'records'/(row['key']+'.json')
        if target.exists():
            record=json.loads(target.read_text())
            if any(record[k]!=row[k] for k in ['image_sha256','label_sha256']): raise ValueError('Cached scene changed')
            if any(digest(out/p)!=h for p,h in record['mask_hashes'].items()): raise ValueError('Cached mask changed')
        else:
            image=Image.open(TEST/row['image']).convert('RGB');start=time.perf_counter()
            predictions,masks,originals,settings=pipeline.predict_native(image);elapsed=time.perf_counter()-start
            truth=read_polygons(TEST/row['label'],image.width,image.height);truth_masks=masks_from_truth(truth,image.width,image.height)
            if [cls for cls,p in truth]!=row['classes']: raise ValueError('Truth sequence changed')
            hashes={}
            for j,mask in enumerate(masks):
                asset=f"masks/{row['key']}/{j+1:02d}.png";path=out/asset;path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.asarray(mask,np.uint8)*255).save(path);hashes[asset]=digest(path)
            record=dict(key=row['key'],image=row['image'],image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],
                        domain=row['domain'],truth_classes=row['classes'],predictions=predictions,mask_hashes=hashes,
                        inference_seconds=elapsed,**pairwise_geometry(truth_masks,masks,image.width,image.height))
            write_json(target,record)
        records.append(record)
        if (index+1)%50==0 or index+1==len(rows):
            status=dict(stage='sealed_candidate',images=index+1,total=len(rows));write_json(out/'status.json',status);print(json.dumps(status),flush=True)
    groups={'synthetic_test':summarize(records),**{d:summarize([r for r in records if r['domain']==d]) for d in ['legacy','focused','negative']}}
    negative=[r for r in records if r['domain']=='negative']
    groups['negative']['empty_false_detection_rate']=sum(bool(r['predictions']) for r in negative)/len(negative)
    write_json(out/'summary.json',dict(name='candidate',provenance=identity,images=len(records),groups=groups,test_used_for_selection=False))
    write_json(out/'status.json',dict(stage='complete',images=len(records)))


def adjusted_intervals(candidate,baseline,paired,strata,replicates=5000,alpha=.025):
    rng=np.random.default_rng(20270203);values=[]
    for _ in range(replicates):
        indices=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata])
        values.append(gains(candidate[indices],baseline[indices],paired[indices]))
    tail=alpha/(2*4*2)
    return np.quantile(np.asarray(values),[tail,1-tail],axis=0).T.tolist()


def compare():
    directory=RUN/'sealed_test'
    reports={n:json.loads((directory/n/'summary.json').read_text()) for n in ['v1_deployed','v2_deployed','candidate']}
    if any(r['images']!=N_IMAGES for r in reports.values()): raise ValueError('Incomplete held-out comparison')
    load=lambda name:{p.stem:json.loads(p.read_text()) for p in (directory/name/'records').glob('*.json')}
    candidate=load('candidate');keys=sorted(candidate)
    if len(keys)!=N_IMAGES: raise ValueError('Incomplete candidate records')
    features={k:image_features(candidate[k]) for k in keys};a=np.stack([features[k][0] for k in keys])
    strata=[np.array([i for i,k in enumerate(keys) if candidate[k]['domain']==d]) for d in ['legacy','focused','negative']]
    comparisons={}
    for name in ['v1_deployed','v2_deployed']:
        old=load(name)
        if set(old)!=set(candidate): raise ValueError('Different scenes were evaluated')
        b=[];paired=[]
        for key in keys:
            if old[key]['truth_classes']!=candidate[key]['truth_classes']: raise ValueError('Different truth labels')
            bf,bb,bi=image_features(old[key]);af,ab,ai=features[key];common=set(ab)&set(bb)
            b.append(bf);paired.append([sum(ab[i]-bb[i] for i in common),sum(ai[i]-bi[i] for i in common),len(common)])
        b=np.stack(b);paired=np.asarray(paired,float);point=gains(a,b,paired)
        intervals=adjusted_intervals(a,b,paired,strata)
        comparisons[name]=dict(gains=dict(zip(['strict_score','false_positives_per_image_reduction','coverage_boundary_iou','common_match_boundary_iou'],point.tolist())),
                               simultaneous_adjusted_intervals=intervals,all_improvement_lower_bounds_positive=all(low>0 for low,high in intervals),
                               common_match_mask_iou_delta=float(paired[:,1].sum()/paired[:,2].sum()) if paired[:,2].sum() else None)
    groups=reports['candidate']['groups']
    style=all(groups[d]['scores']['0.5']['strict_instance_score']>=max(reports[n]['groups'][d]['scores']['0.5']['strict_instance_score'] for n in ['v1_deployed','v2_deployed']) for d in ['legacy','focused'])
    negative=groups['negative']['empty_false_detection_rate']<=min(reports[n]['groups']['negative']['empty_false_detection_rate'] for n in ['v1_deployed','v2_deployed'])
    passed=style and negative and all(v['all_improvement_lower_bounds_positive'] and v['common_match_mask_iou_delta'] is not None and v['common_match_mask_iou_delta']>=0 for v in comparisons.values())
    result=dict(images=N_IMAGES,scope='Second round: entirely new synthetic scenes, no independent real-camera test',
                comparisons=comparisons,both_styles_do_not_regress=style,negative_alarm_does_not_regress=negative,
                passes_joint_test_gate=bool(passed),bootstrap_replicates=5000,per_round_family_alpha=.025,
                bootstrap_unit='paired scene; stratified by domain',interval_adjustment='Bonferroni 8 two-sided comparisons; per-round family alpha 0.025',
                independent_real_test_images=0,automatically_deployed=False,candidate_lock_sha256=digest(RUN/'candidate_lock.json'))
    write_json(RUN/'sealed_test_result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['lock','evaluate','compare']);args=parser.parse_args()
    torch.set_num_threads(7);torch_utils.NUM_THREADS=7
    if args.action=='lock': print(json.dumps(lock_candidate()),flush=True);return
    lock=validate_lock()
    if args.action=='evaluate':
        expected=dict(candidate_lock_sha256=digest(RUN/'candidate_lock.json'),purpose='one fixed second-round complete recipe')
        if (TEST/'usage.json').exists() and json.loads((TEST/'usage.json').read_text())!=expected: raise ValueError('This test was already exposed to another candidate')
        write_json(TEST/'usage.json',expected)
        rows=[json.loads(line) for line in (TEST/'manifest.jsonl').read_text().splitlines()]
        specs=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text())
        # Reuse the exact original baseline algorithm with explicit round-specific paths.
        baseline_evaluator.RUN=RUN;baseline_evaluator.TEST=TEST
        for name in ['v1_deployed','v2_deployed']: baseline_evaluator.evaluate_one(name,specs[name],rows)
        evaluate_candidate(lock,rows)
    print(json.dumps(compare()),flush=True)


if __name__=='__main__':
    try:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        main()
    finally:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
