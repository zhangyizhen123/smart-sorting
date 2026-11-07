"""Close development selection and test one immutable complete inference recipe."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
import ultralytics.utils.torch_utils as torch_utils
from bundle_development_recipe import verified_copy
from compare_optimization_candidates import compare
from contour_quality import pairwise_geometry
from evaluate_focused import masks_from_truth
from evaluate_replay_optimization import ROOT,digest,write_json,summarize,resolve_spec
from prepare_unseen_shape_test import verify as verify_test
from shape_refiner import read_polygons
from pipeline_v2 import NativeShapePipeline
from pipeline_v3 import RecipePipeline
from pipeline_v4 import SpecialistPipeline
from sealed_optimization_test import evaluate_one,compare_test

RUN=ROOT/'runs/segment/replay_optimization_v3'
TEST=ROOT/'dataset/unseen_shape_test_v3'
SOURCE_FILES=['sealed_recipe_test.py','sealed_optimization_test.py','evaluate_replay_optimization.py',
              'evaluate_focused.py','contour_quality.py','run_contour_audit.py','compare_optimization_candidates.py',
              'pipeline_v2.py','pipeline_v3.py','pipeline_v4.py','image_guided_masks.py','shape_verifier.py',
              'foreground_gate.py','consensus_shapes.py','specialist_shapes.py','shape_refiner.py','mask_display.py',
              'prepare_unseen_shape_test.py','bundle_development_recipe.py']


def lock_candidate():
    lock_file=RUN/'candidate_lock.json'
    if lock_file.exists():
        return validate_lock()
    if (TEST/'usage.json').exists():
        raise ValueError('Cannot select a new candidate after this test was exposed')
    comparison=compare(RUN);name=comparison.get('selected_for_sealed_test')
    if not name or not comparison['candidates'][name]['passes_development_gate']:
        raise ValueError('No development candidate passes every quality gate')
    source=RUN/'runtime_recipes'/name;audit=json.loads((source/'runtime_check.json').read_text(encoding='utf-8'))
    summary=RUN/'evaluation'/name/'summary.json'
    if not audit['passed'] or audit['candidate_summary_sha256']!=digest(summary) or audit['pipeline_sha256']!=digest(source/'pipeline.json'):
        raise ValueError('A current native-runtime equivalence check is required')
    if audit.get('torch_runtime_threads')!=7:
        raise ValueError('Runtime equivalence must use the original seven-thread detector profile')
    manifest=json.loads((source/'pipeline.json').read_text(encoding='utf-8'))
    {2:NativeShapePipeline,3:RecipePipeline,4:SpecialistPipeline}[manifest['version']](source/'pipeline.json')
    test_lock=verify_test(TEST)
    bundle=RUN/'sealed_candidate'
    for path in source.rglob('*'):
        if path.is_file():
            verified_copy(path,bundle/path.relative_to(source),digest(path))
    write_json(bundle/'selection_snapshot.json',comparison)
    verified_copy(summary,bundle/'candidate_development_summary.json',digest(summary))
    specs_path=ROOT/'vision/configs/optimization_baselines_v3.json'
    specs=json.loads(specs_path.read_text(encoding='utf-8'))
    baseline_identities={name:resolve_spec(specs[name])[3] for name in ['v1_deployed','v2_deployed']}
    hashes={}
    for name in SOURCE_FILES:
        path=ROOT/'vision/scripts'/name;value=digest(path)
        verified_copy(path,bundle/'source'/name,value);hashes[name]=value
    training_states={}
    for label,path in [('main',RUN/'status.json'),('focused_320',RUN/'focus_replay_320/status.json')]:
        if path.exists():
            training_states[label]=json.loads(path.read_text(encoding='utf-8'))
    lock=dict(candidate=comparison['selected_for_sealed_test'],candidate_summary_sha256=digest(summary),
              development_comparison_sha256=digest(bundle/'selection_snapshot.json'),
              pipeline=(bundle/'pipeline.json').relative_to(ROOT).as_posix(),pipeline_sha256=digest(bundle/'pipeline.json'),
              test_manifest_sha256=test_lock['manifest_sha256'],source_sha256=hashes,
              baseline_specs_sha256=digest(specs_path),baseline_identities=baseline_identities,
              locked_at=datetime.now(timezone.utc).isoformat(),selection_closed=True,test_used_for_selection=False,
              training_states_at_selection=training_states,
              selection_rule='All completed development candidates compared; future training results excluded from this sealed round',
              independent_real_test_images=0,automatic_deployment=False)
    write_json(lock_file,lock)
    return lock


def validate_lock():
    lock=json.loads((RUN/'candidate_lock.json').read_text(encoding='utf-8'))
    if lock.get('selection_closed') is not True:
        raise ValueError('Development selection is not closed')
    if digest(ROOT/lock['pipeline'])!=lock['pipeline_sha256']:
        raise ValueError('Locked complete pipeline changed')
    if digest(RUN/'sealed_candidate/selection_snapshot.json')!=lock['development_comparison_sha256']:
        raise ValueError('Locked development comparison changed')
    if verify_test(TEST)['manifest_sha256']!=lock['test_manifest_sha256']:
        raise ValueError('Sealed test data changed')
    specs_path=ROOT/'vision/configs/optimization_baselines_v3.json'
    if digest(specs_path)!=lock['baseline_specs_sha256']:
        raise ValueError('Deployed baseline settings changed')
    specs=json.loads(specs_path.read_text(encoding='utf-8'))
    if {name:resolve_spec(specs[name])[3] for name in ['v1_deployed','v2_deployed']}!=lock['baseline_identities']:
        raise ValueError('Deployed baseline identities changed')
    for name,expected in lock['source_sha256'].items():
        if digest(ROOT/'vision/scripts'/name)!=expected or digest(RUN/'sealed_candidate/source'/name)!=expected:
            raise ValueError('Locked evaluator or inference source changed')
    return lock


def evaluate_recipe(lock,rows):
    path=ROOT/lock['pipeline'];manifest=json.loads(path.read_text(encoding='utf-8'))
    pipeline={2:NativeShapePipeline,3:RecipePipeline,4:SpecialistPipeline}[manifest['version']](path)
    out=RUN/'sealed_test/candidate'
    provenance=dict(pipeline_sha256=digest(path),candidate_lock_sha256=digest(RUN/'candidate_lock.json'),
                    evaluator_sha256=digest(Path(__file__)),test_manifest_sha256=digest(TEST/'manifest.jsonl'),
                    role='locked_complete_recipe',source_sha256=lock['source_sha256'])
    if (out/'freeze.json').exists() and json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Sealed candidate evaluation changed')
    write_json(out/'freeze.json',provenance);records=[]
    for index,row in enumerate(rows):
        target=out/'records'/(row['key']+'.json')
        if target.exists():
            record=json.loads(target.read_text(encoding='utf-8'))
            if record['image_sha256']!=row['image_sha256'] or record['label_sha256']!=row['label_sha256']:
                raise ValueError('Cached sealed inputs changed')
            for asset,expected in record['mask_hashes'].items():
                if digest(out/asset)!=expected:
                    raise ValueError('Cached sealed geometry changed')
        else:
            image=Image.open(TEST/row['image']).convert('RGB')
            tick=time.perf_counter();predictions,masks,originals,settings=pipeline.predict_native(image)
            elapsed=time.perf_counter()-tick;hashes={}
            for j,mask in enumerate(masks):
                asset=out/'masks'/row['key']/f'{j+1:02d}.png';asset.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.asarray(mask,np.uint8)*255).save(asset);hashes[asset.relative_to(out).as_posix()]=digest(asset)
            truth=read_polygons(TEST/row['label'],image.width,image.height)
            truth_masks=masks_from_truth(truth,image.width,image.height)
            if [cls for cls,polygon in truth]!=row['classes']:
                raise ValueError('Sealed class sequence differs from manifest')
            record=dict(key=row['key'],image=row['image'],image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],
                        domain=row['domain'],truth_classes=[cls for cls,polygon in truth],predictions=predictions,
                        inference_seconds=elapsed,mask_hashes=hashes,**pairwise_geometry(truth_masks,masks,image.width,image.height))
            write_json(target,record)
        records.append(record)
        if (index+1)%50==0 or index+1==len(rows):
            state=dict(stage='evaluating',images=index+1,total=len(rows),candidate=lock['candidate'])
            write_json(out/'status.json',state);print(json.dumps(state),flush=True)
    groups={'synthetic_test':summarize(records),**{d:summarize([r for r in records if r['domain']==d]) for d in ['legacy','focused','negative']}}
    negative=[r for r in records if r['domain']=='negative']
    groups['negative']['empty_false_detection_rate']=sum(bool(r['predictions']) for r in negative)/len(negative)
    report=dict(name='candidate',provenance=provenance,images=len(records),test_used_for_selection=False,groups=groups)
    write_json(out/'summary.json',report);write_json(out/'status.json',dict(stage='complete',images=len(records)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['lock','evaluate','compare']);args=parser.parse_args()
    torch.set_num_threads(7);torch_utils.NUM_THREADS=7
    if args.action=='lock':
        lock=lock_candidate();print(json.dumps(dict(candidate=lock['candidate'],selection_closed=True,pipeline_sha256=lock['pipeline_sha256'])),flush=True);return
    lock=validate_lock()
    if args.action=='evaluate':
        usage=TEST/'usage.json';expected=dict(candidate_lock_sha256=digest(RUN/'candidate_lock.json'),purpose='final comparison of immutable complete recipe')
        if usage.exists() and json.loads(usage.read_text(encoding='utf-8'))!=expected:
            raise ValueError('Sealed data already assigned to another selection')
        write_json(usage,expected)
        rows=[json.loads(line) for line in (TEST/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows)!=2900:
            raise ValueError('Incomplete sealed population')
        specs=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text(encoding='utf-8'))
        for name in ['v1_deployed','v2_deployed']:
            evaluate_one(name,specs[name],rows)
        evaluate_recipe(lock,rows)
    print(json.dumps(compare_test(),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        if hasattr(ctypes,'windll'):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        main()
    finally:
        if hasattr(ctypes,'windll'):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
