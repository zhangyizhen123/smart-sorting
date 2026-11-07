"""Verify real composed inference against cached development results, without test exposure."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import ultralytics.utils.torch_utils as torch_utils

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vision/scripts'))
from bundle_development_recipe import build
from evaluate_replay_optimization import digest,write_json
from pipeline_v2 import NativeShapePipeline
from pipeline_v3 import RecipePipeline
from pipeline_v4 import SpecialistPipeline


def check(name):
    torch.set_num_threads(7)
    torch_utils.NUM_THREADS=7
    run=ROOT/'runs/segment/replay_optimization_v3';candidate=run/'evaluation'/name
    path=build(name,run/'runtime_recipes'/name)
    manifest=json.loads(path.read_text(encoding='utf-8'))
    pipeline={2:NativeShapePipeline,3:RecipePipeline,4:SpecialistPipeline}[manifest['version']](path)
    groups=defaultdict(list)
    for file in (candidate/'records').glob('*.json'):
        row=json.loads(file.read_text(encoding='utf-8'));groups[row['domain']].append(row)
    samples=[]
    for domain,rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda r:hashlib.sha256(r['key'].encode()).hexdigest())
        samples.extend(ordered[:3])
        consulted=[r for r in ordered if any('specialist_source_index' in p for p in r['predictions'])]
        samples.extend(consulted[:2])
        additions=[r for r in ordered if any(p.get('consensus_source')=='secondary' for p in r['predictions'])]
        samples.extend(additions[:1])
    samples=list({r['key']:r for r in samples}.values())
    results=[]
    for row in samples:
        image=Image.open(ROOT/'dataset/replay_optimization_v3'/row['image']).convert('RGB')
        predictions,masks,originals,settings=pipeline.predict_native(image)
        assert len(predictions)==len(row['predictions']),row['key']
        assert [p['class_id'] for p in predictions]==[p['class_id'] for p in row['predictions']],row['key']
        differences=[];box_differences=[];confidence_differences=[]
        for j,prediction in enumerate(predictions):
            expected=row['predictions'][j]
            confidence_difference=abs(prediction['confidence']-expected['confidence'])
            box_difference=float(np.max(np.abs(np.asarray(prediction['box'])-np.asarray(expected['box']))))
            assert confidence_difference==0,(row['key'],j,'confidence',confidence_difference)
            assert box_difference==0,(row['key'],j,'box',box_difference)
            confidence_differences.append(confidence_difference);box_differences.append(box_difference)
            mask=np.asarray(Image.open(candidate/'masks'/row['key']/f'{j+1:02d}.png'))>0
            difference=int(np.count_nonzero(mask!=masks[j]));differences.append(difference)
            assert difference==0,(row['key'],j,difference)
            assert originals[prediction['source_prediction_index']].shape==mask.shape
        results.append(dict(key=row['key'],domain=row['domain'],predictions=len(predictions),mask_pixel_differences=differences,
                            exact_classes_and_masks=True,box_differences_pixels=box_differences,
                            confidence_differences=confidence_differences))
        print(json.dumps(results[-1]),flush=True)
    result=dict(passed=True,candidate=name,pipeline=path.relative_to(ROOT).as_posix(),pipeline_sha256=digest(path),
                candidate_summary_sha256=digest(candidate/'summary.json'),cases=results,
                scope='native runtime equivalence only; not sealed acceptance',sealed_test_used=False,deployed=False,
                torch_runtime_threads=torch.get_num_threads(),box_tolerance_pixels=0,confidence_tolerance=0)
    write_json(run/'runtime_recipes'/name/'runtime_check.json',result)
    print(json.dumps(dict(runtime_check_passed=True,candidate=name,cases=len(results))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('candidate')
    check(parser.parse_args().candidate)
