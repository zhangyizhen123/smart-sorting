"""Verify selected development predictions against the actual native V5 pipeline."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import ultralytics.utils.torch_utils as torch_utils
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_replay_optimization import digest,write_json
from pipeline_v5 import ProtectedPipeline

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/joint_round_v4'


def main():
    torch.set_num_threads(7);torch_utils.NUM_THREADS=7
    pipeline=ProtectedPipeline(RUN/'runtime_candidate/pipeline.json');chosen=[]
    for kind in ['original','expanded']:
        files=list((RUN/'selected'/kind/'records').glob('*.json'))
        rows=[(p,json.loads(p.read_text(encoding='utf-8'))) for p in sorted(files)]
        seen=set()
        predicates=[lambda r,d=d:r['domain']==d for d in ['legacy','focused','negative','real']]
        predicates += [lambda r:any(p.get('foreground_protected') for p in r['predictions']),
                       lambda r:any(p.get('specialist_source_index') is not None for p in r['predictions']),
                       lambda r:any(p.get('consensus_source')=='secondary' for p in r['predictions'])]
        for predicate in predicates:
            count=0
            for path,row in rows:
                if path.name not in seen and predicate(row):
                    chosen.append((kind,path,row));seen.add(path.name);count+=1
                    if count==2: break
    checked=[]
    for kind,path,row in chosen:
        data=ROOT/('dataset/replay_optimization_v3' if kind=='original' else 'dataset/joint_round_v4_development')
        image_path=data/row['image']
        if digest(image_path)!=row['image_sha256']: raise ValueError('Runtime-check image changed')
        predictions,masks,originals,settings=pipeline.predict_native(Image.open(image_path).convert('RGB'))
        if len(predictions)!=len(row['predictions']): raise AssertionError('Detection count mismatch: '+row['key'])
        for j,(actual,expected,mask) in enumerate(zip(predictions,row['predictions'],masks,strict=True)):
            if actual['class_id']!=expected['class_id'] or actual['confidence']!=expected['confidence'] or actual['box']!=expected['box']:
                raise AssertionError('Class/confidence/box mismatch: '+row['key'])
            expected_mask=np.asarray(Image.open(path.parent.parent/'masks'/row['key']/f'{j+1:02d}.png'))>0
            if not np.array_equal(mask>0,expected_mask): raise AssertionError('Native mask pixel mismatch: '+row['key'])
            if actual['source_prediction_index']>=len(originals): raise AssertionError('Original-mask index is invalid')
        checked.append(dict(population=kind,key=row['key'],detections=len(predictions),protected=sum(bool(p.get('foreground_protected')) for p in predictions)))
        print(json.dumps(dict(runtime_cases=len(checked),total=len(chosen))),flush=True)
    result=dict(passed=True,pipeline_sha256=digest(RUN/'runtime_candidate/pipeline.json'),selection_sha256=digest(RUN/'selection.json'),
                torch_runtime_threads=7,box_tolerance_pixels=0,confidence_tolerance=0,mask_pixel_tolerance=0,cases=checked,
                independent_real_images=0,test_accessed=False)
    write_json(RUN/'runtime_candidate/runtime_check.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
