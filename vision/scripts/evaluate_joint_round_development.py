"""Cache unchanged baselines and unfiltered components on NEW development scenes."""
import ctypes
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO

from contour_quality import pairwise_geometry
from consensus_evidence import combine_native
from evaluate_focused import masks_from_truth
from evaluate_replay_optimization import digest,resolve_spec,summarize,write_json
from foreground_gate import ForegroundGate
from pipeline_v3 import RecipePipeline
from prepare_joint_round_data import verify
from shape_refiner import PairwiseRefiner,read_polygons
from shape_verifier import features

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/segment/replay_optimization_v3'
OUT=ROOT/'runs/segment/joint_round_v4/development'
CONFIG=ROOT/'vision/configs/joint_round_v4_development.json'


def baseline_predict(model,refiner,settings,image):
    result=model.predict(image,verbose=False,save=False,**settings)[0]
    predictions=[];masks=[]
    if result.boxes is not None and len(result.boxes):
        if result.masks is None or len(result.masks)!=len(result.boxes): raise ValueError('Missing instance masks')
        if tuple(result.masks.data.shape[-2:])!=(image.height,image.width): raise ValueError('Mask coordinate mismatch')
        polygons=result.masks.xy
        for j,mask in enumerate(result.masks.data.cpu().numpy()):
            base=int(result.boxes.cls[j]);confidence=float(result.boxes.conf[j])
            cls=refiner.refine(np.asarray(image),polygons[j],base,confidence) if refiner else base
            predictions.append(dict(class_id=int(cls),base_class_id=base,confidence=confidence,box=result.boxes.xyxy[j].cpu().tolist()))
            masks.append(mask>0)
    return predictions,masks


def main():
    torch.set_num_threads(7);torch_utils.NUM_THREADS=7;cv2.setNumThreads(1)
    frozen=verify(CONFIG);cfg=json.loads(CONFIG.read_text());data=ROOT/cfg['output']
    if cfg['role']!='development': raise ValueError('This evaluator never opens test data')
    specs_path=ROOT/'vision/configs/optimization_baselines_v3.json';specs=json.loads(specs_path.read_text())
    resolved={name:resolve_spec(specs[name]) for name in ['v1_deployed','v2_deployed']}
    base_manifest=OLD/'sealed_candidate/primary/source/pipeline.json'
    freeze=dict(dataset_manifest_sha256=frozen['manifest_sha256'],config_sha256=digest(CONFIG),
                baseline_specs_sha256=digest(specs_path),baseline_identities={k:v[3] for k,v in resolved.items()},
                base_pipeline_sha256=digest(base_manifest),foreground_model_sha256=digest(OLD/'foreground_gate_v1/model.pt'),
                consensus_evidence_sha256=digest(ROOT/'vision/scripts/consensus_evidence.py'),
                evaluator_sha256=digest(Path(__file__)),torch_runtime_threads=7,test_data_accessed=False)
    if (OUT/'freeze.json').exists() and json.loads((OUT/'freeze.json').read_text())!=freeze:
        raise ValueError('Development evaluator inputs changed')
    write_json(OUT/'freeze.json',freeze)
    baselines={name:(YOLO(str(model)),PairwiseRefiner(refiner) if refiner else None,settings) for name,(model,settings,refiner,p) in resolved.items()}
    base=RecipePipeline(base_manifest);gate=ForegroundGate(OLD/'foreground_gate_v1/model.pt')
    rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines()]
    records={name:[] for name in [*baselines,'base_agreement']}
    for index,row in enumerate(rows):
        image=None;truth=None;truth_masks=None
        for name in records:
            directory=OUT/name;path=directory/'records'/(row['key']+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if any(record[k]!=row[k] for k in ['image_sha256','label_sha256']): raise ValueError('Cached scene changed')
                if any(digest(directory/asset)!=h for asset,h in record['mask_hashes'].items()): raise ValueError('Cached mask changed')
            else:
                if image is None:
                    image=Image.open(data/row['image']).convert('RGB');truth=read_polygons(data/row['label'],image.width,image.height)
                    truth_masks=masks_from_truth(truth,image.width,image.height)
                    if [cls for cls,p in truth]!=row['classes']: raise ValueError('Truth order changed')
                tick=time.perf_counter();probabilities=None
                if name=='base_agreement':
                    components={k:v.predict_native(image) for k,v in base.children.items()}
                    predictions,masks,originals,settings=combine_native(components,base.policy)
                    rgb=np.asarray(image)
                    probabilities=gate.probabilities(np.stack([features(rgb,m) for m in masks])).tolist() if masks else []
                else:
                    predictions,masks=baseline_predict(*baselines[name],image)
                elapsed=time.perf_counter()-tick;hashes={}
                for j,mask in enumerate(masks):
                    asset=f"masks/{row['key']}/{j+1:02d}.png";target=directory/asset;target.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.asarray(mask,np.uint8)*255).save(target);hashes[asset]=digest(target)
                record=dict(key=row['key'],image=row['image'],image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],
                            domain=row['domain'],truth_classes=row['classes'],predictions=predictions,mask_hashes=hashes,
                            inference_seconds=elapsed,foreground_probabilities=probabilities,
                            **pairwise_geometry(truth_masks,masks,image.width,image.height))
                write_json(path,record)
            records[name].append(record)
        if (index+1)%50==0 or index+1==len(rows):
            status=dict(stage='development_evaluation',images=index+1,total=len(rows),sources=list(records))
            write_json(OUT/'status.json',status);print(json.dumps(status),flush=True)
    for name,values in records.items():
        groups={'synthetic_validation':summarize(values),**{d:summarize([r for r in values if r['domain']==d]) for d in ['legacy','focused','negative']}}
        negative=[r for r in values if r['domain']=='negative']
        groups['negative']['empty_false_detection_rate']=sum(bool(r['predictions']) for r in negative)/len(negative)
        write_json(OUT/name/'summary.json',dict(name=name,provenance=freeze,images=len(values),groups=groups,test_used_for_selection=False))
    write_json(OUT/'status.json',dict(stage='complete',images=len(rows),test_accessed=False))


if __name__=='__main__':
    try:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        main()
    finally:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
