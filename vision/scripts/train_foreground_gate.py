"""Mine actual empty-training-scene detections, then learn a binary ROI gate."""
from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
from evaluate_replay_optimization import ROOT,digest,write_json
from foreground_gate import ForegroundNetwork,pooled_features
from shape_verifier import features,FEATURES

RUN=ROOT/'runs/segment/replay_optimization_v3'
DATA=ROOT/'dataset/replay_optimization_v3'
OUT=RUN/'foreground_gate_v1'
SEED=20261102


def mine():
    base=ROOT/'models/trained/stl_shape_focus_v2'
    manifest=json.loads((base/'pipeline.json').read_text(encoding='utf-8'))
    source=base/manifest['model']
    if digest(source)!=manifest['model_sha256']:
        raise ValueError('Mining model changed')
    freeze=dict(dataset_sha256=digest(DATA/'manifest.jsonl'),model_sha256=digest(source),settings=manifest['settings'],
                features_sha256=digest(ROOT/'vision/scripts/shape_verifier.py'),gate_sha256=digest(ROOT/'vision/scripts/foreground_gate.py'),
                trainer_sha256=digest(Path(__file__)),base_features_sha256=digest(RUN/'verifier_v1/features.npz'),
                mining_scope='empty training scenes only',test_accessed=False)
    if (OUT/'freeze.json').exists() and json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))!=freeze:
        raise ValueError('Foreground training inputs changed')
    write_json(OUT/'freeze.json',freeze)
    rows=[json.loads(line) for line in (DATA/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split']=='train' and r['domain']=='negative']
    if len(rows)!=200 or any(r['classes'] for r in rows):
        raise ValueError('Expected 200 labeled-empty training scenes')
    model=None;mined=[]
    for i,row in enumerate(rows):
        saved=OUT/'mined'/f"{row['key']}.npz"
        if saved.exists():
            with np.load(saved) as cached:
                mined.extend(cached['x'])
        else:
            if digest(DATA/row['image'])!=row['image_sha256'] or digest(DATA/row['label'])!=row['label_sha256']:
                raise ValueError('Training asset changed')
            if model is None:
                model=YOLO(str(source))
            rgb=np.asarray(Image.open(DATA/row['image']).convert('RGB'))
            result=model.predict(Image.fromarray(rgb),verbose=False,**manifest['settings'])[0]
            values=[];hashes={}
            if result.boxes is not None and len(result.boxes):
                if result.masks is None or tuple(result.masks.data.shape[-2:])!=rgb.shape[:2]:
                    raise ValueError('Native training proposal mismatch')
                for j,mask in enumerate(result.masks.data.cpu().numpy()):
                    binary=np.asarray(mask>0,np.uint8)
                    values.append(features(rgb,binary))
                    asset=OUT/'masks'/row['key']/f'{j+1:02d}.png';asset.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(binary*255).save(asset);hashes[asset.relative_to(OUT).as_posix()]=digest(asset)
            saved.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(saved,x=np.asarray(values,np.float32).reshape(-1,FEATURES))
            write_json(saved.with_suffix('.json'),dict(key=row['key'],image_sha256=row['image_sha256'],
                       label_sha256=row['label_sha256'],mask_hashes=hashes,features_sha256=digest(saved)))
            mined.extend(values)
        if (i+1)%25==0:
            state=dict(stage='hard_negative_mining',images=i+1,total=len(rows),native_negative_rois=len(mined))
            write_json(OUT/'status.json',state);print(json.dumps(state),flush=True)
    return np.asarray(mined,np.float32).reshape(-1,FEATURES)


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch_utils.NUM_THREADS=1;torch.manual_seed(SEED)
    if (OUT/'model.pt').exists():
        raise FileExistsError('Foreground checkpoint already exists')
    negatives=mine()
    with np.load(RUN/'verifier_v1/features.npz') as cached:
        x=np.concatenate([pooled_features(cached['x']),pooled_features(negatives)])
        y=np.concatenate([(cached['y']!=9).astype(np.float32),np.zeros(len(negatives),np.float32)])
        weights=np.concatenate([cached['weight'],np.full(len(negatives),4.,np.float32)])
        vx=pooled_features(cached['vx']);vy=(cached['vy']!=9).astype(np.float32)
    x,vx=torch.from_numpy(x),torch.from_numpy(vx)
    y,vy,w=torch.from_numpy(y),torch.from_numpy(vy),torch.from_numpy(weights)
    mean,scale=x.mean(0),x.std(0).clamp_min(.05)
    x,vx=((x-mean)/scale).clamp(-6,6),((vx-mean)/scale).clamp(-6,6)
    # Equal total foreground/background weighting; real training proposals retain their weight.
    for label in [0.,1.]:
        w[y==label]*=len(y)/(2*w[y==label].sum())
    model=ForegroundNetwork();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best,best_epoch,best_state,curve=float('inf'),0,None,[]
    for epoch in range(100):
        tick=time.perf_counter();model.train()
        for ids in torch.randperm(len(y)).split(128):
            optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ids]),y[ids],reduction='none')
            (loss*w[ids]).mean().backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits=torch.cat([model(batch) for batch in vx.split(512)])
            individual=torch.nn.functional.binary_cross_entropy_with_logits(logits,vy,reduction='none')
            value=float((individual[vy==0].mean()+individual[vy==1].mean())/2)
            probability=torch.sigmoid(logits)
        entry=dict(epoch=epoch+1,balanced_validation_loss=value,foreground_recall=float((probability[vy==1]>=.5).float().mean()),
                   background_rejection=float((probability[vy==0]<.5).float().mean()),seconds=time.perf_counter()-tick)
        curve.append(entry);print(json.dumps(entry),flush=True)
        write_json(OUT/'status.json',dict(stage='training',**entry))
        if value<best:
            best,best_epoch,best_state=value,epoch+1,copy.deepcopy(model.state_dict())
        if epoch+1-best_epoch>=12:
            break
    torch.save(dict(version=1,mean=mean,scale=scale,state_dict=best_state),OUT/'model.pt')
    write_json(OUT/'metadata.json',dict(best_epoch=best_epoch,curve=curve,model_sha256=digest(OUT/'model.pt'),
               hard_negative_rois=len(negatives),training_rois=len(y),development_validation_rois=len(vy),
               test_accessed=False,deployed=False,scope='binary ROI development; complete-scene evaluation still required'))
    write_json(OUT/'status.json',dict(stage='complete',best_epoch=best_epoch,deployed=False))


if __name__=='__main__':
    main()
