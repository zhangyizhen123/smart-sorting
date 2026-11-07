"""End-to-end development selection of a learned foreground filter."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
from compare_optimization_candidates import compare
from evaluate_focused import score_records
from evaluate_replay_optimization import ROOT,digest,summarize,write_json
from foreground_gate import ForegroundGate
from shape_verifier import features

RUN=ROOT/'runs/segment/replay_optimization_v3'
DATA=ROOT/'dataset/replay_optimization_v3'
GATE=RUN/'foreground_gate_v1'


def filter_record(record,probabilities,threshold):
    if len(record['predictions'])!=len(probabilities):
        raise ValueError('Foreground probability alignment mismatch')
    row=copy.deepcopy(record);row['predictions']=[];indices=[]
    for j,(pred,p) in enumerate(zip(record['predictions'],probabilities,strict=True)):
        if p<threshold:
            continue
        pred=copy.deepcopy(pred);pred['foreground_probability']=p;pred['foreground_source_index']=j
        row['predictions'].append(pred);indices.append(j)
    for field in ['ious','boundary_ious','center_errors_pixels']:
        row[field]=[[values[j] for j in indices] for values in record[field]]
    return row


def evaluate(source_name):
    source=RUN/'evaluation'/source_name;work=GATE/'evaluation'/source_name
    if json.loads((source/'status.json').read_text(encoding='utf-8'))['stage']!='complete':
        raise ValueError('Source evaluation incomplete')
    freeze=dict(source_summary_sha256=digest(source/'summary.json'),source_freeze_sha256=digest(source/'freeze.json'),
                foreground_model_sha256=digest(GATE/'model.pt'),foreground_algorithm_sha256=digest(ROOT/'vision/scripts/foreground_gate.py'),
                features_sha256=digest(ROOT/'vision/scripts/shape_verifier.py'),evaluator_sha256=digest(Path(__file__)))
    if (work/'freeze.json').exists() and json.loads((work/'freeze.json').read_text(encoding='utf-8'))!=freeze:
        raise ValueError('Foreground evaluation changed')
    write_json(work/'freeze.json',freeze)
    gate=ForegroundGate(GATE/'model.pt');rows=[];probabilities={};seconds={}
    files=sorted((source/'records').glob('*.json'))
    for i,path in enumerate(files):
        row=json.loads(path.read_text(encoding='utf-8'));saved=work/'probabilities'/path.name
        if saved.exists():
            item=json.loads(saved.read_text(encoding='utf-8'))
            if item['source_record_sha256']!=digest(path):
                raise ValueError('Source predictions changed')
        else:
            if digest(DATA/row['image'])!=row['image_sha256']:
                raise ValueError('Source image changed')
            rgb=np.asarray(Image.open(DATA/row['image']).convert('RGB'));values=[];tick=time.perf_counter()
            for j in range(len(row['predictions'])):
                asset=f"masks/{row['key']}/{j+1:02d}.png"
                if digest(source/asset)!=row['mask_hashes'][asset]:
                    raise ValueError('Source mask changed')
                values.append(features(rgb,np.asarray(Image.open(source/asset))>0))
            item=dict(source_record_sha256=digest(path),probabilities=gate.probabilities(np.stack(values)).tolist() if values else [],
                      seconds=time.perf_counter()-tick)
            write_json(saved,item)
        rows.append(row);probabilities[row['key']]=item['probabilities'];seconds[row['key']]=item['seconds']
        if (i+1)%100==0 or i+1==len(files):
            print(json.dumps(dict(source=source_name,foreground_images=i+1,total=len(files))),flush=True)
    if len(rows)!=1154:
        raise ValueError('Full development population required')
    old=[json.loads((RUN/'evaluation'/n/'summary.json').read_text(encoding='utf-8')) for n in ['v1_deployed','v2_deployed']]
    incumbent=json.loads((RUN/'evaluation/pilot_640/summary.json').read_text(encoding='utf-8'))
    trials=[]
    for threshold in [.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]:
        revised=[filter_record(r,probabilities[r['key']],threshold) for r in rows]
        groups={d:[r for r in revised if r['domain']==d] for d in ['legacy','focused','negative','real']}
        groups['synthetic_validation']=[r for r in revised if r['domain']!='real']
        scores={d:score_records(v,.5) for d,v in groups.items()}
        fp=scores['synthetic_validation']['spurious_detections']/1108
        negative=sum(bool(r['predictions']) for r in groups['negative'])/100
        empty=[r for r in revised if r['category']=='empty_tray']
        real_alarm=sum(bool(r['predictions']) for r in empty)/len(empty)
        guards=dict(recognition=scores['synthetic_validation']['strict_instance_score']>max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in old),
                    false_positives=fp<min(r['groups']['synthetic_validation']['false_positives_per_image'] for r in old),
                    styles=all(scores[d]['strict_instance_score']>=max(r['groups'][d]['scores']['0.5']['strict_instance_score'] for r in old) for d in ['legacy','focused']),
                    real_fit=scores['real']['strict_instance_score']>=incumbent['groups']['real']['scores']['0.5']['strict_instance_score'],
                    real_empty=real_alarm<=incumbent['groups']['real_empty']['empty_false_detection_rate'],
                    negative=negative<=min(r['groups']['negative']['empty_false_detection_rate'] for r in old))
        trials.append(dict(threshold=threshold,partial_guards=guards,strict_score=scores['synthetic_validation']['strict_instance_score'],
                           false_positives_per_image=fp,legacy_score=scores['legacy']['strict_instance_score'],
                           focused_score=scores['focused']['strict_instance_score'],real_fit_score=scores['real']['strict_instance_score'],
                           negative_alarm=negative,real_empty_alarm=real_alarm))
    eligible=[r for r in trials if r['partial_guards']['real_fit'] and r['partial_guards']['real_empty']]
    selected=max(eligible or trials,key=lambda r:(all(r['partial_guards'].values()),sum(r['partial_guards'].values()),r['strict_score'],-r['false_positives_per_image']))
    name=source_name+'_foreground';out=RUN/'evaluation'/name
    original=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    provenance={**original['provenance'],**freeze,'role':'candidate_foreground','source_evaluation':source_name,
                'foreground_threshold':selected['threshold']}
    if (out/'freeze.json').exists() and json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Registered foreground candidate changed')
    write_json(out/'freeze.json',provenance);final=[]
    for before in rows:
        after=filter_record(before,probabilities[before['key']],selected['threshold'])
        after['inference_seconds']+=seconds[before['key']];after['mask_hashes']={}
        for j,p in enumerate(after['predictions']):
            old_asset=f"masks/{before['key']}/{p['foreground_source_index']+1:02d}.png"
            asset=f"masks/{before['key']}/{j+1:02d}.png";target=out/asset;target.parent.mkdir(parents=True,exist_ok=True)
            if digest(source/old_asset)!=before['mask_hashes'][old_asset]:
                raise ValueError('Source geometry changed')
            shutil.copy2(source/old_asset,target);after['mask_hashes'][asset]=digest(target)
        write_json(out/'records'/(before['key']+'.json'),after);final.append(after)
    groups={'synthetic_validation':[r for r in final if r['domain']!='real'],
            'positive_validation':[r for r in final if r['domain'] in {'legacy','focused'}],
            **{d:[r for r in final if r['domain']==d] for d in ['legacy','focused','negative','real']},
            'real_empty':[r for r in final if r['category']=='empty_tray'],
            'real_white':[r for r in final if r['category']=='white_cube']}
    report=dict(name=name,provenance=provenance,image_count=1154,validation_images=1108,real_train_fit_images=46,
                real_independent_test_images=0,test_used_for_selection=False,groups={d:summarize(v) for d,v in groups.items()})
    for d in ['negative','real_empty']:
        report['groups'][d]['empty_false_detection_images']=sum(bool(r['predictions']) for r in groups[d])
        report['groups'][d]['empty_false_detection_rate']=report['groups'][d]['empty_false_detection_images']/len(groups[d])
    write_json(out/'selection.json',dict(selected=selected,trials=trials,test_accessed=False))
    write_json(out/'summary.json',report);write_json(out/'status.json',dict(stage='complete',images=1154,deployed=False))
    print(json.dumps(dict(selected=selected,joint_gate=compare(RUN)['candidates'][name])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source');args=parser.parse_args()
    if not args.source or not all(c.isalnum() or c=='_' for c in args.source):
        raise ValueError('Safe evaluation name required')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    evaluate(args.source)
