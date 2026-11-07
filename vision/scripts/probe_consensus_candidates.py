"""Develop two-detector agreement policies on frozen validation, never test."""
from __future__ import annotations
import argparse
import copy
from itertools import product
import json
from pathlib import Path
import shutil

import torch
from consensus_shapes import box_ious, select_predictions
from compare_optimization_candidates import compare
from evaluate_focused import score_records
from evaluate_replay_optimization import ROOT, digest, summarize, write_json

RUN=ROOT/'runs/segment/replay_optimization_v3'


def combine_record(primary,secondary,policy,overlap):
    if any(primary[k] != secondary[k] for k in ['key','image_sha256','label_sha256','truth_classes']):
        raise ValueError('Different scene inputs')
    predictions=select_predictions(primary['predictions'],secondary['predictions'],policy,overlap)
    output={k:v for k,v in primary.items() if k not in {'predictions','ious','boundary_ious','center_errors_pixels','mask_hashes'}}
    output['predictions']=predictions
    output['inference_seconds']=primary['inference_seconds']+secondary['inference_seconds']
    for field in ['ious','boundary_ious','center_errors_pixels']:
        output[field]=[[({'primary':primary,'secondary':secondary}[p['consensus_source']])[field][i][p['consensus_source_index']]
                        for p in predictions] for i in range(len(primary['truth_classes']))]
    return output


def run(primary_name,secondary_name):
    sources={k:RUN/'evaluation'/v for k,v in [('primary',primary_name),('secondary',secondary_name)]}
    for source in sources.values():
        if json.loads((source/'status.json').read_text(encoding='utf-8'))['stage']!='complete':
            raise ValueError('Incomplete source evaluation')
    name=primary_name+'_consensus'
    work=RUN/'diagnostics/consensus_v1'/name
    freeze={k+'_summary_sha256':digest(v/'summary.json') for k,v in sources.items()}
    freeze.update(primary=primary_name,secondary=secondary_name,algorithm_sha256=digest(ROOT/'vision/scripts/consensus_shapes.py'),
                  evaluator_sha256=digest(Path(__file__)),test_used_for_selection=False)
    if (work/'freeze.json').exists() and json.loads((work/'freeze.json').read_text(encoding='utf-8'))!=freeze:
        raise ValueError('Frozen consensus run changed')
    write_json(work/'freeze.json',freeze)
    pairs=[]
    for file in sorted((sources['primary']/'records').glob('*.json')):
        a=json.loads(file.read_text(encoding='utf-8'))
        b=json.loads((sources['secondary']/'records'/file.name).read_text(encoding='utf-8'))
        pairs.append((a,b,box_ious(a['predictions'],b['predictions'])))
    if len(pairs)!=1154:
        raise ValueError('Expected full development population')
    baselines=[json.loads((RUN/'evaluation'/n/'summary.json').read_text(encoding='utf-8')) for n in ['v1_deployed','v2_deployed']]
    incumbent=json.loads((RUN/'evaluation/pilot_640/summary.json').read_text(encoding='utf-8'))
    trials=[]
    for match,keep,add,mode in product([.3,.5],[.25,.6,.85,1.01],[.6,.85,1.01],['primary','stronger_secondary']):
        policy=dict(match_iou=match,unmatched_primary_confidence=keep,unmatched_secondary_confidence=add,
                    class_source=mode,secondary_margin=.1)
        revised=[combine_record(a,b,policy,o) for a,b,o in pairs]
        groups={d:[r for r in revised if r['domain']==d] for d in ['legacy','focused','negative','real']}
        groups['synthetic_validation']=[r for r in revised if r['domain']!='real']
        scores={d:score_records(rows,.5) for d,rows in groups.items()}
        fp=scores['synthetic_validation']['spurious_detections']/1108
        negative=sum(bool(r['predictions']) for r in groups['negative'])/100
        empty=[r for r in revised if r.get('category')=='empty_tray']
        real_alarm=sum(bool(r['predictions']) for r in empty)/len(empty)
        guards={
            'recognition':scores['synthetic_validation']['strict_instance_score']>max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in baselines),
            'false_positives':fp<min(r['groups']['synthetic_validation']['false_positives_per_image'] for r in baselines),
            'styles':all(scores[d]['strict_instance_score']>=max(r['groups'][d]['scores']['0.5']['strict_instance_score'] for r in baselines) for d in ['legacy','focused']),
            'real_fit':scores['real']['strict_instance_score']>=incumbent['groups']['real']['scores']['0.5']['strict_instance_score'],
            'real_empty':real_alarm<=incumbent['groups']['real_empty']['empty_false_detection_rate'],
            'negative':negative<=min(r['groups']['negative']['empty_false_detection_rate'] for r in baselines)}
        trials.append(dict(policy=policy,partial_guards=guards,strict_score=scores['synthetic_validation']['strict_instance_score'],
                           false_positives_per_image=fp,negative_alarm=negative,real_empty_alarm=real_alarm,
                           legacy_score=scores['legacy']['strict_instance_score'],focused_score=scores['focused']['strict_instance_score'],
                           real_fit_score=scores['real']['strict_instance_score']))
        if len(trials)%12==0:
            print(json.dumps({'source':primary_name,'policies':len(trials),'total':48}),flush=True)
    eligible=[r for r in trials if r['partial_guards']['real_fit'] and r['partial_guards']['real_empty']]
    selected=max(eligible or trials,key=lambda r:(all(r['partial_guards'].values()),sum(r['partial_guards'].values()),r['strict_score'],-r['false_positives_per_image']))
    write_json(work/'selection.json',dict(selected=selected,trials=trials,scope='development only; geometry gate still required'))
    output=RUN/'evaluation'/name
    primary_report=json.loads((sources['primary']/'summary.json').read_text(encoding='utf-8'))
    provenance={**primary_report['provenance'],'role':'candidate_consensus','consensus':freeze,'consensus_policy':selected['policy']}
    if (output/'freeze.json').exists() and json.loads((output/'freeze.json').read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Registered consensus changed')
    write_json(output/'freeze.json',provenance)
    final=[]
    for a,b,overlap in pairs:
        row=combine_record(a,b,selected['policy'],overlap);row['mask_hashes']={}
        for j,p in enumerate(row['predictions']):
            old_asset=f"masks/{row['key']}/{p['consensus_source_index']+1:02d}.png"
            old_row={'primary':a,'secondary':b}[p['consensus_source']]
            source=sources[p['consensus_source']]/old_asset
            if digest(source)!=old_row['mask_hashes'][old_asset]:
                raise ValueError('Source mask changed')
            asset=f"masks/{row['key']}/{j+1:02d}.png"
            target=output/asset;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);row['mask_hashes'][asset]=digest(target)
        write_json(output/'records'/(row['key']+'.json'),row);final.append(row)
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
    write_json(output/'summary.json',report);write_json(output/'status.json',dict(stage='complete',images=1154,deployed=False))
    print(json.dumps({'selected':selected,'joint_gate':compare(RUN)['candidates'][name]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('primary');parser.add_argument('secondary')
    args=parser.parse_args()
    if any(not n or not all(c.isalnum() or c=='_' for c in n) for n in [args.primary,args.secondary]):
        raise ValueError('Safe evaluation names required')
    torch.set_num_threads(1)
    run(args.primary,args.secondary)
