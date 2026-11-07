"""Joint development selection of class agreement and learned background filtering."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
import shutil
import torch
from consensus_shapes import box_ious
from probe_consensus_candidates import combine_record
from evaluate_foreground_gate import filter_record
from evaluate_replay_optimization import ROOT,digest,summarize,write_json
from evaluate_focused import score_records
from compare_optimization_candidates import compare

RUN=ROOT/'runs/segment/replay_optimization_v3'
NAMES={'primary':'v2_deployed_guided_verified','secondary':'pilot_640_guided_verified'}
BASE='guided_agreement_full_v1'


def publish(name,rows,provenance,asset_source):
    out=RUN/'evaluation'/name
    if (out/'freeze.json').exists() and json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Candidate identity changed')
    write_json(out/'freeze.json',provenance)
    for row in rows:
        row['mask_hashes']={}
        for j,p in enumerate(row['predictions']):
            source,expected=asset_source(row,p,j)
            if digest(source)!=expected:
                raise ValueError('Source geometry changed')
            asset=f"masks/{row['key']}/{j+1:02d}.png";target=out/asset;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);row['mask_hashes'][asset]=digest(target)
        write_json(out/'records'/(row['key']+'.json'),row)
    groups={'synthetic_validation':[r for r in rows if r['domain']!='real'],
            'positive_validation':[r for r in rows if r['domain'] in {'legacy','focused'}],
            **{d:[r for r in rows if r['domain']==d] for d in ['legacy','focused','negative','real']},
            'real_empty':[r for r in rows if r['category']=='empty_tray'],
            'real_white':[r for r in rows if r['category']=='white_cube']}
    report=dict(name=name,provenance=provenance,image_count=1154,validation_images=1108,real_train_fit_images=46,
                real_independent_test_images=0,test_used_for_selection=False,groups={d:summarize(v) for d,v in groups.items()})
    for d in ['negative','real_empty']:
        report['groups'][d]['empty_false_detection_images']=sum(bool(r['predictions']) for r in groups[d])
        report['groups'][d]['empty_false_detection_rate']=report['groups'][d]['empty_false_detection_images']/len(groups[d])
    write_json(out/'summary.json',report);write_json(out/'status.json',dict(stage='complete',images=1154,deployed=False))
    return report


def run():
    torch.set_num_threads(1)
    sources={k:RUN/'evaluation'/v for k,v in NAMES.items()}
    source_reports={k:json.loads((v/'summary.json').read_text(encoding='utf-8')) for k,v in sources.items()}
    probability_root=RUN/'foreground_gate_v1/evaluation'/BASE
    probability_freeze=json.loads((probability_root/'freeze.json').read_text(encoding='utf-8'))
    work=RUN/'diagnostics/agreement_foreground_v2'
    freeze=dict(source_sha256={k:digest(v/'summary.json') for k,v in sources.items()},
                probabilities_freeze_sha256=digest(probability_root/'freeze.json'),runner_sha256=digest(Path(__file__)),
                algorithm_sha256=digest(ROOT/'vision/scripts/consensus_shapes.py'),test_accessed=False)
    if (work/'freeze.json').exists() and json.loads((work/'freeze.json').read_text(encoding='utf-8'))!=freeze:
        raise ValueError('Joint selection inputs changed')
    write_json(work/'freeze.json',freeze)
    pairs=[];probabilities={};times={};originals={}
    for file in sorted((sources['primary']/'records').glob('*.json')):
        a=json.loads(file.read_text(encoding='utf-8'))
        b=json.loads((sources['secondary']/'records'/file.name).read_text(encoding='utf-8'))
        cached=json.loads((probability_root/'probabilities'/file.name).read_text(encoding='utf-8'))
        base_path=RUN/'evaluation'/BASE/'records'/file.name
        if digest(base_path)!=cached['source_record_sha256']:
            raise ValueError('Cached foreground source changed')
        original=json.loads(base_path.read_text(encoding='utf-8'))
        probabilities[a['key']]=cached['probabilities'];times[a['key']]=cached['seconds'];originals[a['key']]=original
        pairs.append((a,b,box_ious(a['predictions'],b['predictions'])))
    if len(pairs)!=1154:
        raise ValueError('Full validation population required')
    old=[json.loads((RUN/'evaluation'/n/'summary.json').read_text(encoding='utf-8')) for n in ['v1_deployed','v2_deployed']]
    incumbent=json.loads((RUN/'evaluation/pilot_640/summary.json').read_text(encoding='utf-8'))
    trials=[]
    base_policy=dict(match_iou=.3,unmatched_primary_confidence=.25,unmatched_secondary_confidence=.6,class_source='stronger_secondary')
    for margin in [-.1,0.,.05,.1,.2]:
        policy={**base_policy,'secondary_margin':margin}
        combined=[combine_record(a,b,policy,o) for a,b,o in pairs]
        for r in combined:
            expected=originals[r['key']]['predictions']
            if [(p['consensus_source'],p['consensus_source_index']) for p in r['predictions']]!=[(p['consensus_source'],p['consensus_source_index']) for p in expected]:
                raise ValueError('Class-only policy changed geometry or order')
        for threshold in [.05,.1,.2,.3,.5,.7]:
            revised=[filter_record(r,probabilities[r['key']],threshold) for r in combined]
            groups={d:[r for r in revised if r['domain']==d] for d in ['legacy','focused','negative','real']}
            groups['synthetic_validation']=[r for r in revised if r['domain']!='real']
            scores={d:score_records(v,.5) for d,v in groups.items()}
            fp=scores['synthetic_validation']['spurious_detections']/1108
            negative=sum(bool(r['predictions']) for r in groups['negative'])/100
            empty=[r for r in revised if r['category']=='empty_tray'];real_alarm=sum(bool(r['predictions']) for r in empty)/len(empty)
            guards=dict(recognition=scores['synthetic_validation']['strict_instance_score']>max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in old),
                        false_positives=fp<min(r['groups']['synthetic_validation']['false_positives_per_image'] for r in old),
                        styles=all(scores[d]['strict_instance_score']>=max(r['groups'][d]['scores']['0.5']['strict_instance_score'] for r in old) for d in ['legacy','focused']),
                        real_fit=scores['real']['strict_instance_score']>=incumbent['groups']['real']['scores']['0.5']['strict_instance_score'],
                        real_empty=real_alarm<=incumbent['groups']['real_empty']['empty_false_detection_rate'],
                        negative=negative<=min(r['groups']['negative']['empty_false_detection_rate'] for r in old))
            trials.append(dict(policy=policy,foreground_threshold=threshold,partial_guards=guards,
                               strict_score=scores['synthetic_validation']['strict_instance_score'],false_positives_per_image=fp,
                               legacy_score=scores['legacy']['strict_instance_score'],focused_score=scores['focused']['strict_instance_score'],
                               real_fit_score=scores['real']['strict_instance_score'],negative_alarm=negative,real_empty_alarm=real_alarm))
        print(json.dumps(dict(class_margin=margin,policies=len(trials),total=30)),flush=True)
    eligible=[r for r in trials if r['partial_guards']['real_fit'] and r['partial_guards']['real_empty']]
    selected=max(eligible or trials,key=lambda r:(all(r['partial_guards'].values()),sum(r['partial_guards'].values()),r['strict_score'],-r['false_positives_per_image']))
    write_json(work/'selection.json',dict(selected=selected,trials=trials,test_accessed=False,geometry_gate_still_required=True))
    combined=[combine_record(a,b,selected['policy'],o) for a,b,o in pairs]
    intermediate='guided_agreement_tuned_v1'
    provenance={**source_reports['primary']['provenance'],'role':'candidate_consensus',
                'consensus':dict(**NAMES,**{k+'_summary_sha256':digest(v/'summary.json') for k,v in sources.items()},algorithm_sha256=freeze['algorithm_sha256']),
                'consensus_policy':selected['policy'],'joint_selection_sha256':digest(work/'selection.json')}
    def original_asset(row,p,index):
        file=RUN/'evaluation'/BASE/'masks'/row['key']/f'{index+1:02d}.png'
        asset=file.relative_to(RUN/'evaluation'/BASE).as_posix()
        return file,originals[row['key']]['mask_hashes'][asset]
    publish(intermediate,combined,provenance,original_asset)
    filtered=[filter_record(r,probabilities[r['key']],selected['foreground_threshold']) for r in combined]
    by_key={r['key']:r for r in combined}
    for row in filtered:
        row['inference_seconds']+=times[row['key']]
    def selected_asset(row,p,index):
        asset=f"masks/{row['key']}/{p['foreground_source_index']+1:02d}.png"
        return RUN/'evaluation'/intermediate/asset,by_key[row['key']]['mask_hashes'][asset]
    p={**provenance,**probability_freeze,'role':'candidate_foreground','source_evaluation':intermediate,
       'source_summary_sha256':digest(RUN/'evaluation'/intermediate/'summary.json'),
       'source_freeze_sha256':digest(RUN/'evaluation'/intermediate/'freeze.json'),
       'foreground_threshold':selected['foreground_threshold'],'joint_selection_sha256':digest(work/'selection.json')}
    publish(intermediate+'_foreground',filtered,p,selected_asset)
    print(json.dumps(dict(selected=selected,joint_gate=compare(RUN)['candidates'][intermediate+'_foreground'])),flush=True)


if __name__=='__main__':
    run()
