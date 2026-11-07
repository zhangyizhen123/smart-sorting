"""Develop a class-only expert consultation; detection geometry remains fixed."""
from itertools import product
import json
from pathlib import Path
import copy
import torch
from specialist_shapes import consult,specialist_matches
from tune_agreement_foreground import publish
from evaluate_replay_optimization import ROOT,digest,write_json
from evaluate_focused import score_records
from compare_optimization_candidates import compare

RUN=ROOT/'runs/segment/replay_optimization_v3'


def main():
    torch.set_num_threads(1)
    primary_name='guided_agreement_full_v1_foreground';secondary_name='v1_deployed'
    source=RUN/'evaluation'/primary_name;specialist=RUN/'evaluation'/secondary_name
    work=RUN/'diagnostics/specialist_v1'
    freeze=dict(primary=primary_name,secondary=secondary_name,primary_summary_sha256=digest(source/'summary.json'),
                secondary_summary_sha256=digest(specialist/'summary.json'),algorithm_sha256=digest(ROOT/'vision/scripts/specialist_shapes.py'),
                evaluator_sha256=digest(Path(__file__)),test_accessed=False)
    if (work/'freeze.json').exists() and json.loads((work/'freeze.json').read_text(encoding='utf-8'))!=freeze:
        raise ValueError('Specialist inputs changed')
    write_json(work/'freeze.json',freeze)
    pairs=[]
    for file in sorted((source/'records').glob('*.json')):
        a=json.loads(file.read_text(encoding='utf-8'));b=json.loads((specialist/'records'/file.name).read_text(encoding='utf-8'))
        if any(a[k]!=b[k] for k in ['image_sha256','label_sha256','truth_classes']):
            raise ValueError('Specialist scenes differ')
        pairs.append((a,b,specialist_matches(a['predictions'],b['predictions'])))
    old=[json.loads((RUN/'evaluation'/n/'summary.json').read_text(encoding='utf-8')) for n in ['v1_deployed','v2_deployed']]
    incumbent=json.loads((RUN/'evaluation/pilot_640/summary.json').read_text(encoding='utf-8'))
    trials=[]
    for maximum,minimum,margin,allowed in product([.6,.8,1.01],[.5,.7,.85],[0.,.1,.2],[[6,7],[0,6,7,8]]):
        policy=dict(match_iou=.5,maximum_primary_confidence=maximum,minimum_specialist_confidence=minimum,
                    confidence_margin=margin,allowed_classes=allowed)
        rows=[]
        for a,b,m in pairs:
            r={**a,'predictions':consult(a['predictions'],b['predictions'],policy,m)};rows.append(r)
        scores={d:score_records([r for r in rows if r['domain']==d],.5) for d in ['legacy','focused','real']}
        scores['synthetic_validation']=score_records([r for r in rows if r['domain']!='real'],.5)
        guards=dict(recognition=scores['synthetic_validation']['strict_instance_score']>max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in old),
                    styles=all(scores[d]['strict_instance_score']>=max(r['groups'][d]['scores']['0.5']['strict_instance_score'] for r in old) for d in ['legacy','focused']),
                    real_fit=scores['real']['strict_instance_score']>=incumbent['groups']['real']['scores']['0.5']['strict_instance_score'])
        trials.append(dict(policy=policy,partial_guards=guards,strict_score=scores['synthetic_validation']['strict_instance_score'],
                           legacy_score=scores['legacy']['strict_instance_score'],focused_score=scores['focused']['strict_instance_score'],
                           real_fit_score=scores['real']['strict_instance_score']))
        if len(trials)%18==0:
            print(json.dumps(dict(policies=len(trials),total=54)),flush=True)
    eligible=[r for r in trials if r['partial_guards']['real_fit']]
    selected=max(eligible or trials,key=lambda r:(all(r['partial_guards'].values()),r['strict_score'],r['legacy_score']))
    write_json(work/'selection.json',dict(selected=selected,trials=trials,test_accessed=False))
    final=[];by_key={a['key']:a for a,b,m in pairs}
    for a,b,m in pairs:
        after=copy.deepcopy(a);after['predictions']=consult(a['predictions'],b['predictions'],selected['policy'],m)
        after['inference_seconds']+=b['inference_seconds'];final.append(after)
    original=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    provenance={**original['provenance'],'role':'candidate_specialist','specialist':freeze,'specialist_policy':selected['policy'],
                'selection_sha256':digest(work/'selection.json')}
    def asset(row,p,j):
        relative=f"masks/{row['key']}/{j+1:02d}.png"
        return source/relative,by_key[row['key']]['mask_hashes'][relative]
    name='guided_foreground_specialist_v1'
    publish(name,final,provenance,asset)
    print(json.dumps(dict(selected=selected,joint_gate=compare(RUN)['candidates'][name])),flush=True)


if __name__=='__main__':
    main()
