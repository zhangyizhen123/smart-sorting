"""Compare foreground safeguards on the original DEVELOPMENT cache only."""
import copy
from itertools import product
import json
from pathlib import Path
import torch
from evaluate_focused import score_records
from evaluate_replay_optimization import digest,write_json
from protected_foreground import retained_indices
from specialist_shapes import consult

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/segment/replay_optimization_v3'
OUT=ROOT/'runs/segment/joint_round_v4'
SPECIALIST=dict(match_iou=.5,maximum_primary_confidence=1.01,minimum_specialist_confidence=.5,confidence_margin=0.,allowed_classes=[6,7])


def revise(row, probabilities, expert, policy, specialist_policy=None):
    indices,protected=retained_indices(row['predictions'],probabilities,policy)
    result={k:v for k,v in row.items() if k not in {'predictions','ious','boundary_ious','center_errors_pixels'}}
    predictions=[]
    for index,rescued in zip(indices,protected,strict=True):
        p=copy.deepcopy(row['predictions'][index]);p.update(foreground_probability=float(probabilities[index]),
                         foreground_source_index=index,foreground_protected=rescued)
        predictions.append(p)
    result['predictions']=consult(predictions,expert['predictions'],specialist_policy or SPECIALIST)
    for field in ['ious','boundary_ious','center_errors_pixels']:
        result[field]=[[values[j] for j in indices] for values in row[field]]
    return result


def main():
    torch.set_num_threads(1)
    source=OLD/'evaluation/guided_agreement_full_v1';expert=OLD/'evaluation/v1_deployed'
    probabilities=OLD/'foreground_gate_v1/evaluation/guided_agreement_full_v1/probabilities'
    freeze=dict(source_summary_sha256=digest(source/'summary.json'),expert_summary_sha256=digest(expert/'summary.json'),
                algorithm_sha256=digest(ROOT/'vision/scripts/protected_foreground.py'),specialist_policy=SPECIALIST,
                scope='original development only; sealed_test records are never read')
    rows=[]
    for path in sorted((source/'records').glob('*.json')):
        item=json.loads((probabilities/path.name).read_text())
        if item['source_record_sha256']!=digest(path): raise ValueError('Probability cache changed')
        rows.append((json.loads(path.read_text()),item['probabilities'],json.loads((expert/'records'/path.name).read_text())))
    if len(rows)!=1154: raise ValueError('Incomplete development population')
    trials=[]
    for threshold,confidence,partner in product([.1,.2,.3],[.5,.7,.85,1.01],[True,False]):
        policy=dict(foreground_threshold=threshold,protect_confidence=confidence,require_partner=partner)
        revised=[revise(row,p,expert,policy) for row,p,expert in rows]
        groups={d:[r for r in revised if r['domain']==d] for d in ['legacy','focused','negative','real']}
        groups['synthetic_validation']=[r for r in revised if r['domain']!='real']
        groups['real_empty']=[r for r in revised if r.get('category')=='empty_tray']
        scores={d:score_records(rs,.5) for d,rs in groups.items()}
        trials.append(dict(policy=policy,strict=scores['synthetic_validation']['strict_instance_score'],
                           legacy=scores['legacy']['strict_instance_score'],focused=scores['focused']['strict_instance_score'],
                           extra_fp=scores['synthetic_validation']['spurious_detections'],
                           real_fit=scores['real']['strict_instance_score'],
                           negative_alarm=sum(bool(r['predictions']) for r in groups['negative'])/len(groups['negative']),
                           real_empty_alarm=sum(bool(r['predictions']) for r in groups['real_empty'])/len(groups['real_empty']),
                           protected=sum(p['foreground_protected'] for r in revised for p in r['predictions'])))
    write_json(OUT/'original_development_protection_probe.json',dict(freeze=freeze,selection_closed=False,
               expanded_development_required=True,trials=trials))
    print(json.dumps(sorted(trials,key=lambda r:(r['focused'],r['legacy'],r['strict']),reverse=True)[:8]),flush=True)


if __name__=='__main__':main()
