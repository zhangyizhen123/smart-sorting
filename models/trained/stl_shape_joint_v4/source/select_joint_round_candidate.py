"""Choose one recipe using original and newly expanded DEVELOPMENT data only."""
import copy
from itertools import product
import json
from pathlib import Path
import shutil
import torch
from consensus_evidence import add_evidence,reclassify
from evaluate_focused import score_records,match_instances
from evaluate_replay_optimization import digest,summarize,write_json
from protected_foreground import retained_indices
from specialist_shapes import consult,specialist_matches
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/segment/replay_optimization_v3'
RUN=ROOT/'runs/segment/joint_round_v4'


def load_population(kind):
    if kind=='original':
        source=OLD/'evaluation/guided_agreement_full_v1';expert=OLD/'evaluation/v1_deployed'
        primary=OLD/'evaluation/v2_deployed_guided_verified';secondary=OLD/'evaluation/pilot_640_guided_verified'
        probability_root=OLD/'foreground_gate_v1/evaluation/guided_agreement_full_v1/probabilities'
        baseline_root=OLD/'evaluation'
    else:
        source=RUN/'development/base_agreement';expert=RUN/'development/v1_deployed';baseline_root=RUN/'development'
        if json.loads((RUN/'development/status.json').read_text())['stage']!='complete': raise ValueError('Expanded development is incomplete')
    rows=[]
    for path in sorted((source/'records').glob('*.json')):
        row=json.loads(path.read_text());specialist=json.loads((expert/'records'/path.name).read_text())
        if kind=='original':
            a=json.loads((primary/'records'/path.name).read_text());b=json.loads((secondary/'records'/path.name).read_text())
            row['predictions']=[add_evidence(p,a['predictions'],b['predictions']) for p in row['predictions']]
            cached=json.loads((probability_root/path.name).read_text())
            if cached['source_record_sha256']!=digest(path): raise ValueError('Foreground cache differs')
            probabilities=cached['probabilities']
        else: probabilities=row['foreground_probabilities']
        rows.append((row,probabilities,specialist))
    if len(rows)!=(1154 if kind=='original' else 2900): raise ValueError('Incomplete development population')
    baselines={n:json.loads((baseline_root/n/'summary.json').read_text()) for n in ['v1_deployed','v2_deployed']}
    return rows,baselines,source


def filter_row(row,probabilities,policy):
    indices,protected=retained_indices(row['predictions'],probabilities,policy)
    result={k:v for k,v in row.items() if k not in {'predictions','ious','boundary_ious','center_errors_pixels'}}
    result['predictions']=[]
    for index,keep in zip(indices,protected,strict=True):
        p=copy.deepcopy(row['predictions'][index]);p.update(foreground_source_index=index,foreground_protected=keep,foreground_probability=float(probabilities[index]))
        result['predictions'].append(p)
    for field in ['ious','boundary_ious','center_errors_pixels']:
        result[field]=[[values[j] for j in indices] for values in row[field]]
    return result


def apply_classes(row,expert,policy,matches=None):
    return {**row,'predictions':consult(reclassify(row['predictions'],policy['secondary_margin']),expert['predictions'],policy['specialist'],matches)}


def quick_scores(rows,baselines):
    domains={d:[r for r in rows if r['domain']==d] for d in ['legacy','focused','negative','real']}
    synthetic=[r for r in rows if r['domain']!='real']
    scores={d:score_records(rs,.5) for d,rs in {**domains,'all':synthetic}.items()}
    old={n:r['groups'] for n,r in baselines.items()}
    strict=scores['all']['strict_instance_score'];fp=scores['all']['spurious_detections']/len(synthetic)
    alarm=sum(bool(r['predictions']) for r in domains['negative'])/len(domains['negative'])
    deltas={d:scores[d]['strict_instance_score']-max(r[d]['scores']['0.5']['strict_instance_score'] for r in old.values()) for d in ['legacy','focused']}
    gates=dict(recognition=strict>max(r['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in old.values()),
               false_positives=fp<min(r['synthetic_validation']['false_positives_per_image'] for r in old.values()),
               styles=all(v>=-1e-12 for v in deltas.values()),
               negative_alarm=alarm<=min(r['negative']['empty_false_detection_rate'] for r in old.values())+1e-12)
    result=dict(strict=strict,extra_fp=scores['all']['spurious_detections'],negative_alarm=alarm,style_deltas=deltas,
                legacy=scores['legacy']['strict_instance_score'],focused=scores['focused']['strict_instance_score'],gates=gates,
                clean_correct=scores['all']['clean_correct_instances'],truth_instances=scores['all']['instance_count'])
    if domains['real']:
        incumbent=json.loads((OLD/'evaluation/pilot_640/summary.json').read_text())['groups']
        empty=[r for r in rows if r.get('category')=='empty_tray']
        real_alarm=sum(bool(r['predictions']) for r in empty)/len(empty)
        gates.update(real_fit=scores['real']['strict_instance_score']>=incumbent['real']['scores']['0.5']['strict_instance_score']-1e-12,
                     real_empty=real_alarm<=incumbent['real_empty']['empty_false_detection_rate']+1e-12)
        result.update(real_fit=scores['real']['strict_instance_score'],real_empty_alarm=real_alarm)
    return result


def geometry_gates(rows,baseline_root):
    selected={}
    for row in rows:
        if row['domain']=='real': continue
        n,p=len(row['truth_classes']),len(row['predictions'])
        for i,j in match_instances(np.asarray(row['ious']).reshape(n,p),.5): selected[(row['key'],i)]=(row['boundary_ious'][i][j],row['ious'][i][j])
    comparisons={}
    summary=summarize([r for r in rows if r['domain']!='real'])
    for name in ['v1_deployed','v2_deployed']:
        paired=[]
        for path in (baseline_root/name/'records').glob('*.json'):
            row=json.loads(path.read_text())
            if row['domain']=='real': continue
            n,p=len(row['truth_classes']),len(row['predictions'])
            for i,j in match_instances(np.asarray(row['ious']).reshape(n,p),.5):
                key=(row['key'],i)
                if key in selected: paired.append([selected[key][0]-row['boundary_ious'][i][j],selected[key][1]-row['ious'][i][j]])
        old=json.loads((baseline_root/name/'summary.json').read_text())['groups']['synthetic_validation']
        delta=np.mean(paired,axis=0)
        comparisons[name]=dict(common_boundary_delta=float(delta[0]),common_mask_delta=float(delta[1]),
                              coverage_delta=summary['geometry']['boundary_iou_with_unmatched_zero']-old['geometry']['boundary_iou_with_unmatched_zero'])
    passed=all(v['common_boundary_delta']>0 and v['common_mask_delta']>=0 and v['coverage_delta']>0 for v in comparisons.values())
    return dict(passed=passed,comparisons=comparisons)


def main():
    torch.set_num_threads(1)
    populations={k:load_population(k) for k in ['original','expanded']};trials=[]
    for threshold,confidence in product([.1,.2,.3],[.5,.7,.85,1.01]):
        foreground=dict(foreground_threshold=threshold,protect_confidence=confidence,require_partner=True)
        filtered={k:[(filter_row(row,p,foreground),expert) for row,p,expert in values[0]] for k,values in populations.items()}
        matches={k:[specialist_matches(r['predictions'],e['predictions'],.5) for r,e in values] for k,values in filtered.items()}
        for margin,maximum,expert_margin in product([.1,.2,1.01],[.8,1.01],[0.,.1]):
            policy=dict(foreground=foreground,secondary_margin=margin,specialist=dict(match_iou=.5,maximum_primary_confidence=maximum,
                       minimum_specialist_confidence=.5,confidence_margin=expert_margin,allowed_classes=[6,7]))
            scores={k:quick_scores([apply_classes(r,e,policy,m) for (r,e),m in zip(values,matches[k],strict=True)],populations[k][1]) for k,values in filtered.items()}
            eligible=all(all(s['gates'].values()) for s in scores.values())
            buffer=min(v for s in scores.values() for v in s['style_deltas'].values())
            pooled=sum(s['clean_correct'] for s in scores.values())/sum(s['truth_instances']+s['extra_fp'] for s in scores.values())
            trials.append(dict(policy=policy,scores=scores,eligible_before_geometry=eligible,minimum_style_margin=buffer,pooled_strict=pooled))
        print(json.dumps(dict(stage='development_policy_comparison',policies=len(trials),total=144)),flush=True)
    ranked=sorted([t for t in trials if t['eligible_before_geometry']],key=lambda t:(t['minimum_style_margin'],t['pooled_strict']),reverse=True)
    selected=None
    for trial in ranked:
        final={k:[apply_classes(filter_row(r,p,trial['policy']['foreground']),e,trial['policy']) for r,p,e in values[0]] for k,values in populations.items()}
        geometry={k:geometry_gates(final[k],OLD/'evaluation' if k=='original' else RUN/'development') for k in populations}
        if all(g['passed'] for g in geometry.values()):
            selected={**trial,'geometry':geometry};break
    selection=dict(stage='development_selection',selected=selected,trials=trials,images_synthetic=4008,real_training_fit_images=46,
                   test_accessed=False,source_hashes={k:{n:digest((OLD/'evaluation' if k=='original' else RUN/'development')/n/'summary.json') for n in ['v1_deployed','v2_deployed']} for k in populations},
                   algorithm_source_sha256={n:digest(ROOT/'vision/scripts'/n) for n in ['select_joint_round_candidate.py','protected_foreground.py','consensus_evidence.py','specialist_shapes.py']},
                   base_summary_sha256={k:digest(v[2]/'summary.json') for k,v in populations.items()},
                   foreground_model_sha256=digest(OLD/'foreground_gate_v1/model.pt'),
                   dataset_manifest_sha256={k:digest(ROOT/path/'manifest.jsonl') for k,path in [('original','dataset/replay_optimization_v3'),('expanded','dataset/joint_round_v4_development')]},
                   selection_rule='All original and expanded gates; maximize worst style margin, then pooled strict score')
    write_json(RUN/'selection.json',selection)
    if selected is None:
        print(json.dumps(dict(selected=None,closest=sorted(trials,key=lambda t:(sum(sum(s['gates'].values()) for s in t['scores'].values()),t['minimum_style_margin'],t['pooled_strict']),reverse=True)[:3])),flush=True)
        return
    for kind,rows in final.items():
        out=RUN/'selected'/kind;source=populations[kind][2]
        for row in rows:
            hashes={}
            for j,prediction in enumerate(row['predictions']):
                previous=f"masks/{row['key']}/{prediction['foreground_source_index']+1:02d}.png";asset=f"masks/{row['key']}/{j+1:02d}.png"
                if digest(source/previous)!=row['mask_hashes'][previous]: raise ValueError('Selected source mask changed')
                target=out/asset;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/previous,target);hashes[asset]=digest(target)
            row['mask_hashes']=hashes;write_json(out/'records'/(row['key']+'.json'),row)
        groups={'synthetic_validation':summarize([r for r in rows if r['domain']!='real']),
                **{d:summarize([r for r in rows if r['domain']==d]) for d in ['legacy','focused','negative','real']}}
        for d in ['negative']:
            values=[r for r in rows if r['domain']==d];groups[d]['empty_false_detection_rate']=sum(bool(r['predictions']) for r in values)/len(values)
        original_provenance=json.loads((OLD/'evaluation/guided_agreement_full_v1/summary.json').read_text())['provenance']
        write_json(out/'summary.json',dict(name='protected_consensus_v4',groups=groups,provenance=original_provenance,
                                         selection_sha256=digest(RUN/'selection.json'),test_accessed=False))
    print(json.dumps(dict(selected=selected)),flush=True)


if __name__=='__main__':main()
