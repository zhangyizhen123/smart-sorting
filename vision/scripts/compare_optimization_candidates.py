"""Development-only joint gate; a passing candidate still requires sealed testing."""
from pathlib import Path
import json
import numpy as np
from evaluate_focused import match_instances
from evaluate_replay_optimization import ROOT,write_json


def matched_geometry(directory):
    values={}
    for path in (directory/'records').glob('*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['domain']=='real':continue
        n,p=len(row['truth_classes']),len(row['predictions'])
        matrix=np.asarray(row['ious']).reshape(n,p)
        for i,j in match_instances(matrix,.5):
            values[(row['key'],i)]=(row['boundary_ious'][i][j],float(matrix[i,j]))
    return values


def compare(root):
    evaluation=root/'evaluation'
    reports={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in evaluation.glob('*/summary.json')}
    baseline_names=['v1_deployed','v2_deployed']
    if any(name not in reports for name in baseline_names):return {'stage':'waiting_for_baselines'}
    baselines={name:reports[name] for name in baseline_names}
    geometry={name:matched_geometry(evaluation/name) for name in baseline_names}
    baseline_strict=max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in baselines.values())
    baseline_fp=min(r['groups']['synthetic_validation']['false_positives_per_image'] for r in baselines.values())
    baseline_boundary=max(r['groups']['synthetic_validation']['geometry']['boundary_iou_with_unmatched_zero'] for r in baselines.values())
    candidates={}
    for name,report in reports.items():
        if name in baseline_names or name=='pilot_640':continue
        group=report['groups']['synthetic_validation']
        common_geometry=matched_geometry(evaluation/name)
        paired={}
        for old,values in geometry.items():
            keys=sorted(common_geometry.keys()&values.keys())
            paired[old]={'matched_instances':len(keys),
                         'mean_boundary_delta':float(np.mean([common_geometry[key][0]-values[key][0] for key in keys])) if keys else None,
                         'mean_mask_iou_delta':float(np.mean([common_geometry[key][1]-values[key][1] for key in keys])) if keys else None}
        gates={
            'recognition_strictly_better_than_both':group['scores']['0.5']['strict_instance_score']>baseline_strict+1e-12,
            'false_positives_strictly_lower_than_both':group['false_positives_per_image']<baseline_fp-1e-12,
            'coverage_boundary_strictly_better_than_both':group['geometry']['boundary_iou_with_unmatched_zero']>baseline_boundary+1e-12,
            'common_match_boundary_better_than_each':all(v['mean_boundary_delta'] is not None and v['mean_boundary_delta']>1e-12 for v in paired.values()),
            'common_match_mask_iou_not_worse':all(v['mean_mask_iou_delta'] is not None and v['mean_mask_iou_delta']>=-1e-12 for v in paired.values()),
            'both_styles_do_not_regress':all(report['groups'][domain]['scores']['0.5']['strict_instance_score']>=max(r['groups'][domain]['scores']['0.5']['strict_instance_score'] for r in baselines.values())-1e-12 for domain in ['legacy','focused']),
            'negative_scene_false_alarm_does_not_regress':report['groups']['negative']['empty_false_detection_rate']<=min(r['groups']['negative']['empty_false_detection_rate'] for r in baselines.values())+1e-12}
        if 'pilot_640' in reports:
            incumbent=reports['pilot_640']['groups']
            gates['real_training_fit_does_not_regress']=report['groups']['real']['scores']['0.5']['strict_instance_score']>=incumbent['real']['scores']['0.5']['strict_instance_score']-1e-12
            gates['real_empty_false_alarm_does_not_regress']=report['groups']['real_empty']['empty_false_detection_rate']<=incumbent['real_empty']['empty_false_detection_rate']+1e-12
        candidates[name]={'passes_development_gate':all(gates.values()),'gates':gates,'paired_geometry':paired,
                          'strict_score':group['scores']['0.5']['strict_instance_score'],
                          'false_positives_per_image':group['false_positives_per_image'],
                          'coverage_boundary_iou':group['geometry']['boundary_iou_with_unmatched_zero'],
                          'real_fit_score':report['groups']['real']['scores']['0.5']['strict_instance_score']}
    passing=[name for name,value in candidates.items() if value['passes_development_gate']]
    selected=max(passing,key=lambda name:(candidates[name]['strict_score'],candidates[name]['coverage_boundary_iou'],-candidates[name]['false_positives_per_image'])) if passing else None
    result={'stage':'development_comparison','baselines':baseline_names,'baseline_best_strict':baseline_strict,
            'baseline_lowest_false_positives_per_image':baseline_fp,'baseline_best_coverage_boundary':baseline_boundary,
            'candidates':candidates,'selected_for_sealed_test':selected,'sealed_test_evaluated':False,
            'automatic_deployment':False,'real_independent_test_available':False}
    write_json(root/'development_comparison.json',result)
    return result


if __name__=='__main__':print(json.dumps(compare(ROOT/'runs/segment/replay_optimization_v3'),ensure_ascii=False,indent=2))
