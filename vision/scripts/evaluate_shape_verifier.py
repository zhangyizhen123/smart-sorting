"""Select a verifier policy on development data and measure complete scenes."""
from __future__ import annotations
import argparse
import copy
from itertools import product
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
from evaluate_replay_optimization import ROOT, digest, summarize, write_json
from shape_verifier import BACKGROUND, ShapeVerifier, features

RUN = ROOT/'runs/segment/replay_optimization_v3'
DATA = ROOT/'dataset/replay_optimization_v3'
VERIFIER = RUN/'verifier_v1'


def apply_policy(record, probabilities, policy):
    result = copy.deepcopy(record)
    retained, predictions = [], []
    for j, (prediction, probability) in enumerate(zip(record['predictions'], probabilities, strict=True)):
        p = np.asarray(probability)
        if p[BACKGROUND] >= policy['background_probability']:
            continue
        revised = copy.deepcopy(prediction)
        winner = int(p[:BACKGROUND].argmax())
        if prediction['confidence'] < policy['maximum_base_confidence'] and p[winner] >= policy['class_probability']:
            revised['class_id'] = winner
        revised['verifier_class_probability'] = float(p[winner])
        revised['verifier_background_probability'] = float(p[BACKGROUND])
        revised['source_prediction_index'] = j
        retained.append(j);predictions.append(revised)
    result['predictions'] = predictions
    for key in ['ious', 'boundary_ious', 'center_errors_pixels']:
        result[key] = [[values[j] for j in retained] for values in record[key]]
    return result


def cache_probabilities(source_name):
    source = RUN/'evaluation'/source_name
    work = VERIFIER/'evaluation'/source_name
    if json.loads((source/'status.json').read_text(encoding='utf-8'))['stage'] != 'complete':
        raise ValueError('Source evaluation is incomplete')
    freeze = {'source_summary_sha256':digest(source/'summary.json'),
              'source_freeze_sha256':digest(source/'freeze.json'),
              'model_sha256':digest(VERIFIER/'model.pt'),
              'verifier_source_sha256':digest(ROOT/'vision/scripts/shape_verifier.py'),
              'evaluator_source_sha256':digest(Path(__file__)),
              'scope':'development selection only; no held-out test access'}
    if (work/'freeze.json').exists():
        if json.loads((work/'freeze.json').read_text(encoding='utf-8')) != freeze:
            raise ValueError('Verifier evaluation changed')
    else:
        write_json(work/'freeze.json', freeze)
    verifier = ShapeVerifier(VERIFIER/'model.pt')
    rows, values, times = [], {}, {}
    paths = sorted((source/'records').glob('*.json'))
    for i, path in enumerate(paths):
        row = json.loads(path.read_text(encoding='utf-8'))
        output = work/'probabilities'/path.name
        if output.exists():
            cached = json.loads(output.read_text(encoding='utf-8'))
            if cached['source_record_sha256'] != digest(path):
                raise ValueError('Cached prediction changed')
        else:
            if digest(DATA/row['image']) != row['image_sha256']:
                raise ValueError('Image changed')
            image = np.asarray(Image.open(DATA/row['image']).convert('RGB'))
            tick = time.perf_counter();crops = []
            for j in range(len(row['predictions'])):
                asset = f"masks/{row['key']}/{j+1:02d}.png"
                if digest(source/asset) != row['mask_hashes'][asset]:
                    raise ValueError('Native mask changed')
                crops.append(features(image, np.asarray(Image.open(source/asset)) > 0))
            probabilities = verifier.probabilities(np.stack(crops)).tolist() if crops else []
            cached = {'source_record_sha256':digest(path), 'probabilities':probabilities, 'seconds':time.perf_counter()-tick}
            write_json(output, cached)
        if len(cached['probabilities']) != len(row['predictions']):
            raise ValueError('Probability/detection alignment changed')
        rows.append(row);values[row['key']] = cached['probabilities'];times[row['key']] = cached['seconds']
        if (i+1) % 100 == 0 or i+1 == len(paths):
            print(json.dumps({'source':source_name, 'probabilities':i+1, 'total':len(paths)}), flush=True)
    return rows, values, times, freeze


def evaluate(source_name):
    rows, probabilities, times, freeze = cache_probabilities(source_name)
    source = RUN/'evaluation'/source_name
    old_reports = [json.loads((RUN/'evaluation'/n/'summary.json').read_text(encoding='utf-8'))
                   for n in ['v1_deployed', 'v2_deployed']]
    incumbent = json.loads((RUN/'evaluation/pilot_640/summary.json').read_text(encoding='utf-8'))
    candidates = []
    for gate, minimum, background in product([.6, .8, 1.01], [.55, .75, .9], [.9, .99, 1.01]):
        policy = {'maximum_base_confidence':gate, 'class_probability':minimum, 'background_probability':background}
        revised = [apply_policy(r, probabilities[r['key']], policy) for r in rows]
        groups = {domain:[r for r in revised if r['domain'] == domain] for domain in ['legacy', 'focused', 'negative', 'real']}
        groups['synthetic_validation'] = [r for r in revised if r['domain'] != 'real']
        scores = {domain:score_records(values, .5) for domain, values in groups.items()}
        real_empty = [r for r in revised if r['category'] == 'empty_tray']
        fp = scores['synthetic_validation']['spurious_detections']/len(groups['synthetic_validation'])
        negative_alarm = sum(bool(r['predictions']) for r in groups['negative'])/len(groups['negative'])
        real_alarm = sum(bool(r['predictions']) for r in real_empty)/len(real_empty)
        guards = {
            'recognition':scores['synthetic_validation']['strict_instance_score'] > max(r['groups']['synthetic_validation']['scores']['0.5']['strict_instance_score'] for r in old_reports),
            'false_positives':fp < min(r['groups']['synthetic_validation']['false_positives_per_image'] for r in old_reports),
            'styles':all(scores[d]['strict_instance_score'] >= max(r['groups'][d]['scores']['0.5']['strict_instance_score'] for r in old_reports) for d in ['legacy', 'focused']),
            'real_fit':scores['real']['strict_instance_score'] >= incumbent['groups']['real']['scores']['0.5']['strict_instance_score'],
            'real_empty':real_alarm <= incumbent['groups']['real_empty']['empty_false_detection_rate'],
            'negative':negative_alarm <= min(r['groups']['negative']['empty_false_detection_rate'] for r in old_reports),
        }
        candidates.append({'policy':policy, 'partial_guards':guards, 'strict_score':scores['synthetic_validation']['strict_instance_score'],
                           'false_positives_per_image':fp, 'real_fit_score':scores['real']['strict_instance_score'],
                           'legacy_score':scores['legacy']['strict_instance_score'], 'focused_score':scores['focused']['strict_instance_score']})
    # A useful diagnostic policy must first respect the real training-fit guard.
    # A winning diagnostic is still not a passing candidate: the full gate follows.
    eligible = [c for c in candidates if c['partial_guards']['real_fit'] and c['partial_guards']['real_empty']]
    selected = max(eligible or candidates, key=lambda c:(all(c['partial_guards'].values()), c['strict_score'], -c['false_positives_per_image']))
    policy = selected['policy']
    name = source_name+'_verified'
    out = RUN/'evaluation'/name
    original = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    provenance = {**freeze, 'role':'candidate_verifier', 'model_sha256':original['provenance']['model_sha256'],
                  'verifier_sha256':freeze['model_sha256'], 'settings':original['provenance']['settings'],
                  'verifier_policy':policy, 'source_evaluation':source_name}
    if (out/'freeze.json').exists():
        if json.loads((out/'freeze.json').read_text(encoding='utf-8')) != provenance:
            raise ValueError('Selected verifier policy changed')
    else:
        write_json(out/'freeze.json', provenance)
    final = []
    for before in rows:
        after = apply_policy(before, probabilities[before['key']], policy)
        after['inference_seconds'] += times[before['key']]
        after['mask_hashes'] = {}
        for j, pred in enumerate(after['predictions']):
            old_asset = f"masks/{before['key']}/{pred['source_prediction_index']+1:02d}.png"
            asset = f"masks/{before['key']}/{j+1:02d}.png"
            target = out/asset;target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != before['mask_hashes'][old_asset]:
                    raise ValueError('Retained mask changed')
            else:
                shutil.copy2(source/old_asset, target)
            after['mask_hashes'][asset] = digest(target)
        write_json(out/'records'/(after['key']+'.json'), after)
        final.append(after)
    groups = {'synthetic_validation':[r for r in final if r['domain'] != 'real'],
              'positive_validation':[r for r in final if r['domain'] in {'legacy','focused'}],
              **{d:[r for r in final if r['domain'] == d] for d in ['legacy','focused','negative','real']},
              'real_empty':[r for r in final if r['category'] == 'empty_tray'],
              'real_white':[r for r in final if r['category'] == 'white_cube']}
    report = {'name':name, 'provenance':provenance, 'image_count':len(final), 'validation_images':1108,
              'real_train_fit_images':46, 'real_independent_test_images':0, 'test_used_for_selection':False,
              'groups':{d:summarize(group) for d, group in groups.items()}}
    for d in ['negative','real_empty']:
        report['groups'][d]['empty_false_detection_images'] = sum(bool(r['predictions']) for r in groups[d])
        report['groups'][d]['empty_false_detection_rate'] = report['groups'][d]['empty_false_detection_images']/len(groups[d])
    write_json(out/'selection.json', {'candidates':candidates, 'selected':selected, 'test_used_for_selection':False,
                                     'geometry_must_pass_joint_gate_separately':True})
    write_json(out/'summary.json', report)
    write_json(out/'status.json', {'stage':'complete', 'images':len(final)})
    joint = compare(RUN)
    print(json.dumps({'candidate':name, 'result':joint['candidates'][name]}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    args = parser.parse_args()
    if not args.source or not all(c.isalnum() or c == '_' for c in args.source):
        raise ValueError('Expected safe evaluation name')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    evaluate(args.source)
