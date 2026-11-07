"""Register an audited RGB-guided result as a development candidate, never deploy."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import shutil
from pathlib import Path
from evaluate_replay_optimization import ROOT, digest, summarize, write_json
from compare_optimization_candidates import compare

RUN = ROOT/'runs/segment/replay_optimization_v3'


def register(source_name):
    source = RUN/'evaluation'/source_name
    audit = RUN/'diagnostics/image_guided_v1'/(source_name+'_full')
    report = json.loads((audit/'summary.json').read_text(encoding='utf-8'))
    freeze = json.loads((audit/'freeze.json').read_text(encoding='utf-8'))
    if not report['full_validation'] or report['images'] != 1154:
        raise ValueError('A complete large development audit is required')
    if freeze['source_summary_sha256'] != digest(source/'summary.json'):
        raise ValueError('Audit/source mismatch')
    if freeze['algorithm_sha256'] != digest(ROOT/'vision/scripts/image_guided_masks.py'):
        raise ValueError('Audited contour algorithm changed')
    before = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    name = source_name+'_guided';out = RUN/'evaluation'/name
    provenance = {**before['provenance'], 'role':'candidate_rgb_guided', 'source_evaluation':source_name,
                  'mask_refinement':{'method':'rgb_grabcut_v1','settings':freeze['settings'],
                                     'algorithm_sha256':freeze['algorithm_sha256']},
                  'audit_freeze_sha256':digest(audit/'freeze.json'),
                  'audit_summary_sha256':digest(audit/'summary.json'), 'registration_source_sha256':digest(Path(__file__))}
    if (out/'freeze.json').exists():
        if json.loads((out/'freeze.json').read_text(encoding='utf-8')) != provenance:
            raise ValueError('Registered candidate changed')
    else:
        write_json(out/'freeze.json', provenance)
    rows = []
    for file in sorted((audit/'records').glob('*.json')):
        row = json.loads(file.read_text(encoding='utf-8'))
        original = json.loads((source/'records'/file.name).read_text(encoding='utf-8'))
        for key in ['key','image','image_sha256','label_sha256','truth_classes','predictions']:
            if row[key] != original[key]:
                raise ValueError('Contour-only audit changed detection identity or data')
        for asset, value in row['mask_hashes'].items():
            if digest(audit/asset) != value:
                raise ValueError('Audited mask changed')
            target = out/asset;target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != value:
                    raise ValueError('Registered mask changed')
            else:
                shutil.copy2(audit/asset, target)
        write_json(out/'records'/file.name, row)
        rows.append(row)
    if Counter(r['domain'] for r in rows) != Counter(legacy=108, focused=900, negative=100, real=46):
        raise ValueError('Incorrect registered evaluation population')
    groups = {'synthetic_validation':[r for r in rows if r['domain'] != 'real'],
              'positive_validation':[r for r in rows if r['domain'] in {'legacy','focused'}],
              **{d:[r for r in rows if r['domain'] == d] for d in ['legacy','focused','negative','real']},
              'real_empty':[r for r in rows if r['category'] == 'empty_tray'],
              'real_white':[r for r in rows if r['category'] == 'white_cube']}
    result = {'name':name,'provenance':provenance,'image_count':1154,'validation_images':1108,'real_train_fit_images':46,
              'real_independent_test_images':0,'test_used_for_selection':False,'groups':{d:summarize(v) for d,v in groups.items()}}
    for d in ['negative','real_empty']:
        result['groups'][d]['empty_false_detection_images'] = sum(bool(r['predictions']) for r in groups[d])
        result['groups'][d]['empty_false_detection_rate'] = result['groups'][d]['empty_false_detection_images']/len(groups[d])
    write_json(out/'summary.json', result)
    write_json(out/'status.json', {'stage':'complete','images':1154,'deployed':False})
    comparison = compare(RUN)
    print(json.dumps({'registered_development_candidate':name,'result':comparison['candidates'][name]},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=['v2_deployed','pilot_640'])
    register(parser.parse_args().source)
