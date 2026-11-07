"""Keep all primary proposals, use agreement for classes, defer background filtering."""
import json
from pathlib import Path
import shutil
from consensus_shapes import box_ious
from probe_consensus_candidates import combine_record
from evaluate_replay_optimization import ROOT,digest,summarize,write_json
from compare_optimization_candidates import compare

RUN=ROOT/'runs/segment/replay_optimization_v3'


def register():
    names={'primary':'v2_deployed_guided_verified','secondary':'pilot_640_guided_verified'}
    sources={k:RUN/'evaluation'/v for k,v in names.items()}
    policy=dict(match_iou=.3,unmatched_primary_confidence=.25,unmatched_secondary_confidence=.6,
                class_source='stronger_secondary',secondary_margin=.1)
    name='guided_agreement_full_v1';out=RUN/'evaluation'/name
    reports={k:json.loads((v/'summary.json').read_text(encoding='utf-8')) for k,v in sources.items()}
    provenance={**reports['primary']['provenance'],'role':'candidate_consensus',
                'consensus':dict(**names,**{k+'_summary_sha256':digest(v/'summary.json') for k,v in sources.items()},
                                 algorithm_sha256=digest(ROOT/'vision/scripts/consensus_shapes.py')),
                'consensus_policy':policy,'registration_sha256':digest(Path(__file__))}
    if (out/'freeze.json').exists() and json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=provenance:
        raise ValueError('Consensus candidate changed')
    write_json(out/'freeze.json',provenance);rows=[]
    for file in sorted((sources['primary']/'records').glob('*.json')):
        before={k:json.loads((v/'records'/file.name).read_text(encoding='utf-8')) for k,v in sources.items()}
        a,b=before['primary'],before['secondary']
        row=combine_record(a,b,policy,box_ious(a['predictions'],b['predictions']));row['mask_hashes']={}
        for j,p in enumerate(row['predictions']):
            old_asset=f"masks/{row['key']}/{p['consensus_source_index']+1:02d}.png"
            source=sources[p['consensus_source']]/old_asset
            if digest(source)!=before[p['consensus_source']]['mask_hashes'][old_asset]:
                raise ValueError('Source mask changed')
            asset=f"masks/{row['key']}/{j+1:02d}.png";target=out/asset;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);row['mask_hashes'][asset]=digest(target)
        write_json(out/'records'/file.name,row);rows.append(row)
    if len(rows)!=1154:
        raise ValueError('Complete development population required')
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
    print(json.dumps(compare(RUN)['candidates'][name]),flush=True)


if __name__=='__main__':
    register()
