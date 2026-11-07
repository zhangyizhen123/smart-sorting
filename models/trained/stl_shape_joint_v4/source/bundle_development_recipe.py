"""Rebuild exact evaluated processing recipes without deploying or opening sealed data."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
from evaluate_replay_optimization import ROOT,digest,write_json,resolve_spec

RUN=ROOT/'runs/segment/replay_optimization_v3'


def verified_copy(source,target,expected):
    if digest(source)!=expected:
        raise ValueError(f'Changed source asset: {source.name}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=expected:
            raise ValueError('Existing bundle asset changed')
    else:
        shutil.copy2(source,target)


def report(name):
    if not name or not all(c.isalnum() or c=='_' for c in name):
        raise ValueError('Safe evaluation name required')
    directory=RUN/'evaluation'/name
    if json.loads((directory/'status.json').read_text(encoding='utf-8'))['stage']!='complete':
        raise ValueError('Cannot bundle unfinished evaluation')
    result=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if json.loads((directory/'freeze.json').read_text(encoding='utf-8'))!=result['provenance']:
        raise ValueError('Evaluation summary/freeze mismatch')
    return result


def find_checkpoint(expected):
    specs=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text(encoding='utf-8'))
    for spec in specs.values():
        path,_,_,provenance=resolve_spec(spec)
        if provenance['model_sha256']==expected:
            return path
    for path in (RUN/'candidates').glob('*/model.pt'):
        if digest(path)==expected:
            return path
    raise ValueError('No verified model matches evaluated checkpoint')


def build(name,directory,stack=()):
    if name in stack or len(stack)>8:
        raise ValueError('Cyclic/deep evaluation recipe')
    summary=report(name);p=summary['provenance'];role=p['role']
    evaluation=dict(stage='development_runtime_check',not_deployed=True,joint_gate_passed=False,independent_real_images=0)
    if role in {'candidate_foreground','candidate_consensus','candidate_specialist'}:
        descriptor=p.get('specialist') if role=='candidate_specialist' else p.get('consensus')
        source_names=({'source':p['source_evaluation']} if role=='candidate_foreground' else
                      {key:descriptor[key] for key in ['primary','secondary']})
        sources={}
        for key,child in source_names.items():
            expected=p.get('source_summary_sha256') if role=='candidate_foreground' else descriptor[key+'_summary_sha256']
            if expected!=digest(RUN/'evaluation'/child/'summary.json'):
                raise ValueError('Evaluated recipe source changed')
            manifest=build(child,directory/key,stack+(name,))
            sources[key]=dict(pipeline=manifest.relative_to(directory).as_posix(),pipeline_sha256=digest(manifest))
        pipeline=dict(version=3,runtime_sha256=digest(ROOT/'vision/scripts/pipeline_v3.py'),sources=sources,
                      operation='foreground' if role=='candidate_foreground' else 'agreement',evaluation=evaluation)
        if role=='candidate_specialist':
            if descriptor['algorithm_sha256']!=digest(ROOT/'vision/scripts/specialist_shapes.py'):
                raise ValueError('Evaluated specialist algorithm changed')
            pipeline.update(version=4,runtime_sha256=digest(ROOT/'vision/scripts/pipeline_v4.py'),operation='specialist',
                            policy=p['specialist_policy'],algorithm_sha256=descriptor['algorithm_sha256'])
        elif role=='candidate_consensus':
            if p['consensus']['algorithm_sha256']!=digest(ROOT/'vision/scripts/consensus_shapes.py'):
                raise ValueError('Evaluated agreement algorithm changed')
            pipeline.update(policy=p['consensus_policy'],algorithm_sha256=p['consensus']['algorithm_sha256'])
        else:
            gate=RUN/'foreground_gate_v1/model.pt';verified_copy(gate,directory/'foreground.pt',p['foreground_model_sha256'])
            pipeline['foreground']=dict(model='foreground.pt',model_sha256=p['foreground_model_sha256'],
                                        algorithm_sha256=p['foreground_algorithm_sha256'],features_sha256=p['features_sha256'],
                                        threshold=p['foreground_threshold'])
    else:
        # Inference v2 has a fixed order: native detector/refiner, guided masks, verifier.
        native=copy.deepcopy(p);postprocess={};visited=[];current_name=name
        while native['role'] in {'candidate_verifier','candidate_rgb_guided'}:
            visited.append(current_name)
            if native['role']=='candidate_verifier':
                if 'mask_refinement' in postprocess:
                    raise ValueError('Verifier-before-contour order is unsupported by inference v2')
                if 'verifier' in postprocess:
                    raise ValueError('Repeated verifier stage')
                postprocess['verifier']=dict(model='verifier.pt',model_sha256=native['verifier_sha256'],
                    features_sha256=native['verifier_source_sha256'],policy=native['verifier_policy'])
                verified_copy(RUN/'verifier_v1/model.pt',directory/'verifier.pt',native['verifier_sha256'])
            else:
                if 'mask_refinement' in postprocess:
                    raise ValueError('Repeated contour stage')
                postprocess['mask_refinement']=native['mask_refinement']
                audit=RUN/'diagnostics/image_guided_v1'/(native['source_evaluation']+'_full')/'freeze.json'
                if digest(audit)!=native['audit_freeze_sha256']:
                    raise ValueError('Contour audit changed')
                audited=json.loads(audit.read_text(encoding='utf-8'))
                if audited['source_summary_sha256']!=digest(RUN/'evaluation'/native['source_evaluation']/'summary.json'):
                    raise ValueError('Contour source changed after audit')
            child=native['source_evaluation']
            if child in visited:
                raise ValueError('Cyclic native recipe')
            if native.get('source_summary_sha256') and native['source_summary_sha256']!=digest(RUN/'evaluation'/child/'summary.json'):
                raise ValueError('Native recipe source changed')
            current_name=child;native=report(child)['provenance']
        if native['role'] in {'candidate_foreground','candidate_consensus'}:
            raise ValueError('Contour/verifier after a composed recipe needs a separate runtime')
        if native['role'] not in {'deployed_baseline','training_fit_incumbent','candidate','candidate_refined'}:
            raise ValueError('Unrecognized native evaluation role')
        checkpoint=find_checkpoint(native['model_sha256'])
        verified_copy(checkpoint,directory/'best.pt',native['model_sha256'])
        pipeline=dict(version=2,runtime_sha256=digest(ROOT/'vision/scripts/pipeline_v2.py'),model='best.pt',
                      model_sha256=native['model_sha256'],settings=native['settings'],postprocess=postprocess,evaluation=evaluation)
        if native.get('refiner_sha256'):
            refiner=ROOT/'models/trained/stl_shape_focus_v2/refiner'
            for file,key in [('refiner.pt','refiner_sha256'),('metadata.json','refiner_metadata_sha256')]:
                verified_copy(refiner/file,directory/'refiner'/file,native[key])
            pipeline.update(refiner='refiner',refiner_sha256=native['refiner_sha256'],refiner_metadata_sha256=native['refiner_metadata_sha256'])
    pipeline['source_evaluation']=name;pipeline['source_summary_sha256']=digest(RUN/'evaluation'/name/'summary.json')
    path=directory/'pipeline.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=pipeline:
        raise ValueError('Existing recipe manifest changed; use a new bundle directory')
    write_json(path,pipeline)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('candidate');args=parser.parse_args()
    path=build(args.candidate,RUN/'runtime_recipes'/args.candidate)
    print(json.dumps(dict(pipeline=path.relative_to(ROOT).as_posix(),sha256=digest(path),deployed=False)),flush=True)
