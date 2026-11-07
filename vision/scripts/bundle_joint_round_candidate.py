"""Bundle only a candidate passing both original and expanded development gates."""
import json
from pathlib import Path
from bundle_development_recipe import verified_copy
from evaluate_replay_optimization import digest,write_json
from pipeline_v5 import ProtectedPipeline

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/segment/replay_optimization_v3'
RUN=ROOT/'runs/segment/joint_round_v4'


def main():
    selection=json.loads((RUN/'selection.json').read_text(encoding='utf-8'));selected=selection['selected']
    if not selected or not selected['eligible_before_geometry'] or not all(v['passed'] for v in selected['geometry'].values()):
        raise ValueError('Both development populations must pass before bundling')
    if (RUN/'candidate_lock.json').exists(): raise ValueError('This round is already locked')
    for name,expected in selection['algorithm_source_sha256'].items():
        if digest(ROOT/'vision/scripts'/name)!=expected: raise ValueError('Selected algorithm source changed')
    out=RUN/'runtime_candidate'
    if (out/'pipeline.json').exists():
        existing=json.loads((out/'pipeline.json').read_text())
        if existing['selection_sha256']!=digest(RUN/'selection.json'): raise ValueError('Existing bundle belongs to another selection')
        ProtectedPipeline(out/'pipeline.json');print(json.dumps(dict(pipeline=str(out/'pipeline.json'),existing=True)));return
    sources={}
    for name,source in [('base',OLD/'sealed_candidate/primary/source'),('specialist',OLD/'sealed_candidate/secondary')]:
        for path in source.rglob('*'):
            if path.is_file(): verified_copy(path,out/name/path.relative_to(source),digest(path))
        sources[name]=dict(pipeline=name+'/pipeline.json',pipeline_sha256=digest(out/name/'pipeline.json'))
    weight=OLD/'foreground_gate_v1/model.pt';verified_copy(weight,out/'foreground/model.pt',selection['foreground_model_sha256'])
    algorithms=['consensus_evidence.py','foreground_gate.py','protected_foreground.py','shape_verifier.py','specialist_shapes.py']
    manifest=dict(version=5,operation='protected_agreement',runtime_sha256=digest(ROOT/'vision/scripts/pipeline_v5.py'),
                  algorithm_sha256={name:digest(ROOT/'vision/scripts'/name) for name in algorithms},sources=sources,
                  foreground=dict(model='foreground/model.pt',model_sha256=digest(weight)),policy=selected['policy'],
                  selection_sha256=digest(RUN/'selection.json'),evaluation=dict(stage='expanded_development_passed',not_deployed=True,
                  strict_instance_score=selected['pooled_strict'],synthetic_development_images=4008,independent_real_images=0))
    write_json(out/'pipeline.json',manifest);ProtectedPipeline(out/'pipeline.json')
    print(json.dumps(dict(pipeline=(out/'pipeline.json').relative_to(ROOT).as_posix(),sha256=digest(out/'pipeline.json'),deployed=False)),flush=True)


if __name__=='__main__':main()
