"""Check real model inference against cached, complete development predictions."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vision/scripts'))
from pipeline_v2 import NativeShapePipeline, digest
from image_guided_masks import DEFAULTS
from evaluate_replay_optimization import write_json


def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/segment/replay_optimization_v3'
    candidate = run/'evaluation/v2_deployed_guided_verified'
    selection = json.loads((candidate/'selection.json').read_text(encoding='utf-8'))
    original = ROOT/'models/trained/stl_shape_focus_v2'
    old = json.loads((original/'pipeline.json').read_text(encoding='utf-8'))
    bundle = run/'runtime_validation_bundle';bundle.mkdir(exist_ok=True)
    assets = [(original/'best.pt', bundle/'best.pt'),
              (original/'refiner/refiner.pt', bundle/'refiner/refiner.pt'),
              (original/'refiner/metadata.json', bundle/'refiner/metadata.json'),
              (run/'verifier_v1/model.pt', bundle/'verifier.pt')]
    for source, target in assets:
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError('Runtime check bundle changed')
        else:
            shutil.copy2(source, target)
    manifest = {'version':2, 'model':'best.pt', 'model_sha256':digest(bundle/'best.pt'),
                'runtime_sha256':digest(ROOT/'vision/scripts/pipeline_v2.py'), 'settings':old['settings'],
                'refiner':'refiner','refiner_sha256':digest(bundle/'refiner/refiner.pt'),
                'refiner_metadata_sha256':digest(bundle/'refiner/metadata.json'),
                'postprocess':{'mask_refinement':{'method':'rgb_grabcut_v1','settings':DEFAULTS,
                    'algorithm_sha256':digest(ROOT/'vision/scripts/image_guided_masks.py')},
                    'verifier':{'model':'verifier.pt','model_sha256':digest(bundle/'verifier.pt'),
                        'features_sha256':digest(ROOT/'vision/scripts/shape_verifier.py'),
                        'policy':selection['selected']['policy']}},
                'evaluation':{'stage':'development_runtime_check','not_deployed':True,
                    'joint_gate_passed':False,'independent_real_images':0}}
    write_json(bundle/'pipeline.json',manifest)
    pipeline = NativeShapePipeline(bundle/'pipeline.json')
    keys = ['legacy_scene000462__frame000',
            'focused_focused_v2_20261102_val_08_00032',
            'real_5ceaab71bf413582_r0']
    negative = next(p.stem for p in sorted((candidate/'records').glob('*.json'))
                    if json.loads(p.read_text(encoding='utf-8'))['domain'] == 'negative')
    keys.append(negative)
    results = []
    for key in keys:
        record = json.loads((candidate/'records'/(key+'.json')).read_text(encoding='utf-8'))
        with Image.open(ROOT/'dataset/replay_optimization_v3'/record['image']) as photo:
            image = photo.convert('RGB')
        predictions, masks, originals, settings = pipeline.predict_native(image)
        assert len(predictions) == len(record['predictions']), key
        assert [p['class_id'] for p in predictions] == [p['class_id'] for p in record['predictions']], key
        assert [p['source_prediction_index'] for p in predictions] == [p['source_prediction_index'] for p in record['predictions']], key
        changed = []
        for j, mask in enumerate(masks):
            expected = np.asarray(Image.open(candidate/'masks'/key/f'{j+1:02d}.png')) > 0
            difference = int(np.count_nonzero(mask != expected))
            changed.append(difference)
            assert difference == 0, (key,j,difference)
        results.append({'key':key,'domain':record['domain'],'detections':len(predictions),
                        'mask_pixel_differences':changed,'classes_equal':True,'source_indices_equal':True})
        print(json.dumps(results[-1]),flush=True)
    result = {'passed':True,'scope':'runtime consistency only; not model acceptance or deployment',
              'pipeline_sha256':pipeline.pipeline_hash,'candidate_summary_sha256':digest(candidate/'summary.json'),
              'cases':results,'sealed_test_used':False,'deployed':False}
    write_json(ROOT/'runs/task_audit/2026-11-01_pipeline_v2_runtime.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
