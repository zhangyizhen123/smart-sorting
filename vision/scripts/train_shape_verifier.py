"""Train an independent ROI verifier on training-only masks and empty scenes."""
from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch

from evaluate_replay_optimization import ROOT, digest, write_json
from shape_refiner import read_polygons, polygon_mask
from shape_verifier import features, VerifierNetwork, BACKGROUND

RUN = ROOT / "runs/segment/replay_optimization_v3"
DATA = ROOT / "dataset/replay_optimization_v3"
OUT = RUN / "verifier_v1"
SEED = 20261101


def feature_data():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = {"dataset_manifest_sha256": digest(DATA/'manifest.jsonl'),
              "feature_source_sha256": digest(ROOT/'vision/scripts/shape_verifier.py'),
              "training_source_sha256": digest(Path(__file__)), "seed": SEED,
              "real_images_training_only": True, "sealed_test_used": False,
              "prediction_sources": {name: digest(RUN/'evaluation'/name/'summary.json')
                                     for name in ['v2_deployed', 'pilot_640']}}
    if (OUT/'freeze.json').exists():
        if json.loads((OUT/'freeze.json').read_text(encoding='utf-8')) != freeze:
            raise ValueError('Verifier inputs changed')
    else:
        write_json(OUT/'freeze.json', freeze)
    if (OUT/'features.npz').exists():
        with np.load(OUT/'features.npz') as saved:
            return {k: saved[k] for k in saved.files}
    rows = [json.loads(line) for line in (DATA/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    by_image = {r['image']:r for r in rows}
    rng = np.random.default_rng(SEED)
    tx, ty, weights, vx, vy = [], [], [], [], []
    train_rows = [r for r in rows if r['split'] == 'train' and r.get('repetition', 0) == 0]
    for index, row in enumerate(train_rows):
        for kind in ['image', 'label']:
            if digest(DATA/row[kind]) != row[kind+'_sha256']:
                raise ValueError('Training data changed')
        image = np.asarray(Image.open(DATA/row['image']).convert('RGB'))
        height, width = image.shape[:2]
        truth = read_polygons(DATA/row['label'], width, height)
        weight = 4. if row['domain'] == 'real' else 1.
        for cls, polygon in truth:
            mask = (polygon_mask(polygon, width, height) > 0).astype(np.uint8)
            tx.append(features(image, mask));ty.append(cls);weights.append(weight)
            # Train-only proposal perturbation imitates modest detector mask error.
            dx, dy = rng.integers(-2, 3, size=2)
            shifted = cv2.warpAffine(mask, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height), flags=cv2.INTER_NEAREST)
            if rng.random() < .5:
                shifted = cv2.dilate(shifted, np.ones((3, 3), np.uint8))
            else:
                shifted = cv2.erode(shifted, np.ones((3, 3), np.uint8))
            if shifted.any():
                tx.append(features(image, shifted));ty.append(cls);weights.append(weight)
        if not truth:
            # Background supervision comes only from known-empty training images.
            for _ in range(8):
                mask = np.zeros((height, width), np.uint8)
                w, h = rng.integers(max(6, min(width, height)//20), max(7, min(width, height)//3), size=2)
                x, y = int(rng.integers(0, max(1, width-w))), int(rng.integers(0, max(1, height-h)))
                if rng.random() < .5:
                    cv2.ellipse(mask, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, 1, -1)
                else:
                    mask[y:y+h, x:x+w] = 1
                tx.append(features(image, mask));ty.append(BACKGROUND);weights.append(weight)
        if (index+1) % 100 == 0:
            write_json(OUT/'status.json', {'stage':'training_features', 'images': index+1, 'total':len(train_rows)})
            print(json.dumps({'training_images':index+1, 'total':len(train_rows)}), flush=True)
    for name in ['v2_deployed', 'pilot_640']:
        source = RUN/'evaluation'/name
        for path in sorted((source/'records').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            origin = by_image[row['image']]
            expected_split = 'train' if row['domain'] == 'real' else 'val'
            if origin['split'] != expected_split or origin['domain'] != row['domain']:
                raise ValueError('Proposal assigned to the wrong training/validation group')
            if row['label_sha256'] != origin['label_sha256'] or row['truth_classes'] != origin['classes']:
                raise ValueError('Proposal truth differs from frozen labels')
            if not row['predictions']:
                continue
            image_path = DATA/row['image']
            if digest(image_path) != row['image_sha256']:
                raise ValueError('Prediction image changed')
            image = np.asarray(Image.open(image_path).convert('RGB'))
            matrix = np.asarray(row['ious']).reshape(len(row['truth_classes']), len(row['predictions']))
            for j in range(len(row['predictions'])):
                overlap = float(matrix[:, j].max()) if len(matrix) else 0.
                if .1 < overlap < .5:
                    continue  # Ambiguous proposals are not mislabeled as empty.
                label = row['truth_classes'][int(matrix[:, j].argmax())] if overlap >= .5 else BACKGROUND
                asset = f"masks/{row['key']}/{j+1:02d}.png"
                if digest(source/asset) != row['mask_hashes'][asset]:
                    raise ValueError('Prediction mask changed')
                mask = np.asarray(Image.open(source/asset)) > 0
                value = features(image, mask)
                if row['domain'] == 'real':
                    # These 46 photographs were already assigned to training.
                    tx.append(value);ty.append(label);weights.append(4.)
                else:
                    vx.append(value);vy.append(label)
    arrays = dict(x=np.asarray(tx, np.float32), y=np.asarray(ty, np.int64),
                  weight=np.asarray(weights, np.float32), vx=np.asarray(vx, np.float32), vy=np.asarray(vy, np.int64))
    if set(arrays['y'].tolist()) != set(range(10)) or set(arrays['vy'].tolist()) != set(range(10)):
        raise ValueError('All shape classes and background required')
    np.savez_compressed(OUT/'features.npz', **arrays)
    write_json(OUT/'feature_counts.json', {'training':dict(Counter(map(int, arrays['y']))),
               'validation':dict(Counter(map(int, arrays['vy']))), 'feature_file_sha256':digest(OUT/'features.npz'),
               'validation_scope':'predicted ROIs on synthetic development validation; correlated proposals, not independent accuracy'})
    return arrays


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(SEED)
    data = feature_data()
    if (OUT/'model.pt').exists():
        raise FileExistsError('Verifier checkpoint already exists; inspect before retraining')
    x, vx = torch.from_numpy(data['x']), torch.from_numpy(data['vx'])
    y, vy, weight = torch.from_numpy(data['y']), torch.from_numpy(data['vy']), torch.from_numpy(data['weight'])
    mean, scale = x.mean(0), x.std(0).clamp_min(.05)
    x, vx = ((x-mean)/scale).clamp(-6, 6), ((vx-mean)/scale).clamp(-6, 6)
    model = VerifierNetwork();optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    class_weight = (torch.bincount(y).float().mean()/torch.bincount(y).float()).sqrt().clamp(.5, 3.)
    best, best_epoch, best_state, curve = float('inf'), 0, None, []
    for epoch in range(45):
        model.train();start = time.perf_counter()
        for ids in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[ids]), y[ids], weight=class_weight, reduction='none')
            (loss*weight[ids]).mean().backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(batch) for batch in vx.split(256)])
            value = float(torch.nn.functional.cross_entropy(logits, vy, weight=class_weight))
            accuracy = float((logits.argmax(1) == vy).float().mean())
        entry = {'epoch':epoch+1, 'validation_loss':value, 'roi_validation_accuracy':accuracy, 'seconds':time.perf_counter()-start}
        curve.append(entry);print(json.dumps(entry), flush=True)
        write_json(OUT/'status.json', {'stage':'training', **entry})
        if value < best:
            best, best_epoch, best_state = value, epoch+1, copy.deepcopy(model.state_dict())
        if epoch+1-best_epoch >= 10:
            break
    torch.save({'version':1, 'mean':mean, 'scale':scale, 'state_dict':best_state}, OUT/'model.pt')
    write_json(OUT/'metadata.json', {'best_epoch':best_epoch, 'curve':curve, 'model_sha256':digest(OUT/'model.pt'),
                                   'development_only':True, 'deployed':False, 'end_to_end_evaluation_required':True})
    write_json(OUT/'status.json', {'stage':'complete', 'best_epoch':best_epoch, 'end_to_end_evaluation_required':True})


if __name__ == '__main__':
    main()
