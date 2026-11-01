#!/usr/bin/env python3
"""Train three pairwise appearance+geometry heads and a cube/prism arbiter."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from shape_refiner import RefinerNetwork, crop_features, read_rgb
from evaluate_focused import match_instances

HEAD_CLASSES = ((0, 6), (3, 2), (1, 2), (1, 2, 3))


def load_features(records_path, root, split):
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    features, labels = [], []
    for record in records:
        if not record["image"].replace("\\", "/").startswith(f"images/{split}/"):
            raise ValueError(f"Refusing {split} features from another split")
        image = read_rgb(root / record["image"])
        ious = np.asarray(record["ious"]).reshape(len(record["truth_classes"]), len(record["predictions"]))
        for i, j in match_instances(ious):
            cls = record["truth_classes"][i]
            if cls in (0, 1, 2, 3, 6):
                features.append(crop_features(image, np.asarray(record["predictions"][j]["polygon"])))
                labels.append(cls)
        # True masks supplement training only. Validation is always predicted masks.
        if split == "train":
            for cls, polygon in zip(record["truth_classes"], record["truth_polygons"]):
                if cls in (0, 1, 2, 3, 6):
                    features.append(crop_features(image, np.asarray(polygon)))
                    labels.append(cls)
    if not features or set(labels) != {0, 1, 2, 3, 6}:
        raise ValueError(f"All five focus classes required in {split}")
    return np.asarray(features, np.float32), np.asarray(labels, np.int64)


def loss_for_heads(outputs, labels):
    losses = []
    for output, classes in zip(outputs, HEAD_CLASSES):
        target = torch.full_like(labels, -1)
        for index, cls in enumerate(classes):
            target[labels == cls] = index
        valid = target >= 0
        if valid.any():
            losses.append(nn.functional.cross_entropy(output[valid], target[valid]))
    return torch.stack(losses).mean()


def train(root, train_records, val_records, out, epochs=60):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
    x, y = load_features(train_records, root, "train")
    vx, vy = load_features(val_records, root, "val")
    mean = torch.tensor(x.mean(0)); scale = torch.tensor(x.std(0)).clamp_min(0.05)
    tx = ((torch.from_numpy(x) - mean) / scale).clamp(-6, 6); ty = torch.from_numpy(y)
    tvx = ((torch.from_numpy(vx) - mean) / scale).clamp(-6, 6); tvy = torch.from_numpy(vy)
    model = RefinerNetwork(x.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    curve = []
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(tx)).split(64):
            optimizer.zero_grad(); loss = loss_for_heads(model(tx[indices]), ty[indices])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float(loss_for_heads(model(tvx), tvy))
        curve.append({"epoch": epoch + 1, "val_loss": value})
        if value < best_loss:
            best_loss, best_state, best_epoch = value, copy.deepcopy(model.state_dict()), epoch + 1
        if (epoch + 1) % 10 == 0:
            print(f"refiner epoch {epoch + 1}: val loss={value:.4f}", flush=True)
        if epoch + 1 - best_epoch >= 15:
            break
    out.mkdir(parents=True)
    torch.save({"mean": mean, "scale": scale, "state_dict": best_state}, out / "refiner.pt")
    model.load_state_dict(best_state); model.eval()
    validation = []
    with torch.inference_mode():
        outputs = model(tvx)
        for output, classes in zip(outputs, HEAD_CLASSES):
            selected = torch.tensor([int(cls) in classes for cls in vy])
            predicted = output[selected].argmax(1).numpy()
            target = np.array([classes.index(int(cls)) for cls in vy[selected.numpy()]])
            validation.append({"class_ids": classes, "instances": len(target), "accuracy": float((predicted == target).mean())})
    metadata = {"version": 2, "head_classes": HEAD_CLASSES, "training_features": len(x), "validation_features": len(vx),
        "best_epoch": best_epoch, "validation": validation, "curve": curve, "max_base_confidence": 0.8, "minimum_probability": 0.65,
        "training_records_sha256": hashlib.sha256(train_records.read_bytes()).hexdigest(),
        "validation_records_sha256": hashlib.sha256(val_records.read_bytes()).hexdigest(),
        "features": "32x32 masked grayscale/mask/edges plus contour geometry; no external classifier weights",
        "purpose": "Experimental until end-to-end validation beats detector-only; no test fitting"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "validation": validation}), flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--train-records", type=Path, required=True)
    p.add_argument("--val-records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args(); train(a.dataset.resolve(), a.train_records, a.val_records, a.out, a.epochs)


if __name__ == "__main__":
    main()
