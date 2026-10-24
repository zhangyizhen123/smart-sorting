#!/usr/bin/env python3
"""Evaluate the confusion between cuboid and cylinder on a YOLO test split."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

DEFAULT_NAMES = [
    "cylinder", "cube", "cuboid", "tri_prism", "pent_prism",
    "hex_prism", "cone", "tetrahedron", "sphere",
]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def read_ground_truth(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        cls = int(fields[0])
        values = np.asarray([float(x) for x in fields[1:]], dtype=np.float32)
        if len(values) == 4:
            cx, cy, w, h = values
            box = np.array([
                (cx - w / 2) * width, (cy - h / 2) * height,
                (cx + w / 2) * width, (cy + h / 2) * height,
            ])
        else:
            points = values.reshape(-1, 2)
            box = np.array([
                points[:, 0].min() * width, points[:, 1].min() * height,
                points[:, 0].max() * width, points[:, 1].max() * height,
            ])
        result.append((cls, box))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--names", default=",".join(DEFAULT_NAMES))
    args = p.parse_args()
    names = [x.strip() for x in args.names.split(",")]
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    missed = defaultdict(int)
    false_positive = 0
    total_gt = 0
    model = YOLO(args.model)

    for image in sorted(args.images.rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = args.labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        prediction = model.predict(
            source=str(image), imgsz=args.imgsz, conf=args.conf, iou=0.7,
            device=args.device, verbose=False,
        )[0]
        with Image.open(image) as im:
            width, height = im.size
        gt = read_ground_truth(label, width, height)
        total_gt += len(gt)
        if prediction.boxes is None:
            for cls, _ in gt:
                missed[cls] += 1
            continue
        boxes = prediction.boxes.xyxy.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        used: set[int] = set()
        for true_cls, true_box in gt:
            candidates = sorted(
                ((iou(true_box, box), idx) for idx, box in enumerate(boxes) if idx not in used),
                reverse=True,
            )
            if not candidates or candidates[0][0] < args.match_iou:
                missed[true_cls] += 1
                continue
            _, idx = candidates[0]
            used.add(idx)
            pred_cls = int(classes[idx])
            if 0 <= true_cls < len(names) and 0 <= pred_cls < len(names):
                matrix[true_cls, pred_cls] += 1
        false_positive += max(0, len(boxes) - len(used))

    print("classes:", names)
    print("confusion rows=true, cols=pred:")
    print(matrix)
    print("ground_truth:", total_gt)
    print("missed:", dict(missed))
    print("false_positive:", false_positive)
    for source, target in (("cylinder", "cuboid"), ("cuboid", "cylinder")):
        if source in names and target in names:
            si, ti = names.index(source), names.index(target)
            denominator = matrix[si].sum() + missed[si]
            error = matrix[si, ti] / max(1, denominator)
            print(f"{source}_to_{target}: {matrix[si, ti]}/{denominator} = {error:.4%}")


if __name__ == "__main__":
    main()
