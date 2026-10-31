#!/usr/bin/env python3
"""Evaluate an instance-segmentation model on a fixed YOLO test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rooted_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def split_images(data_path: Path, split: str) -> tuple[list[Path], Path]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = rooted_path(config.get("path", data_path.parent))
    split_path = Path(config[split])
    image_dir = (split_path if split_path.is_absolute() else dataset_root / split_path).resolve()
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}' in {image_dir}")
    return images, dataset_root


def metric_values(metric) -> dict[str, float]:
    return {
        "precision": round(float(metric.mp), 4),
        "recall": round(float(metric.mr), 4),
        "mAP50": round(float(metric.map50), 4),
        "mAP50_95": round(float(metric.map), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/trained/stl_shape_v1/best.pt")
    parser.add_argument("--data", default="dataset/stl_synthetic_yolo/data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--name", default="stl_shape_v1_test_verified")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--metrics-out", default="docs/视觉训练指标_STL首版.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = rooted_path(args.model)
    data_path = rooted_path(args.data)
    project = rooted_path(args.project)
    metrics_out = rooted_path(args.metrics_out)
    images, dataset_root = split_images(data_path, args.split)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    prediction_name = f"{args.name}_predictions"
    model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        save=True,
        project=str(project),
        name=prediction_name,
        exist_ok=True,
        verbose=False,
    )

    per_class = {}
    for index, class_id in enumerate(metrics.seg.ap_class_index):
        precision, recall, ap50, ap = metrics.seg.class_result(index)
        class_id = int(class_id)
        per_class[metrics.names[class_id]] = {
            "instances": int(metrics.nt_per_class[class_id]),
            "mask_precision": round(float(precision), 4),
            "mask_recall": round(float(recall), 4),
            "mask_mAP50": round(float(ap50), 4),
            "mask_mAP50_95": round(float(ap), 4),
        }

    payload = {
        "model": str(model_path.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "data": str(data_path.relative_to(ROOT)),
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "split": args.split,
        "image_count": len(images),
        "instance_count": int(metrics.nt_per_class.sum()),
        "imgsz": args.imgsz,
        "device": args.device,
        "ultralytics": ultralytics_version,
        "box": metric_values(metrics.box),
        "mask": metric_values(metrics.seg),
        "per_class": per_class,
        "prediction_dir": str((project / prediction_name).relative_to(ROOT)),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
