#!/usr/bin/env python3
"""Train a YOLO11 instance-segmentation model with reproducible defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--name", default="shape_top_n")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cache", choices=("none", "ram", "disk"), default="disk")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--hsv-h", type=float, default=0.05)
    p.add_argument("--hsv-s", type=float, default=0.5)
    p.add_argument("--hsv-v", type=float, default=0.35)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--focused", action="store_true", help="Preserve geometry for pairwise shape training")
    p.add_argument("--threads", type=int, default=2, help="CPU intra-op threads; avoids laptop oversubscription")
    p.add_argument("--save-period", type=int, default=-1)
    p.add_argument("--resume", action="store_true", help="Resume --model last.pt with its saved optimizer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    import ultralytics.utils.torch_utils as torch_utils
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = False if args.cache == "none" else args.cache
    amp = not args.no_amp and device != "cpu"
    os.environ.setdefault("WANDB_PROJECT", "smart-sort")
    data = project_path(args.data)
    project = project_path(args.project)
    model_path = project_path(args.model)
    model_name = str(model_path) if model_path.exists() else args.model

    model = YOLO(model_name)
    if args.resume:
        model.train(resume=True, device=device, workers=args.workers)
        return
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=cache,
        seed=args.seed,
        project=str(project),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        patience=args.patience,
        amp=amp,
        close_mosaic=min(15, max(1, args.epochs // 5)),
        mosaic=0.0 if args.focused else 0.5,
        mixup=0.0 if args.focused else 0.05,
        copy_paste=0.0 if args.focused else 0.1,
        degrees=10.0 if args.focused else 180.0,
        translate=0.05,
        scale=0.2,
        shear=0.0 if args.focused else 2.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        overlap_mask=True,
        mask_ratio=2,
        plots=True,
        save=True,
        save_period=args.save_period,
        verbose=True,
    )


if __name__ == "__main__":
    main()
