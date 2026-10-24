#!/usr/bin/env python3
"""Train a YOLO11 instance-segmentation model with reproducible defaults."""

from __future__ import annotations

import argparse
import os

import torch
from ultralytics import YOLO


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = False if args.cache == "none" else args.cache
    amp = not args.no_amp and device != "cpu"
    os.environ.setdefault("WANDB_PROJECT", "smart-sort")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=cache,
        seed=args.seed,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        patience=args.patience,
        amp=amp,
        close_mosaic=15,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.1,
        degrees=180.0,
        translate=0.05,
        scale=0.2,
        shear=2.0,
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
        verbose=True,
    )


if __name__ == "__main__":
    main()
