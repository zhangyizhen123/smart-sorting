#!/usr/bin/env python3
"""Run the first small, leakage-checked hyperparameter sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=128)
    args = p.parse_args()
    script = Path(__file__).with_name("train_classifier.py")
    configs = [
        {"seed": 42, "lr": 0.001, "weight_decay": 1e-4, "label_smoothing": 0.0, "rotate": 15, "color_jitter": 0.08},
        {"seed": 43, "lr": 0.0005, "weight_decay": 1e-4, "label_smoothing": 0.02, "rotate": 25, "color_jitter": 0.12},
        {"seed": 44, "lr": 0.001, "weight_decay": 5e-4, "label_smoothing": 0.0, "rotate": 8, "color_jitter": 0.05},
    ]
    summary = []
    for i, cfg in enumerate(configs, 1):
        name = f"round1_trial{i}"
        command = [sys.executable, str(script), "--data", str(args.data), "--device", args.device,
                   "--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--name", name]
        for key, value in cfg.items(): command += [f"--{key.replace('_', '-')}", str(value)]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        metrics = json.loads((Path("runs/stage1") / name / "metrics.json").read_text(encoding="utf-8"))
        summary.append({"name": name, **cfg, **metrics})
    # Test is an untouched audit split. It must never affect trial ordering or
    # checkpoint selection, including when validation scores tie.
    summary.sort(key=lambda x: -x["best_val_accuracy"])
    Path("runs/stage1/round1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
