#!/usr/bin/env python3
"""Export per-image predictions and a contact sheet for the stage-one test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_classifier import CLASSES, Classifier


def load_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/stage1/error_analysis"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Classifier().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size = int(checkpoint.get("imgsz", 128))
    rows = []
    errors = []
    with torch.no_grad():
        for true_id, cls in enumerate(CLASSES):
            for path in sorted((args.data / "test" / cls).glob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                logits = model(load_tensor(path, size).unsqueeze(0).to(device))[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                pred_id = int(probs.argmax())
                row = {
                    "path": str(path),
                    "true": cls,
                    "pred": CLASSES[pred_id],
                    "confidence": f"{probs[pred_id]:.6f}",
                    "true_probability": f"{probs[true_id]:.6f}",
                }
                rows.append(row)
                if pred_id != true_id:
                    errors.append((path, row))
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / "error_summary.txt").write_text(
        "\n".join(f"{r['true']} -> {r['pred']} ({r['confidence']}) {r['path']}" for _, r in errors) + "\n",
        encoding="utf-8",
    )
    if errors:
        thumb_w, thumb_h = 192, 220
        sheet = Image.new("RGB", (thumb_w * min(4, len(errors)), thumb_h * ((len(errors) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (path, row) in enumerate(errors):
            im = Image.open(path).convert("RGB").resize((thumb_w, thumb_w))
            x, y = (idx % 4) * thumb_w, (idx // 4) * thumb_h
            sheet.paste(im, (x, y))
            draw.text((x + 3, y + thumb_w + 2), f"{row['true']} → {row['pred']} {row['confidence']}", fill="black")
        sheet.save(args.out / "errors_contact_sheet.jpg", quality=95)
    print(f"samples={len(rows)} errors={len(errors)} output={args.out}")


if __name__ == "__main__":
    main()
