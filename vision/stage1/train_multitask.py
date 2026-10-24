#!/usr/bin/env python3
"""Train a shared-encoder shape + color classifier with leakage checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset

SHAPES = ["cuboid", "tetrahedron", "prism", "sphere"]
COLORS = ["red", "green", "blue", "yellow"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def items(root):
    out = []
    for s, shape in enumerate(SHAPES):
        for c, color in enumerate(COLORS):
            for path in sorted((root / shape / color).glob("*")):
                if path.suffix.lower() in EXTS:
                    out.append((path, s, c))
    return out


def assert_no_leakage(root):
    seen = {}
    for split in ("train", "val", "test"):
        for path, _, _ in items(root / split):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise RuntimeError(f"Exact image leakage: {path} and {seen[digest][1]}")
            seen[digest] = (split, str(path))


class MultiDataset(Dataset):
    def __init__(self, root, size, train, rotate=0.0):
        self.records, self.size, self.train, self.rotate = items(root), size, train, rotate
        if not self.records:
            raise RuntimeError(f"No images under {root}")

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        path, shape, color = self.records[i]
        image = Image.open(path).convert("RGB").resize((self.size, self.size))
        if self.train and self.rotate and random.random() < 0.7:
            image = image.rotate(random.uniform(-self.rotate, self.rotate), resample=Image.Resampling.BILINEAR)
        if self.train:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1))
        return (x - 0.5) / 0.5, shape, color


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x): return self.features(x).flatten(1)


class MultiTaskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.shape_head = nn.Sequential(nn.Dropout(0.15), nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, len(SHAPES)))
        # Robust center statistics provide a low-variance chromatic cue. The
        # learned encoder still handles object/background separation.
        self.color_head = nn.Sequential(nn.Dropout(0.15), nn.Linear(198, 64), nn.ReLU(), nn.Linear(64, len(COLORS)))

    def forward(self, x):
        feat = self.encoder(x)
        h, w = x.shape[-2:]
        crop = x[:, :, h // 5:4 * h // 5, w // 5:4 * w // 5]
        stats = torch.cat((crop.mean((2, 3)), crop.std((2, 3))), dim=1)
        return self.shape_head(feat), self.color_head(torch.cat((feat, stats), dim=1))


def evaluate(model, loader, device):
    model.eval(); st = co = total = pair = 0
    with torch.no_grad():
        for x, ys, yc in loader:
            ps, pc = model(x.to(device))
            ys, yc = ys.to(device), yc.to(device)
            ps, pc = ps.argmax(1), pc.argmax(1)
            st += int((ps == ys).sum()); co += int((pc == yc).sum())
            pair += int(((ps == ys) & (pc == yc)).sum()); total += len(ys)
    return {"shape_accuracy": st / total, "color_accuracy": co / total, "joint_accuracy": pair / total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rotate", type=float, default=15); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shape-loss-weight", type=float, default=1.5)
    p.add_argument("--color-loss-weight", type=float, default=1.0)
    p.add_argument("--device", default="auto"); p.add_argument("--name", default="multitask_round1")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    assert_no_leakage(args.data)
    train = MultiDataset(args.data / "train", args.imgsz, True, args.rotate)
    val = MultiDataset(args.data / "val", args.imgsz, False)
    test = MultiDataset(args.data / "test", args.imgsz, False)
    loaders = [DataLoader(d, batch_size=args.batch, shuffle=i == 0, num_workers=0) for i, d in enumerate((train, val, test))]
    model = MultiTaskClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    loss_fn = nn.CrossEntropyLoss()
    out = Path("runs/stage1") / args.name; out.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out))
    except Exception:
        writer = None
    best = (-1.0, -1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for x, ys, yc in loaders[0]:
            x, ys, yc = x.to(device), ys.to(device), yc.to(device)
            opt.zero_grad(set_to_none=True); ps, pc = model(x)
            loss = args.shape_loss_weight * loss_fn(ps, ys) + args.color_loss_weight * loss_fn(pc, yc)
            loss.backward(); opt.step(); running += loss.item() * len(ys)
        metrics = evaluate(model, loaders[1], device); scheduler.step()
        if writer:
            writer.add_scalar("loss/train", running / len(train), epoch)
            for key, value in metrics.items(): writer.add_scalar(f"val/{key}", value, epoch)
        score = (metrics["joint_accuracy"], metrics["shape_accuracy"], metrics["color_accuracy"])
        print(f"epoch={epoch:03d} loss={running / len(train):.4f} shape={metrics['shape_accuracy']:.4f} color={metrics['color_accuracy']:.4f} joint={metrics['joint_accuracy']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "shapes": SHAPES, "colors": COLORS, "imgsz": args.imgsz}, out / "best.pt")
    checkpoint = torch.load(out / "best.pt", map_location=device); model.load_state_dict(checkpoint["model"])
    metrics = {"best_val_joint": best[0], "best_val_shape": best[1], "best_val_color": best[2], "test": evaluate(model, loaders[2], device), "shapes": SHAPES, "colors": COLORS}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if writer: writer.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
