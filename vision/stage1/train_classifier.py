#!/usr/bin/env python3
"""Train the stage-one single-image four-class classifier."""

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

CLASSES = ["cuboid", "tetrahedron", "prism", "sphere"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_items(root: Path):
    items = []
    for cls_id, cls in enumerate(CLASSES):
        for path in sorted((root / cls).glob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                items.append((path, cls_id))
    return items


def assert_no_exact_leakage(data_root: Path):
    seen = {}
    for split in ("train", "val", "test"):
        for path, _ in collect_items(data_root / split):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = seen.get(digest)
            if previous and previous[0] != split:
                raise RuntimeError(f"Exact image leakage: {path} appears in {previous[0]} as {previous[1]}")
            seen[digest] = (split, str(path))


class FolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int, train: bool, rotate: float = 0.0, color_jitter: float = 0.0):
        self.items = []
        self.image_size = image_size
        self.train = train
        self.rotate = rotate
        self.color_jitter = color_jitter
        self.items = collect_items(root)
        if not self.items:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        if self.train and self.rotate > 0 and random.random() < 0.7:
            image = image.rotate(random.uniform(-self.rotate, self.rotate), resample=Image.Resampling.BILINEAR)
        if self.train and self.color_jitter > 0:
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(factor)
        x = np.asarray(image, dtype=np.float32) / 255.0
        if self.train and random.random() < 0.5:
            x = x[:, ::-1].copy()
        x = torch.from_numpy(x.transpose(2, 0, 1))
        x = (x - 0.5) / 0.5
        return x, label


class Encoder(nn.Module):
    """Small CPU-friendly encoder that can be shared by the later two-view model."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(192, 4))

    def forward(self, x):
        return self.head(self.encoder(x))


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    matrix = torch.zeros((4, 4), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            y = y.cpu()
            correct += int((pred == y).sum())
            total += len(y)
            for t, p in zip(y, pred): matrix[int(t), int(p)] += 1
    return correct / max(1, total), matrix.tolist()


def load_best_and_evaluate(model, checkpoint_path, loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return accuracy(model, loader, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rotate", type=float, default=0.0)
    p.add_argument("--color-jitter", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--name", default="stage1")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    assert_no_exact_leakage(args.data)
    train = FolderDataset(args.data / "train", args.imgsz, True, args.rotate, args.color_jitter)
    val = FolderDataset(args.data / "val", args.imgsz, False)
    test = FolderDataset(args.data / "test", args.imgsz, False)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    out = Path("runs/stage1") / args.name
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out))
    except Exception:
        pass
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); optimizer.step(); running += loss.detach().item() * len(y)
        val_acc, val_matrix = accuracy(model, val_loader, device)
        train_loss = running / len(train)
        scheduler.step()
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"epoch={epoch:03d} loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best:
            best = val_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES, "imgsz": args.imgsz}, out / "best.pt")
    test_acc, test_matrix = load_best_and_evaluate(model, out / "best.pt", test_loader, device)
    metrics = {"best_val_accuracy": best, "test_accuracy_of_best_checkpoint": test_acc, "test_confusion": test_matrix, "classes": CLASSES}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    if writer: writer.close()


if __name__ == "__main__":
    main()
