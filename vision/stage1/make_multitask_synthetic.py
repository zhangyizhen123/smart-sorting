#!/usr/bin/env python3
"""Create a leakage-free smoke dataset with shape and color labels.

Images are stored as split/shape/color/image.jpg so labels are explicit and
can later be replaced by real camera data without changing the trainer.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SHAPES = ["cuboid", "tetrahedron", "prism", "sphere"]
COLORS = {"red": (210, 55, 55), "green": (55, 175, 85), "blue": (55, 105, 210), "yellow": (220, 185, 45)}


def polygon(draw, points, fill, outline=(35, 35, 35), width=2):
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def render(shape: str, color_name: str, rng: random.Random, size: int = 128) -> Image.Image:
    base = COLORS[color_name]
    bg = tuple(rng.randint(145, 220) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-8, 8)
    scale = rng.randint(28, 42)
    shade = tuple(int(v * 0.60) for v in base)
    light = tuple(min(255, int(v * 1.20)) for v in base)
    angle = math.radians(rng.uniform(0, 360))
    if shape == "sphere":
        box = (cx - scale, cy - scale, cx + scale, cy + scale)
        d.ellipse(box, fill=base, outline=(35, 35, 35), width=2)
        # Multiple highlights make the sphere less like a flat disk.
        d.ellipse((cx - scale // 3, cy - scale // 3, cx - scale // 3 + scale // 4,
                   cy - scale // 3 + scale // 4), fill=light)
    elif shape == "tetrahedron":
        pts = [(cx, cy - scale), (cx - scale, cy + scale // 2), (cx + scale, cy + scale // 2)]
        polygon(d, pts, base)
        d.polygon([(cx, cy - scale), (cx, cy + scale // 2), (cx + scale, cy + scale // 2)], fill=shade)
        d.line((cx, cy - scale, cx, cy + scale // 2), fill=light, width=2)
    elif shape == "prism":
        w, h = scale * 2, max(18, scale)
        left, top = cx - w // 2, cy - h // 2
        d.rounded_rectangle((left, top, left + w, top + h), radius=h // 2, fill=base, outline=(35, 35, 35), width=2)
        d.ellipse((left, top, left + h, top + h), fill=light, outline=(35, 35, 35), width=2)
        d.line((left + h // 2, top + 3, left + h // 2, top + h - 3), fill=shade, width=2)
    else:
        w, h = scale * 2, max(28, int(scale * 1.35))
        dx, dy = int(scale * 0.35 * math.cos(angle)), int(scale * 0.25 * math.sin(angle))
        front = [(cx - w // 2, cy - h // 2), (cx + w // 2, cy - h // 2),
                 (cx + w // 2, cy + h // 2), (cx - w // 2, cy + h // 2)]
        top = [front[0], front[1], (front[1][0] + dx, front[1][1] - dy),
               (front[0][0] + dx, front[0][1] - dy)]
        side = [front[1], front[2], (front[2][0] + dx, front[2][1] - dy),
                (front[1][0] + dx, front[1][1] - dy)]
        polygon(d, top, light); polygon(d, side, shade); polygon(d, front, base)
    # Simulate illumination changes while preserving the color identity.
    if rng.random() < 0.8:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.72, 1.28))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.45:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.7)))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", type=int, default=80)
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--test", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        for shape in SHAPES:
            for color in COLORS:
                target = args.out / split / shape / color
                target.mkdir(parents=True, exist_ok=True)
                for idx in range(count):
                    render(shape, color, rng).save(target / f"{shape}_{color}_{idx:05d}.jpg", quality=95)
    print(f"Generated {sum((args.train, args.val, args.test)) * len(SHAPES) * len(COLORS)} images at {args.out}")


if __name__ == "__main__":
    main()
