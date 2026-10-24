#!/usr/bin/env python3
"""Create a small, labeled synthetic four-class dataset for pipeline testing."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["cuboid", "tetrahedron", "prism", "sphere"]


def polygon(draw: ImageDraw.ImageDraw, points, fill, outline=(35, 35, 35), width=2):
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def render(cls: str, rng: random.Random, size: int = 128) -> Image.Image:
    bg = tuple(rng.randint(150, 220) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    cx, cy = size // 2 + rng.randint(-12, 12), size // 2 + rng.randint(-8, 8)
    scale = rng.randint(27, 42)
    color = tuple(rng.randint(55, 220) for _ in range(3))
    shade = tuple(max(0, int(x * 0.65)) for x in color)
    light = tuple(min(255, int(x * 1.18)) for x in color)
    angle = math.radians(rng.uniform(0, 360))

    if cls == "sphere":
        box = (cx - scale, cy - scale, cx + scale, cy + scale)
        d.ellipse(box, fill=color, outline=(35, 35, 35), width=2)
        hi = max(3, scale // 4)
        d.ellipse((cx - scale // 3, cy - scale // 3, cx - scale // 3 + hi, cy - scale // 3 + hi), fill=light)
    elif cls == "tetrahedron":
        pts = [(cx, cy - scale), (cx - scale, cy + scale // 2), (cx + scale, cy + scale // 2)]
        polygon(d, pts, color)
        d.line((cx, cy - scale, cx, cy + scale // 2), fill=light, width=2)
        d.polygon([(cx, cy - scale), (cx, cy + scale // 2), (cx + scale, cy + scale // 2)], fill=shade)
        d.line((cx, cy - scale, cx, cy + scale // 2), fill=light, width=2)
    elif cls == "prism":
        # A cylinder-like body with two curved ends and an oblique side cue.
        w, h = scale * 2, max(18, scale)
        left, top = cx - w // 2, cy - h // 2
        d.rounded_rectangle((left, top, left + w, top + h), radius=h // 2, fill=color, outline=(35, 35, 35), width=2)
        d.ellipse((left, top, left + h, top + h), fill=light, outline=(35, 35, 35), width=2)
        d.line((left + h // 2, top + 3, left + h // 2, top + h - 3), fill=shade, width=2)
    else:  # cuboid
        w, h = scale * 2, max(28, int(scale * 1.35))
        dx, dy = int(scale * 0.35 * math.cos(angle)), int(scale * 0.25 * math.sin(angle))
        front = [(cx - w // 2, cy - h // 2), (cx + w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (cx - w // 2, cy + h // 2)]
        top_face = [front[0], front[1], (front[1][0] + dx, front[1][1] - dy), (front[0][0] + dx, front[0][1] - dy)]
        side_face = [front[1], front[2], (front[2][0] + dx, front[2][1] - dy), (front[1][0] + dx, front[1][1] - dy)]
        polygon(d, top_face, light)
        polygon(d, side_face, shade)
        polygon(d, front, color)

    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 0.6)))
    return im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", type=int, default=240)
    p.add_argument("--val", type=int, default=60)
    p.add_argument("--test", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        for cls in CLASSES:
            target = args.out / split / cls
            target.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                render(cls, rng).save(target / f"{cls}_{idx:05d}.jpg", quality=95)
    print(f"Generated {sum((args.train, args.val, args.test)) * len(CLASSES)} images at {args.out}")


if __name__ == "__main__":
    main()
