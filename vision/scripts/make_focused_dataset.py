#!/usr/bin/env python3
"""Versioned paired STL scenes with visible-instance masks and frozen splits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageFilter
import yaml

from make_stl_synthetic import CLASSES, MODEL_DIR, load_binary_stl, make_background, make_sphere_proxy, render_instance

VERSION = "focused_v2"
MODES = ("single", "separated", "multi", "low_view", "high_view", "small", "edge", "low_contrast", "contact", "occlusion")
PARTNERS = {0: (6,), 1: (2,), 2: (1, 3), 3: (2,), 4: (5,), 5: (4,), 6: (0,), 7: (3,), 8: (0,)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rotation(mesh: np.ndarray, rng: random.Random) -> np.ndarray:
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals = np.unique(np.round(normals, 4), axis=0)
    normal = normals[rng.randrange(len(normals))]
    normal /= max(np.linalg.norm(normal), 1e-12)
    down = np.array([0., 0., -1.])
    cross = np.cross(normal, down)
    sine, cosine = np.linalg.norm(cross), float(normal @ down)
    if sine < 1e-8:
        align = np.diag([1., -1., -1.]) if cosine < 0 else np.eye(3)
    else:
        kx, ky, kz = cross / sine
        skew = np.array([[0., -kz, ky], [kz, 0., -kx], [-ky, kx, 0.]])
        align = np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)
    yaw = rng.uniform(-math.pi, math.pi)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    return rotation @ align


def place_mask(layer: Image.Image, position: tuple[int, int], size: int) -> np.ndarray:
    canvas = Image.new("L", (size, size))
    canvas.paste(layer.getchannel("A").point(lambda value: 255 if value >= 128 else 0), position)
    return np.asarray(canvas) != 0


def visible_polygons(masks: list[np.ndarray]) -> list[np.ndarray]:
    visible = [mask.copy() for mask in masks]
    for index in range(len(masks)):
        for later in masks[index + 1:]:
            visible[index] &= ~later
    polygons = []
    for full, mask in zip(masks, visible):
        if mask.sum() < 120 or mask.sum() / max(1, full.sum()) < 0.6:
            raise ValueError("Object too occluded or too small")
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1 or hierarchy is None:
            raise ValueError("Visible YOLO polygon must be connected and have no holes")
        polygon = cv2.approxPolyDP(contours[0], 0.35, True).reshape(-1, 2)
        if len(polygon) < 3 or cv2.contourArea(polygon) < 100:
            raise ValueError("Degenerate polygon")
        polygons.append(polygon)
    return polygons


def render_scene(meshes: dict, target: int, index: int, size: int, seed: int) -> tuple[Image.Image, list, dict]:
    mode = MODES[index % len(MODES)]
    for attempt in range(100):
        rng = random.Random(seed + attempt * 1000003)
        partner = PARTNERS[target][(index // len(MODES)) % len(PARTNERS[target])]
        ids = [target] if mode == "single" else [target, partner]
        if mode == "multi":
            ids += rng.sample([i for i in range(9) if i not in ids], rng.choice((1, 2)))
        rng.shuffle(ids)
        elevation = rng.uniform(12, 25) if mode == "low_view" else rng.uniform(75, 85) if mode == "high_view" else rng.uniform(35, 65)
        azimuth = rng.uniform(0, 360)
        ppm = size / 190 * rng.uniform(0.8, 1.02)
        if mode == "small":
            ppm *= 0.48
        background = make_background(size, rng)
        material = tuple(rng.randint(70, 230) for _ in range(3))
        if mode == "low_contrast":
            mean = np.asarray(background)[..., :3].mean(axis=(0, 1))
            material = tuple(int(np.clip(v + rng.uniform(-12, 12), 20, 245)) for v in mean)
        light = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(0.4, 1.)])
        layers, poses = [], []
        same_color = rng.random() < 0.5 or mode == "low_contrast"
        for cls in ids:
            rotation = stable_rotation(meshes[CLASSES[cls]], rng)
            layer, _ = render_instance(CLASSES[cls], meshes[CLASSES[cls]], rng, 1., elevation, azimuth,
                                      pixels_per_mm=ppm, rotation=rotation,
                                      material_override=material if same_color else None, light_override=light)
            layers.append(layer)
            poses.append(rotation.round(6).tolist())
        positions = []
        for slot, layer in enumerate(layers):
            if len(ids) == 1:
                cx, cy = size * rng.uniform(0.3, 0.7), size * rng.uniform(0.3, 0.7)
            elif len(ids) == 2:
                cx, cy = size * (0.28 + slot * 0.44), size * rng.uniform(0.38, 0.62)
            else:
                cx, cy = size * (0.25 + (slot % 2) * 0.5), size * (0.25 + (slot // 2) * 0.5)
            positions.append((round(cx - layer.width / 2), round(cy - layer.height / 2)))
        if mode in {"contact", "occlusion"}:
            positions[0] = (round(size * 0.32 - layers[0].width / 2), round(size * 0.5 - layers[0].height / 2))
            first = place_mask(layers[0], positions[0], size)
            y = round(size * 0.5 - layers[1].height / 2)
            second_at_zero = place_mask(layers[1], (0, y), size)
            common_rows = first.any(axis=1) & second_at_zero.any(axis=1)
            if not common_rows.any():
                continue
            first_right = size - 1 - first[:, ::-1].argmax(axis=1)
            second_left = second_at_zero.argmax(axis=1)
            contact_x = int((first_right - second_left)[common_rows].max()) + 1
            positions[1] = (contact_x, y)
            if mode == "occlusion":
                target_overlap = rng.uniform(0.1, 0.25)
                left, right = contact_x - min(layer.width for layer in layers) * 0.6, contact_x
                for _ in range(10):
                    x = round((left + right) / 2)
                    overlap = (first & place_mask(layers[1], (x, y), size)).sum() / max(1, first.sum())
                    if overlap < target_overlap:
                        right = x
                    else:
                        left = x
                positions[1] = (round((left + right) / 2), y)
        if mode == "edge":
            positions[0] = (-rng.randint(2, max(3, layers[0].width // 12)), positions[0][1])
        masks = [place_mask(layer, pos, size) for layer, pos in zip(layers, positions)]
        if mode != "occlusion" and any(np.any(a & b) for i, a in enumerate(masks) for b in masks[i + 1:]):
            continue
        try:
            polygons = visible_polygons(masks)
        except ValueError:
            continue
        for layer, (x, y) in zip(layers, positions):
            shadow = Image.new("RGBA", layer.size, (15, 15, 15, 0))
            shadow.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(3)).point(lambda v: v // 5))
            background.alpha_composite(shadow, (x + 4, y + 4))
        for layer, pos in zip(layers, positions):
            background.alpha_composite(layer, pos)
        return background.convert("RGB"), list(zip(ids, polygons)), {
            "mode": mode, "camera_elevation_deg": elevation, "camera_azimuth_deg": azimuth,
            "pixels_per_mm": ppm, "poses": poses, "seed": seed, "attempt": attempt,
            "visible_fractions": [float(np.count_nonzero(mask & ~np.logical_or.reduce(masks[i + 1:]))) / max(1, mask.sum()) if i + 1 < len(masks) else 1. for i, mask in enumerate(masks)],
        }
    raise RuntimeError(f"Could not render valid scene target={target}, index={index}, seed={seed}")


def verify_dataset(root: Path) -> dict:
    lock = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    if digest(root / "manifest.jsonl") != lock["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    seen, groups = {}, set()
    for row in records:
        if row["group_id"] in groups:
            raise ValueError("Duplicate scene group")
        groups.add(row["group_id"])
        for key in ("image", "label"):
            if digest(root / row[key]) != row[key + "_sha256"]:
                raise ValueError(f"Frozen {key} changed: {row[key]}")
        key = row["image_sha256"]
        if key in seen and seen[key] != row["split"]:
            raise ValueError("Image hash leaked across splits")
        seen[key] = row["split"]
    return lock


def build(root: Path, counts: dict, size: int, seed: int) -> dict:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new output directory; refusing to modify {root}")
    root.mkdir(parents=True, exist_ok=True)
    meshes = {name: load_binary_stl(MODEL_DIR / f"{name}.stl") for name in CLASSES}
    meshes["sphere"] = make_sphere_proxy(meshes["sphere"])
    records, summary = [], {}
    for split, count in counts.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        totals = Counter()
        for target in range(9):
            for index in range(count):
                key = f"{VERSION}_{seed}_{split}_{target:02d}_{index:05d}"
                scene_seed = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
                image, instances, meta = render_scene(meshes, target, index, size, scene_seed)
                image_path, label_path = f"images/{split}/{key}.jpg", f"labels/{split}/{key}.txt"
                image.save(root / image_path, quality=95, subsampling=0)
                lines = [str(cls) + " " + " ".join(f"{v:.7f}" for v in (poly.astype(float) / size).ravel()) for cls, poly in instances]
                (root / label_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
                totals.update(CLASSES[cls] for cls, _ in instances)
                records.append({"group_id": key, "split": split, "target_class": CLASSES[target],
                    "image": image_path, "label": label_path, "image_sha256": digest(root / image_path),
                    "label_sha256": digest(root / label_path), "classes": [cls for cls, _ in instances], **meta})
            print(f"{split}: {CLASSES[target]} {count} scenes complete", flush=True)
        summary[split] = {"images": count * 9, "instances": sum(totals.values()), "per_class": dict(totals)}
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    lock = {"version": VERSION, "seed": seed, "image_size": size, "counts": counts, "summary": summary,
            "manifest_sha256": digest(manifest), "stl_sha256": {name: digest(MODEL_DIR / f"{name}.stl") for name in CLASSES},
            "generator_sha256": digest(Path(__file__)), "renderer_sha256": digest(Path(__file__).with_name("make_stl_synthetic.py")),
            "note": "Same-scene physical scale; stable face-supported poses; visible connected polygons; no external photos"}
    (root / "freeze.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": root.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES, "nc": 9}, allow_unicode=True), encoding="utf-8")
    verify_dataset(root)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=320)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_dataset(args.out), indent=2)); return
    counts = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    if min(counts.values()) < 1 or args.image_size < 256:
        parser.error("Positive split sizes and image-size >=256 required")
    print(json.dumps(build(args.out, counts, args.image_size, args.seed), indent=2))


if __name__ == "__main__":
    main()
