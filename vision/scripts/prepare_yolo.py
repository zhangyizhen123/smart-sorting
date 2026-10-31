#!/usr/bin/env python3
"""Validate a YOLO dataset and split it by physical scene/group.

Expected source layout:
  raw/images/**/*.jpg
  raw/labels/**/*.txt

Images from the same tray arrangement must share the same group id. By default
the part before the first ``__`` in the filename is used, e.g.
``scene042__frame003.jpg`` and ``scene042__frame004.jpg``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = [
    "cylinder",
    "cube",
    "cuboid",
    "tri_prism",
    "pent_prism",
    "hex_prism",
    "cone",
    "tetrahedron",
    "sphere",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--group-csv", type=Path, default=None)
    p.add_argument("--group-regex", default=None, help="Regex with named group 'group' or group 1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--names", type=Path, default=None)
    return p.parse_args()


def load_names(path: Path | None) -> list[str]:
    if path is None:
        return NAMES
    names = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not names:
        raise ValueError("--names is empty")
    return names


def load_group_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["image"]] = row["group_id"]
    return result


def group_id(stem: str, explicit: dict[str, str], regex: re.Pattern[str] | None) -> str:
    if stem in explicit:
        return explicit[stem]
    if regex is not None:
        match = regex.search(stem)
        if not match:
            raise ValueError(f"group regex did not match {stem}")
        return match.groupdict().get("group") or match.group(1)
    return stem.split("__", 1)[0]


def validate_label(path: Path, nclasses: int) -> list[int]:
    class_ids: list[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"{path}:{line_no}: fewer than 5 fields")
        cls = int(fields[0])
        if not 0 <= cls < nclasses:
            raise ValueError(f"{path}:{line_no}: class {cls} outside [0, {nclasses})")
        values = [float(x) for x in fields[1:]]
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in values):
            raise ValueError(f"{path}:{line_no}: coordinates must be finite and normalized")
        if len(values) != 4 and (len(values) < 6 or len(values) % 2 != 0):
            raise ValueError(f"{path}:{line_no}: invalid bbox or polygon length")
        class_ids.append(cls)
    return class_ids


def choose_splits(groups: list[str], sizes: dict[str, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    rng.shuffle(groups)
    assigned: dict[str, str] = {}
    counts = defaultdict(int)
    for group in groups:
        split = max(
            sizes,
            key=lambda key: sizes[key] - counts[key] / max(1, len(groups)),
        )
        assigned[group] = split
        counts[split] += 1
    return assigned


def main() -> None:
    args = parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must equal 1")
    names = load_names(args.names)
    explicit_groups = load_group_csv(args.group_csv)
    regex = re.compile(args.group_regex) if args.group_regex else None

    records: list[dict] = []
    errors: list[str] = []
    for image in sorted(args.images.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            label = args.labels / image.relative_to(args.images).with_suffix(".txt")
            if not label.exists():
                label = args.labels / f"{image.stem}.txt"
            if not label.exists():
                raise ValueError("missing label")
            classes = validate_label(label, len(names))
            records.append({
                "image": image,
                "label": label,
                "group_id": group_id(image.stem, explicit_groups, regex),
                "width": width,
                "height": height,
                "classes": ",".join(map(str, sorted(set(classes)))),
            })
        except Exception as exc:
            errors.append(f"{image}: {exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    if errors:
        (args.out / "dataset_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Skipped {len(errors)} invalid files; see {args.out / 'dataset_errors.txt'}")
    if not records:
        raise RuntimeError("No valid image/label pairs found")

    groups = sorted({row["group_id"] for row in records})
    splits = choose_splits(
        groups,
        {"train": args.train, "val": args.val, "test": args.test},
        args.seed,
    )
    manifest_rows = []
    for row in records:
        split = splits[row["group_id"]]
        image_name = row["image"].name
        if not image_name.startswith(f"{row['group_id']}__"):
            image_name = f"{row['group_id']}__{image_name}"
        target_image = args.out / "images" / split / image_name
        target_label = args.out / "labels" / split / f"{target_image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image"], target_image)
        shutil.copy2(row["label"], target_label)
        manifest_rows.append({
            "image": str(target_image.relative_to(args.out)),
            "label": str(target_label.relative_to(args.out)),
            "split": split,
            "group_id": row["group_id"],
            "width": row["width"],
            "height": row["height"],
            "classes": row["classes"],
        })

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_lines = [
        f"path: {args.out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        f"names: {json.dumps(names, ensure_ascii=False)}",
    ]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Valid pairs: {len(records)}, groups: {len(groups)}")
    for split in ("train", "val", "test"):
        print(split, sum(row["split"] == split for row in manifest_rows))


if __name__ == "__main__":
    main()
