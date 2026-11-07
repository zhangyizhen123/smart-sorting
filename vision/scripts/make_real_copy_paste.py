"""Compose training scenes from human-approved real instances pasted onto real empty-tray photos.

Training data only (D-34). Every instance and background must come from the training split; groups held
out for validation or testing are refused. Labels are exported from the approved masks with the same
round-trip check as prepare_real_training.mask_polygon, so a pasted object is never labelled by a
prediction. Outputs are marked final_test_eligible=False: a pasted photo is not independent real data.

Typical use on the machine that holds dataset/real:

    python vision/scripts/make_real_copy_paste.py bank --review dataset/real/annotations/review_v2 \
        --out dataset/real_copy_paste_bank_v1.json
    python vision/scripts/make_real_copy_paste.py generate --bank dataset/real_copy_paste_bank_v1.json \
        --config vision/configs/real_copy_paste_v1.json --out dataset/real_copy_paste_v1
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

from import_real_review import NAMES, contained, digest, validate_review
from prepare_real_training import mask_polygon

ROOT = Path(__file__).resolve().parents[2]
BANK_VERSION = 1
DEFAULTS = {
    "scenes": 2000,
    "image_size": 640,
    "min_instances": 1,
    "max_instances": 6,
    "empty_fraction": 0.05,
    "scale_jitter": [0.85, 1.15],
    "max_rotation_deg": 10.0,
    "flip_probability": 0.5,
    "gain_range": [0.9, 1.1],
    "feather_sigma_px": 0.8,
    "shadow_strength": 0.25,
    "shadow_offset_px": [2, 6],
    "shadow_blur_px": 3.0,
    "min_gap_px": 3,
    "roi_margin_px": 4,
    "placement_attempts": 40,
    "min_instance_side_px": 12,
    "jpeg_quality": 95,
    "seed": 42,
}


@dataclass(frozen=True)
class Cutout:
    key: str
    class_id: int
    group_id: str
    rgb: np.ndarray          # crop of the source photo around the mask, uint8 HxWx3
    mask: np.ndarray         # bool HxW, same crop
    relative_diag: float     # mask bbox diagonal / source image diagonal


@dataclass(frozen=True)
class Background:
    key: str
    group_id: str
    rgb: np.ndarray          # full photo, uint8
    roi: list[list[float]] | None   # optional polygon in normalized image coordinates


# ----------------------------------------------------------------------------- bank


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def bank_from_review(review_dir: Path, out_path: Path, *, exclude_groups: tuple[str, ...] = (),
                     source_root: Path = ROOT) -> dict:
    """Write a bank JSON from the current approved review snapshot.

    Uses the same snapshot pointer and validation as prepare_real_training. Only approved images are used:
    their instances become paste sources, approved empty-tray photos become backgrounds.
    """
    review_dir = review_dir.resolve()
    draft = json.loads((review_dir / "annotations.json").read_text(encoding="utf-8"))
    pointer = json.loads((review_dir / "current_review.json").read_text(encoding="utf-8"))
    snapshot = contained(review_dir, pointer["snapshot"])
    if digest(snapshot) != pointer["snapshot_sha256"]:
        raise ValueError("Review snapshot changed")
    approved = validate_review(draft, json.loads(snapshot.read_text(encoding="utf-8"))["export"])
    base = out_path.resolve().parent
    instances, backgrounds = [], []
    for row in approved["images"]:
        if row.get("human_review_status") != "approved":
            continue
        if row["leakage_group"] in exclude_groups:
            continue
        if digest(contained(source_root, row["source"])) != row["source_sha256"]:
            raise ValueError(f"Source photograph changed: {row['source']}")
        image = contained(review_dir, row["image"])
        entry_image = Path(image).resolve()
        if row.get("category") == "empty_tray" and not row["instances"]:
            backgrounds.append({"key": row["image_id"], "image": _relative(entry_image, base),
                                "image_sha256": digest(entry_image), "group_id": row["leakage_group"]})
            continue
        for instance in row["instances"]:
            if not instance.get("boundary_human_approved"):
                raise ValueError("Unapproved mask in an approved image")
            mask = contained(review_dir, instance["mask"])
            if digest(mask) != instance["mask_sha256"]:
                raise ValueError("Mask changed since review")
            instances.append({"key": instance["instance_id"], "image": _relative(entry_image, base),
                              "image_sha256": digest(entry_image), "mask": _relative(Path(mask), base),
                              "mask_sha256": instance["mask_sha256"], "class_id": instance["class_id"],
                              "group_id": row["leakage_group"]})
    bank = {"version": BANK_VERSION, "split": "train", "source": "approved_review_snapshot",
            "review_snapshot_sha256": pointer["snapshot_sha256"], "held_out_groups": sorted(exclude_groups),
            "instances": instances, "backgrounds": backgrounds}
    if not instances or not backgrounds:
        raise ValueError("Bank needs at least one approved instance and one approved empty-tray background")
    out_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"instances": len(instances), "backgrounds": len(backgrounds),
            "classes": dict(sorted(Counter(NAMES[i["class_id"]] for i in instances).items()))}


def _relative(path: Path, base: Path) -> str:
    return Path(path).resolve().relative_to(base).as_posix() if Path(path).resolve().is_relative_to(base) \
        else Path(path).resolve().as_posix()


def load_bank(path: Path) -> tuple[list[Cutout], list[Background], dict]:
    """Load and verify a bank: training split only, no held-out groups, unchanged files, valid masks."""
    bank = json.loads(path.read_text(encoding="utf-8"))
    if bank.get("version") != BANK_VERSION:
        raise ValueError("Unsupported bank version")
    if bank.get("split") != "train":
        raise ValueError("Copy-paste sources must come from the training split only")
    held_out = set(bank.get("held_out_groups", []))
    base = path.resolve().parent
    cutouts: list[Cutout] = []
    images: dict[str, np.ndarray] = {}
    for entry in bank["instances"]:
        if entry["group_id"] in held_out:
            raise ValueError(f"Instance {entry['key']} belongs to held-out group {entry['group_id']}")
        if not 0 <= int(entry["class_id"]) < len(NAMES):
            raise ValueError(f"Unknown class id for {entry['key']}")
        image_path, mask_path = base / entry["image"], base / entry["mask"]
        if digest(image_path) != entry["image_sha256"] or digest(mask_path) != entry["mask_sha256"]:
            raise ValueError(f"Source file changed for instance {entry['key']}")
        if entry["image"] not in images:
            images[entry["image"]] = _load_rgb(image_path)
        rgb = images[entry["image"]]
        mask = _load_mask(mask_path)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask size differs from image for {entry['key']}")
        mask_polygon(mask)  # one connected instance without holes, else refuse
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        diag = math.hypot(x1 - x0, y1 - y0) / math.hypot(*rgb.shape[:2])
        cutouts.append(Cutout(entry["key"], int(entry["class_id"]), entry["group_id"],
                              rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), diag))
    backgrounds: list[Background] = []
    for entry in bank["backgrounds"]:
        if entry["group_id"] in held_out:
            raise ValueError(f"Background {entry['key']} belongs to held-out group {entry['group_id']}")
        image_path = base / entry["image"]
        if digest(image_path) != entry["image_sha256"]:
            raise ValueError(f"Background changed: {entry['key']}")
        backgrounds.append(Background(entry["key"], entry["group_id"], _load_rgb(image_path), entry.get("roi")))
    if not cutouts or not backgrounds:
        raise ValueError("Bank needs instances and backgrounds")
    return cutouts, backgrounds, bank


# ----------------------------------------------------------------------------- composition


def _resize_longest(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _roi_mask(shape: tuple[int, int], roi: list[list[float]] | None, margin: int) -> np.ndarray:
    h, w = shape
    region = np.zeros((h, w), np.uint8)
    if roi:
        points = np.rint(np.asarray(roi, np.float64) * [w, h]).astype(np.int32)
        cv2.fillPoly(region, [points], 1)
    else:
        region[:] = 1
    if margin > 0:
        region = cv2.erode(region, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8), borderValue=0)
    return region.astype(bool)


def transform_cutout(cutout: Cutout, target_diag: float, angle_deg: float, flip: bool,
                     gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale, mirror and rotate a cutout. Returns (rgb float32, mask bool) on a canvas that fits the result."""
    h, w = cutout.mask.shape
    scale = target_diag / math.hypot(w, h)
    rgb = cutout.rgb.astype(np.float32) * gain
    mask = cutout.mask.astype(np.float32)
    if flip:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    cos, sin = abs(math.cos(math.radians(angle_deg))), abs(math.sin(math.radians(angle_deg)))
    out_w = max(1, math.ceil((w * cos + h * sin) * scale))
    out_h = max(1, math.ceil((w * sin + h * cos) * scale))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    matrix[0, 2] += out_w / 2 - w / 2
    matrix[1, 2] += out_h / 2 - h / 2
    rgb = cv2.warpAffine(np.ascontiguousarray(rgb), matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 0.5
    return np.clip(rgb, 0, 255), mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def _class_weights(cutouts: list[Cutout]) -> list[float]:
    counts = Counter(c.class_id for c in cutouts)
    return [1.0 / counts[c.class_id] for c in cutouts]


def compose_scene(background: Background, cutouts: list[Cutout], weights: list[float], count: int,
                  rng: random.Random, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """Paste up to `count` instances without overlap inside the ROI. Returns image and placed instances."""
    canvas = _resize_longest(background.rgb, cfg["image_size"]).astype(np.float32)
    h, w = canvas.shape[:2]
    allowed = _roi_mask((h, w), background.roi, cfg["roi_margin_px"])
    occupied = np.zeros((h, w), bool)
    gap = cfg["min_gap_px"]
    dilate_kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8) if gap > 0 else None
    frame_diag = math.hypot(w, h)
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(*cfg["shadow_offset_px"])
    shadow_offset = (round(math.cos(angle) * distance), round(math.sin(angle) * distance))
    placed: list[dict] = []
    for _ in range(count):
        index = rng.choices(range(len(cutouts)), weights=weights, k=1)[0]
        cutout = cutouts[index]
        target_diag = cutout.relative_diag * frame_diag * rng.uniform(*cfg["scale_jitter"])
        rotation = rng.uniform(-cfg["max_rotation_deg"], cfg["max_rotation_deg"])
        flip = rng.random() < cfg["flip_probability"]
        gain = rng.uniform(*cfg["gain_range"])
        rgb, mask = transform_cutout(cutout, target_diag, rotation, flip, gain)
        mask = _largest_component(mask)
        mh, mw = mask.shape
        if min(mh, mw) < cfg["min_instance_side_px"] or mh >= h or mw >= w or not mask.any():
            continue
        for _attempt in range(cfg["placement_attempts"]):
            y, x = rng.randrange(0, h - mh + 1), rng.randrange(0, w - mw + 1)
            full = np.zeros((h, w), bool)
            full[y:y + mh, x:x + mw] = mask
            if (full & ~allowed).any():
                continue
            if (full & occupied).any():
                continue
            try:
                polygon, iou = mask_polygon(full.astype(np.uint8))
            except ValueError:
                continue
            _paste(canvas, rgb, mask, x, y, shadow_offset, cfg)
            grown = cv2.dilate(full.astype(np.uint8), dilate_kernel).astype(bool) if dilate_kernel is not None else full
            occupied |= grown
            placed.append({"source": cutout.key, "class_id": cutout.class_id, "group_id": cutout.group_id,
                           "x": x, "y": y, "width": mw, "height": mh, "rotation_deg": round(rotation, 4),
                           "flip": flip, "gain": round(gain, 4), "polygon": polygon, "roundtrip_iou": iou,
                           "mask": full})
            break
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8), placed


def _paste(canvas: np.ndarray, rgb: np.ndarray, mask: np.ndarray, x: int, y: int,
           shadow_offset: tuple[int, int], cfg: dict) -> None:
    h, w = canvas.shape[:2]
    mh, mw = mask.shape
    strength = cfg["shadow_strength"]
    if strength > 0:
        dx, dy = shadow_offset
        shadow = np.zeros((h, w), np.float32)
        sx0, sy0 = x + dx, y + dy
        cx0, cy0 = max(0, sx0), max(0, sy0)
        cx1, cy1 = min(w, sx0 + mw), min(h, sy0 + mh)
        if cx1 > cx0 and cy1 > cy0:
            shadow[cy0:cy1, cx0:cx1] = mask[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0]
            shadow = cv2.GaussianBlur(shadow, (0, 0), cfg["shadow_blur_px"])
            canvas *= (1.0 - strength * shadow)[..., None]
    alpha = mask.astype(np.float32)
    if cfg["feather_sigma_px"] > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), cfg["feather_sigma_px"])
        alpha = np.where(mask, np.maximum(alpha, 0.5), alpha)
    region = canvas[y:y + mh, x:x + mw]
    region[:] = alpha[..., None] * rgb + (1.0 - alpha[..., None]) * region


# ----------------------------------------------------------------------------- dataset


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if path is not None:
        overrides = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(overrides) - set(DEFAULTS) - {"description"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update({k: v for k, v in overrides.items() if k != "description"})
    if not 0 <= cfg["empty_fraction"] < 1 or not 1 <= cfg["min_instances"] <= cfg["max_instances"]:
        raise ValueError("Invalid scene counts")
    return cfg


def generate(bank_path: Path, out: Path, cfg: dict) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty dataset directory; use a new version")
    cutouts, backgrounds, bank = load_bank(bank_path)
    weights = _class_weights(cutouts)
    rng = random.Random(cfg["seed"])
    (out / "images/train").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    records, class_counts, skipped = [], Counter(), 0
    for index in range(cfg["scenes"]):
        background = backgrounds[rng.randrange(len(backgrounds))]
        empty = rng.random() < cfg["empty_fraction"]
        wanted = 0 if empty else rng.randint(cfg["min_instances"], cfg["max_instances"])
        image, placed = compose_scene(background, cutouts, weights, wanted, rng, cfg)
        skipped += wanted - len(placed)
        key = f"copy_paste_{index:06d}"
        image_path = out / "images/train" / f"{key}.jpg"
        label_path = out / "labels/train" / f"{key}.txt"
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=cfg["jpeg_quality"], subsampling=0)
        image_path.write_bytes(buffer.getvalue())
        lines = [f"{p['class_id']} " + " ".join(f"{v:.8f}" for v in p["polygon"].ravel()) for p in placed]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        class_counts.update(NAMES[p["class_id"]] for p in placed)
        groups = sorted({background.group_id, *(p["group_id"] for p in placed)})
        records.append({
            "key": key, "split": "train", "domain": "real_copy_paste",
            "image": image_path.relative_to(out).as_posix(), "label": label_path.relative_to(out).as_posix(),
            "image_sha256": digest(image_path), "label_sha256": digest(label_path),
            "background": background.key, "requested_instances": wanted,
            "instances": [{k: v for k, v in p.items() if k not in {"polygon", "mask"}} for p in placed],
            "classes": [p["class_id"] for p in placed], "source_groups": groups,
            "group_id": "copy_paste:" + "+".join(groups), "final_test_eligible": False,
        })
    manifest = out / "manifest.jsonl"
    # LF on every OS: the frozen hashes then match whichever machine regenerates the same data.
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8",
                        newline="\n")
    summary = {
        "version": 1, "task": "D-34", "split": "train", "domain": "real_copy_paste",
        "bank_sha256": digest(bank_path), "bank_source": bank.get("source"),
        "review_snapshot_sha256": bank.get("review_snapshot_sha256"), "config": cfg,
        "manifest_sha256": digest(manifest), "images": len(records),
        "empty_images": sum(not r["instances"] for r in records), "instances": sum(class_counts.values()),
        "skipped_placements": skipped, "class_counts": dict(sorted(class_counts.items())),
        "source_instances": len(cutouts), "source_backgrounds": len(backgrounds),
        "final_test_eligible": False,
        "scope": "Training augmentation from approved real masks; not independent real validation or test data.",
    }
    (out / "freeze.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                                     newline="\n")
    verify(out)
    return summary


def verify(out: Path) -> dict:
    """Re-hash every generated file against the frozen manifest; refuse any eval-eligible or non-train row."""
    lock = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    manifest = out / "manifest.jsonl"
    if digest(manifest) != lock["manifest_sha256"]:
        raise ValueError("Copy-paste manifest changed")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if len(records) != lock["images"]:
        raise ValueError("Frozen image count changed")
    for row in records:
        if row["split"] != "train" or row["final_test_eligible"]:
            raise ValueError("Copy-paste data may only be used for training")
        for key in ("image", "label"):
            if digest(contained(out, row[key])) != row[key + "_sha256"]:
                raise ValueError(f"Changed {key}: {row[key]}")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    bank = commands.add_parser("bank", help="build a bank JSON from the approved review snapshot")
    bank.add_argument("--review", type=Path, required=True)
    bank.add_argument("--out", type=Path, required=True)
    bank.add_argument("--exclude-group", action="append", default=[],
                      help="leakage group reserved for validation/test; repeatable")
    gen = commands.add_parser("generate", help="compose a frozen training dataset from a bank")
    gen.add_argument("--bank", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--config", type=Path)
    check = commands.add_parser("verify", help="re-check a generated dataset")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bank":
        result = bank_from_review(args.review, args.out, exclude_groups=tuple(args.exclude_group))
    elif args.command == "generate":
        result = generate(args.bank, args.out, load_config(args.config))
        result = {k: v for k, v in result.items() if k != "config"}
    else:
        result = verify(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
