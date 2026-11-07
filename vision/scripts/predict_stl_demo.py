#!/usr/bin/env python3
"""Run the archived model on bundled test examples and build a standalone demo."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import webbrowser

from PIL import Image
from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs/stl_shape_v1_demo"
MODEL = ROOT / "models/trained/stl_shape_v1/best.pt"
METRICS = ROOT / "docs/视觉训练指标_STL首版.json"
TEMPLATE = ROOT / "vision/templates/stl_demo.html"
NAMES = [
    "cylinder", "cube", "cuboid", "tri_prism", "pent_prism",
    "hex_prism", "cone", "tetrahedron", "sphere",
]
CHINESE_NAMES = ["圆柱体", "正方体", "长方体", "三棱柱", "五棱柱", "六棱柱", "圆锥", "四面体", "球体"]
SELECTION = "每类取实例数最少、场景编号最小的测试图；再取尚未选中的两件、三件、四件场景各一张。选取不使用预测结果。"
PREDICTION_SETTINGS = {"imgsz": 320, "conf": 0.25, "iou": 0.7, "device": "cpu", "agnostic_nms": False, "retina_masks": True}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def prepare_samples() -> None:
    """Only needed when initially bundling the examples; inference needs no dataset/."""
    source = ROOT / "dataset/stl_synthetic_yolo"
    raw_manifest = ROOT / "dataset/stl_synthetic_raw/manifest.csv"
    with raw_manifest.open(encoding="utf-8-sig", newline="") as stream:
        raw = {row["group_id"]: row for row in csv.DictReader(stream)}
    with (source / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        splits = list(csv.DictReader(stream))

    groups: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for row in splits:
        groups.setdefault(row["group_id"], set()).add(row["split"])
        hashes.setdefault(digest(source / portable_path(row["image"])), set()).add(row["split"])
    if any(len(values) != 1 for values in groups.values()) or any(len(values) != 1 for values in hashes.values()):
        raise ValueError("A scene or exact image hash occurs across dataset splits")
    test = sorted((row for row in splits if row["split"] == "test"), key=lambda row: row["group_id"])
    chosen = []
    for name in NAMES:
        candidates = [row for row in test if raw[row["group_id"]]["target_class"] == name]
        chosen.append(min(candidates, key=lambda row: (int(raw[row["group_id"]]["instances"]), row["group_id"])))
    for count in (2, 3, 4):
        chosen.append(next(row for row in test if row not in chosen and int(raw[row["group_id"]]["instances"]) == count))

    (DEMO / "samples").mkdir(parents=True, exist_ok=True)
    cases = []
    for index, row in enumerate(chosen):
        original = raw[row["group_id"]]
        stem = f"{row['group_id']}__frame000"
        image = DEMO / "samples" / f"{stem}.jpg"
        label = DEMO / "samples" / f"{stem}.txt"
        shutil.copy2(source / portable_path(row["image"]), image)
        shutil.copy2(source / portable_path(row["label"]), label)
        cases.append({
            "id": row["group_id"],
            "title": CHINESE_NAMES[index] if index < len(NAMES) else f"{original['instances']} 个物块",
            "target_class": original["target_class"],
            "instances": int(original["instances"]),
            "split": "test",
            "image": image.relative_to(DEMO).as_posix(),
            "label": label.relative_to(DEMO).as_posix(),
            "image_sha256": digest(image),
            "label_sha256": digest(label),
        })
    manifest = {
        "selection": SELECTION,
        "source_manifest_sha256": digest(source / "manifest.csv"),
        "split_audit": {"scene_overlap": 0, "exact_image_hash_overlap": 0, "images_checked": len(splits)},
        "names": NAMES,
        "cases": cases,
    }
    (DEMO / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_truth(path: Path) -> list[dict]:
    objects = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
        if not 0 <= class_id < len(NAMES) or len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"Invalid segmentation label: {path}")
        if not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f"Invalid normalized polygon: {path}")
        objects.append({"class_id": class_id, "polygon": list(zip(coordinates[0::2], coordinates[1::2]))})
    return objects


def load_model() -> YOLO:
    model = YOLO(str(MODEL))
    if model.task != "segment" or [model.names[i] for i in range(len(model.names))] != NAMES:
        raise ValueError("The model is not the expected nine-class segmentation model")
    return model


def serialize_predictions(result, *, include_mask_geometry: bool = False) -> list[dict]:
    boxes, masks = result.boxes, result.masks
    if boxes is None or not len(boxes):
        return []
    if masks is None or len(boxes) != len(masks):
        raise ValueError("Missing instance masks")
    predictions = [
        {
            "class_id": int(class_id),
            "confidence": round(float(confidence), 6),
            "box": [round(float(value), 6) for value in box],
            "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in polygon],
        }
        for class_id, confidence, box, polygon in zip(
            boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxyn.cpu().tolist(), masks.xyn, strict=True,
        )
    ]
    if include_mask_geometry:
        from mask_display import native_mask_geometry
        if tuple(masks.data.shape[-2:]) != tuple(result.orig_shape):
            raise ValueError("Native mask display requires retina_masks=True and original-image mask coordinates")
        for prediction, mask in zip(predictions, masks.data.cpu().numpy(), strict=True):
            prediction.update(native_mask_geometry(mask))
    return predictions


def render_demo(payload: dict) -> str:
    cases = []
    for case in payload["cases"]:
        image_path = DEMO / case["image"]
        if digest(image_path) != case["image_sha256"]:
            raise ValueError(f"Bundled image changed: {case['id']}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        cases.append({**case, "image_data": f"data:image/jpeg;base64,{encoded}"})
    embedded = json.dumps({**payload, "cases": cases}, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DEMO_DATA__", embedded)


def write_demo_page(payload: dict, open_page: bool) -> None:
    output = DEMO / "index.html"
    output.write_text(render_demo(payload), encoding="utf-8", newline="\n")
    print(f"Wrote {output}")
    if open_page:
        webbrowser.open(output.as_uri())


def build_demo(open_page: bool) -> dict:
    manifest = json.loads((DEMO / "samples.json").read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    model_hash = digest(MODEL)
    if metrics["model_sha256"] != model_hash:
        raise ValueError("Archived metrics refer to a different model; re-evaluate before presenting them")
    if manifest["names"] != NAMES:
        raise ValueError("Sample class order does not match the demo")
    for case in manifest["cases"]:
        for kind in ("image", "label"):
            if digest(DEMO / case[kind]) != case[f"{kind}_sha256"]:
                raise ValueError(f"Bundled {kind} changed: {case['id']}")

    model = load_model()
    settings = dict(PREDICTION_SETTINGS)
    start = time.perf_counter()
    results = model.predict(
        source=[str(DEMO / case["image"]) for case in manifest["cases"]],
        batch=4, stream=True, verbose=False, **settings,
    )
    cases = []
    for expected, result in zip(manifest["cases"], results, strict=True):
        if Path(result.path).resolve() != (DEMO / expected["image"]).resolve():
            raise ValueError("Predictions and source examples are out of order")
        truth = load_truth(DEMO / expected["label"])
        if len(truth) != expected["instances"]:
            raise ValueError("Sample instance count disagrees with labels")
        predictions = serialize_predictions(result)
        image_path = DEMO / expected["image"]
        with Image.open(image_path) as source:
            width, height = source.size
        cases.append({**expected, "width": width, "height": height, "truth": truth, "predictions": predictions})

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_sha256": model_hash,
        "ultralytics": ultralytics_version,
        "settings": settings,
        "inference_elapsed_seconds": round(time.perf_counter() - start, 3),
        "selection": manifest["selection"],
        "names": NAMES,
        "chinese_names": CHINESE_NAMES,
        "evaluation": metrics,
        "cases": cases,
    }
    (DEMO / "predictions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_demo_page(payload, open_page)
    print(f"Ran {len(cases)} examples using {MODEL.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-samples", action="store_true", help="Bundle fixed examples from the local held-out dataset")
    mode.add_argument("--render-only", action="store_true", help="Update the page using the existing predictions")
    parser.add_argument("--no-open", action="store_true", help="Regenerate without opening a browser")
    args = parser.parse_args()
    if args.prepare_samples:
        prepare_samples()
    if args.render_only:
        write_demo_page(json.loads((DEMO / "predictions.json").read_text(encoding="utf-8")), open_page=not args.no_open)
    else:
        build_demo(open_page=not args.no_open)


if __name__ == "__main__":
    main()
