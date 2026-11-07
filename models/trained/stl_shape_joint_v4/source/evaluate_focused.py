#!/usr/bin/env python3
"""Class-blind one-to-one mask matching; misses, duplicates and false alarms fail."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO
import yaml
from shape_refiner import NAMES, FOCUSED_PAIRS, PairwiseRefiner, read_polygons

ROOT = Path(__file__).resolve().parents[2]


def rooted(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masks_from_truth(truth, width, height):
    masks = []
    for _, polygon in truth:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    return masks


def match_instances(ious, threshold=0.5):
    matrix = np.asarray(ious, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape:
        return []
    valid = matrix >= threshold
    cost = -(valid.astype(float) * (min(matrix.shape) + 1) + matrix * valid)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def score_records(records, threshold=0.5):
    matrix = np.zeros((9, 9), dtype=int)
    totals, pred_totals, missed, dup_classes, clean = (Counter() for _ in range(5))
    correct = extra = duplicates = spurious = clean_scenes = 0
    errors = []
    for record in records:
        true = record["truth_classes"]
        pred = [p["class_id"] for p in record["predictions"]]
        totals.update(true); pred_totals.update(pred)
        ious = np.asarray(record["ious"], dtype=float).reshape(len(true), len(pred))
        matched = match_instances(ious, threshold)
        used_gt, used_pred = {i for i, _ in matched}, {j for _, j in matched}
        duplicate_gt = set()
        for j in set(range(len(pred))) - used_pred:
            extra += 1
            if len(true) and ious[:, j].max() >= threshold:
                i = int(ious[:, j].argmax()); duplicate_gt.add(i)
                duplicates += 1; dup_classes[true[i]] += 1; kind = "duplicate"
            else:
                spurious += 1; kind = "false_positive"
            errors.append({"image": record["image"], "reason": kind, "prediction_index": j, "predicted": NAMES[pred[j]]})
        scene_correct = len(true) == len(pred) and len(matched) == len(true)
        for i, j in matched:
            matrix[true[i], pred[j]] += 1
            if true[i] == pred[j]:
                correct += 1
                if i not in duplicate_gt:
                    clean[true[i]] += 1
            else:
                scene_correct = False
                errors.append({"image": record["image"], "reason": "wrong_class", "truth": NAMES[true[i]],
                               "predicted": NAMES[pred[j]], "prediction_index": j, "mask_iou": float(ious[i, j])})
        for i in set(range(len(true))) - used_gt:
            missed[true[i]] += 1
            errors.append({"image": record["image"], "reason": "missed", "truth": NAMES[true[i]]})
        clean_scenes += int(scene_correct)
    total, total_pred = sum(totals.values()), sum(pred_totals.values())
    per_class = {}
    for cls, name in enumerate(NAMES):
        tp = int(matrix[cls, cls]); n, p = totals[cls], pred_totals[cls]
        precision, recall = tp / p if p else 0., tp / n if n else 0.
        per_class[name] = {"instances": n, "predictions": p, "correct": tp, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "missed": missed[cls], "duplicate_detections": dup_classes[cls], "clean_correct": clean[cls],
            "clean_instance_accuracy": clean[cls] / n if n else None}
    report = {"image_count": len(records), "instance_count": total, "prediction_count": total_pred,
        "correct_instances": correct, "correct_over_gt": correct / total if total else None,
        "clean_correct_instances": sum(clean.values()),
        "strict_instance_score": sum(clean.values()) / (total + spurious) if total + spurious else 1.,
        "score_definition": "one correct prediction and no duplicate per GT; denominator=all GT+nonduplicate false positives; misses/wrong/duplicates fail",
        "missed_instances": sum(missed.values()), "duplicate_detections": duplicates, "spurious_detections": spurious,
        "unmatched_predictions": extra, "whole_scene_accuracy": clean_scenes / len(records) if records else None,
        "per_class": per_class, "confusion_rows_true_cols_pred": matrix.tolist(), "focused_pairs": {}, "errors": errors}
    for a, b in FOCUSED_PAIRS:
        entries = {}
        for source, target in ((a, b), (b, a)):
            si, ti = NAMES.index(source), NAMES.index(target)
            entries[source + "_to_" + target] = {"count": int(matrix[si, ti]), "denominator": totals[si], "rate": float(matrix[si, ti] / totals[si]) if totals[si] else None}
        report["focused_pairs"][a + "_vs_" + b] = entries
    return report


def predict_records(model_path, data_path, split, settings, refiner_path=None, include_images=None):
    import ultralytics.utils.torch_utils as torch_utils
    torch_utils.NUM_THREADS = 2
    torch.set_num_threads(2)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = rooted(config.get("path", data_path.parent))
    if (root / "freeze.json").exists():
        from make_focused_dataset import verify_dataset
        verify_dataset(root)
    metadata = {}
    if (root / "manifest.jsonl").exists():
        metadata = {r["image"]: r for r in map(json.loads, (root / "manifest.jsonl").read_text().splitlines())}
    model = YOLO(str(model_path))
    if model.task != "segment" or list(model.names.values()) != NAMES:
        raise ValueError("Expected nine-class segmentation checkpoint")
    refiner = PairwiseRefiner(refiner_path) if refiner_path else None
    if refiner and not refiner.ready:
        raise ValueError("Incomplete refiner artifact")
    split_source = root / config[split]
    if split_source.is_file():
        images = [rooted(line.strip()) for line in split_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        images = sorted(split_source.glob("*.jpg"))
    if include_images is not None:
        images = [p for p in images if p.relative_to(root).as_posix() in include_images]
    if not images:
        raise ValueError("Empty evaluation split")
    records = []
    for index, path in enumerate(images):
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        truth = read_polygons(root / "labels" / split / (path.stem + ".txt"), width, height)
        result = model.predict(source=image, verbose=False, **settings)[0]
        predictions, predicted_masks = [], []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None or len(result.masks) != len(result.boxes):
                raise ValueError("Missing segmentation masks")
            for j, polygon in enumerate(result.masks.xy):
                cls = int(result.boxes.cls[j])
                revised = refiner.refine(np.asarray(image), polygon, cls, float(result.boxes.conf[j])) if refiner else cls
                predictions.append({"class_id": revised, "base_class_id": cls, "confidence": float(result.boxes.conf[j]),
                                    "polygon": np.asarray(polygon).round(3).tolist(), "box": result.boxes.xyxy[j].cpu().tolist()})
                mask = result.masks.data[j].cpu().numpy()
                predicted_masks.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5)
        ious = np.zeros((len(truth), len(predictions)))
        for i, tm in enumerate(masks_from_truth(truth, width, height)):
            for j, pm in enumerate(predicted_masks):
                ious[i, j] = np.count_nonzero(tm & pm) / max(1, np.count_nonzero(tm | pm))
        key = path.relative_to(root).as_posix()
        records.append({"image": key, "width": width, "height": height, "mode": metadata.get(key, {}).get("mode", "legacy"),
                        "truth_classes": [c for c, _ in truth], "truth_polygons": [p.tolist() for _, p in truth],
                        "predictions": predictions, "ious": ious.tolist()})
        if (index + 1) % 100 == 0:
            print(f"{split}: {index + 1}/{len(images)}", flush=True)
    return records, root


def write_visuals(records, report, root, out):
    out.mkdir(parents=True, exist_ok=True)
    mapping = {r["image"]: r for r in records}
    for index, name in enumerate(list(dict.fromkeys(e["image"] for e in report["errors"]))[:24]):
        record = mapping[name]
        with Image.open(root / name) as source:
            im = source.convert("RGB")
        draw = ImageDraw.Draw(im)
        for cls, polygon in zip(record["truth_classes"], record["truth_polygons"]):
            points = [tuple(p) for p in polygon]; draw.line(points + points[:1], fill="lime", width=2)
            draw.text(points[0], "GT " + NAMES[cls], fill="lime", stroke_fill="black", stroke_width=1)
        for j, prediction in enumerate(record["predictions"]):
            points = [tuple(p) for p in prediction["polygon"]]
            if points:
                draw.line(points + points[:1], fill="red", width=1)
                draw.text((points[0][0], points[0][1] + 12), f"P{j} " + NAMES[prediction["class_id"]], fill="red", stroke_fill="white", stroke_width=1)
        im.save(out / f"error_{index:02d}.jpg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    matrix = np.asarray(report["confusion_rows_true_cols_pred"])
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(9), yticks=range(9), xticklabels=NAMES, yticklabels=NAMES, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out / "confusion.png"); plt.close(fig)


def run_evaluation(model_path, data_path, split, output, settings, refiner_path=None):
    torch.set_num_threads(2)
    records, root = predict_records(model_path, data_path, split, settings, refiner_path)
    report = score_records(records)
    subsets = defaultdict(list)
    for record in records:
        subsets[record["mode"]].append(record)
    keys = {"image_count", "instance_count", "strict_instance_score", "missed_instances", "duplicate_detections", "spurious_detections"}
    report["subsets"] = {name: {k: v for k, v in score_records(rows).items() if k in keys} for name, rows in subsets.items()}
    report.update({"model": str(model_path), "model_sha256": digest(model_path), "data": str(data_path), "split": split,
                   "settings": settings, "refiner": str(refiner_path) if refiner_path else None,
                   "freeze_sha256": digest(root / "freeze.json") if (root / "freeze.json").exists() else None,
                   "meets_99_target": report["strict_instance_score"] >= 0.99})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".records.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    write_visuals(records, report, root, output.with_suffix(""))
    print(json.dumps({k: report[k] for k in ("split", "instance_count", "strict_instance_score", "missed_instances", "duplicate_detections", "spurious_detections", "meets_99_target")}), flush=True)
    return report


def mask_ap(records):
    """AP at this pipeline's fixed confidence floor; not official low-floor mAP."""
    from ultralytics.utils.metrics import ap_per_class
    all_tp, all_conf, all_pred, all_truth = [], [], [], []
    thresholds = np.linspace(0.5, 0.95, 10)
    for record in records:
        true = np.asarray(record["truth_classes"], dtype=int)
        pred = np.asarray([p["class_id"] for p in record["predictions"]], dtype=int)
        ious = np.asarray(record["ious"]).reshape(len(true), len(pred))
        tp = np.zeros((len(pred), 10), bool)
        for k, threshold in enumerate(thresholds):
            class_ious = ious * (true[:, None] == pred[None, :])
            for i, j in match_instances(class_ious, threshold):
                tp[j, k] = True
        all_tp.append(tp); all_conf.extend(p["confidence"] for p in record["predictions"])
        all_pred.extend(pred); all_truth.extend(true)
    if not all_pred:
        return {"mAP50": 0., "mAP50_95": 0., "confidence_floor_limited": True}
    result = ap_per_class(np.concatenate(all_tp), np.asarray(all_conf), np.asarray(all_pred), np.asarray(all_truth), plot=False)
    ap = result[5]
    return {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()), "confidence_floor_limited": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/trained/stl_shape_v1/best.pt")
    p.add_argument("--data", default="dataset/stl_synthetic_yolo/data.yaml")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--refiner")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--metrics-out", required=True)
    a = p.parse_args()
    if not 0 < a.conf < 1:
        p.error("conf must be within (0,1)")
    settings = dict(imgsz=a.imgsz, conf=a.conf, iou=0.7, agnostic_nms=a.agnostic_nms, device=a.device, retina_masks=True)
    run_evaluation(rooted(a.model), rooted(a.data), a.split, rooted(a.metrics_out), settings, rooted(a.refiner) if a.refiner else None)


if __name__ == "__main__":
    main()
