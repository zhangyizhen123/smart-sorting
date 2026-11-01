#!/usr/bin/env python3
"""Reproducible CPU pilot/full training, validation selection, then frozen testing."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import yaml

from evaluate_focused import digest, rooted, predict_records, run_evaluation, score_records, mask_ap
from make_focused_dataset import verify_dataset
from shape_refiner import PairwiseRefiner, read_rgb
from train_shape_refiner import train as train_refiner

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def state(out, stage, **details):
    write_json(out / "status.json", {"stage": stage, "updated_at": datetime.now().astimezone().isoformat(), **details})
    print(stage, flush=True)


def make_training_config(dataset, out, count):
    records = list(map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines()))
    grouped = {}
    for row in records:
        if row["split"] == "train":
            grouped.setdefault((row["target_class"], row["mode"]), []).append(row["image"])
    rng = random.Random(42)
    selected = []
    for key, rows in sorted(grouped.items()):
        rng.shuffle(rows); selected.extend(rows[:count // 10])
    if count % 10 or len(selected) != count * 9:
        raise ValueError("Insufficient training scenes or count not multiple of ten")
    paths = out / "train_images.txt"
    paths.write_text("\n".join((dataset / p).resolve().as_posix() for p in sorted(selected)) + "\n", encoding="utf-8")
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    config["train"] = paths.resolve().as_posix()
    result = out / "data.yaml"
    result.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return result, selected


def load_records(path):
    return list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/focused_v2")
    parser.add_argument("--profile", choices=("cpu_pilot", "full"), default="cpu_pilot")
    parser.add_argument("--out", required=True)
    parser.add_argument("--wait-for-dataset", action="store_true")
    args = parser.parse_args()
    dataset, out = rooted(args.dataset), rooted(args.out)
    profiles = json.loads((ROOT / "vision/configs/focused_training.json").read_text(encoding="utf-8"))
    cfg = profiles[args.profile]
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "summary.json"
    if existing.exists():
        print(existing.read_text(encoding="utf-8")); return
    try:
        state(out, "waiting_for_dataset", dataset=str(dataset), profile=args.profile)
        start = time.monotonic()
        while not (dataset / "freeze.json").exists():
            if not args.wait_for_dataset or time.monotonic() - start > 21600:
                raise FileNotFoundError("Dataset generation incomplete; inspect generation log")
            time.sleep(10)
        frozen = verify_dataset(dataset)
        if any(frozen["summary"]["test"]["per_class"].get(name, 0) < 300 for name in ("cube", "cuboid", "tri_prism", "cone", "cylinder")):
            raise ValueError("Frozen test set needs at least 300 instances per focus class")
        snapshot = {"profile": args.profile, "config": cfg, "dataset_freeze_sha256": digest(dataset / "freeze.json")}
        if (out / "experiment.json").exists():
            if json.loads((out / "experiment.json").read_text(encoding="utf-8")) != snapshot:
                raise ValueError("Changed configuration: use a new output directory")
        else:
            write_json(out / "experiment.json", snapshot)
        data_path, training_images = make_training_config(dataset, out, cfg["train_per_class"])
        checkpoint = out / "segment" / "weights" / "best.pt"
        torch.set_num_threads(cfg["threads"])
        if not (out / "training_complete.json").exists():
            state(out, "training_segmenter", profile=args.profile, training_images=len(training_images), config=cfg, log=str(out / "train.log"))
            device = ("0" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
            command = [sys.executable, str(ROOT / "vision/scripts/train_seg.py"), "--data", str(data_path),
                "--model", cfg["initial_model"], "--project", str(out), "--name", "segment", "--epochs", str(cfg["epochs"]),
                "--imgsz", str(cfg["imgsz"]), "--batch", str(cfg["batch"]), "--patience", str(cfg["patience"]),
                "--lr0", str(cfg["lr0"]), "--threads", str(cfg["threads"]), "--device", device, "--workers", "0", "--cache", "none", "--focused", "--no-amp"]
            last = checkpoint.with_name("last.pt")
            if last.exists():
                command[command.index("--model") + 1] = str(last); command.append("--resume")
            with (out / "train.log").open("a", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not checkpoint.exists():
                raise FileNotFoundError("Training completed without checkpoint")
            write_json(out / "training_complete.json", {"model_sha256": digest(checkpoint)})
        settings = dict(imgsz=cfg["imgsz"], conf=0.25, iou=0.7, device="cpu", retina_masks=True, agnostic_nms=False)
        state(out, "validation_model_selection")
        candidates = []
        baseline_model = ROOT / "models/trained/stl_shape_v1/best.pt"
        for name, model in (("baseline", baseline_model), ("trained_best", checkpoint), ("trained_last", checkpoint.with_name("last.pt"))):
            for agnostic in (False, True):
                candidate_settings = {**settings, "agnostic_nms": agnostic}
                path = out / f"val_{name}_{int(agnostic)}.json"
                if not path.exists():
                    report = run_evaluation(model, data_path, "val", path, candidate_settings)
                else:
                    report = json.loads(path.read_text())
                candidates.append({"model": str(model), "settings": candidate_settings, "report": str(path), "score": report["strict_instance_score"]})
        chosen = max(candidates, key=lambda item: item["score"])
        selected_model = Path(chosen["model"]); settings = chosen["settings"]
        val_records = load_records(Path(chosen["report"]).with_suffix(".records.jsonl"))
        state(out, "training_refiner", selected_detector=chosen)
        train_records_path = out / "train_refiner.records.jsonl"
        if not train_records_path.exists():
            records, _ = predict_records(selected_model, data_path, "train", settings)
            train_records_path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
        refiner_dir = out / "refiner"
        if not (refiner_dir / "metadata.json").exists():
            train_refiner(dataset, train_records_path, Path(chosen["report"]).with_suffix(".records.jsonl"), refiner_dir)
        state(out, "validation_refiner_selection")
        refiner = PairwiseRefiner(refiner_dir)
        best_refiner_score, best_refiner_config = chosen["score"], None
        comparisons = []
        # Cache crop features: validation labels never enter the refiner's predictions.
        from shape_refiner import crop_features
        feature_cache = {}
        for r in val_records:
            image = read_rgb(dataset / r["image"])
            for j, pred in enumerate(r["predictions"]):
                if pred["base_class_id"] in (0, 1, 2, 3, 6):
                    feature_cache[(r["image"], j)] = crop_features(image, np.asarray(pred["polygon"]))
        probability_cache = {key: refiner.probabilities(feature) for key, feature in feature_cache.items()}
        for gate in (0.6, 0.8, 1.):
            for minimum in (0.65, 0.8):
                refiner.gate, refiner.minimum_probability = gate, minimum
                revised = copy.deepcopy(val_records)
                for r in revised:
                    for j, pred in enumerate(r["predictions"]):
                        if (r["image"], j) in probability_cache:
                            pred["class_id"] = refiner.decide(probability_cache[(r["image"], j)], pred["base_class_id"], pred["confidence"])
                report = score_records(revised)
                score = report["strict_instance_score"]
                comparisons.append({"gate": gate, "minimum_probability": minimum, "strict_instance_score": score})
                if score > best_refiner_score:
                    best_refiner_score, best_refiner_config = score, {"max_base_confidence": gate, "minimum_probability": minimum}
        if best_refiner_config:
            refiner.metadata.update(best_refiner_config); write_json(refiner_dir / "metadata.json", refiner.metadata)
        selection = {"detectors": candidates, "selected": chosen, "refiner_candidates": comparisons,
                     "refiner_enabled": best_refiner_config is not None, "refiner_config": best_refiner_config,
                     "validation_score": best_refiner_score, "test_used_for_selection": False}
        write_json(out / "selection.json", selection)
        state(out, "frozen_test", selection=selection)
        baseline_test = out / "test_baseline.json"
        if not baseline_test.exists():
            run_evaluation(baseline_model, data_path, "test", baseline_test, {**settings, "imgsz": 320, "agnostic_nms": False})
        final_test = out / "test_final.json"
        if not final_test.exists():
            report = run_evaluation(selected_model, data_path, "test", final_test, settings, refiner_dir if best_refiner_config else None)
        else:
            report = json.loads(final_test.read_text())
        test_records = load_records(final_test.with_suffix(".records.jsonl"))
        report["fixed_threshold_mask_ap"] = mask_ap(test_records)
        write_json(final_test, report)
        state(out, "standard_mask_evaluation")
        mask_metrics = out / "standard_mask_metrics.json"
        if not mask_metrics.exists():
            from ultralytics import YOLO
            import ultralytics.utils.torch_utils as torch_utils
            torch_utils.NUM_THREADS = 2
            metrics = YOLO(str(selected_model)).val(data=str(data_path), split="test", imgsz=settings["imgsz"], device="cpu", batch=4,
                workers=0, plots=True, project=str(out), name="mask_validation", exist_ok=True, verbose=False)
            write_json(mask_metrics, {"detector_only": True, "model_sha256": digest(selected_model), "mAP50": float(metrics.seg.map50), "mAP50_95": float(metrics.seg.map),
                "note": "Standard detector mask AP; separate from post-refinement fixed-threshold AP and strict instance score"})
        baseline = json.loads(baseline_test.read_text())
        summary = {"profile": args.profile, "experiment": snapshot, "selection": selection,
            "baseline_strict_score": baseline["strict_instance_score"], "final_strict_score": report["strict_instance_score"],
            "meets_99_target": report["meets_99_target"], "test_instances": report["instance_count"],
            "per_class": report["per_class"], "focused_pairs": report["focused_pairs"], "subsets": report["subsets"],
            "duplicates": report["duplicate_detections"], "missed": report["missed_instances"], "spurious": report["spurious_detections"],
            "standard_mask_metrics": json.loads(mask_metrics.read_text()), "completed_at": datetime.now().astimezone().isoformat()}
        bundle = out / "bundle"; bundle.mkdir(exist_ok=True)
        shutil.copy2(selected_model, bundle / "best.pt")
        if best_refiner_config:
            shutil.copytree(refiner_dir, bundle / "refiner", dirs_exist_ok=True)
        write_json(bundle / "pipeline.json", {"version": 1, "model": "best.pt", "model_sha256": digest(bundle / "best.pt"),
            "settings": settings, "refiner": "refiner" if best_refiner_config else None,
            "refiner_sha256": digest(bundle / "refiner/refiner.pt") if best_refiner_config else None,
            "refiner_metadata_sha256": digest(bundle / "refiner/metadata.json") if best_refiner_config else None,
            "evaluation": {"strict_instance_score": report["strict_instance_score"], "instances": report["instance_count"],
                           "synthetic_only": True, "target_achieved": report["meets_99_target"]}})
        write_json(out / "summary.json", summary)
        state(out, "complete", summary=str(out / "summary.json"), target_achieved=report["meets_99_target"])
    except Exception as exc:
        state(out, "failed", error=str(exc)); raise


if __name__ == "__main__":
    main()
