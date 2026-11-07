"""One config-driven entry for a vision experiment (D-36).

A JSON config names frozen evaluation suites, candidates (a checkpoint, a released pipeline.json of any
version, or a checkpoint produced by this run's training step), optional training, selection, paired
comparisons and gates. The run:

1. verifies every suite (manifest hash and every image/label hash) and refuses leakage between the
   training manifest and any evaluation suite;
2. records an experiment identity (config, suites, code hashes); a changed config or suite under the same
   output directory is refused, so one directory always means one experiment. The config is hashed by its
   parsed content, so a CRLF/LF checkout or reformatting does not count as a change; files this tool
   writes always use LF so their hashes match on Windows and Linux;
3. optionally trains through vision/scripts/train_seg.py and freezes the checkpoint hashes;
4. evaluates each candidate on each suite with the existing strict one-to-one scoring
   (evaluate_focused.score_records) and contour metrics (contour_quality / run_contour_audit), caching
   per-image records so an interrupted run resumes;
5. writes report.json and a Chinese report.md with per-suite tables, per-group breakdowns, empty-image
   alarms, paired bootstrap intervals and gate results.

Sealed test suites are one-shot: they need "selection_closed": true, can never drive selection, and a
ledger in runs/experiments/sealed_ledger.json refuses a second experiment on the same frozen suite.

Usage:
    python vision/scripts/run_experiment.py run --config vision/configs/experiments/<name>.json
    python vision/scripts/run_experiment.py freeze-suite --images DIR --labels DIR --out dataset/<suite>
        --split val --group session_2026_11_20
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

from contour_quality import pairwise_geometry
from evaluate_focused import digest, masks_from_truth, score_records
from run_contour_audit import geometry_summary
from shape_refiner import NAMES, read_polygons

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
LEDGER = "runs/experiments/sealed_ledger.json"
ROLES = {"selection", "regression", "diagnostic", "sealed_test"}
TOP_KEYS = {"name", "purpose", "output", "seed", "threads", "thresholds", "suites", "train", "candidates",
            "selection", "comparisons", "gates", "gates_status", "selection_closed"}
SUITE_KEYS = {"dataset", "split", "role", "group_fields", "independent"}
TRAIN_KEYS = {"data", "manifest", "initial_model", "epochs", "imgsz", "batch", "device", "threads", "workers",
              "lr0", "patience", "warmup_epochs", "cache", "seed", "focused", "no_amp"}
METRICS = {"strict_instance_score": True, "false_positives_per_image": False, "empty_image_alarm_rate": False,
           "missed_instances": False, "boundary_iou_with_unmatched_zero": True, "whole_scene_accuracy": True}
CODE = ["run_experiment.py", "evaluate_focused.py", "contour_quality.py", "run_contour_audit.py", "shape_refiner.py"]


# ----------------------------------------------------------------------------- helpers


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 with LF on every OS, so hashes of files this tool writes match across machines."""
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temp, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def inside(root: Path, value: str) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the project: {value}")
    return path


def canonical(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(data) -> str:
    """Hash of the parsed content: unchanged by CRLF/LF checkouts, indentation or key order."""
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- config


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("name", "output", "suites", "candidates"):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    if not cfg["suites"] or not cfg["candidates"]:
        raise ValueError("Need at least one suite and one candidate")
    for name, suite in cfg["suites"].items():
        if set(suite) - SUITE_KEYS or "dataset" not in suite:
            raise ValueError(f"Suite {name}: unknown keys or missing dataset")
        if suite.get("role", "diagnostic") not in ROLES:
            raise ValueError(f"Suite {name}: role must be one of {sorted(ROLES)}")
        if suite.get("role") == "sealed_test" and not cfg.get("selection_closed"):
            raise ValueError(f"Suite {name} is sealed_test; set selection_closed only after the candidate is locked")
    if "train" in cfg:
        if set(cfg["train"]) - TRAIN_KEYS:
            raise ValueError(f"Unknown train keys: {sorted(set(cfg['train']) - TRAIN_KEYS)}")
        for key in ("data", "manifest", "initial_model", "epochs", "imgsz"):
            if key not in cfg["train"]:
                raise ValueError(f"train.{key} is required")
    for name, spec in cfg["candidates"].items():
        kinds = [k for k in ("model", "pipeline", "trained") if k in spec]
        if len(kinds) != 1 or set(spec) - {"model", "pipeline", "trained", "settings"}:
            raise ValueError(f"Candidate {name}: give exactly one of model / pipeline / trained (+ settings)")
        if kinds[0] in {"model", "trained"} and "settings" not in spec:
            raise ValueError(f"Candidate {name}: settings are required for a checkpoint")
        if kinds[0] == "trained" and ("train" not in cfg or spec["trained"] not in {"best", "last"}):
            raise ValueError(f"Candidate {name}: trained needs a train section and best/last")
    selection = cfg.get("selection")
    if selection:
        if cfg["suites"].get(selection["suite"], {}).get("role") != "selection":
            raise ValueError("Selection must use a suite whose role is 'selection'")
        if selection.get("metric", "strict_instance_score") not in METRICS:
            raise ValueError("Unknown selection metric")
    for item in cfg.get("comparisons", []):
        for key in ("candidate", "baseline"):
            if item[key] not in cfg["candidates"]:
                raise ValueError(f"Comparison refers to unknown candidate {item[key]}")
        if item["suite"] not in cfg["suites"]:
            raise ValueError(f"Comparison refers to unknown suite {item['suite']}")
    for gate in cfg.get("gates", []):
        if gate["metric"] not in METRICS or gate.get("rule", "greater") not in {"greater", "non_inferior"}:
            raise ValueError(f"Gate {gate.get('name')}: unknown metric or rule")
        if gate["suite"] not in cfg["suites"]:
            raise ValueError(f"Gate {gate.get('name')}: unknown suite")
        for key in ("candidate", "baseline"):
            if gate[key] not in cfg["candidates"]:
                raise ValueError(f"Gate {gate.get('name')}: unknown {key}")
        if gate.get("use_ci") and gate["metric"] != "strict_instance_score":
            raise ValueError("Confidence-interval gates are only defined for strict_instance_score")


# ----------------------------------------------------------------------------- suites


def load_suite(root: Path, spec: dict) -> tuple[Path, dict, list[dict]]:
    """Verify a frozen suite: manifest hash plus every used image/label hash."""
    directory = inside(root, spec["dataset"])
    lock = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    manifest = directory / "manifest.jsonl"
    if digest(manifest) != lock["manifest_sha256"]:
        raise ValueError(f"Suite manifest changed: {spec['dataset']}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if spec.get("split"):
        rows = [r for r in rows if r.get("split") == spec["split"]]
    if not rows:
        raise ValueError(f"Suite has no rows: {spec['dataset']}")
    for row in rows:
        for key in ("image", "label"):
            if digest(inside(directory, row[key])) != row[key + "_sha256"]:
                raise ValueError(f"Suite {key} changed: {row[key]}")
    return directory, lock, rows


def freeze_suite(images: Path, labels: Path, out: Path, split: str, group: str, extra: dict | None = None) -> dict:
    """Copy images and YOLO-seg labels into a new frozen suite with per-file hashes."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty suite; use a new version")
    sources = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not sources:
        raise ValueError("No images found")
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    rows, seen = [], set()
    for source in sources:
        label = labels / (source.stem + ".txt")
        if not label.exists():
            raise ValueError(f"Missing label for {source.name}; an empty file marks an empty scene")
        image_hash = digest(source)
        if image_hash in seen:
            raise ValueError(f"Duplicate image content: {source.name}")
        seen.add(image_hash)
        image_dest = out / "images" / split / source.name
        label_dest = out / "labels" / split / label.name
        shutil.copy2(source, image_dest)
        shutil.copy2(label, label_dest)
        classes = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(not 0 <= c < len(NAMES) for c in classes):
            raise ValueError(f"Unknown class id in {label.name}")
        rows.append({"key": source.stem, "split": split, "group_id": group,
                     "image": image_dest.relative_to(out).as_posix(), "label": label_dest.relative_to(out).as_posix(),
                     "image_sha256": image_hash, "label_sha256": digest(label_dest), "classes": classes,
                     "source_image": source.as_posix(), **(extra or {})})
    manifest = out / "manifest.jsonl"
    write_text_lf(manifest, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    lock = {"version": 1, "manifest_sha256": digest(manifest), "images": len(rows), "split": split,
            "group_id": group, "instances": sum(len(r["classes"]) for r in rows), "created_at": now()}
    write_json(out / "freeze.json", lock)
    return lock


def leakage_check(root: Path, train_manifest: Path, suites: dict, specs: dict) -> dict:
    rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = {r["image_sha256"] for r in rows}
    groups = {r["group_id"] for r in rows if r.get("group_id")}
    result = {}
    for name, (_directory, _lock, suite_rows) in suites.items():
        shared_images = images & {r["image_sha256"] for r in suite_rows}
        if shared_images:
            raise ValueError(f"Training data shares {len(shared_images)} image(s) with suite {name}")
        shared_groups = groups & {r["group_id"] for r in suite_rows if r.get("group_id")}
        if shared_groups and specs[name].get("independent"):
            raise ValueError(f"Training data shares group(s) {sorted(shared_groups)[:3]} with independent suite {name}")
        result[name] = {"shared_images": 0, "shared_groups": len(shared_groups)}
    return result


# ----------------------------------------------------------------------------- predictors


class YoloPredictor:
    def __init__(self, model_path: Path, settings: dict, refiner_path: Path | None = None):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        if self.model.task != "segment" or list(self.model.names.values()) != NAMES:
            raise ValueError("Expected a nine-class segmentation checkpoint")
        self.settings = dict(settings)
        self.refiner = None
        if refiner_path is not None:
            from shape_refiner import PairwiseRefiner
            self.refiner = PairwiseRefiner(refiner_path)
            if not self.refiner.ready:
                raise ValueError("Incomplete refiner")

    def predict(self, image: Image.Image):
        result = self.model.predict(source=image, verbose=False, save=False, **self.settings)[0]
        predictions, masks = [], []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None or len(result.masks) != len(result.boxes):
                raise ValueError("Detector returned boxes without masks")
            polygons = result.masks.xy
            for j, mask in enumerate(result.masks.data.cpu().numpy()):
                base, confidence = int(result.boxes.cls[j]), float(result.boxes.conf[j])
                cls = self.refiner.refine(np.asarray(image), polygons[j], base, confidence) if self.refiner else base
                predictions.append({"class_id": int(cls), "confidence": confidence})
                if mask.shape != (image.height, image.width):
                    mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
                masks.append(mask > 0.5)
        return predictions, masks


class PipelinePredictor:
    """Any released pipeline.json; versions 2-5 use the exact runtime the demo serves."""

    def __init__(self, path: Path):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        version = manifest.get("version", 1)
        self.version = version
        if version == 1:
            model = (path.parent / manifest["model"]).resolve()
            if digest(model) != manifest["model_sha256"]:
                raise ValueError("Pipeline model hash mismatch")
            refiner = (path.parent / manifest["refiner"]).resolve() if manifest.get("refiner") else None
            self.inner = YoloPredictor(model, manifest["settings"], refiner)
            self.native = None
            return
        modules = {2: ("pipeline_v2", "NativeShapePipeline"), 3: ("pipeline_v3", "RecipePipeline"),
                   4: ("pipeline_v4", "SpecialistPipeline"), 5: ("pipeline_v5", "ProtectedPipeline")}
        if version not in modules:
            raise ValueError(f"Unsupported pipeline version {version}")
        module = __import__(modules[version][0])
        self.native = getattr(module, modules[version][1])(path)

    def predict(self, image: Image.Image):
        if self.native is None:
            return self.inner.predict(image)
        predictions, masks, _native, _settings = self.native.predict_native(image)
        return ([{"class_id": int(p["class_id"]), "confidence": float(p["confidence"])} for p in predictions],
                [np.asarray(m) > 0 for m in masks])


def candidate_identity(root: Path, spec: dict, out: Path) -> dict:
    if "pipeline" in spec:
        path = inside(root, spec["pipeline"])
        return {"kind": "pipeline", "path": spec["pipeline"], "pipeline_sha256": digest(path)}
    path = inside(root, spec["model"]) if "model" in spec else out / "train" / "weights" / f"{spec['trained']}.pt"
    return {"kind": "trained" if "trained" in spec else "model",
            "path": spec.get("model") or f"train/weights/{spec['trained']}.pt",
            "model_sha256": digest(path), "settings": spec["settings"]}


def default_predictor_factory(root: Path, spec: dict, out: Path, threads: int):
    import torch
    import ultralytics.utils.torch_utils as torch_utils
    torch.set_num_threads(threads)
    torch_utils.NUM_THREADS = threads
    if "pipeline" in spec:
        return PipelinePredictor(inside(root, spec["pipeline"]))
    path = inside(root, spec["model"]) if "model" in spec else out / "train" / "weights" / f"{spec['trained']}.pt"
    return YoloPredictor(path, spec["settings"])


# ----------------------------------------------------------------------------- evaluation


def evaluate_suite(predictor, suite_dir: Path, rows: list[dict], cache: Path, identity: dict,
                   group_fields: list[str]) -> list[dict]:
    identity_path = cache / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Cached evaluation identity changed in {cache}; use a new experiment name")
    else:
        write_json(identity_path, identity)
    records = []
    for row in rows:
        key = row.get("key") or Path(row["image"]).stem
        path = cache / "records" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["image_sha256"] != row["image_sha256"] or record["label_sha256"] != row["label_sha256"]:
                raise ValueError(f"Cached record input changed: {key}")
            records.append(record)
            continue
        with Image.open(suite_dir / row["image"]) as source:
            image = source.convert("RGB")
        truth = read_polygons(suite_dir / row["label"], image.width, image.height)
        tick = time.perf_counter()
        predictions, masks = predictor.predict(image)
        elapsed = time.perf_counter() - tick
        if len(predictions) != len(masks):
            raise ValueError("Predictor returned mismatched predictions and masks")
        truth_masks = masks_from_truth(truth, image.width, image.height)
        record = {"key": key, "image": row["image"], "image_sha256": row["image_sha256"],
                  "label_sha256": row["label_sha256"], "width": image.width, "height": image.height,
                  "groups": {field: row.get(field) for field in group_fields},
                  "truth_classes": [c for c, _ in truth], "predictions": predictions,
                  "inference_seconds": elapsed,
                  **pairwise_geometry(truth_masks, [m.astype(np.uint8) for m in masks], image.width, image.height)}
        write_json(path, record)
        records.append(record)
    return records


def per_image_strict(record: dict, threshold: float = 0.5) -> tuple[int, int]:
    score = score_records([record], threshold)
    return score["clean_correct_instances"], score["instance_count"] + score["spurious_detections"]


def summarize_records(records: list[dict], thresholds: list[float]) -> dict:
    scores = {}
    for threshold in thresholds:
        score = score_records(records, threshold)
        score.pop("errors", None)
        scores[str(threshold)] = score
    main = scores[str(thresholds[0])]
    empty = [r for r in records if not r["truth_classes"]]
    seconds = [r["inference_seconds"] for r in records]
    geometry = geometry_summary(records)
    return {
        "images": len(records), "instances": main["instance_count"],
        "strict_instance_score": main["strict_instance_score"],
        "missed_instances": main["missed_instances"], "duplicate_detections": main["duplicate_detections"],
        "spurious_detections": main["spurious_detections"],
        "false_positives_per_image": main["spurious_detections"] / len(records) if records else None,
        "whole_scene_accuracy": main["whole_scene_accuracy"],
        "empty_images": len(empty),
        "empty_images_with_detections": sum(bool(r["predictions"]) for r in empty),
        "empty_image_alarm_rate": (sum(bool(r["predictions"]) for r in empty) / len(empty)) if empty else None,
        "boundary_iou_with_unmatched_zero": geometry["boundary_iou_with_unmatched_zero"],
        "geometry": geometry,
        "per_class": {name: {k: v[k] for k in ("instances", "clean_correct", "clean_instance_accuracy", "missed")}
                      for name, v in main["per_class"].items() if v["instances"]},
        "confusion_rows_true_cols_pred": main["confusion_rows_true_cols_pred"],
        "scores_by_threshold": {t: {k: s[k] for k in ("strict_instance_score", "missed_instances",
                                                     "duplicate_detections", "spurious_detections")}
                                for t, s in scores.items()},
        "median_inference_seconds": float(np.median(seconds)) if seconds else None,
    }


def summarize_suite(records: list[dict], group_fields: list[str], thresholds: list[float]) -> dict:
    summary = summarize_records(records, thresholds)
    groups = {}
    for field in group_fields:
        buckets = defaultdict(list)
        for record in records:
            buckets[str(record["groups"].get(field))].append(record)
        groups[field] = {value: {k: v for k, v in summarize_records(items, thresholds).items()
                                 if k in ("images", "instances", "strict_instance_score", "missed_instances",
                                          "spurious_detections", "empty_image_alarm_rate")}
                         for value, items in sorted(buckets.items())}
    summary["groups"] = groups
    return summary


def paired_bootstrap(records_a: list[dict], records_b: list[dict], unit: str, resamples: int, alpha: float,
                     seed: int) -> dict:
    """Strict-score difference A-B with a percentile interval, resampling whole units (image or group)."""
    by_key_b = {r["key"]: r for r in records_b}
    if set(by_key_b) != {r["key"] for r in records_a}:
        raise ValueError("Paired comparison needs both candidates on the same images")
    units = defaultdict(lambda: np.zeros(4))
    for record in records_a:
        unit_key = record["key"] if unit == "image" else str(record["groups"].get(unit))
        ca, da = per_image_strict(record)
        cb, db = per_image_strict(by_key_b[record["key"]])
        units[unit_key] += (ca, da, cb, db)
    table = np.stack(list(units.values()))
    delta = lambda t: (t[:, 0].sum() / t[:, 1].sum() if t[:, 1].sum() else 1.0) - \
                      (t[:, 2].sum() / t[:, 3].sum() if t[:, 3].sum() else 1.0)
    rng = np.random.default_rng(seed)
    draws = np.array([delta(table[rng.integers(0, len(table), len(table))]) for _ in range(resamples)])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"unit": unit, "units": len(table), "resamples": resamples, "alpha": alpha,
            "delta": float(delta(table)), "ci_low": float(low), "ci_high": float(high)}


def evaluate_gates(cfg: dict, summaries: dict, comparisons: list[dict]) -> list[dict]:
    results = []
    for gate in cfg.get("gates", []):
        higher = gate.get("higher_is_better", METRICS[gate["metric"]])
        cand = summaries[gate["candidate"]][gate["suite"]][gate["metric"]]
        base = summaries[gate["baseline"]][gate["suite"]][gate["metric"]]
        margin = float(gate.get("margin", 0.0))
        rule = gate.get("rule", "greater")
        result = {"name": gate.get("name"), "suite": gate["suite"], "candidate": gate["candidate"],
                  "baseline": gate["baseline"], "metric": gate["metric"], "rule": rule, "margin": margin,
                  "candidate_value": cand, "baseline_value": base}
        if cand is None or base is None:
            result.update(passed=False, reason="metric undefined on this suite")
            results.append(result)
            continue
        delta = (cand - base) if higher else (base - cand)   # positive = candidate better
        if gate.get("use_ci"):
            match = next((c for c in comparisons if c["candidate"] == gate["candidate"]
                          and c["baseline"] == gate["baseline"] and c["suite"] == gate["suite"]), None)
            if match is None:
                raise ValueError(f"Gate {gate.get('name')} needs a matching comparison for its interval")
            delta = match["ci_low"]
            result["evidence"] = "bootstrap lower bound"
        result["improvement"] = delta
        result["passed"] = bool(delta > margin) if rule == "greater" else bool(delta >= -margin)
        results.append(result)
    return results


def select_candidate(cfg: dict, summaries: dict) -> dict | None:
    selection = cfg.get("selection")
    if not selection:
        return None
    metric = selection.get("metric", "strict_instance_score")
    higher = METRICS[metric]
    pool = selection.get("candidates") or list(cfg["candidates"])
    values = {name: summaries[name][selection["suite"]][metric] for name in pool}
    ranked = sorted((n for n in pool if values[n] is not None), key=lambda n: values[n], reverse=higher)
    return {"suite": selection["suite"], "metric": metric, "values": values, "ranking": ranked,
            "selected": ranked[0] if ranked else None, "used_sealed_test": False}


# ----------------------------------------------------------------------------- training


def build_train_command(root: Path, train: dict, out: Path, seed: int) -> list[str]:
    command = [sys.executable, str(SCRIPTS / "train_seg.py"), "--data", str(inside(root, train["data"])),
               "--model", str(inside(root, train["initial_model"]) if (root / train["initial_model"]).exists()
                              else train["initial_model"]),
               "--project", str(out), "--name", "train", "--epochs", str(train["epochs"]),
               "--imgsz", str(train["imgsz"]), "--batch", str(train.get("batch", 8)),
               "--device", str(train.get("device", "cpu")), "--threads", str(train.get("threads", 2)),
               "--workers", str(train.get("workers", 0)), "--cache", train.get("cache", "none"),
               "--seed", str(train.get("seed", seed)), "--lr0", str(train.get("lr0", 0.001)),
               "--patience", str(train.get("patience", 30)), "--warmup-epochs", str(train.get("warmup_epochs", 3.0))]
    if train.get("focused", True):
        command.append("--focused")
    if train.get("no_amp", True):
        command.append("--no-amp")
    return command


def run_training(root: Path, train: dict, out: Path, seed: int, runner=None) -> dict:
    marker = out / "train_complete.json"
    if marker.exists():
        done = json.loads(marker.read_text(encoding="utf-8"))
        for kind in ("best", "last"):
            if digest(out / "train" / "weights" / f"{kind}.pt") != done[f"{kind}_sha256"]:
                raise ValueError("Trained checkpoint changed after training")
        return done
    if (out / "train").exists():
        raise FileExistsError(f"Partial training directory exists: {out / 'train'}; inspect before resuming")
    command = build_train_command(root, train, out, seed)
    started = now()
    if runner is None:
        with (out / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        runner(command, out)
    done = {"command": command[1:], "started_at": started, "finished_at": now()}
    for kind in ("best", "last"):
        path = out / "train" / "weights" / f"{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint {path}")
        done[f"{kind}_sha256"] = digest(path)
    write_json(marker, done)
    return done


# ----------------------------------------------------------------------------- sealed ledger


def claim_sealed(root: Path, experiment: str, suite_name: str, manifest_sha: str) -> None:
    ledger_path = root / LEDGER
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    entry = ledger.get(manifest_sha)
    if entry and entry["experiment"] != experiment:
        raise ValueError(f"Sealed suite {suite_name} was already used by experiment {entry['experiment']}; "
                         "a sealed test is valid once")
    if not entry:
        ledger[manifest_sha] = {"experiment": experiment, "suite": suite_name, "first_exposed_at": now()}
        write_json(ledger_path, ledger)


# ----------------------------------------------------------------------------- run


def run(config_path: Path, root: Path = ROOT, predictor_factory=None, train_runner=None, quiet: bool = False) -> dict:
    root = root.resolve()
    cfg = load_config(config_path)
    out = inside(root, cfg["output"])
    threads = int(cfg.get("threads", 2))
    thresholds = cfg.get("thresholds", [0.5])
    seed = int(cfg.get("seed", 42))
    suites = {name: load_suite(root, spec) for name, spec in cfg["suites"].items()}
    leakage = None
    if "train" in cfg:
        train_manifest = inside(root, cfg["train"]["manifest"])
        leakage = leakage_check(root, train_manifest, suites, cfg["suites"])
    experiment = {
        "name": cfg["name"], "config": cfg, "config_sha256": content_sha256(cfg),
        "code_sha256": {name: digest(SCRIPTS / name) for name in CODE if (SCRIPTS / name).exists()},
        "suites": {name: {"dataset": cfg["suites"][name]["dataset"], "manifest_sha256": lock["manifest_sha256"],
                          "rows": len(rows), "role": cfg["suites"][name].get("role", "diagnostic")}
                   for name, (_d, lock, rows) in suites.items()},
        "train_manifest_sha256": digest(inside(root, cfg["train"]["manifest"])) if "train" in cfg else None,
        "leakage_check": leakage,
    }
    identity_path = out / "experiment.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if canonical({k: previous.get(k) for k in experiment}) != canonical(experiment):
            raise ValueError("Experiment config, suites or code changed; use a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, {**experiment, "created_at": now()})
    training = run_training(root, cfg["train"], out, seed, train_runner) if "train" in cfg else None
    for name, spec in cfg["suites"].items():
        if spec.get("role") == "sealed_test":
            claim_sealed(root, cfg["name"], name, suites[name][1]["manifest_sha256"])
    factory = predictor_factory or (lambda spec: default_predictor_factory(root, spec, out, threads))
    records, summaries, identities = {}, {}, {}
    for cand_name, spec in cfg["candidates"].items():
        identities[cand_name] = candidate_identity(root, spec, out)
        predictor = factory(spec)
        records[cand_name], summaries[cand_name] = {}, {}
        for suite_name, (directory, lock, rows) in suites.items():
            fields = cfg["suites"][suite_name].get("group_fields", [])
            identity = {"candidate": identities[cand_name], "suite_manifest_sha256": lock["manifest_sha256"],
                        "group_fields": fields}
            cache = out / "cache" / cand_name / suite_name
            records[cand_name][suite_name] = evaluate_suite(predictor, directory, rows, cache, identity, fields)
            summaries[cand_name][suite_name] = summarize_suite(records[cand_name][suite_name], fields, thresholds)
            if not quiet:
                print(json.dumps({"candidate": cand_name, "suite": suite_name,
                                  "strict": summaries[cand_name][suite_name]["strict_instance_score"]}), flush=True)
    comparisons = []
    for item in cfg.get("comparisons", []):
        result = paired_bootstrap(records[item["candidate"]][item["suite"]], records[item["baseline"]][item["suite"]],
                                  item.get("unit", "image"), int(item.get("resamples", 2000)),
                                  float(item.get("alpha", 0.05)), seed)
        comparisons.append({"candidate": item["candidate"], "baseline": item["baseline"], "suite": item["suite"],
                            **result})
    gates = evaluate_gates(cfg, summaries, comparisons)
    report = {
        "name": cfg["name"], "purpose": cfg.get("purpose"), "generated_at": now(),
        "experiment_sha256": digest(identity_path), "training": training, "candidates": identities,
        "suites": experiment["suites"], "summaries": summaries, "comparisons": comparisons,
        "selection": select_candidate(cfg, summaries), "gates": gates,
        "gates_status": cfg.get("gates_status", "none" if not gates else "draft"),
        "all_gates_passed": all(g["passed"] for g in gates) if gates else None,
        "sealed_suites": [n for n, s in cfg["suites"].items() if s.get("role") == "sealed_test"],
        "models_deployed": False,
    }
    write_json(out / "report.json", report)
    write_text_lf(out / "report.md", render_markdown(report))
    return report


# ----------------------------------------------------------------------------- report


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _num(value, digits=4) -> str:
    return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


ROLE_NAMES = {"selection": "选型", "regression": "回归", "diagnostic": "诊断", "sealed_test": "封存测试"}


def render_markdown(report: dict) -> str:
    lines = [f"# 实验报告：{report['name']}", ""]
    if report.get("purpose"):
        lines += [report["purpose"], ""]
    lines += [f"生成时间：{report['generated_at']}。实验身份 SHA-256：`{report['experiment_sha256']}`。"
              "本报告不更换默认模型；封存测试只能使用一次。", ""]
    for suite, info in report["suites"].items():
        lines += [f"## {suite}（{ROLE_NAMES.get(info['role'], info['role'])}，{info['rows']} 张）", "",
                  "| 候选 | 严格实例得分 | 漏检 | 重复 | 额外误报 | 每图误报 | 空图报警率 | 边界 IoU（未匹配记零） | 中位耗时 s |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for cand, by_suite in report["summaries"].items():
            s = by_suite[suite]
            lines.append(f"| {cand} | {_pct(s['strict_instance_score'])} | {s['missed_instances']} | "
                         f"{s['duplicate_detections']} | {s['spurious_detections']} | {_num(s['false_positives_per_image'])} | "
                         f"{_pct(s['empty_image_alarm_rate'])} | {_num(s['boundary_iou_with_unmatched_zero'])} | "
                         f"{_num(s['median_inference_seconds'], 3)} |")
        lines.append("")
        for cand, by_suite in report["summaries"].items():
            for field, values in by_suite[suite]["groups"].items():
                lines += [f"{cand} 按 `{field}` 分组：", "", "| 取值 | 图数 | 实例 | 严格得分 | 额外误报 |",
                          "| --- | ---: | ---: | ---: | ---: |"]
                lines += [f"| {v} | {g['images']} | {g['instances']} | {_pct(g['strict_instance_score'])} | "
                          f"{g['spurious_detections']} |" for v, g in values.items()]
                lines.append("")
    if report["comparisons"]:
        lines += ["## 配对比较（严格得分差，候选减基线）", "", "| 候选 | 基线 | 数据 | 抽样单位 | 差值 | 区间 |",
                  "| --- | --- | --- | --- | ---: | --- |"]
        lines += [f"| {c['candidate']} | {c['baseline']} | {c['suite']} | {c['unit']}（{c['units']}） | "
                  f"{c['delta'] * 100:+.2f} 点 | [{c['ci_low'] * 100:+.2f}, {c['ci_high'] * 100:+.2f}]（{(1 - c['alpha']) * 100:.0f}%） |"
                  for c in report["comparisons"]]
        lines.append("")
    if report["gates"]:
        status = "草案，待 R-13 确认" if report["gates_status"] == "draft" else report["gates_status"]
        lines += [f"## 门槛（{status}）", "", "| 门槛 | 候选 | 基线 | 指标 | 规则 | 改善量 | 结果 |",
                  "| --- | --- | --- | --- | --- | ---: | --- |"]
        lines += [f"| {g['name']} | {g['candidate']} | {g['baseline']} | {g['metric']}@{g['suite']} | "
                  f"{g['rule']}（容差 {g['margin']}） | {_num(g.get('improvement'))} | {'通过' if g['passed'] else '未通过'} |"
                  for g in report["gates"]]
        lines.append("")
    if report.get("selection"):
        sel = report["selection"]
        lines += [f"## 选型", "", f"按 {sel['suite']} 的 {sel['metric']} 排序：{' > '.join(sel['ranking'])}。"
                  f"选中：{sel['selected']}。选型未使用封存测试。", ""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    suite = commands.add_parser("freeze-suite")
    suite.add_argument("--images", type=Path, required=True)
    suite.add_argument("--labels", type=Path, required=True)
    suite.add_argument("--out", type=Path, required=True)
    suite.add_argument("--split", default="val")
    suite.add_argument("--group", required=True, help="leakage group, e.g. the capture session id")
    args = parser.parse_args()
    if args.command == "run":
        report = run(args.config)
        output = inside(ROOT, load_config(args.config)["output"])
        print(json.dumps({"report": (output / "report.md").as_posix(), "selection": report["selection"],
                          "all_gates_passed": report["all_gates_passed"]}, ensure_ascii=False))
    else:
        print(json.dumps(freeze_suite(args.images, args.labels, args.out, args.split, args.group), ensure_ascii=False))


if __name__ == "__main__":
    main()
