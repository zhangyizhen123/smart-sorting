"""Audit RGB-guided refinement on development data, never the sealed test."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from contour_quality import pairwise_geometry
from evaluate_focused import masks_from_truth, match_instances
from evaluate_replay_optimization import ROOT, digest, summarize, write_json
from image_guided_masks import DEFAULTS, refine_masks
from shape_refiner import read_polygons


def select_rows(rows, full=False):
    eligible = [r for r in rows if r["split"] == "val" or
                (r["domain"] == "real" and r.get("repetition") == 0)]
    if full:
        return eligible
    groups = defaultdict(list)
    for row in eligible:
        key = (row["domain"], row.get("target_class", ""), row.get("mode", ""))
        groups[key].append(row)
    selected = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest())
        limit = {"focused": 1, "legacy": 36, "negative": 6, "real": 46}[key[0]]
        selected.extend(group[:limit])
    return selected


def audit(source_name, full=False):
    run = ROOT / "runs/segment/replay_optimization_v3"
    dataset = ROOT / "dataset/replay_optimization_v3"
    source = run / "evaluation" / source_name
    if json.loads((source / "status.json").read_text(encoding="utf-8"))["stage"] != "complete":
        raise ValueError("Source evaluation must finish first")
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = select_rows(rows, full)
    mode = "full" if full else "probe"
    out = run / "diagnostics/image_guided_v1" / (source_name + "_" + mode)
    freeze = {
        "scope": "development_only", "full_validation": full,
        "source_summary_sha256": digest(source / "summary.json"),
        "dataset_manifest_sha256": digest(dataset / "manifest.jsonl"),
        "algorithm_sha256": digest(ROOT / "vision/scripts/image_guided_masks.py"),
        "runner_sha256": digest(Path(__file__)), "settings": DEFAULTS,
        "selection": [r["key"] for r in rows], "no_classes_or_truth_passed_to_algorithm": True,
    }
    if (out / "freeze.json").exists():
        if json.loads((out / "freeze.json").read_text(encoding="utf-8")) != freeze:
            raise ValueError("Probe inputs changed; use a new experiment directory")
    else:
        write_json(out / "freeze.json", freeze)
    originals, records = [], []
    for index, row in enumerate(rows):
        before = json.loads((source / "records" / (row["key"] + ".json")).read_text(encoding="utf-8"))
        target = out / "records" / (row["key"] + ".json")
        if target.exists():
            after = json.loads(target.read_text(encoding="utf-8"))
            for asset, sha in after["mask_hashes"].items():
                if digest(out / asset) != sha:
                    raise ValueError("Cached refined mask changed")
        else:
            for kind in ["image", "label"]:
                if digest(dataset / row[kind]) != row[kind + "_sha256"]:
                    raise ValueError("Development asset changed")
            image = np.asarray(Image.open(dataset / row["image"]).convert("RGB"))
            masks = []
            for j in range(len(before["predictions"])):
                asset = f"masks/{row['key']}/{j+1:02d}.png"
                if digest(source / asset) != before["mask_hashes"][asset]:
                    raise ValueError("Native prediction mask changed")
                with Image.open(source / asset) as saved:
                    masks.append((np.asarray(saved) > 0).astype(np.uint8))
            tick = time.perf_counter()
            refined, details = refine_masks(image, masks)
            elapsed = time.perf_counter() - tick
            after = copy.deepcopy(before)
            after["mask_hashes"] = {}
            for j, mask in enumerate(refined):
                path = out / "masks" / row["key"] / f"{j+1:02d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask*255).save(path)
                after["mask_hashes"][path.relative_to(out).as_posix()] = digest(path)
            # Ground truth is loaded only after prediction/refinement, for scoring.
            truth = read_polygons(dataset / row["label"], before["width"], before["height"])
            truth_masks = masks_from_truth(truth, before["width"], before["height"])
            after.update(pairwise_geometry(truth_masks, refined, before["width"], before["height"]))
            after.update(refinement=details, refinement_seconds=elapsed)
            after["inference_seconds"] += elapsed
            write_json(target, after)
        originals.append(before)
        records.append(after)
        if (index+1) % 20 == 0 or index+1 == len(rows):
            status = {"stage": "evaluating", "source": source_name, "images": index+1, "total": len(rows)}
            write_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
    groups = {}
    for domain in ["synthetic_validation", "legacy", "focused", "negative", "real"]:
        accept = lambda r: r["domain"] != "real" if domain == "synthetic_validation" else r["domain"] == domain
        old, new = [r for r in originals if accept(r)], [r for r in records if accept(r)]
        paired = []
        for a, b in zip(old, new, strict=True):
            n, p = len(a["truth_classes"]), len(a["predictions"])
            ma = dict(match_instances(np.asarray(a["ious"]).reshape(n, p), .5))
            mb = dict(match_instances(np.asarray(b["ious"]).reshape(n, p), .5))
            for i in ma.keys() & mb.keys():
                paired.append([b["boundary_ious"][i][mb[i]]-a["boundary_ious"][i][ma[i]],
                               b["ious"][i][mb[i]]-a["ious"][i][ma[i]]])
        groups[domain] = {"before": summarize(old), "after": summarize(new),
                          "common_matched_instances": len(paired),
                          "paired_mean_boundary_delta": float(np.mean(np.asarray(paired)[:, 0])) if paired else None,
                          "paired_mean_mask_iou_delta": float(np.mean(np.asarray(paired)[:, 1])) if paired else None}
    report = {"scope": "development_diagnostic_only", "source": source_name, "images": len(rows),
              "full_validation": full, "settings": DEFAULTS, "groups": groups,
              "outcomes": dict(Counter(v["reason"] for r in records for v in r["refinement"])),
              "independent_real_test_images": 0, "deployed": False}
    write_json(out / "summary.json", report)
    write_json(out / "status.json", {"stage": "complete", "images": len(rows)})
    print(json.dumps({"source": source_name, "complete": True,
                     "deltas": {k: v["paired_mean_boundary_delta"] for k, v in groups.items()}}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["v1_deployed", "v2_deployed", "pilot_640"])
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    audit(args.source, args.full)
