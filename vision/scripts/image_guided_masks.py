"""Experimental RGB-guided native-mask refinement, without class/GT inputs.

This module is not used by the deployed demo. Candidates must be evaluated
against annotations; passing the change guards alone is not proof of accuracy.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from contour_quality import binary, centroid, mask_iou, topology

DEFAULTS = {
    "band_fraction": .015, "band_min": 2, "band_max": 12,
    "iterations": 3, "minimum_iou": .90, "maximum_area_change": .08,
    "maximum_center_shift_fraction": .5,
}


def refine_masks(image, masks, settings=None):
    """Use fixed interior/exterior seeds and optimize only a boundary band."""
    cfg = {**DEFAULTS, **(settings or {})}
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image")
    raw = [binary(mask) for mask in masks]
    if any(mask.shape != rgb.shape[:2] for mask in raw):
        raise ValueError("Image and masks must share native coordinates")
    if not raw:
        return [], []
    occupancy = np.sum(raw, axis=0, dtype=np.int32)
    output, records = [], []
    height, width = rgb.shape[:2]
    for original in raw:
        x, y, w, h = cv2.boundingRect(original)
        band = int(np.clip(round(math.hypot(w, h) * cfg["band_fraction"]),
                           cfg["band_min"], cfg["band_max"]))
        record = {"status": "fallback", "reason": "empty_mask", "band_pixels": band}
        candidate = original.copy()
        if original.any():
            pad = band * 2 + 2
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1, y1 = min(width, x+w+pad), min(height, y+h+pad)
            patch = original[y0:y1, x0:x1]
            other = (occupancy-original)[y0:y1, x0:x1] > 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band+1, 2*band+1))
            inner = cv2.erode(patch, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            outer = cv2.dilate(patch, kernel)
            seeds = np.full(patch.shape, cv2.GC_BGD, dtype=np.uint8)
            seeds[outer > 0] = cv2.GC_PR_BGD
            seeds[patch > 0] = cv2.GC_PR_FGD
            seeds[inner > 0] = cv2.GC_FGD
            seeds[other & (patch == 0)] = cv2.GC_BGD
            if np.count_nonzero(seeds == cv2.GC_FGD) < 5:
                record["reason"] = "insufficient_interior_seed"
            elif np.count_nonzero(seeds == cv2.GC_BGD) < 5:
                record["reason"] = "insufficient_exterior_seed"
            else:
                pixels = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                cv2.setRNGSeed(20261101)
                cv2.grabCut(pixels, seeds, None, np.zeros((1, 65)), np.zeros((1, 65)),
                            int(cfg["iterations"]), cv2.GC_INIT_WITH_MASK)
                refined = np.isin(seeds, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
                candidate[y0:y1, x0:x1] = refined
                overlap = mask_iou(original, candidate)
                area_change = abs(int(candidate.sum())-int(original.sum())) / int(original.sum())
                center = centroid(candidate)
                shift = float(np.linalg.norm(center-centroid(original))) if center is not None else None
                record.update(candidate_iou=overlap, candidate_area_change=area_change,
                              candidate_center_shift_pixels=shift)
                reason = None
                if topology(candidate) != topology(original):
                    reason = "topology_changed"
                elif overlap < cfg["minimum_iou"]:
                    reason = "iou_guard"
                elif area_change > cfg["maximum_area_change"]:
                    reason = "area_guard"
                elif shift is None or shift > band*cfg["maximum_center_shift_fraction"]:
                    reason = "center_guard"
                elif np.any((candidate > original) & ((occupancy-original) > 0)):
                    reason = "new_overlap_with_neighbor"
                if reason:
                    candidate = original.copy()
                    record["reason"] = reason
                else:
                    record.update(status="accepted", reason="within_change_guards")
        output.append(candidate)
        records.append(record)
    reject = set()
    for i in range(len(raw)):
        for j in range(i+1, len(raw)):
            if np.any((output[i] & output[j]) & ~(raw[i] & raw[j])):
                reject.update((i, j))
    for i in reject:
        output[i] = raw[i].copy()
        records[i].update(status="fallback", reason="new_pairwise_overlap")
    for before, after, record in zip(raw, output, records, strict=True):
        record["effective_iou_to_raw"] = mask_iou(before, after)
        record["changed_pixels"] = int(np.count_nonzero(before != after))
    return output, records
