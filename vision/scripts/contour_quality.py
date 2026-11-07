"""Conservative, class-agnostic contour simplification and mask diagnostics.

Original masks are never modified. Regularized masks are diagnostic outputs,
not training labels or robot grasp targets.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from mask_display import native_mask_geometry


DEFAULTS = {"epsilon_fraction": .0025, "epsilon_min": .5, "epsilon_max": 2.,
            "minimum_iou": .98, "maximum_area_change": .02}


def binary(mask):
    value = np.asarray(mask)
    if value.ndim != 2 or not all(value.shape):
        raise ValueError("Expected nonempty two-dimensional mask")
    return np.ascontiguousarray(value > 0, dtype=np.uint8)


def centroid(mask):
    m = cv2.moments(mask, binaryImage=True)
    return None if not m['m00'] else np.array([m['m10']/m['m00'], m['m01']/m['m00']])


def topology(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for item in hierarchy[0]:
            depth, parent = 0, int(item[3])
            while parent >= 0:
                depth += 1
                parent = int(hierarchy[0, parent, 3])
            holes += depth % 2
    return int(cv2.connectedComponents(mask, connectivity=8)[0]-1), int(holes)


def self_intersects(points):
    points = np.asarray(points, float).reshape(-1, 2)
    n = len(points)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for i in range(n):
        js = np.array([j for j in range(i+1, n) if j != i+1 and not (i == 0 and j == n-1)])
        if not len(js):
            continue
        a, b = points[i], points[(i+1) % n]
        c, d = points[js], points[(js+1) % n]
        u, v = cross(b-a, c-a), cross(b-a, d-a)
        s, t = cross(d-c, a-c), cross(d-c, b-c)
        overlap = np.all(np.maximum(np.minimum(a,b), np.minimum(c,d)) <=
                         np.minimum(np.maximum(a,b), np.maximum(c,d)), axis=1)
        if np.any((u*v <= 0) & (s*t <= 0) & overlap):
            return True
    return False


def mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 1.


def normalized_rings(rings, width, height):
    return [[[round(float(x)/width, 8), round(float(y)/height, 8)]
             for x, y in ring.reshape(-1, 2)] for ring in rings]


def regularize_masks(masks, settings=None):
    cfg = {**DEFAULTS, **(settings or {})}
    raw = [binary(m) for m in masks]
    if not raw:
        return [], []
    if any(m.shape != raw[0].shape for m in raw):
        raise ValueError("All instance masks must share image coordinates")
    occupancy = np.sum(raw, axis=0, dtype=np.int32)
    output, info = [], []
    for mask in raw:
        h, w = mask.shape
        native = native_mask_geometry(mask)
        rings, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        x, y, bw, bh = cv2.boundingRect(mask)
        eps = float(np.clip(math.hypot(bw, bh)*cfg['epsilon_fraction'], cfg['epsilon_min'], cfg['epsilon_max']))
        record = {**native, 'regularized_contours': native['mask_contours'],
                  'geometry_status': 'fallback', 'geometry_reason': 'empty_mask',
                  'epsilon_pixels': eps, 'raw_vertices': sum(len(r) for r in rings),
                  'regularized_vertices': sum(len(r) for r in rings)}
        candidate = mask.copy()
        if mask.any():
            simplified = [cv2.approxPolyDP(r, eps, True) for r in rings]
            reason = None
            if any(len(r) < 3 for r in simplified):
                reason = 'tiny_or_degenerate_ring'
            elif any(self_intersects(r) for r in simplified):
                reason = 'self_intersection'
            else:
                candidate = np.zeros_like(mask)
                cv2.drawContours(candidate, simplified, -1, 1, cv2.FILLED)
                overlap = mask_iou(mask, candidate)
                area_change = abs(int(candidate.sum())-int(mask.sum()))/int(mask.sum())
                center = centroid(candidate)
                shift = float(np.linalg.norm(center-centroid(mask))) if center is not None else float('inf')
                record.update(candidate_iou=overlap, candidate_area_change=area_change,
                              candidate_center_shift_pixels=shift if np.isfinite(shift) else None)
                if topology(candidate) != topology(mask): reason = 'topology_changed'
                elif overlap < cfg['minimum_iou']: reason = 'iou_guard'
                elif area_change > cfg['maximum_area_change']: reason = 'area_guard'
                elif shift > eps: reason = 'center_guard'
                elif np.any((candidate > mask) & ((occupancy-mask) > 0)): reason = 'new_overlap_with_raw_neighbor'
            if reason:
                candidate = mask.copy()
                record['geometry_reason'] = reason
            else:
                record.update(geometry_status='accepted', geometry_reason='within_conservative_limits',
                              regularized_contours=normalized_rings(simplified,w,h),
                              regularized_vertices=sum(len(r) for r in simplified))
        output.append(candidate)
        info.append(record)
    # Two independently accepted contours could both expand into a previously
    # empty pixel. Revert both symmetrically rather than depend on instance order.
    reject = set()
    for i in range(len(raw)):
        for j in range(i+1,len(raw)):
            if np.any((output[i] & output[j]) & ~(raw[i] & raw[j])):
                reject.update([i,j])
    for i in reject:
        output[i] = raw[i].copy()
        info[i].update(geometry_status='fallback', geometry_reason='new_pairwise_overlap',
                       regularized_contours=info[i]['mask_contours'], regularized_vertices=info[i]['raw_vertices'])
    for a, b, record in zip(raw, output, info):
        record['effective_iou_to_raw'] = mask_iou(a,b)
        record['changed_pixels'] = int(np.count_nonzero(a != b))
    return output, info


def boundary_band(mask, radius):
    """Inner boundary band; padding accounts for objects touching image edges."""
    m = binary(mask)
    padded = np.pad(m, 1)
    eroded = cv2.erode(padded, np.ones((3,3),np.uint8), iterations=radius)[1:-1,1:-1]
    return m-eroded


def pairwise_geometry(truth_masks, pred_masks, width, height):
    diagonal = math.hypot(width,height)
    radius = max(1, int(round(.005*diagonal)))
    truths, predictions = [binary(m) for m in truth_masks], [binary(m) for m in pred_masks]
    tb = [boundary_band(m,radius) for m in truths]
    pb = [boundary_band(m,radius) for m in predictions]
    ious, boundaries, centers = [], [], []
    for i, tm in enumerate(truths):
        ir, br, cr = [], [], []
        for j, pm in enumerate(predictions):
            ir.append(mask_iou(tm,pm))
            br.append(mask_iou(tb[i],pb[j]))
            a,b=centroid(tm),centroid(pm)
            cr.append(float(np.linalg.norm(a-b)) if a is not None and b is not None else None)
        ious.append(ir);boundaries.append(br);centers.append(cr)
    return {'ious':ious,'boundary_ious':boundaries,'center_errors_pixels':centers,
            'boundary_radius_pixels':radius,'image_diagonal':diagonal}
