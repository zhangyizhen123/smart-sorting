"""Preserve disconnected regions and holes when displaying native binary masks."""
from __future__ import annotations

import cv2
import numpy as np


def native_mask_geometry(mask) -> dict:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not all(binary.shape):
        raise ValueError("Expected a nonempty two-dimensional binary mask")
    binary = np.ascontiguousarray(binary > 0, dtype=np.uint8)
    height, width = binary.shape
    # All rings are independent paths; SVG evenodd preserves holes at any depth.
    # Do not merge disjoint contours with synthetic connecting edges.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rings = [
        [[round(float(x) / width, 8), round(float(y) / height, 8)] for x, y in ring.reshape(-1, 2)]
        for ring in contours
    ]
    components = cv2.connectedComponents(binary, connectivity=8)[0] - 1
    return {"mask_contours": rings, "mask_size": [width, height],
            "mask_components": components, "mask_foreground_pixels": int(binary.sum())}
