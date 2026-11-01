#!/usr/bin/env python3
"""Small pairwise classifiers for the shape pairs that are easy to confuse."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn


NAMES = [
    "cylinder", "cube", "cuboid", "tri_prism", "pent_prism",
    "hex_prism", "cone", "tetrahedron", "sphere",
]
FOCUSED_PAIRS = (
    ("cylinder", "cone"),
    ("tri_prism", "cuboid"),
    ("cube", "cuboid"),
)
FEATURE_SIZE = 32


def read_rgb(path: Path) -> np.ndarray:
    """Read Windows paths containing Chinese characters without OpenCV path issues."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def polygon_mask(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def crop_features(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Extract geometry and masked appearance without retaining background pixels."""
    height, width = image.shape[:2]
    points = np.asarray(polygon, dtype=np.float32)
    if points.size < 6 or not np.isfinite(points).all():
        return np.zeros(FEATURE_SIZE * FEATURE_SIZE * 3 + 8, np.float32)
    mask = polygon_mask(points, width, height)
    x, y, w, h = cv2.boundingRect(np.rint(points).astype(np.int32))
    if w < 2 or h < 2:
        return np.zeros(32 * 32 * 3 + 8, dtype=np.float32)
    pad = max(2, int(round(max(w, h) * 0.08)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray[crop_mask == 0] = 0
    small_gray = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small_mask = cv2.resize(crop_mask, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 35, 110)
    edges[crop_mask == 0] = 0
    small_edges = cv2.resize(edges, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    contour = points.reshape(-1, 1, 2).astype(np.float32)
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, abs(float(cv2.contourArea(hull))))
    rect_area = max(1.0, float(w * h))
    moments = cv2.moments(contour)
    mu20 = moments["mu20"] / max(1.0, moments["m00"])
    mu02 = moments["mu02"] / max(1.0, moments["m00"])
    metrics = np.asarray([
        w / max(1.0, h),
        h / max(1.0, w),
        area / rect_area,
        area / hull_area,
        4.0 * np.pi * area / max(1.0, perimeter * perimeter),
        len(cv2.approxPolyDP(contour, max(1.0, perimeter * 0.02), True)),
        np.sqrt(max(0.0, mu20) / max(1e-6, mu02)),
        np.sqrt(max(0.0, mu02) / max(1e-6, mu20)),
    ], dtype=np.float32)
    return np.concatenate([small_gray.ravel(), small_mask.ravel(), small_edges.ravel(), metrics])


def read_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    result = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        result.append((int(fields[0]), values * np.asarray([width, height], dtype=np.float32)))
    return result


def save_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PairwiseRefiner:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        payload = torch.load(self.directory / "refiner.pt", map_location="cpu", weights_only=True)
        self.mean, self.scale = payload["mean"], payload["scale"]
        self.model = RefinerNetwork(len(self.mean))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.gate = float(self.metadata.get("max_base_confidence", 0.8))
        self.minimum_probability = float(self.metadata.get("minimum_probability", 0.65))

    @property
    def ready(self) -> bool:
        return self.metadata.get("version") == 2

    def probabilities(self, features):
        x = torch.from_numpy(features.astype(np.float32).reshape(-1, len(self.mean)))
        x = ((x - self.mean) / self.scale).clamp(-6, 6)
        with torch.inference_mode():
            return [torch.softmax(output, -1).numpy() for output in self.model(x)]

    def refine(self, image: np.ndarray, polygon: np.ndarray, predicted_class: int, confidence: float = 0.) -> int:
        if predicted_class not in (0, 6, 1, 2, 3) or confidence >= self.gate:
            return predicted_class
        probabilities = self.probabilities(crop_features(image, polygon))
        return self.decide(probabilities, predicted_class, confidence)

    def decide(self, probabilities, predicted_class, confidence):
        if predicted_class not in (0, 6, 1, 2, 3) or confidence >= self.gate:
            return predicted_class
        if predicted_class in (0, 6):
            choices, p = (0, 6), probabilities[0][0]
        else:
            choices = (1, 2, 3)
            # Two overlapping binary pairs cannot arbitrate cube vs prism alone.
            # A separately supervised three-way head resolves that ambiguity.
            p = probabilities[3][0].copy()
            p *= np.array([probabilities[2][0, 0], np.sqrt(probabilities[1][0, 1] * probabilities[2][0, 1]), probabilities[1][0, 0]])
            p /= max(1e-8, p.sum())
        winner = int(p.argmax())
        return choices[winner] if p[winner] >= self.minimum_probability else predicted_class


class RefinerNetwork(nn.Module):
    def __init__(self, feature_count=3080):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_count, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, count) for count in (2, 2, 2, 3)])

    def forward(self, x):
        encoded = self.encoder(x)
        return [head(encoded) for head in self.heads]
