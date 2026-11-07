"""Experimental nine-shape/background verifier using appearance and geometry."""
from __future__ import annotations
import cv2
import numpy as np
import torch
from torch import nn

SIZE = 32
BACKGROUND = 9
FEATURES = SIZE*SIZE*4 + 16


def features(image, mask):
    image = np.asarray(image, dtype=np.uint8)
    mask = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
    if image.shape[:2] != mask.shape:
        raise ValueError("Image/mask coordinate mismatch")
    if not mask.any():
        return np.zeros(FEATURES, np.float32)
    x, y, w, h = cv2.boundingRect(mask)
    pad = max(3, round(max(w, h)*.15))
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(mask.shape[1], x+w+pad), min(mask.shape[0], y+h+pad)
    crop = image[y0:y1, x0:x1]
    binary = mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    edges = cv2.Canny(np.rint(gray*255).astype(np.uint8), 30, 90).astype(np.float32)/255.
    # Unmasked context/edges avoid fabricating an edge by zeroing background.
    channels = [gray, gray*binary, binary.astype(np.float32), edges]
    reduced = [cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA).ravel() for c in channels]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(binary.sum())
    perimeter = cv2.arcLength(contour, True)
    hull = max(1., cv2.contourArea(cv2.convexHull(contour)))
    hu = cv2.HuMoments(cv2.moments(binary)).ravel()
    hu = -np.sign(hu)*np.log10(np.maximum(np.abs(hu), 1e-12))/12.
    ring = cv2.dilate(binary, np.ones((5, 5), np.uint8))-binary
    inside = gray[binary > 0]
    outside = gray[ring > 0]
    boundary = binary-cv2.erode(binary, np.ones((3, 3), np.uint8))
    near_edges = cv2.dilate((edges > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    extra = np.asarray([
        np.log(max(w, 1)/max(h, 1)), area/max(w*h, 1), area/hull,
        4*np.pi*area/max(perimeter*perimeter, 1),
        float(inside.std()), float(outside.std()) if outside.size else 0.,
        float(inside.mean()-outside.mean()) if outside.size else 0.,
        float((boundary*near_edges).sum()/max(boundary.sum(), 1)),
        len(cv2.approxPolyDP(contour, max(.5, .02*perimeter), True))/12.,
        *hu,
    ], np.float32)
    result = np.concatenate([*reduced, extra]).astype(np.float32)
    if result.shape != (FEATURES,) or not np.isfinite(result).all():
        raise ValueError("Invalid verifier features")
    return result


class VerifierNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURES, 256), nn.ReLU(), nn.Dropout(.15),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 10))

    def forward(self, x):
        return self.layers(x)


class ShapeVerifier:
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["version"] != 1:
            raise ValueError("Unsupported verifier")
        self.mean, self.scale = payload["mean"], payload["scale"]
        self.model = VerifierNetwork()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def probabilities(self, values):
        x = torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1, FEATURES)
        x = ((x-self.mean)/self.scale).clamp(-6, 6)
        with torch.inference_mode():
            return torch.softmax(self.model(x), -1).numpy()
