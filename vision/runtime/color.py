"""Lighting-calibrated Lab prototype classifier for the ten block colours."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

COLOR_NAMES = ("white", "qing", "skyblue", "yellow", "red", "gray", "green", "purple", "black", "orange")


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an sRGB sample to CIE Lab (D65)."""

    def linearize(value: float) -> float:
        value /= 255.0
        return ((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92

    red, green, blue = (linearize(value) for value in rgb)
    x = (red * 0.4124564 + green * 0.3575761 + blue * 0.1804375) / 0.95047
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = (red * 0.0193339 + green * 0.1191920 + blue * 0.9503041) / 1.08883

    def f(value: float) -> float:
        delta = 6 / 29
        return value ** (1 / 3) if value > delta**3 else value / (3 * delta**2) + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


@dataclass(frozen=True)
class ColorResult:
    name: str
    confidence: float
    distance: float
    margin: float


class ColorPrototypeClassifier:
    """Classify masked RGB pixels against measured per-colour Lab prototypes.

    Prototypes must be collected from the actual MaixCAM2, lens, lighting and
    printed blocks. The classifier abstains when distance or class margin is
    outside configured bounds.
    """

    def __init__(self, prototypes: dict[str, tuple[float, float, float]], max_distance: float = 24.0, min_margin: float = 3.0) -> None:
        missing = set(COLOR_NAMES) - set(prototypes)
        extra = set(prototypes) - set(COLOR_NAMES)
        if missing or extra:
            raise ValueError(f"prototype names mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        self.prototypes = {name: tuple(map(float, prototypes[name])) for name in COLOR_NAMES}
        if any(len(value) != 3 or not all(math.isfinite(x) for x in value) for value in self.prototypes.values()):
            raise ValueError("each Lab prototype must contain three finite values")
        self.max_distance = max_distance
        self.min_margin = min_margin

    @classmethod
    def from_json(cls, path: str | Path) -> "ColorPrototypeClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["lab_prototypes"], data.get("max_distance", 24.0), data.get("min_margin", 3.0))

    def classify(self, rgb_samples: Iterable[tuple[float, float, float]]) -> ColorResult | None:
        samples = list(rgb_samples)
        if not samples:
            return None
        lab_samples = [rgb_to_lab(tuple(map(float, sample))) for sample in samples]
        # The component-wise median rejects highlights and small background leaks.
        observed = tuple(sorted(channel)[len(channel) // 2] for channel in zip(*lab_samples))
        ranked = sorted(
            (math.dist(observed, prototype), name)
            for name, prototype in self.prototypes.items()
        )
        distance, name = ranked[0]
        margin = ranked[1][0] - distance
        if distance > self.max_distance or margin < self.min_margin:
            return None
        confidence = max(0.0, min(1.0, (1.0 - distance / self.max_distance) * min(1.0, margin / (2 * self.min_margin))))
        return ColorResult(name, confidence, distance, margin)
