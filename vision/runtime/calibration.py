"""Planar calibration for a fixed oblique camera.

The camera is fixed to the frame and the objects rest on the tray plane. Four
visible tray markers are enough to map image pixels to tray millimetres. The
implementation intentionally uses only the Python standard library so it can
run in a small MaixCAM 2 application without pulling in OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

Point = tuple[float, float]


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with Gaussian elimination and pivoting."""

    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("calibration points are degenerate")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[col])
                ]
    return [augmented[row][-1] for row in range(n)]


def _homography_from_points(source: Sequence[Point], target: Sequence[Point]) -> tuple[float, ...]:
    if len(source) != 4 or len(target) != 4:
        raise ValueError("exactly four source and target points are required")
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        vector.append(v)
    return tuple(_solve_linear(matrix, vector)) + (1.0,)


@dataclass(frozen=True)
class Homography:
    """Map image pixels to the calibrated tray plane in millimetres."""

    matrix: tuple[float, ...]
    source_points: tuple[Point, ...] = ()
    target_points: tuple[Point, ...] = ()

    @classmethod
    def from_points(cls, image_points: Iterable[Point], tray_points: Iterable[Point]) -> "Homography":
        source = tuple((float(x), float(y)) for x, y in image_points)
        target = tuple((float(x), float(y)) for x, y in tray_points)
        return cls(_homography_from_points(source, target), source, target)

    @classmethod
    def from_json(cls, path: str | Path) -> "Homography":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = tuple(float(x) for x in data["matrix"])
        if len(matrix) != 9:
            raise ValueError("homography matrix must contain nine numbers")
        source = tuple(tuple(map(float, p)) for p in data.get("image_points", ()))
        target = tuple(tuple(map(float, p)) for p in data.get("tray_points_mm", ()))
        return cls(matrix, source, target)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "matrix": list(self.matrix),
            "image_points": [list(p) for p in self.source_points],
            "tray_points_mm": [list(p) for p in self.target_points],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def project(self, pixel: Point) -> Point:
        x, y = pixel
        m = self.matrix
        denominator = m[6] * x + m[7] * y + m[8]
        if abs(denominator) < 1e-12:
            raise ValueError("pixel projects to the homography horizon")
        return (
            (m[0] * x + m[1] * y + m[2]) / denominator,
            (m[3] * x + m[4] * y + m[5]) / denominator,
        )

    def project_many(self, pixels: Iterable[Point]) -> list[Point]:
        return [self.project(pixel) for pixel in pixels]


def calibrate_from_json(input_path: str | Path, output_path: str | Path) -> Homography:
    """Build a calibration file from ``image_points`` and ``tray_points_mm``."""

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = Homography.from_points(data["image_points"], data["tray_points_mm"])
    result.to_json(output_path)
    return result
