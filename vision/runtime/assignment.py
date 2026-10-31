"""Detection stabilization and per-round dynamic bin assignment."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Detection:
    """One object detected in a MaixCAM 2 frame."""

    shape_id: int
    color_id: int
    x_mm: float
    y_mm: float
    confidence: float
    track_id: str | int | None = None
    area_px: float = 0.0

    @property
    def class_key(self) -> tuple[int, int]:
        return self.shape_id, self.color_id


@dataclass(frozen=True)
class StableDetection(Detection):
    """Detection after temporal consensus and dynamic bin assignment."""

    bin_id: int = 0
    observations: int = 0


class DynamicBinMap:
    """Assign the first six stable shape+colour classes to bins 1..6."""

    def __init__(self, max_bins: int = 6) -> None:
        if max_bins < 1:
            raise ValueError("max_bins must be positive")
        self.max_bins = max_bins
        self._mapping: dict[tuple[int, int], int] = {}

    @property
    def mapping(self) -> dict[tuple[int, int], int]:
        return dict(self._mapping)

    @property
    def complete(self) -> bool:
        return len(self._mapping) == self.max_bins

    def assign(self, class_key: tuple[int, int]) -> int:
        if class_key in self._mapping:
            return self._mapping[class_key]
        if len(self._mapping) >= self.max_bins:
            raise ValueError("the frame contains more classes than available bins")
        bin_id = len(self._mapping) + 1
        self._mapping[class_key] = bin_id
        return bin_id

    def assign_detections(self, detections: Iterable[Detection]) -> list[StableDetection]:
        result = []
        for detection in detections:
            result.append(
                StableDetection(
                    **detection.__dict__,
                    bin_id=self.assign(detection.class_key),
                    observations=1,
                )
            )
        return result

    def reset(self) -> None:
        self._mapping.clear()


class TemporalConsensus:
    """Require repeated class/position observations before sending a target."""

    def __init__(self, required_frames: int = 3, min_confidence: float = 0.65, max_position_error_mm: float = 8.0) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self.max_position_error_mm = max_position_error_mm
        self._observations: dict[str | int, list[Detection]] = defaultdict(list)

    def add_frame(self, detections: Sequence[Detection]) -> list[Detection]:
        current_ids = {self._key(detection, index) for index, detection in enumerate(detections)}
        for key in list(self._observations):
            if key not in current_ids:
                del self._observations[key]
        for index, detection in enumerate(detections):
            key = self._key(detection, index)
            if detection.confidence < self.min_confidence:
                self._observations.pop(key, None)
                continue
            history = self._observations[key]
            history.append(detection)
            if len(history) > self.required_frames:
                del history[:-self.required_frames]
        stable: list[Detection] = []
        for history in self._observations.values():
            if len(history) < self.required_frames:
                continue
            class_counts = Counter(d.class_key for d in history)
            class_key, count = class_counts.most_common(1)[0]
            if count != len(history):
                continue
            mean_x = sum(d.x_mm for d in history) / len(history)
            mean_y = sum(d.y_mm for d in history) / len(history)
            if any(math.hypot(d.x_mm - mean_x, d.y_mm - mean_y) > self.max_position_error_mm for d in history):
                continue
            latest = history[-1]
            stable.append(Detection(*class_key, mean_x, mean_y, min(d.confidence for d in history), latest.track_id, latest.area_px))
        return stable

    @staticmethod
    def _key(detection: Detection, index: int) -> str | int:
        return detection.track_id if detection.track_id is not None else index

    def clear(self) -> None:
        self._observations.clear()


def choose_target(detections: Iterable[StableDetection], claimed: set[tuple[int, int]] | None = None) -> StableDetection | None:
    """Choose the largest, most confident unclaimed target."""

    claimed = claimed or set()
    candidates = [d for d in detections if d.class_key not in claimed]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.area_px, d.confidence))
