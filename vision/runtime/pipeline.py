"""Hardware-neutral bridge from MaixCAM detections to the UART session."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .assignment import Detection, TemporalConsensus
from .color import COLOR_NAMES, ColorPrototypeClassifier
from .session import VisionSession


@dataclass(frozen=True)
class FrameObject:
    """One segmented object emitted by the camera/model adapter."""

    shape_id: int
    center_px: tuple[float, float]
    shape_confidence: float
    rgb_samples: tuple[tuple[float, float, float], ...]
    track_id: str | int
    area_px: float


class VisionPipeline:
    """Convert frame objects, stabilize detections and dispatch one target."""

    def __init__(
        self,
        session: VisionSession,
        color_classifier: ColorPrototypeClassifier,
        consensus: TemporalConsensus | None = None,
    ) -> None:
        self.session = session
        self.color_classifier = color_classifier
        self.consensus = consensus or TemporalConsensus(min_confidence=session.min_confidence)
        self._bootstrap: list[Detection] = []
        self._bootstrap_classes: set[tuple[int, int]] = set()

    @property
    def mapping_ready(self) -> bool:
        return self.session.bin_map.complete

    def process_frame(self, objects: Iterable[FrameObject], frame_id: int) -> bytes | None:
        if self.session.fault:
            raise RuntimeError(f"session is faulted: {self.session.fault}")

        detections = [self._to_detection(obj) for obj in objects]
        detections = [detection for detection in detections if detection is not None]
        stable = self.consensus.add_frame(detections)

        if self.session.pending_class is not None:
            return None
        if not self.mapping_ready:
            self._collect_round_classes(stable)
            if not self.mapping_ready:
                return None

        return self.session.target_frame(stable, frame_id)

    def handle_controller_frame(self, raw_frame: bytes) -> str:
        result = self.session.handle_stm32_frame(raw_frame)
        if result in {"rescan", "retry_after_rescan", "stop"}:
            self.consensus.clear()
        return result

    def reset_round(self) -> None:
        self.session.reset_round()
        self.consensus.clear()
        self._bootstrap.clear()
        self._bootstrap_classes.clear()

    def _to_detection(self, obj: FrameObject) -> Detection | None:
        if not 0 <= obj.shape_id < 9:
            raise ValueError(f"shape id out of range: {obj.shape_id}")
        if obj.track_id is None:
            raise ValueError("each object needs a stable track_id across frames")
        if not math.isfinite(obj.shape_confidence) or not 0.0 <= obj.shape_confidence <= 1.0:
            raise ValueError("shape confidence must be finite and between zero and one")
        if not math.isfinite(obj.area_px) or obj.area_px < 0.0:
            raise ValueError("object area must be finite and nonnegative")

        color = self.color_classifier.classify(obj.rgb_samples)
        if color is None:
            return None
        x_mm, y_mm = self.session.calibration.project(obj.center_px)
        return Detection(
            shape_id=obj.shape_id,
            color_id=COLOR_NAMES.index(color.name),
            x_mm=x_mm,
            y_mm=y_mm,
            confidence=min(obj.shape_confidence, color.confidence),
            track_id=obj.track_id,
            area_px=obj.area_px,
        )

    def _collect_round_classes(self, stable: list[Detection]) -> None:
        for detection in stable:
            if detection.class_key in self._bootstrap_classes:
                continue
            if len(self._bootstrap) >= self.session.max_bins:
                raise ValueError("more stable classes were detected than available bins")
            self._bootstrap_classes.add(detection.class_key)
            self._bootstrap.append(detection)
        if len(self._bootstrap) == self.session.max_bins:
            self.session.initialize_round(self._bootstrap)
