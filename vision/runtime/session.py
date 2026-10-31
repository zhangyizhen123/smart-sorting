"""Round-level coordination between vision, tray coordinates and UART."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .assignment import Detection, DynamicBinMap, StableDetection, choose_target
from .calibration import Homography
from .protocol import MessageType, StatusCode, decode_frame, decode_status, encode_vision_result


@dataclass(frozen=True)
class RoundMapEntry:
    shape_id: int
    color_id: int
    bin_id: int


class VisionSession:
    """Own category mapping and one-at-a-time retry/rescan decisions."""

    def __init__(self, calibration: Homography, max_bins: int = 6, min_confidence: float = 0.65) -> None:
        self.calibration = calibration
        self.max_bins = max_bins
        self.min_confidence = min_confidence
        self.bin_map = DynamicBinMap(max_bins)
        self.frame_id = 0
        self.success_count = 0
        self.operation_sequence = 0
        self.controller_ready = False
        self.pending_class: tuple[int, int] | None = None
        self.retry_class: tuple[int, int] | None = None
        self.rescan_required = True
        self.fault: str | None = None

    @property
    def mapping(self) -> tuple[RoundMapEntry, ...]:
        return tuple(
            RoundMapEntry(shape_id, color_id, bin_id)
            for (shape_id, color_id), bin_id in self.bin_map.mapping.items()
        )

    def initialize_round(self, stable_detections: Iterable[Detection]) -> tuple[RoundMapEntry, ...]:
        detections = list(stable_detections)
        categories = list(dict.fromkeys(d.class_key for d in detections))
        if len(categories) != self.max_bins:
            raise ValueError(f"expected exactly {self.max_bins} stable classes, got {len(categories)}")
        self.bin_map.reset()
        for category in categories:
            self.bin_map.assign(category)
        self.pending_class = None
        self.retry_class = None
        self.success_count = 0
        self.rescan_required = False
        self.fault = None
        return self.mapping

    def target_frame(self, detections: Iterable[Detection], frame_id: int) -> bytes | None:
        if not self.bin_map.complete:
            raise RuntimeError("round mapping has not been initialized")
        if self.pending_class is not None:
            raise RuntimeError("wait for the pending drop result before sending another target")
        if self.fault:
            raise RuntimeError(f"session is faulted: {self.fault}")

        stable = []
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            bin_id = self.bin_map.mapping.get(detection.class_key)
            if bin_id is None:
                continue
            stable.append(
                StableDetection(
                    **detection.__dict__,
                    bin_id=bin_id,
                    observations=1,
                )
            )
        retry_candidates = [d for d in stable if d.class_key == self.retry_class] if self.retry_class else []
        # Prefer a failed category if it is still visible. If it has left the
        # tray, continue with the freshly recognized objects instead of
        # waiting forever on a stale category.
        target = choose_target(retry_candidates or stable)
        self.frame_id = frame_id & 0xFFFF
        if target is None:
            return None
        if not self.controller_ready:
            self.rescan_required = True
            return None
        self.operation_sequence = (self.operation_sequence + 1) & 0xFFFF
        self.rescan_required = False
        self.pending_class = target.class_key
        confidence_percent = max(0.0, min(1.0, target.confidence))
        return encode_vision_result(
            target.shape_id,
            target.color_id,
            target.bin_id,
            target.x_mm,
            target.y_mm,
            confidence_percent,
            self.frame_id,
        )

    def handle_stm32_frame(self, raw_frame: bytes) -> str:
        frame = decode_frame(raw_frame)
        status, sequence = decode_status(frame)
        if frame.message_type == MessageType.ACK:
            if not self.controller_ready and status == StatusCode.OK and sequence == 0:
                self.controller_ready = True
                return "ready"
            if status in {StatusCode.BUSY, StatusCode.BAD_CRC, StatusCode.BAD_LENGTH,
                          StatusCode.LOW_CONFIDENCE, StatusCode.NO_OBJECT}:
                self.operation_sequence = sequence
                self.pending_class = None
                self.rescan_required = True
                return "retry_after_rescan"
            if status == StatusCode.LIMIT_ERROR:
                self.pending_class = None
                self.controller_ready = False
                self.fault = "controller_limit_error"
                return "stop"
            if sequence != self.operation_sequence:
                raise RuntimeError(f"unexpected ACK sequence {sequence}; expected {self.operation_sequence}")
            return "wait"
        if frame.message_type == MessageType.DROP_SUCCESS:
            if self.pending_class is None:
                raise RuntimeError("drop success arrived without an active target")
            if sequence != self.operation_sequence or status != StatusCode.OK:
                raise RuntimeError("drop success status does not match the active operation")
            self.success_count += 1
            self.pending_class = None
            self.retry_class = None
            self.rescan_required = True
            return "rescan"
        if frame.message_type == MessageType.DROP_FAILED:
            if self.pending_class is None:
                raise RuntimeError("drop failure arrived without an active target")
            if sequence != self.operation_sequence or status != StatusCode.SENSOR_TIMEOUT:
                raise RuntimeError("drop failure status does not match the active operation")
            # No retry cap: rescan, prefer the same class if it is still present,
            # and otherwise continue with the current tray contents.
            self.retry_class = self.pending_class
            self.pending_class = None
            self.rescan_required = True
            return "retry_after_rescan"
        if frame.message_type in {MessageType.MOTION_ERROR, MessageType.BIN_POSITION_ERROR, MessageType.EMERGENCY_STOP}:
            if sequence != self.operation_sequence:
                raise RuntimeError(f"unexpected fault sequence {sequence}; expected {self.operation_sequence}")
            self.fault = frame.message_type.name.lower()
            self.pending_class = None
            self.controller_ready = False
            return "stop"
        if frame.message_type == MessageType.ACTION_STARTED:
            if sequence != self.operation_sequence or status != StatusCode.OK:
                raise RuntimeError("action status does not match the active operation")
            return "wait"
        raise ValueError(f"unexpected STM32 message: {frame.message_type.name}")

    def reset_round(self) -> None:
        self.bin_map.reset()
        self.pending_class = None
        self.retry_class = None
        self.success_count = 0
        self.rescan_required = True
        self.fault = None
