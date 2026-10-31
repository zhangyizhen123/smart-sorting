"""UART protocol shared by MaixCAM 2 and the STM32 controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct

MAGIC = b"SS"
VERSION = 1
MAX_PAYLOAD = 128


class MessageType(IntEnum):
    VISION_RESULT = 0x01
    ACK = 0x02
    ACTION_STARTED = 0x03
    DROP_SUCCESS = 0x04
    DROP_FAILED = 0x05
    MOTION_ERROR = 0x06
    BIN_POSITION_ERROR = 0x07
    EMERGENCY_STOP = 0x08
    RESET_ROUND = 0x09


class StatusCode(IntEnum):
    OK = 0
    BAD_CRC = 1
    BAD_LENGTH = 2
    BUSY = 3
    LOW_CONFIDENCE = 4
    NO_OBJECT = 5
    SENSOR_TIMEOUT = 6
    LIMIT_ERROR = 7


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    payload: bytes = b""
    version: int = VERSION


VISION_RESULT = struct.Struct("<BBBBfffH")
"""shape, colour, bin, reserved, x_mm, y_mm, confidence, frame_id."""

STATUS = struct.Struct("<BBH")
"""status code, reserved, operation sequence."""


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE used by both endpoints."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(frame: Frame) -> bytes:
    if frame.version != VERSION:
        raise ValueError(f"unsupported protocol version {frame.version}")
    if len(frame.payload) > MAX_PAYLOAD:
        raise ValueError("payload is too large")
    body = struct.pack("<BBH", frame.version, int(frame.message_type), len(frame.payload)) + frame.payload
    return MAGIC + body + struct.pack("<H", crc16(body))


def decode_frame(raw: bytes) -> Frame:
    if len(raw) < 8 or raw[:2] != MAGIC:
        raise ValueError("invalid frame header")
    version, message_type, length = struct.unpack_from("<BBH", raw, 2)
    expected = 2 + 4 + length + 2
    if length > MAX_PAYLOAD or len(raw) != expected:
        raise ValueError("invalid frame length")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    body = raw[2:-2]
    received_crc = struct.unpack_from("<H", raw, len(raw) - 2)[0]
    if crc16(body) != received_crc:
        raise ValueError("CRC mismatch")
    try:
        message = MessageType(message_type)
    except ValueError as exc:
        raise ValueError(f"unknown message type {message_type}") from exc
    return Frame(message, raw[6:-2], version)


class StreamDecoder:
    """Collect complete frames from arbitrary UART read chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[Frame]:
        self._buffer.extend(chunk)
        frames: list[Frame] = []
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer.endswith(MAGIC[:1]) else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 6:
                break
            length = struct.unpack_from("<H", self._buffer, 4)[0]
            if length > MAX_PAYLOAD:
                del self._buffer[0]
                continue
            frame_length = 8 + length
            if len(self._buffer) < frame_length:
                break
            candidate = bytes(self._buffer[:frame_length])
            try:
                frames.append(decode_frame(candidate))
            except ValueError:
                del self._buffer[0]
            else:
                del self._buffer[:frame_length]
        return frames


def encode_vision_result(shape_id: int, color_id: int, bin_id: int, x_mm: float, y_mm: float, confidence: float, frame_id: int) -> bytes:
    payload = VISION_RESULT.pack(shape_id, color_id, bin_id, 0, x_mm, y_mm, confidence, frame_id)
    return encode_frame(Frame(MessageType.VISION_RESULT, payload))


def decode_vision_result(frame: Frame) -> dict[str, float | int]:
    if frame.message_type != MessageType.VISION_RESULT or len(frame.payload) != VISION_RESULT.size:
        raise ValueError("frame is not a vision result")
    shape, color, bin_id, _reserved, x_mm, y_mm, confidence, frame_id = VISION_RESULT.unpack(frame.payload)
    return {
        "shape_id": shape,
        "color_id": color,
        "bin_id": bin_id,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "confidence": confidence,
        "frame_id": frame_id,
    }


def decode_status(frame: Frame) -> tuple[StatusCode, int]:
    if frame.message_type == MessageType.VISION_RESULT or len(frame.payload) != STATUS.size:
        raise ValueError("frame is not a status message")
    status, reserved, sequence = STATUS.unpack(frame.payload)
    if reserved != 0:
        raise ValueError("status reserved byte must be zero")
    try:
        return StatusCode(status), sequence
    except ValueError as exc:
        raise ValueError(f"unknown status code {status}") from exc


def encode_status(message_type: MessageType, status: StatusCode, sequence: int = 0) -> bytes:
    if message_type == MessageType.VISION_RESULT:
        raise ValueError("status message cannot be a vision result")
    return encode_frame(Frame(message_type, STATUS.pack(int(status), 0, sequence)))
