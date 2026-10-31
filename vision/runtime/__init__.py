"""Runtime building blocks for the MaixCAM 2 vision controller."""

from .assignment import DynamicBinMap, Detection, StableDetection
from .calibration import Homography
from .color import ColorPrototypeClassifier
from .pipeline import FrameObject, VisionPipeline
from .protocol import (
    Frame,
    MessageType,
    StatusCode,
    StreamDecoder,
    decode_frame,
    decode_status,
    encode_frame,
)

__all__ = [
    "Detection",
    "DynamicBinMap",
    "ColorPrototypeClassifier",
    "FrameObject",
    "Frame",
    "Homography",
    "MessageType",
    "StableDetection",
    "StatusCode",
    "StreamDecoder",
    "VisionPipeline",
    "decode_frame",
    "decode_status",
    "encode_frame",
]
