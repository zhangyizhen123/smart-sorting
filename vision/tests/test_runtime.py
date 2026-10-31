from __future__ import annotations

import struct
import unittest

from vision.runtime.assignment import Detection, DynamicBinMap, TemporalConsensus
from vision.runtime.calibration import Homography
from vision.runtime.color import COLOR_NAMES, ColorPrototypeClassifier, rgb_to_lab
from vision.runtime.protocol import (
    Frame,
    MessageType,
    StatusCode,
    StreamDecoder,
    VERSION,
    decode_frame,
    decode_status,
    decode_vision_result,
    encode_status,
    encode_vision_result,
    crc16,
)
from vision.runtime.pipeline import FrameObject, VisionPipeline
from vision.runtime.session import VisionSession


class CalibrationTests(unittest.TestCase):
    def test_maps_oblique_image_corners_to_tray_plane(self) -> None:
        image = [(100, 100), (900, 150), (800, 850), (200, 800)]
        tray = [(0, 0), (160, 0), (160, 160), (0, 160)]
        transform = Homography.from_points(image, tray)
        for source, expected in zip(image, tray):
            actual = transform.project(source)
            self.assertAlmostEqual(actual[0], expected[0], places=5)
            self.assertAlmostEqual(actual[1], expected[1], places=5)

    def test_rejects_degenerate_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "degenerate"):
            Homography.from_points([(0, 0)] * 4, [(0, 0), (1, 0), (1, 1), (0, 1)])


class ProtocolTests(unittest.TestCase):
    def test_crc_known_vector_and_round_trip(self) -> None:
        self.assertEqual(crc16(b"123456789"), 0x29B1)
        raw = encode_vision_result(2, 4, 3, 31.25, 88.5, 0.91, 65535)
        frame = decode_frame(raw)
        fields = decode_vision_result(frame)
        self.assertEqual(frame.message_type, MessageType.VISION_RESULT)
        self.assertEqual(fields["bin_id"], 3)
        self.assertEqual(fields["frame_id"], 65535)
        self.assertAlmostEqual(fields["x_mm"], 31.25)

    def test_rejects_bad_crc_and_length(self) -> None:
        raw = bytearray(encode_status(MessageType.DROP_FAILED, StatusCode.SENSOR_TIMEOUT, 9))
        raw[-1] ^= 0x80
        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_frame(bytes(raw))
        with self.assertRaisesRegex(ValueError, "length"):
            decode_frame(bytes(raw[:-1]))

    def test_rejects_unsupported_version(self) -> None:
        raw = bytearray(encode_status(MessageType.ACK, StatusCode.OK, 1))
        raw[2] = VERSION + 1
        raw[-2:] = struct.pack("<H", crc16(raw[2:-2]))
        with self.assertRaisesRegex(ValueError, "unsupported protocol version"):
            decode_frame(bytes(raw))

    def test_decodes_status_payload_and_sequence(self) -> None:
        frame = decode_frame(encode_status(MessageType.ACK, StatusCode.OK, 23))
        self.assertEqual(decode_status(frame), (StatusCode.OK, 23))

    def test_stream_decoder_handles_fragmentation_and_noise(self) -> None:
        first = encode_status(MessageType.ACK, StatusCode.OK, 1)
        second = encode_status(MessageType.DROP_SUCCESS, StatusCode.OK, 2)
        stream = StreamDecoder()
        self.assertEqual(stream.feed(b"noise" + first[:3]), [])
        decoded = stream.feed(first[3:] + second)
        self.assertEqual([frame.message_type for frame in decoded], [MessageType.ACK, MessageType.DROP_SUCCESS])


class AssignmentTests(unittest.TestCase):
    def test_first_stable_classes_get_first_detection_order(self) -> None:
        detections = [
            Detection(2, 1, 120, 80, 0.9),
            Detection(0, 2, 40, 20, 0.9),
            Detection(2, 1, 30, 20, 0.9),
            Detection(4, 7, 70, 50, 0.9),
            Detection(3, 4, 10, 80, 0.9),
            Detection(1, 5, 150, 50, 0.9),
            Detection(6, 8, 20, 120, 0.9),
        ]
        bins = DynamicBinMap()
        assigned = bins.assign_detections(detections)
        self.assertEqual(bins.mapping[(2, 1)], 1)
        self.assertEqual(bins.mapping[(0, 2)], 2)
        self.assertEqual(len(bins.mapping), 6)
        self.assertEqual(assigned[0].bin_id, 1)
        with self.assertRaisesRegex(ValueError, "more classes"):
            bins.assign((8, 9))

    def test_temporal_consensus_requires_three_stable_frames(self) -> None:
        consensus = TemporalConsensus(required_frames=3, min_confidence=0.65)
        frame = [Detection(0, 4, 20.0, 30.0, 0.8, track_id=7)]
        self.assertEqual(consensus.add_frame(frame), [])
        self.assertEqual(consensus.add_frame(frame), [])
        stable = consensus.add_frame(frame)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].class_key, (0, 4))

    def test_session_rescans_after_drop_failure_and_counts_only_success(self) -> None:
        calibration = Homography.from_points(
            [(0, 0), (160, 0), (160, 160), (0, 160)],
            [(0, 0), (160, 0), (160, 160), (0, 160)],
        )
        session = VisionSession(calibration)
        initial = [Detection(i, i, i * 20, i * 20, 0.9) for i in range(6)]
        session.initialize_round(initial)
        self.assertIsNone(session.target_frame(initial, 9))
        self.assertIsNone(session.pending_class)
        self.assertEqual(session.handle_stm32_frame(encode_status(MessageType.ACK, StatusCode.OK, 0)), "ready")
        packet = session.target_frame(initial, 10)
        self.assertIsNotNone(packet)
        self.assertEqual(session.success_count, 0)
        self.assertEqual(session.handle_stm32_frame(encode_status(MessageType.DROP_FAILED, StatusCode.SENSOR_TIMEOUT, 1)), "retry_after_rescan")
        self.assertTrue(session.rescan_required)
        session.target_frame(initial, 11)
        self.assertEqual(session.pending_class, (0, 0))
        self.assertEqual(session.handle_stm32_frame(encode_status(MessageType.DROP_SUCCESS, StatusCode.OK, 2)), "rescan")
        self.assertEqual(session.success_count, 1)

    def test_session_falls_back_when_failed_class_is_no_longer_visible(self) -> None:
        calibration = Homography.from_points(
            [(0, 0), (160, 0), (160, 160), (0, 160)],
            [(0, 0), (160, 0), (160, 160), (0, 160)],
        )
        session = VisionSession(calibration)
        initial = [Detection(i, i, i * 20, i * 20, 0.9) for i in range(6)]
        session.initialize_round(initial)
        session.handle_stm32_frame(encode_status(MessageType.ACK, StatusCode.OK, 0))
        session.target_frame(initial, 10)
        self.assertEqual(session.pending_class, (0, 0))
        session.handle_stm32_frame(encode_status(MessageType.DROP_FAILED, StatusCode.SENSOR_TIMEOUT, 1))
        refreshed = initial[1:]
        session.target_frame(refreshed, 11)
        self.assertEqual(session.pending_class, (1, 1))


class PipelineTests(unittest.TestCase):
    def test_stabilizes_maps_six_classes_and_waits_for_controller_ready(self) -> None:
        colors = [
            (245, 245, 245), (30, 70, 180), (90, 180, 235), (245, 220, 20),
            (220, 20, 30), (128, 128, 128), (20, 180, 40), (130, 80, 170),
            (15, 15, 15), (245, 100, 35),
        ]
        prototypes = {name: rgb_to_lab(rgb) for name, rgb in zip(COLOR_NAMES, colors)}
        classifier = ColorPrototypeClassifier(prototypes, max_distance=30, min_margin=4)
        calibration = Homography.from_points(
            [(0, 0), (160, 0), (160, 160), (0, 160)],
            [(0, 0), (160, 0), (160, 160), (0, 160)],
        )
        session = VisionSession(calibration)
        pipeline = VisionPipeline(session, classifier)
        objects = [
            FrameObject(
                shape_id=index,
                center_px=(20.0 + index, 30.0 + index),
                shape_confidence=0.95,
                rgb_samples=((220.0, 20.0, 30.0), (221.0, 21.0, 31.0)),
                track_id=f"piece-{index}",
                area_px=100.0 + index,
            )
            for index in range(6)
        ]

        self.assertIsNone(pipeline.process_frame(objects, 1))
        self.assertIsNone(pipeline.process_frame(objects, 2))
        self.assertFalse(pipeline.mapping_ready)
        self.assertEqual(
            pipeline.handle_controller_frame(encode_status(MessageType.ACK, StatusCode.OK, 0)),
            "ready",
        )
        packet = pipeline.process_frame(objects, 3)
        self.assertTrue(pipeline.mapping_ready)
        self.assertIsNotNone(packet)
        target = decode_vision_result(decode_frame(packet))
        self.assertEqual((target["shape_id"], target["color_id"], target["bin_id"]), (5, 4, 6))
        self.assertAlmostEqual(target["x_mm"], 25.0)
        self.assertEqual(pipeline.handle_controller_frame(
            encode_status(MessageType.ACTION_STARTED, StatusCode.OK, 1)
        ), "wait")

        self.assertEqual(pipeline.handle_controller_frame(
            encode_status(MessageType.DROP_FAILED, StatusCode.SENSOR_TIMEOUT, 1)
        ), "retry_after_rescan")
        self.assertIsNone(pipeline.process_frame(objects, 4))
        self.assertIsNone(pipeline.process_frame(objects, 5))
        retried_packet = pipeline.process_frame(objects, 6)
        self.assertIsNotNone(retried_packet)
        retry = decode_vision_result(decode_frame(retried_packet))
        self.assertEqual((retry["shape_id"], retry["color_id"], retry["bin_id"]), (5, 4, 6))


class ColorTests(unittest.TestCase):
    def test_lab_conversion_and_classifier_abstention(self) -> None:
        prototypes = {name: rgb_to_lab(rgb) for name, rgb in zip(COLOR_NAMES, [
            (245, 245, 245), (30, 70, 180), (90, 180, 235), (245, 220, 20), (220, 20, 30),
            (128, 128, 128), (20, 180, 40), (130, 80, 170), (15, 15, 15), (245, 100, 35),
        ])}
        classifier = ColorPrototypeClassifier(prototypes, max_distance=30, min_margin=4)
        result = classifier.classify([(222, 22, 32), (219, 21, 30), (223, 18, 31)])
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "red")
        self.assertIsNone(classifier.classify([]))


if __name__ == "__main__":
    unittest.main()
