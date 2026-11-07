import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_guided_masks import refine_masks
from contour_quality import mask_iou, topology


class ImageGuidedTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((128, 160, 3), 220, np.uint8)
        self.truth = np.zeros((128, 160), np.uint8)
        self.truth[25:100, 30:120] = 1
        self.image[self.truth > 0] = (25, 60, 170)

    def test_corrects_visible_jagged_boundary(self):
        raw = self.truth.copy()
        raw[25:100:2, 119:121] = 1
        raw[25:100:2, 29:31] = 0
        saved = raw.copy()
        refined, info = refine_masks(self.image, [raw])
        self.assertGreater(mask_iou(refined[0], self.truth), mask_iou(raw, self.truth))
        self.assertEqual(info[0]["status"], "accepted")
        np.testing.assert_array_equal(raw, saved)

    def test_empty_and_thin_masks_remain_valid(self):
        thin = np.zeros_like(self.truth)
        thin[50, 20:60] = 1
        masks, info = refine_masks(self.image, [np.zeros_like(thin), thin])
        np.testing.assert_array_equal(masks[1], thin)
        self.assertEqual(info[1]["reason"], "insufficient_interior_seed")

    def test_topology_and_existing_holes_are_preserved(self):
        ring = self.truth.copy()
        ring[45:65, 55:75] = 0
        refined, _ = refine_masks(self.image, [ring])
        self.assertEqual(topology(refined[0]), topology(ring))

    def test_cannot_expand_onto_neighbor(self):
        left = self.truth.copy()
        left[:, 76:] = 0
        right = self.truth.copy()
        right[:, :76] = 0
        refined, _ = refine_masks(self.image, [left, right])
        self.assertFalse(np.any(refined[0] & refined[1]))

    def test_deterministic_result(self):
        raw = cv2.dilate(self.truth, np.ones((3, 3), np.uint8))
        a, ai = refine_masks(self.image, [raw])
        b, bi = refine_masks(self.image, [raw])
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(ai, bi)

    def test_native_coordinates_required(self):
        with self.assertRaises(ValueError):
            refine_masks(self.image, [np.zeros((50, 50), np.uint8)])


if __name__ == "__main__":
    unittest.main()
