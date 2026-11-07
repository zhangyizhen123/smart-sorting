"""Regression cases for false contour bridges, holes and non-square coordinates."""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mask_display import native_mask_geometry
from predict_stl_demo import serialize_predictions


class MaskDisplayTests(unittest.TestCase):
    def test_disconnected_regions_and_hole_remain_separate(self):
        mask = np.zeros((80, 120), np.uint8)
        mask[10:60, 10:55] = 1
        mask[25:40, 25:40] = 0
        mask[15:40, 85:110] = 1
        result = native_mask_geometry(mask)
        self.assertEqual(result["mask_components"], 2)
        self.assertEqual(len(result["mask_contours"]), 3)
        def inside(x, y):
            return sum(cv2.pointPolygonTest(np.array(r, np.float32), (x/120, y/80), False) > 0 for r in result["mask_contours"]) % 2 == 1
        self.assertTrue(inside(15, 15))
        self.assertTrue(inside(95, 25))
        self.assertFalse(inside(30, 30), "Hole must remain empty")
        self.assertFalse(inside(70, 25), "Disconnected objects must not acquire a bridge")

    def test_serializer_preserves_classes_confidence_and_legacy_polygon(self):
        image = np.zeros((80, 120, 3), np.uint8)
        mask = np.zeros((1, 80, 120), np.uint8); mask[0, 10:35, 20:90] = 1
        result = Results(image, "fixture", {0: "cylinder"}, boxes=torch.tensor([[20, 10, 90, 35, .8, 0]]), masks=torch.tensor(mask))
        old = serialize_predictions(result)
        new = serialize_predictions(result, include_mask_geometry=True)
        self.assertEqual({key: new[0][key] for key in old[0]}, old[0])
        self.assertEqual(new[0]["mask_size"], [120, 80])
        self.assertTrue(all(0 <= x <= 1 and 0 <= y <= 1 for ring in new[0]["mask_contours"] for x, y in ring))

    def test_unscaled_letterbox_masks_are_rejected(self):
        result = Results(np.zeros((80, 120, 3), np.uint8), "fixture", {0: "cylinder"}, boxes=torch.tensor([[1, 1, 50, 50, .8, 0]]), masks=torch.ones((1, 32, 32)))
        with self.assertRaises(ValueError):
            serialize_predictions(result, include_mask_geometry=True)


if __name__ == "__main__":
    unittest.main()
