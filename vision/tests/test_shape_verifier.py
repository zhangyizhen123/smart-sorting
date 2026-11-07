import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from shape_verifier import features, FEATURES, VerifierNetwork


class ShapeVerifierTests(unittest.TestCase):
    def test_empty_and_single_pixel_features_are_finite(self):
        image = np.zeros((64, 64, 3), np.uint8)
        for on in (False, True):
            mask = np.zeros((64, 64), np.uint8)
            mask[32, 32] = on
            value = features(image, mask)
            self.assertEqual(value.shape, (FEATURES,))
            self.assertTrue(np.isfinite(value).all())

    def test_background_cannot_invent_visible_edges(self):
        flat = np.full((100, 100, 3), 220, np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        mask[25:75, 25:75] = 1
        object_image = flat.copy()
        object_image[mask > 0] = 30
        a, b = features(flat, mask), features(object_image, mask)
        self.assertEqual(float(a[-9]), 0.)
        self.assertGreater(float(b[-9]), .9)
        np.testing.assert_array_equal(flat, np.full((100, 100, 3), 220, np.uint8))

    def test_nine_shapes_plus_background(self):
        torch.set_num_threads(1)
        model = VerifierNetwork().eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.zeros((2, FEATURES))).shape), (2, 10))

    def test_mismatched_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            features(np.zeros((50, 50, 3), np.uint8), np.zeros((20, 20), np.uint8))


if __name__ == '__main__':
    unittest.main()
