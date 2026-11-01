"""Regression tests for strict counting, overlap arbitration and visible labels."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_focused import match_instances, score_records
from make_focused_dataset import visible_polygons
from shape_refiner import PairwiseRefiner
from prepare_yolo import load_group_csv, group_id


def record(truth, prediction, ious):
    return {"image": "test.jpg", "truth_classes": truth, "predictions": [{"class_id": c} for c in prediction], "ious": ious}


class StrictScoringTests(unittest.TestCase):
    def test_bom_group_csv_uses_explicit_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "groups.csv"
            p.write_text("image,group_id\nframe01.jpg,physical_scene_A\n", encoding="utf-8-sig")
            self.assertEqual(group_id("frame01", load_group_csv(p), None), "physical_scene_A")

    def test_perfect(self):
        self.assertEqual(score_records([record([0, 6], [0, 6], [[1, 0], [0, 1]])])["strict_instance_score"], 1.)

    def test_miss_cannot_disappear_from_denominator(self):
        r = score_records([record([0, 6], [0], [[1], [0]])])
        self.assertEqual(r["strict_instance_score"], 0.5)
        self.assertEqual(r["missed_instances"], 1)

    def test_duplicate_invalidates_affected_instance(self):
        r = score_records([record([0, 6], [0, 6, 0], [[1, 0, 0.9], [0, 1, 0]])])
        self.assertEqual(r["strict_instance_score"], 0.5)
        self.assertEqual(r["duplicate_detections"], 1)

    def test_false_positive_counts(self):
        r = score_records([record([0], [0, 6], [[1, 0]])])
        self.assertEqual(r["strict_instance_score"], 0.5)
        self.assertEqual(r["spurious_detections"], 1)

    def test_wrong_class_not_used_for_matching(self):
        r = score_records([record([1, 2], [2, 1], [[0.95, 0.6], [0.6, 0.95]])])
        self.assertEqual(r["strict_instance_score"], 0.)
        self.assertEqual(r["confusion_rows_true_cols_pred"][1][2], 1)

    def test_optimal_assignment_not_label_order_greedy(self):
        self.assertEqual(set(match_instances([[0.9, 0.8], [0.8, 0.1]])), {(0, 1), (1, 0)})

    def test_empty_predictions(self):
        self.assertEqual(score_records([record([0], [], [[]])])["strict_instance_score"], 0.)
        self.assertEqual(score_records([record([], [0], [])])["strict_instance_score"], 0.)

    def test_three_way_arbiter_resolves_shared_cuboid(self):
        refiner = PairwiseRefiner.__new__(PairwiseRefiner)
        refiner.gate = 1.; refiner.minimum_probability = 0.5
        refiner.probabilities = lambda _: [np.array([[.5, .5]]), np.array([[.01, .99]]), np.array([[.99, .01]]), np.array([[.99, .005, .005]])]
        image = np.zeros((32, 32, 3), np.uint8); polygon = np.array([[2, 2], [28, 2], [28, 28], [2, 28]])
        self.assertEqual(refiner.refine(image, polygon, 2, .5), 1)
        self.assertEqual(refiner.refine(image, polygon, 8, .5), 8)
        self.assertEqual(refiner.refine(image, polygon, 2, 1.), 2)

    def test_visible_polygon_excludes_front_object(self):
        a = np.zeros((60, 60), bool); a[5:45, 5:45] = True
        b = np.zeros((60, 60), bool); b[25:55, 35:55] = True
        import cv2
        polygons = visible_polygons([a, b])
        canvas = np.zeros(a.shape, np.uint8)
        cv2.fillPoly(canvas, [polygons[0]], 1)
        self.assertEqual(np.count_nonzero(canvas.astype(bool) & b), 0)
        self.assertGreater(np.count_nonzero(canvas), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
