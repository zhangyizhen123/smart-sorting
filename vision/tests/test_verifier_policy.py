import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_shape_verifier import apply_policy


class VerifierPolicyTests(unittest.TestCase):
    def fixture(self):
        return {'predictions':[{'class_id':1,'confidence':.5}, {'class_id':8,'confidence':.4}],
                'ious':[[.9,.1]], 'boundary_ious':[[.8,.05]], 'center_errors_pixels':[[1.,50.]]}

    def test_retention_keeps_geometry_columns_aligned(self):
        row = self.fixture()
        result = apply_policy(row, [[0,0,.95,0,0,0,0,0,0,.05],[0,0,0,0,0,0,0,0,.01,.99]],
                              {'maximum_base_confidence':.8,'class_probability':.75,'background_probability':.95})
        self.assertEqual([r['class_id'] for r in result['predictions']], [2])
        self.assertEqual(result['ious'], [[.9]])
        self.assertEqual(result['center_errors_pixels'], [[1.]])
        self.assertEqual(row['predictions'][0]['class_id'], 1)

    def test_all_filtered_preserves_truth_rows(self):
        row = self.fixture()
        result = apply_policy(row, [[0]*9+[1],[0]*9+[1]],
                              {'maximum_base_confidence':.8,'class_probability':.75,'background_probability':.95})
        self.assertEqual(result['predictions'], [])
        self.assertEqual(result['ious'], [[]])

    def test_class_cannot_override_confident_detection(self):
        row = self.fixture()
        result = apply_policy(row, [[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]],
                              {'maximum_base_confidence':.3,'class_probability':.75,'background_probability':1.01})
        self.assertEqual([r['class_id'] for r in result['predictions']], [1,8])


if __name__ == '__main__':
    unittest.main()
