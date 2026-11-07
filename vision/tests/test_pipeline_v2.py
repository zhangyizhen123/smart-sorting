import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_v2 import apply_verifier
from evaluate_shape_verifier import apply_policy


class RuntimeAlignmentTests(unittest.TestCase):
    def test_runtime_and_development_decisions_match(self):
        row = {'predictions':[{'class_id':1,'confidence':.7999998},{'class_id':8,'confidence':.4}],
               'ious':[[.9,.1]], 'boundary_ious':[[.8,.05]], 'center_errors_pixels':[[1.,50.]]}
        probabilities = [[0,0,.95,0,0,0,0,0,0,.05],[0,0,0,0,0,0,0,0,.01,.99]]
        policy = {'maximum_base_confidence':.8,'class_probability':.75,'background_probability':.95}
        masks = [np.ones((5,5),np.uint8),np.zeros((5,5),np.uint8)]
        expected = apply_policy(row, probabilities, policy)
        actual, retained = apply_verifier(row['predictions'], masks, probabilities, policy)
        self.assertEqual(actual, expected['predictions'])
        self.assertEqual(actual[0]['class_id'], 2)  # Never round confidence before the gate.
        np.testing.assert_array_equal(retained[0], masks[0])

    def test_alignment_errors_fail(self):
        with self.assertRaises(ValueError):
            apply_verifier([{'class_id':0,'confidence':.5}], [], [], {})

    def test_empty_output_is_valid(self):
        predictions, masks = apply_verifier([], [], [], {})
        self.assertEqual(predictions, []);self.assertEqual(masks, [])


if __name__ == '__main__':
    unittest.main()
