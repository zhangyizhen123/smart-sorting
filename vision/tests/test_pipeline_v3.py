import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_v3 import reindex_native
from evaluate_foreground_gate import filter_record


class RecipeAlignmentTests(unittest.TestCase):
    def test_original_mapping_survives_two_rejections(self):
        predictions=[dict(class_id=1,source_prediction_index=2),dict(class_id=4,source_prediction_index=4)]
        effective=[np.full((3,3),10),np.full((3,3),20)]
        native=[np.full((3,3),j) for j in range(5)]
        p,m,o=reindex_native(predictions,effective,native,[1])
        self.assertEqual(p[0]['class_id'],4)
        self.assertEqual(p[0]['source_prediction_index'],0)
        self.assertEqual(p[0]['native_source_prediction_index'],4)
        np.testing.assert_array_equal(m[0],20);np.testing.assert_array_equal(o[0],4)
        self.assertEqual(predictions[1]['source_prediction_index'],4)

    def test_foreground_filter_preserves_matrix_alignment(self):
        before=dict(predictions=[dict(class_id=0),dict(class_id=1)],ious=[[.1,.9]],
                    boundary_ious=[[.2,.8]],center_errors_pixels=[[30.,1.]])
        row=filter_record(before,[.1,.9],.5)
        self.assertEqual(row['ious'],[[.9]])
        self.assertEqual(row['boundary_ious'],[[.8]])
        self.assertEqual(row['predictions'][0]['foreground_source_index'],1)
        self.assertEqual(len(before['predictions']),2)


if __name__=='__main__':
    unittest.main()
