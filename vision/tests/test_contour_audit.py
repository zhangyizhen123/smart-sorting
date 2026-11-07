import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_contour_audit import metric_record, geometry_summary
from evaluate_focused import score_records


class ContourAuditTests(unittest.TestCase):
    def test_truth_order_comes_from_labels_not_sorted_manifest(self):
        a=np.zeros((40,80),np.uint8);b=a.copy();a[5:30,5:25]=1;b[5:30,45:65]=1
        row={'image':'fixture','domain':'legacy','classes':[0,5]}
        predictions=[{'class_id':5},{'class_id':0}]
        record=metric_record(row,predictions,[a,b],[a,b],80,40,[5,0])
        self.assertEqual(record['truth_classes'],[5,0])
        self.assertEqual(score_records([record])['strict_instance_score'],1.)

    def test_unmatched_geometry_and_background_counted(self):
        a=np.zeros((40,80),np.uint8);a[5:30,5:25]=1;b=np.roll(a,40,axis=1)
        row={'image':'fixture','domain':'real','classes':[0]}
        record=metric_record(row,[{'class_id':0}],[b],[a],80,40,[0])
        g=geometry_summary([record]);s=score_records([record])
        self.assertEqual(g['unmatched_truth'],1);self.assertIsNone(g['matched_mean_center_error_pixels'])
        self.assertEqual(g['boundary_iou_with_unmatched_zero'],0.)
        self.assertEqual(s['missed_instances'],1);self.assertEqual(s['spurious_detections'],1)


if __name__=='__main__':unittest.main()
