from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from consensus_evidence import combine_native,reclassify


class ConsensusEvidenceTests(unittest.TestCase):
    def test_class_change_can_be_replayed_without_altering_geometry(self):
        a=dict(class_id=1,confidence=.6,box=[0,0,10,10],source_prediction_index=1)
        b=dict(class_id=2,confidence=.95,box=[0,0,10,10],source_prediction_index=0)
        masks=[np.ones((2,2),bool)];originals=[np.zeros((2,2),bool),np.eye(2,dtype=bool)]
        result=combine_native({'primary':([a],masks,originals,{'conf':.25}),
                               'secondary':([b],masks,[masks[0]],{'conf':.25})},
                              dict(match_iou=.3,class_source='stronger_secondary',secondary_margin=.1,
                                   unmatched_primary_confidence=.25,unmatched_secondary_confidence=.6))
        self.assertEqual(result[0][0]['class_id'],2)
        self.assertEqual(reclassify(result[0],1.01)[0]['class_id'],1)
        self.assertTrue(np.array_equal(result[2][0],originals[1]))
        self.assertIs(result[1][0],masks[0])
        self.assertEqual(a['class_id'],1)

    def test_unmatched_secondary_keeps_own_class(self):
        a=dict(class_id=1,confidence=.9,box=[0,0,10,10],source_prediction_index=0)
        b=dict(class_id=5,confidence=.9,box=[20,20,30,30],source_prediction_index=0)
        masks=[np.ones((2,2),bool)]
        predictions=combine_native({'primary':([a],masks,masks,{}),'secondary':([b],masks,masks,{})},
                    dict(match_iou=.3,class_source='primary',unmatched_primary_confidence=.25,unmatched_secondary_confidence=.6))[0]
        self.assertEqual([p['class_id'] for p in reclassify(predictions,.1)],[1,5])
        self.assertEqual([p['source_prediction_index'] for p in predictions],[0,1])


if __name__=='__main__':unittest.main()
