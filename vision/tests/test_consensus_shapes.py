import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from consensus_shapes import select_predictions


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(match_iou=.5, unmatched_primary_confidence=.7,
                           unmatched_secondary_confidence=.7, class_source='primary')

    def pred(self, cls=0, confidence=.5, box=(0,0,10,10)):
        return dict(class_id=cls, confidence=confidence, box=list(box))

    def test_no_detections(self):
        self.assertEqual(select_predictions([],[],self.policy),[])

    def test_agreement_is_class_blind_and_keeps_primary_geometry(self):
        a,b = self.pred(1),self.pred(4)
        selected = select_predictions([a],[b],self.policy)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['class_id'],1)
        self.assertEqual(selected[0]['consensus_source_index'],0)
        self.assertNotIn('consensus_source',a)

    def test_unmatched_confidence_and_secondary_duplicates(self):
        a = self.pred(confidence=.8)
        b = self.pred(confidence=.9)
        far = self.pred(3,.8,(30,30,40,40))
        selected = select_predictions([a],[b,b,far],self.policy)
        self.assertEqual([(p['consensus_source'],p['consensus_source_index']) for p in selected],
                         [('primary',0),('secondary',2)])
        self.assertEqual(select_predictions([self.pred()],[],self.policy),[])

    def test_class_switch_uses_only_prediction_confidence(self):
        policy = {**self.policy,'class_source':'stronger_secondary','secondary_margin':.1}
        selected = select_predictions([self.pred(1,.5)],[self.pred(4,.8)],policy)
        self.assertEqual(selected[0]['class_id'],4)


if __name__ == '__main__':
    unittest.main()
