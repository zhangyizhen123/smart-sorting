import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from specialist_shapes import consult


class SpecialistTests(unittest.TestCase):
    def test_high_confidence_primary_is_unchanged(self):
        policy=dict(match_iou=.5,maximum_primary_confidence=.8,minimum_specialist_confidence=.7,confidence_margin=0.,allowed_classes=[6,7])
        primary=[dict(class_id=0,confidence=.9,box=[0,0,10,10])]
        secondary=[dict(class_id=6,confidence=.95,box=[0,0,10,10])]
        self.assertEqual(consult(primary,secondary,policy),primary)

    def test_strongest_supported_class_is_chosen_before_specialty_filter(self):
        policy=dict(match_iou=.5,maximum_primary_confidence=.8,minimum_specialist_confidence=.7,confidence_margin=0.,allowed_classes=[6,7])
        primary=[dict(class_id=0,confidence=.5,box=[0,0,10,10])]
        secondary=[dict(class_id=6,confidence=.8,box=[0,0,10,10]),dict(class_id=2,confidence=.9,box=[0,0,10,10])]
        self.assertEqual(consult(primary,secondary,policy),primary)
        result=consult(primary,secondary[:1],policy)
        self.assertEqual(result[0]['class_id'],6)
        self.assertEqual(primary[0]['class_id'],0)
        self.assertEqual(result[0]['box'],primary[0]['box'])


if __name__=='__main__':
    unittest.main()
