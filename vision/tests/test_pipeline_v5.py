from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_v5 import ProtectedPipeline


class Source:
    def __init__(self,result): self.result=result
    def predict_native(self,image,confidence=None): return self.result


class PipelineV5Tests(unittest.TestCase):
    def test_protection_keeps_native_masks_aligned_after_rejection_and_addition(self):
        one=np.ones((4,4),np.uint8);eye=np.eye(4,dtype=np.uint8);zero=np.zeros((4,4),np.uint8)
        def p(cls,confidence,box,index): return dict(class_id=cls,base_class_id=cls,confidence=confidence,box=box,source_prediction_index=index,verifier_background_probability=.01)
        a=[p(1,.8,[0,0,10,10],2),p(2,.4,[20,0,30,10],0)]
        b=[p(1,.9,[0,0,10,10],0),p(5,.9,[40,0,50,10],1)]
        base=SimpleNamespace(children={'primary':Source((a,[one,eye],[eye,zero,one],{})),
                                      'secondary':Source((b,[one,eye],[one,eye],{}))},
                             policy=dict(match_iou=.3,class_source='stronger_secondary',secondary_margin=.1,unmatched_primary_confidence=.25,unmatched_secondary_confidence=.6))
        pipeline=ProtectedPipeline.__new__(ProtectedPipeline);pipeline.base=base
        pipeline.specialist=Source(([],[],[],{}));pipeline.foreground=SimpleNamespace(probabilities=lambda x:np.array([.01,.01,.9]))
        pipeline.policy=dict(secondary_margin=.1,foreground=dict(foreground_threshold=.2,protect_confidence=.7,require_partner=True),
                             specialist=dict(match_iou=.5,maximum_primary_confidence=1.01,minimum_specialist_confidence=.5,confidence_margin=0,allowed_classes=[6,7]))
        predictions,masks,originals,settings=pipeline.predict_native(Image.new('RGB',(4,4),'gray'))
        self.assertEqual([p['class_id'] for p in predictions],[1,5])
        self.assertEqual([p['source_prediction_index'] for p in predictions],[0,1])
        self.assertEqual([p['foreground_source_index'] for p in predictions],[0,2])
        self.assertEqual([p['foreground_protected'] for p in predictions],[True,False])
        self.assertTrue(np.array_equal(originals[0],one));self.assertTrue(np.array_equal(originals[1],eye))
        self.assertTrue(np.array_equal(masks[0],one));self.assertTrue(np.array_equal(masks[1],eye))


if __name__=='__main__':unittest.main()
