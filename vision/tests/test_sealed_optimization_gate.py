import json,sys,tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import sealed_optimization_test as sealed


class SealedGateTests(unittest.TestCase):
    def setUp(self):
        self.c=np.tile([3,4,0,3.2],(20,1)).astype(float)
        self.b=np.tile([2,4,1,2.4],(20,1)).astype(float)
        self.p=np.tile([.4,.2,4],(20,1)).astype(float)
        self.strata=[np.arange(10),np.arange(10,20)]

    def test_paired_improvements_and_intervals(self):
        point=sealed.gains(self.c,self.b,self.p)
        np.testing.assert_allclose(point,[.35,1,.2,.1])
        intervals=sealed.bootstrap_gains(self.c,self.b,self.p,self.strata,replicates=100)
        self.assertTrue(all(lo>0 for lo,hi in intervals))
        self.assertEqual(intervals,sealed.bootstrap_gains(self.c,self.b,self.p,self.strata,replicates=100))

    def test_false_positive_regression_remains_negative(self):
        self.c[:,2]=2
        self.assertLess(sealed.gains(self.c,self.b,self.p)[1],0)

    def test_no_common_match_cannot_pass(self):
        self.p[:]=0
        self.assertLess(sealed.gains(self.c,self.b,self.p)[3],0)

    def test_no_development_passing_candidate_cannot_unlock_test(self):
        original=sealed.RUN
        try:
            with tempfile.TemporaryDirectory() as tmp:
                sealed.RUN=Path(tmp)
                for name,value in [('status.json',{'stage':'training_complete'}),('watcher_status.json',{'stage':'decision_needed'}),('development_comparison.json',{'selected_for_sealed_test':None})]:
                    (sealed.RUN/name).write_text(json.dumps(value),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'No development candidate'):
                    sealed.lock_candidate()
        finally:sealed.RUN=original


if __name__=='__main__':unittest.main()
