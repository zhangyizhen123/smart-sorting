from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from release_joint_round_v4 import require_success


class SecondRoundReleaseGuardTests(unittest.TestCase):
    def report(self):
        return dict(passes_joint_test_gate=True,images=5800,both_styles_do_not_regress=True,negative_alarm_does_not_regress=True,
                    per_round_family_alpha=.025,bootstrap_replicates=5000,comparisons={name:dict(simultaneous_adjusted_intervals=[[.01,.1]]*4,
                    common_match_mask_iou_delta=.01) for name in ['v1_deployed','v2_deployed']})

    def test_complete_second_round_protocol_is_required(self):
        require_success(self.report())
        for field,value in [('images',2900),('per_round_family_alpha',.05),('bootstrap_replicates',500),('both_styles_do_not_regress',False),('negative_alarm_does_not_regress',False)]:
            report=self.report();report[field]=value
            with self.assertRaises(ValueError): require_success(report)

    def test_nonpositive_interval_or_mask_regression_blocks_release(self):
        report=self.report();report['comparisons']['v2_deployed']['simultaneous_adjusted_intervals'][0]=[-.001,.1]
        with self.assertRaises(ValueError): require_success(report)
        report=self.report();report['comparisons']['v1_deployed']['common_match_mask_iou_delta']=-.001
        with self.assertRaises(ValueError): require_success(report)


if __name__=='__main__':unittest.main()
