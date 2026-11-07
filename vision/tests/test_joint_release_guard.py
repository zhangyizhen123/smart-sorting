import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from release_joint_recipe import require_success


class ReleaseGuardTests(unittest.TestCase):
    def report(self):
        return dict(passes_joint_test_gate=True,images=2900,both_styles_do_not_regress=True,
                    negative_alarm_does_not_regress=True,comparisons={name:dict(simultaneous_adjusted_intervals=[[.01,.1]]*4,
                    common_match_mask_iou_delta=.01) for name in ['v1_deployed','v2_deployed']})

    def test_full_passing_report_is_eligible(self):
        require_success(self.report())

    def test_development_only_or_failed_results_do_not_release(self):
        for field,value in [('passes_joint_test_gate',False),('images',1108),('both_styles_do_not_regress',False),('negative_alarm_does_not_regress',False)]:
            report=self.report();report[field]=value
            with self.assertRaises(ValueError):
                require_success(report)

    def test_nonpositive_interval_blocks_even_with_pass_flag(self):
        report=self.report();report['comparisons']['v2_deployed']['simultaneous_adjusted_intervals'][0]=[-.01,.1]
        with self.assertRaises(ValueError):
            require_success(report)


if __name__=='__main__':
    unittest.main()
