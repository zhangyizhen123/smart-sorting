import json,sys,tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_optimization_candidates import compare


class OptimizationGateTests(unittest.TestCase):
    def fixture(self,root,name,strict,fp,coverage,boundary,iou,legacy=None,focused=None):
        directory=root/'evaluation'/name;(directory/'records').mkdir(parents=True)
        def group(score):return {'scores':{'0.5':{'strict_instance_score':score}},'false_positives_per_image':fp,
                                'geometry':{'boundary_iou_with_unmatched_zero':coverage}}
        groups={'synthetic_validation':group(strict),'legacy':group(strict if legacy is None else legacy),
                'focused':group(strict if focused is None else focused),'negative':{'empty_false_detection_rate':fp},
                'real':group(strict),'real_empty':{'empty_false_detection_rate':fp}}
        (directory/'summary.json').write_text(json.dumps({'groups':groups}),encoding='utf-8')
        (directory/'records/example.json').write_text(json.dumps({'key':'same_scene','domain':'focused','truth_classes':[0],
                'predictions':[{'class_id':0}],'ious':[[iou]],'boundary_ious':[[boundary]]}),encoding='utf-8')

    def check_candidate(self,**overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root,'v1_deployed',.7,.2,.6,.7,.85)
            self.fixture(root,'v2_deployed',.72,.15,.62,.71,.86)
            args={'strict':.8,'fp':.1,'coverage':.75,'boundary':.8,'iou':.9,**overrides}
            self.fixture(root,'candidate',**args)
            return compare(root)

    def test_all_three_improvements_pass(self):
        result=self.check_candidate()
        self.assertEqual(result['selected_for_sealed_test'],'candidate')
        self.assertFalse(result['automatic_deployment'])
        self.assertFalse(result['sealed_test_evaluated'])

    def test_more_false_positives_fail_despite_higher_recognition(self):
        result=self.check_candidate(fp=.3)
        self.assertIsNone(result['selected_for_sealed_test'])

    def test_smoother_but_less_accurate_boundary_fails(self):
        result=self.check_candidate(boundary=.6)
        self.assertIsNone(result['selected_for_sealed_test'])

    def test_one_style_regression_cannot_be_hidden_by_total(self):
        result=self.check_candidate(legacy=.6,focused=.99)
        self.assertIsNone(result['selected_for_sealed_test'])

    def test_ties_are_not_claimed_as_strict_improvement(self):
        result=self.check_candidate(strict=.72,fp=.15,coverage=.62)
        self.assertIsNone(result['selected_for_sealed_test'])


if __name__=='__main__':unittest.main()
