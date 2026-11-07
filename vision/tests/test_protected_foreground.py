from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from protected_foreground import retained_indices


class ProtectedForegroundTests(unittest.TestCase):
    def test_two_detector_support_protects_without_using_labels(self):
        policy=dict(foreground_threshold=.2,protect_confidence=.7,require_partner=True)
        supported=dict(consensus_source='primary',consensus_partner_index=0,confidence=.9,verifier_background_probability=.01)
        unsupported={**supported,'consensus_partner_index':None}
        self.assertEqual(retained_indices([supported,unsupported,supported],[.01,.01,.8],policy),([0,2],[True,False]))

    def test_background_evidence_or_low_confidence_does_not_gain_protection(self):
        policy=dict(foreground_threshold=.2,protect_confidence=.7,require_partner=True)
        base=dict(consensus_source='primary',consensus_partner_index=0,confidence=.9,verifier_background_probability=.01)
        values=[{**base,'confidence':.6},{**base,'verifier_background_probability':.9},{**base,'consensus_source':'secondary'}]
        self.assertEqual(retained_indices(values,[.01]*3,policy),([],[]))

    def test_misaligned_or_invalid_probabilities_are_rejected(self):
        policy=dict(foreground_threshold=.2,protect_confidence=.7,require_partner=True)
        with self.assertRaises(ValueError): retained_indices([{}],[],policy)
        with self.assertRaises(ValueError): retained_indices([{}],[float('nan')],policy)


if __name__=='__main__':unittest.main()
