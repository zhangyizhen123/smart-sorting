"""Review approval must bind exact sources, masks, identities and explicit intent."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_real_review import validate_review


class ReviewValidation(unittest.TestCase):
    def setUp(self):
        self.data={'review_revision':'revision','images':[{'image_id':'image','source_sha256':'source',
            'instances':[{'instance_id':'i1','mask_sha256':'mask','class_name':'cube','class_id':1}]}]}
        self.export={'version':1,'source':'explicit_user_review','review_revision':'revision','reviews':[
            {'image_id':'image','image_sha256':'source','approved':False,'actual_count':1,'notes':'',
             'instances':[{'instance_id':'i1','mask_sha256':'mask','class_name':'cube'}]}]}

    def test_unapproved_remains_pending(self):
        out=validate_review(self.data,self.export)
        self.assertEqual(out['reviewed_image_count'],0)
        self.assertFalse(out['images'][0]['training_approved'])
        self.assertNotIn('human_review_status',self.data['images'][0])

    def test_explicit_approval_allows_correction_but_not_training(self):
        r=self.export['reviews'][0];r['approved']=True;r['instances'][0]['class_name']='tri_prism'
        out=validate_review(self.data,self.export);image=out['images'][0]
        self.assertEqual(out['reviewed_image_count'],1)
        self.assertEqual(image['instances'][0]['class_id'],3)
        self.assertTrue(image['instances'][0]['boundary_human_approved'])
        self.assertFalse(image['training_approved'])

    def test_reject_invalid_exports(self):
        for key,value in [('review_revision','stale'),('source','automatic_prediction')]:
            with self.subTest(key=key):
                e=copy.deepcopy(self.export);e[key]=value
                with self.assertRaises(ValueError):validate_review(self.data,e)

    def test_reject_missing_and_duplicate_images(self):
        for records in [[],self.export['reviews']*2]:
            e=copy.deepcopy(self.export);e['reviews']=records
            with self.assertRaises(ValueError):validate_review(self.data,e)

    def test_reject_approval_coercion_and_bad_counts(self):
        for key,value in [('approved','true'),('approved',1),('actual_count',True),('actual_count',-1),('notes',None)]:
            e=copy.deepcopy(self.export);e['reviews'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_review(self.data,e)

    def test_approved_count_and_known_class_required(self):
        for kind in ['count','class']:
            e=copy.deepcopy(self.export);r=e['reviews'][0];r['approved']=True
            if kind=='count':r['actual_count']=2
            else:r['instances'][0]['class_name']='unknown'
            with self.assertRaises(ValueError):validate_review(self.data,e)

    def test_changed_source_mask_or_instance_rejected(self):
        for key in ['image_sha256','mask_sha256','instance_id']:
            e=copy.deepcopy(self.export);r=e['reviews'][0]
            (r if key=='image_sha256' else r['instances'][0])[key]='different'
            with self.subTest(key=key),self.assertRaises(ValueError):validate_review(self.data,e)

if __name__=='__main__':unittest.main()
