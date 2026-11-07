import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from contour_quality import regularize_masks, topology, mask_iou, pairwise_geometry, self_intersects


class ContourQualityTests(unittest.TestCase):
    def test_oblique_rectangle_simplifies_without_changing_source(self):
        m=np.zeros((240,320),np.uint8)
        cv2.fillPoly(m,[np.array([[30,40],[250,83],[221,190],[10,140]])],1)
        source=m.copy();out,meta=regularize_masks([m])
        np.testing.assert_array_equal(m,source)
        self.assertEqual(meta[0]['geometry_status'],'accepted')
        self.assertLess(meta[0]['regularized_vertices'],meta[0]['raw_vertices'])
        self.assertGreaterEqual(mask_iou(m,out[0]),.98)

    def test_curve_and_point_remain_faithful(self):
        for kind in ['ellipse','triangle']:
            m=np.zeros((240,320),np.uint8)
            if kind=='ellipse':cv2.ellipse(m,(160,120),(95,70),17,0,360,1,-1)
            else:cv2.fillPoly(m,[np.array([[160,10],[25,215],[300,215]])],1)
            out,meta=regularize_masks([m]);self.assertGreaterEqual(mask_iou(m,out[0]),.98)
            if kind=='triangle':self.assertTrue(out[0][10,160])

    def test_holes_components_and_edge_coordinates(self):
        m=np.zeros((90,160),np.uint8);m[:65,:55]=1;m[20:40,20:35]=0;m[15:70,100:150]=1
        out,meta=regularize_masks([m])
        self.assertEqual(topology(m),topology(out[0]));self.assertFalse(out[0][30,25]);self.assertFalse(out[0][40,80])
        self.assertEqual(meta[0]['mask_size'],[160,90])
        self.assertTrue(all(0<=x<=1 and 0<=y<=1 for ring in meta[0]['regularized_contours'] for x,y in ring))

    def test_thin_and_empty_masks_fall_back(self):
        m=np.zeros((50,80),np.uint8);m[20,10:70]=1
        for source in [m,np.zeros_like(m)]:
            out,meta=regularize_masks([source]);np.testing.assert_array_equal(out[0],source)
            self.assertEqual(meta[0]['geometry_status'],'fallback')
        self.assertEqual(regularize_masks([]),([],[]))

    def test_no_new_overlaps_even_when_order_reversed(self):
        a=np.zeros((120,200),np.uint8);b=a.copy()
        for y in range(15,105):
            edge=90+(y%3)
            a[y,15:edge]=1;b[y,edge:180]=1
        out,_=regularize_masks([a,b]);rev,_=regularize_masks([b,a])
        self.assertFalse(np.any(out[0]&out[1]))
        np.testing.assert_array_equal(out[0],rev[1]);np.testing.assert_array_equal(out[1],rev[0])

    def test_metrics_identical_shifted_and_unmatched(self):
        m=np.zeros((80,120),np.uint8);m[20:60,30:90]=1
        r=pairwise_geometry([m],[m],120,80)
        self.assertEqual(r['ious'],[[1.]]);self.assertEqual(r['boundary_ious'],[[1.]])
        self.assertEqual(r['center_errors_pixels'],[[0.]])
        moved=np.roll(m,4,axis=1);r=pairwise_geometry([m],[moved],120,80)
        self.assertAlmostEqual(r['center_errors_pixels'][0][0],4.)
        self.assertLess(r['boundary_ious'][0][0],r['ious'][0][0])
        self.assertEqual(pairwise_geometry([m],[],120,80)['ious'],[[]])

    def test_self_crossing_and_invalid_shapes(self):
        self.assertTrue(self_intersects([[0,0],[10,10],[0,10],[10,0]]))
        self.assertFalse(self_intersects([[0,0],[10,0],[10,10],[0,10]]))
        with self.assertRaises(ValueError):regularize_masks([np.zeros((3,4)),np.zeros((4,3))])


if __name__=='__main__':unittest.main()
