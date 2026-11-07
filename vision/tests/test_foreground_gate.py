import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from foreground_gate import DIMENSIONS,ForegroundNetwork,ForegroundGate,pooled_features
from shape_verifier import FEATURES


class ForegroundTests(unittest.TestCase):
    def test_spatial_pooling_preserves_channels_and_geometry(self):
        full=np.concatenate([np.full(1024,k,dtype=np.float32) for k in range(4)]+[np.arange(16,dtype=np.float32)])
        reduced=pooled_features(full)
        self.assertEqual(reduced.shape,(1,DIMENSIONS))
        for k in range(4):
            np.testing.assert_array_equal(reduced[0,k*64:(k+1)*64],k)
        np.testing.assert_array_equal(reduced[0,-16:],np.arange(16))

    def test_saved_runtime_matches_native_probability(self):
        torch.manual_seed(1);model=ForegroundNetwork().eval()
        features=np.random.default_rng(1).normal(size=(3,FEATURES)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(version=1,mean=torch.zeros(DIMENSIONS),scale=torch.ones(DIMENSIONS),state_dict=model.state_dict()),path)
            with torch.inference_mode():
                expected=torch.sigmoid(model(torch.from_numpy(pooled_features(features)).clamp(-6,6))).numpy()
            np.testing.assert_allclose(ForegroundGate(path).probabilities(features),expected)


if __name__=='__main__':
    unittest.main()
