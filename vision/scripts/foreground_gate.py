"""Small foreground/background gate trained on native hard negative proposals."""
import numpy as np
import torch
from torch import nn
from shape_verifier import FEATURES

DIMENSIONS=4*8*8+16


def pooled_features(values):
    values=np.asarray(values,dtype=np.float32).reshape(-1,FEATURES)
    spatial=values[:,:4*32*32].reshape(-1,4,8,4,8,4).mean(axis=(3,5)).reshape(-1,4*8*8)
    return np.concatenate([spatial,values[:,-16:]],axis=1)


class ForegroundNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(DIMENSIONS,64),nn.ReLU(),nn.Dropout(.1),
                                  nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        return self.layers(x).squeeze(-1)


class ForegroundGate:
    def __init__(self,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['version']!=1:
            raise ValueError('Unsupported foreground gate')
        self.mean,self.scale=payload['mean'],payload['scale']
        self.model=ForegroundNetwork();self.model.load_state_dict(payload['state_dict']);self.model.eval()

    def probabilities(self,features):
        x=torch.from_numpy(pooled_features(features))
        x=((x-self.mean)/self.scale).clamp(-6,6)
        with torch.inference_mode():
            return torch.sigmoid(self.model(x)).numpy()
