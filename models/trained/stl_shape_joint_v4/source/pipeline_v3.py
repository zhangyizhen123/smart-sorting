"""Hash-bound composition of native inference, detector agreement and foreground filtering."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import consensus_shapes
import foreground_gate
import shape_verifier
from pipeline_v2 import NativeShapePipeline,checked_asset,digest


def reindex_native(predictions,masks,originals,indices):
    """Select outputs while preserving each prediction's true original detector mask."""
    selected=[];effective=[];native=[]
    for j in indices:
        item=copy.deepcopy(predictions[j])
        original_index=item['source_prediction_index']
        native.append(originals[original_index])
        effective.append(masks[j])
        item['native_source_prediction_index']=item.get('native_source_prediction_index',original_index)
        item['source_prediction_index']=len(selected)
        selected.append(item)
    return selected,effective,native


class RecipePipeline:
    def __init__(self,path,*,_depth=0):
        if _depth>8:
            raise ValueError('Inference recipe nesting exceeds limit')
        path=Path(path).resolve();root=path.parent
        self.manifest=json.loads(path.read_text(encoding='utf-8'))
        if self.manifest.get('version')!=3 or self.manifest.get('runtime_sha256')!=digest(Path(__file__)):
            raise ValueError('Recipe runtime version/hash mismatch')
        self.pipeline_hash=digest(path);self.operation=self.manifest['operation'];self.children={}
        for name,child in self.manifest['sources'].items():
            manifest=checked_asset(root,child['pipeline'],child['pipeline_sha256'])
            version=json.loads(manifest.read_text(encoding='utf-8')).get('version')
            self.children[name]=NativeShapePipeline(manifest) if version==2 else RecipePipeline(manifest,_depth=_depth+1)
        if self.operation=='agreement':
            if set(self.children)!={'primary','secondary'}:
                raise ValueError('Agreement needs exactly two sources')
            if self.manifest['algorithm_sha256']!=digest(Path(consensus_shapes.__file__)):
                raise ValueError('Agreement algorithm changed')
            self.policy=self.manifest['policy'];primary=self.children['primary'];self.foreground=None
        elif self.operation=='foreground':
            if set(self.children)!={'source'}:
                raise ValueError('Foreground filtering needs exactly one source')
            cfg=self.manifest['foreground']
            if cfg['algorithm_sha256']!=digest(Path(foreground_gate.__file__)) or cfg['features_sha256']!=digest(Path(shape_verifier.__file__)):
                raise ValueError('Foreground inference algorithm changed')
            if not np.isfinite(cfg['threshold']) or not 0<cfg['threshold']<1:
                raise ValueError('Invalid foreground threshold')
            model=checked_asset(root,cfg['model'],cfg['model_sha256'])
            self.foreground=foreground_gate.ForegroundGate(model);self.threshold=cfg['threshold'];primary=self.children['source']
        else:
            raise ValueError('Unknown inference operation')
        self.model=primary.model;self.model_hash=primary.model_hash
        self.settings=dict(primary.settings);self.refiner=primary.refiner
        self.mask_refinement=any(bool(c.mask_refinement) for c in self.children.values())
        self.verifier=next((c.verifier for c in self.children.values() if c.verifier is not None),None)

    def predict_native(self,image,confidence=None):
        if self.operation=='agreement':
            results={k:v.predict_native(image,confidence) for k,v in self.children.items()}
            a,b=results['primary'],results['secondary']
            selection=consensus_shapes.select_predictions(a[0],b[0],self.policy)
            predictions=[];masks=[];originals=[]
            for item in selection:
                source=results[item['consensus_source']];index=item['consensus_source_index']
                original_index=source[0][index]['source_prediction_index']
                originals.append(source[2][original_index]);masks.append(source[1][index])
                item['native_source_prediction_index']=source[0][index].get('native_source_prediction_index',original_index)
                item['source_prediction_index']=len(predictions);predictions.append(item)
            return predictions,masks,originals,a[3]
        predictions,masks,originals,settings=self.children['source'].predict_native(image,confidence)
        if not predictions:
            return [],[],[],settings
        rgb=np.asarray(image.convert('RGB'))
        values=np.stack([shape_verifier.features(rgb,mask) for mask in masks])
        probabilities=self.foreground.probabilities(values)
        indices=[i for i,p in enumerate(probabilities) if p>=self.threshold]
        selected,effective,native=reindex_native(predictions,masks,originals,indices)
        for item,index in zip(selected,indices,strict=True):
            item['foreground_probability']=float(probabilities[index]);item['foreground_source_index']=index
        return selected,effective,native,settings

    def predict(self,image,confidence=None,*,include_mask_geometry=False):
        return NativeShapePipeline.predict(self,image,confidence,include_mask_geometry=include_mask_geometry)
