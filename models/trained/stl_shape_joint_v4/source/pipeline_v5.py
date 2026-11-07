"""Reproducible detector-evidence protection with class-only specialist review."""
import json
from pathlib import Path
import numpy as np
import consensus_evidence
import foreground_gate
import protected_foreground
import shape_verifier
import specialist_shapes
from pipeline_v2 import NativeShapePipeline,checked_asset,digest
from pipeline_v3 import RecipePipeline,reindex_native


class ProtectedPipeline:
    def __init__(self,path):
        path=Path(path).resolve();root=path.parent
        self.manifest=json.loads(path.read_text(encoding='utf-8'))
        if self.manifest.get('version')!=5 or self.manifest.get('runtime_sha256')!=digest(Path(__file__)):
            raise ValueError('Protected pipeline version/hash mismatch')
        for module in [consensus_evidence,foreground_gate,protected_foreground,shape_verifier,specialist_shapes]:
            if self.manifest['algorithm_sha256'][Path(module.__file__).name]!=digest(Path(module.__file__)):
                raise ValueError('Protected pipeline algorithm changed')
        sources=self.manifest['sources']
        if set(sources)!={'base','specialist'}: raise ValueError('Expected exactly two inference sources')
        source=lambda key:checked_asset(root,sources[key]['pipeline'],sources[key]['pipeline_sha256'])
        self.base=RecipePipeline(source('base'));self.specialist=NativeShapePipeline(source('specialist'))
        if self.base.operation!='agreement': raise ValueError('Protection requires unfiltered detector agreement')
        cfg=self.manifest['foreground'];self.foreground=foreground_gate.ForegroundGate(checked_asset(root,cfg['model'],cfg['model_sha256']))
        self.policy=self.manifest['policy'];self.pipeline_hash=digest(path)
        self.children={'base':self.base,'specialist':self.specialist}
        self.model=self.base.model;self.model_hash=self.base.model_hash;self.settings=dict(self.base.settings)
        self.refiner=self.base.refiner;self.mask_refinement=self.base.mask_refinement;self.verifier=self.base.verifier

    def predict_native(self,image,confidence=None):
        components={key:child.predict_native(image,confidence) for key,child in self.base.children.items()}
        predictions,masks,originals,settings=consensus_evidence.combine_native(components,self.base.policy)
        predictions=consensus_evidence.reclassify(predictions,self.policy['secondary_margin'])
        expert=self.specialist.predict_native(image,confidence)
        if not predictions: return [],[],[],settings
        rgb=np.asarray(image.convert('RGB'))
        probabilities=self.foreground.probabilities(np.stack([shape_verifier.features(rgb,m) for m in masks]))
        indices,protected=protected_foreground.retained_indices(predictions,probabilities,self.policy['foreground'])
        selected,effective,native=reindex_native(predictions,masks,originals,indices)
        for item,index,rescued in zip(selected,indices,protected,strict=True):
            item.update(foreground_probability=float(probabilities[index]),foreground_source_index=index,foreground_protected=rescued)
        selected=specialist_shapes.consult(selected,expert[0],self.policy['specialist'])
        return selected,effective,native,settings

    def predict(self,image,confidence=None,*,include_mask_geometry=False):
        return NativeShapePipeline.predict(self,image,confidence,include_mask_geometry=include_mask_geometry)
