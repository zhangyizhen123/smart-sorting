"""Frozen composed inference with a class-only legacy shape specialist."""
import json
from pathlib import Path
import specialist_shapes
from pipeline_v2 import NativeShapePipeline,checked_asset,digest
from pipeline_v3 import RecipePipeline


class SpecialistPipeline:
    def __init__(self,path):
        path=Path(path).resolve();root=path.parent
        self.manifest=json.loads(path.read_text(encoding='utf-8'))
        if self.manifest.get('version')!=4 or self.manifest.get('runtime_sha256')!=digest(Path(__file__)):
            raise ValueError('Specialist runtime version/hash mismatch')
        if self.manifest.get('operation')!='specialist' or self.manifest['algorithm_sha256']!=digest(Path(specialist_shapes.__file__)):
            raise ValueError('Specialist algorithm mismatch')
        if set(self.manifest['sources'])!={'primary','secondary'}:
            raise ValueError('Specialist requires two source pipelines')
        self.children={}
        for key,value in self.manifest['sources'].items():
            source=checked_asset(root,value['pipeline'],value['pipeline_sha256'])
            version=json.loads(source.read_text(encoding='utf-8')).get('version')
            if version not in [2,3]:
                raise ValueError('Unsupported specialist source pipeline')
            self.children[key]=NativeShapePipeline(source) if version==2 else RecipePipeline(source)
        primary=self.children['primary']
        self.policy=self.manifest['policy'];self.pipeline_hash=digest(path)
        self.model=primary.model;self.model_hash=primary.model_hash;self.settings=dict(primary.settings)
        self.refiner=primary.refiner;self.mask_refinement=primary.mask_refinement;self.verifier=primary.verifier

    def predict_native(self,image,confidence=None):
        primary=self.children['primary'].predict_native(image,confidence)
        secondary=self.children['secondary'].predict_native(image,confidence)
        predictions=specialist_shapes.consult(primary[0],secondary[0],self.policy)
        return predictions,primary[1],primary[2],primary[3]

    def predict(self,image,confidence=None,*,include_mask_geometry=False):
        return NativeShapePipeline.predict(self,image,confidence,include_mask_geometry=include_mask_geometry)
