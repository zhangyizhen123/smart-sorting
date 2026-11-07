"""Native-coordinate inference for optional, provenance-checked processing stages.

No deployed launcher enables this pipeline yet. Dataset labels and evaluation
modules are deliberately not imported by inference code.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import image_guided_masks
import shape_verifier
from mask_display import native_mask_geometry
from shape_refiner import NAMES, PairwiseRefiner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_asset(root, relative, expected):
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Pipeline assets must stay inside the bundle')
    if digest(path) != expected:
        raise ValueError('Pipeline asset hash mismatch')
    return path


def apply_verifier(predictions, masks, probabilities, policy):
    """Keep prediction identities/masks aligned when the verifier rejects an ROI."""
    if len(predictions) != len(masks) or len(predictions) != len(probabilities):
        raise ValueError('Prediction/mask/probability alignment mismatch')
    result, retained = [], []
    for index, (prediction, mask, probability) in enumerate(zip(predictions, masks, probabilities, strict=True)):
        p = np.asarray(probability)
        if p.shape != (10,) or not np.isfinite(p).all():
            raise ValueError('Invalid verifier probability vector')
        if p[9] >= policy['background_probability']:
            continue
        item = copy.deepcopy(prediction)
        winner = int(p[:9].argmax())
        if item['confidence'] < policy['maximum_base_confidence'] and p[winner] >= policy['class_probability']:
            item['class_id'] = winner
        item['verifier_class_probability'] = float(p[winner])
        item['verifier_background_probability'] = float(p[9])
        item['source_prediction_index'] = index
        result.append(item);retained.append(mask)
    return result, retained


class NativeShapePipeline:
    def __init__(self, manifest_path):
        cv2.setNumThreads(1)  # Match the audited native-image processing profile.
        manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('version') != 2:
            raise ValueError('Expected native-processing pipeline version 2')
        root = manifest_path.parent
        if self.manifest.get('runtime_sha256') != digest(Path(__file__)):
            raise ValueError('Evaluated inference runtime changed')
        model = checked_asset(root, self.manifest['model'], self.manifest['model_sha256'])
        self.model_hash = digest(model);self.pipeline_hash = digest(manifest_path)
        self.model = YOLO(str(model))
        if self.model.task != 'segment' or list(self.model.names.values()) != NAMES:
            raise ValueError('Expected the nine-class segmentation model')
        self.settings = dict(self.manifest['settings'])
        if self.settings.get('retina_masks') is not True:
            raise ValueError('Native mask coordinates are required')
        self.refiner = None
        if self.manifest.get('refiner'):
            directory = (root/self.manifest['refiner']).resolve()
            checked_asset(root, str(directory/'refiner.pt'), self.manifest['refiner_sha256'])
            checked_asset(root, str(directory/'metadata.json'), self.manifest['refiner_metadata_sha256'])
            self.refiner = PairwiseRefiner(directory)
            if not self.refiner.ready:
                raise ValueError('Incomplete class refiner')
        postprocess = self.manifest.get('postprocess', {})
        if set(postprocess)-{'mask_refinement', 'verifier'}:
            raise ValueError('Unknown processing stage')
        self.mask_refinement = postprocess.get('mask_refinement')
        if self.mask_refinement:
            cfg = self.mask_refinement
            if cfg['method'] != 'rgb_grabcut_v1' or cfg['settings'] != image_guided_masks.DEFAULTS:
                raise ValueError('This runtime supports only the audited contour profile')
            if cfg['algorithm_sha256'] != digest(Path(image_guided_masks.__file__)):
                raise ValueError('Evaluated contour algorithm changed')
        self.verifier = None;self.verifier_policy = None
        if postprocess.get('verifier'):
            cfg = postprocess['verifier']
            if cfg['features_sha256'] != digest(Path(shape_verifier.__file__)):
                raise ValueError('Evaluated verifier features changed')
            path = checked_asset(root, cfg['model'], cfg['model_sha256'])
            self.verifier = shape_verifier.ShapeVerifier(path)
            self.verifier_policy = cfg['policy']
            required = {'maximum_base_confidence', 'class_probability', 'background_probability'}
            if set(self.verifier_policy) != required or any(not np.isfinite(v) or not 0 <= v <= 1.01 for v in self.verifier_policy.values()):
                raise ValueError('Invalid verifier policy')

    def predict_native(self, image, confidence=None):
        """Return exact classes/confidences, pixel boxes and binary native masks."""
        rgb = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected an RGB uint8 image')
        settings = dict(self.settings)
        if confidence is not None:
            settings['conf'] = confidence
        # PIL explicitly defines RGB; ndarray input to the detector uses BGR.
        result = self.model.predict(source=Image.fromarray(rgb), verbose=False, **settings)[0]
        predictions, masks = [], []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None or len(result.masks) != len(result.boxes):
                raise ValueError('Native inference is missing instance masks')
            if tuple(result.masks.data.shape[-2:]) != tuple(rgb.shape[:2]):
                raise ValueError('Native inference coordinate mismatch')
            polygons = result.masks.xy
            for j, mask in enumerate(result.masks.data.cpu().numpy()):
                base, confidence_value = int(result.boxes.cls[j]), float(result.boxes.conf[j])
                cls = self.refiner.refine(rgb, polygons[j], base, confidence_value) if self.refiner else base
                predictions.append({'class_id':int(cls), 'base_class_id':base, 'confidence':confidence_value,
                                    'box':result.boxes.xyxy[j].cpu().tolist(), 'source_prediction_index':j})
                masks.append(np.ascontiguousarray(mask > 0, dtype=np.uint8))
        originals = [m.copy() for m in masks]
        if self.mask_refinement:
            masks, info = image_guided_masks.refine_masks(rgb, masks, self.mask_refinement['settings'])
            for item, details in zip(predictions, info, strict=True):
                item['contour_refinement'] = details
        if self.verifier and predictions:
            values = np.stack([shape_verifier.features(rgb, mask) for mask in masks])
            probabilities = self.verifier.probabilities(values)
            predictions, masks = apply_verifier(predictions, masks, probabilities, self.verifier_policy)
        return predictions, masks, originals, settings

    def predict(self, image, confidence=None, *, include_mask_geometry=False):
        predictions, masks, originals, settings = self.predict_native(image, confidence)
        width, height = image.size
        result = []
        for prediction, mask in zip(predictions, masks, strict=True):
            item = copy.deepcopy(prediction)
            # Rounding belongs only to presentation, after every decision.
            item['confidence'] = round(item['confidence'], 6)
            item['box'] = [round(float(v)/s, 6) for v, s in zip(item['box'], [width,height,width,height], strict=True)]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest = max(contours, key=cv2.contourArea) if contours else np.empty((0,1,2))
            item['polygon'] = [[round(float(x)/width, 6),round(float(y)/height, 6)] for x,y in largest.reshape(-1,2)]
            item['refined'] = item['class_id'] != item['base_class_id']
            if include_mask_geometry:
                item.update(native_mask_geometry(mask))
                original = native_mask_geometry(originals[item['source_prediction_index']])
                item['model_mask_contours'] = original['mask_contours']
                item['contour_method'] = 'rgb_guided' if self.mask_refinement else 'native_model'
            result.append(item)
        return result, settings
