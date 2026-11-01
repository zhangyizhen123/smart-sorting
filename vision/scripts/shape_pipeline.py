"""Load an evaluated detector/refiner bundle for the local image demo."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ultralytics import YOLO
from predict_stl_demo import NAMES, serialize_predictions
from shape_refiner import PairwiseRefiner


class ShapePipeline:
    def __init__(self, manifest_path):
        manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("version") != 1:
            raise ValueError("Unsupported pipeline manifest")
        root = manifest_path.parent
        model_path = (root / self.manifest["model"]).resolve()
        if not model_path.is_relative_to(root):
            raise ValueError("Bundle model must remain inside bundle")
        self.model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if self.model_hash != self.manifest["model_sha256"]:
            raise ValueError("Pipeline model hash mismatch")
        self.pipeline_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.model = YOLO(str(model_path))
        if self.model.task != "segment" or list(self.model.names.values()) != NAMES:
            raise ValueError("Expected nine-class segmentation model")
        self.settings = dict(self.manifest["settings"])
        self.refiner = None
        if self.manifest.get("refiner"):
            refiner_path = (root / self.manifest["refiner"]).resolve()
            if not refiner_path.is_relative_to(root):
                raise ValueError("Refiner must remain inside bundle")
            for name, field in (("refiner.pt", "refiner_sha256"), ("metadata.json", "refiner_metadata_sha256")):
                if hashlib.sha256((refiner_path / name).read_bytes()).hexdigest() != self.manifest.get(field):
                    raise ValueError("Refiner checksum mismatch")
            self.refiner = PairwiseRefiner(refiner_path)
            if not self.refiner.ready:
                raise ValueError("Incomplete refiner")

    def predict(self, image, confidence=None):
        settings = {**self.settings}
        if confidence is not None:
            settings["conf"] = confidence
        result = self.model.predict(source=image, verbose=False, **settings)[0]
        predictions = serialize_predictions(result)
        if self.refiner:
            for item, polygon in zip(predictions, result.masks.xy if result.masks is not None else []):
                item["base_class_id"] = item["class_id"]
                item["class_id"] = self.refiner.refine(np.asarray(image), polygon, item["class_id"], item["confidence"])
                item["refined"] = item["class_id"] != item["base_class_id"]
        return predictions, settings
