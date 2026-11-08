"""Label a new capture session: drafts -> human review in a local page -> training/evaluation export (D-39).

A session is a folder of camera photos plus its manifest, as produced by the MaixCAM capture tool:
    dataset/raw_sessions/<session>/*.jpg
    dataset/raw_sessions/<session>.manifest.json   (sha256 per image, session group such as tray_v1)

Commands (run from the project root with the project venv):
    python vision/scripts/session_labeling.py draft  --session pilot_s01
    python vision/scripts/session_labeling.py review --session pilot_s01      # opens a local page
    python vision/scripts/session_labeling.py status --session pilot_s01
    python vision/scripts/session_labeling.py export --sessions pilot_s01 --split train --out dataset/suites/pilot_train_v1

Drafts come from the current default pipeline (V5) for boxes and classes, with MobileSAM outlining each box.
Drafts are never truth: only instances a person confirms in the review page are exported, with the class
the person chose. Every exported label is the reviewed mask converted with the same round-trip check used for
the earlier real photos (prepare_real_training.mask_polygon).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from PIL import Image

from prepare_real_training import mask_polygon

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset/raw_sessions"
WORK = ROOT / "dataset/sessions"
SAM_WEIGHTS = ROOT / "dataset/tools/mobile_sam.pt"
SAM_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
V5_PIPELINE = ROOT / "models/trained/stl_shape_joint_v4/pipeline.json"
TEMPLATE = ROOT / "vision/templates/session_review.html"
NAMES = ["cylinder", "cube", "cuboid", "tri_prism", "pent_prism", "hex_prism", "cone", "tetrahedron", "sphere"]
CHINESE = ["圆柱体", "正方体", "长方体", "三棱柱", "五棱柱", "六棱柱", "圆锥体", "四面体", "球体"]
NOT_BLOCK = "not_block"
MASK_KINDS = ("sam", "model")
# Each pixel shows at most one block, so neighbouring outlines only touch at the edge (pilot_s01: at most 1.2% of
# the smaller outline). More than this means an outline swallowed the block behind it, or one block was confirmed
# twice (pilot_s01 frames 14 and 26 before the v2 fix: 99-100%).
MAX_MASK_OVERLAP = 0.05


# ----------------------------------------------------------------------------- small helpers


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_mask(mask) -> np.ndarray | None:
    """Keep the largest 8-connected region and fill its interior holes. Blocks are solid objects, so a hole
    or a detached speck in a draft mask is an artefact; the cleaned mask is what the reviewer sees and approves."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not binary.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == keep).astype(np.uint8)
    padded = np.pad(region, 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (region.astype(bool) | holes)


def contour_points(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(int).tolist()


def sam_box_fit(mask: np.ndarray, box: list[float], pad_fraction: float = 0.1) -> tuple[float, float]:
    """How well a SAM outline fits the box it was prompted with: share of the outline inside the (slightly
    padded) box, and outline area relative to the box. SAM is judged against its own prompt, not against the
    model mask, because the model mask is the one that more often merges neighbours or drops a face."""
    height, width = mask.shape
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * pad_fraction, (y2 - y1) * pad_fraction
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(np.ceil(x2 + pad_x))), min(height, int(np.ceil(y2 + pad_y)))
    area = int(mask.sum())
    if area == 0:
        return 0.0, 0.0
    inside = int(mask[top:bottom, left:right].sum()) / area
    fill = area / max(1.0, (x2 - x1) * (y2 - y1))
    return inside, fill


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def overlap_problems(masks: list[tuple[str, np.ndarray]], max_overlap: float = MAX_MASK_OVERLAP) -> list[dict]:
    """Pairs of block outlines in one image sharing more than max_overlap of the smaller outline."""
    areas = [int(np.count_nonzero(mask)) for _id, mask in masks]
    problems = []
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            shared = int(np.count_nonzero(masks[a][1] & masks[b][1]))
            smaller = min(areas[a], areas[b])
            fraction = shared / smaller if smaller else 0.0
            if fraction > max_overlap:
                problems.append({"ids": [masks[a][0], masks[b][0]], "pixels": shared, "fraction": round(fraction, 4)})
    return problems


def save_mask(mask: np.ndarray, path: Path) -> dict:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    ys, xs = np.nonzero(mask)
    return {"file": path.name, "sha256": sha256_bytes(data), "area": int(mask.sum()),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "contour": contour_points(mask)}


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


# ----------------------------------------------------------------------------- session files


class Session:
    """Paths and verified inputs for one capture session."""

    def __init__(self, name: str, raw: Path = RAW, work: Path = WORK):
        if not name or any(c in name for c in "/\\:") or name.startswith("."):
            raise ValueError(f"Invalid session name: {name!r}")
        self.name = name
        self.raw_dir = raw / name
        self.manifest_path = raw / f"{name}.manifest.json"
        self.work = work / name
        self.masks = self.work / "masks"
        self.drafts = self.work / "drafts"
        self.manual = self.work / "manual"
        self.edits = self.work / "edits"
        self.reviews = self.work / "reviews"
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("session") != name:
            raise ValueError("Manifest session name does not match")
        self.images = {row["file"]: row for row in self.manifest["images"]}

    @property
    def group(self) -> str:
        return self.manifest.get("group", self.name)

    def image_path(self, file: str) -> Path:
        if file not in self.images:
            raise KeyError(f"Unknown image {file}")
        return self.raw_dir / file

    def verified_image(self, file: str) -> Image.Image:
        path = self.image_path(file)
        data = path.read_bytes()
        if sha256_bytes(data) != self.images[file]["sha256"]:
            raise ValueError(f"Raw image changed since capture: {file}")
        return Image.open(io.BytesIO(data)).convert("RGB")

    def stem(self, file: str) -> str:
        return Path(file).stem

    def draft(self, file: str) -> dict:
        return read_json(self.drafts / f"{self.stem(file)}.json")

    def manual_instances(self, file: str) -> list[dict]:
        path = self.manual / f"{self.stem(file)}.json"
        return read_json(path)["instances"] if path.exists() else []

    def review(self, file: str) -> dict | None:
        path = self.reviews / f"{self.stem(file)}.json"
        return read_json(path) if path.exists() else None

    def outline_edits(self, file: str) -> dict:
        path = self.edits / f"{self.stem(file)}.json"
        return read_json(path)["edits"] if path.exists() else {}

    def instance_list(self, file: str) -> list[dict]:
        """Draft instances, then manual ones, each with any redrawn or hand-traced outlines added as options."""
        edits = self.outline_edits(file)
        merged = []
        for instance in self.draft(file)["instances"] + self.manual_instances(file):
            instance = dict(instance, masks=dict(instance["masks"]))
            instance["masks"].update(edits.get(instance["id"], {}))
            merged.append(instance)
        return merged

    def instances(self, file: str) -> dict[str, dict]:
        return {i["id"]: i for i in self.instance_list(file)}


# ----------------------------------------------------------------------------- drafting


def v5_predictor(confidence: float):
    """Default pipeline as a draft source: returns [(class_id, confidence, box_xyxy, mask_uint8), ...]."""
    import torch
    import ultralytics.utils.torch_utils as torch_utils
    from pipeline_v5 import ProtectedPipeline
    torch.set_num_threads(7)
    torch_utils.NUM_THREADS = 7
    pipeline = ProtectedPipeline(V5_PIPELINE)

    def predict(image: Image.Image):
        predictions, masks, _native, _settings = pipeline.predict_native(image, confidence)
        return [(int(p["class_id"]), float(p["confidence"]), [float(v) for v in p["box"]], np.asarray(m))
                for p, m in zip(predictions, masks, strict=True)]

    predict.identity = {"pipeline": V5_PIPELINE.relative_to(ROOT).as_posix(), "pipeline_sha256": sha256_file(V5_PIPELINE),
                        "confidence": confidence}
    return predict


def sam_segmenter(weights: Path = SAM_WEIGHTS):
    """MobileSAM prompted by boxes: returns one boolean mask per box."""
    if sha256_file(weights) != SAM_SHA256:
        raise ValueError("MobileSAM weights do not match the recorded SHA-256")
    from ultralytics import SAM
    model = SAM(str(weights))

    def segment(rgb: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        if not boxes:
            return []
        result = model.predict(source=rgb[..., ::-1].copy(), bboxes=boxes, verbose=False)[0]
        masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else np.zeros((0,) + rgb.shape[:2], bool)
        if len(masks) != len(boxes):
            raise ValueError("MobileSAM returned a different number of masks than boxes")
        return list(masks)

    segment.identity = {"weights": weights.name, "sha256": SAM_SHA256}
    return segment


def describe_instance(session: Session, file: str, instance_id: str, origin: str, masks: dict[str, np.ndarray],
                      **extra) -> dict:
    record = {"id": instance_id, "origin": origin, "masks": {}, **extra}
    for kind, mask in masks.items():
        record["masks"][kind] = save_mask(mask, session.masks / f"{instance_id}_{kind}.png")
    return record


def draft_session(session: Session, predictor, segmenter, *, min_inside: float = 0.9, min_fill: float = 0.15) -> dict:
    if session.drafts.exists() and any(session.drafts.iterdir()):
        raise FileExistsError(f"Drafts already exist for {session.name}; they are kept so reviews stay valid")
    summary = {"images": 0, "instances": 0, "sam_default": 0, "model_default": 0, "sam_failed": 0}
    for file in session.images:
        image = session.verified_image(file)
        rgb = np.asarray(image)
        predictions = predictor(image)
        boxes = [p[2] for p in predictions]
        sam_masks = segmenter(rgb, boxes) if boxes else []
        instances = []
        for k, ((cls, conf, box, model_mask), sam_mask) in enumerate(zip(predictions, sam_masks, strict=True)):
            model_clean = clean_mask(model_mask)
            sam_clean = clean_mask(sam_mask)
            masks = {}
            if sam_clean is not None:
                masks["sam"] = sam_clean
            if model_clean is not None:
                masks["model"] = model_clean
            if not masks:
                continue
            agreement = mask_iou(sam_clean, model_clean) if sam_clean is not None and model_clean is not None else 0.0
            inside, fill = sam_box_fit(sam_clean, box) if sam_clean is not None else (0.0, 0.0)
            sam_ok = sam_clean is not None and inside >= min_inside and fill >= min_fill
            default = "sam" if sam_ok or model_clean is None else "model"
            summary["sam_default" if default == "sam" else "model_default"] += 1
            summary["sam_failed"] += sam_clean is None
            instances.append(describe_instance(
                session, file, f"{session.stem(file)}_{k + 1:02d}", "draft", masks,
                draft_class=NAMES[cls], draft_confidence=round(conf, 4), draft_box=[round(v, 1) for v in box],
                sam_model_iou=round(agreement, 4), sam_inside_box=round(inside, 4), sam_box_fill=round(fill, 4),
                default_mask=default))
        write_json(session.drafts / f"{session.stem(file)}.json",
                   {"image": file, "image_sha256": session.images[file]["sha256"], "instances": instances})
        summary["images"] += 1
        summary["instances"] += len(instances)
    write_json(session.work / "draft_run.json", {
        "session": session.name, "group": session.group, "created_at": now(),
        "raw_manifest_sha256": sha256_file(session.manifest_path),
        "predictor": getattr(predictor, "identity", "custom"), "segmenter": getattr(segmenter, "identity", "custom"),
        "default_rule": {"sam_when_inside_box_at_least": min_inside, "and_box_fill_at_least": min_fill,
                         "otherwise": "model outline"},
        "summary": summary,
        "note": "Drafts only. Classes and outlines become labels only after human review."})
    return summary


# ----------------------------------------------------------------------------- review


def add_manual_instance(session: Session, file: str, box: list[float], segmenter) -> dict:
    """Outline a block the draft missed: the reviewer draws a box, MobileSAM outlines it."""
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("Box must be [x1, y1, x2, y2] with positive size")
    image = session.verified_image(file)
    width, height = image.size
    box = [max(0.0, min(float(box[0]), width)), max(0.0, min(float(box[1]), height)),
           max(0.0, min(float(box[2]), width)), max(0.0, min(float(box[3]), height))]
    mask = clean_mask(segmenter(np.asarray(image), [box])[0])
    if mask is None:
        raise ValueError("No object outline found inside that box; draw a slightly larger box")
    path = session.manual / f"{session.stem(file)}.json"
    existing = read_json(path)["instances"] if path.exists() else []
    instance = describe_instance(session, file, f"{session.stem(file)}_m{len(existing) + 1:02d}", "manual",
                                 {"sam": mask}, prompt_box=[round(v, 1) for v in box], default_mask="sam")
    write_json(path, {"image": file, "instances": existing + [instance]})
    return instance


def add_outline_option(session: Session, file: str, instance_id: str, kind: str, mask: np.ndarray) -> dict:
    """Store a reviewer-made outline as a new option of an existing instance; earlier files are never overwritten."""
    if kind not in {"redo", "polygon"}:
        raise ValueError("Outline kind must be redo or polygon")
    instances = session.instances(file)
    if instance_id not in instances:
        raise ValueError(f"Unknown instance id: {instance_id}")
    path = session.edits / f"{session.stem(file)}.json"
    edits = read_json(path)["edits"] if path.exists() else {}
    version = sum(1 for p in session.masks.glob(f"{instance_id}_{kind}*.png")) + 1
    info = save_mask(mask, session.masks / f"{instance_id}_{kind}{version}.png")
    edits.setdefault(instance_id, {})[kind] = info
    write_json(path, {"image": file, "edits": edits})
    return session.instances(file)[instance_id]


def reoutline_instance(session: Session, file: str, instance_id: str, box: list[float], segmenter) -> dict:
    """Redraw an existing instance's outline from a tighter box (MobileSAM)."""
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("Box must be [x1, y1, x2, y2] with positive size")
    image = session.verified_image(file)
    width, height = image.size
    box = [max(0.0, min(float(box[0]), width)), max(0.0, min(float(box[1]), height)),
           max(0.0, min(float(box[2]), width)), max(0.0, min(float(box[3]), height))]
    mask = clean_mask(segmenter(np.asarray(image), [box])[0])
    if mask is None:
        raise ValueError("No object outline found inside that box; draw a slightly larger box or trace it by hand")
    return add_outline_option(session, file, instance_id, "redo", mask)


def trace_instance(session: Session, file: str, instance_id: str, points: list[list[float]]) -> dict:
    """Hand-traced outline: the reviewer clicks the corners of the block, in order."""
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Click at least three corners to trace an outline")
    image = session.verified_image(file)
    width, height = image.size
    polygon = np.array([[min(max(float(x), 0.0), width - 1), min(max(float(y), 0.0), height - 1)] for x, y in points])
    raster = np.zeros((height, width), np.uint8)
    cv2.fillPoly(raster, [np.rint(polygon).astype(np.int32)], 1)
    mask = clean_mask(raster)
    if mask is None or mask.sum() < 20:
        raise ValueError("The traced outline is too small")
    return add_outline_option(session, file, instance_id, "polygon", mask)


def save_review(session: Session, file: str, status: str, decisions: list[dict], notes: str = "") -> dict:
    """Record one image's human decision. Every outlined instance must be decided: a block class or not_block."""
    if file not in session.images:
        raise ValueError(f"Unknown image {file}")
    if status not in {"approved", "unusable"}:
        raise ValueError("Status must be approved or unusable")
    instances = session.instances(file)
    by_id = {}
    for decision in decisions:
        instance_id = decision.get("id")
        if instance_id not in instances or instance_id in by_id:
            raise ValueError(f"Unknown or repeated instance id: {instance_id}")
        by_id[instance_id] = decision
    if status == "approved" and set(by_id) != set(instances):
        raise ValueError("Every outlined instance needs a decision before approval")
    recorded = []
    for instance_id, decision in by_id.items():
        class_name = decision.get("class_name")
        if class_name not in NAMES and class_name != NOT_BLOCK:
            raise ValueError(f"Unknown class for {instance_id}: {class_name}")
        entry = {"id": instance_id, "class_name": class_name}
        if class_name != NOT_BLOCK:
            kind = decision.get("mask", instances[instance_id]["default_mask"])
            if kind not in instances[instance_id]["masks"]:
                raise ValueError(f"Mask {kind} is not available for {instance_id}")
            mask_info = instances[instance_id]["masks"][kind]
            if sha256_file(session.masks / mask_info["file"]) != mask_info["sha256"]:
                raise ValueError(f"Mask file changed: {mask_info['file']}")
            entry.update(mask=kind, mask_file=mask_info["file"], mask_sha256=mask_info["sha256"],
                         draft_class=instances[instance_id].get("draft_class"))
        recorded.append(entry)
    review = {"image": file, "image_sha256": session.images[file]["sha256"], "status": status,
              "instances": recorded, "notes": str(notes)[:2000], "reviewed_at": now(), "reviewer": "human"}
    write_json(session.reviews / f"{session.stem(file)}.json", review)
    return review


def review_overlaps(session: Session, file: str, review: dict | None = None,
                    max_overlap: float = MAX_MASK_OVERLAP) -> list[dict]:
    """Overlapping confirmed blocks in one approved review (see MAX_MASK_OVERLAP)."""
    review = session.review(file) if review is None else review
    if not review or review["status"] != "approved":
        return []
    masks = [(entry["id"], load_mask(session.masks / entry["mask_file"]))
             for entry in review["instances"] if entry["class_name"] != NOT_BLOCK]
    return overlap_problems(masks, max_overlap)


def status(session: Session, overlaps: bool = False) -> dict:
    counts = {"images": len(session.images), "approved": 0, "unusable": 0, "pending": 0,
              "blocks": 0, "not_block": 0, "class_changed_from_draft": 0}
    if overlaps:
        counts["overlap_problems"] = [{"image": file, **problem} for file in session.images
                                      for problem in review_overlaps(session, file)]
    for file in session.images:
        review = session.review(file)
        if review is None:
            counts["pending"] += 1
            continue
        counts[review["status"]] += 1
        for entry in review["instances"]:
            if entry["class_name"] == NOT_BLOCK:
                counts["not_block"] += 1
            else:
                counts["blocks"] += 1
                counts["class_changed_from_draft"] += bool(entry.get("draft_class")) and entry["class_name"] != entry["draft_class"]
    return counts


def page_payload(session: Session) -> dict:
    images = []
    for file in session.images:
        review = session.review(file)
        images.append({"file": file, "instances": session.instance_list(file), "review": review,
                       "overlaps": review_overlaps(session, file, review)})
    return {"session": session.name, "group": session.group, "names": NAMES, "chinese": CHINESE,
            "not_block": NOT_BLOCK, "images": images}


def serve_review(session: Session, segmenter_factory, port: int = 8770, open_browser: bool = True) -> None:
    server = make_review_server(session, segmenter_factory, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Review page: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_review_server(session: Session, segmenter_factory, port: int = 8770):
    """Local-only review server (127.0.0.1); SAM loads on first manual box."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    template = TEMPLATE.read_text(encoding="utf-8")
    lock = threading.Lock()
    segmenter_holder = {}

    def segmenter():
        if "fn" not in segmenter_holder:
            segmenter_holder["fn"] = segmenter_factory()
        return segmenter_holder["fn"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code: int, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self) -> bool:
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in hosts or (origin is not None and origin != f"http://{host}"):
                self.reply(403, {"error": "请从本机审核页面访问。"})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = unquote(urlsplit(self.path).path)
            try:
                if path in {"/", "/index.html"}:
                    payload = json.dumps(page_payload(session), ensure_ascii=False).replace("<", "\\u003c")
                    self.reply(200, template.replace("__PAYLOAD__", payload).encode("utf-8"), "text/html; charset=utf-8")
                elif path.startswith("/image/"):
                    self.reply(200, session.image_path(path[len("/image/"):]).read_bytes(), "image/jpeg")
                else:
                    self.reply(404, {"error": "没有这个页面。"})
            except (KeyError, ValueError) as exc:
                self.reply(404, {"error": str(exc)})

        def do_POST(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Request too large or empty")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                with lock:
                    if path == "/api/segment":
                        self.reply(200, add_manual_instance(session, body["image"], body["box"], segmenter()))
                    elif path == "/api/reoutline":
                        self.reply(200, reoutline_instance(session, body["image"], body["id"], body["box"], segmenter()))
                    elif path == "/api/trace":
                        self.reply(200, trace_instance(session, body["image"], body["id"], body["points"]))
                    elif path == "/api/review":
                        review = save_review(session, body["image"], body["status"], body.get("instances", []),
                                             body.get("notes", ""))
                        self.reply(200, {"review": review, "status": status(session),
                                         "overlaps": review_overlaps(session, body["image"], review)})
                    else:
                        self.reply(404, {"error": "没有这个接口。"})
            except (KeyError, ValueError, TypeError, FileNotFoundError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


# ----------------------------------------------------------------------------- export


def export_sessions(sessions: list[Session], out: Path, split: str, *, allow_pending: bool = False,
                    final_test_eligible: bool = False, max_overlap: float = MAX_MASK_OVERLAP) -> dict:
    """Write reviewed blocks as YOLO segmentation labels, in the frozen-suite layout run_experiment.py reads.
    Everything is checked before the first file is written, so a refused export leaves no partial suite."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Split must be train, val or test")
    if not 0 <= max_overlap < 0.5:
        raise ValueError("max_overlap must be in [0, 0.5)")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty export; use a new name")
    rows, skipped, pending = [], [], []
    plans = []
    for session in sessions:
        for file in session.images:
            review = session.review(file)
            if review is None:
                pending.append(f"{session.name}/{file}")
                continue
            if review["image_sha256"] != session.images[file]["sha256"]:
                raise ValueError(f"Review refers to a different image version: {session.name}/{file}")
            if review["status"] != "approved":
                skipped.append(f"{session.name}/{file}")
                continue
            plans.append((session, file, review))
    if pending and not allow_pending:
        raise ValueError(f"{len(pending)} image(s) are not reviewed yet, e.g. {pending[:3]}")
    if not plans:
        raise ValueError("Nothing approved to export")
    overlapping = []
    for session, file, review in plans:
        for entry in review["instances"]:
            if entry["class_name"] != NOT_BLOCK and sha256_file(session.masks / entry["mask_file"]) != entry["mask_sha256"]:
                raise ValueError(f"Reviewed mask changed: {entry['mask_file']}")
        overlapping += [f"{session.name}/{file}: {' + '.join(p['ids'])} share {p['fraction']:.0%}"
                        for p in review_overlaps(session, file, review, max_overlap)]
    if overlapping:
        raise ValueError(f"{len(overlapping)} pair(s) of confirmed blocks overlap by more than {max_overlap:.0%} "
                         "(an outline swallowed another block, or one block was confirmed twice); fix them in the "
                         "review page: " + "; ".join(overlapping))
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    for session, file, review in plans:
        image_bytes = session.image_path(file).read_bytes()
        if sha256_bytes(image_bytes) != session.images[file]["sha256"]:
            raise ValueError(f"Raw image changed since capture: {session.name}/{file}")
        lines, classes = [], []
        for entry in review["instances"]:
            if entry["class_name"] == NOT_BLOCK:
                continue
            mask_path = session.masks / entry["mask_file"]
            if sha256_file(mask_path) != entry["mask_sha256"]:
                raise ValueError(f"Reviewed mask changed: {entry['mask_file']}")
            polygon, _iou = mask_polygon(load_mask(mask_path).astype(np.uint8))
            class_id = NAMES.index(entry["class_name"])
            classes.append(class_id)
            lines.append(f"{class_id} " + " ".join(f"{v:.8f}" for v in polygon.ravel()))
        key = f"{session.name}__{Path(file).stem}"
        image_dest = out / "images" / split / f"{key}.jpg"
        label_dest = out / "labels" / split / f"{key}.txt"
        image_dest.write_bytes(image_bytes)
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        rows.append({"key": key, "split": split, "group_id": session.name, "session": session.name,
                     "session_group": session.group, "image": image_dest.relative_to(out).as_posix(),
                     "label": label_dest.relative_to(out).as_posix(), "image_sha256": sha256_bytes(image_bytes),
                     "label_sha256": sha256_file(label_dest), "classes": classes,
                     "review_sha256": sha256_file(session.reviews / f"{Path(file).stem}.json"),
                     "final_test_eligible": final_test_eligible})
    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
    lock = {"version": 1, "task": "D-39", "split": split, "sessions": [s.name for s in sessions],
            "session_groups": sorted({s.group for s in sessions}), "manifest_sha256": sha256_file(manifest),
            "images": len(rows), "instances": sum(len(r["classes"]) for r in rows),
            "empty_images": sum(not r["classes"] for r in rows), "skipped_unusable": skipped,
            "pending_not_exported": pending, "final_test_eligible": final_test_eligible,
            "max_mask_overlap": max_overlap, "created_at": now()}
    write_json(out / "freeze.json", lock)
    data_yaml = (f"path: {out.resolve().as_posix()}\ntrain: images/{split}\nval: images/{split}\nnc: {len(NAMES)}\n"
                 "names: [" + ", ".join(NAMES) + "]\n")
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")
    return lock


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft")
    draft.add_argument("--session", required=True)
    draft.add_argument("--confidence", type=float, default=0.15, help="draft threshold; low on purpose, reviewers delete")
    review = commands.add_parser("review")
    review.add_argument("--session", required=True)
    review.add_argument("--port", type=int, default=8770)
    review.add_argument("--no-open", action="store_true")
    stat = commands.add_parser("status")
    stat.add_argument("--session", required=True)
    export = commands.add_parser("export")
    export.add_argument("--sessions", nargs="+", required=True)
    export.add_argument("--split", required=True, choices=["train", "val", "test"])
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--allow-pending", action="store_true")
    export.add_argument("--final-test-eligible", action="store_true")
    export.add_argument("--max-overlap", type=float, default=MAX_MASK_OVERLAP,
                        help="largest share of the smaller outline two confirmed blocks may share (default 0.05)")
    args = parser.parse_args()
    if args.command == "draft":
        print(json.dumps(draft_session(Session(args.session), v5_predictor(args.confidence), sam_segmenter()), ensure_ascii=False))
    elif args.command == "review":
        serve_review(Session(args.session), sam_segmenter, args.port, not args.no_open)
    elif args.command == "status":
        print(json.dumps(status(Session(args.session), overlaps=True), ensure_ascii=False))
    else:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        result = export_sessions([Session(s) for s in args.sessions], out, args.split,
                                 allow_pending=args.allow_pending, final_test_eligible=args.final_test_eligible,
                                 max_overlap=args.max_overlap)
        print(json.dumps({k: v for k, v in result.items() if k not in {"skipped_unusable", "pending_not_exported"}},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
