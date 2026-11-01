#!/usr/bin/env python3
"""Serve the local demo and run the archived segmentation model on user images."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import socket
import threading
import time
import traceback
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from PIL import Image, ImageOps, UnidentifiedImageError

from predict_stl_demo import (
    CHINESE_NAMES, DEMO, MODEL, NAMES, PREDICTION_SETTINGS,
    digest, load_model, render_demo, serialize_predictions,
)


MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 1600


def decode_image(content: bytes) -> tuple[Image.Image, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP", "BMP"} or getattr(source, "n_frames", 1) != 1:
                raise ValueError("请选择静态 JPG、PNG、WebP 或 BMP 图片。")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("图片超过 4000 万像素，请先缩小后再导入。")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            original_size = oriented.size
            oriented.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
            return image, original_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取这张图片，请重新导出为 JPG 或 PNG 后重试。") from exc


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], model, page: bytes, model_hash: str, pipeline=None):
        super().__init__(address, DemoHandler)
        self.model = model
        self.page = page
        self.model_hash = model_hash
        self.pipeline = pipeline
        self.inference_lock = threading.Lock()


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status: int, payload, content_type: str = "application/json; charset=utf-8") -> None:
        content = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # The user may close the page during inference.

    def allowed_request(self) -> bool:
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in hosts or (origin is not None and origin != f"http://{host}"):
            self.reply(403, {"error": "请从本机演示窗口打开此页面。"})
            return False
        return True

    def do_GET(self) -> None:
        if not self.allowed_request():
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self.reply(200, self.server.page, "text/html; charset=utf-8")
        elif path == "/api/health":
            self.reply(200, {"ready": True, "model_sha256": self.server.model_hash, "names": NAMES, "chinese_names": CHINESE_NAMES, "max_bytes": MAX_BYTES,
                             "pipeline_sha256": self.server.pipeline.pipeline_hash if self.server.pipeline else None})
        elif path == "/favicon.ico":
            self.reply(204, b"", "image/x-icon")
        else:
            self.reply(404, {"error": "没有这个页面。"})

    def do_POST(self) -> None:
        if not self.allowed_request():
            return
        url = urlsplit(self.path)
        if url.path != "/api/predict":
            self.reply(404, {"error": "没有这个接口。"})
            return
        if self.headers.get_content_type() != "application/octet-stream":
            self.reply(415, {"error": "请使用页面中的选图按钮导入图片。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            confidence = float(parse_qs(url.query).get("confidence", [str(PREDICTION_SETTINGS["conf"])])[0])
        except (ValueError, IndexError):
            self.reply(400, {"error": "图片大小或置信度参数无效。"})
            return
        if not 0 < length <= MAX_BYTES:
            self.reply(413, {"error": "请选择不超过 20 MB 的图片。"})
            return
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            self.reply(400, {"error": "最低置信度需在 0.05～0.95 之间。"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(429, {"error": "模型正在处理上一张图片，请稍后再点识别。"})
            return
        try:
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("图片传输不完整，请重新选择并识别。")
            image, original_size = decode_image(content)
            settings = {**PREDICTION_SETTINGS, "conf": confidence}
            start = time.perf_counter()
            if self.server.pipeline:
                predictions, settings = self.server.pipeline.predict(image, confidence)
            else:
                result = self.server.model.predict(source=image, verbose=False, **settings)[0]
                predictions = serialize_predictions(result)
            preview = io.BytesIO()
            image.save(preview, format="PNG")
            filename = unquote(self.headers.get("X-Filename", "自选图片"))[:200]
            self.reply(200, {
                "id": "uploaded-image", "title": filename, "source": "upload",
                "width": image.width, "height": image.height,
                "original_width": original_size[0], "original_height": original_size[1],
                "image_data": "data:image/png;base64," + base64.b64encode(preview.getvalue()).decode("ascii"),
                "predictions": predictions, "settings": settings,
                "model_sha256": self.server.model_hash,
                "pipeline_sha256": self.server.pipeline.pipeline_hash if self.server.pipeline else None,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            })
        except ValueError as exc:
            self.reply(422, {"error": str(exc)})
        except socket.timeout:
            self.reply(408, {"error": "图片传输超时，请重试。"})
        except Exception:
            traceback.print_exc()
            self.reply(500, {"error": "识别失败，请查看启动窗口，或重新启动演示后重试。"})
        finally:
            self.server.inference_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--pipeline", help="Path to an evaluated pipeline.json; bundled examples remain the v1 reference")
    args = parser.parse_args()
    payload = json.loads((DEMO / "predictions.json").read_text(encoding="utf-8"))
    model_hash = digest(MODEL)
    if model_hash != payload["model_sha256"]:
        raise ValueError("Saved examples use a different model; regenerate them before starting the demo")
    print("Loading local shape model...", flush=True)
    pipeline = None
    if args.pipeline:
        from shape_pipeline import ShapePipeline
        pipeline = ShapePipeline(args.pipeline)
        model, model_hash = pipeline.model, pipeline.model_hash
        payload["active_pipeline"] = {"model_sha256": model_hash, "pipeline_sha256": pipeline.pipeline_hash,
                                      "evaluation": pipeline.manifest["evaluation"], "refiner_enabled": pipeline.refiner is not None}
    else:
        model = load_model()
    page = render_demo(payload).encode("utf-8")
    try:
        server = DemoServer(("127.0.0.1", args.port), model, page, model_hash, pipeline)
    except OSError as exc:
        if exc.errno not in {48, 98, 10048} and getattr(exc, "winerror", None) != 10048:
            raise
        server = DemoServer(("127.0.0.1", 0), model, page, model_hash, pipeline)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Demo ready: {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
