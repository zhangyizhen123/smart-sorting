"""Exercise user image import against the live local model and browser UI."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re

from PIL import Image
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[2]


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--browser-exe", type=Path)
    parser.add_argument("--pipeline", type=Path, help="Expected evaluated pipeline manifest")
    parser.add_argument("--audit-name", default="2026-11-01_stl_upload.json")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    audit_dir = ROOT / "runs/task_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "docs/stl_shape_v1_demo/samples/scene000000__frame000.jpg"
    with Image.open(fixture) as source:
        base = source.convert("RGB")
    landscape = Image.new("RGB", (640, 360), (224, 225, 219))
    landscape.paste(base, (160, 20))
    portrait = Image.new("RGB", (320, 640), (224, 225, 219))
    portrait.paste(base, (0, 160))
    images = [
        ("圆柱样例.jpg", fixture.read_bytes(), (320, 320)),
        ("横向测试.png", png_bytes(landscape), (640, 360)),
        ("竖向测试.png", png_bytes(portrait), (320, 640)),
    ]
    errors = []
    cases_checked = []
    with sync_playwright() as playwright:
        options = {"headless": True}
        if args.browser_exe:
            options["executable_path"] = str(args.browser_exe)
        elif os.name == "nt":
            options["channel"] = "msedge"
        browser = playwright.chromium.launch(**options)
        try:
            context = browser.new_context(viewport={"width": 1440, "height": 1100})
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            health = context.request.get(url + "/api/health", timeout=60000).json()
            assert health["ready"]
            if args.pipeline:
                manifest = json.loads(args.pipeline.read_text(encoding="utf-8"))
                assert health["model_sha256"] == manifest["model_sha256"]
                assert health["pipeline_sha256"] == hashlib.sha256(args.pipeline.read_bytes()).hexdigest()
            else:
                assert health["model_sha256"] == hashlib.sha256((ROOT / "models/trained/stl_shape_v1/best.pt").read_bytes()).hexdigest()
            page.goto(url, wait_until="load")
            expect(page.locator("#mode-badge")).to_contain_text("本机模型已就绪", timeout=15000)
            expect(page.get_by_label("选择图片", exact=True)).to_be_enabled()
            if args.pipeline:
                expect(page.locator("#active-pipeline")).to_contain_text("专项模型")
                expect(page.locator("#active-pipeline")).to_contain_text("首版对照")
                expect(page.locator("#start-command")).to_contain_text("演示专项模型.cmd")
                expect(page.locator("#confidence-value")).to_have_text(f"{manifest['settings']['conf']:.2f}")
                expect(page.locator("#active-pipeline")).to_contain_text(f"验收使用置信度 {manifest['settings']['conf']:.2f}")

            for name, content, (width, height) in images:
                page.get_by_label("选择图片", exact=True).set_input_files({"name": name, "mimeType": "image/jpeg" if name.endswith("jpg") else "image/png", "buffer": content})
                expect(page.locator("#case-title")).to_contain_text(name)
                expect(page.locator("#prediction-overlay polygon, #prediction-overlay path[data-mask-native]")).to_have_count(0)
                expect(page.get_by_label("显示参考轮廓")).to_be_disabled()
                with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=60000) as pending:
                    page.get_by_role("button", name="识别形状", exact=True).click()
                response = pending.value
                assert response.status == 200, response.text()
                result = response.json()
                assert (result["width"], result["height"]) == (width, height)
                assert result["model_sha256"] == health["model_sha256"]
                if args.pipeline:
                    assert result["pipeline_sha256"] == health["pipeline_sha256"]
                expect(page.locator("#case-status")).to_contain_text("本机实时预测", timeout=15000)
                expect(page.locator("#prediction-list li")).to_have_count(max(1, len(result["predictions"])))
                assert page.locator("#original").evaluate("image => [image.naturalWidth, image.naturalHeight]") == [width, height]
                stage = page.locator(".stage").first.bounding_box()
                assert abs(stage["width"] / stage["height"] - width / height) < 0.01
                for position, prediction in enumerate(result["predictions"]):
                    expect(page.locator("#prediction-list li").nth(position)).to_contain_text(health["chinese_names"][prediction["class_id"]])
                    if prediction.get("refined"):
                        expect(page.locator("#prediction-list li").nth(position)).to_contain_text("基础检测置信度")
                    native = page.locator("#prediction-overlay path[data-mask-native]").nth(position)
                    assert native.get_attribute("fill-rule") == "evenodd"
                    commands = native.get_attribute("d")
                    rings = [ring for ring in prediction["mask_contours"] if ring]
                    assert commands.count("M") == len(rings) == commands.count("Z")
                    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", commands, re.I)]
                    expected = [v for ring in rings for x, y in ring for v in (x*320, y*320*height/width)]
                    assert len(numbers) == len(expected) and all(abs(a-b)<1e-4 for a,b in zip(numbers,expected,strict=True))
                    assert prediction["mask_size"] == [width,height]
                expect(page.locator("#prediction-overlay polygon")).to_have_count(0)
                if name.endswith("jpg") and not args.pipeline:
                    assert any(item["class_id"] == 0 for item in result["predictions"]), "A known cylinder must be inferred, not replaced by an unrelated saved sample"
                page.locator(".case-button").nth(8).click()
                expect(page.get_by_label("显示参考轮廓")).to_be_enabled()
                page.get_by_role("button", name="查看导入图片").click()
                expect(page.locator("#case-title")).to_contain_text(name)
                if width > height:
                    page.screenshot(path=str(audit_dir / "stl_upload_desktop.png"), full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    page.screenshot(path=str(audit_dir / "stl_upload_mobile.png"), full_page=True)
                    page.set_viewport_size({"width": 1440, "height": 1100})
                cases_checked.append({"name": name, "size": [width, height], "predictions": len(result["predictions"]),
                                      "class_ids": [item["class_id"] for item in result["predictions"]]})

            # Changing the threshold must clear stale results and run actual inference again.
            page.locator("#confidence").focus()
            page.locator("#confidence").press("End")
            expect(page.locator("#confidence-value")).to_have_text("0.95")
            expect(page.locator("#prediction-overlay polygon, #prediction-overlay path[data-mask-native]")).to_have_count(0)
            if args.pipeline:
                expect(page.locator("#active-pipeline")).to_contain_text("当前阈值已改变")
            page.get_by_label("选择图片", exact=True).set_input_files({"name": "空白.png", "mimeType": "image/png", "buffer": png_bytes(Image.new("RGB", (320, 320), "white"))})
            with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=60000) as pending:
                page.get_by_role("button", name="识别形状", exact=True).click()
            assert pending.value.json()["predictions"] == []
            expect(page.locator("#upload-status")).to_contain_text("未检出物块")
            expect(page.locator("#prediction-list")).to_contain_text("未检出")

            # Invalid files must remove the previous image's detections and allow a new selection.
            page.get_by_label("选择图片", exact=True).set_input_files({"name": "坏图.png", "mimeType": "image/png", "buffer": b"invalid image"})
            expect(page.locator("#upload-status")).to_contain_text("无法读取")
            expect(page.get_by_role("button", name="识别形状", exact=True)).to_be_disabled()
            page.get_by_label("选择图片", exact=True).set_input_files({"name": "恢复.jpg", "mimeType": "image/jpeg", "buffer": fixture.read_bytes()})
            expect(page.get_by_role("button", name="识别形状", exact=True)).to_be_enabled()

            headers = {"Content-Type": "application/octet-stream"}
            response = context.request.post(url + "/api/predict", data=b"not an image", headers=headers)
            assert response.status == 422 and "error" in response.json()
            response = context.request.post(url + "/api/predict?confidence=nan", data=fixture.read_bytes(), headers=headers)
            assert response.status == 400
            response = context.request.post(url + "/api/predict", data=fixture.read_bytes(), headers={**headers, "Origin": "https://unrelated.example"})
            assert response.status == 403
            assert context.request.get(url + "/AGENTS.md").status == 404

            # Phone orientation and large originals are normalized before both inference and display.
            rotated = landscape.transpose(Image.Transpose.ROTATE_90)
            exif = Image.Exif(); exif[274] = 6
            encoded = io.BytesIO(); rotated.save(encoded, format="JPEG", exif=exif)
            oriented = context.request.post(url + "/api/predict", data=encoded.getvalue(), headers=headers, timeout=60000).json()
            assert (oriented["width"], oriented["height"]) == (640, 360)
            large = context.request.post(url + "/api/predict", data=png_bytes(landscape.resize((2400, 1350))), headers=headers, timeout=60000).json()
            assert (large["width"], large["height"]) == (1600, 900)
            assert (large["original_width"], large["original_height"]) == (2400, 1350)

            # Known binary geometry proves holes and disjoint regions survive the actual SVG renderer.
            # Independent geometry fixture: two rectangles, one with a rectangular hole.
            rings = [[[10,10],[54,10],[54,59],[10,59]],
                     [[24,24],[40,24],[40,40],[24,40]],
                     [[85,15],[109,15],[109,39],[85,39]]]
            fixture_image=Image.new("RGB",(120,80),"white")
            fixture_payload={"id":"mask-fixture","title":"多连通与孔洞.png","width":120,"height":80,
                "model_sha256":health["model_sha256"],"pipeline_sha256":health.get("pipeline_sha256"),
                "image_data":"data:image/png;base64,"+base64.b64encode(png_bytes(fixture_image)).decode(),
                "predictions":[{"class_id":0,"confidence":.8,"box":[10/120,10/80,110/120,60/80],
                    "polygon":[[10/120,10/80],[110/120,10/80],[110/120,60/80]],
                    "mask_contours":[[[x/120,y/80] for x,y in ring] for ring in rings],"mask_size":[120,80]}],
                "settings":{"conf":.95,"imgsz":320},"elapsed_seconds":0.0}
            page.route("**/api/predict?*",lambda route:route.fulfill(status=200,content_type="application/json",body=json.dumps(fixture_payload)))
            page.get_by_label("选择图片",exact=True).set_input_files({"name":"多连通与孔洞.png","mimeType":"image/png","buffer":png_bytes(fixture_image)})
            page.get_by_role("button",name="识别形状",exact=True).click()
            native=page.locator("#prediction-overlay path[data-mask-native]")
            expect(native).to_have_count(1)
            self_check=native.evaluate("p => [[15,15],[95,25],[30,30],[70,25]].map(([x,y]) => p.isPointInFill(new DOMPoint(x*320/120,y*320/120)))")
            assert self_check == [True,True,False,False], self_check
            page.get_by_label("显示预测轮廓").uncheck()
            expect(native).to_have_count(0)
            page.get_by_label("显示预测轮廓").check()
            expect(native).to_have_count(1)
            page.unroute("**/api/predict?*")
            assert not errors, errors
        finally:
            browser.close()
    audit = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_sha256": health["model_sha256"],
        "pipeline_sha256": health.get("pipeline_sha256"),
        "scope": "UI/API consistency; model accuracy is evaluated separately on the frozen dataset",
        "cases": cases_checked,
        "real_model_inference": True,
        "non_square_mask_alignment": True,
        "native_mask_holes_and_disconnected_regions": True,
        "sample_and_import_navigation": True,
        "threshold_clears_old_result": True,
        "empty_result_and_bad_file_recovery": True,
        "exif_orientation_and_large_image_resize": True,
        "request_validation": True,
        "browser_errors": errors,
    }
    (audit_dir / args.audit_name).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
