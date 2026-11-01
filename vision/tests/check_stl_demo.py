"""Verify the standalone teacher demo in a real headless browser."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs/stl_shape_v1_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-exe", type=Path)
    args = parser.parse_args()
    payload = json.loads((DEMO / "predictions.json").read_text(encoding="utf-8"))
    model = ROOT / "models/trained/stl_shape_v1/best.pt"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == payload["model_sha256"]
    assert len(payload["cases"]) == 12
    assert {item["class_id"] for case in payload["cases"] for item in case["truth"]} == set(range(9))
    assert all(case["split"] == "test" for case in payload["cases"])
    audit_dir = ROOT / "runs/task_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    requests = []
    with tempfile.TemporaryDirectory(prefix="sorting-demo-check-") as temporary:
        standalone = Path(temporary) / "index.html"
        shutil.copyfile(DEMO / "index.html", standalone)
        with sync_playwright() as playwright:
            options = {"headless": True}
            if args.browser_exe:
                options["executable_path"] = str(args.browser_exe)
            elif os.name == "nt":
                options["channel"] = "msedge"
            browser = playwright.chromium.launch(**options)
            try:
                context = browser.new_context(viewport={"width": 1440, "height": 1100}, offline=True)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                page.on("request", lambda request: requests.append(request.url) if request.url.startswith(("http:", "https:")) else None)
                page.goto(standalone.as_uri(), wait_until="load", timeout=15000)
                expect(page.get_by_role("heading", level=1)).to_contain_text("九类形状模型")
                expect(page.locator(".case-button")).to_have_count(12)
                expect(page.locator("#map50")).to_have_text(f"{payload['evaluation']['mask']['mAP50']:.3f}")
                expect(page.locator("#map5095")).to_have_text(f"{payload['evaluation']['mask']['mAP50_95']:.3f}")
                embedded = page.locator("#demo-data").text_content()
                actual_payload = json.loads(embedded)
                for actual in actual_payload["cases"]:
                    actual.pop("image_data")
                assert actual_payload == payload, "Page and saved inference output disagree"
                for index, case in enumerate(payload["cases"]):
                    page.locator(".case-button").nth(index).click()
                    expect(page.locator("#case-title")).to_contain_text(case["title"])
                    expect(page.locator("#case-status")).to_contain_text(case["id"])
                    expect(page.locator(".case-button[aria-pressed=true]")).to_have_count(1)
                    expect(page.locator("#truth-list li")).to_have_count(len(case["truth"]))
                    expect(page.locator("#prediction-list li")).to_have_count(max(1, len(case["predictions"])))
                    assert page.locator("#original").evaluate("image => image.complete && image.naturalWidth === 320")
                    assert page.locator("#predicted").evaluate("image => image.complete && image.naturalWidth === 320")
                    for position, item in enumerate(case["predictions"]):
                        expect(page.locator("#prediction-list li").nth(position)).to_contain_text(payload["chinese_names"][item["class_id"]])
                        polygon = page.locator("#prediction-overlay polygon").nth(position).get_attribute("points")
                        coordinates = [[float(value) for value in pair.split(",")] for pair in polygon.split()]
                        assert len(coordinates) == len(item["polygon"])
                        assert all(abs(x - nx * 320) < 0.0001 and abs(y - ny * 320) < 0.0001 for (x, y), (nx, ny) in zip(coordinates, item["polygon"], strict=True))
                    markers = page.locator('#prediction-overlay [data-marker]').evaluate_all("nodes => nodes.map(node => [Number(node.getAttribute('x')), Number(node.getAttribute('y'))])")
                    assert len(markers) == len(case["predictions"])
                    for marker_index, (x, y) in enumerate(markers):
                        assert 0 <= x <= 299 and 0 <= y <= 299
                        assert all(abs(x - px) >= 21 or abs(y - py) >= 21 for px, py in markers[:marker_index]), "Prediction numbers overlap"
                    page.get_by_label("显示参考轮廓").check()
                    expect(page.locator("#truth-overlay polygon")).to_have_count(len(case["truth"]))
                    page.get_by_label("显示参考轮廓").uncheck()
                    expect(page.locator("#truth-overlay polygon")).to_have_count(0)
                    page.get_by_label("显示预测轮廓").uncheck()
                    expect(page.locator("#prediction-overlay polygon")).to_have_count(0)
                    page.get_by_label("显示预测框").uncheck()
                    expect(page.locator("#prediction-overlay *")).to_have_count(0)
                    page.get_by_label("显示预测轮廓").check()
                    page.get_by_label("显示预测框").check()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Desktop horizontal overflow"
                page.locator(".case-button").nth(2).click()
                page.get_by_label("显示参考轮廓").check()
                page.screenshot(path=str(audit_dir / "stl_demo_desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile horizontal overflow"
                page.screenshot(path=str(audit_dir / "stl_demo_mobile.png"), full_page=True)
                assert not requests, requests
                assert not errors, errors
            finally:
                browser.close()
    audit = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_sha256": payload["model_sha256"],
        "demo_html_sha256": hashlib.sha256((DEMO / "index.html").read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((DEMO / "predictions.json").read_bytes()).hexdigest(),
        "cases_checked": len(payload["cases"]),
        "classes_covered": 9,
        "copied_single_html_works_offline": True,
        "all_case_navigation_and_overlay_toggles_passed": True,
        "displayed_masks_and_classes_match_saved_inference": True,
        "horizontal_overflow": False,
        "prediction_number_overlap": False,
        "external_http_requests": requests,
        "browser_errors": errors,
    }
    path = audit_dir / "2026-11-01_stl_demo.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
