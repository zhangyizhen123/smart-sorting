#!/usr/bin/env python3
"""Render a readable contact sheet from the exact predictions shown by the demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#205baa", "#9a3d78", "#887000", "#076f64", "#a3471b", "#624ca3", "#455e72", "#0877a0"]


def rooted_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def font(size: int):
    for path in [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("Install Microsoft YaHei or Noto Sans CJK to render Chinese captions")


def draw_prediction(image_path: Path, objects: list[dict]) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGBA").resize((320, 320), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    markers = []
    for index, item in enumerate(objects):
        color = ImageColor.getrgb(COLORS[index % len(COLORS)])
        points = [(round(x * 320), round(y * 320)) for x, y in item["polygon"]]
        if len(points) >= 3:
            draw.polygon(points, fill=(*color, 62), outline=(*color, 255), width=2)
        x1, y1, x2, y2 = [round(value * 320) for value in item["box"]]
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=2)
        candidates = (
            (min(297, max(1, x1 + offset)), min(297, max(1, y1 - 24 + row * 24)))
            for row in range(14)
            for offset in (0, 24, -24, 48, -48, 72, -72, 96, -96, 120, -120)
        )
        x, y = next((x, y) for x, y in candidates if all(abs(x - px) >= 23 or abs(y - py) >= 23 for px, py in markers))
        markers.append((x, y))
        draw.rounded_rectangle((x, y, x + 21, y + 21), radius=4, fill=(*color, 255))
        draw.text((x + 6, y + 1), str(index + 1), font=font(13), fill="white")
    return Image.alpha_composite(image, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="docs/stl_shape_v1_demo/predictions.json")
    parser.add_argument("--out", default="docs/视觉训练首版预测拼图.jpg")
    args = parser.parse_args()
    results_path = rooted_path(args.results)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    names = payload["chinese_names"]
    cases = payload["cases"]
    columns, gap, padding, header = 4, 18, 22, 104
    max_predictions = max(len(case["predictions"]) for case in cases)
    tile_height = 320 + 68 + 22 * max(1, max_predictions)
    rows = (len(cases) + columns - 1) // columns
    canvas = Image.new("RGB", (2 * padding + columns * 320 + (columns - 1) * gap, header + rows * tile_height + (rows - 1) * gap + padding), "#f3f6f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 16), "九类形状模型 | 合成测试样例", font=font(26), fill="#172b3d")
    draw.text((padding, 55), "固定规则取样；保留误识别与重复框。编号对应下方预测，结果与中文演示页一致。", font=font(16), fill="#536679")
    for index, case in enumerate(cases):
        x = padding + index % columns * (320 + gap)
        y = header + index // columns * (tile_height + gap)
        image_path = results_path.parent / case["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError(f"Image changed since inference: {image_path}")
        canvas.paste(draw_prediction(image_path, case["predictions"]), (x, y))
        draw.text((x, y + 325), f"{index + 1:02}  {case['title']} / {case['id']}", font=font(15), fill="#172b3d")
        truth = "、".join(names[item["class_id"]] for item in case["truth"])
        draw.text((x, y + 350), "标注：" + truth, font=font(13), fill="#536679")
        for number, prediction in enumerate(case["predictions"]):
            label = f"{number + 1}  {names[prediction['class_id']]}  {prediction['confidence']:.1%}"
            draw.text((x, y + 375 + number * 22), label, font=font(14), fill=COLORS[number % len(COLORS)])
        if not case["predictions"]:
            draw.text((x, y + 375), "未检出超过阈值的物块", font=font(14), fill="#536679")
    output = rooted_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94, optimize=True)
    print(f"Saved {len(cases)} cases from the demo predictions to {output}")


if __name__ == "__main__":
    main()
