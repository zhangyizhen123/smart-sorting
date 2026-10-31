#!/usr/bin/env python3
"""Build a balanced contact sheet from predictions on the held-out STL test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def rooted_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def target_box(label_path: Path, class_id: int, width: int, height: int) -> list[float] | None:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or int(fields[0]) != class_id:
            continue
        coordinates = [float(value) for value in fields[1:]]
        xs = [value * width for value in coordinates[0::2]]
        ys = [value * height for value in coordinates[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def select_scenes(
    raw_manifest: Path,
    split_manifest: Path,
    predictions: Path,
    model_path: Path,
    class_names: list[str],
    extra_count: int,
    imgsz: int,
    device: str,
) -> list[tuple[Path, str]]:
    with raw_manifest.open(encoding="utf-8-sig", newline="") as stream:
        raw_by_group = {row["group_id"]: row for row in csv.DictReader(stream)}
    with split_manifest.open(encoding="utf-8-sig", newline="") as stream:
        test_rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]

    available = {}
    for row in test_rows:
        group_id = row["group_id"]
        image_path = predictions / f"{group_id}__frame000.jpg"
        if image_path.is_file() and group_id in raw_by_group:
            available[group_id] = {
                "prediction": image_path,
                "image": split_manifest.parent / row["image"],
                "label": split_manifest.parent / row["label"],
                "row": raw_by_group[group_id],
                "metadata": row,
            }

    model = YOLO(str(model_path))
    model_outputs = model.predict(
        source=[str(available[group_id]["image"]) for group_id in sorted(available)],
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    predictions_by_group = {}
    for output in model_outputs:
        group_id = Path(output.path).stem.split("__", 1)[0]
        detections = []
        if output.boxes is not None:
            class_ids = output.boxes.cls.cpu().tolist()
            confidences = output.boxes.conf.cpu().tolist()
            boxes = output.boxes.xyxy.cpu().tolist()
            detections = [
                (int(class_id), float(confidence), [float(value) for value in box])
                for class_id, confidence, box in zip(class_ids, confidences, boxes)
            ]
        predictions_by_group[group_id] = detections

    selected: list[tuple[Path, str]] = []
    used: set[str] = set()
    for class_name in class_names:
        class_id = class_names.index(class_name)
        candidates = []
        fallback = []
        for group_id, item in available.items():
            row = item["row"]
            if row["target_class"] != class_name:
                continue
            metadata = item["metadata"]
            truth = target_box(
                item["label"],
                class_id,
                int(metadata["width"]),
                int(metadata["height"]),
            )
            if truth is None:
                continue
            matching = [
                (confidence, box_iou(box, truth))
                for detected_class, confidence, box in predictions_by_group.get(group_id, [])
                if detected_class == class_id
            ]
            if matching:
                best_confidence, best_iou = max(matching, key=lambda value: (value[1], value[0]))
                fallback.append((best_confidence, best_iou, group_id))
                if best_iou >= 0.5:
                    candidates.append((best_confidence, best_iou, group_id))
        if not candidates:
            if fallback:
                confidence, iou, candidate = min(
                    fallback,
                    key=lambda value: (
                        int(available[value[2]]["row"]["instances"]),
                        -value[0],
                        -value[1],
                        value[2],
                    ),
                )
                match_label = f"CHECK: {class_name} | IoU {iou:.2f}"
            else:
                matching_groups = [
                    group_id for group_id, item in available.items()
                    if item["row"]["target_class"] == class_name
                ]
                if not matching_groups:
                    raise ValueError(f"No test prediction found for target class: {class_name}")
                candidate = min(
                    matching_groups,
                    key=lambda group_id: (int(available[group_id]["row"]["instances"]), group_id),
                )
                match_label = f"MISSED: {class_name}"
        else:
            confidence, iou, candidate = min(
                candidates,
                key=lambda value: (
                    int(available[value[2]]["row"]["instances"]),
                    -value[0],
                    -value[1],
                    value[2],
                ),
            )
            match_label = f"MATCHED: {class_name} | IoU {iou:.2f}"
        image_path = available[candidate]["prediction"]
        row = available[candidate]["row"]
        selected.append((image_path, f"{match_label} | INSTANCES: {row['instances']}"))
        used.add(candidate)

    extras = [
        (group_id, value)
        for group_id, value in available.items()
        if group_id not in used and int(value["row"]["instances"]) >= 2
    ]
    extras.sort(key=lambda item: (int(item[1]["row"]["instances"]), item[0]))
    if len(extras) < extra_count:
        extras = sorted(
            ((group_id, value) for group_id, value in available.items() if group_id not in used),
            key=lambda item: (int(item[1]["row"]["instances"]), item[0]),
        )
    for group_id, item in extras[:extra_count]:
        row = item["row"]
        selected.append((
            item["prediction"],
            f"MULTI-OBJECT | TARGET: {row['target_class']} | {row['instances']} instances",
        ))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", default="dataset/stl_synthetic_raw/manifest.csv")
    parser.add_argument("--split-manifest", default="dataset/stl_synthetic_yolo/manifest.csv")
    parser.add_argument("--predictions", default="runs/segment/stl_shape_v1_test_verified_predictions")
    parser.add_argument("--model", default="models/trained/stl_shape_v1/best.pt")
    parser.add_argument("--names", default="vision/configs/classes.txt")
    parser.add_argument("--out", default="docs/视觉训练首版预测拼图.jpg")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--extra-scenes", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [line.strip() for line in rooted_path(args.names).read_text(encoding="utf-8").splitlines() if line.strip()]
    scenes = select_scenes(
        rooted_path(args.raw_manifest),
        rooted_path(args.split_manifest),
        rooted_path(args.predictions),
        rooted_path(args.model),
        names,
        args.extra_scenes,
        args.imgsz,
        args.device,
    )

    image_size = 320
    caption_height = 34
    header_height = 68
    padding = 20
    gap = 12
    columns = max(1, args.columns)
    rows = (len(scenes) + columns - 1) // columns
    tile_height = image_size + caption_height
    canvas_size = (
        padding * 2 + columns * image_size + (columns - 1) * gap,
        header_height + padding + rows * tile_height + (rows - 1) * gap + padding,
    )
    canvas = Image.new("RGB", canvas_size, (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 15), "STL SHAPE V1 | HELD-OUT TEST PREDICTIONS", fill=(24, 35, 43), font=font(21))
    draw.text((padding, 42), "One box-matched held-out scene per shape where available, plus multi-object scenes. Synthetic data only.", fill=(74, 83, 90), font=font(13))

    for index, (path, caption) in enumerate(scenes):
        row, column = divmod(index, columns)
        x = padding + column * (image_size + gap)
        y = header_height + padding + row * (tile_height + gap)
        with Image.open(path) as source:
            image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + image_size - 1, y + image_size - 1), outline=(211, 216, 220), width=1)
        draw.text((x + 2, y + image_size + 7), caption, fill=(35, 45, 53), font=font(11))

    output = rooted_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)
    print(f"Saved {len(scenes)} test predictions to {output}")


if __name__ == "__main__":
    main()
