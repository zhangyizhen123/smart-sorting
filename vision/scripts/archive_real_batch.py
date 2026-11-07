"""Copy the reviewed real-photo batch without deleting or relabelling originals."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import filecmp
import hashlib
import html
import io
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / "dataset/real/raw/batch_20261101_import"
DEFAULT_SOURCE = ROOT / "dataset/一些实物图"
CANONICAL = "1793533894726..jpg"
GROUPS = {
    "single_yellow": [f"IMG_20260930_{s}..jpg" for s in ("001819", "001834", "001950", "001956")],
    "single_gray": [f"IMG_20260930_{s}..jpg" for s in ("002033", "002039")],
    "single_blue_pending": [f"IMG_20260930_{s}..jpg" for s in ("002114", "002128")],
    "tray_multi": [CANONICAL, "IMG_20261101_194706..jpg"],
    "duplicates": ["1793533894726. - 副本.jpg"],
}
DESCRIPTIONS = {
    "single_yellow": "黄色圆柱状物块；类别待人工确认",
    "single_gray": "灰色方块；正方体/长方体待尺寸确认",
    "single_blue_pending": "蓝色物块；形状待端面与侧面确认",
    "tray_multi": "托盘多物块；逐实例类别和轮廓待确认",
    "duplicates": "完全重复副本；保留但不重复采样",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite changed file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def manifest_rows(source: Path, destination: Path) -> list[dict]:
    expected = {name for names in GROUPS.values() for name in names}
    found = {p.name for p in source.iterdir() if p.is_file()}
    if expected != found:
        raise ValueError(f"Batch changed; review before copying. Missing={expected-found}; extra={found-expected}")
    records = []
    for category, names in GROUPS.items():
        for name in names:
            path = source / name
            if path.is_symlink():
                raise ValueError(f"Symlink not allowed: {path}")
            with Image.open(path) as image:
                image.load()
                exif = image.getexif()
                width, height = ImageOps.exif_transpose(image).size
                record = {
                    "image_id": digest(path)[:16],
                    "source": path.relative_to(ROOT).as_posix(),
                    "destination": (destination / category / name).relative_to(ROOT).as_posix(),
                    "category": category,
                    "description": DESCRIPTIONS[category],
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "width": width, "height": height,
                    "capture_time_exif": exif.get(36867, exif.get(306, "")),
                    "camera_exif": exif.get(272, ""),
                    "import_batch": "batch_20261101_import",
                    "session_group": "phone_20260930" if name.startswith("IMG_20260930") else "tray_capture_unconfirmed",
                    # Conservative: the same physical objects may occur in several sessions.
                    "leakage_group": "real_pilot_20261101_shared_objects",
                    "duplicate_of": CANONICAL if category == "duplicates" else "",
                    "sample_eligible": category != "duplicates",
                    "annotation_status": "duplicate_excluded" if category == "duplicates" else "pending_manual_review",
                    "split": "development_unassigned",
                    "training_approved": False,
                    "final_test_eligible": False,
                    "shape_label": "", "color_label": "",
                }
            records.append(record)
    if not filecmp.cmp(source / CANONICAL, source / GROUPS["duplicates"][0], shallow=False):
        raise ValueError("Expected duplicate is no longer byte-identical")
    return records


def audit(destination: Path) -> dict:
    records = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["images"]
    for row in records:
        src, dst = ROOT / row["source"], ROOT / row["destination"]
        assert digest(src) == digest(dst) == row["sha256"], row["source"]
        assert filecmp.cmp(src, dst, shallow=False), row["source"]
        assert not row["training_approved"] and not row["final_test_eligible"]
    assert len(records) == 11 and len({r["sha256"] for r in records}) == 10
    assert sum(r["sample_eligible"] for r in records) == 10
    assert sum(bool(r["duplicate_of"]) for r in records) == 1
    assert len({r["leakage_group"] for r in records}) == 1
    assert not list(destination.rglob("*.txt")), "No unapproved YOLO labels allowed"
    return {"files": len(records), "unique_images": 10, "duplicates": 1,
            "category_counts": dict(Counter(r["category"] for r in records)),
            "originals_preserved": True, "copy_hashes_match": True,
            "training_labels_created": False, "final_test_eligible": False}


def archive(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_relative_to(ROOT / "dataset") or not destination.is_relative_to(ROOT / "dataset/real/raw"):
        raise ValueError("Source/destination must stay inside the intended project dataset directories")
    records = manifest_rows(source, destination)
    document = {"version": 1, "note": "Grouping only, not ground truth. Reviewed diagnostic/development batch; never final test.", "images": records}
    data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Preflight conflicts before copying anything.
    manifest = destination / "manifest.json"
    if manifest.exists() and manifest.read_bytes() != data:
        raise FileExistsError(f"Refusing to overwrite changed manifest: {manifest}")
    for row in records:
        target = ROOT / row["destination"]
        if target.exists() and digest(target) != row["sha256"]:
            raise FileExistsError(f"Refusing to overwrite changed image: {target}")
    for row in records:
        target = ROOT / row["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT / row["source"], target)
    write_once(manifest, data)
    csv_out = io.StringIO(newline="")
    writer = csv.DictWriter(csv_out, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_once(destination / "manifest.csv", csv_out.getvalue().encode("utf-8-sig"))
    cards = []
    for row in records:
        thumb = destination / "preview" / f"{row['image_id']}.jpg"
        with Image.open(ROOT / row["destination"]) as source_image:
            preview = ImageOps.exif_transpose(source_image).convert("RGB")
            preview.thumbnail((640, 480))
            buffer = io.BytesIO(); preview.save(buffer, "JPEG", quality=88)
        write_once(thumb, buffer.getvalue())
        rel = (ROOT / row["destination"]).relative_to(destination).as_posix()
        cards.append(f'<article><h2>{html.escape(row["category"])}</h2><a href="{html.escape(rel, quote=True)}"><img src="preview/{row["image_id"]}.jpg"></a><p>{html.escape(Path(rel).name)}</p><p>{html.escape(row["description"])}</p><p>状态：{row["annotation_status"]}；不是最终测试集。</p></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>首批实物图归档</title><style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#182433}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{background:white;padding:16px;border:1px solid #ddd;border-radius:10px}img{max-width:100%;height:250px;object-fit:contain}h2{font-size:18px}p{overflow-wrap:anywhere}</style><h1>首批实物图：11个文件 / 10张唯一图片</h1><p>原图保留；点击图片查看归档原图。以下分类用于整理，不是已确认的形状或颜色标签。所有图片均待标注，不计入最终独立验收。</p><div class="grid">' + ''.join(cards) + '</div></html>'
    write_once(destination / "index.html", page.encode("utf-8"))
    return audit(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = audit(args.destination) if args.check else archive(args.source, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
