"""Archive the second real-photo batch and prepare evidence-preserving review data."""
from __future__ import annotations
import csv
import filecmp
import hashlib
import io
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "vision/configs/real_photo_review_v2.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_new_batch():
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    source=ROOT/config["source_directory"]; destination=ROOT/config["new_archive"]
    old=json.loads((ROOT/"dataset/real/raw/batch_20261101_import/manifest.json").read_text(encoding="utf-8"))
    existing={r["sha256"] for r in old["images"]}
    planned={f"IMG_20261102_{suffix}..jpg":category for category,ids in config["new_groups"].items() for suffix in ids}
    assert len(planned)==36
    rows=[]
    for name,category in planned.items():
        src=source/name; dst=destination/category/name
        if not src.is_file() or src.is_symlink(): raise ValueError(f"Missing/unsupported source {src}")
        value=sha(src)
        if value in existing: raise ValueError(f"Unexpected duplicate: {name}")
        existing.add(value)
        if dst.exists() and not filecmp.cmp(src,dst,shallow=False): raise FileExistsError(f"Destination conflict: {dst}")
        with Image.open(src) as im:
            im.load(); ex=im.getexif(); width,height=ImageOps.exif_transpose(im).size
        rows.append({"source":src.relative_to(ROOT).as_posix(),"destination":dst.relative_to(ROOT).as_posix(),
            "image_id":value[:16],"sha256":value,"category":category,"width":width,"height":height,
            "capture_time_exif":ex.get(36867,ex.get(306,"")),"camera_exif":ex.get(272,""),
            "session_group":"phone_20261102_0020_0022","leakage_group":config["annotation_policy"]["shared_object_leakage_group"],
            "split":"development_unassigned","training_approved":False,"final_test_eligible":False})
    content=(json.dumps({"version":1,"date_note":config["date_note"],"images":rows},ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    manifest=destination/"manifest.json"
    if manifest.exists() and manifest.read_bytes()!=content: raise FileExistsError("Manifest changed; refusing overwrite")
    for row in rows:
        dst=ROOT/row["destination"]; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(ROOT/row["source"],dst)
        assert sha(dst)==row["sha256"] and filecmp.cmp(ROOT/row["source"],dst,shallow=False)
    manifest.write_bytes(content)
    buffer=io.StringIO(newline="");writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_path=destination/"manifest.csv"
    if not csv_path.exists():csv_path.write_text(buffer.getvalue(),encoding="utf-8-sig",newline="")
    return rows


if __name__=="__main__":
    rows=archive_new_batch()
    print(json.dumps({"archived":len(rows),"groups":{key:sum(r["category"]==key for r in rows) for key in sorted({r["category"] for r in rows})},"originals_preserved":True},ensure_ascii=False,indent=2))
