"""Run reproducible 320/640 real-photo comparisons and unapproved annotation drafts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

from archive_real_batch import DEFAULT_BATCH, ROOT, audit, write_once
from predict_stl_demo import serialize_predictions, NAMES
from serve_stl_demo import decode_image
from shape_pipeline import ShapePipeline

DEFAULT_OUT = ROOT / "dataset/real/diagnostics/batch_20261101_import"
COLORS = [(32, 103, 179), (161, 58, 131), (145, 122, 0), (0, 120, 108), (175, 75, 25), (105, 74, 167)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_overlay(image, result, predictions):
    canvas = np.asarray(image).copy()
    for index, mask in enumerate(result.masks.data.cpu().numpy() if result.masks is not None else []):
        mask = mask.astype(bool)
        color = np.array(COLORS[index % len(COLORS)])
        canvas[mask] = np.rint(canvas[mask] * .7 + color * .3).astype(np.uint8)
        rings, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, rings, -1, color.tolist(), 2)
    output = Image.fromarray(canvas)
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    for index, item in enumerate(predictions):
        x1, y1, x2, y2 = item["box"]
        x1, y1, x2, y2 = x1*image.width, y1*image.height, x2*image.width, y2*image.height
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x1,y1,x2,y2), outline=color, width=3)
        label = f'{index+1}: {NAMES[item["class_id"]]} {item["confidence"]:.3f}'
        tx,ty=max(0,min(x1,image.width-340)),max(0,y1-29)
        bounds=draw.textbbox((tx,ty),label,font=font);draw.rectangle(bounds,fill="white");draw.text((tx,ty),label,fill=color,font=font)
    return output


def run(batch: Path, output: Path) -> dict:
    audit_result = audit(batch)
    records = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))["images"]
    unique = [r for r in records if r["sample_eligible"]]
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    drafts = output / "drafts"; drafts.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    v1_path = ROOT / "models/trained/stl_shape_v1/best.pt"
    pipe = ShapePipeline(ROOT / "models/trained/stl_shape_focus_v2/pipeline.json")
    v1 = YOLO(str(v1_path))
    configs = [
        ("v1_320",v1,320,False,None), ("v1_640_diagnostic",v1,640,False,None),
        ("v2_320",pipe.model,320,True,pipe.refiner), ("v2_640_diagnostic",pipe.model,640,True,pipe.refiner),
    ]
    report = {"version":1,"dataset_audit":audit_result,"ultralytics":ultralytics_version,
              "model_sha256":{"v1":digest(v1_path),"v2":pipe.model_hash},
              "pipeline_sha256":pipe.pipeline_hash,"batch_manifest_sha256":digest(batch/"manifest.json"),
              "purpose":"Development diagnosis only; no approved truth, no accuracy computation. 640 changes inference only, not training.",
              "ground_truth_available":False,"accuracy":None,"images":[]}
    cards=[]; review_rows=[]
    for row in unique:
        path = ROOT / row["destination"]
        image,original_size = decode_image(path.read_bytes())
        image_record = {"image_id":row["image_id"],"source":row["destination"],"image_sha256":row["sha256"],
                        "original_size":original_size,"inference_image_size":image.size,"runs":[]}
        original_name=f'{row["image_id"]}_original.jpg';image.save(figures/original_name,quality=92)
        panels=[f'<div><h3>原图</h3><a href="figures/{original_name}"><img src="figures/{original_name}"></a><p>{html.escape(row["description"])}</p></div>']
        for name,model,imgsz,agnostic,refiner in configs:
            settings={"imgsz":imgsz,"conf":.25,"iou":.7,"device":"cpu","retina_masks":True,"agnostic_nms":agnostic}
            start=time.perf_counter()
            result=model.predict(source=image,verbose=False,save=False,**settings)[0]
            preds=serialize_predictions(result,include_mask_geometry=True)
            if refiner:
                for item,polygon in zip(preds,result.masks.xy if result.masks is not None else [],strict=True):
                    item["base_class_id"]=item["class_id"]
                    item["class_id"]=refiner.refine(np.asarray(image),polygon,item["class_id"],item["confidence"])
                    item["refined"]=item["class_id"]!=item["base_class_id"]
            elapsed=time.perf_counter()-start
            # Save the actual binary mask for independent verification of rendering.
            for index,mask in enumerate(result.masks.data.cpu().numpy() if result.masks is not None else []):
                Image.fromarray((mask>0).astype(np.uint8)*255).save(figures/f'{row["image_id"]}_{name}_mask{index+1}.png')
            figure=f'{row["image_id"]}_{name}.jpg'
            raw_overlay(image,result,preds).save(figures/figure,quality=92)
            image_record["runs"].append({"name":name,"settings":settings,"elapsed_seconds":round(elapsed,3),"predictions":preds,"figure":f"figures/{figure}"})
            summary='<br>'.join(f'{i+1}. {NAMES[p["class_id"]]} · {p["confidence"]*100:.1f}% · {p["mask_components"]}块连通区域' for i,p in enumerate(preds)) or '未检出'
            panels.append(f'<div><h3>{name}</h3><a href="figures/{figure}"><img src="figures/{figure}"></a><p>{summary}</p></div>')
            if name=="v2_320":
                draft={"version":1,"image_id":row["image_id"],"image":row["destination"],"image_sha256":row["sha256"],
                       "coordinate_space":"normalized_to_exif_oriented_image", "annotation_status":"unreviewed_model_proposals",
                       "approved_for_training":False,"model_sha256":pipe.model_hash,"pipeline_sha256":pipe.pipeline_hash,
                       "note":"May contain missing objects, wrong classes and background detections. Must manually correct and approve. No YOLO labels created.","predictions":preds}
                write_once(drafts/f'{row["image_id"]}.json',(json.dumps(draft,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        if Path(row["source"]).name=="1793533894726..jpg":
            first=image_record["runs"][0]["predictions"]
            expected=[(0,.869),(5,.394),(0,.373),(5,.370),(0,.343),(5,.307)]
            report["screenshot_reproduction"] = len(first)==6 and all(p["class_id"]==c and abs(p["confidence"]-v)<.002 for p,(c,v) in zip(first,expected))
        report["images"].append(image_record)
        cards.append(f'<section><h2>{html.escape(Path(row["source"]).name)}</h2><p>{html.escape(row["category"])} · 全部为待人工确认预测</p><div class="panels">'+''.join(panels)+'</div></section>')
        review_rows.append({"image_id":row["image_id"],"image":row["destination"],"review_status":"pending",
                            "confirmed_object_count":"","confirmed_shapes":"","confirmed_colors":"","missing_objects":"",
                            "background_false_positives":"","mask_corrections":"","reviewer":"","approved_for_training":False})
        print(f'Completed {len(report["images"])}/{len(unique)}: {path.name}',flush=True)
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    buffer=io.StringIO(newline=''); writer=csv.DictWriter(buffer,fieldnames=list(review_rows[0]));writer.writeheader();writer.writerows(review_rows)
    # Never reset human review entries on a diagnostic rerun.
    if not (output/"review.csv").exists():
        write_once(output/"review.csv",buffer.getvalue().encode('utf-8-sig'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实拍模型对照与待复核草稿</title><style>body{font:16px system-ui;margin:24px;color:#172433;background:#f3f5f8}section{background:white;padding:18px;margin:22px 0;border:1px solid #ccd3db;border-radius:10px}.panels{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}img{width:100%;height:250px;object-fit:contain;background:#edf0f3}h3{font-size:15px}p{overflow-wrap:anywhere}.notice{border-left:5px solid #c67619;padding:14px;background:#fff3df}@media(max-width:1000px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){.panels{grid-template-columns:1fr}}</style><h1>首批实拍：原图与四组模型对照</h1><p class="notice">10张唯一图片；没有人工核准真值，因此不计算准确率。640只改变本次推理输入，模型没有重新训练，演示默认设置没有改变。置信度不等于正确率；高置信度也会把圆孔识别成球。此页按原始二值掩膜着色，模型预测错误完整保留。</p><p>点击图片查看大图。请逐图确认物块数量、形状、颜色、漏检和背景误检。复核表：<a href="review.csv">review.csv</a>；自动草稿位于drafts，不能直接用于训练。</p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8',newline='\n')
    return {"images":len(unique),"runs":len(unique)*4,"drafts":len(list(drafts.glob('*.json'))),
            "screenshot_reproduction":report.get('screenshot_reproduction'),"accuracy":None,
            "report":str(output/'report.json'),"gallery":str(output/'index.html')}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    print(json.dumps(run(args.batch,args.output),ensure_ascii=False,indent=2))
