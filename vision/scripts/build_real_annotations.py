"""Build independent, prompted mask drafts; never promote predictions to approved truth."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from ultralytics import SAM

from mask_display import native_mask_geometry
from prepare_real_review import ROOT, CONFIG
from predict_stl_demo import NAMES

PROMPTS = ROOT / "vision/configs/real_review_prompts.json"
OUT = ROOT / "dataset/real/annotations/review_v2"
PALETTE = [(220,55,45),(30,140,240),(235,160,20),(145,70,215),(20,180,145),(225,60,165),(90,145,30),(45,70,220),(190,100,30),(60,190,210)]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records():
    old=json.loads((ROOT/"dataset/real/raw/batch_20261101_import/manifest.json").read_text(encoding="utf-8"))["images"]
    new=json.loads((ROOT/"dataset/real/raw/batch_20261102_capture/manifest.json").read_text(encoding="utf-8"))["images"]
    rows=[r for r in old if r.get("sample_eligible",True)]+new
    assert len(rows)==46 and len({r["sha256"] for r in rows})==46
    return rows


def load_prompts():
    p=json.loads(PROMPTS.read_text(encoding="utf-8"))
    result=p["old"].copy()
    result.update({f"IMG_20261102_{name}..jpg":[["cube","white",*box]] for name,box in p["white_cube"].items()})
    result.update({f"IMG_20261102_{name}..jpg":rows for name,rows in p["multi_object"].items()})
    return result


def annotate(*, only=None):
    config=json.loads(CONFIG.read_text(encoding="utf-8"));prompts=load_prompts()
    if (OUT/'accepted_reviews').exists() and any((OUT/'accepted_reviews').glob('*.json')):
        raise ValueError('Reviewed data exists. Create a new review revision/directory before rebuilding drafts.')
    OUT.mkdir(parents=True,exist_ok=True)
    for sub in ("images","masks","overlays","records"): (OUT/sub).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    checkpoint=ROOT/"dataset/real/tools/mobile_sam.pt"
    model=SAM(str(checkpoint))
    records=[]
    override_path=ROOT/"vision/configs/real_review_mask_overrides.json"
    overrides=json.loads(override_path.read_text(encoding="utf-8")) if override_path.exists() else {}
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",24)
    for index,row in enumerate(source_records()):
        path=ROOT/row["destination"];name=path.name;target=OUT/"records"/f'{row["image_id"]}.json'
        if only and name not in only:
            if not target.exists(): raise ValueError(f"No previous record for {name}")
            records.append(json.loads(target.read_text(encoding="utf-8")));continue
        assert sha(path)==row["sha256"]
        with Image.open(path) as source:
            im=ImageOps.exif_transpose(source).convert("RGB");original_size=im.size;im.thumbnail((1280,1280))
        im.save(OUT/"images"/f'{row["image_id"]}.jpg',quality=95)
        instructions=prompts.get(name,[])
        if not instructions and row["category"]!="empty_tray":raise ValueError(f"Missing object prompts: {name}")
        items=[];masks=[];arr=np.array(im)
        if instructions:
            boxes=np.asarray([r[2:6] for r in instructions],dtype=float)*im.width/480
            boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,im.width);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,im.height)
            # Run all object prompts against one image encoding. No detector class guesses are used.
            result=model.predict(im,bboxes=boxes.tolist(),verbose=False,device="cpu",save=False)[0]
            if result.masks is None or len(result.masks)!=len(instructions):raise ValueError(f"Prompt/mask count mismatch: {name}")
            for k,(instruction,raw) in enumerate(zip(instructions,result.masks.data.cpu().numpy(),strict=True)):
                class_name,color,*_=instruction;mask=(raw>0).astype(np.uint8)
                method="MobileSAM_box_prompt"
                if str(k+1) in overrides.get(name,{}):
                    correction=overrides[name][str(k+1)]
                    if correction.get('keep_largest_component'):
                        count,components,stats,_=cv2.connectedComponentsWithStats(mask)
                        if count>1: mask=(components==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
                        method="MobileSAM_with_assistant_reviewed_speck_removal"
                    else:
                        polygon=np.asarray(correction['polygon'],float)*im.width/480
                        mask[:]=0;cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
                        method="assistant_visual_polygon_correction"
                geometry=native_mask_geometry(mask)
                area=geometry["mask_foreground_pixels"]
                x1,y1,x2,y2=np.rint(boxes[k]).astype(int)
                outside=mask.copy();outside[y1:y2,x1:x2]=0
                warnings=[]
                if area<20:warnings.append("empty_or_tiny_mask")
                if outside.sum()/max(1,area)>.04:warnings.append("mask_extends_outside_prompt")
                if geometry["mask_components"]>1:warnings.append("disconnected_mask_review")
                mask_path=OUT/"masks"/f'{row["image_id"]}_{k+1:02d}.png';Image.fromarray(mask*255).save(mask_path)
                instance={"instance_id":f'{row["image_id"]}_{k+1:02d}',"class_name":class_name,
                    "class_id":NAMES.index(class_name) if class_name in NAMES else None,"color_observed":color,
                    "color_class_id":None,"color_note":"Appearance only; not calibrated ten-color ground truth.",
                    "class_status":"user_confirmed" if (name in {'IMG_20260930_002114..jpg','IMG_20260930_002128..jpg','IMG_20261101_194706..jpg'} or name.startswith('IMG_20260930_00203') or (name=='IMG_20261102_002244..jpg' and k+1 in {2,3,5,10})) else "assistant_proposed",
                    "method":method,"prompt_box":boxes[k].tolist(),"mask":mask_path.relative_to(OUT).as_posix(),
                    "prompt_box_coordinate_space":"annotation_image_pixels_xyxy",
                    "assistant_correction":overrides.get(name,{}).get(str(k+1)),
                    "mask_sha256":sha(mask_path),"boundary_human_approved":False,"warnings":warnings,**geometry}
                if class_name=="unknown":instance["class_status"]="pending_user_identity"
                items.append(instance);masks.append(mask)
            # Occluded visible masks must not silently double-label shared pixels.
            for a in range(len(masks)):
                for b in range(a+1,len(masks)):
                    overlap=int((masks[a]&masks[b]).sum())
                    if overlap/max(1,min(int(masks[a].sum()),int(masks[b].sum())))>.01:
                        items[a]["warnings"].append(f'overlap_with_{b+1}');items[b]["warnings"].append(f'overlap_with_{a+1}')
        record={"image_id":row["image_id"],"name":name,"source":row["destination"],"source_sha256":row["sha256"],
            "category":row["category"],"original_size":original_size,"annotation_size":im.size,
            "image":f'images/{row["image_id"]}.jpg',"overlay":f'overlays/{row["image_id"]}.jpg',
            "coordinate_space":"normalized_exif_oriented_image","instances":items,
            "is_empty_tray":row["category"]=="empty_tray","empty_tray_verification":"assistant_visual_inspection" if not items else None,
            "split":"development_unassigned","leakage_group":"real_pilot_shared_physical_objects",
            "training_approved":False,"human_review_status":"pending","final_test_eligible":False}
        for k,mask in enumerate(masks):
            region=mask.astype(bool);color=np.array(PALETTE[k%len(PALETTE)])
            arr[region]=np.rint(arr[region]*.78+color*.22).astype('uint8')
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,contours,-1,color.tolist(),2)
        overlay=Image.fromarray(arr);draw=ImageDraw.Draw(overlay)
        for k,item in enumerate(items):
            ys,xs=np.where(masks[k]>0)
            if not len(xs):continue
            x,y=int(np.median(xs)),int(np.median(ys))
            label=f'{k+1} {item["class_name"]}';bounds=draw.textbbox((x,y),label,font=font)
            draw.rectangle(bounds,fill='white');draw.text((x,y),label,fill=PALETTE[k%len(PALETTE)],font=font)
        overlay.save(OUT/record['overlay'],quality=94)
        target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        records.append(record)
        print(f'{index+1}/46 {name}: {len(items)} instances, {sum(bool(v["warnings"]) for v in items)} flagged',flush=True)
    payload={"version":1,"annotation_model":"MobileSAM (offline annotation helper, not deployed sorting detector)",
             "annotation_model_sha256":sha(checkpoint),"annotation_model_source":"https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt",
             "source_config_sha256":sha(CONFIG),"prompts_sha256":sha(PROMPTS),"mask_overrides_sha256":sha(override_path) if override_path.exists() else None,"scope":"Draft reference masks for human review, not approved ground truth or final evaluation.",
             "images":records}
    payload['review_revision']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    (OUT/'annotations.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from render_real_review import render
    render(payload,OUT)
    return {"images":len(records),"empty_tray":sum(r['is_empty_tray'] for r in records),"instances":sum(len(r['instances']) for r in records),
        "unknown_classes":sum(i['class_id'] is None for r in records for i in r['instances']),
        "instances_flagged":sum(bool(i['warnings']) for r in records for i in r['instances']),"approved_images":0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--only',nargs='+')
    args=parser.parse_args();print(json.dumps(annotate(only=args.only),ensure_ascii=False,indent=2))
