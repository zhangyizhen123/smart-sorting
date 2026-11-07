"""Freeze existing detector outputs on the new phone batch; no unapproved accuracy."""
from __future__ import annotations
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO, __version__
from diagnose_real_batch import raw_overlay, digest
from prepare_real_review import ROOT
from predict_stl_demo import serialize_predictions, NAMES
from serve_stl_demo import decode_image
from shape_pipeline import ShapePipeline

OUT=ROOT/'dataset/real/diagnostics/batch_20261102_capture'


def run():
    manifest=ROOT/'dataset/real/raw/batch_20261102_capture/manifest.json'
    rows=json.loads(manifest.read_text(encoding='utf-8'))['images']
    OUT.mkdir(parents=True,exist_ok=True);figures=OUT/'figures';figures.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    v1path=ROOT/'models/trained/stl_shape_v1/best.pt';v1=YOLO(str(v1path))
    v2=ShapePipeline(ROOT/'models/trained/stl_shape_focus_v2/pipeline.json')
    configs=[('v1_320',v1,320,False,None),('v1_640',v1,640,False,None),
             ('v2_320',v2.model,320,True,v2.refiner),('v2_640',v2.model,640,True,v2.refiner)]
    report={'version':1,'scope':'Phone development diagnostics; no approved positive truth and no real-accuracy claim.',
        'accuracy':None,'positive_ground_truth_approved':False,'ultralytics':__version__,
        'manifest_sha256':digest(manifest),'model_sha256':{'v1':digest(v1path),'v2':v2.model_hash},
        'pipeline_sha256':v2.pipeline_hash,'images':[],'empty_tray_reference':'Assistant visually inspected all 11 empty scenes; user review pending.'}
    cards=[]
    for row in rows:
        path=ROOT/row['destination'];assert digest(path)==row['sha256']
        image,original_size=decode_image(path.read_bytes())
        original=f'{row["image_id"]}_original.jpg';image.save(figures/original,quality=92)
        panels=[f'<div><h3>原图</h3><a href="figures/{original}"><img src="figures/{original}" loading="lazy"></a></div>']
        record={'image_id':row['image_id'],'name':path.name,'category':row['category'],
            'source':row['destination'],'source_sha256':row['sha256'],'original_size':original_size,
            'inference_image_size':image.size,'runs':[]}
        for name,model,size,agnostic,refiner in configs:
            settings=dict(imgsz=size,conf=.25,iou=.7,agnostic_nms=agnostic,retina_masks=True,device='cpu')
            start=time.perf_counter();result=model.predict(image,verbose=False,save=False,**settings)[0]
            preds=serialize_predictions(result,include_mask_geometry=False)
            if refiner:
                for p,polygon in zip(preds,result.masks.xy if result.masks is not None else [],strict=True):
                    p['base_class_id']=p['class_id'];p['class_id']=refiner.refine(np.asarray(image),polygon,p['class_id'],p['confidence'])
            for index,mask in enumerate(result.masks.data.cpu().numpy() if result.masks is not None else []):
                maskfile=f'{row["image_id"]}_{name}_{index+1:02d}.png'
                Image.fromarray((mask>0).astype('uint8')*255).save(figures/maskfile)
                preds[index]['mask']=f'figures/{maskfile}';preds[index]['mask_sha256']=digest(figures/maskfile)
            figure=f'{row["image_id"]}_{name}.jpg';raw_overlay(image,result,preds).save(figures/figure,quality=90)
            record['runs'].append({'name':name,'settings':settings,'elapsed_seconds':round(time.perf_counter()-start,3),
                                   'predictions':preds,'figure':f'figures/{figure}'})
            counts=Counter(NAMES[p['class_id']] for p in preds)
            panels.append(f'<div><h3>{name}：{len(preds)}个预测</h3><a href="figures/{figure}"><img src="figures/{figure}" loading="lazy"></a><p>{html.escape(str(dict(counts)))}</p></div>')
        report['images'].append(record)
        cards.append(f'<section><h2>{html.escape(path.name)}</h2><p>{row["category"]}</p><div class="panels">'+''.join(panels)+'</div></section>')
        print(f'{len(report["images"])}/{len(rows)} {path.name}',flush=True)
    report['empty_tray_summary']={}
    for name,*_ in configs:
        predictions=[next(r for r in image['runs'] if r['name']==name)['predictions'] for image in report['images'] if image['category']=='empty_tray']
        report['empty_tray_summary'][name]={'images':len(predictions),'images_with_false_detections':sum(bool(p) for p in predictions),
            'total_false_detections':sum(map(len,predictions)),'largest_false_confidence':max((p['confidence'] for ps in predictions for p in ps),default=None)}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    notice='新增36张手机图；首版/专项各320和640，共144次推理。空盘应无物块，任何检出都需检查。正样本仍无用户核准边界，因此本页不计算识别准确率。640仅为推理对照，未重新训练，也未更改正式演示默认设置。'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二批实拍模型对照</title><style>body{font:16px system-ui;margin:20px;background:#f3f5f8}section{padding:16px;margin:20px 0;background:white}h2,p{overflow-wrap:anywhere}.panels{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}img{width:100%;height:260px;object-fit:contain}h3{font-size:15px}@media(max-width:900px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.panels{grid-template-columns:1fr}}</style><h1>第二批实拍：已有模型基线</h1><p>'+notice+'</p>'+''.join(cards)+'</html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(report['empty_tray_summary'],ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':run()
