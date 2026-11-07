"""Publish completed CPU pilot evidence without promoting a candidate to deployment."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from ultralytics import YOLO
from diagnose_real_batch import raw_overlay
from predict_stl_demo import serialize_predictions

ROOT=Path(__file__).resolve().parents[2]


def report():
    cfg=json.loads((ROOT/'vision/configs/real_pilot_v1.json').read_text(encoding='utf-8'))
    run=ROOT/cfg['output'];summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    dataset=ROOT/cfg['dataset'];selection=summary['selection'];diagnostic=selection['diagnostic_candidate']
    labels={'from_v1_initial':'首版起点','from_v2_initial':'专项起点','from_v1_best':'首版微调best','from_v1_last':'首版微调last','from_v2_best':'专项微调best','from_v2_last':'专项微调last'}
    pct=lambda v:f'{100*v:.2f}%'
    lines=['# 已审核实拍混合短跑结果','',
        '日期：2026-11-01。已完成两个起点各3轮的本地CPU训练；输入640，训练154张，合成验证198张。未购买云资源，原有两个部署模型未替换。','',
        '## 合成验证回归','',
        '全部按相同640、置信度0.25、类别无关NMS与原始分割器比较，不使用旧专项复核器。以下是固定合成验证子集结果，不是历史完整测试集结果。','',
        '| 检查点 | 旧风格严格得分 | 专项风格严格得分 |','| --- | ---: | ---: |']
    for name,domains in summary['validation'].items():lines.append(f'| {labels[name]} | {pct(domains["legacy"]["strict_instance_score"])} | {pct(domains["focused"]["strict_instance_score"])} |')
    origin,kind=diagnostic.rsplit('_',1)
    before=json.loads((run/'evaluation'/f'val_{origin}_initial.json').read_text(encoding='utf-8'))
    after=json.loads((run/'evaluation'/f'val_{diagnostic}.json').read_text(encoding='utf-8'))
    chinese={'cylinder':'圆柱','cube':'正方体','cuboid':'长方体','tri_prism':'三棱柱','pent_prism':'五棱柱','hex_prism':'六棱柱','cone':'圆锥','tetrahedron':'四面体','sphere':'球体'}
    error_analysis={}
    for domain in ['legacy','focused']:
        changes=[]
        for cls,a in before['by_domain'][domain]['per_class'].items():
            b=after['by_domain'][domain]['per_class'][cls]
            changes.append({'class':cls,'instances':a['instances'],'clean_correct_before':a['clean_correct'],
                            'clean_correct_after':b['clean_correct'],'change':b['clean_correct']-a['clean_correct']})
        error_analysis[domain]=sorted(changes,key=lambda r:r['change'])
    (run/'class_regression_analysis.json').write_text(json.dumps(error_analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines+=['',('本轮通过合成不退步门槛的候选：'+labels[selection['selected']]) if selection['selected'] else '**本轮没有候选同时通过两种合成风格的不退步门槛，不升级默认模型。**',
            f'用于展示训练拟合的候选：{labels[diagnostic]}。选择只使用合成验证，未使用真实训练图或测试图选检查点。','',
            '旧风格主要退步项（正确且无重复的实例数）：'+ '；'.join(f'{chinese[r["class"]]} {r["clean_correct_before"]}/{r["instances"]} → {r["clean_correct_after"]}/{r["instances"]}' for r in error_analysis['legacy'][:3])+'。因此后续需要重点检查类别混淆，不能只看轮廓或漏检是否改善。','',
            '## 真实训练图上的拟合','',
            '**这46张照片已经参与训练，以下不是未见实物准确率。** 同一批实物的独立真实验证/测试照片为0张，因此不能据此承诺泛化或99%。','',
            '| 模型 | 118实例训练拟合严格得分 | 11空盘中误报图数 | 空盘误报总数 | 12张单白块严格得分 |','| --- | ---: | ---: | ---: | ---: |']
    for name,data in summary['real_training_fit'].items():
        lines.append(f'| {labels[name]} | {pct(data["strict_instance_score"])} | {data["empty_tray"]["images_with_false_detections"]}/11 | {data["empty_tray"]["false_detections"]} | {pct(data["white_cube_strict_score"])} |')
    lines+=['','严格得分将漏检、错分、重复和额外误检计入；阈值固定。空盘没有物块，所以空盘任何检测都记为误报。',
        '', '## 如何使用结果','',
        '- 本轮确认数据能进入训练、掩膜转换不变形，并提供双起点回归证据。3轮只是短跑，不能代替正式收敛实验。',
        '- 优先保留空盘负样本和白块低对比度；下一轮扩大旧风格回放及专项覆盖，在独立验证上重新选择训练参数。',
        '- 当前真实数据缺圆锥和长方体，最终固定相机、独立拍摄会话及板上兼容性仍待补齐。',
        '- 不把训练图再当测试集。正式分拣需另验颜色、抓取坐标、连续帧稳定和正确投放。','',
        '## 文件','',f'- 参数：`vision/configs/real_pilot_v1.json`。',f'- 冻结数据：`{cfg["dataset"]}/freeze.json`。',
        f'- 完整指标：`{cfg["output"]}/summary.json`，逐图记录在 `evaluation/`。',
        f'- 候选权重：`{cfg["output"]}/from_v1/weights/`、`from_v2/weights/`。',
        '- [固定样例的训练前后对照](../runs/segment/real_mixed_pilot_v1/comparison/index.html)。',
        '',f'本轮从启动到完成约 {summary["elapsed_seconds"]/60:.1f} 分钟，包含训练、评估及中途保存目录修复；云支出0元。']
    (ROOT/'docs/实拍混合训练短跑结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Cases fixed by category/filename, not selected for looking successful.
    filenames=['IMG_20261102_002025..jpg','IMG_20261102_002039..jpg','IMG_20261102_002109..jpg','IMG_20261102_002115..jpg',
               'IMG_20261102_002219..jpg','IMG_20261102_002241..jpg','IMG_20261102_002244..jpg','IMG_20260930_002114..jpg']
    review=ROOT/cfg['review'];draft=json.loads((review/'annotations.json').read_text(encoding='utf-8'))
    rows={r['name']:r for r in draft['images']};origin,kind=diagnostic.rsplit('_',1)
    torch.set_num_threads(2)
    models=[('before',YOLO(str(ROOT/cfg['initial_models'][origin]))),('after',YOLO(str(run/origin/'weights'/f'{kind}.pt')))]
    output=run/'comparison';output.mkdir(exist_ok=True);settings={k:v for k,v in cfg['evaluation'].items() if k!='match_iou'};settings['device']='cpu'
    cards=[]
    for name in filenames:
        row=rows[name];image=Image.open(dataset/'images/train'/f'real_{row["image_id"]}.jpg').convert('RGB')
        stem=row['image_id'];image.save(output/f'{stem}_original.jpg',quality=92)
        Image.open(review/row['overlay']).save(output/f'{stem}_reference.jpg',quality=92)
        panels=[('原图',f'{stem}_original.jpg'),('已确认标注',f'{stem}_reference.jpg')]
        for suffix,model in models:
            result=model.predict(image,verbose=False,save=False,**settings)[0];predictions=serialize_predictions(result)
            filename=f'{stem}_{suffix}.jpg';raw_overlay(image,result,predictions).save(output/filename,quality=92)
            title=('训练前' if suffix=='before' else '短跑后')+f'：{len(predictions)}个预测';panels.append((title,filename))
        cards.append('<section><h2>'+html.escape(name)+'</h2><div class="panels">'+''.join('<div><h3>'+html.escape(title)+f'</h3><a href="{file}"><img loading="lazy" src="{file}"></a></div>' for title,file in panels)+'</div></section>')
    title='已审核实拍：3轮短跑前后对照'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{font:16px system-ui;margin:20px;background:#f3f5f8;color:#172433}section{background:white;padding:16px;margin:20px 0}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}img{width:100%;height:330px;object-fit:contain}h2{overflow-wrap:anywhere}@media(max-width:900px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.panels{grid-template-columns:1fr}}</style><h1>'+title+'</h1><p>这8张固定样例来自已经参加训练的照片，只展示训练拟合，不代表新照片准确率。左至右：原图、已确认标注、同一起点训练前、微调3轮后。置信度不等于正确率；所有预测完整显示，包括失败情况。</p><p>当前候选：'+labels[diagnostic]+'；默认分拣模型未替换。</p>'+''.join(cards)+'</html>'
    if (ROOT/'runs/segment/contour_audit_v1/comparison/index.html').exists():
        page=page.replace('<h1>', '<p id="contour-audit-link"><a href="../../contour_audit_v1/comparison/index.html">打开可切换原始/规整轮廓、640/960及局部放大的新版对照页</a></p><h1>', 1)
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'report':'docs/实拍混合训练短跑结果.md','comparison':str(output/'index.html'),'fixed_cases':len(filenames),'candidate':diagnostic},ensure_ascii=False))


if __name__=='__main__':report()
