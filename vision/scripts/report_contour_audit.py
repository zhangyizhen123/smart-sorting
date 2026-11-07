"""Render an offline, original/regularized comparison using saved predictions."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from evaluate_focused import score_records

ROOT=Path(__file__).resolve().parents[2]


def main():
    cfg=json.loads((ROOT/'vision/configs/contour_audit_v1.json').read_text(encoding='utf-8'))
    out=ROOT/cfg['output'];summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if json.loads((out/'status.json').read_text(encoding='utf-8'))['stage']!='complete':
        raise ValueError('Cannot publish incomplete evaluation')
    page=out/'comparison';(page/'data').mkdir(parents=True,exist_ok=True);(page/'images').mkdir(exist_ok=True)
    variants=list(summary['variants']);cases=[]
    for source in sorted((out/'records'/variants[0]).glob('*.json')):
        first=json.loads(source.read_text(encoding='utf-8'));key=first['key']
        image='images/'+key+'.jpg';shutil.copyfile(ROOT/cfg['dataset']/first['image'],page/image)
        truth=[{'class_id':c,'mask_contours':[[[x/first['width'],y/first['height']] for x,y in poly]],
                'box':[min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)]}
               for c,poly in first['truth_polygons']]
        payload={'key':key,'name':first['name'],'image':image,'width':first['width'],'height':first['height'],'truth':truth,'variants':{}}
        for variant in variants:
            record=json.loads((out/'records'/variant/(key+'.json')).read_text(encoding='utf-8'))
            scores={}
            for mode in ['raw','regularized']:
                scored=score_records([record[mode]],.5)
                scores[mode]={k:scored[k] for k in ['instance_count','clean_correct_instances','missed_instances','spurious_detections','duplicate_detections']}
                scores[mode]['wrong_class']=sum(e['reason']=='wrong_class' for e in scored['errors'])
            payload['variants'][variant]={'predictions':[{**p,**g} for p,g in zip(record['raw']['predictions'],record['geometry'],strict=True)],'scores':scores}
        data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
        (page/'data'/(key+'.js')).write_text('window.registerContourCase('+json.dumps(key)+','+data+');\n',encoding='utf-8')
        category=first['category'];group='multi_object' if category in {'multi_object','tray_multi'} else category
        cases.append({'key':key,'name':first['name'],'domain':first['domain'],'group':group,'occlusion':first['confirmed_occlusion_example']})
    cases.sort(key=lambda c:({'real':0,'legacy':1,'focused':2}[c['domain']],c['name']))
    public={'cases':cases,'summary':summary}
    (page/'manifest.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    body=(ROOT/'vision/templates/contour_audit.html').read_text(encoding='utf-8')
    body=body.replace('__AUDIT_DATA__',json.dumps(public,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (page/'index.html').write_text(body,encoding='utf-8')
    pct=lambda value:'—' if value is None else f'{value*100:.2f}%'
    names={'initial_640':'训练前640','initial_960':'训练前960','pilot_640':'短跑后640','pilot_960':'短跑后960'}
    lines=['# 轮廓规整与640／960对照结果','',
        '记录日期：2026-11-01。本轮只推理和诊断，没有新增训练、云支出或默认模型替换。',
        '两个固定权重 × 两种输入 × 244张图 = 976图次。46张实拍已参与此前训练，其结果不代表新照片准确率；另有108张旧风格及90张专项合成验证图。','',
        '## 原始预测：分辨率对照','',
        '| 组合 | 实拍严格分≥50% | 实拍严格分≥90% | 实拍漏检 | 实拍额外误报 | 旧风格严格分≥50% | 专项严格分≥50% | 推理中位秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,groups in summary['variants'].items():
        r=groups['real']['raw'];a=r['0.5']
        lines.append(f"| {names[key]} | {pct(a['strict_instance_score'])} | {pct(r['0.9']['strict_instance_score'])} | {a['missed_instances']} | {a['spurious_detections']} | {pct(groups['legacy']['raw']['0.5']['strict_instance_score'])} | {pct(groups['focused']['raw']['0.5']['strict_instance_score'])} | {groups['all']['median_inference_seconds']:.3f} |")
    lines+=['','## 几何规整：效果与代价','',
        '参数固定：轮廓误差为物块框对角线0.25%，限制0.5—2像素；原始/规整掩膜IoU≥98%，面积变化≤2%，中心偏移不超过允许误差。保留拓扑、不新增实例重叠。未满足保护条件的实例退回原始结果。',
        '规整不改变类别或检测数量；原始掩膜和规整掩膜均逐实例保存PNG与SHA256。不会把几何规整结果写回训练标签或抓取控制。','',
        '| 组合（实拍） | 可规整/全部预测 | 顶点数：原始→规整 | 严格分≥90%：原始→规整 | 边界IoU：原始→规整 | 同配对边界改善/变差/不变 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,groups in summary['variants'].items():
        g=groups['real'];r=g['raw'];s=g['regularized'];d=g['paired_boundary_comparison_on_raw_matches']
        lines.append(f"| {names[key]} | {g['geometry_statuses'].get('accepted',0)}/{r['0.5']['prediction_count']} | {g['raw_vertices']}→{g['regularized_vertices']} | {pct(r['0.9']['strict_instance_score'])}→{pct(s['0.9']['strict_instance_score'])} | {pct(r['geometry']['matched_mean_boundary_iou'])}→{pct(s['geometry']['matched_mean_boundary_iou'])} | {d['improved']}/{d['worsened']}/{d['unchanged']} |")
    lines+=['','边界带为图像对角线0.5%对应的内侧像素带。配对使用类别无关、IoU≥50%的一对一匹配；边界/中心均值只针对匹配实例，必须与未匹配数一起看。`summary.json`另存未匹配按0计的边界均值、中心像素偏差/P95和归一化偏差。',
        '顶点减少代表显示轮廓更简洁，不等于识别准确率提高；即使通过保护，也仍可能更偏离真实轮廓，逐项结果如实保留。','',
        '## 实拍子组','',
        '| 组合 | 子组 | 图数 | 实例数 | 原始严格分≥50% | 漏检 | 额外误报 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,groups in summary['variants'].items():
        for group,label in [('empty_tray','空托盘'),('white_cube','单白块'),('multi_object','多物块'),('occlusion_example','已确认遮挡示例')]:
            g=groups[group];r=g['raw']['0.5']
            lines.append(f"| {names[key]} | {label} | {g['images']} | {r['instance_count']} | {pct(r['strict_instance_score']) if r['instance_count'] else '不适用，看误报数'} | {r['missed_instances']} | {r['spurious_detections']} |")
    lines+=['','遮挡组只含已确认的IMG_20261102_002244，且属于多物块组；分组不是互斥数据，不相加为新样本。空盘没有真值实例，不用其严格分宣称准确率。','',
        '## 下一轮训练调整结论','',
        '- 圆孔、盘沿、桌面误检：保留空盘负样本和相应背景，训练与评估独立统计误报。当前不同视角手机图不套固定托盘像素框。',
        '- 白块、侧面和接触遮挡：保留完整可见轮廓，按失败组增加视角和背景覆盖；不补画不可见区域。',
        '- 易混类别：类别错误与边界质量分开统计；优先检查长方体与正方体/三棱柱，实拍仍缺长方体和圆锥。',
        '- 几何规整仅用于诊断显示，不能代替漏检、背景误检或类别错误的训练修复。',
        '- 任何分辨率变化须同时检查旧风格、专项风格和耗时；本轮不自动升级。固定相机及独立实拍数据未齐，不能据本轮承诺99%。','']
    for model in ['initial','pilot']:
        a=summary['variants'][model+'_640'];b=summary['variants'][model+'_960']
        domains=['legacy','focused'];passes=all(b[d]['raw']['0.5']['strict_instance_score']>=a[d]['raw']['0.5']['strict_instance_score'] for d in domains)
        delta=b['real']['raw']['0.9']['strict_instance_score']-a['real']['raw']['0.9']['strict_instance_score']
        lines.append(f"- {names[model+'_640'].replace('640','')}改960：实拍≥90%严格分变化{delta*100:+.2f}个百分点；两类合成验证均不退步：{'是' if passes else '否'}。这是分辨率诊断，不是部署通过。")
    lines+=['','## 使用方法','',
        '在项目根目录双击`查看轮廓诊断.cmd`，或打开[交互对照页](../runs/segment/contour_audit_v1/comparison/index.html)。选择图片与640/960，切换原始/规整轮廓；检测框默认关闭。点图片或预测列表可放大。',
        '完整结果在`runs/segment/contour_audit_v1/summary.json`；逐图记录和原始/规整掩膜分别位于`records/`、`masks/`。本阶段不需要用户逐个点确认或补拍。']
    (ROOT/'docs/轮廓规整与分辨率对照结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    old=ROOT/'runs/segment/real_mixed_pilot_v1/comparison/index.html'
    old_body=old.read_text(encoding='utf-8')
    marker='id="contour-audit-link"'
    if marker not in old_body:
        old_body=old_body.replace('<h1>', '<p '+marker+'><a href="../../contour_audit_v1/comparison/index.html">打开可切换原始/规整轮廓、640/960及局部放大的新版对照页</a></p><h1>',1)
        old.write_text(old_body,encoding='utf-8')
    print(json.dumps({'cases':len(cases),'variants':variants,'page':str(page/'index.html')},ensure_ascii=False))


if __name__=='__main__':main()
