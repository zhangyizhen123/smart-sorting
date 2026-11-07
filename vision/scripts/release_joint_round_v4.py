"""Package only a jointly accepted sealed recipe; activate only after UI validation."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/joint_round_v4'
TARGET=ROOT/'models/trained/stl_shape_joint_v4'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def require_success(report):
    if report.get('per_round_family_alpha')!=.025 or report.get('bootstrap_replicates')!=5000:
        raise ValueError('The second-round joint confidence protocol is required')
    if not report.get('passes_joint_test_gate') or report.get('images')!=5800:
        raise ValueError('A complete passing sealed test is required before release')
    if not report.get('both_styles_do_not_regress') or not report.get('negative_alarm_does_not_regress'):
        raise ValueError('Sealed style/negative safeguards did not pass')
    if set(report.get('comparisons',{}))!={'v1_deployed','v2_deployed'}:
        raise ValueError('Both unchanged old models must be compared')
    for comparison in report['comparisons'].values():
        intervals=comparison.get('simultaneous_adjusted_intervals',[])
        if len(intervals)!=4 or not all(low>0 for low,high in intervals):
            raise ValueError('Every joint improvement interval must be positive')
        if comparison.get('common_match_mask_iou_delta') is None or comparison['common_match_mask_iou_delta']<0:
            raise ValueError('Paired mask overlap regressed')


def inputs():
    result=json.loads((RUN/'sealed_test_result.json').read_text(encoding='utf-8'));require_success(result)
    # Import only when a release is actually eligible; this validates the exact frozen sources/data.
    from sealed_joint_round_test import validate_lock
    lock=validate_lock();summary=json.loads((RUN/'sealed_test/candidate/summary.json').read_text(encoding='utf-8'))
    if result.get('candidate_lock_sha256')!=digest(RUN/'candidate_lock.json'):
        raise ValueError('Passing statistics belong to another candidate lock')
    if summary['provenance']['candidate_lock_sha256']!=digest(RUN/'candidate_lock.json') or summary['provenance']['pipeline_sha256']!=lock['pipeline_sha256']:
        raise ValueError('Passing test belongs to a different inference recipe')
    return lock,result,summary


def prepare():
    lock,result,summary=inputs()
    source=ROOT/lock['pipeline'];original=json.loads(source.read_text(encoding='utf-8'))
    identity=dict(source_pipeline_sha256=lock['pipeline_sha256'],sealed_test_result_sha256=digest(RUN/'sealed_test_result.json'),
                  candidate_lock_sha256=digest(RUN/'candidate_lock.json'))
    manifest=TARGET/'pipeline.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text(encoding='utf-8'))
        if existing.get('release_origin')!=identity:
            raise ValueError('An existing release contains a different validated recipe')
        return manifest
    from bundle_development_recipe import verified_copy
    for path in source.parent.rglob('*'):
        if path.is_file() and path!=source:
            verified_copy(path,TARGET/path.relative_to(source.parent),digest(path))
    group=summary['groups']['synthetic_test'];scores=group['scores']['0.5']
    released=copy.deepcopy(original)
    released['evaluation']=dict(stage='sealed_synthetic_test_passed',joint_gate_passed=True,not_deployed=False,
        strict_instance_score=scores['strict_instance_score'],test_images=5800,instances=scores['instance_count'],
        extra_false_positives=scores['spurious_detections'],false_positives_per_image=group['false_positives_per_image'],
        boundary_iou=group['geometry']['boundary_iou_with_unmatched_zero'],independent_real_images=0,
        scope='New synthetic scenes only; the 46 real photographs were training data',
        test_result_sha256=identity['sealed_test_result_sha256'])
    released['release_origin']=identity
    save(manifest,released)
    save(TARGET/'release_preparation.json',dict(prepared_at=datetime.now(timezone.utc).isoformat(),candidate=lock['candidate'],
         pipeline_sha256=digest(manifest),default_activated=False,ui_check_required=True,**identity))
    lines=['# 联合优化模型验收结果','',
           '本报告为第二轮验收，只描述 5800 张新生成、未用于选型的合成场景。46 张已有实拍全部参与过训练，不能据此宣称真实相机达到 99%。','',
           '| 流程 | 严格实例得分 | 额外误报数 | 重复检测数 | 总体边界 IoU |',
           '|---|---:|---:|---:|---:|']
    reports={}
    versions=[('v1_deployed','首版原配置'),('v2_deployed','专项版原配置'),('candidate','联合优化完整流程')]
    for name,title in versions:
        report=json.loads((RUN/'sealed_test'/name/'summary.json').read_text(encoding='utf-8'))
        reports[name]=report
        group=report['groups']['synthetic_test'];score=group['scores']['0.5']
        lines.append(f"| {title} | {score['strict_instance_score']:.2%} | {score['spurious_detections']} | {score['duplicate_detections']} | {group['geometry']['boundary_iou_with_unmatched_zero']:.4f} |")
    lines.extend(['',f"共 {scores['instance_count']} 个标注物块。严格实例得分要求每个真实物块只有一个类别正确、掩膜 IoU 至少 0.5 的对应预测；错类、漏检、重复都会扣分，额外误报也计入分母。它不等同于普通分类准确率。",'',
                  '| 流程 | 漏检数 | 整张图全部正确 | 旧风格严格得分 | 专项风格严格得分 | 空背景报警率 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for name,title in versions:
        groups=reports[name]['groups'];score=groups['synthetic_test']['scores']['0.5']
        lines.append(f"| {title} | {score['missed_instances']} | {score['whole_scene_accuracy']:.2%} | {groups['legacy']['scores']['0.5']['strict_instance_score']:.2%} | {groups['focused']['scores']['0.5']['strict_instance_score']:.2%} | {groups['negative']['empty_false_detection_rate']:.2%} |")
    lines.extend(['','| 新流程相对旧版 | 共同匹配物块的边界 IoU 增量 | 共同匹配物块的掩膜 IoU 增量 |',
                  '|---|---:|---:|'])
    for name,title in versions[:2]:
        comparison=result['comparisons'][name]
        lines.append(f"| {title} | {comparison['gains']['common_match_boundary_iou']:+.4f} | {comparison['common_match_mask_iou_delta']:+.4f} |")
    lines.extend(['','以下逐类得分为“无重复的正确实例数 / 该类标注物块数”，与包含额外误报惩罚的总体严格得分口径不同。总体改善不代表每张照片或每一类都改善。','',
                  '| 类别 | 标注物块数 | 首版 | 专项版 | 联合优化 |',
                  '|---|---:|---:|---:|---:|'])
    per_class={name:report['groups']['synthetic_test']['scores']['0.5']['per_class'] for name,report in reports.items()}
    labels={'cylinder':'圆柱体','cube':'正方体','cuboid':'长方体','tri_prism':'三棱柱',
            'pent_prism':'五棱柱','hex_prism':'六棱柱','cone':'圆锥体','tetrahedron':'四面体','sphere':'球体'}
    for label,values in per_class['candidate'].items():
        cells=[f"{per_class[name][label]['clean_instance_accuracy']:.2%}" if per_class[name][label]['clean_instance_accuracy'] is not None else '无样本' for name,title in versions]
        lines.append(f"| {labels.get(label,label)} | {values['instances']} | {' | '.join(cells)} |")
    lines.extend(['','首轮因专项场景保护未通过而未发布，原报告及结果保留在「联合优化首轮封存测试结果.md」。本轮重新扩大开发集、锁定候选，并使用另一批全新场景验收。',
                  '', '数据构成：1800 张旧风格、3600 张专项风格、400 张空背景。三版在同一批图片上对照，旧版保持原输入尺寸、阈值和分类复核设置。',
                  '', '最终门槛同时检查识别、额外误报、轮廓覆盖、共同匹配边界、掩膜重叠、旧/专项风格和空背景。按场景配对、分风格分层的 5000 次 bootstrap，对两版旧模型与四项改善作本轮联合区间校正（本轮家族 alpha=0.025）；全部要求通过后才允许更新默认。',
                  '', '启动方式：完成页面检查及默认切换后，双击根目录的「演示模型.cmd」。旧首版保存在「演示首版对照.cmd」，旧专项版保存在「演示专项模型.cmd」。',
                  '', '导入图片会使用完整的新流程；首页显示本次新合成场景的识别、误报和轮廓指标，预置样例仍明确标为首版对照。每个输出可切换查看校正后与模型原始轮廓。改变置信度后，固定验收分数不代表新阈值下的表现。',
                  '', '现阶段未完成固定相机独立实拍、机械抓取或分拣闭环验收。模型使用多步复核，当前优先保证效果，速度需要在最终设备上另测。',
                  '', '已有实拍的页面检查也暴露了限制：IMG_20261101_194706 原有五个物块，当前流程输出四个，白色正方体漏检，蓝色正方体及绿色六棱柱被标为圆柱，部分轮廓带碎片。此照片属于已有训练材料；这里记录实际失败现象，不把页面功能通过当作实拍准确率通过。检查截图：runs/task_audit/pipeline_v5_development_real_training_photo.png。',
                  '', f"证据：`runs/segment/joint_round_v4/sealed_test_result.json`；原封存流程 SHA-256 `{lock['pipeline_sha256']}`。",''])
    (ROOT/'docs/联合优化模型验收结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(prepared_pipeline=manifest.relative_to(ROOT).as_posix(),sha256=digest(manifest),default_activated=False)),flush=True)
    return manifest


def activate(audit_path):
    lock,result,summary=inputs();manifest=TARGET/'pipeline.json'
    prepared=json.loads((TARGET/'release_preparation.json').read_text(encoding='utf-8'))
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if digest(manifest)!=prepared['pipeline_sha256'] or not audit.get('passed') or audit.get('pipeline_sha256')!=digest(manifest) or audit.get('javascript_errors'):
        raise ValueError('Current release UI/upload checks must pass before activation')
    launcher=ROOT/'演示模型.cmd';baseline=ROOT/'演示首版对照.cmd';backup=TARGET/'rollback/演示模型.cmd'
    state=TARGET/'activation.json'
    if state.exists():
        activation=json.loads(state.read_text(encoding='utf-8'))
        if digest(launcher)!=activation['launcher_sha256']:
            raise ValueError('Default launcher changed after activation')
        return activation
    before=launcher.read_bytes();text=before.decode('utf-8-sig')
    needle='"%SORTING_DEMO_PYTHON%" "%~dp0vision\\scripts\\serve_stl_demo.py" %*'
    replacement='"%SORTING_DEMO_PYTHON%" "%~dp0vision\\scripts\\serve_stl_demo.py" --pipeline "%~dp0models\\trained\\stl_shape_joint_v4\\pipeline.json" %*'
    if text.count(needle)!=1:
        raise ValueError('Default launcher layout changed; inspect before editing')
    if baseline.exists() and baseline.read_bytes()!=before:
        raise ValueError('An existing legacy launcher must not be overwritten')
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists() and backup.read_bytes()!=before:
        raise ValueError('Rollback launcher differs from the current original')
    backup.write_bytes(before)
    if not baseline.exists():
        baseline.write_bytes(before)
    after=text.replace(needle,replacement).replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')
    temporary=launcher.with_suffix('.tmp');temporary.write_bytes(after);temporary.replace(launcher)
    activation=dict(activated_at=datetime.now(timezone.utc).isoformat(),candidate=lock['candidate'],pipeline_sha256=digest(manifest),
                    launcher_sha256=digest(launcher),original_launcher_sha256=hashlib.sha256(before).hexdigest(),
                    ui_audit_sha256=digest(audit_path),rollback=backup.relative_to(ROOT).as_posix(),independent_real_test_images=0)
    save(state,activation);print(json.dumps(activation,ensure_ascii=False),flush=True);return activation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','activate'])
    parser.add_argument('--ui-audit',default='runs/task_audit/2026-11-01_joint_round4_release_ui.json')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        activate(ROOT/args.ui_audit)
