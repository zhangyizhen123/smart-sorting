#!/usr/bin/env python3
"""Archive a completed, evaluated run without overwriting the v1 baseline."""
from __future__ import annotations
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from evaluate_focused import digest, rooted
from shape_pipeline import ShapePipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/segment/focused_v2_cpu")
    args = p.parse_args(); run = rooted(args.run)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status["stage"] != "complete":
        raise ValueError("Cannot archive an unfinished run")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((run / "test_final.json").read_text(encoding="utf-8"))
    baseline = json.loads((run / "test_baseline.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["test_used_for_selection"] or summary["final_strict_score"] != report["strict_instance_score"]:
        raise ValueError("Evaluation or selection provenance mismatch")
    import yaml
    dataset = Path(yaml.safe_load((run / "data.yaml").read_text(encoding="utf-8"))["path"])
    expected_freeze = summary["experiment"]["dataset_freeze_sha256"]
    if digest(dataset / "freeze.json") != expected_freeze or report["freeze_sha256"] != expected_freeze:
        raise ValueError("Dataset freeze changed during the experiment")
    if baseline["freeze_sha256"] != expected_freeze or baseline["instance_count"] != report["instance_count"]:
        raise ValueError("Baseline and final model must use the same frozen test set")
    selected_validation = json.loads(Path(selection["selected"]["report"]).read_text(encoding="utf-8"))
    if report["model_sha256"] != selected_validation["model_sha256"]:
        raise ValueError("Model changed between validation and testing")
    pipeline = ShapePipeline(run / "bundle/pipeline.json")
    if (pipeline.model_hash != report["model_sha256"] or pipeline.settings != report["settings"]
            or pipeline.manifest["evaluation"]["strict_instance_score"] != report["strict_instance_score"]
            or bool(pipeline.refiner) != selection["refiner_enabled"]):
        raise ValueError("Demo bundle differs from the evaluated pipeline")
    with (run / "segment/results.csv").open(encoding="utf-8", newline="") as stream:
        curve = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    completed_epochs = int(curve[-1]["epoch"])
    archive = rooted("models/trained/stl_shape_focus_v2")
    if archive.exists() and any(archive.iterdir()):
        if digest(archive / "pipeline.json") != digest(run / "bundle/pipeline.json"):
            raise FileExistsError("A different focused bundle is already archived")
    shutil.copytree(run / "bundle", archive, dirs_exist_ok=True)
    portable = {**summary, "run": str(run.relative_to(rooted("."))), "model_sha256": digest(archive / "best.pt"),
                "pipeline_sha256": digest(archive / "pipeline.json"), "synthetic_only": True,
                "completed_epochs": completed_epochs,
                "software": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "ultralytics", "numpy", "Pillow", "scipy")}},
                "baseline_evaluation": {k: v for k, v in baseline.items() if k != "errors"},
                "final_evaluation": {k: v for k, v in report.items() if k != "errors"}}
    legacy = None
    if (run / "legacy_final.json").exists():
        old_baseline = json.loads((run / "legacy_baseline.json").read_text(encoding="utf-8"))
        old_final = json.loads((run / "legacy_final.json").read_text(encoding="utf-8"))
        if (old_final["model_sha256"] != report["model_sha256"] or old_final["settings"] != report["settings"]
                or old_final["image_count"] != old_baseline["image_count"]
                or old_final["instance_count"] != old_baseline["instance_count"]):
            raise ValueError("Legacy regression must use the fixed final pipeline and identical old test images")
        legacy = {"used_for_selection": False, "separate_from_frozen_acceptance": True,
                  "baseline": {k: v for k, v in old_baseline.items() if k != "errors"},
                  "final": {k: v for k, v in old_final.items() if k != "errors"}}
        portable["legacy_regression"] = legacy
    metrics = rooted("docs/视觉训练指标_三组专项.json")
    metrics.write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(dataset / "freeze.json", rooted("docs/三组专项数据冻结清单.json"))
    examples = rooted("docs/focused_error_examples"); examples.mkdir(exist_ok=True)
    for path in sorted((run / "test_final").glob("error_*.jpg"))[:8]:
        shutil.copy2(path, examples / path.name)
    shutil.copy2(run / "test_final/confusion.png", examples / "confusion.png")
    error_images = list(dict.fromkeys(e["image"] for e in report["errors"]))[:8]
    (examples / "examples.json").write_text(json.dumps({"selection": "First eight erroneous images in deterministic test order; no cherry picking", "examples": error_images}, indent=2) + "\n", encoding="utf-8")
    names = {"cylinder": "圆柱体", "cone": "圆锥", "cube": "正方体", "cuboid": "长方体", "tri_prism": "三棱柱", "pent_prism": "五棱柱", "hex_prism": "六棱柱", "tetrahedron": "四面体", "sphere": "球体"}
    run_description = "本轮是 CPU 试运行，不是 640 输入、150 轮的正式训练。实物效果未验证。" if summary["profile"] == "cpu_pilot" else "本轮采用正式训练配置；实物效果仍未验证。"
    regression_note = ""
    if legacy and legacy["final"]["strict_instance_score"] < legacy["baseline"]["strict_instance_score"]:
        regression_note = (f"**回归限制：首版旧测试集严格得分由 {legacy['baseline']['strict_instance_score']:.2%} 降至 "
                           f"{legacy['final']['strict_instance_score']:.2%}。专项版尚不能视为全面替代原模型，原演示入口继续保留。**")
    lines = ["# 三组易混形状专项实验结果", "", run_description, "",
        f"原模型严格实例得分 **{summary['baseline_strict_score']:.2%}**；验证集选定流程在冻结测试集为 **{summary['final_strict_score']:.2%}**。",
        f"测试包含 {report['image_count']} 张图片、{report['instance_count']} 个物块；99%目标{'达到' if summary['meets_99_target'] else '尚未达到'}。", "",
        regression_note, "",
        "评分要求每个真实物块只检出一次且类别正确；漏检、错分、重复检出都会失败，背景额外误检也计入分母。两版在同一套冻结图片上比较。", "",
        "原模型按原演示设置评估；专项流程包含本轮训练、验证选定的去重和类别复核。因此提升是整条流程的收益，不能全部归功于训练。", "",
        "| 指标 | 原模型 | 专项流程 |", "|---|---:|---:|",
        f"| 严格实例得分 | {baseline['strict_instance_score']:.2%} | {report['strict_instance_score']:.2%} |",
        f"| 整张图片全部识别正确 | {baseline['whole_scene_accuracy']:.2%} | {report['whole_scene_accuracy']:.2%} |",
        f"| 漏检物块数 | {baseline['missed_instances']} | {report['missed_instances']} |",
        f"| 重复检测数 | {baseline['duplicate_detections']} | {report['duplicate_detections']} |",
        f"| 额外误检数 | {baseline['spurious_detections']} | {report['spurious_detections']} |", "",
        f"复核器{'启用' if selection['refiner_enabled'] else '未启用：验证集未证明优于主模型'}；所有选择只使用验证集。", "",
        "## 各类识别", "",
        "类别的唯一正确比例已扣重复检出；Precision、Recall、F1 使用一对一匹配的常规定义。", "",
        "| 形状 | 实例数 | 原模型唯一正确 | 专项唯一正确 | 专项 Precision | 专项 Recall | 专项 F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in report["per_class"].items():
        before = baseline["per_class"][name]
        lines.append(f"| {names[name]} | {values['instances']} | {before['clean_instance_accuracy']:.2%} | {values['clean_instance_accuracy']:.2%} | {values['precision']:.2%} | {values['recall']:.2%} | {values['f1']:.2%} |")
    lines.extend(["", "## 三组双向错分", "", "比例以对应真实类别的全部实例为分母，漏检和其他类别错误另计。", "",
                  "| 真值→预测 | 原模型错分数/实例数 | 原模型错分比例 | 专项错分数/实例数 | 专项错分比例 |", "|---|---:|---:|---:|---:|"])
    for pair_name, pair in report["focused_pairs"].items():
        for direction, values in pair.items():
            a, b = direction.split("_to_")
            before = baseline["focused_pairs"][pair_name][direction]
            lines.append(f"| {names[a]}→{names[b]} | {before['count']}/{before['denominator']} | {before['rate']:.2%} | {values['count']}/{values['denominator']} | {values['rate']:.2%} |")
    lines.extend(["", "## 困难子集", "", "| 场景 | 图片 | 原模型严格得分 | 专项严格得分 |", "|---|---:|---:|---:|"])
    modes = {"single": "单个物块", "separated": "两个物块分开摆放", "multi": "三个或四个物块", "low_view": "较低拍摄角度", "high_view": "接近俯视", "small": "物块在图中较小", "edge": "靠近画面边缘", "low_contrast": "物块与背景颜色接近", "contact": "物块相接触", "occlusion": "部分遮挡"}
    for name, values in report["subsets"].items():
        before = baseline["subsets"][name]
        lines.append(f"| {modes.get(name, name)} | {values['image_count']} | {before['strict_instance_score']:.2%} | {values['strict_instance_score']:.2%} |")
    class_names = list(report["per_class"])
    wrong_directions = sorted(
        [(count, class_names[i], class_names[j]) for i, row in enumerate(report["confusion_rows_true_cols_pred"])
         for j, count in enumerate(row) if i != j and count], reverse=True)
    if wrong_directions:
        lines.extend(["", "## 仍需解决的问题", "", "错分最多的五个方向如下；这些计数不包含漏检和额外误检。", ""])
        for count, source, target in wrong_directions[:5]:
            lines.append(f"- {names[source]}→{names[target]}：{count} 个。")
        weakest = sorted(report["subsets"].items(), key=lambda item: item[1]["strict_instance_score"])[:3]
        lines.extend(["", "分数最低的三个场景：" + "；".join(f"{modes.get(name, name)} {values['strict_instance_score']:.2%}" for name, values in weakest) + "。",
                      "这些结果用于记录局限；下一轮仍用验证集选择改进，并另留新测试批次，不能反复针对本测试集调分。"])
    if legacy:
        old_baseline, old_final = legacy["baseline"], legacy["final"]
        lines.extend(["", "## 首版测试集回归", "",
            f"另外检查首版 {old_final['image_count']} 张、{old_final['instance_count']} 个实例的旧测试集。该检查不用于选择参数，也不与新冻结集混算。",
            f"相同严格口径：原模型 {old_baseline['strict_instance_score']:.2%} → 专项流程 {old_final['strict_instance_score']:.2%}。", "",
            "| 形状 | 原模型唯一正确 | 专项唯一正确 |", "|---|---:|---:|"])
        for name, values in old_final["per_class"].items():
            lines.append(f"| {names[name]} | {old_baseline['per_class'][name]['clean_instance_accuracy']:.2%} | {values['clean_instance_accuracy']:.2%} |")
    ap = summary["standard_mask_metrics"]
    cfg = summary["experiment"]["config"]
    lines.extend(["", "## 实验记录", "",
        f"- 完成时间：{summary['completed_at']}；配置 `{summary['profile']}`。",
        f"- 主模型训练：{cfg['train_per_class'] * 9} 张图、输入 {cfg['imgsz']}、实际 {completed_epochs} 轮（上限 {cfg['epochs']}）。",
        f"- 验证选定候选：`{Path(selection['selected']['report']).stem}`；最终验证严格得分 {selection['validation_score']:.2%}。",
        f"- 推理置信度阈值 {report['settings']['conf']}；类别无关去重 {'开启' if report['settings']['agnostic_nms'] else '关闭'}。",
        f"- 模型 SHA-256：`{portable['model_sha256']}`。",
        f"- 流程配置 SHA-256：`{portable['pipeline_sha256']}`。",
        f"- 冻结清单 SHA-256：`{expected_freeze}`。",
        "- 运行环境版本：" + "；".join(f"{name} {value}" for name, value in portable["software"].items()) + "。",
        "- 完整数值、验证选择与两版逐类统计见 [指标 JSON](视觉训练指标_三组专项.json)。"])
    lines.extend(["", f"主分割模型标准 mask mAP50={ap['mAP50']:.4f}，mAP50-95={ap['mAP50_95']:.4f}；它们不是逐物块准确率。", "",
        "## 使用与下一步", "", "双击根目录 `演示专项模型.cmd`，等待页面显示“本机模型已就绪”；点“选择图片”，再点“识别形状”。识别时保持启动窗口打开。",
        "先保持最低置信度为 0.25，与本次验收一致；修改阈值后的表现没有单独验收。旧入口继续展示首版。页面明确区分两版指标。",
        "完整方法与复现步骤见 [专项训练说明](三组易混形状专项训练.md)。本轮无需手工标注。", "",
        "错误图绿线为真值、红线为预测，图中文字使用类别英文缩写。按固定测试顺序保存错误图，不挑选好看的结果。",
        "![专项混淆矩阵](focused_error_examples/confusion.png)", "",
        "后续需完成完整规模、640 输入的正式训练；使用验证集分析错误和选择参数。如果根据本次测试结果继续改方案，最终99%验收应另留新批次。未达到目标时不能把任务表标为达标。"])
    rooted("docs/视觉训练结果_三组专项.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (archive / "README.md").write_text("\n".join([
        "# 三组易混形状专项模型", "", run_description, "",
        regression_note, "",
        f"冻结合成测试严格实例得分：{report['strict_instance_score']:.2%}；99%目标{'达到' if summary['meets_99_target'] else '未达到'}。",
        f"输入 {report['settings']['imgsz']}；置信度 {report['settings']['conf']}；九类顺序与首版一致。",
        "使用根目录 `演示专项模型.cmd`；程序从 `pipeline.json` 加载权重、推理参数和可选的复核器。",
        "权重及复核文件在加载时校验 SHA-256。不要单独替换权重而保留旧配置。", "",
        "[实验结果与适用范围](../../../docs/视觉训练结果_三组专项.md)",
        "[训练和评估复现](../../../docs/三组易混形状专项训练.md)", "",
        "该模型只完成合成数据实验。实拍图片、固定相机和 MaixCAM 2 部署尚未验收。", ""
    ]), encoding="utf-8")
    print(json.dumps({"archived": str(archive), "score": summary["final_strict_score"], "target_achieved": summary["meets_99_target"]}), flush=True)


if __name__ == "__main__":
    main()
