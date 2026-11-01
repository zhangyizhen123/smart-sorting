#!/usr/bin/env python3
"""Archive a completed, evaluated run without overwriting the v1 baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
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
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["test_used_for_selection"] or summary["final_strict_score"] != report["strict_instance_score"]:
        raise ValueError("Evaluation or selection provenance mismatch")
    import yaml
    dataset = Path(yaml.safe_load((run / "data.yaml").read_text(encoding="utf-8"))["path"])
    expected_freeze = summary["experiment"]["dataset_freeze_sha256"]
    if digest(dataset / "freeze.json") != expected_freeze or report["freeze_sha256"] != expected_freeze:
        raise ValueError("Dataset freeze changed during the experiment")
    selected_validation = json.loads(Path(selection["selected"]["report"]).read_text(encoding="utf-8"))
    if report["model_sha256"] != selected_validation["model_sha256"]:
        raise ValueError("Model changed between validation and testing")
    ShapePipeline(run / "bundle/pipeline.json")
    archive = rooted("models/trained/stl_shape_focus_v2")
    if archive.exists() and any(archive.iterdir()):
        if digest(archive / "pipeline.json") != digest(run / "bundle/pipeline.json"):
            raise FileExistsError("A different focused bundle is already archived")
    shutil.copytree(run / "bundle", archive, dirs_exist_ok=True)
    portable = {**summary, "run": str(run.relative_to(rooted("."))), "model_sha256": digest(archive / "best.pt"),
                "pipeline_sha256": digest(archive / "pipeline.json"), "synthetic_only": True}
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
    lines = ["# 三组易混形状专项实验结果", "", run_description, "",
        f"原模型严格实例得分 **{summary['baseline_strict_score']:.2%}**；验证集选定流程在冻结测试集为 **{summary['final_strict_score']:.2%}**。",
        f"测试包含 {report['image_count']} 张图片、{report['instance_count']} 个物块；99%目标{'达到' if summary['meets_99_target'] else '尚未达到'}。", "",
        f"漏检 {report['missed_instances']} 个，重复检出 {report['duplicate_detections']} 个，额外误检 {report['spurious_detections']} 个。",
        f"复核器{'启用' if selection['refiner_enabled'] else '未启用：验证集未证明优于主模型'}；所有选择只使用验证集。", "",
        "| 形状 | 实例数 | 唯一正确识别比例 | Precision | Recall |", "|---|---:|---:|---:|---:|"]
    for name, values in report["per_class"].items():
        lines.append(f"| {names[name]} | {values['instances']} | {values['clean_instance_accuracy']:.2%} | {values['precision']:.2%} | {values['recall']:.2%} |")
    lines.extend(["", "## 三组双向错分", "", "比例以对应真实类别的全部实例为分母，漏检和其他类别错误另计。", ""])
    for pair in report["focused_pairs"].values():
        for direction, values in pair.items():
            a, b = direction.split("_to_")
            lines.append(f"- {names[a]}→{names[b]}：{values['count']}/{values['denominator']}（{values['rate']:.2%}）。")
    lines.extend(["", "## 困难子集", "", "| 场景 | 图片 | 严格实例得分 |", "|---|---:|---:|"])
    for name, values in report["subsets"].items():
        lines.append(f"| {name} | {values['image_count']} | {values['strict_instance_score']:.2%} |")
    ap = summary["standard_mask_metrics"]
    lines.extend(["", f"主分割模型标准 mask mAP50={ap['mAP50']:.4f}，mAP50-95={ap['mAP50_95']:.4f}；它们不是逐物块准确率。", "",
        "## 使用与下一步", "", "双击根目录 `演示专项模型.cmd` 可以导入自己的图片；旧入口继续展示首版。页面明确区分两版指标。",
        "完整方法与复现步骤见 [专项训练说明](三组易混形状专项训练.md)。本轮无需手工标注。", "",
        "错误图绿线为真值、红线为预测，图中文字使用类别英文缩写。按固定测试顺序保存错误图，不挑选好看的结果。",
        "![专项混淆矩阵](focused_error_examples/confusion.png)", "",
        "后续需完成完整规模、640 输入的正式训练；使用验证集分析错误和选择参数。如果根据本次测试结果继续改方案，最终99%验收应另留新批次。未达到目标时不能把任务表标为达标。"])
    rooted("docs/视觉训练结果_三组专项.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"archived": str(archive), "score": summary["final_strict_score"], "target_achieved": summary["meets_99_target"]}), flush=True)


if __name__ == "__main__":
    main()
