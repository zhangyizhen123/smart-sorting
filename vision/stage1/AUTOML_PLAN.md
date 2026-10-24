# AutoML 阶段一调优记录与后续计划

## 当前基线归因

基线模型是四层 CPU CNN，训练集/验证集/测试集为 80/20/20 张/类的合成数据。基线保存的最佳验证 checkpoint 在测试集出现 1 个错误：

```text
sphere -> prism, confidence=0.497562
```

错误样本分析表明，合成球被画成了近似平面圆，只带一个小高光，缺少真实球面的径向明暗；模型置信度接近 0.5，属于数据外观歧义和轻微欠拟合，不是数据泄露。`analyze_errors.py` 会输出完整 CSV、错误摘要和 contact sheet。

## 第一轮已经执行

入口：`sweep_round1.py`。每个 trial 都会：

1. 对 train/val/test 做 SHA-256 完全重复检查；重复图片跨 split 会直接报错。
2. 使用训练集增强，验证集和测试集保持确定性。
3. 使用 AdamW + cosine learning rate；每轮保存验证集最优 checkpoint。
4. 只用最优 checkpoint 计算测试指标，避免最后一轮过拟合导致指标失真。
5. 将参数、混淆矩阵和测试准确率写入 `runs/stage1/round1_summary.json`。

启动命令：

```bash
python vision/stage1/sweep_round1.py \
  --data dataset/stage1 \
  --device cpu \
  --epochs 30 \
  --imgsz 128
```

当前合成 smoke 数据的第一轮结果：

| trial | 关键设置 | 最佳 val | 最优 checkpoint test |
|---|---|---:|---:|
| 1 | lr 1e-3，旋转 15°，颜色扰动 0.08，无 label smoothing | 100% | 100% |
| 2 | lr 5e-4，旋转 25°，颜色扰动 0.12，smoothing 0.02 | 98.75% | 95% |
| 3 | lr 1e-3，旋转 8°，颜色扰动 0.05，weight decay 5e-4 | 100% | 100% |

这两个 100% 只说明合成测试集在当前生成规则下已经拟合充分，不能迁移为真实物块准确率。随后按验证集排序重新执行了同一轮 sweep；排序逻辑已明确禁止使用测试集作为 tie-breaker。当前 trial1 的逐样本测试审计为 80/80 正确、0 个错误。

测试集只在每个 trial 的验证集最佳 checkpoint 固定后计算一次，结果写入报告供审计，不参与 trial 排序或模型选择。这样即使多个 trial 的验证准确率相同，也不会把测试集间接暴露给调参过程。

## 下一轮顺序

1. **数据审计**：对真实数据按物块 ID、拍摄日期、相机视角分组；禁止同一物块连续帧跨 split。统计每类、每姿态、每光照条件数量。
2. **错误归因**：按类别、姿态、相机位置、置信度和背景输出错误表；优先补采 `cuboid/prism` 和低置信度样本。
3. **特征工程**：加入真实物块的边缘/轮廓保持增强；保留颜色扰动，避免模型使用颜色捷径；必要时增加 silhouette/edge 辅助输入。
4. **超参数搜索**：在固定测试集不参与调参的前提下，搜索学习率、weight decay、输入分辨率、增强强度和模型宽度；以验证集 macro-F1 与两类误判率排序。
5. **模型融合**：仅在独立验证集上比较 seed ensemble；双相机阶段使用共享 encoder 的 top/oblique 特征拼接，最终一次性在独立测试集报告结果。
6. **停止标准**：验证集达到 100% 仍需继续检查独立拍摄日和 hard-case 测试集；任何一个 `cuboid ↔ prism` 错误都要保留样本并归因，不能通过重复调参掩盖。
