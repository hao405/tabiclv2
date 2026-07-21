# Graph-SCM 合成数据实验汇总（最佳 Optuna F-aware）

- 数据：512 个 Graph-SCM 分类任务，Stage1/2/3 = 171/171/170。
- 配对：严格使用 dataset-name shared-`status=ok`；本次所有逻辑 cell 均为 512/512 `status=ok`。
- TabPFN v3 F-aware：用 Optuna 最佳 Trial 3 替换默认 ratio=0.05 的结果。
- 最佳参数：`ttt_c_reserve_ratio=0.2908641795`；搜索目标 mean Accuracy = `0.8149421368`。

| 模型 | 比较 | Accuracy Δ | Balanced Accuracy Δ | W/L/T（Accuracy） | 判断 |
|---|---|---:|---:|---:|---|
| TabICL v2 | FT − Infer | +0.0265 pp | +0.0353 pp | 63/41/408 | 满足 |
| TabICL v2 | 最佳 F-aware − FT | +0.0153 pp | -0.0038 pp | 57/55/400 | 未满足 |
| TabPFN v3 | FT − Infer | +0.0942 pp | +0.1788 pp | 166/122/224 | 满足 |
| TabPFN v3 | 最佳 F-aware − FT | +0.0099 pp | +0.0238 pp | 108/104/300 | 满足 |

## 调参收益

相对默认 TabPFN v3 F-aware（ratio=0.05），最佳 Trial 的 Accuracy 变化为 `+0.0532 pp`，Balanced Accuracy 变化为 `+0.0706 pp`。
相对 FT，最佳 F-aware 的 Accuracy 变化从调参前的 `−0.0432 pp` 翻转为 `+0.0099 pp`，Balanced Accuracy 为 `+0.0238 pp`。
Accuracy 的 dataset-bootstrap 95% CI 为 `[-0.0606, +0.0807] pp`；Balanced Accuracy CI 为 `[-0.0859, +0.1364] pp`。

## 结论

最佳 reserve ratio 使 TabPFN v3 的平均 Accuracy 略高于 FT，但是否同时满足两个指标应以 Balanced Accuracy 结果和 bootstrap CI 为准。完整 Stage 结果见 `paired_summary.csv`，数据集级结果见 `paired_detail.csv`。
