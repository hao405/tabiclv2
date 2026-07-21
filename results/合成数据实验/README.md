# 合成数据实验本地归档

- `best_matrix/`：用于后续分析的规范化 2×3 矩阵，其中 TabPFN v3 F-aware 已替换为最佳 Trial 3。
- `matrix/`：TabICL v2 / TabPFN v3 × infer / ft / faware_ft 原始 2×3 正式矩阵。
- `optuna/`：TabPFN v3 reserve ratio 正式 4-trial 搜参和 smoke。
- `data_manifest/`：512 个 Graph-SCM 任务的 Stage manifest。
- `analysis/`：用最佳 Optuna trial 替换 TabPFN v3 F-aware 后的 paired comparison。
