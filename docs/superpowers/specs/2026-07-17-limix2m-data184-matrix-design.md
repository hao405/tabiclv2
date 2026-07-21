# LimiX-2M data184 八方法矩阵设计

## 目标与边界

为 LimiX-2M 实现 `infer`、`ft`、`faware_ft`、`lora`、`prefix`、
`last_block`、`micp` 和 `mixturepfn` 八种原生分类实验方法，并用一个 shell
编排 seed42 的 `8 × 184` data184 矩阵。

本次只交付 runner、shell、测试和 dry-run，不启动 GPU smoke 或完整矩阵。现有
`baseline_compare/LimiX/limix_ttt.py` 保持删除状态；新增实现放在独立 runner
目录，不把实验编排继续写入第三方源码目录。

## 统一数据与训练协议

- 模型固定为 `baseline_compare/LimiX/LimiX-2M.ckpt`。
- 推理配置固定为 `baseline_compare/LimiX/config/cls_default_noretrieval.json`。
- seed 固定为 42；shell 保留扩展 `SEEDS` 的接口。
- `train + val` 构成 labeled target pool，分层留出 10% 仅用于 early stopping。
- test labels 只用于最终指标；只有方法定义允许时才读取未标注 test features。
- 最终预测使用完整 `train + val` context。
- 训练型方法默认使用 AdamW、`lr=1e-5`、weight decay `0.01`、30 epochs、
  chunk size `10000`、query ratio `0.2`、cosine warmup `0.1`、grad clip `1.0`
  和 patience `8`。
- 训练使用两个 preprocessing pipelines，最终推理使用全部四个。

## 方法定义

- `infer`：原始 checkpoint，无参数更新。
- `ft`：随机 context/query，全参数更新。
- `faware_ft`：全参数更新，固定使用未标注 test features、
  `f_mmd + standardized_l2` 和 `reserve_ratio=0.05`。
- `lora`：沿用 `PEFT_Tabicl/1C_Chunk_PEFT.py` 的默认协议，
  `rank=4, alpha=8, dropout=0`。覆盖 LimiX encoder、attention、MLP 和
  decoder 中兼容的线性权重；4D QKV/O 权重使用可恢复原形状的低秩参数化。
- `prefix`：在 12 层 sequence attention 注入长度 8 的 deep K/V prefix，只训练
  prefix 参数。
- `last_block`：只训练最后一个 Transformer layer 和 `cls_y_decoder`。
- `micp`：严格采用论文的 routing-only MICP 语义，不更新模型参数。使用
  `B=3000`、`K=ceil(gamma*N/B)`、KMeans route-shared support；大 cluster
  随机采样 B 行，小 cluster 以 centroid 的 B-NN 扩充；在验证集上从
  `gamma={5,1}` 选择。
- `mixturepfn`：论文完整模型，即 `MICP + C_A PFN`。冻结 backbone，只训练
  每层零初始化 residual adapter；大数据 bootstrap 先取 anchor 的 B-NN，
  再取 64 个 decoder query；小数据使用随机 90/10 bootstrap；固定
  `128 Adam steps, lr=1e-3`。推理 batch 为 1024，ensemble 为 16。

## 编排、结果与恢复

shell 默认运行 seed42 的八个方法，按 `GPUS` 和 `MAX_PARALLEL` 建立方法级 GPU
队列。默认结果目录为：

```text
results/limix/data184/limix2m/seed42/<method>/
```

每个方法保存 `all_classification_results.csv`、`summary.txt`、
`run_manifest.json` 和日志。矩阵根目录保存 `matrix_manifest.json` 和合并汇总。
结果包含 Accuracy、Balanced Accuracy、F1、ROC-AUC、Log Loss、wall time、峰值
显存、trainable params/ratio、`ttt_applied`、fallback 和失败原因。

每个 dataset 先原子写入独立结果，再重建方法汇总。resume 只复用 `status=ok`
且配置、checkpoint、config 和 runner 哈希一致的结果。OOM 时只允许缩小不改变
算法语义的 query micro-batch 并重试一次；仍失败则写 `status=error`，不回退到
Infer。比较时使用 shared-`status=ok` intersection，并单独报告 coverage。

## 验收

- data184 恰有 184 个可解析任务，且全部不超过 LimiX 的 10 类上限。
- fake LimiX 测试覆盖 LoRA 初始等价与冻结范围、Prefix 注入、Last-block、
  full FT 和 MICP adapter 初始恒等。
- 测试随机 split、`f_mmd`、KNN、KMeans route/support、标签覆盖修复和
  test-label 隔离。
- 测试原子写入、配置哈希、resume、失败重跑和 shared-`status=ok` 汇总。
- 通过 Python compile、focused pytest 和 `bash -n`。
- `DRY_RUN=1` 只能生成 seed42 的八个独立 cell，并报告 1472 个预期
  dataset-method 任务。
