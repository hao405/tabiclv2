# 两模型 × 七方法 × 两数据集 × 三种子实验记录模板

## 实验配置

| 项目 | 设置 |
|---|---|
| 模型 | `TabICLv2`、`TabPFNv3` |
| 方法 | `Infer`、`FT`、`F-aware FT`、`LoRA`、`Prefix-tuning`、`Last-block`、`MICP` |
| 数据集 | `data184 (Talent)`、`OpenML-CC18` |
| Seed 1 | `待填写` |
| Seed 2 | `待填写` |
| Seed 3 | `待填写` |
| 主评测口径 | 各比较双方共同的 `status=ok` 数据集交集 |
| 结果格式 | 三个 seed 的 `mean ± std` |

> 记录约定：Accuracy 和 Balanced Accuracy 均填写百分数；`W/L/T` 以同一
> model、dataset、seed 下的指定 reference 为基准。`Infer` 的 reference 填 `—`；
> `FT`、`LoRA`、`Prefix-tuning`、`Last-block` 默认对比 `Infer`；
> `F-aware FT`、`MICP` 默认对比 `FT`。

## 主结果

### data184 (Talent) — TabICLv2

| 方法 | Reference | Seed 1 Acc / BAcc | Seed 2 Acc / BAcc | Seed 3 Acc / BAcc | Acc mean ± std | BAcc mean ± std | Average Rank | W/L/T | 备注 |
|---|---|---|---|---|---|---|---:|---|---|
| Infer | — | — / — | — / — | — / — | — | — | — | — | |
| FT | Infer | — / — | — / — | — / — | — | — | — | — | |
| F-aware FT | FT | — / — | — / — | — / — | — | — | — | — | |
| LoRA | Infer | — / — | — / — | — / — | — | — | — | — | |
| Prefix-tuning | Infer | — / — | — / — | — / — | — | — | — | — | |
| Last-block | Infer | — / — | — / — | — / — | — | — | — | — | |
| MICP | FT | — / — | — / — | — / — | — | — | — | — | |

### data184 (Talent) — TabPFNv3

| 方法 | Reference | Seed 1 Acc / BAcc | Seed 2 Acc / BAcc | Seed 3 Acc / BAcc | Acc mean ± std | BAcc mean ± std | Average Rank | W/L/T | 备注 |
|---|---|---|---|---|---|---|---:|---|---|
| Infer | — | — / — | — / — | — / — | — | — | — | — | |
| FT | Infer | — / — | — / — | — / — | — | — | — | — | |
| F-aware FT | FT | — / — | — / — | — / — | — | — | — | — | |
| LoRA | Infer | — / — | — / — | — / — | — | — | — | — | |
| Prefix-tuning | Infer | — / — | — / — | — / — | — | — | — | — | |
| Last-block | Infer | — / — | — / — | — / — | — | — | — | — | |
| MICP | FT | — / — | — / — | — / — | — | — | — | — | |

### OpenML-CC18 — TabICLv2

| 方法 | Reference | Seed 1 Acc / BAcc | Seed 2 Acc / BAcc | Seed 3 Acc / BAcc | Acc mean ± std | BAcc mean ± std | Average Rank | W/L/T | 备注 |
|---|---|---|---|---|---|---|---:|---|---|
| Infer | — | — / — | — / — | — / — | — | — | — | — | |
| FT | Infer | — / — | — / — | — / — | — | — | — | — | |
| F-aware FT | FT | — / — | — / — | — / — | — | — | — | — | |
| LoRA | Infer | — / — | — / — | — / — | — | — | — | — | |
| Prefix-tuning | Infer | — / — | — / — | — / — | — | — | — | — | |
| Last-block | Infer | — / — | — / — | — / — | — | — | — | — | |
| MICP | FT | — / — | — / — | — / — | — | — | — | — | |

### OpenML-CC18 — TabPFNv3

| 方法 | Reference | Seed 1 Acc / BAcc | Seed 2 Acc / BAcc | Seed 3 Acc / BAcc | Acc mean ± std | BAcc mean ± std | Average Rank | W/L/T | 备注 |
|---|---|---|---|---|---|---|---:|---|---|
| Infer | — | — / — | — / — | — / — | — | — | — | — | |
| FT | Infer | — / — | — / — | — / — | — | — | — | — | |
| F-aware FT | FT | — / — | — / — | — / — | — | — | — | — | |
| LoRA | Infer | — / — | — / — | — / — | — | — | — | — | |
| Prefix-tuning | Infer | — / — | — / — | — / — | — | — | — | — | |
| Last-block | Infer | — / — | — / — | — / — | — | — | — | — | |
| MICP | FT | — / — | — / — | — / — | — | — | — | — | |

## 效率与运行审计

每个模型、数据集和方法填写一行；跨三个 seed 汇总时写成 `mean ± std`，失败和
fallback 保留原始计数，不要并入 `status=ok` 主结果。

| 模型 | 数据集 | 方法 | Trainable Params | Wall Time | Peak GPU Memory | status=ok / total | ttt_applied / ok | OOM | Fallback | 结果目录或日志 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TabICLv2 | data184 | Infer | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | FT | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | F-aware FT | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | LoRA | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | Prefix-tuning | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | Last-block | — | — | — | — | — | — | — | |
| TabICLv2 | data184 | MICP | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | Infer | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | FT | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | F-aware FT | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | LoRA | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | Prefix-tuning | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | Last-block | — | — | — | — | — | — | — | |
| TabPFNv3 | data184 | MICP | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | Infer | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | FT | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | F-aware FT | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | LoRA | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | Prefix-tuning | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | Last-block | — | — | — | — | — | — | — | |
| TabICLv2 | OpenML-CC18 | MICP | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | Infer | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | FT | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | F-aware FT | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | LoRA | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | Prefix-tuning | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | Last-block | — | — | — | — | — | — | — | |
| TabPFNv3 | OpenML-CC18 | MICP | — | — | — | — | — | — | — | |

## 统计与复现说明

- 三个 seed 必须在所有模型、方法和数据集间保持一致。
- 主结果只使用各比较双方共同的 `status=ok` 数据集交集；同时记录交集大小。
- `ttt_applied=False`、OOM 和 fallback 单独审计，不用成功状态掩盖实际未适配结果。
- Accuracy、Balanced Accuracy、Average Rank 和 `W/L/T` 均按 dataset-level
  结果计算，不按样本数加权，除非另行标注。
- 最终报告除 `mean ± std` 外，建议补充 paired bootstrap 95% CI。
- F-aware FT 使用 test features 时注明其 transductive setting，并确认未使用 test labels。
