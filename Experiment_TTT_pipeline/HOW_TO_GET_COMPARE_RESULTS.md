# 如何生成 compare_results 风格结果

这个文件只回答一件事：怎么用 `Experiment_TTT_pipeline` 跑出每个模型的
`all_classification_results.csv` 和 `summary.txt`，并得到类似
`baseline_compare/compare_results` 的结果目录。

## 结果目录约定

推荐不要直接覆盖现有 `baseline_compare/compare_results`，先写到一个新目录：

```text
baseline_compare/compare_results_pipeline/
```

pipeline 会按策略多加一层目录：

```text
baseline_compare/compare_results_pipeline/
  inference/
    tabicl/all_classification_results.csv
    tabicl/summary.txt
    tabpfnv3/all_classification_results.csv
    tabpfnv3/summary.txt
    ...
  ttt/
    tabicl/all_classification_results.csv
    tabicl/summary.txt
    tabpfnv3/all_classification_results.csv
    tabpfnv3/summary.txt
    ...
  manifests/
    pipeline_runs_all.csv
    pipeline_runs_all.json
```

其中 `inference/<model_key>/` 和 `ttt/<model_key>/` 都是
`make_compare_results_table.py` 能读取的“每个方法一个子目录”的格式。

## 先做 dry-run

先确认命令会调到正确后端，不启动重模型：

```bash
python -m Experiment_TTT_pipeline \
  --strategy both \
  --models all \
  --out-root baseline_compare/compare_results_pipeline \
  --max-datasets 1 \
  --workers 1 \
  --gpus auto \
  --dry-run
```

确认没问题后，把 `--dry-run` 去掉；如果要全量 Data178，也去掉
`--max-datasets 1`。

## 一次跑所有 inference 结果

```bash
python -m Experiment_TTT_pipeline \
  --strategy inference \
  --models all \
  --out-root baseline_compare/compare_results_pipeline \
  --workers 1 \
  --gpus auto \
  --verbose
```

会生成：

```text
baseline_compare/compare_results_pipeline/inference/<model_key>/all_classification_results.csv
baseline_compare/compare_results_pipeline/inference/<model_key>/summary.txt
```

## 一次跑所有 1C/chunk TTT 结果

```bash
python -m Experiment_TTT_pipeline \
  --strategy ttt \
  --models all_ttt \
  --out-root baseline_compare/compare_results_pipeline \
  --workers 1 \
  --gpus auto \
  --verbose
```

会生成：

```text
baseline_compare/compare_results_pipeline/ttt/<model_key>/all_classification_results.csv
baseline_compare/compare_results_pipeline/ttt/<model_key>/summary.txt
```

`tabr` 当前没有注册 TTT 后端，所以不在 `all_ttt` 里。

## 单模型 inference 命令

下面每条都会生成：

```text
baseline_compare/compare_results_pipeline/inference/<model_key>/all_classification_results.csv
baseline_compare/compare_results_pipeline/inference/<model_key>/summary.txt
```

TabICL:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models tabicl \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TabPFN v2:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models tabpfnv2 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TabPFN v2.5:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models tabpfnv25 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TabPFN v3:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models tabpfnv3 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

LimiX:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models limix \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TaBR:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models tabr \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

Orion-MSP:

```bash
python -m Experiment_TTT_pipeline --strategy inference --models orion_msp \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

## 单模型 1C/chunk TTT 命令

下面每条都会生成：

```text
baseline_compare/compare_results_pipeline/ttt/<model_key>/all_classification_results.csv
baseline_compare/compare_results_pipeline/ttt/<model_key>/summary.txt
```

TabICL:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models tabicl \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

如果要让 TabICL TTT 使用 `1C_Chunk_TTT.py` 的 intra-worker GPU group：

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models tabicl \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpu-groups 0,1 --verbose
```

TabPFN v2:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models tabpfnv2 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TabPFN v2.5:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models tabpfnv25 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TabPFN v3:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models tabpfnv3 \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

LimiX:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models limix \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

Orion-MSP:

```bash
python -m Experiment_TTT_pipeline --strategy ttt --models orion_msp \
  --out-root baseline_compare/compare_results_pipeline --workers 1 --gpus auto --verbose
```

TaBR:

```text
当前没有 TaBR 的 1C/chunk TTT pipeline 后端；只能跑 inference。
```

## 一次跑 inference + TTT

如果想连续跑 inference 和 TTT：

```bash
python -m Experiment_TTT_pipeline \
  --strategy both \
  --models all \
  --out-root baseline_compare/compare_results_pipeline \
  --workers 1 \
  --gpus auto \
  --verbose
```

这会先生成 `inference/`，再生成 `ttt/`。TaBR 的 TTT 会被记录为 skip，
不会影响其他模型继续跑。

## 生成 LaTeX 对比表

Inference 对比表：

```bash
python baseline_compare/make_compare_results_table.py \
  --results-dir baseline_compare/compare_results_pipeline/inference \
  --output baseline_compare/compare_results_pipeline/inference/all_metrics_table.tex
```

TTT 对比表：

```bash
python baseline_compare/make_compare_results_table.py \
  --results-dir baseline_compare/compare_results_pipeline/ttt \
  --output baseline_compare/compare_results_pipeline/ttt/all_metrics_table.tex
```

注意：`make_compare_results_table.py` 只会读取直接子目录里的
`all_classification_results.csv`，所以要分别对 `inference/` 和 `ttt/`
运行，而不是直接对 `baseline_compare/compare_results_pipeline/` 运行。

## 常用参数

- `--max-datasets 1`：只跑一个数据集，适合 smoke test。
- `--workers 1 --gpus auto`：单 worker 自动找 GPU。
- `--workers 2 --gpus 0,1`：两个 worker 分别绑定 GPU 0/1。
- `--dry-run`：只打印后端命令，不启动模型。
- `--model-extra-arg=MODEL:ARG`：给某个模型传专属参数。

例子：只把 TabPFN v3 TTT 改成 8 epoch：

```bash
python -m Experiment_TTT_pipeline \
  --strategy ttt \
  --models tabpfnv3 \
  --out-root baseline_compare/compare_results_pipeline \
  --workers 1 \
  --gpus auto \
  --model-extra-arg=tabpfnv3:--ttt-epochs=8
```
