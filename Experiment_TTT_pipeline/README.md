# Experiment_TTT_pipeline

TabTune-style experiment runner for the tabular foundation-model baselines under
`baseline_compare` plus the repo-level TabICL benchmark scripts.

The pipeline does not reimplement each model. It keeps the existing, tested
entrypoints as model backends and adds a single registry-driven interface for:

- inference
- 1C/chunk-style test-time training (`TTT`)
- sequential `both` runs

## Supported Models

| key | inference | 1C/chunk TTT backend |
|---|---|---|
| `tabicl` | `benchmark.py` with TabICL v1.1 checkpoint | `1C_Chunk_TTT.py` with TabICL v1.1 checkpoint |
| `tabiclv2` | `benchmark.py` with TabICL v2 checkpoint | `1C_Chunk_TTT.py` with TabICL v2 checkpoint |
| `tabpfnv2` | `baseline_compare/TabPFN-main/benchmark_infer.py` | `baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py` |
| `tabpfnv25` | `baseline_compare/TabPFN-main/benchmark_infer.py` | `baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py` |
| `tabpfnv3` | `baseline_compare/TabPFN-main/benchmark_infer.py` | `baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py` |
| `limix` | `baseline_compare/LimiX/benchmark_infer.py` | `baseline_compare/LimiX/limix_ttt.py` |
| `tabr` | `baseline_compare/tabular-dl-tabr/benchmark_infer.py` | not registered |
| `orion_msp` | `baseline_compare/Orion-MSP/benchmark_infer.py` | `baseline_compare/Orion-MSP/orion_msp_ttt.py` |

## Quick Start

如果目标是复现 `baseline_compare/compare_results` 那种每个模型一个结果目录的产物，
直接看 [HOW_TO_GET_COMPARE_RESULTS.md](HOW_TO_GET_COMPARE_RESULTS.md)。

List the registry:

```bash
python -m Experiment_TTT_pipeline --list-models
```

Preview commands without launching heavy model code:

```bash
python -m Experiment_TTT_pipeline \
  --strategy both \
  --models tabicl,tabpfnv25,limix,tabr,orion_msp \
  --max-datasets 1 --workers 1 --gpus auto --dry-run
```

Run inference only:

```bash
python -m Experiment_TTT_pipeline \
  --strategy inference \
  --models all \
  --workers 1 --gpus auto
```

Run 1C/chunk TTT only:

```bash
python -m Experiment_TTT_pipeline \
  --strategy ttt \
  --models all_ttt \
  --workers 1 --gpus auto --verbose
```

Run inference and then TTT:

```bash
python -m Experiment_TTT_pipeline \
  --strategy both \
  --models tabicl,tabpfnv3,limix \
  --workers 1 --gpus auto
```

Run OpenML-CC18 and TabZilla through the dedicated jiqun-aware runner:

```bash
python Experiment_TTT_pipeline/run_openml_cc18_tabzilla_all_models_jiqun.py \
  --strategy both \
  --benchmarks openml_cc18,tabzilla \
  --max-datasets 1 \
  --dry-run
```

That runner uses `openml_cc18/` and `tabzilla/` by default. It runs inference
for all registered models, and runs TTT for
`tabicl`, `tabiclv2`, `tabpfnv2`, `tabpfnv25`, `tabpfnv3`, `limix`, and
`orion_msp`. `tabr` is included in inference and recorded as a TTT skip.

For jiqun smoke testing, sync and launch into the existing `tmux zh` session:

```bash
python Experiment_TTT_pipeline/run_openml_cc18_tabzilla_all_models_jiqun.py \
  --strategy both \
  --benchmarks openml_cc18,tabzilla \
  --max-datasets 1 \
  --sync-to-jiqun \
  --launch-jiqun \
  --jiqun-target-pane zh:0
```

The runner only does lightweight dataset-file and `info.json` classification
checks before launching backends; it does not run full-root `np.load` shape
validation.

Default outputs are written under:

```text
Experiment_TTT_pipeline/results/<strategy>/<model_key>/
```

Each strategy directory also receives `pipeline_runs.csv` and
`pipeline_runs.json`; a combined manifest is written to
`Experiment_TTT_pipeline/results/manifests/`.

## Passing Backend Arguments

Use `--model-extra-arg MODEL:ARG` for model-specific backend flags:

```bash
python -m Experiment_TTT_pipeline \
  --strategy ttt --models tabpfnv3 \
  --model-extra-arg=tabpfnv3:--ttt-epochs=8 \
  --model-extra-arg=tabpfnv3:--ttt-max-chunk-size=2000
```

Use `--extra-arg` only for flags accepted by every selected backend script.

For TabICL TTT, `--gpus` clears `1C_Chunk_TTT.py`'s default `--gpu-groups`.
If you want intra-worker TabICL TTT data parallelism, pass groups explicitly:

```bash
python -m Experiment_TTT_pipeline \
  --strategy ttt --models tabicl \
  --workers 1 --gpu-groups 0,1
```

## Notes

- `tabr` is inference-only because no 1C TTT backend is present in
  `baseline_compare`.
- The pipeline preserves existing backend behavior, output CSV schemas, retry
  logic, and checkpoint/model path arguments.
- For full benchmark validation, run on the configured GPU server/environment
  after first checking `--dry-run` commands.
