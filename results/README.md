# Results directory layout

Experiment artifacts are organized by model family, model version, method, and
run:

```text
results/
├── tabicl/
│   ├── v1.1/<method>/<run>/
│   └── v2/<method>/<run>/
├── tabpfn/
│   ├── v2/<method>/<run>/
│   ├── v2.5/<method>/<run>/
│   └── v3/<method>/<run>/
├── analysis/
│   ├── average_rank/
│   ├── comparisons/
│   └── data_shift/
└── other_models/<model>/<version>/<method>/<run>/
```

Canonical method names include `infer_baseline`, `ft`, `ttt`, `faware_c`,
`faware_c_wise_ft`, `optuna_lr`, `optuna_reserve_ratio`, and
`prior_synthetic_eval`. A wrapper that emits several methods together uses
`multi_method_sweep` as its method directory.

Run directories preserve the original experiment label, seed-sweep timestamp,
or protocol name so that results from different benchmark surfaces are not
silently merged.

The `analysis/` tree stores cross-model reports, average-rank artifacts, and
dataset-shift audits. `other_models/` stores non-TabICL/TabPFN baselines such as
LimiX and TabR. Historical macOS metadata files were retained under
`analysis/macos_metadata/`; no result artifact was deleted during the
reorganization.
