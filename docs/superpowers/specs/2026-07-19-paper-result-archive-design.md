# Paper Result Archive Design

## Goal

Build a reproducible, paper-oriented `result/` archive from the experiment
requirements in `analysis/experiment_results.md` and the available artifacts in
local `results/` plus `jiqun:~/zh/tabiclv2/results/`.

The archive must:

- cover the complete paper matrix, including missing and running cells;
- select the best available formal `FT` and `F-aware FT` results;
- preserve failed rows and adaptation/fallback evidence;
- show every metric available for each selected method without putting direct
  method comparisons in the main result display;
- remain traceable to immutable source paths and file hashes;
- avoid modifying original local or remote experiment results and active jobs.

## Target Matrix and Naming

The inventory contains exactly 72 primary cells:

- models: `TabICLv2`, `TabPFNv3`, `LimiX-2M`;
- datasets: `data184`, the repository's filtered `OpenML-CC18` suite, and
  `Graph-SCM`;
- methods: `Infer`, `FT`, `F-aware FT`, `LoRA`, `ln_head_embedding`,
  `Last-block`, `LoCalPFN`, and `MICP`.

Normalize implementation names only where they are semantically equivalent:

- `infer` and `base_infer` display as `Infer`;
- `faware_ft`, `faware_c`, and formal `f_mmd` F-aware runs display as
  `F-aware FT`;
- `last_layers` and `last_block` display as `Last-block`;
- `ln_head_embedding` remains `ln_head_embedding` and is not called
  Prefix-tuning.

An unavailable or unimplemented target combination is `missing`, not `n/a`.
Use `n/a` only when the experiment design explicitly establishes that a method
cannot apply to a cell.

Expected dataset membership comes from manifests or materialized dataset
directories, not directory names alone. The current expected sizes are
`data184=184`, filtered `OpenML-CC18=67`, and `Graph-SCM=512`; the builder must
record the manifest-derived count and flag a mismatch instead of silently
accepting a hard-coded count.

## Archive Layout

Create the following structure:

```text
result/
├── README.md
├── inventory.csv
├── provenance.csv
├── raw/
│   └── <dataset>/<model>/<method>/
├── tables/
│   ├── method_metrics.md
│   ├── method_metrics.csv
│   ├── completeness_matrix.md
│   ├── completeness_matrix.csv
│   ├── reference_comparisons.csv
│   └── reference_rankings.csv
└── scripts/
    └── build_result_archive.py
```

`raw/` is a curated snapshot, not a complete experiment mirror. For each
selected cell, copy the final `all_classification_results.csv` and any directly
relevant manifest, summary, or best-parameter file. Do not copy worker logs,
caches, smoke outputs, non-selected Optuna trials, or per-dataset recovery
fragments after a merged formal result exists.

`README.md` explains the selection timestamp, source hosts, completeness
semantics, metric aggregation, and the fact that comparison files are
diagnostic references rather than the main presentation.

## Candidate Discovery and Best-Result Selection

The builder scans local `results/` and
`jiqun:~/zh/tabiclv2/results/`. Candidate discovery is read-only and must not
send commands to, interrupt, or reuse active `tmux zh:*` panes.

Exclude these candidates from best-result selection:

- smoke runs;
- explicitly interrupted runs;
- unmerged single-dataset recovery fragments;
- superseded default trials when a formal best Optuna trial is available;
- malformed CSVs or results with duplicate `dataset_name` rows.

Select one result per primary cell in this deterministic order:

1. formal run over smoke, debug, recovery-fragment, or compatibility run;
2. greater unique target-dataset coverage;
3. greater `status=ok` count;
4. greater mean Accuracy over the candidate's `status=ok` rows;
5. newer file modification time, only as the final deterministic tie-breaker.

This rule applies to every method and specifically guarantees that `FT` and
`F-aware FT` use the best available formal result. A high mean from a smaller
partial subset cannot displace a more complete formal result.

If a selected run or Optuna trial was chosen on the same evaluation tasks being
reported, set `selected_on_eval=true` in `inventory.csv` and
`provenance.csv`. The artifact may still be selected as requested, but the
archive must not describe it as an unbiased held-out estimate.

## Completeness and Provenance

`inventory.csv` has one row for each of the 72 primary cells and includes:

```text
model,dataset,method,overall_status,artifact_complete,
source_host,source_path,selected_run,selected_on_eval,
expected_datasets,result_rows,unique_datasets,status_ok,
adaptation_expected,adaptation_applied,fallback_count,failed_count,
selection_note,snapshot_timestamp
```

Use these `overall_status` values:

- `complete`: all target datasets are present and `status=ok`; adaptive
  methods also applied adaptation on every expected adaptive row;
- `complete_with_failures`: the artifact covers the full target set but has
  failures, skips, fallback, or unapplied adaptation;
- `partial`: a stable formal artifact exists but lacks target rows;
- `running`: an incomplete artifact is confirmed active from a current remote
  process, pane, or log;
- `missing`: no qualifying artifact exists;
- `invalid`: an artifact exists but cannot be safely parsed or has duplicate
  dataset identities;
- `n/a`: explicitly inapplicable by design.

`artifact_complete` concerns dataset membership only and is separate from
`status_ok` and adaptation coverage. Failed rows remain in copied raw CSVs.

`provenance.csv` records every selected source and every conflicting
non-selected candidate needed to explain selection:

```text
model,dataset,method,selected,source_host,source_path,
source_mtime,size_bytes,sha256,rejection_reason,snapshot_timestamp
```

When local and remote files have the same SHA-256, archive one copy and record
both origins. When hashes differ, apply the selection rule and retain both
candidate records in provenance; never overwrite silently.

## Result Tables

### Main method display

`method_metrics.md` and `method_metrics.csv` are grouped by
`dataset → model → method`. They display each selected method's own metrics and
do not show direct deltas, W/L/T, or rankings.

Aggregate every meaningful result metric actually present in the selected CSV:

- predictive metrics such as Accuracy, Balanced Accuracy, AUC, F1, and Log
  Loss;
- robustness statistics such as mean, standard deviation, and valid-seed
  count when multiple comparable seeds are available;
- coverage metrics such as result rows, `status=ok`, failures, fallback, and
  `ttt_applied`/`ft_applied`;
- efficiency metrics such as fit, adaptation, and prediction time and peak
  memory when present.

Configuration values and identifiers remain provenance/configuration fields,
not averaged metrics. Use `—` for a metric unavailable to a method. Never
invent values or coerce semantically different columns into the same metric.
The CSV may use a stable superset of metric columns; the Markdown table may be
split into performance, coverage, and efficiency blocks to stay readable while
still showing all available metrics.

### Completeness display

`completeness_matrix.md` and `.csv` show all 72 cells with:

- overall status;
- `unique_datasets / expected_datasets`;
- `status_ok`;
- adaptation coverage for adaptive methods;
- a short note for missing, running, failed, or fallback cells.

Include separate lists of missing cells, running cells with their remote paths,
and stable partial cells with their exact deficits.

### Reference-only comparisons

Do not put comparisons in the main Markdown result display.

Generate `reference_comparisons.csv` using shared-`status=ok` intersections:

- `FT` against `Infer`;
- `F-aware FT` against `FT`;
- `LoRA`, `ln_head_embedding`, and `Last-block` against `Infer` and,
  secondarily, `FT`;
- `LoCalPFN` and `MICP` against `FT`.

Store metric deltas and W/L/T here only. Generate
`reference_rankings.csv` only where the compared methods share a common
successful dataset intersection. Never mix rankings calculated on different
dataset sets.

## Remote Snapshot Safety

For a remote CSV that may still be written:

1. record size, mtime, and SHA-256;
2. copy to a temporary archive path;
3. record the remote values again;
4. accept the copy only if both observations match its hash;
5. retry at most three times.

If the file continues changing, do not archive a potentially torn CSV. Record
the cell as `running`, including the current row count and remote source path.
All archive writes use temporary files followed by atomic rename. Re-running
the builder updates generated archive files deterministically without deleting
or modifying source experiments.

## Validation and Acceptance

The archive is accepted when:

- `inventory.csv` contains exactly 72 unique
  `(model, dataset, method)` rows;
- every selected source exists at collection time and every archived copy
  matches its recorded SHA-256;
- selected CSVs parse successfully and contain unique dataset identities;
- manifest-derived target membership agrees with reported coverage;
- the deterministic selector chooses complete formal candidates before
  partial high-mean candidates and selects the current best formal `FT` and
  `F-aware FT`;
- completeness labels agree with artifact, `status=ok`, adaptation, fallback,
  and active-run evidence;
- main Markdown tables contain method metrics only;
- delta, W/L/T, and ranking outputs appear only in the reference CSVs;
- a second builder run against unchanged sources produces the same archive
  contents and no duplicate inventory rows;
- no local `results/` artifact, remote result, or active `jiqun` job is moved,
  deleted, or modified.

The implementation should include focused automated tests for candidate
filtering, deterministic best-result selection, status classification,
duplicate detection, shared-intersection comparisons, metric discovery, and
idempotent output generation. A final live read-only audit against local
`results/` and `jiqun` verifies the produced completeness matrix.
