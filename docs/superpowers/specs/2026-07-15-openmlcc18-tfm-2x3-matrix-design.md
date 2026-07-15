# TabICL v2 / TabPFN v3 OpenML-CC18 2x3 Matrix Design

## Summary

`scripts/run_tfm_experiment.py --matrix` will default to TabICL v2 and
TabPFN v3 across `infer`, `ft`, and `faware_ft`. Single-cell launches and
explicit matrix subsets continue to support TabICL v1.1 and TabPFN v2.

The default dataset becomes OpenML-CC18. To give all six cells the same task
surface, the launcher creates and reuses a validated, read-only view containing
the 67 tasks with at most 10 classes. The five larger-class tasks remain in the
source dataset and are recorded as exclusions in the view manifest.

## Launcher and Dataset View

- Keep the four-model CLI choice set and introduce
  `DEFAULT_MATRIX_MODELS = ("tabicl-v2", "tabpfn-v3")`.
- Change the default data root to `openml_cc18`. When that default is used and
  no override is supplied, apply a 10-class limit. A zero class limit disables
  filtering; explicitly supplied non-OpenML data roots are not filtered unless
  requested.
- Build `results/dataset_views/openml_cc18_max10` from directory symlinks. Use a
  temporary directory and atomic rename, validate benchmark files and labels,
  and write a manifest with source, included, and excluded tasks.
- Record source and effective data roots, class limit, and view manifest in
  launch and matrix manifests.
- Use `model_native_l2` for the experiment, mapping to
  `tabicl_encoded_l2` for TabICL and `raw_l2` for TabPFN.

## Orchestration and Reporting

Add an OpenML-specific wrapper with `smoke` and `full` presets. Smoke uses
`OpenML-ID-1063`, `OpenML-ID-23381`, and `OpenML-ID-188` across all six cells
with two estimators, one TTT epoch, and patience one. Full uses the 67-task view,
seed 42, 32/8 estimators, 30 TTT epochs, learning rate `1e-5`, patience eight,
one worker, independent run IDs, logs, and matrix resume.

Add a paired summarizer that uses only shared `status=ok` rows. For accuracy
and balanced accuracy, report `FT-Infer` and `F-aware-FT` means, dataset
bootstrap 95% confidence intervals, win/loss/tie counts, and negative transfer.
Also audit coverage, failures, OOM/fallback, `ttt_applied`, selection, and the
persisted native metric.

## Validation and Rollout

Local tests cover the default 2x3 matrix, explicit legacy 4x3 matrix, data-view
creation/reuse/corruption recovery, metric routing, paired intersections, and
missing-row exclusion. Run Python compilation, shell syntax checks, targeted
pytest, and both default and compatibility dry-runs.

On `jiqun`, preserve other tmux work and use `zh:2`. Recheck the environment,
checkpoints, data, dependencies, and script hashes. Select the most-free visible
physical GPU among indices 1 and 2 only when at least 90% is free. Require all
six smoke cells to produce three successful rows, with adaptation applied and
the expected selectors and native metrics. Only then start the full run in the
same pane. Full execution continues past individual failures, and the final
summary reports all missing, failed, skipped, OOM, or fallback rows explicitly.

## Acceptance Criteria

- Default matrix manifest contains exactly two models, three methods, and six
  uniquely routed cells.
- Every full cell discovers 67 unique tasks; no many-class skip is expected.
- F-aware rows persist `tabicl_encoded_l2` for TabICL and `raw_l2` for TabPFN.
- Paired statistics exclude failures, missing rows, fallbacks, and unapplied
  adaptation, while reporting those conditions separately.
- Explicit data roots and explicit legacy model axes remain supported.
