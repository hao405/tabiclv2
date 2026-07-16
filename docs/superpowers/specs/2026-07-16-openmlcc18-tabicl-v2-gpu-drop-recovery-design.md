# OpenML-CC18 TabICL v2 GPU-Drop Recovery Design

## Objective

Recover the interrupted TabICL v2 cells in
`openmlcc18_2x3_full_20260715_223032` without rerunning successful tasks or the
three complete TabPFN v3 cells. Run only on physical GPU 2 in `jiqun` tmux
`zh:2`, merge validated recovery rows into the original run atomically, and
then regenerate the paired OpenML summary.

The experiment remains the default TabICL v2 / TabPFN v3 by `infer`, `ft`, and
`faware_ft` 2x3 matrix on the 67-task OpenML-CC18 view with at most ten classes.
Single-cell launches and explicit legacy 4x3 matrix axes remain supported.

## Existing State

The smoke run has six complete three-task cells. In the full run, all three
TabPFN v3 cells contain 67 result rows. Each TabICL v2 cell contains 54 real
dataset rows plus a `__WORKER_EXIT__0` sentinel caused by worker exit code -15.
The cells share 13 missing tasks and contain one or two failed tasks. Successful
TabICL rows are retained.

The ordinary matrix resume path is insufficient: it resumes at cell granularity,
and the interrupted TabICL inference manager was persisted as successful despite
incomplete dataset coverage.

## Recovery Architecture

Add a dedicated OpenML recovery manager rather than changing generic matrix
resume semantics. The manager reads the 67 expected names from the dataset view
manifest and audits each TabICL v2 result CSV independently. It ignores synthetic
worker sentinel rows, retains unique real rows whose status is `ok`, and defines
the recovery set as every missing or non-ok dataset.

For each of `infer`, `ft`, and `faware_ft`, the manager creates a validated
symlink-only dataset view containing exactly that method's recovery set. It runs
the existing single-cell launcher with the original model, seed, estimator,
checkpoint, and TTT parameters into a separate recovery result root. TabPFN v3
outputs are read-only and are never relaunched.

After a recovery cell finishes, the manager requires one unique successful row
for every requested dataset and no unexpected row. It then combines retained and
recovered rows in dataset-view order. Only a complete 67-row result is eligible
to replace the original cell artifacts.

## Parameters and Audit Contract

Recovery preserves seed 42, TabICL v2 with 32 estimators, one worker, TTT 30
epochs, learning rate `1e-5`, and patience 8. FT uses selector `random`; F-aware
FT uses `f_mmd`. Both adaptation methods persist `tabicl_encoded_l2`, and all
successful adaptation rows must report `ttt_applied=True` without OOM fallback.
After a confirmed OOM at the original inference batch size 8 on an otherwise
empty 24 GiB GPU, the approved recovery retry uses batch size 4 while retaining
32 estimators and every other experiment parameter.

The recovery manifest records the original run ID, source and recovery data
roots, expected, retained, retried, and replaced dataset names, commands, GPU
audit, source CSV hashes, recovery CSVs, backup artifacts, timestamps, and final
validation status.

## Atomicity and Failure Handling

Recovery is restartable per method. A failed or incomplete recovery attempt is
left isolated and does not modify the original cell. Before the first merge, the
manager makes non-overwriting timestamped backups of the original result CSV,
summary, and manager manifest.

Merged CSV, summary, and manifest files are written beside their destinations as
temporary files, flushed, and atomically renamed. A merge is rejected if it has
duplicates, sentinels, missing or unexpected datasets, non-ok rows, invalid TTT
metadata, or fallback rows. Repeated execution recognizes already validated
67-row cells and skips them.

The matrix manifest is marked successful only after all six cells pass exact
67-task coverage and method-specific metadata audits. Until then, its status
continues to reflect an incomplete or failed run.

## Validation and Remote Rollout

Local tests use temporary manifests and result CSVs to cover recovery-set
calculation, sentinel removal, failed-row replacement, dataset ordering,
duplicate and unexpected-row rejection, partial recovery isolation, atomic
merge, backup idempotence, and already-complete resume behavior. Static checks
include Python compilation and targeted pytest. A dry run must print the three
method-specific recovery sets and commands without changing experiment results.

Before remote execution, verify local and remote script hashes, the OpenML view,
checkpoints, conda environment, and the current six-cell results. Confirm that
tmux `zh:2` is attached to compute node `gpu2` and that physical GPU 2 has at
least 90 percent free memory; do not fall back to GPU 0 or 1. If GPU 2 is busy,
wait in the same pane without preempting another process.

Run the three isolated recovery cells sequentially on physical GPU 2. After each
cell, validate its recovery output before merging. Finally audit six cells by 67
unique tasks, update the matrix manifest, and run the OpenML paired summarizer.
Preserve complete logs and a failure inventory even when final summarization
cannot proceed.

## Acceptance Criteria

- The original run ID remains `openmlcc18_2x3_full_20260715_223032`.
- Existing successful TabICL rows and all TabPFN v3 outputs remain unchanged.
- Each TabICL v2 cell ends with exactly 67 unique real datasets, all `status=ok`,
  and no worker sentinel.
- FT and F-aware FT have valid adaptation, selector, native metric, and fallback
  metadata on all 67 rows.
- The six-cell matrix manifest is successful and traceable to recovery artifacts.
- Paired statistics use shared successful datasets only and produce coverage,
  paired detail, paired summary, and human-readable summary artifacts.
