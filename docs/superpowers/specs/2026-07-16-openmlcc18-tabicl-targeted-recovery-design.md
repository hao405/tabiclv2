# OpenML-CC18 TabICL v2 Targeted Recovery Design

## Summary

Recover the interrupted TabICL v2 cells in the existing OpenML-CC18 2x3 run:

```text
results/managed_experiments/openmlcc18_2x3_full_20260715_223032
```

The completed TabPFN v3 cells and the 52--53 successful TabICL v2 rows per
method remain unchanged. Recovery runs only the missing and failed TabICL v2
tasks in isolated Python processes on physical GPU 2 in `jiqun tmux zh:2`.
Validated results are merged atomically into the existing run ID.

The default launcher remains TabICL v2 / TabPFN v3 across `infer`, `ft`, and
`faware_ft` on the 67-task OpenML-CC18 view. Single-cell launches continue to
support the legacy models, and explicit matrix axes can still request the old
4x3 matrix.

## Existing State and Recovery Scope

The authoritative task list is the 67 included datasets in:

```text
results/dataset_views/openml_cc18_max10/dataset_view_manifest.json
```

Each TabICL v2 result CSV currently contains 54 real dataset rows plus a
`__WORKER_EXIT__0` sentinel caused by worker termination with exit code -15.
All three methods are missing the same 13 datasets. Existing failures add one
retry for `infer`, two for `ft`, and one for `faware_ft`:

- `infer`: 13 missing plus 1 failed, for 14 isolated launches.
- `ft`: 13 missing plus 2 failed, for 15 isolated launches.
- `faware_ft`: 13 missing plus 1 failed, for 14 isolated launches.
- Total: at most 43 isolated launches.

The observed failed datasets include `OpenML-ID-40927` (CIFAR-10) and
`OpenML-ID-40978` (Internet-Advertisements). Their existing failures are CUDA
OOMs, so they must be retried in clean processes without changing experiment
hyperparameters.

## Recovery Orchestration

Add an OpenML-specific recovery sidecar. It reads the view manifest and each
TabICL v2 result CSV, discards synthetic worker sentinel rows, and computes the
union of missing datasets and real rows whose status is not `ok`.

Each dataset-method pair runs in a separate Python process. This resets the
CUDA context between tasks and prevents memory retained or fragmented by one
dataset from affecting the next. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to improve allocation
behavior without changing the model or evaluation contract.

Recovery preserves the full-run parameters:

- model: TabICL v2;
- random seed: 42;
- inference estimators: 32;
- TTT epochs: 30;
- TTT learning rate: `1e-5`;
- TTT patience: 8;
- FT selector: `random`;
- F-aware selector: `f_mmd`;
- F-aware metric: `tabicl_encoded_l2`;
- one worker on physical GPU 2.

All recovery artifacts are written beneath a separate
`openmlcc18_2x3_full_20260715_223032_recovery_attempt_N` root. A durable
attempt manifest records the requested task-method pairs, commands, process
exit codes, result paths, validation status, timestamps, and source run ID.
On restart, valid successful single-task artifacts are reused and only
unfinished or unsuccessful pairs run again.

## Result Validation and Atomic Merge

A successful single-task artifact must contain exactly one row whose dataset
name matches the request. It must not contain a worker sentinel. Successful
rows require finite accuracy and balanced accuracy. FT and F-aware rows also
require `ttt_applied=True`, the expected selector, `tabicl_encoded_l2`, and no
undeclared OOM fallback.

After one method's recovery attempts finish, construct a candidate 67-row CSV
in view-manifest order:

1. retain every existing successful real row;
2. replace existing failed rows with their latest recovery rows;
3. fill missing datasets from their latest recovery rows;
4. remove all worker sentinel rows;
5. reject duplicate, unexpected, or still-missing dataset names.

Back up the original CSV before the first merge. Write the candidate to a
temporary sibling file, validate its schema, uniqueness, exact 67-task
coverage, and audit fields, then atomically rename it over both the worker and
aggregate result CSVs. Regenerate the cell summary from the merged rows.

Update each cell manager manifest with recovery provenance, backup paths,
counts, and the persistent failure list. Mark a cell `ok` only when all 67 rows
are successful; otherwise mark it `fail`. Recompute the matrix manifest from
all six cells under the same rule. Never relabel an OOM, fallback, or unapplied
TTT row as successful.

If an isolated clean-process retry still OOMs, retain the latest explicit
failure row and merge the other valid recovery results. Do not lower the
estimator count, alter the dataset, or change any other experimental
hyperparameter. Downstream paired statistics exclude persistent failures via
the shared-`status=ok` rule and report them separately.

## Validation and Remote Rollout

Local tests cover recovery inventory (`14/15/14`), sentinel removal, successful
row preservation, retry precedence, deterministic manifest ordering, duplicate
rejection, atomic backup and replacement, persistent OOM handling, and
idempotent resume. Run Python compilation, targeted pytest, and a complete
recovery dry-run before remote rollout.

On `jiqun`, preserve unrelated remote edits and verify hashes for only the
recovery sidecar and any required OpenML summarizer changes. Use the existing
`zh:2` pane, confirm it is attached to compute node `gpu2`, and require physical
GPU 2 to have at least 90% free memory. Do not use physical GPUs 0 or 1 and do
not preempt another process.

Use one small missing dataset across `infer`, `ft`, and `faware_ft` as a
preflight with the full production hyperparameters. Its valid artifacts count
toward the recovery attempt. After the preflight passes, continue the remaining
isolated launches. Because each result is persisted immediately, another GPU
loss resumes from the first unfinished pair.

After merging, run the OpenML paired summarizer on the original run ID. It must
produce coverage and exception audits plus accuracy and balanced-accuracy
comparisons for `FT-Infer` and `F-aware-FT`, using only dataset-name
shared-`status=ok` pairs.

## Acceptance Criteria

- Recovery targets only TabICL v2 missing or failed rows in the existing
  OpenML-CC18 run; completed TabPFN v3 results are not rerun or modified.
- Recovery inventory is exactly 14 `infer`, 15 `ft`, and 14 `faware_ft`
  dataset-method pairs before reusable recovery artifacts are considered.
- Every launch uses an isolated process, physical GPU 2, and the original full
  hyperparameters.
- Merged TabICL v2 CSVs contain exactly 67 unique real dataset rows and no
  worker sentinel rows.
- Successful FT and F-aware rows pass the TTT selector, metric, application,
  finite-value, and fallback audits.
- Persistent OOMs remain explicit failures; no hyperparameter downgrade is
  used to manufacture full coverage.
- Original CSVs are backed up, merge replacement is atomic, and the attempt and
  manager manifests provide complete recovery provenance.
- Final paired summaries use only shared successful datasets and report every
  failure, OOM, fallback, unapplied TTT row, and coverage difference.
