# OpenML-CC18 PEFT 2×3 Matrix Implementation Plan

**Design:** `docs/superpowers/specs/2026-07-17-openmlcc18-peft-2x3-matrix-design.md`

**Goal:** Make the validated 67-task OpenML-CC18 view the default dataset for
the existing TabICLv2 / TabPFNv3 × three-PEFT-method matrix, add a guarded
smoke-to-full launcher and result audit, validate locally, and start the full
run in `jiqun tmux zh:1` on physical GPU 1 and GPU 2.

## Constraints

- Preserve all unrelated dirty-worktree changes.
- Build on the existing uncommitted TabPFNv3 PEFT backend work in
  `PEFT_Tabicl/1C_Chunk_PEFT.py`; do not replace or revert it.
- Do not alter the fixed six-cell model-major matrix order.
- Keep TabPFNv2 and explicit `data184` single-model compatibility.
- Do not overwrite existing `data184` or OpenML result directories.
- Do not use GPU 1 or GPU 2 while either is occupied by another process.

## Task 1: Establish the PEFT Runner Baseline

**Files:**

- Inspect: `PEFT_Tabicl/1C_Chunk_PEFT.py`
- Inspect: `tests/test_peft_tabicl_runner.py`
- Inspect: `results/dataset_views/openml_cc18_max10/dataset_view_manifest.json`

**Steps:**

1. Record `git status --short` and the current diff for the PEFT runner and its
   tests.
2. Confirm the current matrix expansion is exactly:
   `tabiclv2/{lora,last_layers,ln_head_embedding}` followed by
   `tabpfnv3/{lora,last_layers,ln_head_embedding}`.
3. Confirm the existing CLI can explicitly accept `--data-root data184`.
4. Validate the local OpenML view manifest contains 67 unique included tasks,
   `max_classes=10`, and valid dataset-directory symlinks.
5. Run the existing targeted PEFT tests before changing the OpenML interface.

**Verification:**

```bash
python -m pytest tests/test_peft_tabicl_runner.py -q
```

## Task 2: Change the Default Dataset Without Breaking Explicit Roots

**Files:**

- Modify: `PEFT_Tabicl/1C_Chunk_PEFT.py`
- Modify: `tests/test_peft_tabicl_runner.py`

**Steps:**

1. Add a repository-relative constant for
   `results/dataset_views/openml_cc18_max10`.
2. Make that value the default for `--data-root`.
3. Keep path resolution behavior unchanged for explicitly supplied roots.
4. Add tests asserting:
   - A default parse selects the OpenML view.
   - `--data-root data184` remains unchanged.
   - Matrix expansion remains exactly six cells.
   - Explicit `tabpfnv2` remains accepted.

**Verification:**

```bash
python -m pytest tests/test_peft_tabicl_runner.py -q
python PEFT_Tabicl/1C_Chunk_PEFT.py --help
```

## Task 3: Add a Reusable OpenML View Validator

**Files:**

- Add: `scripts/validate_openmlcc18_peft_view.py`
- Add: `tests/test_validate_openmlcc18_peft_view.py`

**Interface:**

```text
python scripts/validate_openmlcc18_peft_view.py \
  --data-root results/dataset_views/openml_cc18_max10 \
  --expected-datasets 67
```

**Steps:**

1. Read `dataset_view_manifest.json`.
2. Validate `max_classes=10`, 67 included tasks, unique names, and the expected
   OpenML source root.
3. For every included task, validate that the view entry is a symlink to an
   existing directory containing `info.json` and the required train/validation/
   test NumPy arrays.
4. Reject extra dataset directories not represented in the included manifest.
5. Print a compact machine-readable JSON audit for the wrapper.
6. Unit-test valid, duplicate, missing-link, missing-array, wrong-count,
   wrong-class-limit, and extra-directory cases.

**Verification:**

```bash
python -m pytest tests/test_validate_openmlcc18_peft_view.py -q
python scripts/validate_openmlcc18_peft_view.py \
  --data-root results/dataset_views/openml_cc18_max10 \
  --expected-datasets 67
```

## Task 4: Add a PEFT Matrix Auditor

**Files:**

- Add: `scripts/summarize_openmlcc18_peft_matrix.py`
- Add: `tests/test_summarize_openmlcc18_peft_matrix.py`

**Interface:**

```text
python scripts/summarize_openmlcc18_peft_matrix.py \
  --matrix-root results/PEFT/_matrix/<run_name> \
  --view-manifest results/dataset_views/openml_cc18_max10/dataset_view_manifest.json \
  --expected-datasets 67
```

For smoke, pass `--expected-datasets 3 --require-all-success`.

**Steps:**

1. Read the matrix manifest and verify the exact six-cell axes and order.
2. Resolve each trial output directory without assuming it is under the matrix
   directory.
3. Load each `all_classification_results.csv`.
4. Reject duplicate datasets, unexpected names, `__WORKER_EXIT__` rows, and
   unexplained coverage gaps.
5. Audit per-cell row count, status, `ttt_applied`, OOM fallback, PEFT method,
   model family, trainable parameter count/ratio, and TabPFNv3 checkpoint route.
6. Compute metrics only over rows with `status=ok`, `ttt_applied=True`, and
   `peft_trainable_params > 0`.
7. Compute the shared eligible dataset intersection across all six cells.
8. Write:
   - `coverage.csv`
   - `failures.csv`
   - `eligible_results.csv`
   - `metric_summary.csv`
   - `audit_summary.md`
9. In strict smoke mode, return non-zero unless all 18 rows satisfy the complete
   PEFT contract.
10. Test missing, failed, skipped, OOM, fallback, duplicate, wrong-method,
    wrong-model, zero-trainable-parameter, checkpoint-routing, and shared-set
    behavior.

**Verification:**

```bash
python -m pytest tests/test_summarize_openmlcc18_peft_matrix.py -q
```

## Task 5: Add the Guarded Smoke-to-Full Wrapper

**Files:**

- Add: `scripts/run_openmlcc18_peft_matrix.sh`
- Add: `tests/test_run_openmlcc18_peft_matrix.py`

**Presets:**

- `smoke`: three fixed OpenML tasks covering binary and multiclass,
  one epoch, two estimators, patience one, strict six-cell audit.
- `full`: all 67 view tasks, 30 epochs, learning rate `1e-5`, weight decay
  `0.01`, query ratio `0.2`, patience eight, established PEFT defaults.
- `smoke-then-full`: run smoke, audit it, and start full only after a successful
  strict audit.

**Steps:**

1. Accept environment overrides for run names, output root, log root, devices,
   and dry-run mode.
2. Always run the OpenML view validator before launching.
3. Materialize a deterministic read-only three-task smoke view from symlinks;
   do not copy datasets.
4. Launch the existing runner with:

   ```text
   --model-family all
   --ttt-peft-method all
   --workers 2
   --gpu-groups 1;2
   ```

5. Use independent smoke and full run names containing `openmlcc18_peft_2x3`.
6. Append stdout and stderr to persistent launcher logs.
7. Run the matrix auditor after smoke and full.
8. Do not start full if smoke fails.
9. Preserve the runner's `--resume` semantics for an explicitly reused run name.
10. Test command construction, quoting of `1;2`, run-name separation, dry-run,
    validation failure, smoke failure, and smoke-to-full gating.

**Verification:**

```bash
bash -n scripts/run_openmlcc18_peft_matrix.sh
python -m pytest tests/test_run_openmlcc18_peft_matrix.py -q
DRY_RUN=1 bash scripts/run_openmlcc18_peft_matrix.sh smoke-then-full
```

## Task 6: Complete Local Validation

**Files:**

- Verify all modified and added files from Tasks 2–5.

**Steps:**

1. Compile all affected Python files.
2. Run shell syntax validation.
3. Run targeted test suites.
4. Run the real local view validator.
5. Run the wrapper dry-run and inspect all six expanded cells.
6. Run an explicit `data184` dry-run to confirm backward compatibility.
7. Run `git diff --check`.

**Verification:**

```bash
python -m py_compile \
  PEFT_Tabicl/1C_Chunk_PEFT.py \
  scripts/validate_openmlcc18_peft_view.py \
  scripts/summarize_openmlcc18_peft_matrix.py
bash -n scripts/run_openmlcc18_peft_matrix.sh
python -m pytest \
  tests/test_peft_tabicl_runner.py \
  tests/test_validate_openmlcc18_peft_view.py \
  tests/test_summarize_openmlcc18_peft_matrix.py \
  tests/test_run_openmlcc18_peft_matrix.py -q
git diff --check
```

## Task 7: Synchronize and Validate in `jiqun tmux zh:1`

**Remote root:** `/home/crc00006699/zh/tabiclv2`

**Steps:**

1. Recheck the exact `zh:1` pane and ensure it is at an idle shell.
2. Probe physical GPU 1 and GPU 2 from inside the nested compute-node shell.
3. If either GPU is busy, wait in the same pane; do not preempt.
4. Check remote dirty files before synchronization. Back up a remote target
   only if it differs and cannot be safely reconciled.
5. Synchronize only:
   - `PEFT_Tabicl/1C_Chunk_PEFT.py`
   - `scripts/validate_openmlcc18_peft_view.py`
   - `scripts/summarize_openmlcc18_peft_matrix.py`
   - `scripts/run_openmlcc18_peft_matrix.sh`
6. Compare local and remote SHA-256 hashes.
7. Activate the `tabicl` conda environment.
8. Run remote static checks and the OpenML view validator.
9. Start `smoke-then-full` in `zh:1`.

**Remote launch shape:**

```bash
cd ~/zh/tabiclv2
conda activate tabicl
GPU_GROUPS='1;2' \
bash scripts/run_openmlcc18_peft_matrix.sh smoke-then-full
```

## Task 8: Verify Smoke and Full Startup

**Steps:**

1. Confirm the smoke matrix has six successful trials and 18 eligible rows.
2. Confirm both binary and multiclass TabPFNv3 checkpoints were exercised.
3. Confirm every smoke row has non-zero trainable parameters and no fallback.
4. Confirm the wrapper started a distinct full run only after smoke success.
5. Capture the full run name and paths for:
   - Matrix manifest
   - Launcher log
   - Current trial output
   - Audit directory
6. Confirm two worker processes are bound to physical GPU 1 and GPU 2.
7. Confirm both GPUs show memory use from the intended processes.
8. Confirm the full matrix manifest contains the six planned cells and the first
   cell is running.
9. Report the run ID and authoritative remote paths to the user.

## Full-Run Completion Contract

When the long run completes:

1. Run the matrix auditor even if a dataset or cell failed.
2. Require 67 discovered unique tasks per cell or list every discrepancy.
3. Report cell-level status, ok/fail/skip, OOM/fallback, and shared eligible
   coverage.
4. Do not include failed, missing, fallback, or unapplied rows in metric means.
5. Preserve complete failure lists and logs for later paired comparison with
   the existing OpenML `infer/ft/faware_ft` baseline.
