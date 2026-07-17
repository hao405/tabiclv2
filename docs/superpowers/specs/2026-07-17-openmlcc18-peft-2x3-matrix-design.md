# OpenML-CC18 TabICLv2 / TabPFNv3 PEFT 2×3 Matrix Design

## Summary

Run the existing PEFT matrix on the validated OpenML-CC18 view containing the
67 classification tasks with at most 10 classes. The matrix remains:

- Models: `tabiclv2`, `tabpfnv3`
- PEFT methods: `lora`, `last_layers`, `ln_head_embedding`

Each cell uses two workers bound to physical GPU 1 and GPU 2. The six cells run
sequentially so different models do not contend for the same devices. A
three-task smoke matrix must pass before the full matrix starts in
`jiqun tmux zh:1`.

## Runner and Dataset Interface

Change the default data root in `PEFT_Tabicl/1C_Chunk_PEFT.py` to:

```text
results/dataset_views/openml_cc18_max10
```

Explicit `--data-root` values remain supported, including `data184`. Preserve
the existing single-model backends for TabICLv2, TabPFNv2, and TabPFNv3. The
matrix selection `--model-family all --ttt-peft-method all` continues to expand
exactly these six cells in model-major order:

1. `tabiclv2/lora`
2. `tabiclv2/last_layers`
3. `tabiclv2/ln_head_embedding`
4. `tabpfnv3/lora`
5. `tabpfnv3/last_layers`
6. `tabpfnv3/ln_head_embedding`

The PEFT runner remains dataset-agnostic. An OpenML-specific wrapper owns
dataset-view validation, smoke and full presets, rollout logging, and the
transition from smoke to full.

## OpenML Dataset Contract

Before launching either preset, the wrapper reads:

```text
results/dataset_views/openml_cc18_max10/dataset_view_manifest.json
```

It validates all of the following:

- The source is the existing `openml_cc18` benchmark root.
- The class limit is 10.
- The manifest contains exactly 67 included tasks.
- Included dataset names are unique.
- Every included entry has a valid read-only dataset-directory symlink.
- Each target contains the benchmark arrays and `info.json` required by the
  PEFT runner.

The wrapper refuses to launch if this contract is not satisfied. It does not
copy or modify the source OpenML data.

## Experiment Configuration

The full run keeps the established PEFT defaults:

- Random seed: 42
- Epochs: 30
- Learning rate: `1e-5`
- Weight decay: `0.01`
- Query ratio: `0.2`
- Early-stopping patience: 8
- LoRA rank: 4
- LoRA alpha: 8
- LoRA dropout: 0
- Last-layer tuning: final one ICL block
- TabPFNv3 checkpoint: selected from the task type using the existing binary
  and multiclass checkpoint arguments

Every cell uses:

```text
--workers 2 --gpu-groups '1;2'
```

Worker 0 is restricted to physical GPU 1 and worker 1 to physical GPU 2. The
workers divide the 67 datasets within one cell. Cells run sequentially.

## Smoke and Full Presets

The smoke preset uses three fixed OpenML tasks covering binary and multiclass
classification. It runs all six cells with:

- One epoch
- Two estimators
- Early-stopping patience of one
- Two workers on physical GPU 1 and GPU 2

Smoke succeeds only when all 18 result rows satisfy:

- `status=ok`
- `ttt_applied=True`
- `peft_trainable_params > 0`
- The persisted model family and PEFT method match the cell
- The TabPFNv3 checkpoint route matches the task type
- No OOM fallback, worker crash, or silent non-PEFT fallback occurred

The full preset starts automatically only after the smoke audit passes. It runs
all 67 tasks in all six cells. A dataset-level failure does not prevent later
datasets from running, and a failed cell is recorded before the launcher moves
to the next cell.

## Outputs and Auditing

Trial outputs use:

```text
results/PEFT/<model>/<method>/<run_name>/
```

Matrix artifacts use:

```text
results/PEFT/_matrix/<run_name>/
```

Run names are unique to the OpenML smoke and full launches, so existing
`data184` PEFT results are not overwritten.

The matrix audit records:

- Expected, observed, successful, failed, and skipped dataset counts per cell
- Missing and duplicate dataset names
- `ttt_applied`, OOM fallback, and worker-crash counts
- PEFT method, trainable parameter count, and trainable ratio
- Binary or multiclass checkpoint routing
- The shared successful dataset intersection across all six cells
- Accuracy, balanced accuracy, ROC-AUC, log loss, and runtime summaries
- Complete failure and missing-task lists

Only rows with `status=ok`, `ttt_applied=True`, and non-zero trainable
parameters enter aggregate PEFT metrics. Failures, missing rows, fallbacks, and
unapplied adaptations remain visible in the audit.

## Resume and Integrity Semantics

Using the same run name with `--resume` skips only a cell whose manifest and
required artifacts are complete and successful. Interrupted or failed cells
are rerun without changing completed cells.

Worker aggregation rejects:

- `__WORKER_EXIT__` sentinel rows
- Duplicate dataset names
- Rows outside the validated 67-task view
- Unexplained missing tasks

The full audit is still written when dataset-level failures occur. Such rows
are never represented as successful PEFT results or included in means.

## Validation and Remote Rollout

Local validation includes:

- Python compilation
- Shell syntax checking
- Targeted PEFT runner and wrapper tests
- Dataset-manifest validation tests
- Default OpenML and explicit `data184` compatibility checks
- A dry-run proving the fixed six-cell order and GPU assignments

Remote rollout uses `ssh jiqun`, the existing `tmux zh:1` pane, and the
`~/zh/tabiclv2` checkout:

1. Confirm that the pane is idle.
2. Confirm that physical GPU 1 and GPU 2 are available; wait rather than
   preempting another process if either device is busy.
3. Synchronize only the intended files and verify SHA-256 hashes.
4. Run the complete three-task smoke matrix.
5. Audit all 18 smoke rows.
6. Start the full 67-task matrix automatically after smoke passes.
7. Verify the process tree, device visibility, GPU memory use, launcher log,
   matrix manifest, and first active cell.

## Acceptance Criteria

- The default PEFT data root is the validated 67-task OpenML-CC18 view.
- Explicit `data184` and explicit single-model invocations remain compatible.
- Matrix expansion remains exactly TabICLv2 / TabPFNv3 × three PEFT methods.
- Smoke produces 18 valid PEFT rows with no fallback or worker failure.
- The full matrix discovers exactly 67 unique datasets in every cell.
- Both physical GPU 1 and GPU 2 are used as isolated dataset workers.
- Results and logs use independent OpenML run names and do not overwrite older
  PEFT experiments.
- All missing, failed, skipped, OOM, fallback, or unapplied rows are reported
  and excluded from aggregate PEFT metrics.
