# TabICLv2 / TabPFNv3 PEFT 2×3 Matrix Design

## Summary

Extend `PEFT_Tabicl/1C_Chunk_PEFT.py` so the unified `all` matrix contains
TabICLv2 and TabPFNv3 with `lora`, `last_layers`, and
`ln_head_embedding`. Reuse the three completed TabICLv2 trials and run only
the missing TabPFNv3 trials. Preserve TabPFNv2 as an explicit single-model
compatibility backend.

The remote rollout must not interrupt the TabPFNv2 process currently using
physical GPU 2. After that process exits, run binary and multiclass smoke tests
in `jiqun` `tmux zh:1`, then automatically start the full `data184` run when
all smoke cases pass.

## Backend and CLI Design

The shared runner owns dataset discovery, worker scheduling, metrics, output
rows, manifests, resume behavior, logging, and device cleanup. Model-specific
behavior is separated into `TabICLBackend`, `TabPFNv2Backend`, and
`TabPFNv3Backend`.

`--model-family` accepts `tabiclv2`, `tabpfnv2`, `tabpfnv3`, and `all`.
`all` expands in this fixed order:

1. `tabiclv2/lora`
2. `tabiclv2/last_layers`
3. `tabiclv2/ln_head_embedding`
4. `tabpfnv3/lora`
5. `tabpfnv3/last_layers`
6. `tabpfnv3/ln_head_embedding`

TabPFNv2 remains available only through an explicit `tabpfnv2` selection.
The existing `--tabpfn-model-path` continues to mean the v2 checkpoint. New
`--tabpfn-v3-binary-model-path` and
`--tabpfn-v3-multiclass-model-path` arguments select the v3 checkpoint from
the loaded dataset task type. `--reuse-matrix-manifest` imports completed
TabICLv2 trials into a new matrix without copying their result directories.

## TabPFNv3 PEFT Policies

- `lora` freezes the base model and installs LoRA on v3 linear and attention
  projection weights. Only LoRA parameters are trainable.
- `last_layers` trains the final N `icl_blocks`, `output_norm`, and the
  task-specific output head: `output_projection` for binary classification or
  `many_class_decoder` for multiclass classification.
- `ln_head_embedding` trains normalization modules, `x_embed`,
  `col_y_encoder`, `icl_y_encoder`, and the task-specific output head.

All three methods retain the established defaults: 30 epochs, learning rate
`1e-5`, weight decay `0.01`, query ratio `0.2`, seed 42, and early stopping.
LoRA uses rank 4, alpha 8, and dropout 0.

## Reuse, Output, and Failure Semantics

A source matrix may still be globally `running`; each reused TabICLv2 trial
must independently have status `success` and non-empty
`all_classification_results.csv`, `summary.txt`, and `run_config.json`.
Reused manifest entries record the source manifest, source output directory,
and validation summary.

New trial outputs use
`results/PEFT/<model>/<method>/<run-name>/`; matrix artifacts use
`results/PEFT/_matrix/<run-name>/`. Cross-model summaries use only the common
datasets satisfying `status=ok` and `ttt_applied=true`.

Smoke is fail-fast and requires every row to be successful, actually adapted,
and backed by non-zero trainable parameters. Full runs continue after a failed
method while recording failure in the manifest. OOM fallback remains
`ttt_applied=false` with `ttt_oom_fallback=true`; unsupported inputs are
`skipped` rather than reported as baseline PEFT results. Resume skips only
successful trials whose required artifacts remain complete.

## Validation and Remote Rollout

Unit tests use fake binary and multiclass v3 architectures to verify exact
trainable parameter sets, counts, ratios, output-head routing, LoRA checkpoint
merging, matrix order, compatibility, reuse validation, overwrite protection,
and resume behavior. Local validation consists of `py_compile`, CLI help, and
targeted pytest.

Remote files are synchronized to `~/zh/tabiclv2` and checked with SHA-256.
The launcher waits until the existing command in `tmux zh:1` has returned to
the remote shell before sending any input. Smoke uses one worker on physical
GPU 2, one-epoch/one-estimator settings, and a temporary data root containing
`Basketball_c` and `cmc`. The three TabPFNv3 methods therefore exercise both
binary and multiclass checkpoints. After all six smoke rows pass, the same
pane starts the full `data184` TabPFNv3 three-method run and builds the new
matrix from those trials plus the reused TabICLv2 trials.

The full launch is accepted only after verifying the process tree, device
visibility, physical GPU 2 memory use, launcher log, and current manifest
trial. Completed trials must contain 184 rows, no worker crash, non-zero
applied PEFT coverage, and auditable skip/fallback rows.

## Constraints

- Preserve unrelated worktree changes and existing result directories.
- Use the existing local TabPFNv3 binary and multiclass checkpoints.
- Do not add a long-term shell wrapper; the Python runner remains the only
  experiment entrypoint.
- Refuse same-name output overwrite unless valid resume behavior is requested.
