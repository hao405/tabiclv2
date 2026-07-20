# OpenML-CC18 PEFT Non-OOM Repair and Targeted Backfill Design

## Scope

Repair only the five non-OOM failures in the completed OpenML-CC18 PEFT
matrix `openmlcc18_peft_2x3_full_20260717_113939`:

- TabICLv2 `lora`, `last_layers`, and `ln_head_embedding` on
  `OpenML-ID-40996`
- TabPFNv3 `last_layers` on `OpenML-ID-40978` and `OpenML-ID-4134`

Existing successful rows and every existing OOM or OOM-fallback row are
immutable. The repair keeps the original checkpoints, seed 42, 30 epochs,
learning rate `1e-5`, patience 8, two fine-tuning estimators, two validation
estimators, and 32 final inference estimators.

## Runtime Fixes

TabPFN exposes `--ttt-auto-scale-n-estimators`. Its default preserves the
existing generic runner behavior. The PEFT bridge explicitly disables it so
high-dimensional tasks retain the required `2/2/32` estimator configuration
instead of silently scaling fine-tuning estimators and violating the
fine-tuning loss shape contract. The vendored loss implementation is not
changed.

TabICL PEFT exposes two bounded-memory controls:

- `--ttt-max-feature-cells-per-chunk`, default `2000000`
- `--tabicl-predict-batch-size`, default `2048`

A zero value disables the corresponding bound and restores the previous
behavior. For a positive feature-cell budget, the effective TTT chunk size is:

```text
max(min_chunk_size,
    min(max_chunk_size, floor(feature_cell_budget / n_features)))
```

Final `predict_proba` is evaluated in deterministic row batches when the
prediction batch size is positive. Concatenated batched probabilities must
match the former whole-array result within numerical tolerance.

Result rows and logs record the requested and effective chunk size, feature
cell budget, prediction batch size and count, execution phase, and TabPFN
estimator auto-scale setting.

## Recovery and Transaction Semantics

The recovery sidecar derives its inventory from the formal matrix and requires
exactly the five logical targets above after excluding OOM failures. Attempts
are stored under:

```text
results/PEFT/_recovery/openmlcc18_peft_2x3_full_20260717_113939_nonoom_logicfix_20260720
```

The ordered preflight is TabPFNv3 `40978`, TabICLv2
`40996/last_layers`, and then the remaining three targets. A successful
preflight artifact is reused as the formal attempt.

If a TabICL target's first attempt reports worker `SIGTERM(-15)`, that target
receives exactly one resource-backoff retry using one million feature cells
per TTT chunk and a 1024-row prediction batch. OOM failures never trigger this
retry.

No formal matrix artifact is changed unless all five final attempts satisfy:

- `status=ok`
- `ttt_applied=True`
- `peft_trainable_params > 0`
- no OOM or fallback

After all five pass, the recovery process locks source hashes, stages all four
affected cell CSVs and summaries, records a global journal and snapshots, and
commits the complete set. A crash can only be recovered by completing the
prepared commit or restoring every pre-write artifact. Re-running the sidecar
is idempotent.

After a successful commit, the OpenML PEFT audit is regenerated and the local
`result/` paper archive is first dry-run and then rebuilt. If any target
remains invalid, all attempt diagnostics are retained but the formal matrix
and paper archive remain unchanged.

## Verification

Local verification covers the feature-cell boundary, disabled compatibility,
batched prediction equivalence, fixed TabPFN `2/2/32` estimator routing, exact
five-target inventory, OOM exclusion, one-time SIGTERM backoff, all-or-nothing
transactions, hash conflicts, crash recovery, and idempotent resume. It also
runs Python compilation, shell syntax checks, targeted tests, recovery
dry-run, and `git diff --check`.

Remote execution uses `jiqun tmux zh:1`, physical GPU2, one worker, and
`--gpu-groups 2`. It waits until GPU2 has at least 90 percent free memory and
does not preempt an existing process. The master log is:

```text
/home/crc00006699/zh/tabiclv2/logs/openmlcc18_peft_nonoom_logicfix/master.log
```

## Acceptance Criteria

- All six formal CSVs contain 67 unique real tasks.
- The five targets are strictly valid and have no OOM or fallback.
- TabICLv2 `OpenML-ID-40996` no longer exits with `SIGTERM(-15)`.
- The two TabPFNv3 tasks keep two fine-tuning estimators and do not raise the
  estimator-count assertion.
- Every non-target success row and every pre-existing OOM row remains
  field-for-field unchanged.
- Expected strict counts become TabICLv2 `60/65/61` and TabPFNv3 `56/64/58`
  for `lora/last_layers/ln_head_embedding`; the six-cell shared strict
  intersection remains 56.
