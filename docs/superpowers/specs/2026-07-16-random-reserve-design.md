# `random_reserve` Context/Query Control Design

## Goal

Add `--ttt-c-selection random_reserve` to
`1C_Chunk_FT/1C_Chunk_FT_Faware_C.py` as a controlled ablation for
`f_mmd`. Both methods reserve the same number of global training rows as
context-only and share the same downstream context/query split. They differ
only in how the reserved indices are selected: uniform random sampling versus
distance from the unlabeled test distribution.

## Interface and behavior

- Extend `--ttt-c-selection` with `random_reserve`; preserve the existing
  `random` and `f_mmd` defaults and behavior.
- Reuse `--ttt-c-reserve-ratio`, whose default remains `0.05`.
- Normalize `random_reserve` to `c_source=none` and `c_metric=none`; it must not
  read validation or test features.
- Require `0 < reserve_ratio < 1` and
  `reserve_ratio + query_ratio < 1` for both reserved-context selectors.
- Sample `min(n-1, max(1, ceil(reserve_ratio*n)))` global training indices once
  per dataset. Derive an independent RNG from `random_state` and a fixed
  namespace so the reserve sample is reproducible but not coupled to epoch
  chunk shuffling.
- Keep the reserved set fixed across all epochs and chunks. Reserved rows may
  enter context but never query.

## Internal structure

- Keep selector-specific builders for the distance-ranked and random reserved
  sets.
- Extract one shared reserved-context splitter for global-to-local mapping,
  allowed-row query splitting, label-coverage repair, and failure handling.
- Preserve existing `f_mmd` split-strategy strings and behavior. Use
  `random_reserved_context:none:<split_strategy>` for the new method.
- Keep random-reserve distance telemetry empty. Record the selection, source,
  metric, ratio, and user-facing random seed in result metadata and output
  naming.

## Verification

- Add focused tests against the exact runner for CLI/config normalization,
  ratio validation, reserve count, deterministic sampling, seed variation,
  context/query membership, fallback behavior, empty distance telemetry, and
  output naming.
- Add regression coverage for ordinary `random` and existing `f_mmd` behavior.
- Run `py_compile` and the focused pytest file locally.
- Run a one-dataset GPU smoke on `ssh 238` from an isolated `/tmp` mirror. Do
  not overwrite the remote untracked runner. Use `data200/ada_prior`, one GPU,
  one worker, one estimator, one TTT epoch, no checkpoint saving, and no data
  parallelism. Validate the persisted CSV contract and absence of fallback/OOM;
  do not interpret its accuracy as a performance result.

## Scope boundaries

Do not change experiment matrix launchers, aggregators, or full-benchmark
method routing. A later performance trial may compare `random`,
`random_reserve`, and `f_mmd` under identical seeds and TTT hyperparameters.
