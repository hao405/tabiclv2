# TabICLv2 Graph-SCM Three-Stage Matrix Design

## Goal

Evaluate the existing fixed four-model by three-method TFM matrix on 512 classification datasets sampled directly from the TabICLv2 `graph_scm` prior. The launcher matrix, checkpoints, method routing, and default estimator counts remain unchanged; only `--data-root` changes.

## Dataset contract

The suite contains 171 Stage 1, 171 Stage 2, and 170 Stage 3 tasks. Stage settings match `scripts/train_v2_clf_stage{1,2,3}.sh`: Stage 1 uses 1,024 rows and a 0.3–0.9 train fraction; Stage 2 uses log-uniform 400–10,240 rows and a 0.79–0.81 train fraction; Stage 3 uses log-uniform 400–60,000 rows and the same train fraction. All stages use `graph_scm`, 1–100 features, 2–10 classes, no graph noise, both unpredictability filters, and 2–32 graph nodes.

Each task is exported as `N_train.npy`, `N_test.npy`, `y_train.npy`, `y_test.npy`, and `info.json`. `prior_manifest.csv` records the stage, deterministic group seed, sampled dimensions, split sizes, and prior configuration. Stage 1 retains groups of four tasks sharing prior hyperparameters; Stage 2 and Stage 3 use groups of one. Generation is resumable and atomically replaces incomplete or invalid task directories.

## Execution and analysis

`scripts/run_tabiclv2_graph_scm_matrix.sh` provides smoke and full presets. Smoke generates one task per stage and runs all 12 cells with two estimators and one TTT epoch. Full generation and execution start only after smoke succeeds, use seed 42 and GPU1 in `jiqun tmux zh:2`, and retain the launcher's normal TabICL/TabPFN estimator defaults, 30 TTT epochs, learning rate `1e-5`, and `standardized_l2`.

The stage-aware summarizer joins each result CSV to the manifest, reports coverage and TTT telemetry, and computes `FT−Infer` and `F-aware−FT` only on dataset-name intersections where both cells have `status=ok`. Results are reported overall and separately for all three stages; failed, missing, OOM, or fallback rows remain visible but do not enter paired means.

## Acceptance

The formal suite must contain exactly 512 unique, valid tasks with a `171/171/170` stage split. The matrix manifest must contain the unchanged four models, three methods, and 12 cells. All 12 smoke cells must complete with three `status=ok` rows before the full run starts. Full-run coverage and any fallback are reported exactly rather than inferred from directory names.
