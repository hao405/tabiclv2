# TabICLv2 / TabPFNv2 PEFT Matrix Design

## Goal

Extend `PEFT_Tabicl/1C_Chunk_PEFT.py` into the single CLI for running TabICLv2
and TabPFNv2 with `lora`, `last_layers`, and `ln_head_embedding`. Preserve the
existing single-trial behavior, add an explicit 2 x 3 sequential matrix mode,
and isolate every result under `results/PEFT`.

## Architecture

The runner owns dataset discovery, metrics, worker orchestration, output
artifacts, resume state, and matrix scheduling. Model-specific behavior is
isolated behind TabICL and TabPFN backends. Matrix order is TabICLv2's three
methods followed by TabPFNv2's three methods, with model and CUDA resources
released between trials.

The method semantics are architecture-equivalent rather than name-equivalent:

- `lora` freezes the backbone and injects trainable low-rank updates into
  linear and attention projection weights.
- `last_layers` trains the final N transformer blocks and prediction head.
- `ln_head_embedding` trains LayerNorm, input/target embeddings, and the
  prediction head.

TabPFN's local finetuning base receives a protected no-op hook immediately
before optimizer construction. The PEFT subclass overrides that hook to freeze
or parameterize the loaded model. Existing TabPFN runners retain their current
behavior.

## CLI and Outputs

Add `--model-family {tabiclv2,tabpfnv2,all}`, extend
`--ttt-peft-method` with `all`, and add `--output-root`, `--run-name`,
`--resume`, `--tabicl-model-path`, and `--tabpfn-model-path`. Defaults remain a
single TabICLv2 LoRA trial for backward safety. Ambiguous legacy `--model-path`
and `--out-dir` arguments are rejected in matrix mode.

Each trial writes to
`results/PEFT/<model>/<method>/<run-name>/`; matrix metadata writes to
`results/PEFT/_matrix/<run-name>/`. Trial output includes worker and aggregate
CSV files, `summary.txt`, `run_config.json`, logs, and optional checkpoints.
Matrix output includes a manifest, aggregate CSV, summary, and launcher log.
Existing successful complete trials are skipped only with `--resume`; otherwise
an existing run name is protected from overwrite.

The six trials share epochs, learning rate, weight decay, query ratio, random
seed, early stopping, and estimator settings. Output telemetry distinguishes
successful applied PEFT from skipped, failed, and inference fallback rows.

## Validation and Execution

Unit tests cover exact trainable parameter sets for both architectures, LoRA
checkpoint merging, the TabPFN hook's default behavior, matrix expansion,
output routing, overwrite protection, and resume decisions. Local validation is
`py_compile`, CLI help, and focused pytest.

Remote validation uses `jiqun -> tmux zh:1 -> gpu2`, with one worker assigned
to physical GPU 2. A six-trial one-dataset smoke using `Basketball_c` and reduced
epochs/estimators must produce `status=ok`, `ttt_applied=true`, nonzero
trainable parameter counts, and valid metrics for every combination. Only then
is the full `data184` matrix launched in the same tmux window. The launch is
accepted only after checking the live process, CUDA visibility, GPU memory,
logs, and manifest state.

OOM inference fallback remains observable but is not counted as PEFT success.
Unsupported TabPFNv2 datasets are marked `skipped`. Cross-model comparisons use
only the shared `status=ok` and `ttt_applied=true` intersection.
