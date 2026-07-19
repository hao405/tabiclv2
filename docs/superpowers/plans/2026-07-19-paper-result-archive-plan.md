# Paper Result Archive Implementation Plan

**Design:** `docs/superpowers/specs/2026-07-19-paper-result-archive-design.md`

**Goal:** Build a reproducible `result/` archive for the complete
`3 models × 3 datasets × 8 methods` paper matrix by selecting the best formal
artifacts from local `results/` and `jiqun`, preserving provenance and failures,
and generating method-only metric tables plus complete/missing audits.

## Constraints

- Preserve all unrelated dirty-worktree changes.
- Do not modify, move, delete, or rename any source artifact under local or
  remote `results/`.
- Do not send commands to or interrupt active `jiqun tmux zh:*` panes.
- Treat `ln_head_embedding` as its own method name; never label it Prefix.
- Select the best formal `FT` and `F-aware FT` by the approved deterministic
  policy, with coverage and success preceding mean Accuracy.
- Keep direct comparisons, W/L/T, and rankings out of the main Markdown result
  display.
- Preserve failed rows and report artifact, success, and adaptation coverage
  separately.
- Exclude smoke, debug, interrupted, compatibility-only, superseded Optuna,
  and unmerged recovery-fragment artifacts from selection.
- Build with standard-library Python plus the repository's existing `pandas`
  dependency; do not add a new package.

## Task 1: Establish the Source and Manifest Baseline

**Files:**

- Inspect: `analysis/experiment_results.md`
- Inspect: `results/`
- Inspect remotely: `jiqun:~/zh/tabiclv2/results/`
- Inspect: dataset manifests for `data184`, filtered `OpenML-CC18`, and
  `Graph-SCM`

**Steps:**

1. Record `git status --short`; do not alter existing user changes.
2. Capture current `jiqun` sessions and pane commands with
   `tmux list-windows` and `tmux capture-pane` only.
3. Build a read-only candidate listing of `all_classification_results.csv`,
   manifests, summaries, and best-parameter files under these roots:
   - local `results/tabicl/v2`, `results/tabpfn/v3`,
     `results/PEFT_results`, `results/合成数据实验`,
     `results/PEFT_compare`, and `results/limix`;
   - remote `results/managed_experiments`,
     `results/PEFT`, `results/PEFT_compare`, `results/limix`,
     `results/tabicl`, and `results/tabpfn`.
4. Exclude directories named or tagged `smoke`, `debug`, `interrupted`,
   `_legacy`, and raw per-dataset recovery attempts from the formal candidate
   listing.
5. Derive expected dataset-name sets:
   - `data184`: directory names containing valid dataset artifacts;
   - OpenML: included task names from
     `results/dataset_views/openml_cc18_max10/dataset_view_manifest.json`, with
     the corresponding remote manifest as a cross-check;
   - Graph-SCM: `dataset_name` values from the canonical 512-task manifest
     under `results/合成数据实验/data_manifest/`.
6. Fail the later build if a manifest set is unavailable or differs from the
   expected 184/67/512 counts; do not silently substitute a row count.
7. Save no output yet; this task supplies verified constants and fixtures for
   the builder.

**Verification:**

```bash
find data184 -mindepth 1 -maxdepth 1 -type d | wc -l
python - <<'PY'
import json
from pathlib import Path
p = Path("results/dataset_views/openml_cc18_max10/dataset_view_manifest.json")
d = json.loads(p.read_text())
print(len(d["included"]))
PY
ssh jiqun 'cd ~/zh/tabiclv2 && tmux list-windows -t zh'
```

## Task 2: Implement Candidate Discovery and Canonical Classification

**Files:**

- Add: `scripts/build_paper_result_archive.py`
- Add: `tests/test_build_paper_result_archive.py`

**CLI:**

```text
python scripts/build_paper_result_archive.py \
  --local-results results \
  --remote-host jiqun \
  --remote-results '~/zh/tabiclv2/results' \
  --out result
```

Additional modes:

```text
--dry-run       discover and select without writing result/
--local-only    skip SSH and classify remote-only cells as unavailable
--snapshot-time <ISO-8601>  fixed timestamp for deterministic tests
```

**Architecture:**

- `MatrixCell`: canonical `(model, dataset, method)` identity.
- `Candidate`: source host/path, run class, row/schema audit, coverage,
  success count, mean Accuracy, mtime, and selection flags.
- `LocalSourceBackend`: lists, reads, hashes, and copies local files.
- `SSHSourceBackend`: uses non-interactive `ssh` for listing/stat/hash and
  `rsync` for selected copies; it never uses `tmux send-keys`.
- `CandidateClassifier`: maps path and CSV metadata to canonical cells.
- `CandidateSelector`: applies the approved ordering and records rejection
  reasons.

**Canonical methods:**

```text
infer/base_infer/infer_baseline -> Infer
ft/random/ttt_baseline          -> FT
faware_ft/faware_c/f_mmd        -> F-aware FT
lora                            -> LoRA
ln_head_embedding               -> ln_head_embedding
last_layers/last_block          -> Last-block
localpfn                        -> LoCalPFN
micp                            -> MICP
```

The classifier must require both path evidence and compatible row/config
metadata for ambiguous `random` and `f_mmd` paths. Unknown or contradictory
artifacts go to provenance with `rejection_reason=unclassified` and cannot be
selected.

**Steps:**

1. Define the 72 expected cells in a fixed model/dataset/method order.
2. Implement allowlisted candidate roots and path-tag exclusion.
3. Read CSVs without rewriting them and validate:
   - a usable dataset identity column;
   - unique dataset identities;
   - `status`, Accuracy, adaptation, and fallback columns when present;
   - no `__WORKER_EXIT__` pseudo-row.
4. Record malformed or duplicate candidates as invalid rather than raising
   away the whole inventory.
5. Implement the selection tuple:

   ```text
   formal_class,
   unique_target_coverage,
   status_ok_count,
   mean_accuracy_on_status_ok,
   source_mtime
   ```

6. Prefer the larger value at every stage, with formal class encoded so a
   formal run always outranks excluded run classes.
7. Mark an Optuna/best-parameter result `selected_on_eval=true` unless its
   metadata identifies a disjoint selection split.
8. Unit-test aliases, unknown paths, excluded tags, duplicate rows, malformed
   CSVs, partial-high-mean rejection, and deterministic tie-breaking.

**Verification:**

```bash
python -m pytest tests/test_build_paper_result_archive.py \
  -k 'classif or candidate or select' -q
python scripts/build_paper_result_archive.py --local-only --dry-run
```

## Task 3: Implement Stable Remote Snapshot and Provenance

**Files:**

- Modify: `scripts/build_paper_result_archive.py`
- Modify: `tests/test_build_paper_result_archive.py`

**Steps:**

1. Represent source observations as `(size, mtime, sha256)`.
2. For a selected remote file:
   - observe it remotely;
   - `rsync` it to an output-side temporary file;
   - observe it remotely again;
   - hash the temporary copy;
   - accept only when both remote observations and the local copy hash agree.
3. Retry an unstable file at most three times.
4. If it remains unstable, remove only the temporary output file, classify the
   cell as `running`, and record its current remote row count and path.
5. For stable local files, hash before and after copy and apply the same
   acceptance rule.
6. Use temporary files and `os.replace` for all generated CSV, Markdown, JSON,
   and copied raw artifacts.
7. Deduplicate identical local/remote artifacts by SHA-256 while retaining both
   provenance rows.
8. Preserve differing candidates in provenance with explicit rejection
   reasons; never overwrite based only on the filename.
9. Mock both source backends in tests; unit tests must not require SSH.

**Verification:**

```bash
python -m pytest tests/test_build_paper_result_archive.py \
  -k 'snapshot or provenance or unstable or hash' -q
```

## Task 4: Implement Completeness Classification and the 72-Cell Inventory

**Files:**

- Modify: `scripts/build_paper_result_archive.py`
- Modify: `tests/test_build_paper_result_archive.py`

**Steps:**

1. For each selected CSV, compare unique dataset identities with its exact
   manifest set; record missing and unexpected names.
2. Distinguish:
   - `artifact_complete`;
   - `status_ok`;
   - `adaptation_expected` and `adaptation_applied`;
   - failure, skip, OOM, and fallback counts.
3. Set `overall_status` in this order:
   - `invalid` for parse/schema/duplicate-identity failures;
   - `running` for an unstable or partial artifact confirmed active;
   - `missing` when no qualifying artifact exists;
   - `partial` for a stable artifact missing target rows;
   - `complete_with_failures` for full membership with any failure, skip,
     fallback, or unapplied required adaptation;
   - `complete` otherwise;
   - `n/a` only from an explicit design-level applicability entry.
4. Do not infer `complete` from a summary file or directory name.
5. Generate exactly one `inventory.csv` row per expected cell with the fields
   specified in the design.
6. Generate `provenance.csv` for selected and rejected candidates.
7. Add tests for all status states, exact 72-row uniqueness, manifest
   mismatches, unexpected datasets, and adaptive/non-adaptive methods.

**Verification:**

```bash
python -m pytest tests/test_build_paper_result_archive.py \
  -k 'status or inventory or coverage or manifest' -q
```

## Task 5: Implement All-Metric Method Tables

**Files:**

- Modify: `scripts/build_paper_result_archive.py`
- Modify: `tests/test_build_paper_result_archive.py`

**Metric representation:**

Use long form for `method_metrics.csv`:

```text
dataset,model,method,metric,n,mean,std,min,max,source_path
```

In `method_metrics.md`, group by `dataset → model` and show one compact table
per metric family with methods as rows. Do not add delta, W/L/T, reference
method, or rank columns.

**Metric discovery rules:**

1. Treat identifiers, status/error fields, dataset-size metadata, seeds, and
   hyperparameters as non-metrics.
2. Explicitly exclude columns matching:
   - identifiers: dataset/model/method/path/worker/trial/run;
   - dataset metadata: `n_train`, `n_val`, `n_test`, `n_features`,
     `n_classes`, and split sizes;
   - configuration: seed/random-state, learning rate, epochs, steps, batch
     size, estimators, rank, ratio, patience, and context/query sizes.
3. Include every other numeric outcome or telemetry column with at least one
   finite value.
4. Normalize only approved aliases such as `acc -> accuracy`; keep
   semantically distinct metrics separate.
5. For each metric compute `n`, mean, sample standard deviation, minimum, and
   maximum over eligible rows.
6. Eligibility for method-only metrics is `status=ok`; adaptation coverage is
   shown separately and does not silently filter fallback rows.
7. Add explicit coverage metrics for result rows, `status=ok`, failures,
   fallback, and adaptation applied.
8. Render unavailable metrics as `—` in Markdown and absent rows in the
   long-form CSV.
9. Test metric inclusion, configuration exclusion, aliases, NaN handling,
   one-row standard deviation, and Markdown content.

**Verification:**

```bash
python -m pytest tests/test_build_paper_result_archive.py \
  -k 'metric or markdown' -q
```

## Task 6: Implement Reference-Only Comparisons and Rankings

**Files:**

- Modify: `scripts/build_paper_result_archive.py`
- Modify: `tests/test_build_paper_result_archive.py`

**Steps:**

1. Generate comparison pairs exactly as defined by the design.
2. Join each pair by canonical dataset identity and keep only shared
   `status=ok` rows.
3. For predictive metrics present on both sides, write means, deltas, shared
   count, and W/L/T to `reference_comparisons.csv`.
4. Keep loss direction explicit:
   - higher is better for Accuracy, Balanced Accuracy, AUC, and F1;
   - lower is better for Log Loss and other explicitly named loss metrics.
5. Generate rankings only for methods sharing the same successful dataset set
   within one `(model, dataset, metric)` block.
6. Never put comparison or ranking output into `method_metrics.md`.
7. Test shared intersections, unequal coverage, ties, loss direction,
   incomparable blocks, and the absence of comparisons from main Markdown.

**Verification:**

```bash
python -m pytest tests/test_build_paper_result_archive.py \
  -k 'comparison or ranking or intersection' -q
```

## Task 7: Assemble the Curated Archive and Documentation

**Files:**

- Modify: `scripts/build_paper_result_archive.py`
- Generate: `result/README.md`
- Generate: `result/inventory.csv`
- Generate: `result/provenance.csv`
- Generate: `result/raw/**`
- Generate: `result/tables/**`
- Generate: `result/scripts/build_result_archive.py`

**Steps:**

1. Create a staging directory adjacent to `result/`.
2. Copy only selected stable CSVs and directly relevant manifests, summaries,
   and best-parameter files into canonical
   `raw/<dataset>/<model>/<method>/` directories.
3. Preserve the original basename and add a short source hash when two selected
   support files would otherwise collide.
4. Generate inventory, provenance, method metrics, completeness Markdown/CSV,
   and reference-only CSVs in the staging directory.
5. Copy the exact authoritative builder source into
   `result/scripts/build_result_archive.py` so the archive records how it was
   produced.
6. Generate `README.md` with:
   - snapshot time and source roots;
   - 72-cell definition and canonical names;
   - best-result selection policy;
   - completeness definitions;
   - metric aggregation rules;
   - warning for every `selected_on_eval=true` cell;
   - counts by overall status.
7. Replace generated archive files atomically. Do not delete unknown,
   user-created files already under `result/`; abort on an unmanaged collision.
8. Ensure repeated builds against unchanged sources produce byte-identical
   files by using the supplied snapshot time and stable ordering.

**Verification:**

```bash
python scripts/build_paper_result_archive.py --local-only --out result
find result -maxdepth 4 -type f | sort
```

## Task 8: Complete Automated and Static Validation

**Files:**

- Verify: `scripts/build_paper_result_archive.py`
- Verify: `tests/test_build_paper_result_archive.py`

**Steps:**

1. Compile the builder.
2. Run the focused test module.
3. Run a local-only dry-run.
4. Build twice into two temporary output roots with the same fixed snapshot
   time and compare hashes.
5. Assert:
   - exactly 72 unique inventory cells;
   - no duplicate archived dataset identities;
   - no comparison/ranking fields in `method_metrics.md`;
   - no selected smoke/debug/interrupted paths;
   - every archived raw file matches provenance SHA-256.
6. Run `git diff --check` only on files created or modified for this task.

**Verification:**

```bash
python -m py_compile scripts/build_paper_result_archive.py
python -m pytest tests/test_build_paper_result_archive.py -q
python scripts/build_paper_result_archive.py --local-only --dry-run
git diff --check -- \
  scripts/build_paper_result_archive.py \
  tests/test_build_paper_result_archive.py
```

## Task 9: Run the Live Read-Only Collection and Final Audit

**Remote root:** `/home/crc00006699/zh/tabiclv2`

**Steps:**

1. Recheck `jiqun` connectivity, repository root, current tmux panes, and
   active result-producing processes.
2. Run the builder from the local checkout; use SSH/rsync read-only access to
   remote sources.
3. Do not wait for or interfere with running LoCalPFN/LimiX jobs. Mark unstable
   or active partial cells `running` with exact paths.
4. Inspect the selected `FT` and `F-aware FT` source paths for all available
   model/dataset blocks and confirm no lower-coverage high-mean run displaced a
   fuller formal run.
5. Validate the final archive:

   ```bash
   python - <<'PY'
   import pandas as pd
   x = pd.read_csv("result/inventory.csv")
   assert len(x) == 72
   assert not x.duplicated(["model", "dataset", "method"]).any()
   print(x.groupby("overall_status").size().sort_index())
   PY
   ```

6. Re-hash every file represented in provenance.
7. Review `result/tables/method_metrics.md` for method-only presentation and
   `result/tables/completeness_matrix.md` for complete, partial, running, and
   missing lists.
8. Report:
   - exact `result/` path;
   - status counts;
   - selected best `FT` and `F-aware FT` source paths;
   - current missing and running cells;
   - any `selected_on_eval` caveats.

**Final verification:**

```bash
python scripts/build_paper_result_archive.py \
  --local-results results \
  --remote-host jiqun \
  --remote-results '~/zh/tabiclv2/results' \
  --out result
python -m pytest tests/test_build_paper_result_archive.py -q
git diff --check -- result scripts/build_paper_result_archive.py \
  tests/test_build_paper_result_archive.py
```
