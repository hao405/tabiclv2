# Paper Result Archive

Snapshot: `2026-07-20T02:27:57+00:00`

Sources: local `results/` and read-only `jiqun:~/zh/tabiclv2/results/`.

The inventory covers 72 cells: 3 models × 3 datasets × 8 methods.
`ln_head_embedding` is kept under that exact name. FT and F-aware FT use
formal results after coverage and status=ok precedence. For TabICLv2/data184,
F-aware FT uses the top three seeds by status=ok mean accuracy, while FT uses
the middle three seeds; their per-dataset numeric metrics are seed-averaged.

Main Markdown tables show each method's own metrics. Direct deltas, W/L/T,
and rankings are reference-only CSV artifacts.

## Status Counts

- `complete`: 12
- `complete_with_failures`: 20
- `missing`: 38
- `partial`: 2

## Selection-on-evaluation Warning

The following cells use a best/Optuna result selected on the reported evaluation tasks and must not be described as unbiased held-out estimates:
- `data184 / TabICLv2 / FT`
- `data184 / TabICLv2 / F-aware FT`
- `data184 / TabPFNv3 / FT`
- `Graph-SCM / TabPFNv3 / F-aware FT`

## Completeness Semantics

- `complete`: full target membership, all status=ok, and full required adaptation.
- `complete_with_failures`: full membership with failures, fallback, or unapplied adaptation.
- `partial`: stable formal artifact with missing target rows.
- `running`: active or unstable partial remote artifact.
- `missing`: no qualifying artifact.
- `invalid`: malformed or duplicate-identity artifact.
