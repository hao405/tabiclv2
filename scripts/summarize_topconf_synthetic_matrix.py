#!/usr/bin/env python3
"""Summarize the fixed TFM matrix on the paired topconf synthetic suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MODELS = ("tabicl-v1.1", "tabicl-v2", "tabpfn-v2", "tabpfn-v3")
METHODS = ("infer", "ft", "faware_ft")
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "log_loss")


def numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (*METRICS, "fit_seconds", "predict_seconds"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_joined(
    manifest_path: Path,
    matrix_root: Path,
    random_state: int,
    *,
    models: Sequence[str] = MODELS,
    methods: Sequence[str] = METHODS,
    reference_matrix_root: Path | None = None,
    reference_methods: Sequence[str] = (),
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if manifest["dataset_name"].duplicated().any():
        raise ValueError("synthetic_manifest.csv contains duplicate dataset_name values")
    frames: list[pd.DataFrame] = []
    reference_method_set = set(reference_methods)
    for model in models:
        for method in methods:
            use_reference = method in reference_method_set
            source_root = reference_matrix_root if use_reference else matrix_root
            if source_root is None:
                raise ValueError(f"method {method!r} requires --reference-matrix-root")
            result_path = (
                source_root
                / model
                / method
                / f"seed{random_state}"
                / "all_classification_results.csv"
            )
            if result_path.exists():
                result = pd.read_csv(result_path)
                if result["dataset_name"].duplicated().any():
                    raise ValueError(f"duplicate dataset_name values in {result_path}")
                unexpected = sorted(set(result["dataset_name"]) - set(manifest["dataset_name"]))
                if unexpected:
                    raise ValueError(
                        f"{result_path} contains {len(unexpected)} unexpected datasets"
                    )
                joined = manifest.merge(
                    result,
                    on="dataset_name",
                    how="left",
                    validate="one_to_one",
                    suffixes=("", "_result"),
                )
                joined["result_present"] = joined["status"].notna()
                joined["result_csv"] = str(result_path)
            else:
                joined = manifest.copy()
                joined["status"] = "missing_result_file"
                joined["result_present"] = False
                joined["result_csv"] = str(result_path)
            joined["model"] = model
            joined["method"] = method
            joined["result_source_root"] = str(source_root)
            joined["result_source_role"] = "reference" if use_reference else "primary"
            frames.append(joined)
    merged = numeric_columns(pd.concat(frames, ignore_index=True, sort=False))
    duplicate_key = ["model", "method", "dataset_name"]
    if merged.duplicated(duplicate_key).any():
        raise ValueError("duplicate model/method/dataset_name rows across result roots")
    return merged


def coverage_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method), group in joined.groupby(["model", "method"], sort=False):
        status = group["status"].fillna("missing_result_row").astype(str)
        rows.append(
            {
                "model": model,
                "method": method,
                "expected_datasets": len(group),
                "present_rows": int(group["result_present"].fillna(False).sum()),
                "ok_rows": int((status == "ok").sum()),
                "failed_rows": int((status == "fail").sum()),
                "skipped_rows": int((status == "skip").sum()),
                "missing_rows": int(status.str.startswith("missing").sum()),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = joined[joined["status"] == "ok"].copy()
    group_columns = ["model", "method", "variant", "generator_family"]
    for keys, group in ok.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        row["ok_rows"] = len(group)
        for metric in METRICS:
            if metric in group.columns:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
        if {"fit_seconds", "predict_seconds"}.issubset(group.columns):
            row["dataset_seconds_mean"] = (
                group["fit_seconds"] + group["predict_seconds"]
            ).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_stress(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = joined[joined["status"] == "ok"].copy()
    clean_columns = ["model", "method", "base_task_id", *METRICS]
    available_clean = [column for column in clean_columns if column in ok.columns]
    clean = ok[ok["variant"] == "clean"][available_clean].copy()
    clean = clean.rename(
        columns={metric: f"{metric}_clean" for metric in METRICS if metric in clean}
    )
    stress = ok[ok["variant"] != "clean"].copy()
    detail = stress.merge(
        clean,
        on=["model", "method", "base_task_id"],
        how="inner",
        validate="many_to_one",
    )
    for metric in METRICS:
        clean_column = f"{metric}_clean"
        if metric in detail.columns and clean_column in detail.columns:
            detail[f"{metric}_stress_minus_clean"] = detail[metric] - detail[clean_column]

    rows = []
    delta_columns = [column for column in detail if column.endswith("_stress_minus_clean")]
    for keys, group in detail.groupby(
        ["model", "method", "stress_axis", "stress_level"],
        dropna=False,
        sort=False,
    ):
        row = dict(zip(["model", "method", "stress_axis", "stress_level"], keys))
        row["paired_rows"] = len(group)
        for column in delta_columns:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std()
        rows.append(row)
    return detail, pd.DataFrame(rows)


def method_deltas(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = joined[joined["status"] == "ok"].copy()
    identity = ["model", "dataset_name"]
    infer_columns = [*identity, *[metric for metric in METRICS if metric in ok.columns]]
    infer = ok[ok["method"] == "infer"][infer_columns].rename(
        columns={metric: f"{metric}_infer" for metric in METRICS if metric in ok}
    )
    candidates = ok[ok["method"].isin(("ft", "faware_ft"))].copy()
    detail = candidates.merge(infer, on=identity, how="inner", validate="many_to_one")
    for metric in METRICS:
        infer_column = f"{metric}_infer"
        if metric in detail and infer_column in detail:
            detail[f"{metric}_minus_infer"] = detail[metric] - detail[infer_column]

    rows = []
    for (model, method, variant), group in detail.groupby(
        ["model", "method", "variant"], dropna=False, sort=False
    ):
        row = {
            "model": model,
            "method": method,
            "variant": variant,
            "paired_rows": len(group),
        }
        for metric in METRICS:
            column = f"{metric}_minus_infer"
            if column in group:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_std"] = group[column].std()
                if metric in {"accuracy", "balanced_accuracy"}:
                    row[f"{metric}_negative_transfer_count"] = int(
                        (group[column] < 0).sum()
                    )
        rows.append(row)
    return detail, pd.DataFrame(rows)


def truthy_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin({"1", "true", "yes"})


def telemetry_audit(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method, source_role), group in joined.groupby(
        ["model", "method", "result_source_role"], sort=False, dropna=False
    ):
        ok = group[group["status"] == "ok"]
        row = {
            "model": model,
            "method": method,
            "source_role": source_role,
            "expected_rows": len(group),
            "ok_rows": len(ok),
            "ttt_applied_true": (
                int(truthy_mask(ok["ttt_applied"]).sum()) if "ttt_applied" in ok else 0
            ),
            "ttt_oom_fallback_true": (
                int(truthy_mask(ok["ttt_oom_fallback"]).sum())
                if "ttt_oom_fallback" in ok
                else 0
            ),
            "ttt_fallback_reason_nonempty": (
                int(ok["ttt_fallback_reason"].fillna("").astype(str).str.strip().ne("").sum())
                if "ttt_fallback_reason" in ok
                else 0
            ),
            "ttt_c_fallback_reason_nonempty": (
                int(ok["ttt_c_fallback_reason"].fillna("").astype(str).str.strip().ne("").sum())
                if "ttt_c_fallback_reason" in ok
                else 0
            ),
        }
        for column in ("ttt_c_selection", "ttt_c_source", "ttt_c_metric"):
            row[f"{column}_values"] = (
                "|".join(sorted(ok[column].dropna().astype(str).unique()))
                if column in ok
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_model_native_faware(joined: pd.DataFrame) -> None:
    expected_metrics = {
        "tabicl-v1.1": "tabicl_encoded_l2",
        "tabicl-v2": "tabicl_encoded_l2",
        "tabpfn-v2": "raw_l2",
        "tabpfn-v3": "raw_l2",
    }
    primary = joined[
        joined["result_source_role"].eq("primary") & joined["method"].eq("faware_ft")
    ]
    if primary.empty:
        raise ValueError("model-native F-aware validation requested but no primary faware_ft rows exist")
    errors: list[str] = []
    for model, group in primary.groupby("model", sort=False):
        present = group[truthy_mask(group["result_present"])]
        ok = present[present["status"] == "ok"]
        if present.empty:
            errors.append(f"{model}: no primary result rows are present")
            continue
        if len(ok) != len(present):
            errors.append(
                f"{model}: expected all {len(present)} present rows status=ok, got {len(ok)}"
            )
            continue
        if "ttt_applied" not in ok or not truthy_mask(ok["ttt_applied"]).all():
            errors.append(f"{model}: not every row has ttt_applied=True")
        selections = set(ok.get("ttt_c_selection", pd.Series(dtype=str)).dropna().astype(str))
        if selections != {"f_mmd"}:
            errors.append(f"{model}: ttt_c_selection={sorted(selections)} expected ['f_mmd']")
        metrics = set(ok.get("ttt_c_metric", pd.Series(dtype=str)).dropna().astype(str))
        expected_metric = expected_metrics[model]
        if metrics != {expected_metric}:
            errors.append(f"{model}: ttt_c_metric={sorted(metrics)} expected [{expected_metric!r}]")
        if "ttt_oom_fallback" in ok and truthy_mask(ok["ttt_oom_fallback"]).any():
            errors.append(f"{model}: ttt_oom_fallback=True rows detected")
        for column in ("ttt_fallback_reason", "ttt_c_fallback_reason"):
            if column in ok and ok[column].fillna("").astype(str).str.strip().ne("").any():
                errors.append(f"{model}: non-empty {column} rows detected")
    if errors:
        raise ValueError("model-native F-aware telemetry validation failed: " + "; ".join(errors))


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    delta_column: str,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[int, float | None, float | None]:
    cluster_means = (
        frame[["base_task_id", delta_column]]
        .dropna(subset=[delta_column])
        .groupby("base_task_id", sort=False)[delta_column]
        .mean()
        .to_numpy()
    )
    cluster_count = len(cluster_means)
    if cluster_count == 0:
        return 0, None, None
    indices = rng.integers(0, cluster_count, size=(samples, cluster_count))
    bootstrap_means = cluster_means[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return cluster_count, float(low), float(high)


def stage_transitions(
    joined: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260715,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = joined[joined["status"] == "ok"].copy()
    transitions = (("infer_to_ft", "infer", "ft"), ("ft_to_faware", "ft", "faware_ft"))
    identity = ["model", "dataset_name"]
    detail_frames: list[pd.DataFrame] = []
    for comparison, baseline_method, candidate_method in transitions:
        baseline = ok[ok["method"] == baseline_method][
            [*identity, *[metric for metric in METRICS if metric in ok]]
        ].rename(columns={metric: f"{metric}_baseline" for metric in METRICS if metric in ok})
        candidate = ok[ok["method"] == candidate_method].copy()
        if baseline.empty or candidate.empty:
            continue
        detail = candidate.merge(baseline, on=identity, how="inner", validate="one_to_one")
        detail["comparison"] = comparison
        detail["baseline_method"] = baseline_method
        detail["candidate_method"] = candidate_method
        for metric in METRICS:
            baseline_column = f"{metric}_baseline"
            if metric in detail and baseline_column in detail:
                detail[f"{metric}_delta"] = detail[metric] - detail[baseline_column]
        detail_frames.append(detail)

    if not detail_frames:
        return pd.DataFrame(), pd.DataFrame()
    detail = pd.concat(detail_frames, ignore_index=True, sort=False)
    rng = np.random.default_rng(bootstrap_seed)
    summary_rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", detail)]
    scopes.extend((str(variant), group) for variant, group in detail.groupby("variant", sort=False))
    for scope, scoped in scopes:
        for (model, comparison), group in scoped.groupby(["model", "comparison"], sort=False):
            row: dict[str, object] = {
                "model": model,
                "comparison": comparison,
                "variant": scope,
                "paired_rows": len(group),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            }
            for metric in METRICS:
                delta_column = f"{metric}_delta"
                if delta_column not in group:
                    continue
                clean = group.dropna(subset=[delta_column])
                delta = clean[delta_column]
                cluster_count, low, high = cluster_bootstrap_ci(
                    clean,
                    delta_column,
                    rng=rng,
                    samples=bootstrap_samples,
                )
                row[f"{metric}_paired_rows"] = len(clean)
                row[f"{metric}_delta_mean"] = delta.mean() if len(clean) else None
                row[f"{metric}_delta_std"] = delta.std() if len(clean) > 1 else None
                row[f"{metric}_delta_median"] = delta.median() if len(clean) else None
                row[f"{metric}_wins"] = int((delta > 0).sum())
                row[f"{metric}_losses"] = int((delta < 0).sum())
                row[f"{metric}_ties"] = int((delta == 0).sum())
                row[f"{metric}_negative_transfer_count"] = int((delta < 0).sum())
                row[f"{metric}_cluster_count"] = cluster_count
                row[f"{metric}_cluster_ci95_low"] = low
                row[f"{metric}_cluster_ci95_high"] = high
            summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def summarize(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    models = list(dict.fromkeys(args.models or MODELS))
    primary_methods = list(dict.fromkeys(args.methods or METHODS))
    reference_matrix_root = (
        Path(args.reference_matrix_root).expanduser().resolve()
        if args.reference_matrix_root
        else None
    )
    reference_methods = list(
        dict.fromkeys(
            args.reference_methods
            or (("infer", "ft") if reference_matrix_root is not None else ())
        )
    )
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1")
    if reference_methods and reference_matrix_root is None:
        raise ValueError("--reference-methods requires --reference-matrix-root")
    overlap = set(primary_methods) & set(reference_methods)
    if overlap:
        raise ValueError(
            "primary and reference methods must be disjoint: " + ", ".join(sorted(overlap))
        )
    selected_methods = [
        method
        for method in METHODS
        if method in set(primary_methods) | set(reference_methods)
    ]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else matrix_root / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    joined = load_joined(
        manifest_path,
        matrix_root,
        args.random_state,
        models=models,
        methods=selected_methods,
        reference_matrix_root=reference_matrix_root,
        reference_methods=reference_methods,
    )
    if args.validate_model_native_faware:
        validate_model_native_faware(joined)
    coverage = coverage_summary(joined)
    metrics = metric_summary(joined)
    stress_detail, stress_summary = paired_stress(joined)
    method_detail, method_summary = method_deltas(joined)
    transition_detail, transition_summary = stage_transitions(
        joined,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    telemetry = telemetry_audit(joined)

    artifacts = {
        "matrix_results_joined.csv": joined,
        "coverage_summary.csv": coverage,
        "metric_summary.csv": metrics,
        "paired_stress_detail.csv": stress_detail,
        "paired_stress_summary.csv": stress_summary,
        "method_vs_infer_detail.csv": method_detail,
        "method_vs_infer_summary.csv": method_summary,
        "stage_transition_detail.csv": transition_detail,
        "stage_transition_summary.csv": transition_summary,
        "telemetry_audit.csv": telemetry,
    }
    for name, frame in artifacts.items():
        frame.to_csv(out_dir / name, index=False)

    fully_ok = int((coverage["ok_rows"] == coverage["expected_datasets"]).sum())
    lines = [
        "Topconf synthetic selected-matrix summary",
        "",
        f"manifest: {manifest_path}",
        f"matrix_root: {matrix_root}",
        f"models: {','.join(models)}",
        f"primary_methods: {','.join(primary_methods)}",
        f"reference_methods: {','.join(reference_methods) if reference_methods else '(none)'}",
        f"reference_matrix_root: {reference_matrix_root if reference_matrix_root else '(none)'}",
        f"expected_matrix_cells: {len(models) * len(selected_methods)}",
        f"fully_ok_matrix_cells: {fully_ok}",
        f"expected_datasets_per_cell: {coverage['expected_datasets'].iloc[0] if len(coverage) else 0}",
        "",
        "Interpretation boundary:",
        "- clean rows estimate prior-ID behavior.",
        "- non-clean rows are paired controlled OOD tests; stress-minus-clean is the primary robustness contrast.",
        "- method comparisons use exact model/dataset shared-ok pairs.",
        "",
        "Artifacts:",
    ]
    lines.extend(f"- {name}: {out_dir / name}" for name in artifacts)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"analysis_out_dir: {out_dir}")
    print(f"fully_ok_matrix_cells: {fully_ok}/{len(models) * len(selected_methods)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--reference-matrix-root", default=None)
    parser.add_argument("--reference-methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--validate-model-native-faware", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(summarize(build_parser().parse_args()))


if __name__ == "__main__":
    main()
