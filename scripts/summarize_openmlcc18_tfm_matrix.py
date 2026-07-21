#!/usr/bin/env python3
"""Summarize the TabICL v2 / TabPFN v3 OpenML-CC18 2x3 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("tabicl-v2", "tabpfn-v3")
METHODS = ("infer", "ft", "faware_ft")
METRICS = ("accuracy", "balanced_accuracy")
COMPARISONS = (
    ("ft", "infer", "ft_minus_infer"),
    ("faware_ft", "ft", "faware_ft_minus_ft"),
)
EXPECTED_NATIVE_METRIC = {
    "tabicl-v2": "tabicl_encoded_l2",
    "tabpfn-v3": "raw_l2",
}
TIE_ATOL = 1e-12


def truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def nonempty(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def load_expected_names(view_manifest: Path) -> list[str]:
    payload = json.loads(view_manifest.read_text(encoding="utf-8"))
    included = payload.get("included")
    if not isinstance(included, list):
        raise ValueError("dataset view manifest has no included list")
    names = [str(row["dataset_name"]) for row in included]
    if len(names) != len(set(names)):
        raise ValueError("dataset view manifest contains duplicate dataset names")
    if int(payload.get("included_count", -1)) != len(names):
        raise ValueError("dataset view manifest included_count mismatch")
    return names


def load_joined(
    *, matrix_root: Path, expected_names: list[str], random_state: int
) -> pd.DataFrame:
    expected = pd.DataFrame(
        {"dataset_name": expected_names, "dataset_order": range(len(expected_names))}
    )
    expected_set = set(expected_names)
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        for method in METHODS:
            result_path = (
                matrix_root
                / model
                / method
                / f"seed{random_state}"
                / "all_classification_results.csv"
            )
            if result_path.exists():
                result = pd.read_csv(result_path)
                if "dataset_name" not in result:
                    raise ValueError(f"missing dataset_name column: {result_path}")
                result["dataset_name"] = result["dataset_name"].astype(str)
                if result["dataset_name"].duplicated().any():
                    raise ValueError(f"duplicate dataset rows: {result_path}")
                unexpected = sorted(set(result["dataset_name"]) - expected_set)
                if unexpected:
                    raise ValueError(f"unexpected datasets in {result_path}: {unexpected[:5]}")
                frame = expected.merge(result, on="dataset_name", how="left", validate="one_to_one")
                frame["result_present"] = frame["status"].notna()
            else:
                frame = expected.copy()
                frame["status"] = "missing_result_file"
                frame["result_present"] = False
            frame["model"] = model
            frame["method"] = method
            frame["result_csv"] = str(result_path)
            frames.append(frame)
    joined = pd.concat(frames, ignore_index=True, sort=False)
    for metric in METRICS:
        if metric in joined:
            joined[metric] = pd.to_numeric(joined[metric], errors="coerce")
    return joined


def adaptation_eligible(frame: pd.DataFrame) -> pd.Series:
    eligible = frame["status"].eq("ok")
    adapted = frame["method"].ne("infer")
    if "ttt_applied" in frame:
        eligible &= ~adapted | truthy(frame["ttt_applied"])
    else:
        eligible &= ~adapted
    if "ttt_oom_fallback" in frame:
        eligible &= ~truthy(frame["ttt_oom_fallback"])
    for column in ("ttt_fallback_reason", "ttt_c_fallback_reason"):
        if column in frame:
            eligible &= ~nonempty(frame[column])
    return eligible


def coverage_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, method), cell in joined.groupby(["model", "method"], sort=False):
        present = cell["result_present"].fillna(False)
        statuses = cell["status"].fillna("missing_result_row").astype(str)
        adapted = method != "infer"
        applied = truthy(cell["ttt_applied"]) if "ttt_applied" in cell else pd.Series(False, index=cell.index)
        oom = truthy(cell["ttt_oom_fallback"]) if "ttt_oom_fallback" in cell else pd.Series(False, index=cell.index)
        fallback = pd.Series(False, index=cell.index)
        for column in ("ttt_fallback_reason", "ttt_c_fallback_reason"):
            if column in cell:
                fallback |= nonempty(cell[column])
        rows.append(
            {
                "model": model,
                "method": method,
                "expected_rows": len(cell),
                "present_rows": int(present.sum()),
                "ok_rows": int(statuses.eq("ok").sum()),
                "eligible_rows": int(adaptation_eligible(cell).sum()),
                "failed_rows": int(statuses.eq("fail").sum()),
                "skipped_rows": int(statuses.eq("skip").sum()),
                "missing_rows": int((~present).sum()),
                "ttt_applied_true": int(applied.sum()) if adapted else 0,
                "ttt_oom_fallback_true": int(oom.sum()),
                "fallback_rows": int(fallback.sum()),
                "unapplied_adaptation_rows": int((statuses.eq("ok") & ~applied).sum()) if adapted else 0,
            }
        )
    return pd.DataFrame(rows)


def telemetry_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, method), cell in joined.groupby(["model", "method"], sort=False):
        ok = cell[cell["status"].eq("ok")]
        row: dict[str, object] = {"model": model, "method": method}
        for column in (
            "ttt_c_selection",
            "ttt_c_source",
            "ttt_c_metric",
            "ttt_fallback_reason",
            "ttt_c_fallback_reason",
        ):
            row[f"{column}_values"] = (
                "|".join(sorted(value for value in ok[column].dropna().astype(str).str.strip().unique() if value))
                if column in ok
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_results(
    joined: pd.DataFrame, *, random_state: int, bootstrap_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = joined[adaptation_eligible(joined)].copy()
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for model in MODELS:
        model_rows = eligible[eligible["model"].eq(model)]
        for left_method, right_method, comparison in COMPARISONS:
            columns = ["dataset_name", *METRICS]
            left = model_rows[model_rows["method"].eq(left_method)][columns].rename(
                columns={metric: f"{metric}_left" for metric in METRICS}
            )
            right = model_rows[model_rows["method"].eq(right_method)][columns].rename(
                columns={metric: f"{metric}_right" for metric in METRICS}
            )
            detail = left.merge(right, on="dataset_name", how="inner", validate="one_to_one")
            detail["model"] = model
            detail["comparison"] = comparison
            detail["left_method"] = left_method
            detail["right_method"] = right_method
            for metric_index, metric in enumerate(METRICS):
                delta_column = f"{metric}_delta"
                detail[delta_column] = detail[f"{metric}_left"] - detail[f"{metric}_right"]
                valid = detail.dropna(subset=[delta_column])
                delta = valid[delta_column].to_numpy(dtype=float)
                digest = hashlib.sha256(f"{model}|{comparison}|{metric}".encode()).digest()
                ci_seed = random_state + int.from_bytes(digest[:4], "little")
                ci_low, ci_high = bootstrap_ci(
                    delta,
                    seed=ci_seed,
                    resamples=bootstrap_resamples,
                )
                summary_rows.append(
                    {
                        "model": model,
                        "comparison": comparison,
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "paired_rows": len(delta),
                        "left_mean": valid[f"{metric}_left"].mean(),
                        "right_mean": valid[f"{metric}_right"].mean(),
                        "delta_mean": float(delta.mean()) if len(delta) else float("nan"),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "wins": int((delta > TIE_ATOL).sum()),
                        "losses": int((delta < -TIE_ATOL).sum()),
                        "ties": int((np.abs(delta) <= TIE_ATOL).sum()),
                        "negative_transfer_count": int((delta < -TIE_ATOL).sum()),
                    }
                )
            detail_frames.append(detail)
    detail = pd.concat(detail_frames, ignore_index=True, sort=False)
    return detail, pd.DataFrame(summary_rows)


def failure_rows(joined: pd.DataFrame) -> pd.DataFrame:
    eligible = adaptation_eligible(joined)
    failures = joined[~eligible].copy()
    keep = [
        column
        for column in (
            "model",
            "method",
            "dataset_name",
            "status",
            "error",
            "ttt_applied",
            "ttt_oom_fallback",
            "ttt_fallback_reason",
            "ttt_c_fallback_reason",
        )
        if column in failures
    ]
    return failures[keep]


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    display = frame.copy().fillna("")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def smoke_errors(
    *, joined: pd.DataFrame, matrix_root: Path, expected_count: int
) -> list[str]:
    errors: list[str] = []
    manifest_path = matrix_root / "matrix_manifest.json"
    try:
        matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid matrix manifest: {exc}"]
    if matrix_manifest.get("models") != list(MODELS):
        errors.append(f"unexpected matrix models: {matrix_manifest.get('models')}")
    if matrix_manifest.get("methods") != list(METHODS):
        errors.append(f"unexpected matrix methods: {matrix_manifest.get('methods')}")
    for model in MODELS:
        for method in METHODS:
            cell = joined[joined["model"].eq(model) & joined["method"].eq(method)]
            if len(cell) != expected_count:
                errors.append(f"{model}/{method}: expected {expected_count} rows, got {len(cell)}")
                continue
            if not cell["status"].eq("ok").all():
                errors.append(f"{model}/{method}: not all rows are status=ok")
            if method == "infer":
                continue
            if "ttt_applied" not in cell or not truthy(cell["ttt_applied"]).all():
                errors.append(f"{model}/{method}: ttt_applied is not true for every row")
            selections = set(cell.get("ttt_c_selection", pd.Series(dtype=str)).dropna().astype(str))
            allowed_selections = (
                {"random"}
                if method == "ft"
                else {"f_test_centroid_reserve", "f_mmd"}
            )
            if not selections or not selections.issubset(allowed_selections):
                errors.append(f"{model}/{method}: unexpected selections {sorted(selections)}")
            if method == "faware_ft":
                metrics = set(cell.get("ttt_c_metric", pd.Series(dtype=str)).dropna().astype(str))
                if metrics != {EXPECTED_NATIVE_METRIC[model]}:
                    errors.append(f"{model}/{method}: unexpected metrics {sorted(metrics)}")
            if not adaptation_eligible(cell).all():
                errors.append(f"{model}/{method}: contains OOM/fallback/unapplied rows")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--require-smoke", action="store_true")
    parser.add_argument("--expected-datasets", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be >= 1")
    expected_names = load_expected_names(args.view_manifest)
    if args.expected_datasets is not None and len(expected_names) != args.expected_datasets:
        raise ValueError(
            f"view expected {args.expected_datasets} datasets, found {len(expected_names)}"
        )
    joined = load_joined(
        matrix_root=args.matrix_root,
        expected_names=expected_names,
        random_state=args.random_state,
    )
    coverage = coverage_summary(joined)
    telemetry = telemetry_summary(joined)
    detail, paired = paired_results(
        joined,
        random_state=args.random_state,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    failures = failure_rows(joined)
    output_dir = args.output_dir or (args.matrix_root / "openmlcc18_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    telemetry.to_csv(output_dir / "telemetry.csv", index=False)
    detail.to_csv(output_dir / "paired_detail.csv", index=False)
    paired.to_csv(output_dir / "paired_summary.csv", index=False)
    failures.to_csv(output_dir / "failures.csv", index=False)
    summary = [
        "# OpenML-CC18 2x3 Matrix Summary",
        "",
        f"Expected datasets per cell: {len(expected_names)}",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Paired comparisons",
        "",
        markdown_table(paired),
        "",
        "## Telemetry",
        "",
        markdown_table(telemetry),
        "",
        f"Excluded/non-eligible rows: {len(failures)}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"analysis_dir: {output_dir}")
    print(f"expected_datasets: {len(expected_names)}")
    print(f"noneligible_rows: {len(failures)}")

    if args.require_smoke:
        errors = smoke_errors(
            joined=joined,
            matrix_root=args.matrix_root,
            expected_count=len(expected_names),
        )
        if errors:
            for error in errors:
                print(f"smoke_error: {error}")
            return 2
        print("smoke_validation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
