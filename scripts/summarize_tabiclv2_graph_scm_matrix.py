#!/usr/bin/env python3
"""Summarize the fixed TFM matrix on TabICLv2 three-stage Graph-SCM tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODELS = ("tabicl-v1.1", "tabicl-v2", "tabpfn-v2", "tabpfn-v3")
METHODS = ("infer", "ft", "faware_ft")
METRICS = ("accuracy", "balanced_accuracy")
COMPARISONS = (
    ("ft", "infer", "ft_minus_infer"),
    ("faware_ft", "ft", "faware_ft_minus_ft"),
)
TIE_ATOL = 1e-12


def truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def load_joined(manifest_path: Path, matrix_root: Path, random_state: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"dataset_name", "stage"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"prior manifest missing columns: {sorted(missing)}")
    if manifest["dataset_name"].duplicated().any():
        raise ValueError("prior manifest contains duplicate dataset_name values")
    frames: list[pd.DataFrame] = []
    expected_names = set(manifest["dataset_name"].astype(str))
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
                if result["dataset_name"].duplicated().any():
                    raise ValueError(f"duplicate dataset_name rows: {result_path}")
                unexpected = sorted(set(result["dataset_name"].astype(str)) - expected_names)
                if unexpected:
                    raise ValueError(
                        f"{result_path} contains {len(unexpected)} unexpected datasets"
                    )
                frame = manifest.merge(
                    result,
                    on="dataset_name",
                    how="left",
                    validate="one_to_one",
                    suffixes=("", "_result"),
                )
                frame["result_present"] = frame["status"].notna()
            else:
                frame = manifest.copy()
                frame["status"] = "missing_result_file"
                frame["result_present"] = False
            frame["model"] = model
            frame["method"] = method
            frame["result_csv"] = str(result_path)
            frames.append(frame)
    joined = pd.concat(frames, ignore_index=True, sort=False)
    for column in (*METRICS, "fit_seconds", "predict_seconds"):
        if column in joined:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    return joined


def scoped_groups(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "overall", frame
    for stage in ("stage1", "stage2", "stage3"):
        yield stage, frame[frame["stage"] == stage]


def coverage_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, method), cell in joined.groupby(["model", "method"], sort=False):
        for scope, group in scoped_groups(cell):
            statuses = group["status"].fillna("missing_result_row").astype(str)
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "scope": scope,
                    "expected_rows": len(group),
                    "present_rows": int(group["result_present"].fillna(False).sum()),
                    "ok_rows": int(statuses.eq("ok").sum()),
                    "failed_rows": int(statuses.eq("fail").sum()),
                    "skipped_rows": int(statuses.eq("skip").sum()),
                    "missing_rows": int(statuses.str.startswith("missing").sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_method_deltas(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = joined[joined["status"].eq("ok")].copy()
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    identity = ["dataset_name", "stage", "stage_id", "stage_index"]
    identity = [column for column in identity if column in ok]

    for model in MODELS:
        model_rows = ok[ok["model"].eq(model)]
        for left_method, right_method, comparison in COMPARISONS:
            value_columns = [column for column in METRICS if column in model_rows]
            left = model_rows[model_rows["method"].eq(left_method)][
                [*identity, *value_columns]
            ].copy()
            right = model_rows[model_rows["method"].eq(right_method)][
                [*identity, *value_columns]
            ].copy()
            left = left.rename(columns={metric: f"{metric}_left" for metric in value_columns})
            right = right.rename(columns={metric: f"{metric}_right" for metric in value_columns})
            detail = left.merge(right, on=identity, how="inner", validate="one_to_one")
            detail["model"] = model
            detail["comparison"] = comparison
            detail["left_method"] = left_method
            detail["right_method"] = right_method
            for metric in value_columns:
                detail[f"{metric}_delta"] = (
                    detail[f"{metric}_left"] - detail[f"{metric}_right"]
                )
            detail_frames.append(detail)

            for scope, group in scoped_groups(detail):
                for metric in value_columns:
                    delta = group[f"{metric}_delta"].dropna()
                    wins = int((delta > TIE_ATOL).sum())
                    losses = int((delta < -TIE_ATOL).sum())
                    ties = int((delta.abs() <= TIE_ATOL).sum())
                    summary_rows.append(
                        {
                            "model": model,
                            "comparison": comparison,
                            "left_method": left_method,
                            "right_method": right_method,
                            "scope": scope,
                            "metric": metric,
                            "paired_rows": len(delta),
                            "left_mean": group.loc[delta.index, f"{metric}_left"].mean(),
                            "right_mean": group.loc[delta.index, f"{metric}_right"].mean(),
                            "delta_mean": delta.mean(),
                            "delta_std": delta.std(),
                            "wins": wins,
                            "losses": losses,
                            "ties": ties,
                            "negative_transfer_count": losses,
                        }
                    )
    detail_columns = [
        *identity,
        "model",
        "comparison",
        "left_method",
        "right_method",
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("left", "right", "delta")],
    ]
    detail = (
        pd.concat(detail_frames, ignore_index=True, sort=False)
        if detail_frames
        else pd.DataFrame(columns=detail_columns)
    )
    return detail, pd.DataFrame(summary_rows)


def telemetry_audit(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, method), cell in joined.groupby(["model", "method"], sort=False):
        ok = cell[cell["status"].eq("ok")]
        row: dict[str, object] = {
            "model": model,
            "method": method,
            "expected_rows": len(cell),
            "ok_rows": len(ok),
            "ttt_applied_true": int(truthy(ok["ttt_applied"]).sum())
            if "ttt_applied" in ok
            else 0,
            "ttt_oom_fallback_true": int(truthy(ok["ttt_oom_fallback"]).sum())
            if "ttt_oom_fallback" in ok
            else 0,
        }
        for column in (
            "ttt_c_selection",
            "ttt_c_source",
            "ttt_c_metric",
            "ttt_fallback_reason",
            "ttt_c_fallback_reason",
        ):
            row[f"{column}_values"] = (
                "|".join(sorted(ok[column].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()))
                if column in ok
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    output_path: Path,
    coverage: pd.DataFrame,
    paired: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(none)"

        def render(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        headers = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for values in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(render(value) for value in values) + " |")
        return "\n".join(lines)

    lines = [
        "# TabICLv2 Graph-SCM Matrix Summary",
        "",
        "Paired comparisons use only dataset-name intersections where both cells have `status=ok`.",
        "",
        "## Coverage",
        "",
    ]
    overall_coverage = coverage[coverage["scope"].eq("overall")]
    lines.append(markdown_table(overall_coverage))
    lines.extend(["", "## Paired method deltas", ""])
    overall_paired = paired[paired["scope"].eq("overall")]
    lines.append(markdown_table(overall_paired))
    lines.extend(["", "## TTT telemetry", ""])
    lines.append(markdown_table(telemetry))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else matrix_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = load_joined(manifest_path, matrix_root, args.random_state)
    coverage = coverage_summary(joined)
    detail, paired = paired_method_deltas(joined)
    telemetry = telemetry_audit(joined)
    joined.to_csv(output_dir / "joined_results.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    detail.to_csv(output_dir / "paired_detail.csv", index=False)
    paired.to_csv(output_dir / "paired_summary.csv", index=False)
    telemetry.to_csv(output_dir / "telemetry_audit.csv", index=False)
    write_markdown(output_dir / "summary.md", coverage, paired, telemetry)
    print(f"analysis_out_dir: {output_dir}")
    print(
        "fully_ok_matrix_cells: "
        f"{int(((coverage['scope'] == 'overall') & (coverage['ok_rows'] == coverage['expected_rows'])).sum())}/12"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a fixed 4x3 TFM matrix on TabICLv2 Graph-SCM tasks."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(summarize(build_parser().parse_args()))
