#!/usr/bin/env python3
"""Compare two benchmark result folders on their successful dataset intersection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LEFT = Path("1c_result_v2/adaptor_dcnv2_ttt_chunk10000_accuracy")
DEFAULT_RIGHT = Path("1c_result_v2/iclv2_ttt_default")
DEFAULT_DATA_ROOT = Path("data178")
METRIC_CANDIDATES = ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss")
EPS = 1e-12


@dataclass(frozen=True)
class DatasetInfo:
    n_features: float | None
    n_total: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all_classification_results.csv from two result folders on "
            "the intersection of dataset_name where both status columns are ok."
        )
    )
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--left-name", default=None, help="Display name for the left result.")
    parser.add_argument("--right-name", default=None, help="Display name for the right result.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --left-dir.",
    )
    parser.add_argument(
        "--detail-name",
        default=None,
        help="Detail CSV filename. Defaults to compare_<left>_vs_<right>_success_intersection_detail.csv.",
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Summary markdown filename. Defaults to compare_<left>_vs_<right>_success_intersection_summary.md.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the generated markdown summary to stdout.",
    )
    return parser.parse_args()


def read_results(result_dir: Path) -> pd.DataFrame:
    csv_path = result_dir / "all_classification_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing result CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"dataset_name", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    if "accuracy" not in df.columns:
        raise ValueError(f"{csv_path} missing required metric column: accuracy")
    return df


def normalize_ok_mask(df: pd.DataFrame) -> pd.Series:
    return df["status"].fillna("").astype(str).str.lower().eq("ok")


def dedupe_by_dataset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    duplicated = df["dataset_name"].duplicated(keep=False)
    if duplicated.any():
        names = ", ".join(sorted(df.loc[duplicated, "dataset_name"].astype(str).unique()))
        print(f"[warn] {label} has duplicated dataset_name rows; keeping the first: {names}")
        df = df.drop_duplicates("dataset_name", keep="first")
    return df


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_dataset_info(data_root: Path, dataset_name: str) -> DatasetInfo:
    path = data_root / dataset_name / "info.json"
    if not path.exists():
        return DatasetInfo(n_features=None, n_total=None)
    with path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    n_num = info.get("n_num_features")
    n_cat = info.get("n_cat_features")
    n_features = None
    if n_num is not None or n_cat is not None:
        n_features = float(n_num or 0) + float(n_cat or 0)
    train = info.get("train_size")
    val = info.get("val_size")
    test = info.get("test_size")
    n_total = None
    if train is not None or val is not None or test is not None:
        n_total = float(train or 0) + float(val or 0) + float(test or 0)
    return DatasetInfo(n_features=n_features, n_total=n_total)


def sample_bin(n_total: float | None) -> str:
    if n_total is None or pd.isna(n_total):
        return "unknown"
    if n_total <= 2000:
        return "<=2k"
    if n_total <= 5000:
        return "2k-5k"
    if n_total <= 20000:
        return "5k-20k"
    if n_total <= 100000:
        return "20k-100k"
    return ">100k"


def feature_bin(n_features: float | None) -> str:
    if n_features is None or pd.isna(n_features):
        return "unknown"
    if n_features <= 10:
        return "<=10"
    if n_features <= 30:
        return "11-30"
    if n_features <= 100:
        return "31-100"
    if n_features <= 300:
        return "101-300"
    return ">300"


def pct(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{100 * value:+.{digits}f} pp"


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def fmt_signed(value: float | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{value:+.{digits}f}"


def names_from_rows(rows: pd.DataFrame, delta_col: str, ascending: bool = False) -> str:
    if rows.empty:
        return "(none)"
    parts = []
    for row in rows.sort_values(delta_col, ascending=ascending).itertuples(index=False):
        dataset = getattr(row, "dataset_name")
        delta = getattr(row, delta_col)
        n_features = getattr(row, "n_features_info")
        n_total = getattr(row, "n_total_info")
        parts.append(
            f"{dataset} ({pct(delta, 3)}, feat={fmt_float(n_features, 0)}, n={fmt_float(n_total, 0)})"
        )
    return ", ".join(parts)


def summarize_outcome(detail: pd.DataFrame, outcome: str) -> dict[str, float | int | str]:
    subset = detail[detail["accuracy_outcome"] == outcome]
    return {
        "outcome": outcome,
        "count": int(len(subset)),
        "mean_delta_pp": float(subset["accuracy_delta"].mean() * 100) if len(subset) else np.nan,
        "median_delta_pp": float(subset["accuracy_delta"].median() * 100) if len(subset) else np.nan,
        "mean_features": float(subset["n_features_info"].mean()) if len(subset) else np.nan,
        "median_features": float(subset["n_features_info"].median()) if len(subset) else np.nan,
        "mean_samples": float(subset["n_total_info"].mean()) if len(subset) else np.nan,
        "median_samples": float(subset["n_total_info"].median()) if len(subset) else np.nan,
    }


def markdown_table(rows: list[dict[str, object]], columns: Iterable[str]) -> str:
    columns = list(columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def metric_summary(detail: pd.DataFrame, metrics: list[str], left_name: str, right_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        left = as_float(detail[f"{metric}_left"])
        right = as_float(detail[f"{metric}_right"])
        delta = left - right
        delta_mean = float(delta.mean())
        delta_pp = pct(delta_mean, 4) if metric != "log_loss" else ""
        rows.append(
            {
                "metric": metric,
                left_name: fmt_float(float(left.mean())),
                right_name: fmt_float(float(right.mean())),
                "delta": fmt_signed(delta_mean),
                "delta_pp": delta_pp,
            }
        )
    return rows


def bin_table(detail: pd.DataFrame, bin_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    orders = {
        "sample_bin": ["<=2k", "2k-5k", "5k-20k", "20k-100k", ">100k", "unknown"],
        "feature_bin": ["<=10", "11-30", "31-100", "101-300", ">300", "unknown"],
    }
    values = list(detail[bin_col].dropna().unique())
    order = orders.get(bin_col)
    if order is not None:
        values = sorted(values, key=lambda value: order.index(value) if value in order else len(order))
    else:
        values = sorted(values)
    for value in values:
        subset = detail[detail[bin_col] == value]
        rows.append(
            {
                bin_col: value,
                "win": int((subset["accuracy_outcome"] == "win").sum()),
                "loss": int((subset["accuracy_outcome"] == "loss").sum()),
                "tie": int((subset["accuracy_outcome"] == "tie").sum()),
                "mean_delta_pp": f"{subset['accuracy_delta'].mean() * 100:+.4f}",
            }
        )
    return rows


def build_detail(left_ok: pd.DataFrame, right_ok: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    joined = left_ok.merge(right_ok, on="dataset_name", suffixes=("_left", "_right"), how="inner")
    if joined.empty:
        return joined

    metric_cols = [m for m in METRIC_CANDIDATES if f"{m}_left" in joined and f"{m}_right" in joined]
    for metric in metric_cols:
        joined[f"{metric}_left"] = as_float(joined[f"{metric}_left"])
        joined[f"{metric}_right"] = as_float(joined[f"{metric}_right"])
        joined[f"{metric}_delta"] = joined[f"{metric}_left"] - joined[f"{metric}_right"]

    acc_delta = joined["accuracy_delta"]
    joined["accuracy_outcome"] = np.where(
        acc_delta > EPS,
        "win",
        np.where(acc_delta < -EPS, "loss", "tie"),
    )

    n_test_left = as_float(joined["n_test_left"]) if "n_test_left" in joined else pd.Series(np.nan, index=joined.index)
    n_test_right = as_float(joined["n_test_right"]) if "n_test_right" in joined else pd.Series(np.nan, index=joined.index)
    joined["n_test_for_weight"] = n_test_left.fillna(n_test_right)
    joined["accuracy_delta_correct"] = joined["accuracy_delta"] * joined["n_test_for_weight"]

    infos = [load_dataset_info(data_root, str(name)) for name in joined["dataset_name"]]
    joined["n_features_info"] = [info.n_features for info in infos]
    joined["n_total_info"] = [info.n_total for info in infos]
    joined["sample_bin"] = [sample_bin(value) for value in joined["n_total_info"]]
    joined["feature_bin"] = [feature_bin(value) for value in joined["n_features_info"]]

    preferred = [
        "dataset_name",
        "accuracy_left",
        "accuracy_right",
        "accuracy_delta",
        "accuracy_outcome",
        "n_test_for_weight",
        "accuracy_delta_correct",
        "n_features_info",
        "n_total_info",
        "sample_bin",
        "feature_bin",
    ]
    for metric in metric_cols:
        for col in (f"{metric}_left", f"{metric}_right", f"{metric}_delta"):
            if col not in preferred:
                preferred.append(col)
    telemetry = []
    for col in joined.columns:
        is_left_suffix = col.endswith("_left")
        base_col = col.removesuffix("_left") if is_left_suffix else col
        if base_col.startswith("ttt_"):
            telemetry.append(col)
        elif col in {
            "n_train_left",
            "n_val_left",
            "n_test_left",
            "n_features_left",
            "n_classes_left",
            "error_left",
        }:
            telemetry.append(col)
    remaining = [col for col in preferred + telemetry if col in joined.columns]
    return joined[remaining].sort_values(["accuracy_outcome", "accuracy_delta", "dataset_name"], ascending=[True, False, True])


def generate_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    detail: pd.DataFrame,
    left_dir: Path,
    right_dir: Path,
    left_name: str,
    right_name: str,
    detail_path: Path,
) -> str:
    left_ok = normalize_ok_mask(left)
    right_ok = normalize_ok_mask(right)
    left_fail = left.loc[~left_ok, "dataset_name"].astype(str).sort_values().tolist()
    right_fail = right.loc[~right_ok, "dataset_name"].astype(str).sort_values().tolist()
    metrics = [m for m in METRIC_CANDIDATES if f"{m}_left" in detail and f"{m}_right" in detail]
    win = detail[detail["accuracy_outcome"] == "win"]
    loss = detail[detail["accuracy_outcome"] == "loss"]
    tie = detail[detail["accuracy_outcome"] == "tie"]

    metric_rows = metric_summary(detail, metrics, left_name, right_name)
    weighted_rows: list[dict[str, object]] = []
    if detail["n_test_for_weight"].notna().any():
        weights = as_float(detail["n_test_for_weight"])
        valid = weights.notna() & (weights > 0)
        if valid.any():
            left_weighted = float(np.average(detail.loc[valid, "accuracy_left"], weights=weights[valid]))
            right_weighted = float(np.average(detail.loc[valid, "accuracy_right"], weights=weights[valid]))
            weighted_rows.append(
                {
                    "metric": "test-size weighted accuracy",
                    left_name: fmt_float(left_weighted),
                    right_name: fmt_float(right_weighted),
                    "delta": fmt_signed(left_weighted - right_weighted),
                    "delta_pp": pct(left_weighted - right_weighted, 4),
                }
            )
            weighted_rows.append(
                {
                    "metric": "weighted delta_correct",
                    left_name: "",
                    right_name: "",
                    "delta": f"{detail.loc[valid, 'accuracy_delta_correct'].sum():+.2f} test examples",
                    "delta_pp": "",
                }
            )

    outcome_rows = []
    for outcome in ("win", "loss", "tie"):
        row = summarize_outcome(detail, outcome)
        outcome_rows.append(
            {
                "outcome": row["outcome"],
                "count": row["count"],
                "mean_delta_pp": f"{row['mean_delta_pp']:+.4f}" if not pd.isna(row["mean_delta_pp"]) else "nan",
                "median_delta_pp": f"{row['median_delta_pp']:+.4f}" if not pd.isna(row["median_delta_pp"]) else "nan",
                "mean_features": fmt_float(row["mean_features"], 1),
                "median_features": fmt_float(row["median_features"], 1),
                "mean_samples": fmt_float(row["mean_samples"], 1),
                "median_samples": fmt_float(row["median_samples"], 1),
            }
        )

    def bool_count(col: str) -> int | None:
        candidates = [col, f"{col}_left"]
        for candidate in candidates:
            if candidate in detail:
                values = detail[candidate].fillna(False).astype(str).str.lower()
                return int(values.isin({"true", "1", "yes"}).sum())
        return None

    lines = [
        f"# {left_name} vs {right_name} 成功交集对比",
        "",
        "## 口径",
        "",
        f"- 比较文件：`{left_dir / 'all_classification_results.csv'}` vs `{right_dir / 'all_classification_results.csv'}`。",
        "- 主比较只取两边 `status=ok` 的 `dataset_name` 交集。",
        f"- {left_name} 当前 `ok={int(left_ok.sum())}/{len(left)}`；{right_name} 当前 `ok={int(right_ok.sum())}/{len(right)}`；成功交集 `n={len(detail)}`。",
        f"- {left_name} 失败/非 ok 数据集：{', '.join(left_fail) if left_fail else '(none)'}。",
        f"- {right_name} 失败/非 ok 数据集：{', '.join(right_fail) if right_fail else '(none)'}。",
        "",
        "## 总体结果",
        "",
        markdown_table(metric_rows + weighted_rows, ["metric", left_name, right_name, "delta", "delta_pp"]),
        "",
        f"- Win/Loss/Tie（{left_name} 相对 {right_name}，按 accuracy）：`{len(win)}/{len(loss)}/{len(tie)}`。",
    ]

    for col in ("ttt_adaptor_enabled", "ttt_applied", "ttt_oom_fallback"):
        count = bool_count(col)
        if count is not None:
            lines.append(f"- 交集内 `{col}=True`：`{count}/{len(detail)}`。")

    lines.extend(
        [
            "",
            "## 上升/下降/持平数据集",
            "",
            f"- 上升 {len(win)} 个：{names_from_rows(win, 'accuracy_delta')}",
            f"- 下降 {len(loss)} 个：{names_from_rows(loss, 'accuracy_delta', ascending=True)}",
            f"- 持平 {len(tie)} 个：{names_from_rows(tie, 'accuracy_delta')}",
            "",
            "## 上升/下降数据规模特征",
            "",
            markdown_table(
                outcome_rows,
                [
                    "outcome",
                    "count",
                    "mean_delta_pp",
                    "median_delta_pp",
                    "mean_features",
                    "median_features",
                    "mean_samples",
                    "median_samples",
                ],
            ),
            "",
            "### 按样本量分桶",
            "",
            markdown_table(bin_table(detail, "sample_bin"), ["sample_bin", "win", "loss", "tie", "mean_delta_pp"]),
            "",
            "### 按特征数分桶",
            "",
            markdown_table(bin_table(detail, "feature_bin"), ["feature_bin", "win", "loss", "tie", "mean_delta_pp"]),
            "",
            f"逐数据集明细见 `{detail_path.name}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    left_dir = args.left_dir
    right_dir = args.right_dir
    left_name = args.left_name or left_dir.name
    right_name = args.right_name or right_dir.name
    out_dir = args.out_dir or left_dir
    detail_name = args.detail_name or f"compare_{left_name}_vs_{right_name}_success_intersection_detail.csv"
    summary_name = args.summary_name or f"compare_{left_name}_vs_{right_name}_success_intersection_summary.md"
    detail_path = out_dir / detail_name
    summary_path = out_dir / summary_name

    left = dedupe_by_dataset(read_results(left_dir), left_name)
    right = dedupe_by_dataset(read_results(right_dir), right_name)
    left_ok = left.loc[normalize_ok_mask(left)].copy()
    right_ok = right.loc[normalize_ok_mask(right)].copy()
    detail = build_detail(left_ok, right_ok, args.data_root)
    if detail.empty:
        raise SystemExit("No successful dataset intersection found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary = generate_summary(left, right, detail, left_dir, right_dir, left_name, right_name, detail_path)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote detail: {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"success_intersection_n: {len(detail)}")
    print(f"mean_accuracy_delta: {detail['accuracy_delta'].mean():+.9f}")
    print(
        "win_loss_tie: "
        f"{int((detail['accuracy_outcome'] == 'win').sum())}/"
        f"{int((detail['accuracy_outcome'] == 'loss').sum())}/"
        f"{int((detail['accuracy_outcome'] == 'tie').sum())}"
    )
    if args.print_summary:
        print()
        print(summary)


if __name__ == "__main__":
    main()
