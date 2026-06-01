#!/usr/bin/env python3
"""Create a paper-style LaTeX table for compare_results metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


METRICS = ["accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"]
HIGHER_IS_BETTER = {
    "rank": False,
    "accuracy": True,
    "f1": True,
    "balanced_accuracy": True,
    "roc_auc": True,
    "log_loss": False,
}

METHOD_NAME_MAP = {
    "iclv2_ensemble32": "TabICLv2",
    "iclv2_ensmble32": "TabICLv2",
    "tabiclv2_ensmble32": "TabICLv2",
    "tabiclv2_ttt_default": "TabICLv2-TTT",
    "tabiclv1.1_ttt_default": "TabICLv1.1-TTT",
    "tabiclv1.1_ensemble32": "TabICLv1.1",
    "iclv1.1_ttt_default": "TabICLv1.1-TTT",
    "iclv1.1_ensemble32": "TabICLv1.1",
    "tabpfnv3_results_ensemble8": "TabPFNv3",
    "tabpfnv3_1c_ttt_epoch30_chunk2000_lr5e-6": "TabPFNv3-1C-TTT",
    "tabpfnv2.5_results_ensemble8": "TabPFN-2.5",
    "tabpfnv2_results_ensemble8": "TabPFNv2",
    "limix-16m_results_178": "LimiX",
    "limix_ttt_epoch30_chunk150": "LimiX-TTT",
    "orion_msp_results_ensemble32": "Orion-MSP",
    "tabr_results": "TabR",
}


def discover_results(results_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for csv_path in sorted(results_dir.glob("*/all_classification_results.csv")):
        raw_name = csv_path.parent.name
        display_name = METHOD_NAME_MAP.get(raw_name, raw_name)
        results.append((display_name, csv_path))
    if not results:
        raise FileNotFoundError(f"No */all_classification_results.csv files found under {results_dir}")

    names = [name for name, _ in results]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"Duplicate display method names after renaming: {duplicated}")
    return results


def load_metric_tables(results: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    columns_by_metric: dict[str, list[pd.Series]] = {metric: [] for metric in METRICS}
    for method_name, csv_path in results:
        df = pd.read_csv(csv_path)
        required = {"dataset_name", "status", *METRICS}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        ok_df = df.loc[df["status"].eq("ok"), ["dataset_name", *METRICS]].set_index("dataset_name")
        for metric in METRICS:
            columns_by_metric[metric].append(ok_df[metric].rename(method_name))

    return {metric: pd.concat(columns, axis=1) for metric, columns in columns_by_metric.items()}


def common_full_metric_index(metric_tables: dict[str, pd.DataFrame]) -> pd.Index:
    combined = pd.concat(metric_tables.values(), axis=1)
    return combined.dropna(axis=0, how="any").index


def rank_table(metric_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    higher_is_better = HIGHER_IS_BETTER[metric]

    def rank_row(row: pd.Series) -> pd.Series:
        values = row.to_numpy()
        ranking_values = -values if higher_is_better else values
        return pd.Series(rankdata(ranking_values, method="average"), index=row.index)

    return metric_df.apply(rank_row, axis=1)


def summarize(metric_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    common_index = common_full_metric_index(metric_tables)
    if common_index.empty:
        raise ValueError("No common successful full-metric datasets across all methods.")

    rows = {}
    accuracy_ranks = rank_table(metric_tables["accuracy"].loc[common_index], "accuracy").mean(axis=0)
    for method_name in accuracy_ranks.index:
        row = {
            "data178_rank": float(accuracy_ranks[method_name]),
        }
        for metric in METRICS:
            row[metric] = float(metric_tables[metric].loc[common_index, method_name].mean())
        rows[method_name] = row

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary = summary.sort_values(["data178_rank", "accuracy"], ascending=[True, False])
    return summary, len(common_index)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_rank(value: float) -> str:
    return f"{value:.2f}"


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def best_second_masks(summary: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    columns = ["data178_rank", *METRICS]
    masks = {}
    for column in columns:
        decimals = 2 if column == "data178_rank" else 4
        values = summary[column].round(decimals)
        ascending = not HIGHER_IS_BETTER["rank"] if column == "data178_rank" else not HIGHER_IS_BETTER[column]
        ordered_unique = np.array(sorted(values.dropna().unique(), reverse=not ascending))
        if len(ordered_unique) == 0:
            masks[column] = (pd.Series(False, index=summary.index), pd.Series(False, index=summary.index))
            continue
        best_value = ordered_unique[0]
        second_value = ordered_unique[1] if len(ordered_unique) > 1 else np.nan
        best = values.eq(best_value)
        second = values.eq(second_value) if not pd.isna(second_value) else pd.Series(False, index=summary.index)
        masks[column] = (best, second)
    return masks


def decorate(value: str, is_best: bool, is_second: bool) -> str:
    if is_best:
        return rf"\textbf{{{value}}}"
    if is_second:
        return rf"\underline{{{value}}}"
    return value


def render_latex(summary: pd.DataFrame, n_common: int, caption: str, label: str) -> str:
    masks = best_second_masks(summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Models & \multicolumn{6}{c}{Data178} \\",
        r"\cline{2-7}",
        r" & Rank & ACC & F1 & BAcc & AUC & LogLoss \\",
        r"\hline",
    ]

    columns = [
        ("data178_rank", format_rank),
        ("accuracy", format_metric),
        ("f1", format_metric),
        ("balanced_accuracy", format_metric),
        ("roc_auc", format_metric),
        ("log_loss", format_metric),
    ]
    for method_name, row in summary.iterrows():
        cells = [latex_escape(method_name)]
        for column, formatter in columns:
            best, second = masks[column]
            cells.append(decorate(formatter(row[column]), bool(best.loc[method_name]), bool(second.loc[method_name])))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            rf"\multicolumn{{7}}{{l}}{{\footnotesize Results are averaged over the common successful full-metric intersection (N={n_common}).}} \\",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baseline_compare/compare_results"),
        help="Directory containing method subdirectories with all_classification_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output LaTeX table path. Defaults to RESULTS_DIR/all_metrics_table.tex.",
    )
    parser.add_argument(
        "--caption",
        default="Classification results on Data178.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:compare-results-all-metrics",
        help="LaTeX table label.",
    )
    args = parser.parse_args()

    output = args.output or args.results_dir / "all_metrics_table.tex"
    results = discover_results(args.results_dir)
    metric_tables = load_metric_tables(results)
    summary, n_common = summarize(metric_tables)
    latex = render_latex(summary, n_common=n_common, caption=args.caption, label=args.label)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="utf-8")

    print(f"methods: {len(summary)}")
    print(f"common_successful_full_metric_datasets: {n_common}")
    print(summary.round({"all_rank": 2, "data178_rank": 2, **{metric: 4 for metric in METRICS}}).to_string())
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
