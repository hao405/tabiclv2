#!/usr/bin/env python3
"""Refresh the live MICP matrix summary from atomic cell summaries."""

from __future__ import annotations

import sys
from pathlib import Path


METRICS = (
    "avg_accuracy_ok",
    "avg_f1_ok",
    "avg_balanced_accuracy_ok",
    "avg_roc_auc_ok",
    "avg_log_loss_ok",
)


def read_summary(path: Path) -> dict[str, str]:
    return dict(
        line.split(": ", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if ": " in line
    )


def main() -> None:
    root = Path(sys.argv[1])
    summaries = [
        read_summary(root / model / "mixturepfn" / "summary.txt")
        for model in ("tabiclv2", "tabpfnv3")
    ]
    ok_counts = [int(summary["ok_count"]) for summary in summaries]
    total_ok = sum(ok_counts)

    def total(key: str) -> int:
        return sum(int(summary[key]) for summary in summaries)

    def weighted_metric(key: str) -> str:
        values = [
            (float(summary[key]), count)
            for summary, count in zip(summaries, ok_counts)
            if summary[key] != "(none)" and count
        ]
        if not values:
            return "(none)"
        return f"{sum(value * count for value, count in values) / total_ok:.6f}"

    failed_names = [
        summary["failed_datasets"]
        for summary in summaries
        if summary["failed_datasets"] != "(none)"
    ]
    skipped_names = [
        summary["skipped_datasets"]
        for summary in summaries
        if summary["skipped_datasets"] != "(none)"
    ]
    lines = [
        "discovered_datasets: 184",
        f"processed_datasets: {total('processed_datasets')}",
        f"ok_count: {total_ok}",
        f"failed_count: {total('failed_count')}",
        f"skipped_count: {total('skipped_count')}",
        "ft_oom_fallback_count: 0",
        *(f"{metric}: {weighted_metric(metric)}" for metric in METRICS),
        (
            "wall_seconds: "
            f"{sum(float(summary['wall_seconds']) for summary in summaries):.3f}"
        ),
        f"failed_datasets: {', '.join(failed_names) if failed_names else '(none)'}",
        f"skipped_datasets: {', '.join(skipped_names) if skipped_names else '(none)'}",
        "ft_oom_fallback_datasets: (none)",
    ]
    temporary = root / "summary.txt.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(root / "summary.txt")


if __name__ == "__main__":
    main()
