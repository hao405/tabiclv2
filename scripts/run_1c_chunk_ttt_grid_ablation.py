#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a resumable 1C Chunk TTT ablation grid.

This script only orchestrates calls to ``1C_Chunk_TTT.py``. It does not modify
the benchmark implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = Path("1c_result_v2/ablation_1c_chunk_grid40_seed42")
DEFAULT_MODEL_PATH = "tabicl-classifier-v2-20260212.ckpt"

LR_VALUES = ("5e-6", "1e-5", "3e-5")
CHUNK_SIZES = (3000, 5000, 10000, 20000)

FREEZE_MODES = {
    "full": {
        "freeze_col": False,
        "freeze_row": False,
        "freeze_icl": False,
    },
    "only_icl_predictor": {
        "freeze_col": True,
        "freeze_row": True,
        "freeze_icl": False,
    },
    "only_row_interactor": {
        "freeze_col": True,
        "freeze_row": False,
        "freeze_icl": True,
    },
    "only_col_embedder": {
        "freeze_col": False,
        "freeze_row": True,
        "freeze_icl": True,
    },
}

SUMMARY_COLUMNS = (
    "trial_slug",
    "status",
    "returncode",
    "lr",
    "chunk_size",
    "freeze_mode",
    "freeze_col",
    "freeze_row",
    "freeze_icl",
    "ok_count",
    "failed_count",
    "skipped_count",
    "ttt_oom_fallback_count",
    "avg_accuracy_ok",
    "avg_f1_ok",
    "avg_balanced_accuracy_ok",
    "avg_roc_auc_ok",
    "avg_log_loss_ok",
    "avg_fit_seconds_ok",
    "avg_predict_seconds_ok",
    "baseline_intersection_n",
    "baseline_accuracy_ok",
    "accuracy_delta_vs_baseline",
    "accuracy_win_vs_baseline",
    "accuracy_loss_vs_baseline",
    "accuracy_tie_vs_baseline",
    "elapsed_seconds",
    "started_at",
    "finished_at",
    "out_dir",
    "log_path",
    "command",
    "error",
)

EPS = 1e-12


@dataclass(frozen=True)
class Trial:
    slug: str
    lr: str
    chunk_size: int
    freeze_mode: str
    freeze_col: bool
    freeze_row: bool
    freeze_icl: bool
    out_dir: Path
    log_path: Path
    command: tuple[str, ...]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_optional_int(value: str) -> int | None:
    lowered = str(value).strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return int(value)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(none)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_optional(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def generate_trials(args: argparse.Namespace, out_root: Path) -> list[Trial]:
    trials: list[Trial] = []
    trial_root = out_root / "trials"
    log_root = out_root / "logs"
    script_path = REPO_ROOT / "1C_Chunk_TTT.py"

    for lr in LR_VALUES:
        for chunk_size in CHUNK_SIZES:
            for freeze_mode, freeze in FREEZE_MODES.items():
                slug = f"lr{lr}__chunk{chunk_size}__{freeze_mode}"
                out_dir = trial_root / slug
                log_path = log_root / f"{slug}.log"
                command = [
                    args.python_bin,
                    str(script_path),
                    "--data-root",
                    str(resolve_repo_path(args.data_root)),
                    "--model-path",
                    str(resolve_repo_path(args.model_path)),
                    "--out-dir",
                    str(out_dir),
                    "--workers",
                    "1",
                    "--gpu-groups",
                    str(args.gpu_id),
                    "--max-datasets",
                    str(args.max_datasets),
                    "--n-estimators",
                    "32",
                    "--batch-size",
                    "8",
                    "--kv-cache",
                    "False",
                    "--use-amp",
                    "auto",
                    "--use-fa3",
                    "auto",
                    "--offload-mode",
                    "auto",
                    "--random-state",
                    str(args.random_state),
                    "--ttt-lr",
                    lr,
                    "--ttt-scheduler",
                    "cosine_warmup",
                    "--ttt-warmup-proportion",
                    "0.1",
                    "--ttt-grad-clip",
                    "1.0",
                    "--ttt-dtype",
                    "float32",
                    "--ttt-micro-batch-size",
                    "1",
                    "--ttt-weight-decay",
                    "0.01",
                    "--ttt-epochs",
                    "30",
                    "--ttt-max-chunk-size",
                    str(chunk_size),
                    "--ttt-min-chunk-size",
                    "50",
                    "--ttt-query-ratio",
                    "0.2",
                    "--ttt-n-estimators-finetune",
                    "2",
                    "--ttt-early-stopping",
                    "True",
                    "--ttt-patience",
                    "8",
                    "--ttt-min-delta",
                    "1e-4",
                    "--ttt-eval-metric",
                    "accuracy",
                    "--ttt-validation-fraction",
                    "0.1",
                    "--ttt-validation-n-estimators",
                    "2",
                    "--ttt-freeze-col",
                    bool_arg(freeze["freeze_col"]),
                    "--ttt-freeze-row",
                    bool_arg(freeze["freeze_row"]),
                    "--ttt-freeze-icl",
                    bool_arg(freeze["freeze_icl"]),
                    "--ttt-save-ckpt",
                    "False",
                    "--no-ttt-data-parallel",
                ]
                trials.append(
                    Trial(
                        slug=slug,
                        lr=lr,
                        chunk_size=chunk_size,
                        freeze_mode=freeze_mode,
                        freeze_col=bool(freeze["freeze_col"]),
                        freeze_row=bool(freeze["freeze_row"]),
                        freeze_icl=bool(freeze["freeze_icl"]),
                        out_dir=out_dir,
                        log_path=log_path,
                        command=tuple(command),
                    )
                )
    if args.trial_limit is not None:
        return trials[: args.trial_limit]
    return trials


def read_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("trial_slug", "")): dict(row) for row in reader if row.get("trial_slug")}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_optional(row.get(column)) for column in columns})


def read_result_rows(trial: Trial) -> list[dict[str, str]]:
    result_csv = trial.out_dir / "all_classification_results.csv"
    if not result_csv.exists():
        return []
    with result_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary_values(trial: Trial) -> dict[str, str]:
    summary_path = trial.out_dir / "summary.txt"
    if not summary_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        values[key.strip()] = raw_value.strip()
    return values


def load_baseline_accuracy(baseline_dir: Path | None) -> dict[str, float]:
    if baseline_dir is None:
        return {}
    csv_path = baseline_dir / "all_classification_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing baseline result CSV: {csv_path}")
    values: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).lower() != "ok":
                continue
            dataset = str(row.get("dataset_name", "")).strip()
            accuracy = as_float(row.get("accuracy"))
            if dataset and accuracy is not None:
                values[dataset] = accuracy
    return values


def result_metrics(trial: Trial, baseline_accuracy: dict[str, float]) -> dict[str, Any]:
    rows = read_result_rows(trial)
    summary_values = read_summary_values(trial)
    if not rows:
        return {}

    ok_rows = [row for row in rows if str(row.get("status", "")).lower() == "ok"]
    failed_rows = [row for row in rows if str(row.get("status", "")).lower() == "fail"]
    skipped_rows = [row for row in rows if str(row.get("status", "")).lower() == "skip"]

    metrics: dict[str, Any] = {
        "ok_count": len(ok_rows),
        "failed_count": len(failed_rows),
        "skipped_count": len(skipped_rows),
        "ttt_oom_fallback_count": sum(
            str(row.get("ttt_oom_fallback", "")).strip().lower() in {"true", "1", "yes"}
            for row in ok_rows
        ),
        "avg_accuracy_ok": mean(as_float(row.get("accuracy")) for row in ok_rows),
        "avg_f1_ok": mean(as_float(row.get("f1")) for row in ok_rows),
        "avg_balanced_accuracy_ok": mean(as_float(row.get("balanced_accuracy")) for row in ok_rows),
        "avg_roc_auc_ok": mean(as_float(row.get("roc_auc")) for row in ok_rows),
        "avg_log_loss_ok": mean(as_float(row.get("log_loss")) for row in ok_rows),
        "avg_fit_seconds_ok": mean(as_float(row.get("fit_seconds")) for row in ok_rows),
        "avg_predict_seconds_ok": mean(as_float(row.get("predict_seconds")) for row in ok_rows),
    }
    if "wall_seconds" in summary_values:
        metrics["wall_seconds"] = as_float(summary_values["wall_seconds"])

    if baseline_accuracy:
        trial_pairs: list[tuple[float, float]] = []
        for row in ok_rows:
            dataset = str(row.get("dataset_name", "")).strip()
            acc = as_float(row.get("accuracy"))
            base = baseline_accuracy.get(dataset)
            if acc is not None and base is not None:
                trial_pairs.append((acc, base))
        deltas = [acc - base for acc, base in trial_pairs]
        metrics.update(
            {
                "baseline_intersection_n": len(trial_pairs),
                "baseline_accuracy_ok": mean(base for _acc, base in trial_pairs),
                "accuracy_delta_vs_baseline": mean(deltas),
                "accuracy_win_vs_baseline": sum(delta > EPS for delta in deltas),
                "accuracy_loss_vs_baseline": sum(delta < -EPS for delta in deltas),
                "accuracy_tie_vs_baseline": sum(abs(delta) <= EPS for delta in deltas),
            }
        )

    return metrics


def base_record(trial: Trial) -> dict[str, Any]:
    return {
        "trial_slug": trial.slug,
        "lr": trial.lr,
        "chunk_size": trial.chunk_size,
        "freeze_mode": trial.freeze_mode,
        "freeze_col": trial.freeze_col,
        "freeze_row": trial.freeze_row,
        "freeze_icl": trial.freeze_icl,
        "out_dir": str(trial.out_dir),
        "log_path": str(trial.log_path),
        "command": shlex.join(trial.command),
    }


def is_resumable_ok(trial: Trial, previous: dict[str, Any]) -> bool:
    if str(previous.get("status", "")).lower() != "ok":
        return False
    return (
        (trial.out_dir / "all_classification_results.csv").exists()
        and (trial.out_dir / "summary.txt").exists()
    )


def merge_records(
    trials: list[Trial],
    state: dict[str, dict[str, Any]],
    baseline_accuracy: dict[str, float],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for trial in trials:
        record = base_record(trial)
        record.update(state.get(trial.slug, {}))
        record.update(base_record(trial))
        record.update(result_metrics(trial, baseline_accuracy))
        merged.append(record)
    return merged


def sort_key_accuracy(row: dict[str, Any]) -> tuple[int, float, float, str]:
    acc = as_float(row.get("avg_accuracy_ok"))
    oom = as_float(row.get("ttt_oom_fallback_count"))
    if acc is None:
        return (1, 0.0, float("inf"), str(row.get("trial_slug", "")))
    return (0, -acc, oom if oom is not None else float("inf"), str(row.get("trial_slug", "")))


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "") or "pending")
        status_counts[status] = status_counts.get(status, 0) + 1

    ranked = sorted(rows, key=sort_key_accuracy)
    completed = [row for row in ranked if str(row.get("status", "")).lower() == "ok"]
    top_rows = completed[:20]

    top_columns = [
        "trial_slug",
        "avg_accuracy_ok",
        "avg_balanced_accuracy_ok",
        "avg_roc_auc_ok",
        "avg_log_loss_ok",
        "ttt_oom_fallback_count",
        "elapsed_seconds",
    ]
    if any(row.get("accuracy_delta_vs_baseline") not in {None, ""} for row in rows):
        top_columns.extend(
            [
                "baseline_intersection_n",
                "accuracy_delta_vs_baseline",
                "accuracy_win_vs_baseline",
                "accuracy_loss_vs_baseline",
            ]
        )

    best_by_freeze: list[dict[str, Any]] = []
    for freeze_mode in FREEZE_MODES:
        subset = [row for row in completed if row.get("freeze_mode") == freeze_mode]
        if subset:
            best_by_freeze.append(sorted(subset, key=sort_key_accuracy)[0])

    lines = [
        "# 1C Chunk TTT Grid Ablation Summary",
        "",
        f"- total_trials: {len(rows)}",
        f"- status_counts: {json.dumps(status_counts, sort_keys=True)}",
        "",
        "## Top Configs",
        "",
        markdown_table(top_rows, top_columns),
        "",
        "## Best By Freeze Mode",
        "",
        markdown_table(
            best_by_freeze,
            [
                "freeze_mode",
                "trial_slug",
                "avg_accuracy_ok",
                "avg_balanced_accuracy_ok",
                "ttt_oom_fallback_count",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_outputs(
    out_root: Path,
    trials: list[Trial],
    state: dict[str, dict[str, Any]],
    baseline_accuracy: dict[str, float],
) -> None:
    rows = merge_records(trials, state, baseline_accuracy)
    write_csv(out_root / "grid_trials.csv", rows, SUMMARY_COLUMNS)
    write_csv(out_root / "grid_summary.csv", rows, SUMMARY_COLUMNS)
    ranked = sorted(rows, key=sort_key_accuracy)
    write_csv(out_root / "grid_ranked_by_accuracy.csv", ranked, SUMMARY_COLUMNS)
    write_markdown_summary(out_root / "grid_summary.md", ranked)


def run_trial(trial: Trial) -> dict[str, Any]:
    record = base_record(trial)
    started_at = now_iso()
    start = time.time()
    trial.out_dir.mkdir(parents=True, exist_ok=True)
    trial.log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with trial.log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"started_at: {started_at}\n")
        log_handle.write(f"command: {shlex.join(trial.command)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            list(trial.command),
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        finished_at = now_iso()
        elapsed = time.time() - start
        log_handle.write("\n")
        log_handle.write(f"finished_at: {finished_at}\n")
        log_handle.write(f"returncode: {completed.returncode}\n")
        log_handle.write(f"elapsed_seconds: {elapsed:.3f}\n")

    record.update(
        {
            "status": "ok" if completed.returncode == 0 else "fail",
            "returncode": int(completed.returncode),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "error": "" if completed.returncode == 0 else f"returncode={completed.returncode}",
        }
    )
    return record


def validate_data_root(data_root: Path, max_datasets: int) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    dataset_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    if len(dataset_dirs) < max_datasets:
        raise ValueError(
            f"Data root {data_root} has only {len(dataset_dirs)} dataset dirs, "
            f"but --max-datasets={max_datasets} was requested."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 48-config 1C Chunk TTT 40-dataset ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data200")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--trial-limit", type=parse_optional_int, default=None)
    parser.add_argument("--baseline-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_root = resolve_repo_path(args.data_root)
    out_root = resolve_repo_path(args.out_root)
    baseline_dir = resolve_repo_path(args.baseline_dir) if args.baseline_dir else None

    validate_data_root(data_root, int(args.max_datasets))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "trials").mkdir(parents=True, exist_ok=True)

    trials = generate_trials(args, out_root)
    state_path = out_root / "grid_trials.csv"
    state = read_state(state_path)
    baseline_accuracy = load_baseline_accuracy(baseline_dir)

    if args.dry_run:
        for trial in trials:
            record = base_record(trial)
            record.update(
                {
                    "status": "dry_run",
                    "returncode": "",
                    "started_at": "",
                    "finished_at": "",
                    "elapsed_seconds": 0.0,
                    "error": "",
                }
            )
            state[trial.slug] = record
            print(shlex.join(trial.command))
        refresh_outputs(out_root, trials, state, baseline_accuracy)
        print(f"dry_run_trials: {len(trials)}")
        print(f"saved_grid_trials: {state_path}")
        print(f"saved_grid_summary: {out_root / 'grid_summary.csv'}")
        return

    for index, trial in enumerate(trials, start=1):
        previous = state.get(trial.slug, {})
        if not args.force and is_resumable_ok(trial, previous):
            print(f"[{index}/{len(trials)}] skip ok {trial.slug}")
            continue

        started_record = base_record(trial)
        started_record.update(
            {
                "status": "running",
                "returncode": "",
                "started_at": now_iso(),
                "finished_at": "",
                "elapsed_seconds": "",
                "error": "",
            }
        )
        state[trial.slug] = started_record
        refresh_outputs(out_root, trials, state, baseline_accuracy)

        print(f"[{index}/{len(trials)}] run {trial.slug}")
        print(shlex.join(trial.command))
        result = run_trial(trial)
        state[trial.slug] = result
        refresh_outputs(out_root, trials, state, baseline_accuracy)
        if result["status"] != "ok":
            print(f"[{index}/{len(trials)}] fail {trial.slug}; see {trial.log_path}")
        else:
            print(f"[{index}/{len(trials)}] ok {trial.slug}")

    refresh_outputs(out_root, trials, state, baseline_accuracy)
    print(f"saved_grid_trials: {state_path}")
    print(f"saved_grid_summary: {out_root / 'grid_summary.csv'}")
    print(f"saved_grid_ranked: {out_root / 'grid_ranked_by_accuracy.csv'}")
    print(f"saved_markdown_summary: {out_root / 'grid_summary.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
