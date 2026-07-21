#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optuna search for TabPFN F-aware-C learning rate and reserve ratio.

This script is an outer orchestrator for
``baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py``. It does not
modify or import the benchmark runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "baseline_compare"
    / "TabPFN-main"
    / "Tabpfn_1c_ttt_faware_c.py"
)
LR_LOW = 1e-6
LR_HIGH = 1e-5
RESERVE_RATIO_LOW = 0.01
RESERVE_RATIO_HIGH = 0.25
DEFAULT_OUTPUT_METHOD = "data184_lr_reserve_ratio"


@dataclass(frozen=True)
class TrialPaths:
    out_dir: Path
    log_path: Path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def create_indexed_output_root(method_root: Path, run_prefix: str) -> Path:
    method_root.mkdir(parents=True, exist_ok=True)
    run_index = 1
    while True:
        candidate = method_root / f"{run_prefix}_{run_index}"
        try:
            candidate.mkdir()
        except FileExistsError:
            run_index += 1
            continue
        return candidate


def format_float_slug(value: float) -> str:
    text = f"{float(value):.3g}"
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace("e", "e")
    )


def format_trial_dir_name(
    trial_number: int,
    ttt_lr: float,
    ttt_c_reserve_ratio: float,
) -> str:
    return (
        f"trial_{int(trial_number):04d}_lr_{format_float_slug(ttt_lr)}"
        f"_rr_{format_float_slug(ttt_c_reserve_ratio)}"
    )


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


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot compute mean over an empty sequence")
    return float(sum(values) / len(values))


def load_optuna():
    try:
        import optuna  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for this search. Install it in the active "
            "environment, for example: pip install optuna"
        ) from exc
    return optuna


def sample_ttt_lr(trial: Any, *, low: float = LR_LOW, high: float = LR_HIGH) -> float:
    return float(trial.suggest_float("ttt_lr", float(low), float(high), log=True))


def sample_ttt_lr_dry_run(rng: random.Random, *, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(float(low)), math.log(float(high)))))


def sample_ttt_c_reserve_ratio(
    trial: Any,
    *,
    low: float = RESERVE_RATIO_LOW,
    high: float = RESERVE_RATIO_HIGH,
) -> float:
    return float(
        trial.suggest_float(
            "ttt_c_reserve_ratio",
            float(low),
            float(high),
            log=False,
        )
    )


def sample_ttt_c_reserve_ratio_dry_run(
    rng: random.Random,
    *,
    low: float,
    high: float,
) -> float:
    return float(rng.uniform(float(low), float(high)))


def build_trial_command(
    *,
    python_bin: str,
    runner_path: Path,
    data_root: Path,
    out_dir: Path,
    workers: int,
    gpus: str,
    model_version: str,
    model_path: str | None,
    random_state: int,
    max_datasets: int | None,
    n_estimators: int,
    ttt_lr: float,
    ttt_epochs: int,
    ttt_query_ratio: float,
    ttt_c_selection: str,
    ttt_c_metric: str,
    ttt_c_reserve_ratio: float,
    ttt_weight_decay: float,
    ttt_patience: int,
    ttt_min_delta: float,
    ttt_validation_fraction: float,
    ttt_n_estimators_finetune: int,
    ttt_validation_n_estimators: int,
    ttt_grad_accumulation_steps: int,
) -> list[str]:
    command = [
        python_bin,
        str(runner_path),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--workers",
        str(workers),
        "--gpus",
        str(gpus),
        "--random-state",
        str(random_state),
        "--n-estimators",
        str(n_estimators),
        "--model-version",
        str(model_version),
        "--ttt",
        "--ttt-epochs",
        str(ttt_epochs),
        "--ttt-lr",
        f"{float(ttt_lr):.10g}",
        "--ttt-weight-decay",
        f"{float(ttt_weight_decay):.10g}",
        "--ttt-patience",
        str(ttt_patience),
        "--ttt-min-delta",
        f"{float(ttt_min_delta):.10g}",
        "--ttt-eval-metric",
        "acc",
        "--ttt-validation-fraction",
        f"{float(ttt_validation_fraction):.10g}",
        "--ttt-n-estimators-finetune",
        str(ttt_n_estimators_finetune),
        "--ttt-validation-n-estimators",
        str(ttt_validation_n_estimators),
        "--ttt-grad-accumulation-steps",
        str(ttt_grad_accumulation_steps),
        "--ttt-query-ratio",
        f"{float(ttt_query_ratio):.10g}",
        "--ttt-c-selection",
        str(ttt_c_selection),
        "--ttt-c-metric",
        str(ttt_c_metric),
        "--ttt-c-reserve-ratio",
        f"{float(ttt_c_reserve_ratio):.10g}",
    ]
    if model_path:
        command.extend(["--model-path", str(resolve_repo_path(model_path))])
    if max_datasets is not None:
        command.extend(["--max-datasets", str(max_datasets)])
    return command


def has_valid_trial_outputs(out_dir: Path) -> bool:
    return (out_dir / "all_classification_results.csv").exists() and (
        out_dir / "summary.txt"
    ).exists()


def ensure_trial_outputs(
    *,
    out_dir: Path,
    log_path: Path,
    command: list[str],
    cwd: Path = REPO_ROOT,
    force: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
) -> bool:
    """Return True when a cached output was reused."""
    if not force and has_valid_trial_outputs(out_dir):
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    runner = runner or subprocess.run
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = runner(
            command,
            cwd=cwd,
            env=env,
            check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"trial command failed with returncode={completed.returncode}"
        )
    if not has_valid_trial_outputs(out_dir):
        raise FileNotFoundError(
            f"trial completed but expected outputs are missing under {out_dir}"
        )
    return False


def evaluate_trial_outputs(out_dir: Path) -> dict[str, Any]:
    result_csv = out_dir / "all_classification_results.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Missing trial result CSV: {result_csv}")

    ok_accuracies: list[float] = []
    failed_count = 0
    with result_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status == "ok":
                accuracy = as_float(row.get("accuracy"))
                if accuracy is not None:
                    ok_accuracies.append(accuracy)
            else:
                failed_count += 1

    if not ok_accuracies:
        raise ValueError(f"No status=ok accuracy rows found in {result_csv}")

    avg_accuracy_ok = mean(ok_accuracies)
    return {
        "objective": avg_accuracy_ok,
        "avg_accuracy_ok": avg_accuracy_ok,
        "ok_count": len(ok_accuracies),
        "failed_count": failed_count,
    }


def trial_paths(
    output_root: Path,
    trial_number: int,
    ttt_lr: float,
    ttt_c_reserve_ratio: float,
) -> TrialPaths:
    slug = format_trial_dir_name(trial_number, ttt_lr, ttt_c_reserve_ratio)
    return TrialPaths(
        out_dir=output_root / "trials" / slug,
        log_path=output_root / "logs" / f"{slug}.log",
    )


def record_from_trial(trial: Any) -> dict[str, Any]:
    attrs = getattr(trial, "user_attrs", {}) or {}
    params = getattr(trial, "params", {}) or {}
    return {
        "number": getattr(trial, "number", ""),
        "state": trial_state_name(trial),
        "value": getattr(trial, "value", None),
        "ttt_lr": params.get("ttt_lr"),
        "ttt_c_reserve_ratio": params.get("ttt_c_reserve_ratio"),
        "ok_count": attrs.get("ok_count"),
        "failed_count": attrs.get("failed_count"),
        "avg_accuracy_ok": attrs.get("avg_accuracy_ok"),
        "out_dir": attrs.get("out_dir"),
        "log_path": attrs.get("log_path"),
        "command": attrs.get("command"),
        "error": attrs.get("error"),
    }


def trial_state_name(trial: Any) -> str:
    state = getattr(trial, "state", "")
    return str(getattr(state, "name", state))


def get_best_trial_or_none(study: Any) -> Any | None:
    try:
        return study.best_trial
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_best_summary(
    output_root: Path,
    study_name: str,
    trials: list[Any],
    best_trial: Any | None,
) -> None:
    completed_trials = [
        trial for trial in trials if trial_state_name(trial).upper() == "COMPLETE"
    ]
    failed_trials = [
        trial for trial in trials if trial_state_name(trial).upper() == "FAIL"
    ]
    payload: dict[str, Any] = {
        "mode": "optuna",
        "study_name": study_name,
        "direction": "maximize",
        "total_trials": len(trials),
        "completed_trials": len(completed_trials),
        "failed_trials": len(failed_trials),
        "best_trial": None,
    }
    if best_trial is not None:
        best_record = record_from_trial(best_trial)
        best_record["objective"] = best_record.pop("value", None)
        payload["best_trial"] = best_record
    write_json(output_root / "best_summary.json", payload)


def validate_inputs(data_root: Path, runner_path: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner path does not exist: {runner_path}")


def validate_search_space(
    lr_low: float,
    lr_high: float,
    reserve_ratio_low: float,
    reserve_ratio_high: float,
    ttt_query_ratio: float,
    ttt_c_selection: str,
) -> None:
    if float(lr_low) <= 0:
        raise ValueError("--lr-low must be > 0 for log search")
    if float(lr_high) <= float(lr_low):
        raise ValueError("--lr-high must be greater than --lr-low")
    if not 0.0 < float(reserve_ratio_low) < float(reserve_ratio_high) < 1.0:
        raise ValueError(
            "reserve ratio bounds must satisfy "
            "0 < --reserve-ratio-low < --reserve-ratio-high < 1"
        )
    if not 0.0 < float(ttt_query_ratio) < 1.0:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if (
        str(ttt_c_selection) == "f_test_centroid_reserve"
        and float(reserve_ratio_high) + float(ttt_query_ratio) >= 1.0
    ):
        raise ValueError(
            "--reserve-ratio-high + --ttt-query-ratio must be < 1 for "
            "f_test_centroid_reserve"
        )


def create_objective(args: argparse.Namespace, output_root: Path):
    data_root = resolve_repo_path(args.data_root)
    runner_path = resolve_repo_path(args.runner_path)

    def objective(trial: Any) -> float:
        ttt_lr = sample_ttt_lr(
            trial,
            low=float(args.lr_low),
            high=float(args.lr_high),
        )
        ttt_c_reserve_ratio = sample_ttt_c_reserve_ratio(
            trial,
            low=float(args.reserve_ratio_low),
            high=float(args.reserve_ratio_high),
        )
        paths = trial_paths(
            output_root,
            int(trial.number),
            ttt_lr,
            ttt_c_reserve_ratio,
        )
        command = build_trial_command(
            python_bin=args.python_bin,
            runner_path=runner_path,
            data_root=data_root,
            out_dir=paths.out_dir,
            workers=int(args.workers),
            gpus=str(args.gpus),
            model_version=str(args.model_version),
            model_path=args.model_path,
            random_state=int(args.random_state),
            max_datasets=args.max_datasets,
            n_estimators=int(args.n_estimators),
            ttt_lr=ttt_lr,
            ttt_epochs=int(args.ttt_epochs),
            ttt_query_ratio=float(args.ttt_query_ratio),
            ttt_c_selection=str(args.ttt_c_selection),
            ttt_c_metric=str(args.ttt_c_metric),
            ttt_c_reserve_ratio=ttt_c_reserve_ratio,
            ttt_weight_decay=float(args.ttt_weight_decay),
            ttt_patience=int(args.ttt_patience),
            ttt_min_delta=float(args.ttt_min_delta),
            ttt_validation_fraction=float(args.ttt_validation_fraction),
            ttt_n_estimators_finetune=int(args.ttt_n_estimators_finetune),
            ttt_validation_n_estimators=int(args.ttt_validation_n_estimators),
            ttt_grad_accumulation_steps=int(args.ttt_grad_accumulation_steps),
        )
        trial.set_user_attr("out_dir", str(paths.out_dir))
        trial.set_user_attr("log_path", str(paths.log_path))
        trial.set_user_attr("command", shlex.join(command))
        print(f"trial_command: {shlex.join(command)}", flush=True)
        try:
            reused_cache = ensure_trial_outputs(
                out_dir=paths.out_dir,
                log_path=paths.log_path,
                command=command,
                cwd=REPO_ROOT,
                force=bool(args.force),
            )
            metrics = evaluate_trial_outputs(paths.out_dir)
            trial.set_user_attr("reused_cache", reused_cache)
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return float(metrics["objective"])
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    return objective


def dry_run_trials(args: argparse.Namespace, output_root: Path) -> None:
    data_root = resolve_repo_path(args.data_root)
    runner_path = resolve_repo_path(args.runner_path)
    rng = random.Random(int(args.sampler_seed))
    rows: list[dict[str, Any]] = []

    for number in range(int(args.n_trials)):
        ttt_lr = sample_ttt_lr_dry_run(
            rng,
            low=float(args.lr_low),
            high=float(args.lr_high),
        )
        ttt_c_reserve_ratio = sample_ttt_c_reserve_ratio_dry_run(
            rng,
            low=float(args.reserve_ratio_low),
            high=float(args.reserve_ratio_high),
        )
        paths = trial_paths(output_root, number, ttt_lr, ttt_c_reserve_ratio)
        command = build_trial_command(
            python_bin=args.python_bin,
            runner_path=runner_path,
            data_root=data_root,
            out_dir=paths.out_dir,
            workers=int(args.workers),
            gpus=str(args.gpus),
            model_version=str(args.model_version),
            model_path=args.model_path,
            random_state=int(args.random_state),
            max_datasets=args.max_datasets,
            n_estimators=int(args.n_estimators),
            ttt_lr=ttt_lr,
            ttt_epochs=int(args.ttt_epochs),
            ttt_query_ratio=float(args.ttt_query_ratio),
            ttt_c_selection=str(args.ttt_c_selection),
            ttt_c_metric=str(args.ttt_c_metric),
            ttt_c_reserve_ratio=ttt_c_reserve_ratio,
            ttt_weight_decay=float(args.ttt_weight_decay),
            ttt_patience=int(args.ttt_patience),
            ttt_min_delta=float(args.ttt_min_delta),
            ttt_validation_fraction=float(args.ttt_validation_fraction),
            ttt_n_estimators_finetune=int(args.ttt_n_estimators_finetune),
            ttt_validation_n_estimators=int(args.ttt_validation_n_estimators),
            ttt_grad_accumulation_steps=int(args.ttt_grad_accumulation_steps),
        )
        print(shlex.join(command))
        rows.append(
            {
                "number": number,
                "state": "DRY_RUN",
                "value": "",
                "ttt_lr": ttt_lr,
                "ttt_c_reserve_ratio": ttt_c_reserve_ratio,
                "ok_count": "",
                "failed_count": "",
                "avg_accuracy_ok": "",
                "out_dir": str(paths.out_dir),
                "log_path": str(paths.log_path),
                "command": shlex.join(command),
                "error": "",
            }
        )

    write_json(
        output_root / "best_summary.json",
        {
            "mode": "dry_run",
            "study_name": args.study_name,
            "direction": "maximize",
            "planned_trials": len(rows),
            "search_space": {
                "ttt_lr": {
                    "low": float(args.lr_low),
                    "high": float(args.lr_high),
                    "log": True,
                },
                "ttt_c_reserve_ratio": {
                    "low": float(args.reserve_ratio_low),
                    "high": float(args.reserve_ratio_high),
                    "log": False,
                },
            },
            "trials": rows,
            "best_trial": None,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna search for TabPFN F-aware-C --ttt-lr and "
            "--ttt-c-reserve-ratio."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data184", help="Root directory for benchmark datasets")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Result root. Defaults to "
            "results/tabpfn/<model-version>/optuna_lr/<numbered-run>."
        ),
    )
    parser.add_argument("--study-name", default="tabpfnv3_lr_reserve_ratio_data184")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--runner-path", default=str(RUNNER_PATH))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--model-version", choices=["v2", "v2.5", "v2.6", "v3"], default="v3")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--lr-low", type=float, default=LR_LOW)
    parser.add_argument("--lr-high", type=float, default=LR_HIGH)
    parser.add_argument("--reserve-ratio-low", type=float, default=RESERVE_RATIO_LOW)
    parser.add_argument("--reserve-ratio-high", type=float, default=RESERVE_RATIO_HIGH)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-epochs", type=int, default=30)
    parser.add_argument(
        "--ttt-c-selection",
        choices=["f_test_centroid_reserve", "random"],
        default="f_test_centroid_reserve",
    )
    parser.add_argument("--ttt-c-metric", choices=["raw_l2", "standardized_l2"], default="standardized_l2")
    parser.add_argument("--ttt-weight-decay", type=float, default=0.01)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--ttt-min-delta", type=float, default=1e-4)
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=2)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=2)
    parser.add_argument("--ttt-grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_root = resolve_repo_path(args.data_root)
    runner_path = resolve_repo_path(args.runner_path)
    validate_inputs(data_root, runner_path)
    validate_search_space(
        float(args.lr_low),
        float(args.lr_high),
        float(args.reserve_ratio_low),
        float(args.reserve_ratio_high),
        float(args.ttt_query_ratio),
        str(args.ttt_c_selection),
    )
    if args.output_root is None:
        version_slug = str(args.model_version).replace(".", "p")
        method_root = resolve_repo_path(
            f"results/tabpfn/{args.model_version}/{DEFAULT_OUTPUT_METHOD}"
        )
        output_root = create_indexed_output_root(
            method_root,
            f"tabpfn_{version_slug}_optuna_faware_c_lr_reserve_ratio",
        )
    else:
        output_root = resolve_repo_path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "trials").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run_trials(args, output_root)
        print(f"saved_summary: {output_root / 'best_summary.json'}")
        return 0

    optuna = load_optuna()
    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed))
    study = optuna.create_study(
        study_name=str(args.study_name),
        direction="maximize",
        sampler=sampler,
    )

    objective = create_objective(args, output_root)
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=args.timeout,
        catch=(Exception,),
    )

    best_trial = get_best_trial_or_none(study)
    trials = list(getattr(study, "trials", []))
    write_best_summary(
        output_root,
        str(args.study_name),
        trials,
        best_trial,
    )
    print(f"saved_summary: {output_root / 'best_summary.json'}")
    if best_trial is None:
        raise RuntimeError("Optuna search completed without a successful trial")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
