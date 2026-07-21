#!/usr/bin/env python3
"""Optuna search over only TabPFN F-aware-C context reserve ratio."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py"
DEFAULT_DATA_ROOT = "results/tabiclv2_prior_data/graph_scm_3stage_512_seed42"
RESERVE_RATIO_LOW = 0.0001
RESERVE_RATIO_HIGH = 0.3


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_optuna():
    try:
        import optuna  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Optuna is required in the active environment") from exc
    return optuna


def sample_reserve_ratio(trial: Any) -> float:
    return float(
        trial.suggest_float(
            "ttt_c_reserve_ratio", RESERVE_RATIO_LOW, RESERVE_RATIO_HIGH
        )
    )


def format_ratio(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def trial_dir(output_root: Path, number: int, reserve_ratio: float) -> Path:
    return output_root / "trials" / (
        f"trial_{number:04d}_reserve_{format_ratio(reserve_ratio)}"
    )


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
    reserve_ratio: float,
    ttt_lr: float,
    ttt_epochs: int,
    ttt_query_ratio: float,
    ttt_c_selection: str,
    ttt_c_metric: str,
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
        "--data-root", str(data_root),
        "--out-dir", str(out_dir),
        "--workers", str(workers),
        "--gpus", str(gpus),
        "--model-version", str(model_version),
        "--random-state", str(random_state),
        "--n-estimators", str(n_estimators),
        "--ttt",
        "--ttt-epochs", str(ttt_epochs),
        "--ttt-lr", f"{ttt_lr:.10g}",
        "--ttt-weight-decay", f"{ttt_weight_decay:.10g}",
        "--ttt-patience", str(ttt_patience),
        "--ttt-min-delta", f"{ttt_min_delta:.10g}",
        "--ttt-eval-metric", "acc",
        "--ttt-validation-fraction", f"{ttt_validation_fraction:.10g}",
        "--ttt-n-estimators-finetune", str(ttt_n_estimators_finetune),
        "--ttt-validation-n-estimators", str(ttt_validation_n_estimators),
        "--ttt-grad-accumulation-steps", str(ttt_grad_accumulation_steps),
        "--ttt-query-ratio", f"{ttt_query_ratio:.10g}",
        "--ttt-c-selection", str(ttt_c_selection),
        "--ttt-c-metric", str(ttt_c_metric),
        "--ttt-c-reserve-ratio", f"{reserve_ratio:.10g}",
    ]
    if model_path:
        command.extend(["--model-path", str(resolve_repo_path(model_path))])
    if max_datasets is not None:
        command.extend(["--max-datasets", str(max_datasets)])
    return command


def has_valid_trial_outputs(out_dir: Path) -> bool:
    return (out_dir / "all_classification_results.csv").is_file() and (
        out_dir / "summary.txt"
    ).is_file()


def ensure_trial_outputs(
    *,
    out_dir: Path,
    log_path: Path,
    command: list[str],
    cwd: Path = REPO_ROOT,
    force: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
) -> bool:
    if not force and has_valid_trial_outputs(out_dir):
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        completed = (runner or subprocess.run)(
            command,
            cwd=cwd,
            env=env,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"trial command failed with returncode={completed.returncode}")
    if not has_valid_trial_outputs(out_dir):
        raise FileNotFoundError(f"missing trial outputs under {out_dir}")
    return False


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_trial_outputs(
    out_dir: Path, *, expected_datasets: int | None = None
) -> dict[str, Any]:
    path = out_dir / "all_classification_results.csv"
    ok: list[float] = []
    failed_count = 0
    applied_count = 0
    oom_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if expected_datasets is not None and len(rows) != expected_datasets:
        raise ValueError(
            f"dataset count mismatch: expected={expected_datasets} actual={len(rows)}"
        )
    for row in rows:
        if row.get("status") == "ok" and as_float(row.get("accuracy")) is not None:
            ok.append(float(row["accuracy"]))
        else:
            failed_count += 1
        applied_count += str(row.get("ttt_applied", "")).lower() in {"1", "true", "yes"}
        oom_count += str(row.get("ttt_oom_fallback", "")).lower() in {"1", "true", "yes"}
    if not ok:
        raise ValueError(f"no status=ok accuracy rows in {path}")
    if expected_datasets is not None and len(ok) != expected_datasets:
        raise ValueError(
            f"status=ok count mismatch: expected={expected_datasets} actual={len(ok)}"
        )
    objective = sum(ok) / len(ok)
    return {
        "objective": objective,
        "avg_accuracy_ok": objective,
        "ok_count": len(ok),
        "failed_count": failed_count,
        "ttt_applied_count": applied_count,
        "ttt_oom_fallback_count": oom_count,
    }


def trial_state_name(trial: Any) -> str:
    state = getattr(trial, "state", "")
    return str(getattr(state, "name", state))


def trial_record(trial: Any) -> dict[str, Any]:
    attrs = getattr(trial, "user_attrs", {}) or {}
    params = getattr(trial, "params", {}) or {}
    return {
        "number": getattr(trial, "number", ""),
        "state": trial_state_name(trial),
        "value": getattr(trial, "value", None),
        "ttt_c_reserve_ratio": params.get("ttt_c_reserve_ratio"),
        "ok_count": attrs.get("ok_count"),
        "failed_count": attrs.get("failed_count"),
        "ttt_applied_count": attrs.get("ttt_applied_count"),
        "ttt_oom_fallback_count": attrs.get("ttt_oom_fallback_count"),
        "avg_accuracy_ok": attrs.get("avg_accuracy_ok"),
        "out_dir": attrs.get("out_dir"),
        "log_path": attrs.get("log_path"),
        "command": attrs.get("command"),
        "error": attrs.get("error"),
    }


def write_artifacts(
    output_root: Path,
    *,
    study_name: str,
    trials: list[Any],
    best_trial: Any | None,
    dry_run: bool = False,
) -> None:
    records = trials if dry_run else [trial_record(trial) for trial in trials]
    fieldnames = list(records[0]) if records else list(trial_record(type("T", (), {})()))
    with (output_root / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    complete = [r for r in records if str(r.get("state", "")).upper() == "COMPLETE"]
    failed = [r for r in records if str(r.get("state", "")).upper() == "FAIL"]
    best_record = trial_record(best_trial) if best_trial is not None else None
    if best_record is not None:
        (output_root / "best_params.json").write_text(
            json.dumps(
                {"ttt_c_reserve_ratio": best_record["ttt_c_reserve_ratio"]},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        payload = dict(best_record)
        payload["objective"] = payload.pop("value")
        (output_root / "best_trial_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    summary = [
        f"study_name: {study_name}",
        f"total_trials: {len(records)}",
        f"completed_trials: {len(complete)}",
        f"failed_trials: {len(failed)}",
        f"reserve_ratio_low: {RESERVE_RATIO_LOW}",
        f"reserve_ratio_high: {RESERVE_RATIO_HIGH}",
    ]
    if dry_run:
        summary.append(f"dry_run_trials: {len(records)}")
    if best_record is not None:
        summary.extend(
            [
                f"best_trial: {best_record['number']}",
                f"best_objective: {best_record['value']}",
                f"best_ttt_c_reserve_ratio: {best_record['ttt_c_reserve_ratio']}",
            ]
        )
    (output_root / "study_summary.txt").write_text("\n".join(summary) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--study-name", default="tabpfnv3_graph_scm_reserve_ratio")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-startup-trials", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--runner-path", default=str(RUNNER_PATH))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--model-version", choices=["v2", "v2.5", "v2.6", "v3"], default="v3")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--expected-datasets", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-epochs", type=int, default=30)
    parser.add_argument(
        "--ttt-c-selection",
        choices=["f_test_centroid_reserve"],
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


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_root = resolve_repo_path(args.data_root)
    runner_path = resolve_repo_path(args.runner_path)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not runner_path.is_file():
        raise FileNotFoundError(f"runner not found: {runner_path}")
    if not 0 < RESERVE_RATIO_LOW < RESERVE_RATIO_HIGH < 1:
        raise ValueError("invalid reserve ratio bounds")
    if RESERVE_RATIO_HIGH + args.ttt_query_ratio >= 1:
        raise ValueError("reserve_ratio_high + query_ratio must be < 1")
    if args.n_trials < 1 or args.n_startup_trials < 1:
        raise ValueError("trial counts must be >= 1")
    if args.output_root:
        output_root = resolve_repo_path(args.output_root)
    else:
        output_root = resolve_repo_path(
            "results/tabpfn/v3/optuna_reserve_ratio/tabpfnv3_graph_scm_seed42"
        )
    return data_root, runner_path, output_root


def make_command(args: argparse.Namespace, data_root: Path, runner_path: Path, out_dir: Path, ratio: float) -> list[str]:
    return build_trial_command(
        python_bin=args.python_bin, runner_path=runner_path, data_root=data_root,
        out_dir=out_dir, workers=args.workers, gpus=args.gpus,
        model_version=args.model_version, model_path=args.model_path,
        random_state=args.random_state, max_datasets=args.max_datasets,
        n_estimators=args.n_estimators, reserve_ratio=ratio, ttt_lr=args.ttt_lr,
        ttt_epochs=args.ttt_epochs, ttt_query_ratio=args.ttt_query_ratio,
        ttt_c_selection=args.ttt_c_selection, ttt_c_metric=args.ttt_c_metric,
        ttt_weight_decay=args.ttt_weight_decay, ttt_patience=args.ttt_patience,
        ttt_min_delta=args.ttt_min_delta,
        ttt_validation_fraction=args.ttt_validation_fraction,
        ttt_n_estimators_finetune=args.ttt_n_estimators_finetune,
        ttt_validation_n_estimators=args.ttt_validation_n_estimators,
        ttt_grad_accumulation_steps=args.ttt_grad_accumulation_steps,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_root, runner_path, output_root = validate_args(args)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "trials").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    if args.dry_run:
        rng = random.Random(args.sampler_seed)
        records = []
        for number in range(args.n_trials):
            ratio = rng.uniform(RESERVE_RATIO_LOW, RESERVE_RATIO_HIGH)
            out_dir = trial_dir(output_root, number, ratio)
            command = make_command(args, data_root, runner_path, out_dir, ratio)
            print(shlex.join(command))
            records.append({
                "number": number, "state": "DRY_RUN", "value": "",
                "ttt_c_reserve_ratio": ratio, "ok_count": "", "failed_count": "",
                "ttt_applied_count": "", "ttt_oom_fallback_count": "",
                "avg_accuracy_ok": "", "out_dir": str(out_dir),
                "log_path": str(output_root / "logs" / f"trial_{number:04d}.log"),
                "command": shlex.join(command), "error": "",
            })
        write_artifacts(output_root, study_name=args.study_name, trials=records, best_trial=None, dry_run=True)
        return 0
    optuna = load_optuna()
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed, n_startup_trials=args.n_startup_trials
    )
    storage = f"sqlite:///{(output_root / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        sampler=sampler, load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        ratio = sample_reserve_ratio(trial)
        out_dir = trial_dir(output_root, int(trial.number), ratio)
        log_path = output_root / "logs" / f"trial_{int(trial.number):04d}.log"
        command = make_command(args, data_root, runner_path, out_dir, ratio)
        for key, value in {
            "out_dir": str(out_dir), "log_path": str(log_path),
            "command": shlex.join(command),
        }.items():
            trial.set_user_attr(key, value)
        print(f"trial_command: {shlex.join(command)}", flush=True)
        try:
            reused = ensure_trial_outputs(
                out_dir=out_dir, log_path=log_path, command=command,
                cwd=REPO_ROOT, force=args.force,
            )
            metrics = evaluate_trial_outputs(
                out_dir, expected_datasets=args.expected_datasets
            )
            trial.set_user_attr("reused_cache", reused)
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return float(metrics["objective"])
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    existing = len(getattr(study, "trials", []))
    remaining = max(0, args.n_trials - existing)
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=args.timeout, catch=(Exception,))
    trials = list(getattr(study, "trials", []))
    try:
        best_trial = study.best_trial
    except Exception:
        best_trial = None
    write_artifacts(
        output_root, study_name=args.study_name, trials=trials,
        best_trial=best_trial,
    )
    completed = sum(trial_state_name(t).upper() == "COMPLETE" for t in trials)
    if best_trial is None or completed < args.n_trials:
        raise RuntimeError(
            f"search incomplete: required={args.n_trials} completed={completed}"
        )
    print(f"saved_trials: {output_root / 'trials.csv'}")
    print(f"saved_best_params: {output_root / 'best_params.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
