#!/usr/bin/env python3
"""Unified launcher for TabICL/TabPFN infer, random-FT, and F-aware FT runs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


MODEL_CHOICES = ("tabicl-v1.1", "tabicl-v2", "tabpfn-v2", "tabpfn-v3")
METHOD_CHOICES = ("infer", "ft", "faware_ft")
TABICL_MODELS = {"tabicl-v1.1", "tabicl-v2"}
TABPFN_MODELS = {"tabpfn-v2", "tabpfn-v3"}

TABICL_CHECKPOINTS = {
    "tabicl-v1.1": "tabicl-classifier-v1.1-20250506.ckpt",
    "tabicl-v2": "tabicl-classifier-v2-20260212.ckpt",
}
TABPFN_V2_CHECKPOINT = (
    "baseline_compare/TabPFN-main/tabpfn-v2-classifier-v2_default.ckpt"
)
TABPFN_V3_BINARY_CHECKPOINT = (
    "baseline_compare/TabPFN-main/"
    "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
)
TABPFN_V3_MULTICLASS_CHECKPOINT = (
    "baseline_compare/TabPFN-main/"
    "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
)

TABICL_INFER_RUNNER = "benchmark.py"
TABICL_ADAPT_RUNNER = "1C_Chunk_FT/1C_Chunk_FT_Faware_C.py"
TABPFN_INFER_RUNNER = "baseline_compare/TabPFN-main/benchmark_infer.py"
TABPFN_ADAPT_RUNNER = (
    "baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py"
)


@dataclass(frozen=True)
class LaunchSpec:
    model: str
    method: str
    model_family: str
    selection: str | None
    source: str | None
    metric: str | None
    checkpoint: str | None
    runner: str
    data_root: str
    out_dir: str
    run_id: str
    workers: int
    devices: str
    n_estimators: int
    random_state: int
    command: list[str]
    passthrough_args: list[str]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one TabICL/TabPFN model x method experiment. "
            "The ft method uses random context/query selection; faware_ft uses f_mmd."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_CHOICES)
    parser.add_argument("--method", choices=METHOD_CHOICES)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the fixed 4-model x 3-method matrix sequentially on one data root.",
    )
    parser.add_argument(
        "--matrix-resume",
        action="store_true",
        help="In matrix mode, skip cells whose manager_manifest.json is already ok.",
    )
    parser.add_argument(
        "--matrix-fail-fast",
        action="store_true",
        help="In matrix mode, stop after the first failed cell instead of continuing.",
    )
    parser.add_argument(
        "--matrix-models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=None,
        help="Optional model subset for --matrix; defaults to the fixed full model axis.",
    )
    parser.add_argument(
        "--matrix-methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=None,
        help="Optional method subset for --matrix; defaults to the fixed full method axis.",
    )
    parser.add_argument("--data-root", default="data184")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--output-root", default="results/managed_experiments")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier used in the automatic output path; defaults to a timestamp.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated physical GPU ids; one id is assigned to each worker.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Defaults to 32 for TabICL and 8 for TabPFN.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the launcher-selected checkpoint with one explicit checkpoint.",
    )
    parser.add_argument("--ttt-epochs", type=int, default=30)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument(
        "--ttt-eval-metric",
        choices=["accuracy", "roc_auc", "log_loss"],
        default="accuracy",
    )
    parser.add_argument(
        "--ttt-c-metric",
        choices=["standardized_l2", "model_native_l2"],
        default="standardized_l2",
        help=(
            "Unified selection metric. model_native_l2 maps to tabicl_encoded_l2 "
            "for TabICL and raw_l2 for TabPFN."
        ),
    )
    parser.add_argument("--ttt-c-reserve-ratio", type=float, default=0.05)
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=2)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=2)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without creating outputs or launching it.",
    )
    return parser


def split_launcher_and_passthrough(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv_list = list(argv)
    if "--" not in argv_list:
        return argv_list, []
    separator_index = argv_list.index("--")
    return argv_list[:separator_index], argv_list[separator_index + 1 :]


def parse_device_ids(value: str) -> list[str]:
    device_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not device_ids:
        raise ValueError("--devices must contain at least one GPU id")
    if any(not item.isdigit() for item in device_ids):
        raise ValueError("--devices must be a comma-separated list of non-negative integers")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--devices must not contain duplicate GPU ids")
    return device_ids


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_args(args: argparse.Namespace, device_ids: list[str]) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if len(device_ids) != args.workers:
        raise ValueError(
            f"--devices contains {len(device_ids)} ids but --workers={args.workers}; "
            "provide exactly one GPU id per worker"
        )
    if args.n_estimators is not None and args.n_estimators < 1:
        raise ValueError("--n-estimators must be >= 1")
    if args.max_datasets is not None and args.max_datasets < 1:
        raise ValueError("--max-datasets must be >= 1")
    if args.ttt_epochs < 1:
        raise ValueError("--ttt-epochs must be >= 1")
    if args.ttt_lr <= 0:
        raise ValueError("--ttt-lr must be > 0")
    if not 0 < args.ttt_query_ratio < 1:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if not 0 < args.ttt_c_reserve_ratio < 1:
        raise ValueError("--ttt-c-reserve-ratio must be in (0, 1)")
    if args.ttt_query_ratio + args.ttt_c_reserve_ratio >= 1:
        raise ValueError(
            "--ttt-query-ratio + --ttt-c-reserve-ratio must be < 1"
        )
    if not 0 < args.ttt_validation_fraction < 1:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if args.ttt_n_estimators_finetune < 1:
        raise ValueError("--ttt-n-estimators-finetune must be >= 1")
    if args.ttt_validation_n_estimators < 1:
        raise ValueError("--ttt-validation-n-estimators must be >= 1")
    if args.ttt_patience < 1:
        raise ValueError("--ttt-patience must be >= 1")
    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise ValueError("--run-id may contain only letters, digits, '.', '_' and '-'")


def validate_launch_mode(args: argparse.Namespace) -> None:
    if args.matrix:
        if args.model is not None or args.method is not None:
            raise ValueError("--matrix cannot be combined with --model or --method")
        if args.out_dir is not None:
            raise ValueError("--out-dir is not supported in matrix mode; use --output-root")
        if args.model_path is not None:
            raise ValueError("--model-path is ambiguous in a multi-model matrix")
        for flag, values in (
            ("--matrix-models", args.matrix_models),
            ("--matrix-methods", args.matrix_methods),
        ):
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{flag} must not contain duplicate values")
        return
    if args.model is None or args.method is None:
        raise ValueError("single-run mode requires both --model and --method")
    if args.matrix_resume or args.matrix_fail_fast:
        raise ValueError("--matrix-resume/--matrix-fail-fast require --matrix")
    if args.matrix_models is not None or args.matrix_methods is not None:
        raise ValueError("--matrix-models/--matrix-methods require --matrix")


def resolve_out_dir(args: argparse.Namespace, repo_root: Path, run_id: str) -> Path:
    if args.out_dir:
        return resolve_path(args.out_dir, repo_root)
    output_root = resolve_path(args.output_root, repo_root)
    return output_root / run_id / args.model / args.method / f"seed{args.random_state}"


def resolve_tabicl_checkpoint(args: argparse.Namespace, repo_root: Path) -> Path:
    checkpoint = args.model_path or TABICL_CHECKPOINTS[args.model]
    return resolve_path(checkpoint, repo_root)


def add_common_runner_args(
    command: list[str],
    *,
    args: argparse.Namespace,
    data_root: Path,
    out_dir: Path,
    n_estimators: int,
    gpu_flag: str,
    gpu_value: str,
) -> None:
    command.extend(
        [
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--workers",
            str(args.workers),
            gpu_flag,
            gpu_value,
            "--n-estimators",
            str(n_estimators),
            "--random-state",
            str(args.random_state),
        ]
    )
    if args.max_datasets is not None:
        command.extend(["--max-datasets", str(args.max_datasets)])
    if args.verbose:
        command.append("--verbose")


def add_adaptation_args(
    command: list[str],
    *,
    args: argparse.Namespace,
    is_tabicl: bool,
    selection: str,
) -> tuple[str | None, str]:
    metric = args.ttt_c_metric
    if metric == "model_native_l2":
        metric = "tabicl_encoded_l2" if is_tabicl else "raw_l2"

    eval_metric = args.ttt_eval_metric
    if not is_tabicl and eval_metric == "accuracy":
        eval_metric = "acc"

    command.extend(
        [
            "--ttt-epochs",
            str(args.ttt_epochs),
            "--ttt-lr",
            str(args.ttt_lr),
            "--ttt-query-ratio",
            str(args.ttt_query_ratio),
            "--ttt-eval-metric",
            eval_metric,
            "--ttt-validation-fraction",
            str(args.ttt_validation_fraction),
            "--ttt-n-estimators-finetune",
            str(args.ttt_n_estimators_finetune),
            "--ttt-validation-n-estimators",
            str(args.ttt_validation_n_estimators),
            "--ttt-patience",
            str(args.ttt_patience),
            "--ttt-c-selection",
            selection,
            "--ttt-c-metric",
            metric,
            "--ttt-c-reserve-ratio",
            str(args.ttt_c_reserve_ratio),
        ]
    )

    if is_tabicl:
        source = "none" if selection == "random" else "test"
        command.extend(["--ttt-c-source", source, "--ttt-holdout"])
    else:
        source = None
        command.append("--ttt")
    return source, metric


def build_launch_spec(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    passthrough_args: Sequence[str] = (),
) -> LaunchSpec:
    repo_root = (repo_root or repo_root_from_script()).resolve()
    device_ids = parse_device_ids(args.devices)
    validate_args(args, device_ids)

    run_id = args.run_id or default_run_id()
    data_root = resolve_path(args.data_root, repo_root)
    out_dir = resolve_out_dir(args, repo_root, run_id)
    is_tabicl = args.model in TABICL_MODELS
    n_estimators = args.n_estimators or (32 if is_tabicl else 8)
    selection = None if args.method == "infer" else (
        "random" if args.method == "ft" else "f_mmd"
    )
    source: str | None = None
    metric: str | None = None
    checkpoint: Path | None = None

    if is_tabicl:
        runner_relative = (
            TABICL_INFER_RUNNER if args.method == "infer" else TABICL_ADAPT_RUNNER
        )
        runner = resolve_path(runner_relative, repo_root)
        checkpoint = resolve_tabicl_checkpoint(args, repo_root)
        command = [sys.executable, "-u", str(runner)]
        add_common_runner_args(
            command,
            args=args,
            data_root=data_root,
            out_dir=out_dir,
            n_estimators=n_estimators,
            gpu_flag="--gpus" if args.method == "infer" else "--gpu-groups",
            gpu_value=(
                ",".join(device_ids)
                if args.method == "infer"
                else ";".join(device_ids)
            ),
        )
        command.extend(
            [
                "--model-path",
                str(checkpoint),
                "--checkpoint-version",
                TABICL_CHECKPOINTS[args.model],
            ]
        )
        if selection is not None:
            source, metric = add_adaptation_args(
                command,
                args=args,
                is_tabicl=True,
                selection=selection,
            )
        model_family = "tabicl"
    else:
        runner_relative = (
            TABPFN_INFER_RUNNER if args.method == "infer" else TABPFN_ADAPT_RUNNER
        )
        runner = resolve_path(runner_relative, repo_root)
        command = [sys.executable, "-u", str(runner)]
        add_common_runner_args(
            command,
            args=args,
            data_root=data_root,
            out_dir=out_dir,
            n_estimators=n_estimators,
            gpu_flag="--gpus",
            gpu_value=",".join(device_ids),
        )
        version = "v2" if args.model == "tabpfn-v2" else "v3"
        command.extend(["--model-version", version])
        if args.model_path:
            checkpoint = resolve_path(args.model_path, repo_root)
            command.extend(["--model-path", str(checkpoint)])
        elif args.model == "tabpfn-v2":
            checkpoint = resolve_path(TABPFN_V2_CHECKPOINT, repo_root)
            command.extend(["--model-path", str(checkpoint)])
        else:
            binary_path = resolve_path(TABPFN_V3_BINARY_CHECKPOINT, repo_root)
            multiclass_path = resolve_path(TABPFN_V3_MULTICLASS_CHECKPOINT, repo_root)
            command.extend(
                [
                    "--v3-binary-model-path",
                    str(binary_path),
                    "--v3-multiclass-model-path",
                    str(multiclass_path),
                ]
            )
        if selection is not None:
            source, metric = add_adaptation_args(
                command,
                args=args,
                is_tabicl=False,
                selection=selection,
            )
        model_family = "tabpfn"

    command.extend(passthrough_args)
    return LaunchSpec(
        model=args.model,
        method=args.method,
        model_family=model_family,
        selection=selection,
        source=source,
        metric=metric,
        checkpoint=None if checkpoint is None else str(checkpoint),
        runner=str(runner),
        data_root=str(data_root),
        out_dir=str(out_dir),
        run_id=run_id,
        workers=args.workers,
        devices=",".join(device_ids),
        n_estimators=n_estimators,
        random_state=args.random_state,
        command=command,
        passthrough_args=list(passthrough_args),
    )


def validate_required_files(spec: LaunchSpec, repo_root: Path) -> None:
    required = [Path(spec.runner), Path(spec.data_root)]
    if spec.checkpoint is not None:
        required.append(Path(spec.checkpoint))
    if spec.model == "tabpfn-v3" and spec.checkpoint is None:
        required.extend(
            [
                resolve_path(TABPFN_V3_BINARY_CHECKPOINT, repo_root),
                resolve_path(TABPFN_V3_MULTICLASS_CHECKPOINT, repo_root),
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required paths do not exist: " + ", ".join(missing))


def manifest_payload(
    spec: LaunchSpec,
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    elapsed_seconds: float | None = None,
    exit_code: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = asdict(spec)
    payload.update(
        {
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "exit_code": exit_code,
            "command_shell": shlex.join(spec.command),
            "result_csv": str(Path(spec.out_dir) / "all_classification_results.csv"),
            "summary_txt": str(Path(spec.out_dir) / "summary.txt"),
        }
    )
    return payload


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def execute_launch_spec(spec: LaunchSpec, repo_root: Path) -> int:
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manager_manifest.json"
    started_at = datetime.now().astimezone().isoformat()
    write_manifest(
        manifest_path,
        manifest_payload(spec, status="running", started_at=started_at),
    )

    started = time.perf_counter()
    try:
        completed = subprocess.run(spec.command, cwd=repo_root, check=False)
        exit_code = int(completed.returncode)
    except KeyboardInterrupt:
        exit_code = 130
    elapsed = time.perf_counter() - started
    status = "ok" if exit_code == 0 else "fail"
    write_manifest(
        manifest_path,
        manifest_payload(
            spec,
            status=status,
            started_at=started_at,
            finished_at=datetime.now().astimezone().isoformat(),
            elapsed_seconds=elapsed,
            exit_code=exit_code,
        ),
    )
    print(f"manager_manifest: {manifest_path}", flush=True)
    print(f"launcher_status: {status}", flush=True)
    print(f"launcher_exit_code: {exit_code}", flush=True)
    return exit_code


def build_matrix_specs(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    passthrough_args: Sequence[str] = (),
) -> list[LaunchSpec]:
    run_id = args.run_id or default_run_id()
    models = list(dict.fromkeys(args.matrix_models or MODEL_CHOICES))
    methods = list(dict.fromkeys(args.matrix_methods or METHOD_CHOICES))
    specs: list[LaunchSpec] = []
    for model in models:
        for method in methods:
            cell_args = argparse.Namespace(**vars(args))
            cell_args.model = model
            cell_args.method = method
            cell_args.run_id = run_id
            cell_args.out_dir = None
            specs.append(
                build_launch_spec(
                    cell_args,
                    repo_root=repo_root,
                    passthrough_args=passthrough_args,
                )
            )
    return specs


def read_completed_manifest(spec: LaunchSpec) -> bool:
    path = Path(spec.out_dir) / "manager_manifest.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok" and payload.get("exit_code") == 0


def run_matrix(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    passthrough_args: Sequence[str],
) -> int:
    specs = build_matrix_specs(
        args, repo_root=repo_root, passthrough_args=passthrough_args
    )
    for spec in specs:
        validate_required_files(spec, repo_root)

    print(f"matrix_run_id: {specs[0].run_id}", flush=True)
    print(f"matrix_cells: {len(specs)}", flush=True)
    for index, spec in enumerate(specs, start=1):
        print(
            f"matrix_cell[{index:02d}]: {spec.model}/{spec.method} "
            f"command={shlex.join(spec.command)}",
            flush=True,
        )
    if args.dry_run:
        print("dry_run: true", flush=True)
        return 0

    matrix_root = resolve_path(args.output_root, repo_root) / specs[0].run_id
    matrix_root.mkdir(parents=True, exist_ok=True)
    matrix_manifest_path = matrix_root / "matrix_manifest.json"
    started_at = datetime.now().astimezone().isoformat()
    cells: list[dict[str, object]] = [
        {
            "model": spec.model,
            "method": spec.method,
            "out_dir": spec.out_dir,
            "status": "planned",
            "exit_code": None,
        }
        for spec in specs
    ]

    def save_matrix(status: str) -> None:
        write_manifest(
            matrix_manifest_path,
            {
                "run_id": specs[0].run_id,
                "data_root": specs[0].data_root,
                "status": status,
                "started_at": started_at,
                "updated_at": datetime.now().astimezone().isoformat(),
                "models": list(dict.fromkeys(spec.model for spec in specs)),
                "methods": list(dict.fromkeys(spec.method for spec in specs)),
                "cells": cells,
            },
        )

    save_matrix("running")
    first_failure = 0
    interrupted = False
    for index, spec in enumerate(specs):
        if args.matrix_resume and read_completed_manifest(spec):
            cells[index]["status"] = "skipped_existing_ok"
            cells[index]["exit_code"] = 0
            save_matrix("running")
            print(f"matrix_skip_existing_ok: {spec.model}/{spec.method}", flush=True)
            continue
        cells[index]["status"] = "running"
        save_matrix("running")
        print(f"matrix_start: {spec.model}/{spec.method}", flush=True)
        exit_code = execute_launch_spec(spec, repo_root)
        cells[index]["exit_code"] = exit_code
        cells[index]["status"] = "ok" if exit_code == 0 else "fail"
        save_matrix("running")
        if exit_code != 0 and first_failure == 0:
            first_failure = exit_code
        if exit_code == 130:
            interrupted = True
            break
        if exit_code != 0 and args.matrix_fail_fast:
            break

    final_status = "interrupted" if interrupted else ("fail" if first_failure else "ok")
    save_matrix(final_status)
    print(f"matrix_manifest: {matrix_manifest_path}", flush=True)
    print(f"matrix_status: {final_status}", flush=True)
    return 130 if interrupted else first_failure


def run(argv: Sequence[str] | None = None) -> int:
    launcher_argv, passthrough_args = split_launcher_and_passthrough(
        sys.argv[1:] if argv is None else argv
    )
    parser = build_arg_parser()
    args = parser.parse_args(launcher_argv)
    repo_root = repo_root_from_script()

    try:
        validate_launch_mode(args)
        if args.matrix:
            return run_matrix(
                args,
                repo_root=repo_root,
                passthrough_args=passthrough_args,
            )
        spec = build_launch_spec(
            args,
            repo_root=repo_root,
            passthrough_args=passthrough_args,
        )
        validate_required_files(spec, repo_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(asdict(spec), indent=2, ensure_ascii=False), flush=True)
    print(f"command: {shlex.join(spec.command)}", flush=True)
    if args.dry_run:
        print("dry_run: true", flush=True)
        return 0

    return execute_launch_spec(spec, repo_root)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
