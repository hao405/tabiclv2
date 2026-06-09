#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Experiment_TTT_pipeline.pipeline import (
        PipelineConfig,
        expand_models,
        parse_model_extra_args,
        resolve_repo_path,
        run_pipeline,
        write_pipeline_manifest,
    )
    from Experiment_TTT_pipeline.registry import PACKAGE_ROOT, REPO_ROOT
else:
    from .pipeline import (
        PipelineConfig,
        expand_models,
        parse_model_extra_args,
        resolve_repo_path,
        run_pipeline,
        write_pipeline_manifest,
    )
    from .registry import PACKAGE_ROOT, REPO_ROOT


BENCHMARK_ROOT_ARGS = {
    "openml_cc18": "openml_cc18_root",
    "tabzilla": "tabzilla_root",
}
DEFAULT_BENCHMARKS = ("tabzilla",)
DEFAULT_EXCLUDED_MODELS = {"tabpfnv2"}
DEFAULT_TTT_EXCLUDED_MODELS = DEFAULT_EXCLUDED_MODELS | {"orion_msp"}
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
REQUIRED_DATASET_FILES = (
    "N_train.npy",
    "C_train.npy",
    "y_train.npy",
    "N_val.npy",
    "C_val.npy",
    "y_val.npy",
    "N_test.npy",
    "C_test.npy",
    "y_test.npy",
    "info.json",
)
SYNC_FILES = (
    "Experiment_TTT_pipeline/__init__.py",
    "Experiment_TTT_pipeline/__main__.py",
    "Experiment_TTT_pipeline/pipeline.py",
    "Experiment_TTT_pipeline/registry.py",
    "Experiment_TTT_pipeline/run_pipeline.py",
    "Experiment_TTT_pipeline/run_tazilla_openmlcc18_benchmark.py",
    "benchmark.py",
    "1C_Chunk_TTT.py",
    "src/tabicl/_sklearn/preprocessing.py",
    "baseline_compare/tfm_benchmark_common.py",
    "baseline_compare/result_naming.py",
    "baseline_compare/TabPFN-main/benchmark_infer.py",
    "baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py",
    "baseline_compare/LimiX/benchmark_infer.py",
    "baseline_compare/LimiX/limix_ttt.py",
    "baseline_compare/tabular-dl-tabr/benchmark_infer.py",
    "baseline_compare/Orion-MSP/benchmark_infer.py",
    "baseline_compare/Orion-MSP/orion_msp_ttt.py",
    "baseline_compare/Orion-MSP/src/orion_msp/sklearn/preprocessing.py",
)
REMOTE_NOISE_PREFIXES = (
    "bash: warning: setlocale:",
    "/bin/sh: warning: setlocale:",
    "ln: failed to create symbolic link '/home/crc00006699/.ssh/known_hosts':",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected Experiment_TTT_pipeline models on TabZilla by default "
            "(OpenML-CC18 is still available explicitly), with a jiqun tmux launch path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        action="append",
        default=[],
        help="Comma-separated benchmark names: openml_cc18, tabzilla.",
    )
    parser.add_argument("--openml-cc18-root", default="openml_cc18")
    parser.add_argument("--tabzilla-root", default="tabzilla")
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        help=(
            "Comma-separated model keys or aliases. Defaults skip tabpfnv2 and "
            "skip Orion-MSP only for TTT; explicit --models values are not filtered."
        ),
    )
    parser.add_argument("--strategy", choices=["inference", "ttt", "both"], default="both")
    parser.add_argument(
        "--out-root",
        default=str(PACKAGE_ROOT / "results_tabzilla"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--gpu-groups", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--backend-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for each backend model command. Omit for no timeout.",
    )
    parser.add_argument(
        "--backend-kill-grace-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait after SIGTERM before SIGKILL when cleaning backend processes.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--fail-on-unsupported",
        dest="skip_unsupported",
        action="store_false",
        help="Fail instead of recording skip rows for unsupported strategies.",
    )
    parser.set_defaults(skip_unsupported=True)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to every selected backend script.",
    )
    parser.add_argument(
        "--model-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to one model script, formatted as MODEL:ARG.",
    )

    parser.add_argument("--sync-to-jiqun", action="store_true")
    parser.add_argument("--launch-jiqun", action="store_true")
    parser.add_argument("--jiqun-host", default="jiqun")
    parser.add_argument("--jiqun-repo", default="~/zh/tabiclv2")
    parser.add_argument("--jiqun-target-pane", default="zh:0")
    parser.add_argument("--jiqun-python", default="python")
    parser.add_argument("--jiqun-log-dir", default="Experiment_TTT_pipeline/jiqun_logs")
    parser.add_argument(
        "--force-jiqun-launch",
        action="store_true",
        help="Launch even if the tmux pane looks busy.",
    )
    return parser


def parse_csv_values(values: Sequence[str], *, default: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    raw_values = values or default
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def selected_benchmarks(args: argparse.Namespace) -> list[str]:
    benchmarks = parse_csv_values(args.benchmarks, default=DEFAULT_BENCHMARKS)
    unknown = [name for name in benchmarks if name not in BENCHMARK_ROOT_ARGS]
    if unknown:
        known = ", ".join(BENCHMARK_ROOT_ARGS)
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}. Known: {known}")
    return benchmarks


def filter_default_models(model_keys: Sequence[str], strategy: str) -> list[str]:
    excluded = DEFAULT_TTT_EXCLUDED_MODELS if strategy == "ttt" else DEFAULT_EXCLUDED_MODELS
    return [key for key in model_keys if key not in excluded]


def selected_model_stage_specs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.models:
        strategy = "inference" if args.strategy == "both" else args.strategy
        return [(args.strategy, expand_models(args.models, strategy))]

    if args.strategy == "both":
        return [
            ("inference", filter_default_models(expand_models((), "inference"), "inference")),
            ("ttt", filter_default_models(expand_models((), "ttt"), "ttt")),
        ]
    return [
        (
            args.strategy,
            filter_default_models(expand_models((), args.strategy), args.strategy),
        )
    ]


def benchmark_root(args: argparse.Namespace, benchmark: str) -> Path:
    arg_name = BENCHMARK_ROOT_ARGS[benchmark]
    return resolve_repo_path(getattr(args, arg_name))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc


def validate_dataset_dir(dataset_dir: Path) -> str:
    missing = [name for name in REQUIRED_DATASET_FILES if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_dir} is missing required files: {', '.join(missing)}"
        )
    info = read_json(dataset_dir / "info.json")
    task_type = str(info.get("task_type", "")).lower()
    if task_type not in CLASSIFICATION_TASKS:
        raise ValueError(f"{dataset_dir} has unsupported task_type={task_type!r}")
    return task_type


def validate_benchmark_root(root: Path, benchmark: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"{benchmark} root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{benchmark} root is not a directory: {root}")
    dataset_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {root}")

    for dataset_dir in dataset_dirs:
        validate_dataset_dir(dataset_dir)
    print(f"[runner] validated {benchmark}: {root} ({len(dataset_dirs)} datasets)")
    return dataset_dirs


def benchmark_out_root(args: argparse.Namespace, benchmark: str) -> Path:
    return resolve_repo_path(args.out_root) / benchmark


def write_combined_manifest(
    out_root: Path,
    records: Sequence[dict[str, object]],
    *,
    stem: str = "openml_cc18_tabzilla_runs",
) -> None:
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / f"{stem}.json"
    csv_path = manifest_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "benchmark",
        "data_root",
        "datasets_discovered",
        "model_key",
        "display_name",
        "strategy",
        "status",
        "returncode",
        "command",
        "out_dir",
        "elapsed_seconds",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[runner] saved_combined_manifest_json: {json_path}")
    print(f"[runner] saved_combined_manifest_csv: {csv_path}")


def saw_keyboard_interrupt(records: Sequence[dict[str, object]]) -> bool:
    return any(str(record.get("error") or "").startswith("KeyboardInterrupt") for record in records)


def run_local(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.backend_timeout_seconds is not None and args.backend_timeout_seconds <= 0:
        raise ValueError("--backend-timeout-seconds must be positive when provided")
    if args.backend_kill_grace_seconds < 0:
        raise ValueError("--backend-kill-grace-seconds must be non-negative")

    benchmarks = selected_benchmarks(args)
    stage_specs = selected_model_stage_specs(args)
    model_extra_args = parse_model_extra_args(args.model_extra_arg)
    combined_records: list[dict[str, object]] = []

    for benchmark in benchmarks:
        root = benchmark_root(args, benchmark)
        dataset_dirs = validate_benchmark_root(root, benchmark)
        benchmark_results = []
        stop_benchmark = False
        for stage_strategy, model_keys in stage_specs:
            config = PipelineConfig(
                strategy=stage_strategy,
                data_root=root,
                out_root=benchmark_out_root(args, benchmark),
                workers=args.workers,
                gpus=args.gpus,
                gpu_groups=args.gpu_groups,
                max_datasets=args.max_datasets,
                random_state=args.random_state,
                python_executable=args.python_executable,
                verbose=args.verbose,
                dry_run=args.dry_run,
                stop_on_failure=args.stop_on_failure,
                skip_unsupported=args.skip_unsupported,
                backend_timeout_seconds=args.backend_timeout_seconds,
                backend_kill_grace_seconds=args.backend_kill_grace_seconds,
                extra_args=tuple(args.extra_arg),
                model_extra_args=model_extra_args,
            )
            print(
                f"[runner] running {benchmark}: strategy={stage_strategy} "
                f"models={','.join(model_keys)} out_root={benchmark_out_root(args, benchmark)}"
            )
            results = run_pipeline(model_keys, config)
            benchmark_results.extend(results)
            for result in results:
                record = asdict(result)
                record.update(
                    {
                        "benchmark": benchmark,
                        "data_root": str(root),
                        "datasets_discovered": len(dataset_dirs),
                    }
                )
                combined_records.append(record)

            if saw_keyboard_interrupt(combined_records):
                stop_benchmark = True
                break
            if args.stop_on_failure and any(result.status == "fail" for result in results):
                stop_benchmark = True
                break

        if not args.dry_run:
            write_pipeline_manifest(
                benchmark_out_root(args, benchmark) / "manifests",
                benchmark_results,
                stem="pipeline_runs_all",
            )
        if stop_benchmark:
            break

    if not args.dry_run:
        write_combined_manifest(resolve_repo_path(args.out_root), combined_records)
    return combined_records


def run_command(command: Sequence[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    print("[runner] " + shlex.join(command), flush=True)
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        input=input_text,
        text=True,
        check=True,
    )


def existing_sync_files() -> list[str]:
    files = [path for path in SYNC_FILES if (REPO_ROOT / path).exists()]
    missing = [path for path in SYNC_FILES if not (REPO_ROOT / path).exists()]
    if missing:
        print("[runner] sync_skip_missing: " + ", ".join(missing), file=sys.stderr)
    return files


def sync_to_jiqun(args: argparse.Namespace) -> None:
    files = existing_sync_files()
    if not files:
        raise FileNotFoundError("No local files selected for jiqun sync")
    run_command(["ssh", args.jiqun_host, f"mkdir -p {remote_path_expr(args.jiqun_repo)}"])
    files_input = "\n".join(files) + "\n"
    run_command(
        [
            "rsync",
            "-av",
            "--files-from=-",
            ".",
            f"{args.jiqun_host}:{args.jiqun_repo}/",
        ],
        input_text=files_input,
    )


def remote_relative_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return value


def remote_path_expr(value: str) -> str:
    if value == "~":
        return "$HOME"
    if value.startswith("~/"):
        return "$HOME/" + shlex.quote(value[2:])
    return shlex.quote(value)


def append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_remote_runner_command(args: argparse.Namespace) -> str:
    command = [
        args.jiqun_python,
        "Experiment_TTT_pipeline/run_tazilla_openmlcc18_benchmark.py",
        "--strategy",
        args.strategy,
        "--benchmarks",
        ",".join(selected_benchmarks(args)),
        "--openml-cc18-root",
        remote_relative_path(args.openml_cc18_root),
        "--tabzilla-root",
        remote_relative_path(args.tabzilla_root),
        "--out-root",
        remote_relative_path(args.out_root),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--random-state",
        str(args.random_state),
    ]
    append_optional_arg(command, "--gpu-groups", args.gpu_groups)
    append_optional_arg(command, "--max-datasets", args.max_datasets)
    append_optional_arg(command, "--backend-timeout-seconds", args.backend_timeout_seconds)
    append_optional_arg(command, "--backend-kill-grace-seconds", args.backend_kill_grace_seconds)
    for value in args.models:
        command.append(f"--models={value}")
    for value in args.extra_arg:
        command.append(f"--extra-arg={value}")
    for value in args.model_extra_arg:
        command.append(f"--model-extra-arg={value}")
    if args.verbose:
        command.append("--verbose")
    if args.dry_run:
        command.append("--dry-run")
    if args.stop_on_failure:
        command.append("--stop-on-failure")
    if not args.skip_unsupported:
        command.append("--fail-on-unsupported")
    return shlex.join(command)


def run_ssh_capture(args: argparse.Namespace, remote_command: str) -> str:
    completed = subprocess.run(
        ["ssh", args.jiqun_host, remote_command],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if not any(line.startswith(prefix) for prefix in REMOTE_NOISE_PREFIXES)
    ]
    return "\n".join(lines).strip() + ("\n" if lines else "")


def remote_root_test(repo: str, root: str) -> str:
    if Path(root).is_absolute():
        return f"test -d {remote_path_expr(root)}"
    return f"test -d {remote_path_expr(repo.rstrip('/') + '/' + root)}"


def remote_selected_root_test(args: argparse.Namespace) -> str:
    tests = []
    for benchmark in selected_benchmarks(args):
        root_arg = BENCHMARK_ROOT_ARGS[benchmark]
        tests.append(remote_root_test(args.jiqun_repo, remote_relative_path(getattr(args, root_arg))))
    return " && ".join(tests)


def target_session(target_pane: str) -> str:
    return target_pane.split(":", 1)[0].split(".", 1)[0]


def pane_looks_busy(pane_command: str, capture: str) -> bool:
    normalized_command = pane_command.strip().lstrip("-")
    allowed_commands = {"bash", "zsh", "sh", "ssh"}
    if normalized_command and normalized_command not in allowed_commands:
        return True

    lowered_capture = capture.lower()
    busy_markers = (
        "nvitop",
        "press h for help or q to quit",
        "htop",
        "top - ",
        "python experiment_ttt_pipeline",
        "run_tazilla_openmlcc18_benchmark.py",
        "[worker ",
        "1c_chunk_ttt.py",
        "tabpfn_1c_ttt.py",
        "limix_ttt.py",
        "orion_msp_ttt.py",
    )
    return any(marker in lowered_capture for marker in busy_markers)


def check_jiqun_ready(args: argparse.Namespace) -> None:
    session = target_session(args.jiqun_target_pane)
    run_ssh_capture(args, f"tmux has-session -t {shlex.quote(session)}")
    run_ssh_capture(
        args,
        f"test -d {remote_path_expr(args.jiqun_repo)} && "
        + remote_selected_root_test(args),
    )
    pane_command = run_ssh_capture(
        args,
        "tmux display-message -p "
        f"-t {shlex.quote(args.jiqun_target_pane)} "
        "'#{pane_current_command}'",
    ).strip()
    capture = run_ssh_capture(
        args,
        f"tmux capture-pane -p -t {shlex.quote(args.jiqun_target_pane)} | tail -40",
    )
    if pane_looks_busy(pane_command, capture) and not args.force_jiqun_launch:
        remote_command = build_remote_runner_command(args)
        print(
            f"[runner] refusing to launch: tmux pane {args.jiqun_target_pane} "
            f"looks busy (current_command={pane_command!r}).",
            file=sys.stderr,
        )
        print("[runner] manual command after selecting an idle pane:", file=sys.stderr)
        print(f"cd {args.jiqun_repo} && {remote_command}", file=sys.stderr)
        raise SystemExit(2)


def launch_jiqun(args: argparse.Namespace) -> None:
    check_jiqun_ready(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    remote_command = build_remote_runner_command(args)
    log_path = f"{args.jiqun_log_dir}/openml_cc18_tabzilla_{timestamp}.log"
    pane_command = (
        f"cd {remote_path_expr(args.jiqun_repo)} && "
        f"mkdir -p {remote_path_expr(args.jiqun_log_dir)} && "
        f"{remote_command} 2>&1 | tee {shlex.quote(log_path)}"
    )
    run_command(
        [
            "ssh",
            args.jiqun_host,
            "tmux send-keys "
            f"-t {shlex.quote(args.jiqun_target_pane)} "
            f"{shlex.quote(pane_command)} C-m",
        ]
    )
    print(f"[runner] launched on {args.jiqun_host}:{args.jiqun_target_pane}")
    print(f"[runner] remote_log: {args.jiqun_repo}/{log_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.sync_to_jiqun:
        sync_to_jiqun(args)
    if args.launch_jiqun:
        launch_jiqun(args)
        return

    records = run_local(args)
    failed = [record for record in records if record.get("status") == "fail"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
