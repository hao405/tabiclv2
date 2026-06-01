#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


DEFAULT_DATA172_ROOT = "data172"
DEFAULT_DATA200_ROOT = "data200"
DEFAULT_REFERENCE_RESULTS_ROOT = "baseline_compare/compare_results"
DEFAULT_SUBSET_ROOT = "Experiment_TTT_pipeline/generated/data200_minus_data172"
DEFAULT_RAW_OUT_ROOT = "baseline_compare/compare_results_data200_minus_data172"
DEFAULT_MERGED_OUT_ROOT = "baseline_compare/compare_results_data200_completed"

EXPECTED_DATA172_COUNT = 172
EXPECTED_DATA200_COUNT = 200
EXPECTED_COMPLEMENT_COUNT = 28

METRIC_COLUMNS = (
    "accuracy",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "log_loss",
    "fit_seconds",
    "predict_seconds",
)


@dataclass(frozen=True)
class RunTarget:
    strategy: str
    model_key: str
    reference_dir: str


INFERENCE_TARGETS: tuple[RunTarget, ...] = (
    RunTarget("inference", "tabicl", "tabiclv1.1_ensemble32"),
    RunTarget("inference", "tabpfnv2", "tabpfnv2_results_ensemble8"),
    RunTarget("inference", "tabpfnv25", "tabpfnv2.5_results_ensemble8"),
    RunTarget("inference", "tabpfnv3", "tabpfnv3_results_ensemble8"),
    RunTarget("inference", "limix", "limix-16m_results_178"),
    RunTarget("inference", "tabr", "tabr_results"),
    RunTarget("inference", "orion_msp", "orion_msp_results_ensemble32"),
)

TTT_TARGETS: tuple[RunTarget, ...] = (
    RunTarget("ttt", "tabicl", "tabiclv1.1_ttt_default"),
    RunTarget("ttt", "tabpfnv25", "tabpfnv2.5_1c_ttt_epoch30_chunk2000_lr1e-5_all_estimators8"),
    RunTarget("ttt", "tabpfnv3", "tabpfnv3_1c_ttt_epoch30_chunk2000_lr5e-6"),
    RunTarget("ttt", "limix", "limix_ttt_epoch30_chunk150"),
)


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or []


def dataset_names(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def select_targets(strategy: str, inference_models: list[str] | None, ttt_models: list[str] | None) -> list[RunTarget]:
    targets: list[RunTarget] = []
    if strategy in {"inference", "both"}:
        targets.extend(filter_targets(INFERENCE_TARGETS, inference_models, "inference"))
    if strategy in {"ttt", "both"}:
        targets.extend(filter_targets(TTT_TARGETS, ttt_models, "ttt"))
    return targets


def filter_targets(targets: Sequence[RunTarget], selected: list[str] | None, strategy: str) -> list[RunTarget]:
    if selected is None:
        return list(targets)
    known = {target.model_key for target in targets}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(
            f"Unknown {strategy} model(s) for auto-merge: {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}"
        )
    selected_set = set(selected)
    return [target for target in targets if target.model_key in selected_set]


def validate_inputs(
    *,
    data172_root: Path,
    data200_root: Path,
    reference_results_root: Path,
    targets: Sequence[RunTarget],
) -> tuple[list[str], list[str], list[str]]:
    data172_names = dataset_names(data172_root)
    data200_names = dataset_names(data200_root)
    data172_set = set(data172_names)
    data200_set = set(data200_names)
    complement_names = sorted(data200_set - data172_set)

    if len(data172_names) != EXPECTED_DATA172_COUNT:
        raise ValueError(f"Expected {EXPECTED_DATA172_COUNT} data172 datasets, got {len(data172_names)}")
    if len(data200_names) != EXPECTED_DATA200_COUNT:
        raise ValueError(f"Expected {EXPECTED_DATA200_COUNT} data200 datasets, got {len(data200_names)}")
    if not data172_set <= data200_set:
        missing = sorted(data172_set - data200_set)
        raise ValueError(f"data172 is not a subset of data200; missing in data200: {missing[:10]}")
    if len(complement_names) != EXPECTED_COMPLEMENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COMPLEMENT_COUNT} complement datasets, got {len(complement_names)}"
        )

    for target in targets:
        csv_path = reference_csv_path(reference_results_root, target)
        frame = read_results_csv(csv_path)
        names = set(frame["dataset_name"].astype(str))
        if names != data172_set:
            missing = sorted(data172_set - names)
            extra = sorted(names - data172_set)
            raise ValueError(
                f"Reference CSV does not exactly cover data172: {csv_path}; "
                f"missing={len(missing)} extra={len(extra)}"
            )

    return data172_names, data200_names, complement_names


def reference_csv_path(reference_results_root: Path, target: RunTarget) -> Path:
    return reference_results_root / target.reference_dir / "all_classification_results.csv"


def raw_csv_path(raw_out_root: Path, target: RunTarget) -> Path:
    return raw_out_root / target.strategy / target.model_key / "all_classification_results.csv"


def merged_dir_path(merged_out_root: Path, target: RunTarget) -> Path:
    return merged_out_root / target.strategy / target.reference_dir


def read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")
    frame = pd.read_csv(path)
    if "dataset_name" not in frame.columns:
        raise ValueError(f"CSV lacks required dataset_name column: {path}")
    if "status" not in frame.columns:
        raise ValueError(f"CSV lacks required status column: {path}")
    return frame


def prepare_subset_root(
    *,
    subset_root: Path,
    data200_root: Path,
    complement_names: Sequence[str],
    force: bool,
) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    expected = set(complement_names)

    for child in subset_root.iterdir():
        if child.name in {"_dataset_manifest.csv", "_dataset_manifest.json"}:
            child.unlink()
            continue
        if child.is_symlink():
            child.unlink()
            continue
        if force:
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
            continue
        raise RuntimeError(
            f"Refusing to remove non-symlink entry under generated subset root: {child}. "
            "Pass --force-subset to replace it."
        )

    for name in complement_names:
        source = (data200_root / name).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Complement dataset directory missing: {source}")
        os.symlink(source, subset_root / name, target_is_directory=True)

    write_subset_manifest(subset_root, complement_names)


def write_subset_manifest(subset_root: Path, complement_names: Sequence[str]) -> None:
    records = [{"index": idx, "dataset_name": name} for idx, name in enumerate(complement_names)]
    pd.DataFrame(records).to_csv(subset_root / "_dataset_manifest.csv", index=False)
    (subset_root / "_dataset_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_pipeline_stage(
    *,
    strategy: str,
    model_keys: Sequence[str],
    subset_root: Path,
    raw_out_root: Path,
    args: argparse.Namespace,
) -> None:
    if not model_keys:
        return

    command = [
        args.python_executable,
        "-m",
        "Experiment_TTT_pipeline",
        "--strategy",
        strategy,
        "--models",
        ",".join(model_keys),
        "--data-root",
        str(subset_root),
        "--out-root",
        str(raw_out_root),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--random-state",
        str(args.random_state),
    ]
    if args.gpu_groups:
        command.extend(["--gpu-groups", str(args.gpu_groups)])
    if args.max_datasets is not None:
        command.extend(["--max-datasets", str(args.max_datasets)])
    if args.verbose:
        command.append("--verbose")
    if args.dry_run:
        command.append("--dry-run")
    if args.stop_on_failure:
        command.append("--stop-on-failure")
    for value in args.extra_arg:
        command.extend(["--extra-arg", value])
    for value in args.model_extra_arg:
        command.extend(["--model-extra-arg", value])

    print("[data200-minus-data172] run:", shlex.join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def selected_update_names(complement_names: Sequence[str], max_datasets: int | None) -> set[str]:
    if max_datasets is None:
        return set(complement_names)
    return set(complement_names[: max(0, int(max_datasets))])


def merge_all_results(
    *,
    reference_results_root: Path,
    raw_out_root: Path,
    merged_out_root: Path,
    targets: Sequence[RunTarget],
    data200_names: Sequence[str],
    complement_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    expected_update_names = selected_update_names(complement_names, args.max_datasets)
    expected_final_names = set(data200_names) if args.max_datasets is None else set(data200_names) - (
        set(complement_names) - expected_update_names
    )
    order_map = {name: idx for idx, name in enumerate(data200_names)}

    for target in targets:
        reference_csv = reference_csv_path(reference_results_root, target)
        update_csv = raw_csv_path(raw_out_root, target)
        output_dir = merged_dir_path(merged_out_root, target)

        reference_df = read_results_csv(reference_csv)
        update_df = read_results_csv(update_csv)
        update_names = set(update_df["dataset_name"].astype(str))
        if update_names != expected_update_names:
            missing = sorted(expected_update_names - update_names)
            extra = sorted(update_names - expected_update_names)
            raise ValueError(
                f"Raw complement CSV has unexpected datasets: {update_csv}; "
                f"missing={len(missing)} extra={len(extra)}"
            )

        merged_df = merge_frames(reference_df, update_df, order_map)
        merged_names = set(merged_df["dataset_name"].astype(str))
        if merged_names != expected_final_names:
            missing = sorted(expected_final_names - merged_names)
            extra = sorted(merged_names - expected_final_names)
            raise ValueError(
                f"Merged CSV does not cover expected datasets for {target.model_key}; "
                f"missing={len(missing)} extra={len(extra)}"
            )
        if args.max_datasets is None and len(merged_df) != EXPECTED_DATA200_COUNT:
            raise ValueError(f"Expected merged full data200 CSV to have 200 rows, got {len(merged_df)}")

        output_dir.mkdir(parents=True, exist_ok=True)
        merged_csv = output_dir / "all_classification_results.csv"
        merged_df.to_csv(merged_csv, index=False)
        write_summary(output_dir / "summary.txt", merged_df, expected_final_names, target, args.max_datasets)
        print(f"[data200-minus-data172] merged: {merged_csv}", flush=True)


def merge_frames(reference_df: pd.DataFrame, update_df: pd.DataFrame, order_map: dict[str, int]) -> pd.DataFrame:
    update_names = set(update_df["dataset_name"].astype(str))
    base_keep = reference_df[~reference_df["dataset_name"].astype(str).isin(update_names)].copy()
    merged = pd.concat([base_keep, update_df.copy()], ignore_index=True)
    merged["_merge_order"] = merged["dataset_name"].astype(str).map(order_map)
    if merged["_merge_order"].isna().any():
        unknown = sorted(merged.loc[merged["_merge_order"].isna(), "dataset_name"].astype(str).unique())
        raise ValueError(f"Merged CSV contains datasets not present in data200: {unknown[:10]}")
    merged = merged.sort_values("_merge_order", kind="stable").drop(columns="_merge_order")
    return merged.reset_index(drop=True)


def write_summary(
    summary_path: Path,
    frame: pd.DataFrame,
    expected_names: set[str],
    target: RunTarget,
    max_datasets: int | None,
) -> None:
    result_df = frame.copy()
    status = result_df["status"].astype(str)
    ok_df = result_df[status == "ok"].copy()
    failed_df = result_df[status == "fail"].copy()
    skipped_df = result_df[status == "skip"].copy()

    lines = [
        f"strategy: {target.strategy}",
        f"model_key: {target.model_key}",
        f"reference_dir: {target.reference_dir}",
        f"merge_mode: {'partial_max_datasets_' + str(max_datasets) if max_datasets is not None else 'full_data200'}",
        f"discovered_datasets: {len(expected_names)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
    ]

    if "ttt_oom_fallback" in ok_df.columns:
        fallback_mask = ok_df["ttt_oom_fallback"].astype(str).str.lower().isin({"true", "1", "yes"})
        lines.append(f"ttt_oom_fallback_count: {int(fallback_mask.sum())}")

    for column in METRIC_COLUMNS:
        lines.append(mean_line(ok_df, f"avg_{column}_ok", column))

    lines.append(
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)")
    )
    lines.append(
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)")
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_line(frame: pd.DataFrame, label: str, column: str) -> str:
    if len(frame) and column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return f"{label}: {values.mean():.6f}"
    return f"{label}: (none)"


def run_rsync(args: argparse.Namespace) -> None:
    remote_root = args.jiqun_root.rstrip("/")
    host = args.jiqun_host
    mkdir_command = (
        f"mkdir -p {remote_path_for_shell(remote_root)} "
        f"{remote_path_for_shell(remote_root + '/baseline_compare')}"
    )
    subprocess.run(["ssh", host, mkdir_command], check=True)

    sync_specs = [
        (
            "Experiment_TTT_pipeline/",
            f"{host}:{remote_root}/Experiment_TTT_pipeline/",
            ["--exclude", "__pycache__/", "--exclude", "generated/", "--exclude", "results/", "--exclude", "logs/"],
        ),
        ("data172/", f"{host}:{remote_root}/data172/", []),
        ("data200/", f"{host}:{remote_root}/data200/", []),
        ("baseline_compare/compare_results/", f"{host}:{remote_root}/baseline_compare/compare_results/", []),
    ]
    for source, dest, extra in sync_specs:
        command = ["rsync", "-az", *extra, source, dest]
        print("[data200-minus-data172] sync:", shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def launch_on_jiqun(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"Experiment_TTT_pipeline/logs/data200_minus_data172_{timestamp}.log"
    remote_args = build_remote_args(args)
    remote_command = (
        f"cd {remote_path_for_shell(args.jiqun_root)} && "
        "mkdir -p Experiment_TTT_pipeline/logs && "
        f"{shlex.join([args.jiqun_python, 'Experiment_TTT_pipeline/run_data200_minus_data172_jiqun.py', *remote_args])} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )
    tmux_shell_command = shlex.join(
        [
            "tmux",
            "new-window",
            "-t",
            f"{args.jiqun_tmux_session}:",
            "-n",
            args.jiqun_window_name,
            f"bash -lc {shlex.quote(remote_command)}",
        ]
    )
    tmux_command = ["ssh", args.jiqun_host, tmux_shell_command]
    print("[data200-minus-data172] launch:", shlex.join(tmux_command), flush=True)
    subprocess.run(tmux_command, check=True)
    print(f"[data200-minus-data172] jiqun log: {args.jiqun_root}/{log_path}", flush=True)


def remote_path_for_shell(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)


def build_remote_args(args: argparse.Namespace) -> list[str]:
    remote_args = [
        "--strategy",
        args.strategy,
        "--data172-root",
        path_arg_for_remote(args.data172_root),
        "--data200-root",
        path_arg_for_remote(args.data200_root),
        "--reference-results-root",
        path_arg_for_remote(args.reference_results_root),
        "--subset-root",
        path_arg_for_remote(args.subset_root),
        "--raw-out-root",
        path_arg_for_remote(args.raw_out_root),
        "--merged-out-root",
        path_arg_for_remote(args.merged_out_root),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--random-state",
        str(args.random_state),
    ]
    if args.inference_models is not None:
        remote_args.extend(["--inference-models", args.inference_models])
    if args.ttt_models is not None:
        remote_args.extend(["--ttt-models", args.ttt_models])
    if args.gpu_groups:
        remote_args.extend(["--gpu-groups", args.gpu_groups])
    if args.max_datasets is not None:
        remote_args.extend(["--max-datasets", str(args.max_datasets)])
    if args.verbose:
        remote_args.append("--verbose")
    if args.dry_run:
        remote_args.append("--dry-run")
    if args.stop_on_failure:
        remote_args.append("--stop-on-failure")
    if args.no_merge:
        remote_args.append("--no-merge")
    if args.skip_run:
        remote_args.append("--skip-run")
    if args.merge_only:
        remote_args.append("--merge-only")
    if args.prepare_only:
        remote_args.append("--prepare-only")
    if args.force_subset:
        remote_args.append("--force-subset")
    for value in args.extra_arg:
        remote_args.extend(["--extra-arg", value])
    for value in args.model_extra_arg:
        remote_args.extend(["--model-extra-arg", value])
    return remote_args


def path_arg_for_remote(raw: str) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return raw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline inference + TTT on data200 minus data172, then merge "
            "the complement results into existing data172 compare_results CSVs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", choices=["inference", "ttt", "both"], default="both")
    parser.add_argument("--data172-root", default=DEFAULT_DATA172_ROOT)
    parser.add_argument("--data200-root", default=DEFAULT_DATA200_ROOT)
    parser.add_argument("--reference-results-root", default=DEFAULT_REFERENCE_RESULTS_ROOT)
    parser.add_argument("--subset-root", default=DEFAULT_SUBSET_ROOT)
    parser.add_argument("--raw-out-root", default=DEFAULT_RAW_OUT_ROOT)
    parser.add_argument("--merged-out-root", default=DEFAULT_MERGED_OUT_ROOT)
    parser.add_argument("--inference-models", default=None, help="Comma-separated inference model keys to run.")
    parser.add_argument("--ttt-models", default=None, help="Comma-separated TTT model keys to run.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--gpu-groups", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print backend commands without model runs.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--model-extra-arg", action="append", default=[])
    parser.add_argument("--skip-run", action="store_true", help="Skip pipeline execution and merge existing raw outputs.")
    parser.add_argument("--merge-only", action="store_true", help="Only validate and merge existing raw outputs.")
    parser.add_argument("--no-merge", action="store_true", help="Run complement jobs but do not merge outputs.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the complement symlink root and manifest.")
    parser.add_argument("--force-subset", action="store_true", help="Replace non-symlink entries under --subset-root.")

    parser.add_argument("--sync-to-jiqun", action="store_true", help="rsync code/data/reference results to jiqun.")
    parser.add_argument("--launch-jiqun", action="store_true", help="Launch this script in tmux zh on jiqun.")
    parser.add_argument("--jiqun-host", default="jiqun")
    parser.add_argument("--jiqun-root", default="~/zh/tabiclv2")
    parser.add_argument("--jiqun-tmux-session", default="zh")
    parser.add_argument("--jiqun-window-name", default="data200-comp")
    parser.add_argument("--jiqun-python", default="python")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.inference_models = args.inference_models
    args.ttt_models = args.ttt_models

    if args.sync_to_jiqun:
        run_rsync(args)
    if args.launch_jiqun:
        launch_on_jiqun(args)
        return

    inference_models = parse_csv_list(args.inference_models)
    ttt_models = parse_csv_list(args.ttt_models)
    targets = select_targets(args.strategy, inference_models, ttt_models)
    if not targets:
        raise ValueError("No targets selected")

    data172_root = repo_path(args.data172_root)
    data200_root = repo_path(args.data200_root)
    reference_results_root = repo_path(args.reference_results_root)
    subset_root = repo_path(args.subset_root)
    raw_out_root = repo_path(args.raw_out_root)
    merged_out_root = repo_path(args.merged_out_root)

    _, data200_names, complement_names = validate_inputs(
        data172_root=data172_root,
        data200_root=data200_root,
        reference_results_root=reference_results_root,
        targets=targets,
    )
    print(
        "[data200-minus-data172] validated: "
        f"data172={EXPECTED_DATA172_COUNT} data200={EXPECTED_DATA200_COUNT} complement={len(complement_names)}",
        flush=True,
    )

    if not args.merge_only:
        prepare_subset_root(
            subset_root=subset_root,
            data200_root=data200_root,
            complement_names=complement_names,
            force=args.force_subset,
        )
        print(f"[data200-minus-data172] subset_root: {subset_root}", flush=True)

    if args.prepare_only:
        return

    if not args.skip_run and not args.merge_only:
        inference_keys = [target.model_key for target in targets if target.strategy == "inference"]
        ttt_keys = [target.model_key for target in targets if target.strategy == "ttt"]
        run_pipeline_stage(
            strategy="inference",
            model_keys=inference_keys,
            subset_root=subset_root,
            raw_out_root=raw_out_root,
            args=args,
        )
        run_pipeline_stage(
            strategy="ttt",
            model_keys=ttt_keys,
            subset_root=subset_root,
            raw_out_root=raw_out_root,
            args=args,
        )

    if args.dry_run:
        print("[data200-minus-data172] dry-run: skipping merge because backend CSVs were not produced.")
        return
    if args.no_merge:
        print("[data200-minus-data172] no-merge: raw complement outputs left in place.")
        return

    merge_all_results(
        reference_results_root=reference_results_root,
        raw_out_root=raw_out_root,
        merged_out_root=merged_out_root,
        targets=targets,
        data200_names=data200_names,
        complement_names=complement_names,
        args=args,
    )


if __name__ == "__main__":
    main()
