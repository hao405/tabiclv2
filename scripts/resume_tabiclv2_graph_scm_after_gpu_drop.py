#!/usr/bin/env python3
"""Recover one interrupted matrix cell, then resume the original full matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


TRUTHY = {"1", "true", "yes"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def unique_rows(rows: Iterable[dict[str, str]], *, source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("dataset_name", "").strip()
        if not name:
            raise ValueError(f"empty dataset_name in {source}")
        if name in result:
            raise ValueError(f"duplicate dataset_name={name} in {source}")
        result[name] = row
    return result


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def prepare_recovery_root(
    *, data_root: Path, recovery_root: Path, missing_names: list[str]
) -> None:
    recovery_root.mkdir(parents=True, exist_ok=True)
    expected = set(missing_names)
    for child in recovery_root.iterdir():
        if child.name not in expected:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    for name in missing_names:
        source = (data_root / name).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"missing source dataset directory: {source}")
        target = recovery_root / name
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source, target_is_directory=True)


def write_summary(path: Path, rows: list[dict[str, str]], expected_count: int) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") == "fail"]
    skipped = [row for row in rows if row.get("status") == "skip"]
    oom = [
        row
        for row in ok
        if row.get("ttt_oom_fallback", "").strip().lower() in TRUTHY
    ]

    def mean_line(label: str, key: str) -> str:
        values: list[float] = []
        for row in ok:
            try:
                value = float(row.get(key, ""))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return f"{label}: {sum(values) / len(values):.6f}" if values else f"{label}: (none)"

    lines = [
        f"discovered_datasets: {expected_count}",
        f"processed_datasets: {len(rows)}",
        f"ok_count: {len(ok)}",
        f"failed_count: {len(failed)}",
        f"skipped_count: {len(skipped)}",
        f"ttt_oom_fallback_count: {len(oom)}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        "wall_seconds: (recovered; see recovery manager manifest)",
        "failed_datasets: " + (", ".join(row["dataset_name"] for row in failed) or "(none)"),
        "skipped_datasets: " + (", ".join(row["dataset_name"] for row in skipped) or "(none)"),
        "ttt_oom_fallback_datasets: "
        + (", ".join(row["dataset_name"] for row in oom) or "(none)"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument(
        "--data-root",
        default="results/tabiclv2_prior_data/graph_scm_3stage_512_seed42",
    )
    result.add_argument("--output-root", default="results/managed_experiments")
    result.add_argument("--devices", default="0")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--expected-count", type=int, default=512)
    result.add_argument("--prior-src-root", default="")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    matrix_root = output_root / args.run_id
    cell_dir = matrix_root / "tabicl-v2" / "ft" / f"seed{args.seed}"
    partial_csv = cell_dir / "worker_0.csv"
    original_manager = cell_dir / "manager_manifest.json"
    manifest_csv = data_root / "prior_manifest.csv"
    _, manifest_rows = read_csv(manifest_csv)
    manifest_by_name = unique_rows(manifest_rows, source=manifest_csv)
    ordered_names = [row["dataset_name"] for row in manifest_rows]
    if len(ordered_names) != args.expected_count:
        raise ValueError(
            f"manifest row count mismatch: expected={args.expected_count} actual={len(ordered_names)}"
        )
    partial_fields, partial_rows = read_csv(partial_csv)
    partial_by_name = unique_rows(partial_rows, source=partial_csv)
    unexpected = sorted(set(partial_by_name) - set(manifest_by_name))
    if unexpected:
        raise ValueError(f"partial CSV contains unexpected datasets: {unexpected[:5]}")
    completed_by_name = {
        name: row for name, row in partial_by_name.items() if row.get("status") == "ok"
    }
    recovery_csvs: list[Path] = []
    recovery_field_sets: list[list[str]] = []
    prior_recovery_ok_rows = 0
    prior_recovery_bad_rows = 0
    for recovery_root in sorted(output_root.glob(f"{args.run_id}_recovery*")):
        candidate = recovery_root / "tabicl-v2" / "ft" / f"seed{args.seed}" / "worker_0.csv"
        if not candidate.is_file():
            continue
        fields, rows = read_csv(candidate)
        rows_by_name = unique_rows(rows, source=candidate)
        unexpected = sorted(set(rows_by_name) - set(manifest_by_name))
        if unexpected:
            raise ValueError(f"{candidate} contains unexpected datasets: {unexpected[:5]}")
        recovery_csvs.append(candidate)
        recovery_field_sets.append(fields)
        for name, row in rows_by_name.items():
            if row.get("status") == "ok":
                completed_by_name[name] = row
                prior_recovery_ok_rows += 1
            else:
                prior_recovery_bad_rows += 1

    attempt_number = len(recovery_csvs) + 1
    recovery_data_root = (
        data_root.parent
        / f"{data_root.name}_resume_tabicl_v2_ft_attempt{attempt_number}"
    )
    recovery_out = (
        output_root
        / f"{args.run_id}_recovery_attempt_{attempt_number}"
        / "tabicl-v2"
        / "ft"
        / f"seed{args.seed}"
    )
    missing_names = [name for name in ordered_names if name not in completed_by_name]
    print(f"original_completed_rows: {len(partial_by_name)}", flush=True)
    print(f"prior_recovery_csvs: {len(recovery_csvs)}", flush=True)
    print(f"prior_recovery_ok_rows: {prior_recovery_ok_rows}", flush=True)
    print(f"prior_recovery_bad_rows_retried: {prior_recovery_bad_rows}", flush=True)
    print(f"recovery_missing_rows: {len(missing_names)}", flush=True)
    print(f"recovery_data_root: {recovery_data_root}", flush=True)
    print(f"recovery_out: {recovery_out}", flush=True)
    if not missing_names:
        raise ValueError("interrupted cell has no missing datasets; refusing unnecessary recovery")

    prepare_recovery_root(
        data_root=data_root,
        recovery_root=recovery_data_root,
        missing_names=missing_names,
    )
    if len([path for path in recovery_data_root.iterdir() if path.is_dir()]) != len(missing_names):
        raise RuntimeError("recovery root directory count mismatch")

    backup_csv = cell_dir / f"worker_0.before_gpu_drop_{len(partial_rows)}.csv"
    if not backup_csv.exists():
        shutil.copy2(partial_csv, backup_csv)
    print(f"partial_backup: {backup_csv}", flush=True)

    recovery_command = [
        sys.executable,
        "-u",
        "scripts/run_tfm_experiment.py",
        "--model",
        "tabicl-v2",
        "--method",
        "ft",
        "--data-root",
        str(recovery_data_root),
        "--out-dir",
        str(recovery_out),
        "--workers",
        "1",
        "--devices",
        args.devices,
        "--random-state",
        str(args.seed),
        "--ttt-epochs",
        "30",
        "--ttt-lr",
        "1e-5",
        "--ttt-patience",
        "8",
        "--ttt-c-metric",
        "standardized_l2",
    ]
    print("recovery_command: " + " ".join(recovery_command), flush=True)
    if args.dry_run:
        return 0

    recovery_exit = subprocess.run(recovery_command, cwd=repo_root, check=False).returncode
    if recovery_exit != 0:
        raise RuntimeError(f"recovery cell failed with exit code {recovery_exit}")

    recovered_csv = recovery_out / "all_classification_results.csv"
    recovered_fields, recovered_rows = read_csv(recovered_csv)
    recovered_by_name = unique_rows(recovered_rows, source=recovered_csv)
    if set(recovered_by_name) != set(missing_names):
        missing = sorted(set(missing_names) - set(recovered_by_name))
        extra = sorted(set(recovered_by_name) - set(missing_names))
        raise ValueError(
            f"recovery coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    non_ok_recovered = [
        row for row in recovered_rows if row.get("status") != "ok"
    ]
    if non_ok_recovered:
        names = [row.get("dataset_name", "") for row in non_ok_recovered]
        raise RuntimeError(
            f"recovery produced {len(non_ok_recovered)} non-ok rows; retry required: {names[:5]}"
        )
    merged_by_name = {**completed_by_name, **recovered_by_name}
    if set(merged_by_name) != set(ordered_names):
        raise RuntimeError("merged result does not exactly cover the 512-task manifest")
    merged_rows = [merged_by_name[name] for name in ordered_names]
    fieldnames = list(
        dict.fromkeys(
            [
                *partial_fields,
                *[field for fields in recovery_field_sets for field in fields],
                *recovered_fields,
            ]
        )
    )
    atomic_write_csv(cell_dir / "worker_0.csv", fieldnames, merged_rows)
    atomic_write_csv(cell_dir / "all_classification_results.csv", fieldnames, merged_rows)
    write_summary(cell_dir / "summary.txt", merged_rows, args.expected_count)

    manager_payload = json.loads(original_manager.read_text(encoding="utf-8"))
    manager_payload.update(
        {
            "status": "ok",
            "exit_code": 0,
            "finished_at": datetime.now().astimezone().isoformat(),
            "recovered_after_gpu_drop": True,
            "recovery_original_rows": len(partial_rows),
            "recovery_prior_ok_rows": len(completed_by_name) - len(partial_by_name),
            "recovery_added_rows": len(recovered_rows),
            "recovery_source_csvs": [str(path) for path in recovery_csvs],
            "recovery_result_csv": str(recovered_csv),
            "recovery_backup_csv": str(backup_csv),
        }
    )
    original_manager.write_text(
        json.dumps(manager_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"merged_rows: {len(merged_rows)}", flush=True)
    print(f"original_manager_status: ok ({original_manager})", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": args.run_id,
            "DATA_ROOT": str(data_root),
            "OUTPUT_ROOT": str(output_root),
            "DEVICES": args.devices,
            "GENERATION_WORKERS": "4",
        }
    )
    if args.prior_src_root:
        env["PRIOR_SRC_ROOT"] = args.prior_src_root
    resume_command = ["bash", "scripts/run_tabiclv2_graph_scm_matrix.sh", "full"]
    print("matrix_resume_command: " + " ".join(resume_command), flush=True)
    return subprocess.run(resume_command, cwd=repo_root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
