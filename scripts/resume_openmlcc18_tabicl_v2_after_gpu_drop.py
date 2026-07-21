#!/usr/bin/env python3
"""Recover interrupted TabICL v2 OpenML-CC18 cells and merge them safely."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


MODEL = "tabicl-v2"
METHODS = ("infer", "ft", "faware_ft")
MATRIX_MODELS = ("tabicl-v2", "tabpfn-v3")
MATRIX_METHODS = ("infer", "ft", "faware_ft")
TRUTHY = {"1", "true", "yes"}
EXPECTED_METRIC = "tabicl_encoded_l2"
EXPECTED_SELECTION = {
    "ft": "random",
    "faware_ft": "f_test_centroid_reserve",
}
SENTINEL_PREFIX = "__WORKER_EXIT__"
EXPECTED_INITIAL_INVENTORY = {"infer": 14, "ft": 15, "faware_ft": 14}


@dataclass(frozen=True)
class CellPlan:
    method: str
    cell_dir: Path
    fieldnames: list[str]
    retained_rows: dict[str, dict[str, str]]
    recovery_names: list[str]
    source_csv_hash: str


@dataclass(frozen=True)
class TaskArtifact:
    method: str
    dataset_name: str
    attempt_dir: Path
    result_csv: Path
    row: dict[str, str]
    reused: bool


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def real_rows(
    rows: Iterable[dict[str, str]], *, source: Path
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("dataset_name", "").strip()
        if name.startswith(SENTINEL_PREFIX):
            continue
        if not name:
            raise ValueError(f"empty dataset_name in {source}")
        if name in result:
            raise ValueError(f"duplicate dataset_name={name} in {source}")
        result[name] = row
    return result


def load_expected_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    included = payload.get("included")
    if not isinstance(included, list):
        raise ValueError("dataset view manifest has no included list")
    names = [str(row["dataset_name"]) for row in included]
    if len(names) != len(set(names)):
        raise ValueError("dataset view manifest has duplicate dataset names")
    if int(payload.get("included_count", -1)) != len(names):
        raise ValueError("dataset view manifest included_count mismatch")
    return names


def discover_expected_names(data_root: Path, *, max_classes: int = 10) -> list[str]:
    import numpy as np

    names: list[str] = []
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        labels_path = dataset_dir / "y_train.npy"
        if not labels_path.is_file():
            raise ValueError(f"missing y_train.npy: {dataset_dir}")
        labels = np.load(labels_path, mmap_mode="r")
        if labels.ndim != 1:
            raise ValueError(f"invalid y_train rank: {dataset_dir}")
        if len(np.unique(labels)) <= max_classes:
            names.append(dataset_dir.name)
    return names


def expected_names_from_view_or_source(
    *, view_manifest: Path, data_root: Path
) -> list[str]:
    if view_manifest.is_file():
        return load_expected_names(view_manifest)
    return discover_expected_names(data_root)


def build_cell_plan(
    *, matrix_root: Path, method: str, seed: int, expected_names: list[str]
) -> CellPlan:
    cell_dir = matrix_root / MODEL / method / f"seed{seed}"
    source_csv = cell_dir / "all_classification_results.csv"
    fieldnames, rows = read_csv(source_csv)
    by_name = real_rows(rows, source=source_csv)
    unexpected = sorted(set(by_name) - set(expected_names))
    if unexpected:
        raise ValueError(f"unexpected datasets in {source_csv}: {unexpected[:5]}")
    retained = {
        name: row
        for name, row in by_name.items()
        if row.get("status", "").strip() == "ok"
    }
    recovery_names = [name for name in expected_names if name not in retained]
    return CellPlan(
        method=method,
        cell_dir=cell_dir,
        fieldnames=fieldnames,
        retained_rows=retained,
        recovery_names=recovery_names,
        source_csv_hash=sha256(source_csv),
    )


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY


def validate_success_row(
    *,
    method: str,
    row: dict[str, str],
    source: Path,
    allow_declared_fallback: bool = False,
) -> None:
    name = row.get("dataset_name", "").strip()
    if row.get("status", "").strip() != "ok":
        raise ValueError(f"row is not successful for {method}/{name}: {source}")
    for column in ("accuracy", "balanced_accuracy"):
        try:
            value = float(row.get(column, ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric {column} for {method}/{name}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite {column} for {method}/{name}")
    if method == "infer":
        return
    expected_selection = EXPECTED_SELECTION[method]
    actual_selection = row.get("ttt_c_selection", "").strip()
    accepted_selections = {expected_selection}
    if method == "faware_ft":
        accepted_selections.add("f_mmd")
    if actual_selection not in accepted_selections:
        raise ValueError(f"unexpected selector for {method}/{name}")
    if row.get("ttt_c_metric", "").strip() != EXPECTED_METRIC:
        raise ValueError(f"unexpected native metric for {method}/{name}")
    applied = is_truthy(row.get("ttt_applied"))
    oom_fallback = is_truthy(row.get("ttt_oom_fallback"))
    fallback_reasons = [
        row.get(column, "").strip()
        for column in ("ttt_fallback_reason", "ttt_c_fallback_reason")
        if row.get(column, "").strip()
    ]
    if applied and not oom_fallback and not fallback_reasons:
        return
    if (
        allow_declared_fallback
        and not applied
        and oom_fallback
        and fallback_reasons
    ):
        return
    if not applied:
        raise ValueError(f"ttt_applied is not true for {method}/{name}")
    if oom_fallback:
        raise ValueError(f"OOM fallback is not allowed for {method}/{name}")
    raise ValueError(f"fallback is not allowed for {method}/{name}")


def validate_complete_tabicl_rows(
    *,
    method: str,
    rows: Sequence[dict[str, str]],
    expected_names: Sequence[str],
    source: Path,
    require_all_ok: bool = False,
) -> None:
    by_name = real_rows(rows, source=source)
    expected_set = set(expected_names)
    actual_set = set(by_name)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        raise ValueError(
            f"coverage mismatch in {source}: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    bad = [name for name, row in by_name.items() if row.get("status", "").strip() != "ok"]
    if require_all_ok and bad:
        raise ValueError(f"non-ok rows in {source}: {bad[:5]}")
    for name, row in by_name.items():
        if row.get("status", "").strip() == "ok":
            validate_success_row(
                method=method,
                row=row,
                source=source,
                allow_declared_fallback=True,
            )


def merge_rows(
    *,
    plan: CellPlan,
    recovery_rows: Sequence[dict[str, str]],
    expected_names: Sequence[str],
    recovery_source: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    sentinels = [
        row.get("dataset_name", "")
        for row in recovery_rows
        if row.get("dataset_name", "").startswith(SENTINEL_PREFIX)
    ]
    if sentinels:
        raise ValueError(f"recovery contains worker sentinels: {sentinels[:5]}")
    recovered = real_rows(recovery_rows, source=recovery_source)
    if set(recovered) != set(plan.recovery_names):
        missing = sorted(set(plan.recovery_names) - set(recovered))
        unexpected = sorted(set(recovered) - set(plan.recovery_names))
        raise ValueError(
            f"recovery coverage mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    merged = {**plan.retained_rows, **recovered}
    ordered = [merged[name] for name in expected_names]
    fields = list(
        dict.fromkeys(
            [*plan.fieldnames, *[key for row in recovery_rows for key in row.keys()]]
        )
    )
    validate_complete_tabicl_rows(
        method=plan.method,
        rows=ordered,
        expected_names=expected_names,
        source=recovery_source,
        require_all_ok=False,
    )
    return fields, ordered


def atomic_write_text(path: Path, value: str) -> None:
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def copy_stage_atomically(source: Path, destination: Path) -> None:
    atomic_write_bytes(destination, source.read_bytes())


def backup_once(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.before_targeted_recovery")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def result_summary(rows: Sequence[dict[str, str]]) -> str:
    metric_names = (
        ("avg_accuracy_ok", "accuracy"),
        ("avg_f1_ok", "f1"),
        ("avg_balanced_accuracy_ok", "balanced_accuracy"),
        ("avg_roc_auc_ok", "roc_auc"),
        ("avg_log_loss_ok", "log_loss"),
    )
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") == "fail"]
    skipped_rows = [row for row in rows if row.get("status") == "skip"]
    oom_rows = [
        row
        for row in failed_rows
        if "OutOfMemoryError" in row.get("error", "")
        or is_truthy(row.get("ttt_oom_fallback"))
    ]
    lines = [
        f"discovered_datasets: {len(rows)}",
        f"processed_datasets: {len(rows)}",
        f"ok_count: {len(ok_rows)}",
        f"failed_count: {len(failed_rows)}",
        f"skipped_count: {len(skipped_rows)}",
        f"ttt_oom_fallback_count: {len(oom_rows)}",
    ]
    for label, column in metric_names:
        values: list[float] = []
        for row in ok_rows:
            try:
                values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                pass
        lines.append(
            f"{label}: {sum(values) / len(values):.6f}" if values else f"{label}: (none)"
        )
    lines.extend(
        [
            "wall_seconds: (targeted recovery; see recovery manifest)",
            "failed_datasets: "
            + (", ".join(row["dataset_name"] for row in failed_rows) or "(none)"),
            "skipped_datasets: "
            + (", ".join(row["dataset_name"] for row in skipped_rows) or "(none)"),
            "ttt_oom_fallback_datasets: "
            + (", ".join(row["dataset_name"] for row in oom_rows) or "(none)"),
        ]
    )
    return "\n".join(lines) + "\n"


def next_task_attempt_dir(
    recovery_root: Path, method: str, seed: int, dataset_name: str
) -> Path:
    method_root = recovery_root / MODEL / method / f"seed{seed}" / dataset_name
    existing = [
        int(path.name.removeprefix("attempt_"))
        for path in method_root.glob("attempt_*")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    return method_root / f"attempt_{max(existing, default=0) + 1}"


def build_recovery_command(
    *,
    repo_root: Path,
    method: str,
    names: Sequence[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "run_tfm_experiment.py"),
        "--model",
        MODEL,
        "--method",
        method,
        "--data-root",
        str(args.data_root),
        "--dataset-max-classes",
        "10",
        "--dataset-names",
        *names,
        "--out-dir",
        str(out_dir),
        "--workers",
        "1",
        "--devices",
        args.devices,
        "--random-state",
        str(args.seed),
        "--n-estimators",
        "32",
        "--ttt-epochs",
        "30",
        "--ttt-lr",
        "1e-5",
        "--ttt-patience",
        "8",
        "--ttt-c-metric",
        "model_native_l2",
    ]
    original_manager = (
        Path(args.matrix_root) / MODEL / method / f"seed{args.seed}" / "manager_manifest.json"
    )
    original_payload = json.loads(original_manager.read_text(encoding="utf-8"))
    checkpoint = original_payload.get("checkpoint")
    if checkpoint:
        command.extend(["--model-path", str(checkpoint)])
    return command


def read_task_artifact(
    *,
    result_csv: Path,
    method: str,
    dataset_name: str,
    require_ok: bool,
    allow_declared_fallback: bool = False,
) -> dict[str, str]:
    _, rows = read_csv(result_csv)
    sentinels = [
        row.get("dataset_name", "")
        for row in rows
        if row.get("dataset_name", "").startswith(SENTINEL_PREFIX)
    ]
    if sentinels:
        raise ValueError(f"single-task artifact contains worker sentinel: {sentinels}")
    by_name = real_rows(rows, source=result_csv)
    if set(by_name) != {dataset_name}:
        raise ValueError(
            f"single-task artifact mismatch in {result_csv}: "
            f"expected={dataset_name} actual={sorted(by_name)}"
        )
    row = by_name[dataset_name]
    status = row.get("status", "").strip()
    if status == "ok":
        validate_success_row(
            method=method,
            row=row,
            source=result_csv,
            allow_declared_fallback=allow_declared_fallback,
        )
    elif require_ok:
        raise ValueError(f"single-task artifact is not successful: {result_csv}")
    elif status != "fail":
        raise ValueError(f"unexpected task status={status!r}: {result_csv}")
    return row


def find_reusable_success(
    *, recovery_root: Path, method: str, seed: int, dataset_name: str
) -> TaskArtifact | None:
    task_root = recovery_root / MODEL / method / f"seed{seed}" / dataset_name
    attempts = sorted(
        task_root.glob("attempt_*"),
        key=lambda path: int(path.name.removeprefix("attempt_"))
        if path.name.removeprefix("attempt_").isdigit()
        else -1,
        reverse=True,
    )
    for attempt_dir in attempts:
        result_csv = attempt_dir / "validated_task_result.csv"
        if not result_csv.is_file():
            result_csv = attempt_dir / "all_classification_results.csv"
        if not result_csv.is_file():
            continue
        try:
            row = read_task_artifact(
                result_csv=result_csv,
                method=method,
                dataset_name=dataset_name,
                require_ok=True,
            )
        except (OSError, ValueError):
            continue
        return TaskArtifact(
            method=method,
            dataset_name=dataset_name,
            attempt_dir=attempt_dir,
            result_csv=result_csv,
            row=row,
            reused=True,
        )
    return None


def find_latest_task_artifact(
    *, recovery_root: Path, method: str, seed: int, dataset_name: str
) -> TaskArtifact | None:
    task_root = recovery_root / MODEL / method / f"seed{seed}" / dataset_name
    attempts = sorted(
        task_root.glob("attempt_*"),
        key=lambda path: int(path.name.removeprefix("attempt_"))
        if path.name.removeprefix("attempt_").isdigit()
        else -1,
        reverse=True,
    )
    for attempt_dir in attempts:
        for filename in ("all_classification_results.csv", "validated_task_result.csv"):
            result_csv = attempt_dir / filename
            if not result_csv.is_file():
                continue
            try:
                row = read_task_artifact(
                    result_csv=result_csv,
                    method=method,
                    dataset_name=dataset_name,
                    require_ok=False,
                    allow_declared_fallback=True,
                )
            except (OSError, ValueError):
                continue
            return TaskArtifact(
                method=method,
                dataset_name=dataset_name,
                attempt_dir=attempt_dir,
                result_csv=result_csv,
                row=row,
                reused=True,
            )
    return None


def failed_task_row(
    *, plan: CellPlan, dataset_name: str, error: str
) -> dict[str, str]:
    row = {field: "" for field in plan.fieldnames}
    row.update(
        {
            "dataset_name": dataset_name,
            "dataset_dir": "",
            "status": "fail",
            "error": error,
        }
    )
    return row


def dataset_storage_bytes(data_root: Path, dataset_name: str) -> int:
    dataset_dir = data_root / dataset_name
    return sum(path.stat().st_size for path in dataset_dir.iterdir() if path.is_file())


def recovery_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def build_task_schedule(
    *, plans: Sequence[CellPlan], data_root: Path
) -> tuple[str | None, list[tuple[str, str]]]:
    requested = {
        plan.method: list(plan.recovery_names)
        for plan in plans
    }
    common = set.intersection(
        *(set(names) for names in requested.values())
    ) if requested else set()
    preflight = (
        min(common, key=lambda name: (dataset_storage_bytes(data_root, name), name))
        if common
        else None
    )
    schedule: list[tuple[str, str]] = []
    if preflight is not None:
        schedule.extend(
            (method, preflight)
            for method in METHODS
            if preflight in requested.get(method, [])
        )
    for method in METHODS:
        for name in requested.get(method, []):
            pair = (method, name)
            if pair not in schedule:
                schedule.append(pair)
    return preflight, schedule


def validate_run_contract(
    *,
    matrix_root: Path,
    run_id: str,
    data_root: Path,
    view_manifest: Path,
    seed: int,
) -> dict[str, object]:
    matrix_path = matrix_root / "matrix_manifest.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id:
        raise ValueError("matrix manifest run_id mismatch")
    if payload.get("models") != list(MATRIX_MODELS):
        raise ValueError(f"unexpected matrix models: {payload.get('models')}")
    if payload.get("methods") != list(MATRIX_METHODS):
        raise ValueError(f"unexpected matrix methods: {payload.get('methods')}")
    if int(payload.get("dataset_max_classes", -1)) != 10:
        raise ValueError("matrix dataset_max_classes is not 10")
    source_root = Path(str(payload.get("source_data_root", ""))).resolve()
    if source_root != data_root.resolve():
        raise ValueError(f"matrix source data root mismatch: {source_root} != {data_root}")
    recorded_view = Path(str(payload.get("dataset_view_manifest", ""))).resolve()
    if recorded_view != view_manifest.resolve():
        raise ValueError(f"matrix view manifest mismatch: {recorded_view} != {view_manifest}")
    for method in METHODS:
        manager_path = matrix_root / MODEL / method / f"seed{seed}" / "manager_manifest.json"
        manager = json.loads(manager_path.read_text(encoding="utf-8"))
        expected = {
            "model": MODEL,
            "method": method,
            "random_state": seed,
            "n_estimators": 32,
        }
        for key, value in expected.items():
            if manager.get(key) != value:
                raise ValueError(
                    f"manager contract mismatch for {method}: {key}={manager.get(key)!r}"
                )
        if not manager.get("checkpoint"):
            raise ValueError(f"manager checkpoint missing for {method}")
        if method != "infer":
            if manager.get("selection") != EXPECTED_SELECTION[method]:
                raise ValueError(f"manager selector mismatch for {method}")
            if manager.get("metric") != EXPECTED_METRIC:
                raise ValueError(f"manager metric mismatch for {method}")
            command = [str(value) for value in manager.get("command", [])]
            expected_options = {
                "--ttt-epochs": "30",
                "--ttt-lr": "1e-05",
                "--ttt-patience": "8",
            }
            for option, expected_value in expected_options.items():
                try:
                    actual_value = command[command.index(option) + 1]
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"manager command missing {option} for {method}") from exc
                if actual_value != expected_value:
                    raise ValueError(
                        f"manager command mismatch for {method}: "
                        f"{option}={actual_value!r}"
                    )
    return payload


def audit_gpu(*, devices: str, minimum_free_ratio: float) -> dict[str, object]:
    import torch

    hostname = socket.gethostname().split(".")[0]
    if hostname != "gpu2":
        raise RuntimeError(f"recovery must run on node gpu2, found {hostname}")
    if devices.strip() != "2":
        raise RuntimeError(f"recovery requires physical GPU 2, found devices={devices!r}")
    if torch.cuda.device_count() <= 2:
        raise RuntimeError(f"physical GPU 2 is not visible; count={torch.cuda.device_count()}")
    free, total = torch.cuda.mem_get_info(2)
    ratio = free / total
    if ratio < minimum_free_ratio:
        raise RuntimeError(
            f"physical GPU 2 is not sufficiently free: ratio={ratio:.6f} "
            f"required={minimum_free_ratio:.6f}"
        )
    return {
        "hostname": hostname,
        "physical_gpu": 2,
        "name": torch.cuda.get_device_name(2),
        "free_bytes": free,
        "total_bytes": total,
        "free_ratio": ratio,
    }


def merge_cell(
    *,
    plan: CellPlan,
    recovery_csv: Path,
    expected_names: list[str],
    attempt_dir: Path,
) -> dict[str, object]:
    recovery_fields, recovery_rows = read_csv(recovery_csv)
    fields, rows = merge_rows(
        plan=plan,
        recovery_rows=recovery_rows,
        expected_names=expected_names,
        recovery_source=recovery_csv,
    )
    fields = list(dict.fromkeys([*fields, *recovery_fields]))
    current_csv = plan.cell_dir / "all_classification_results.csv"
    current_hash = sha256(current_csv)
    if current_hash != plan.source_csv_hash:
        raise RuntimeError(
            f"source CSV changed during recovery: expected={plan.source_csv_hash} "
            f"actual={current_hash} path={current_csv}"
        )
    backups = {
        name: backup_once(plan.cell_dir / name)
        for name in (
            "all_classification_results.csv",
            "worker_0.csv",
            "summary.txt",
            "manager_manifest.json",
        )
    }
    manager_path = plan.cell_dir / "manager_manifest.json"
    manager = json.loads(manager_path.read_text(encoding="utf-8"))
    failed_names = [
        row["dataset_name"] for row in rows if row.get("status") != "ok"
    ]
    cell_ok = not failed_names
    manager.update(
        {
            "status": "ok" if cell_ok else "fail",
            "exit_code": 0 if cell_ok else 1,
            "finished_at": now(),
            "recovered_after_gpu_drop": True,
            "recovery_retained_rows": len(plan.retained_rows),
            "recovery_added_rows": len(plan.recovery_names),
            "recovery_result_csv": str(recovery_csv),
            "recovery_attempt_dir": str(attempt_dir),
            "recovery_source_csv_sha256": plan.source_csv_hash,
            "recovery_failed_datasets": failed_names,
        }
    )
    transaction_dir = attempt_dir / "merge_transaction"
    transaction_dir.mkdir(parents=True, exist_ok=False)
    staged = {
        "worker_0.csv": transaction_dir / "worker_0.csv",
        "all_classification_results.csv": transaction_dir / "all_classification_results.csv",
        "summary.txt": transaction_dir / "summary.txt",
        "manager_manifest.json": transaction_dir / "manager_manifest.json",
    }
    atomic_write_csv(staged["worker_0.csv"], fields, rows)
    atomic_write_csv(staged["all_classification_results.csv"], fields, rows)
    atomic_write_text(staged["summary.txt"], result_summary(rows))
    atomic_write_json(staged["manager_manifest.json"], manager)
    journal = {
        "status": "prepared",
        "method": plan.method,
        "cell_dir": str(plan.cell_dir),
        "completed": [],
        "artifacts": {
            name: {
                "stage": str(stage),
                "destination": str(plan.cell_dir / name),
                "sha256": sha256(stage),
            }
            for name, stage in staged.items()
        },
    }
    atomic_write_json(transaction_dir / "journal.json", journal)
    apply_merge_transaction(transaction_dir)
    return {
        "method": plan.method,
        "status": "merged",
        "retained_count": len(plan.retained_rows),
        "recovered_count": len(plan.recovery_names),
        "recovered_names": plan.recovery_names,
        "ok_count": len(rows) - len(failed_names),
        "fail_count": len(failed_names),
        "failed_names": failed_names,
        "recovery_csv": str(recovery_csv),
        "backups": {key: None if value is None else str(value) for key, value in backups.items()},
        "transaction_dir": str(transaction_dir),
    }


def apply_merge_transaction(transaction_dir: Path) -> None:
    journal_path = transaction_dir / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("status") == "committed":
        return
    artifacts = journal.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"invalid merge journal: {journal_path}")
    journal["status"] = "applying"
    atomic_write_json(journal_path, journal)
    completed = set(journal.get("completed", []))
    for name, record in artifacts.items():
        stage = Path(record["stage"])
        destination = Path(record["destination"])
        if sha256(stage) != record["sha256"]:
            raise ValueError(f"staged artifact hash mismatch: {stage}")
        copy_stage_atomically(stage, destination)
        completed.add(name)
        journal["completed"] = sorted(completed)
        atomic_write_json(journal_path, journal)
    journal["status"] = "committed"
    journal["committed_at"] = now()
    atomic_write_json(journal_path, journal)


def repair_pending_transactions(recovery_root: Path) -> list[str]:
    repaired: list[str] = []
    for journal_path in sorted(recovery_root.glob("**/merge_transaction/journal.json")):
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        if payload.get("status") != "committed":
            apply_merge_transaction(journal_path.parent)
            repaired.append(str(journal_path.parent))
    return repaired


def audit_matrix(
    *, matrix_root: Path, expected_names: list[str], seed: int
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for model in MATRIX_MODELS:
        for method in MATRIX_METHODS:
            path = matrix_root / model / method / f"seed{seed}" / "all_classification_results.csv"
            _, rows = read_csv(path)
            by_name = real_rows(rows, source=path)
            if set(by_name) != set(expected_names):
                raise ValueError(f"incomplete matrix cell: {model}/{method}")
            if model == MODEL:
                validate_complete_tabicl_rows(
                    method=method,
                    rows=list(by_name.values()),
                    expected_names=expected_names,
                    source=path,
                    require_all_ok=False,
                )
            ok_count = sum(row.get("status") == "ok" for row in by_name.values())
            fail_count = len(by_name) - ok_count
            cells.append(
                {
                    "model": model,
                    "method": method,
                    "rows": len(by_name),
                    "ok": ok_count,
                    "fail": fail_count,
                    "status": "ok" if fail_count == 0 else "fail",
                    "sha256": sha256(path),
                }
            )
    return {
        "status": "ok" if all(cell["status"] == "ok" for cell in cells) else "fail",
        "expected_datasets": len(expected_names),
        "cells": cells,
    }


def update_matrix_manifest(matrix_root: Path, audit: dict[str, object]) -> None:
    path = matrix_root / "matrix_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit_by_cell = {
        (cell["model"], cell["method"]): cell
        for cell in audit["cells"]
    }
    for cell in payload.get("cells", []):
        record = audit_by_cell[(cell.get("model"), cell.get("method"))]
        cell["status"] = record["status"]
        cell["exit_code"] = 0 if record["status"] == "ok" else 1
        if cell.get("model") == MODEL:
            cell["recovered_after_gpu_drop"] = True
    payload.update(
        {
            "status": audit["status"],
            "updated_at": now(),
            "recovered_after_gpu_drop": True,
            "recovery_audit": audit,
        }
    )
    atomic_write_json(path, payload)


def write_recovery_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    atomic_write_json(path, payload)


def ensure_summary_manifest(
    *, view_manifest: Path, recovery_root: Path, data_root: Path, expected_names: list[str]
) -> Path:
    if view_manifest.is_file():
        return view_manifest
    reconstructed = recovery_root / "reconstructed_dataset_view_manifest.json"
    atomic_write_json(
        reconstructed,
        {
            "source_root": str(data_root),
            "effective_root": None,
            "max_classes": 10,
            "included_count": len(expected_names),
            "excluded_count": None,
            "included": [{"dataset_name": name} for name in expected_names],
            "reconstructed_for_summary": True,
            "created_at": now(),
        },
    )
    return reconstructed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="openmlcc18_2x3_full_20260715_223032"
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/managed_experiments"))
    parser.add_argument("--data-root", type=Path, default=Path("openml_cc18"))
    parser.add_argument(
        "--view-manifest",
        type=Path,
        default=Path("results/dataset_views/openml_cc18_max10/dataset_view_manifest.json"),
    )
    parser.add_argument("--devices", default="2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gpu-min-free-ratio", type=float, default=0.90)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge the latest isolated task artifacts without launching model processes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    args.data_root = (repo_root / args.data_root).resolve()
    view_manifest = (repo_root / args.view_manifest).resolve()
    matrix_root = output_root / args.run_id
    args.matrix_root = matrix_root
    recovery_root = output_root / f"{args.run_id}_targeted_recovery"
    manifest_path = recovery_root / "recovery_manifest.json"
    validate_run_contract(
        matrix_root=matrix_root,
        run_id=args.run_id,
        data_root=args.data_root,
        view_manifest=view_manifest,
        seed=args.seed,
    )
    expected_names = expected_names_from_view_or_source(
        view_manifest=view_manifest,
        data_root=args.data_root,
    )
    if len(expected_names) != 67:
        raise ValueError(f"expected 67 OpenML tasks, found {len(expected_names)}")

    lock_handle = None
    repaired_transactions: list[str] = []
    gpu_audit: dict[str, object] | None = None
    if not args.dry_run:
        recovery_root.mkdir(parents=True, exist_ok=True)
        lock_handle = (recovery_root / ".recovery.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another targeted recovery process is already running") from exc
        repaired_transactions = repair_pending_transactions(recovery_root)
        if not args.merge_only:
            gpu_audit = audit_gpu(
                devices=args.devices,
                minimum_free_ratio=args.gpu_min_free_ratio,
            )

    plans = [
        build_cell_plan(
            matrix_root=matrix_root,
            method=method,
            seed=args.seed,
            expected_names=expected_names,
        )
        for method in args.methods
    ]
    plan_by_method = {plan.method: plan for plan in plans}
    inventory = {plan.method: len(plan.recovery_names) for plan in plans}
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    needs_initial_inventory = not previous_manifest.get("initial_inventory")
    if (
        needs_initial_inventory
        and set(args.methods) == set(METHODS)
        and inventory != EXPECTED_INITIAL_INVENTORY
    ):
        raise ValueError(
            f"initial recovery inventory mismatch: expected={EXPECTED_INITIAL_INVENTORY} "
            f"actual={inventory}"
        )
    preflight_dataset, schedule = build_task_schedule(
        plans=plans,
        data_root=args.data_root,
    )
    execution_schedule = [] if args.merge_only else schedule
    for plan in plans:
        print(
            f"recovery_plan: method={plan.method} retained={len(plan.retained_rows)} "
            f"retry={len(plan.recovery_names)} names={','.join(plan.recovery_names) or '(none)'}"
        )
    print(f"preflight_dataset: {preflight_dataset or '(none)'}")
    print(f"isolated_process_count: {len(execution_schedule)}")
    print(f"merge_only: {args.merge_only}")
    for method, dataset_name in execution_schedule:
        attempt_dir = next_task_attempt_dir(
            recovery_root, method, args.seed, dataset_name
        )
        command = build_recovery_command(
            repo_root=repo_root,
            method=method,
            names=[dataset_name],
            out_dir=attempt_dir,
            args=args,
        )
        print(f"recovery_command[{method}/{dataset_name}]: {shlex.join(command)}")
    if args.dry_run:
        print("dry_run: true")
        return 0

    prior_history: list[dict[str, object]] = []
    if previous_manifest:
        prior_history = list(previous_manifest.get("attempt_history", []))
        prior_history.append(
            {
                key: previous_manifest.get(key)
                for key in ("status", "started_at", "finished_at", "updated_at", "cells")
            }
        )
    payload: dict[str, object] = {
        "status": "running",
        "run_id": args.run_id,
        "matrix_root": str(matrix_root),
        "view_manifest": str(view_manifest),
        "expected_names": expected_names,
        "devices": args.devices,
        "allocator": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "gpu_audit": gpu_audit,
        "repaired_transactions": repaired_transactions,
        "initial_inventory": (
            EXPECTED_INITIAL_INVENTORY
            if needs_initial_inventory and set(args.methods) == set(METHODS)
            else previous_manifest.get("initial_inventory", inventory)
        ),
        "current_inventory": inventory,
        "preflight_dataset": preflight_dataset,
        "preflight_status": previous_manifest.get("preflight_status") if args.merge_only else None,
        "isolated_process_count": (
            previous_manifest.get("isolated_process_count", len(schedule))
            if args.merge_only
            else len(execution_schedule)
        ),
        "merge_only_process_count": len(execution_schedule),
        "merge_only": args.merge_only,
        "started_at": now(),
        "cells": [],
        "tasks": list(previous_manifest.get("tasks", [])) if args.merge_only else [],
        "attempt_history": prior_history,
    }
    write_recovery_manifest(manifest_path, payload)
    artifacts: dict[tuple[str, str], TaskArtifact] = {}
    if args.merge_only:
        for plan in plans:
            for dataset_name in plan.recovery_names:
                artifact = find_latest_task_artifact(
                    recovery_root=recovery_root,
                    method=plan.method,
                    seed=args.seed,
                    dataset_name=dataset_name,
                )
                if artifact is None:
                    raise FileNotFoundError(
                        f"no reusable task artifact for {plan.method}/{dataset_name}"
                    )
                artifacts[(plan.method, dataset_name)] = artifact
    preflight_count = sum(
        dataset_name == preflight_dataset for _, dataset_name in execution_schedule
    )
    for index, (method, dataset_name) in enumerate(execution_schedule):
        plan = plan_by_method[method]
        reusable = find_reusable_success(
            recovery_root=recovery_root,
            method=method,
            seed=args.seed,
            dataset_name=dataset_name,
        )
        if reusable is not None:
            artifacts[(method, dataset_name)] = reusable
            task_record: dict[str, object] = {
                "method": method,
                "dataset_name": dataset_name,
                "status": "reused_ok",
                "result_status": "ok",
                "attempt_dir": str(reusable.attempt_dir),
                "result_csv": str(reusable.result_csv),
                "finished_at": now(),
            }
        else:
            attempt_dir = next_task_attempt_dir(
                recovery_root, method, args.seed, dataset_name
            )
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = build_recovery_command(
                repo_root=repo_root,
                method=method,
                names=[dataset_name],
                out_dir=attempt_dir,
                args=args,
            )
            task_record = {
                "method": method,
                "dataset_name": dataset_name,
                "status": "running",
                "attempt_dir": str(attempt_dir),
                "command": command,
                "started_at": now(),
            }
            payload["tasks"].append(task_record)
            write_recovery_manifest(manifest_path, payload)
            started = time.perf_counter()
            env = recovery_environment()
            try:
                exit_code = subprocess.run(
                    command, cwd=repo_root, env=env, check=False
                ).returncode
            except KeyboardInterrupt:
                task_record.update(
                    {
                        "status": "interrupted",
                        "exit_code": 130,
                        "elapsed_seconds": time.perf_counter() - started,
                        "finished_at": now(),
                    }
                )
                payload["status"] = "interrupted"
                payload["finished_at"] = now()
                write_recovery_manifest(manifest_path, payload)
                return 130
            raw_csv = attempt_dir / "all_classification_results.csv"
            validation_error = ""
            try:
                row = read_task_artifact(
                    result_csv=raw_csv,
                    method=method,
                    dataset_name=dataset_name,
                    require_ok=False,
                )
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                row = failed_task_row(
                    plan=plan,
                    dataset_name=dataset_name,
                    error=(
                        f"recovery process exit={exit_code}; artifact validation failed: "
                        f"{validation_error}"
                    ),
                )
            validated_csv = attempt_dir / "validated_task_result.csv"
            validated_fields = list(dict.fromkeys([*plan.fieldnames, *row.keys()]))
            atomic_write_csv(validated_csv, validated_fields, [row])
            artifact = TaskArtifact(
                method=method,
                dataset_name=dataset_name,
                attempt_dir=attempt_dir,
                result_csv=validated_csv,
                row=row,
                reused=False,
            )
            artifacts[(method, dataset_name)] = artifact
            task_record.update(
                {
                    "status": "validated",
                    "result_status": row.get("status"),
                    "exit_code": exit_code,
                    "elapsed_seconds": time.perf_counter() - started,
                    "raw_result_csv": str(raw_csv),
                    "result_csv": str(validated_csv),
                    "validation_error": validation_error or None,
                    "finished_at": now(),
                }
            )
        if reusable is not None:
            payload["tasks"].append(task_record)
        write_recovery_manifest(manifest_path, payload)

        if preflight_count and index + 1 == preflight_count:
            preflight_records = [
                artifacts[(method_name, preflight_dataset)].row
                for method_name in METHODS
                if (method_name, preflight_dataset) in artifacts
            ]
            if len(preflight_records) != preflight_count or any(
                row.get("status") != "ok" for row in preflight_records
            ):
                payload["status"] = "preflight_failed"
                payload["finished_at"] = now()
                write_recovery_manifest(manifest_path, payload)
                return 2
            payload["preflight_status"] = "ok"
            write_recovery_manifest(manifest_path, payload)

    for plan in plans:
        if not plan.recovery_names:
            source_csv = plan.cell_dir / "all_classification_results.csv"
            _, rows = read_csv(source_csv)
            validate_complete_tabicl_rows(
                method=plan.method,
                rows=rows,
                expected_names=expected_names,
                source=source_csv,
                require_all_ok=False,
            )
            payload["cells"].append({"method": plan.method, "status": "already_complete"})
            continue
        recovery_rows = [
            artifacts[(plan.method, name)].row for name in plan.recovery_names
        ]
        merge_dir = next_task_attempt_dir(
            recovery_root, plan.method, args.seed, "__merge__"
        )
        merge_dir.mkdir(parents=True, exist_ok=False)
        aggregate_csv = merge_dir / "recovery_aggregate.csv"
        fields = list(
            dict.fromkeys(
                [*plan.fieldnames, *[key for row in recovery_rows for key in row]]
            )
        )
        atomic_write_csv(aggregate_csv, fields, recovery_rows)
        merged = merge_cell(
            plan=plan,
            recovery_csv=aggregate_csv,
            expected_names=expected_names,
            attempt_dir=merge_dir,
        )
        payload["cells"].append(merged)
        write_recovery_manifest(manifest_path, payload)

    audit = audit_matrix(matrix_root=matrix_root, expected_names=expected_names, seed=args.seed)
    update_matrix_manifest(matrix_root, audit)
    summary_manifest = ensure_summary_manifest(
        view_manifest=view_manifest,
        recovery_root=recovery_root,
        data_root=args.data_root,
        expected_names=expected_names,
    )
    summary_command = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "summarize_openmlcc18_tfm_matrix.py"),
        "--matrix-root",
        str(matrix_root),
        "--view-manifest",
        str(summary_manifest),
        "--random-state",
        str(args.seed),
        "--expected-datasets",
        str(len(expected_names)),
    ]
    print(f"summary_command: {shlex.join(summary_command)}")
    summary_exit = subprocess.run(summary_command, cwd=repo_root, check=False).returncode
    payload.update(
        {
            "status": (
                "ok"
                if summary_exit == 0 and audit["status"] == "ok"
                else "complete_with_failures"
                if summary_exit == 0
                else "summary_failed"
            ),
            "finished_at": now(),
            "matrix_audit": audit,
            "summary_command": summary_command,
            "summary_view_manifest": str(summary_manifest),
            "summary_exit_code": summary_exit,
        }
    )
    write_recovery_manifest(manifest_path, payload)
    print(f"recovery_manifest: {manifest_path}")
    print(f"recovery_status: {payload['status']}")
    return summary_exit


if __name__ == "__main__":
    raise SystemExit(main())
