#!/usr/bin/env python3
"""Targeted recovery for missing/non-OOM OpenML-CC18 PEFT matrix rows."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


MODELS = ("tabiclv2", "tabpfnv3")
METHODS = ("lora", "last_layers", "ln_head_embedding")
EXPECTED_CELLS = tuple((model, method) for model in MODELS for method in METHODS)
SENTINEL_PREFIX = "__WORKER_"
TRUTHY = {"1", "true", "yes"}
OOM_MARKERS = (
    "outofmemoryerror",
    "cuda out of memory",
    "out of memory",
    "cudnn_status_alloc_failed",
    "hip error out of memory",
)
DEFAULT_RUN_NAME = "openmlcc18_peft_2x3_full_20260717_113939"
DEFAULT_RECOVERY_NAME = (
    "openmlcc18_peft_2x3_full_20260717_113939_nonoom_logicfix_20260720"
)
EXPECTED_NONOOM_TARGETS = {
    ("tabpfnv3", "last_layers", "OpenML-ID-40978"),
    ("tabpfnv3", "last_layers", "OpenML-ID-4134"),
}
DEFAULT_SKIPPED_DATASETS = ("OpenML-ID-40996",)
SIGTERM_MARKERS = ("exitcode=-15", "returncode=-15", "sigterm", "signal 15")


@dataclass(frozen=True)
class CellPlan:
    model: str
    method: str
    output_dir: Path
    result_csv: Path
    fieldnames: list[str]
    rows_by_name: dict[str, dict[str, str]]
    missing_names: list[str]
    retry_names: list[str]
    source_sha256: str


def now() -> str:
    return datetime.now().astimezone().isoformat()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY


def is_oom_row(row: dict[str, str]) -> bool:
    error = str(row.get("error", "")).lower()
    return truthy(row.get("ttt_oom_fallback")) or any(
        marker in error for marker in OOM_MARKERS
    )


def is_eligible_success(row: dict[str, str]) -> bool:
    if row.get("status", "").strip() != "ok":
        return False
    if not truthy(row.get("ttt_applied")) or is_oom_row(row):
        return False
    try:
        return float(row.get("peft_trainable_params", "")) > 0
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_text(fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def backup_once(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.before_targeted_recovery")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def load_expected_names(view_manifest: Path) -> list[str]:
    payload = json.loads(view_manifest.read_text(encoding="utf-8"))
    included = payload.get("included")
    if not isinstance(included, list):
        raise ValueError("dataset view manifest has no included list")
    names = [str(item["dataset_name"]) for item in included]
    if len(names) != 67 or len(names) != len(set(names)):
        raise ValueError(
            f"expected 67 unique OpenML datasets, got {len(names)}/{len(set(names))}"
        )
    if int(payload.get("included_count", -1)) != 67:
        raise ValueError("dataset view manifest included_count must be 67")
    return names


def resolve_output_dir(manifest_path: Path, value: object, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    return manifest_path.parent / path


def manifest_trial_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise ValueError("matrix manifest has no trials list")
    result = {
        (str(item.get("model_family")), str(item.get("peft_method"))): item
        for item in trials
    }
    if set(result) != set(EXPECTED_CELLS):
        raise ValueError(f"unexpected matrix cells: {sorted(result)}")
    return result


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


def build_plans(
    *,
    matrix_manifest: Path,
    expected_names: list[str],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], list[CellPlan]]:
    payload = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    trials = manifest_trial_index(payload)
    expected_set = set(expected_names)
    plans: list[CellPlan] = []
    for model, method in EXPECTED_CELLS:
        output_dir = resolve_output_dir(
            matrix_manifest, trials[(model, method)].get("output_dir"), repo_root
        )
        result_csv = output_dir / "all_classification_results.csv"
        fieldnames, rows = read_csv(result_csv)
        by_name = real_rows(rows, source=result_csv)
        unexpected = sorted(set(by_name) - expected_set)
        if unexpected:
            raise ValueError(f"unexpected datasets in {result_csv}: {unexpected}")
        missing = [name for name in expected_names if name not in by_name]
        failed_non_oom = [
            name
            for name in expected_names
            if name in by_name
            and not is_eligible_success(by_name[name])
            and not is_oom_row(by_name[name])
        ]
        plans.append(
            CellPlan(
                model=model,
                method=method,
                output_dir=output_dir,
                result_csv=result_csv,
                fieldnames=fieldnames,
                rows_by_name=by_name,
                missing_names=missing,
                retry_names=list(dict.fromkeys([*missing, *failed_non_oom])),
                source_sha256=sha256(result_csv),
            )
        )
    return payload, trials, plans


def exclude_datasets_from_plans(
    plans: Sequence[CellPlan],
    skipped_datasets: set[str],
) -> list[CellPlan]:
    return [
        replace(
            plan,
            missing_names=[
                name for name in plan.missing_names if name not in skipped_datasets
            ],
            retry_names=[
                name for name in plan.retry_names if name not in skipped_datasets
            ],
        )
        for plan in plans
    ]


def next_attempt_dir(recovery_root: Path, plan: CellPlan, dataset_name: str) -> Path:
    root = recovery_root / "attempts" / plan.model / plan.method / dataset_name
    existing = [
        int(path.name.removeprefix("attempt_"))
        for path in root.glob("attempt_*")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    return root / f"attempt_{max(existing, default=0) + 1}"


def make_single_task_view(
    *, recovery_root: Path, source_view: Path, dataset_name: str
) -> Path:
    destination = recovery_root / "views" / dataset_name
    target = (source_view / dataset_name).resolve(strict=True)
    if destination.exists():
        links = [path for path in destination.iterdir() if path.is_symlink()]
        if (
            len(links) != 1
            or links[0].name != dataset_name
            or links[0].resolve() != target
        ):
            raise ValueError(f"invalid existing single-task view: {destination}")
        return destination
    destination.mkdir(parents=True)
    (destination / dataset_name).symlink_to(target, target_is_directory=True)
    return destination


def replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise ValueError(f"source command has no {option}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"source command has no value for {option}")
    command[index + 1] = value


def recovery_command(
    *,
    source_command: Sequence[object],
    data_root: Path,
    out_dir: Path,
    physical_gpu: int,
    max_feature_cells_per_chunk: int = 2_000_000,
    predict_batch_size: int = 2_048,
) -> list[str]:
    command = [str(value) for value in source_command]
    for option, value in (
        ("--data-root", str(data_root)),
        ("--out-dir", str(out_dir)),
        ("--workers", "1"),
        ("--gpu-groups", str(physical_gpu)),
    ):
        replace_option(command, option, value)
    if "--resume" in command:
        command.remove("--resume")
    for option, value in (
        ("--ttt-max-feature-cells-per-chunk", str(max_feature_cells_per_chunk)),
        ("--tabicl-predict-batch-size", str(predict_batch_size)),
    ):
        if option in command:
            replace_option(command, option, value)
        else:
            command.extend([option, value])
    return command


def validate_attempt_result(
    path: Path, *, plan: CellPlan, dataset_name: str
) -> dict[str, str]:
    _, rows = read_csv(path)
    if any(row.get("dataset_name", "").startswith(SENTINEL_PREFIX) for row in rows):
        raise ValueError(f"attempt contains worker sentinel: {path}")
    by_name = real_rows(rows, source=path)
    if set(by_name) != {dataset_name}:
        raise ValueError(
            f"attempt result mismatch expected={dataset_name} actual={sorted(by_name)}"
        )
    row = by_name[dataset_name]
    if row.get("status") not in {"ok", "fail", "skip"}:
        raise ValueError(f"invalid attempt status in {path}: {row.get('status')!r}")
    if row.get("model_family") != plan.model:
        raise ValueError(f"wrong model_family in {path}")
    if row.get("peft_method") != plan.method:
        raise ValueError(f"wrong peft_method in {path}")
    return row


def failed_attempt_row(
    *, plan: CellPlan, dataset_name: str, error: str
) -> dict[str, str]:
    row = {field: "" for field in plan.fieldnames}
    row.update(
        {
            "dataset_name": dataset_name,
            "status": "fail",
            "error": error,
            "model_family": plan.model,
            "peft_method": plan.method,
            "ttt_applied": "False",
            "ttt_oom_fallback": "False",
        }
    )
    return row


def collect_attempt_result(
    attempt_dir: Path,
    *,
    plan: CellPlan,
    dataset_name: str,
    exit_code: int,
) -> dict[str, str]:
    result_path = attempt_dir / "all_classification_results.csv"
    if result_path.is_file():
        try:
            _, rows = read_csv(result_path)
            candidates = [
                row for row in rows if row.get("dataset_name") == dataset_name
            ]
            if len(candidates) == 1:
                row = candidates[0]
                if row.get("model_family") == plan.model and row.get("peft_method") == plan.method:
                    atomic_write_text(
                        attempt_dir / "validated_task_result.csv",
                        csv_text(list(dict.fromkeys([*plan.fieldnames, *row.keys()])), [row]),
                    )
                    return row
        except (OSError, ValueError):
            pass
    log_path = attempt_dir / "runner.log"
    log_tail = ""
    if log_path.is_file():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    row = failed_attempt_row(
        plan=plan,
        dataset_name=dataset_name,
        error=(
            f"RecoveryProcessError: runner exit_code={exit_code}; "
            f"no valid single-dataset result row. log_tail={log_tail}"
        ),
    )
    atomic_write_text(
        attempt_dir / "validated_task_result.csv",
        csv_text(plan.fieldnames, [row]),
    )
    return row


def find_reusable_attempt(
    recovery_root: Path, *, plan: CellPlan, dataset_name: str
) -> tuple[Path, dict[str, str]] | None:
    root = recovery_root / "attempts" / plan.model / plan.method / dataset_name
    attempts = sorted(
        root.glob("attempt_*"),
        key=lambda path: int(path.name.removeprefix("attempt_"))
        if path.name.removeprefix("attempt_").isdigit()
        else -1,
        reverse=True,
    )
    for attempt in attempts:
        for filename in ("validated_task_result.csv", "all_classification_results.csv"):
            result = attempt / filename
            if not result.is_file():
                continue
            try:
                return attempt, validate_attempt_result(
                    result, plan=plan, dataset_name=dataset_name
                )
            except (OSError, ValueError):
                continue
    return None


def is_sigterm_attempt(row: dict[str, str], attempt_dir: Path) -> bool:
    text = str(row.get("error", ""))
    log_path = attempt_dir / "runner.log"
    if log_path.is_file():
        text += "\n" + log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    lowered = text.lower()
    return any(marker in lowered for marker in SIGTERM_MARKERS)


def dataset_storage_bytes(source_view: Path, dataset_name: str) -> int:
    dataset_dir = (source_view / dataset_name).resolve(strict=True)
    return sum(path.stat().st_size for path in dataset_dir.iterdir() if path.is_file())


def execution_schedule(
    plans: Sequence[CellPlan], source_view: Path
) -> list[tuple[CellPlan, str]]:
    tasks = [
        (plan, dataset_name)
        for plan in plans
        for dataset_name in plan.retry_names
    ]
    priority = {
        ("tabpfnv3", "last_layers", "OpenML-ID-40978"): 0,
        ("tabiclv2", "last_layers", "OpenML-ID-40996"): 1,
        ("tabpfnv3", "last_layers", "OpenML-ID-4134"): 2,
        ("tabiclv2", "lora", "OpenML-ID-40996"): 3,
        ("tabiclv2", "ln_head_embedding", "OpenML-ID-40996"): 4,
    }
    return sorted(
        tasks,
        key=lambda item: (
            priority.get((item[0].model, item[0].method, item[1]), 100),
            dataset_storage_bytes(source_view, item[1]),
            item[1],
            item[0].model,
            item[0].method,
        ),
    )


def check_gpu(physical_gpu: int, minimum_free_ratio: float) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(physical_gpu)
    ratio = free / total
    if ratio < minimum_free_ratio:
        raise RuntimeError(
            f"physical GPU {physical_gpu} free ratio {ratio:.4f} "
            f"is below required {minimum_free_ratio:.4f}"
        )
    return {
        "physical_gpu": physical_gpu,
        "name": torch.cuda.get_device_name(physical_gpu),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "free_ratio": ratio,
    }


def result_summary(rows: Sequence[dict[str, str]]) -> str:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") == "fail"]
    skipped = [row for row in rows if row.get("status") == "skip"]
    oom = [row for row in rows if is_oom_row(row)]
    lines = [
        f"discovered_datasets: {len(rows)}",
        f"processed_datasets: {len(rows)}",
        f"ok_count: {len(ok)}",
        f"failed_count: {len(failed)}",
        f"skipped_count: {len(skipped)}",
        f"ttt_oom_fallback_count: {len(oom)}",
    ]
    for label, column in (
        ("avg_accuracy_ok", "accuracy"),
        ("avg_f1_ok", "f1"),
        ("avg_balanced_accuracy_ok", "balanced_accuracy"),
        ("avg_roc_auc_ok", "roc_auc"),
        ("avg_log_loss_ok", "log_loss"),
    ):
        values = []
        for row in ok:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        lines.append(
            f"{label}: {sum(values) / len(values):.6f}"
            if values
            else f"{label}: (none)"
        )
    lines.extend(
        [
            "wall_seconds: (targeted recovery; see recovery_manifest.json)",
            "failed_datasets: "
            + (", ".join(row["dataset_name"] for row in failed) or "(none)"),
            "skipped_datasets: "
            + (", ".join(row["dataset_name"] for row in skipped) or "(none)"),
            "ttt_oom_fallback_datasets: "
            + (", ".join(row["dataset_name"] for row in oom) or "(none)"),
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_merged_rows(
    *,
    plan: CellPlan,
    expected_names: Sequence[str],
    recovered: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, str]], list[str], list[str]]:
    if set(recovered) != set(plan.retry_names):
        raise ValueError(
            f"recovery coverage mismatch for {plan.model}/{plan.method}: "
            f"expected={plan.retry_names} actual={sorted(recovered)}"
        )
    merged = dict(plan.rows_by_name)
    replaced: list[str] = []
    retained_failure: list[str] = []
    for name in plan.retry_names:
        row = recovered[name]
        if name in plan.missing_names:
            merged[name] = row
            replaced.append(name)
        elif is_eligible_success(row):
            merged[name] = row
            replaced.append(name)
        else:
            retained_failure.append(name)
    rows = [merged[name] for name in expected_names]
    fields = list(
        dict.fromkeys(
            [
                *plan.fieldnames,
                *[key for row in recovered.values() for key in row],
            ]
        )
    )
    if len(rows) != 67 or len({row["dataset_name"] for row in rows}) != 67:
        raise ValueError(f"merged cell is not a unique 67-row result: {plan.result_csv}")
    return fields, rows, replaced, retained_failure


def apply_cell_transaction(
    *,
    plan: CellPlan,
    expected_names: Sequence[str],
    recovered: dict[str, dict[str, str]],
    recovery_root: Path,
) -> dict[str, Any]:
    if sha256(plan.result_csv) != plan.source_sha256:
        raise RuntimeError(f"source CSV changed during recovery: {plan.result_csv}")
    fields, rows, replaced, retained_failure = prepare_merged_rows(
        plan=plan, expected_names=expected_names, recovered=recovered
    )
    run_config_path = plan.output_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    metadata = {
        "recovery_root": str(recovery_root),
        "updated_at": now(),
        "source_csv_sha256": plan.source_sha256,
        "retry_names": plan.retry_names,
        "replaced_names": replaced,
        "retained_failure_names": retained_failure,
    }
    run_config["targeted_recovery"] = metadata

    transaction = recovery_root / "transactions" / plan.model / plan.method
    transaction.mkdir(parents=True, exist_ok=True)
    staged_csv = transaction / "all_classification_results.csv"
    staged_summary = transaction / "summary.txt"
    staged_config = transaction / "run_config.json"
    atomic_write_text(staged_csv, csv_text(fields, rows))
    atomic_write_text(staged_summary, result_summary(rows))
    atomic_write_json(staged_config, run_config)
    journal = {
        "status": "prepared",
        "cell": [plan.model, plan.method],
        "created_at": now(),
        "artifacts": [
            [str(staged_csv), str(plan.result_csv)],
            [str(staged_summary), str(plan.output_dir / "summary.txt")],
            [str(staged_config), str(run_config_path)],
        ],
    }
    journal_path = transaction / "journal.json"
    atomic_write_json(journal_path, journal)
    for _, destination in journal["artifacts"]:
        backup_once(Path(destination))
    journal["status"] = "applying"
    atomic_write_json(journal_path, journal)
    for source, destination in journal["artifacts"]:
        atomic_write_text(Path(destination), Path(source).read_text(encoding="utf-8"))
    journal["status"] = "committed"
    journal["committed_at"] = now()
    atomic_write_json(journal_path, journal)
    return {
        "model_family": plan.model,
        "peft_method": plan.method,
        "status": "merged",
        "rows": 67,
        "retried": len(plan.retry_names),
        "replaced_names": replaced,
        "retained_failure_names": retained_failure,
        "result_csv_sha256": sha256(plan.result_csv),
        "transaction": str(transaction),
    }


def update_matrix_manifest(
    *,
    matrix_manifest: Path,
    recovery_root: Path,
    cell_results: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> None:
    backup_once(matrix_manifest)
    payload = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    payload["targeted_recovery"] = {
        "status": "complete",
        "finished_at": now(),
        "recovery_root": str(recovery_root),
        "attempt_count": len(attempts),
        "cells": cell_results,
        "original_matrix_status_preserved": payload.get("status"),
    }
    atomic_write_json(matrix_manifest, payload)


def recover_global_transaction(recovery_root: Path) -> None:
    journal_path = recovery_root / "transaction" / "journal.json"
    if not journal_path.is_file():
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("status") != "applying":
        return
    for artifact in reversed(journal.get("artifacts", [])):
        destination = Path(artifact["destination"])
        snapshot = Path(artifact["snapshot"])
        if artifact.get("existed"):
            atomic_write_bytes(destination, snapshot.read_bytes())
        elif destination.exists():
            destination.unlink()
    journal["status"] = "rolled_back"
    journal["rolled_back_at"] = now()
    atomic_write_json(journal_path, journal)


def apply_global_transaction(
    *,
    plans: Sequence[CellPlan],
    expected_names: Sequence[str],
    recovered_by_cell: dict[tuple[str, str], dict[str, dict[str, str]]],
    recovery_root: Path,
    matrix_manifest: Path,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    affected = [plan for plan in plans if plan.retry_names]
    if not affected:
        raise ValueError("global recovery transaction has no affected cells")
    for plan in affected:
        if sha256(plan.result_csv) != plan.source_sha256:
            raise RuntimeError(f"source CSV changed during recovery: {plan.result_csv}")
        rows = recovered_by_cell[(plan.model, plan.method)]
        if set(rows) != set(plan.retry_names) or not all(
            is_eligible_success(row) for row in rows.values()
        ):
            raise RuntimeError(
                f"strict recovery gate failed for {plan.model}/{plan.method}"
            )

    transaction = recovery_root / "transaction"
    staged_root = transaction / "staged"
    snapshot_root = transaction / "snapshots"
    transaction.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    cell_results: list[dict[str, Any]] = []

    def stage(destination: Path, content: bytes, label: str) -> None:
        index = len(artifacts)
        staged = staged_root / f"{index:02d}-{label}"
        snapshot = snapshot_root / f"{index:02d}-{label}"
        atomic_write_bytes(staged, content)
        existed = destination.exists()
        if existed:
            atomic_write_bytes(snapshot, destination.read_bytes())
        artifacts.append(
            {
                "destination": str(destination),
                "staged": str(staged),
                "snapshot": str(snapshot),
                "existed": existed,
                "source_sha256": sha256(destination) if existed else None,
            }
        )

    for plan in affected:
        fields, rows, replaced, retained = prepare_merged_rows(
            plan=plan,
            expected_names=expected_names,
            recovered=recovered_by_cell[(plan.model, plan.method)],
        )
        if retained:
            raise RuntimeError(
                f"strict recovery would retain failures for {plan.model}/{plan.method}: {retained}"
            )
        run_config_path = plan.output_dir / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["targeted_recovery"] = {
            "status": "complete",
            "recovery_root": str(recovery_root),
            "updated_at": now(),
            "source_csv_sha256": plan.source_sha256,
            "retry_names": plan.retry_names,
            "replaced_names": replaced,
        }
        stage(
            plan.result_csv,
            csv_text(fields, rows).encode("utf-8"),
            f"{plan.model}-{plan.method}-results.csv",
        )
        stage(
            plan.output_dir / "summary.txt",
            result_summary(rows).encode("utf-8"),
            f"{plan.model}-{plan.method}-summary.txt",
        )
        stage(
            run_config_path,
            (json.dumps(run_config, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            f"{plan.model}-{plan.method}-run-config.json",
        )
        cell_results.append(
            {
                "model_family": plan.model,
                "peft_method": plan.method,
                "status": "merged",
                "rows": 67,
                "retried": len(plan.retry_names),
                "replaced_names": replaced,
                "retained_failure_names": [],
            }
        )

    manifest_payload = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    manifest_payload["targeted_recovery"] = {
        "status": "complete",
        "finished_at": now(),
        "recovery_root": str(recovery_root),
        "attempt_count": len(attempts),
        "cells": cell_results,
        "original_matrix_status_preserved": manifest_payload.get("status"),
    }
    stage(
        matrix_manifest,
        (json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        "matrix-manifest.json",
    )

    journal = {
        "status": "prepared",
        "created_at": now(),
        "artifacts": artifacts,
    }
    journal_path = transaction / "journal.json"
    atomic_write_json(journal_path, journal)
    for artifact in artifacts:
        destination = Path(artifact["destination"])
        current = sha256(destination) if destination.exists() else None
        if current != artifact["source_sha256"]:
            raise RuntimeError(f"artifact changed before global commit: {destination}")
    journal["status"] = "applying"
    atomic_write_json(journal_path, journal)
    try:
        for artifact in artifacts:
            atomic_write_bytes(
                Path(artifact["destination"]),
                Path(artifact["staged"]).read_bytes(),
            )
    except BaseException:
        recover_global_transaction(recovery_root)
        raise
    journal["status"] = "committed"
    journal["committed_at"] = now()
    atomic_write_json(journal_path, journal)
    for result in cell_results:
        plan = next(
            item
            for item in affected
            if item.model == result["model_family"] and item.method == result["peft_method"]
        )
        result["result_csv_sha256"] = sha256(plan.result_csv)
        result["transaction"] = str(transaction)
    return cell_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("results/PEFT/_matrix") / DEFAULT_RUN_NAME,
    )
    parser.add_argument(
        "--view-manifest",
        type=Path,
        default=Path(
            "results/dataset_views/openml_cc18_max10/dataset_view_manifest.json"
        ),
    )
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--minimum-free-ratio", type=float, default=0.9)
    parser.add_argument("--expected-targets", type=int, default=2)
    parser.add_argument(
        "--skip-datasets",
        nargs="*",
        default=list(DEFAULT_SKIPPED_DATASETS),
        help=(
            "Dataset names intentionally excluded from targeted recovery. "
            "OpenML-ID-40996 is skipped by default after its bounded-memory "
            "retry exceeded the practical runtime budget."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    matrix_root = (repo_root / args.matrix_root).resolve()
    matrix_manifest = matrix_root / "matrix_manifest.json"
    view_manifest = (repo_root / args.view_manifest).resolve()
    source_view = view_manifest.parent
    recovery_root = (
        (repo_root / args.recovery_root).resolve()
        if args.recovery_root
        else (repo_root / "results/PEFT/_recovery" / DEFAULT_RECOVERY_NAME)
    )
    manifest_path = recovery_root / "recovery_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            print(f"recovery already complete: {manifest_path}")
            return 0
    recover_global_transaction(recovery_root)
    expected_names = load_expected_names(view_manifest)
    payload, trials, plans = build_plans(
        matrix_manifest=matrix_manifest,
        expected_names=expected_names,
        repo_root=repo_root,
    )
    skipped_datasets = {str(name) for name in args.skip_datasets}
    unknown_skips = skipped_datasets - set(expected_names)
    if unknown_skips:
        raise ValueError(f"unknown --skip-datasets values: {sorted(unknown_skips)}")
    plans = exclude_datasets_from_plans(plans, skipped_datasets)
    total_targets = sum(len(plan.retry_names) for plan in plans)
    print(f"matrix_root: {matrix_root}")
    print(f"recovery_root: {recovery_root}")
    print(f"physical_gpu: {args.physical_gpu}")
    print(f"skipped_datasets: {','.join(sorted(skipped_datasets)) or '(none)'}")
    print(f"target_count: {total_targets}")
    for plan in plans:
        print(
            f"target {plan.model}/{plan.method}: {len(plan.retry_names)} "
            + (",".join(plan.retry_names) or "(none)")
        )
    if total_targets != args.expected_targets:
        raise RuntimeError(
            f"target inventory changed: expected={args.expected_targets} actual={total_targets}"
        )
    actual_targets = {
        (plan.model, plan.method, dataset_name)
        for plan in plans
        for dataset_name in plan.retry_names
    }
    if args.expected_targets == 2 and actual_targets != EXPECTED_NONOOM_TARGETS:
        raise RuntimeError(
            "two-target inventory changed: "
            f"expected={sorted(EXPECTED_NONOOM_TARGETS)} actual={sorted(actual_targets)}"
        )

    schedule = execution_schedule(plans, source_view)
    commands: list[dict[str, Any]] = []
    for plan, dataset_name in schedule:
        source_command = trials[(plan.model, plan.method)].get("command")
        if not isinstance(source_command, list):
            raise ValueError(f"matrix trial has no command: {plan.model}/{plan.method}")
        preview_view = recovery_root / "views" / dataset_name
        preview_out = (
            recovery_root
            / "attempts"
            / plan.model
            / plan.method
            / dataset_name
            / "attempt_1"
        )
        commands.append(
            {
                "model_family": plan.model,
                "peft_method": plan.method,
                "dataset_name": dataset_name,
                "command": recovery_command(
                    source_command=source_command,
                    data_root=preview_view,
                    out_dir=preview_out,
                    physical_gpu=args.physical_gpu,
                ),
            }
        )
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return 0

    recovery_root.mkdir(parents=True, exist_ok=True)
    lock_path = recovery_root / ".recovery.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another recovery process is running") from exc

        gpu = None
        if not args.skip_gpu_check and not args.merge_only:
            gpu = check_gpu(args.physical_gpu, args.minimum_free_ratio)
            print("gpu_check: " + json.dumps(gpu))

        recovery_manifest: dict[str, Any] = {
            "status": "running",
            "started_at": now(),
            "matrix_root": str(matrix_root),
            "matrix_manifest": str(matrix_manifest),
            "view_manifest": str(view_manifest),
            "physical_gpu": args.physical_gpu,
            "gpu_check": gpu,
            "runner_sha256": sha256(repo_root / "PEFT_Tabicl/1C_Chunk_PEFT.py"),
            "source_matrix_status": payload.get("status"),
            "target_count": total_targets,
            "skipped_datasets": sorted(skipped_datasets),
            "attempts": [],
            "cells": [],
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            recovery_manifest["started_at"] = previous.get(
                "started_at", recovery_manifest["started_at"]
            )
            recovery_manifest["attempts"] = list(previous.get("attempts", []))
        atomic_write_json(manifest_path, recovery_manifest)

        recovered_by_cell: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
        attempt_records: list[dict[str, Any]] = []
        for plan in plans:
            recovered_by_cell[(plan.model, plan.method)] = {}
        for plan, dataset_name in schedule:
            source_command = trials[(plan.model, plan.method)]["command"]
            view = make_single_task_view(
                recovery_root=recovery_root,
                source_view=source_view,
                dataset_name=dataset_name,
            )
            environment = os.environ.copy()
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            reusable = find_reusable_attempt(
                recovery_root, plan=plan, dataset_name=dataset_name
            )
            if reusable:
                attempt_dir, row = reusable
                exit_code = 0
                execution = "reused"
                command: list[str] = []
            elif args.merge_only:
                raise RuntimeError(
                    f"no reusable attempt for {plan.model}/{plan.method}/{dataset_name}"
                )
            else:
                attempt_dir = next_attempt_dir(recovery_root, plan, dataset_name)
                attempt_dir.mkdir(parents=True)
                command = recovery_command(
                    source_command=source_command,
                    data_root=view,
                    out_dir=attempt_dir,
                    physical_gpu=args.physical_gpu,
                )
                log_path = attempt_dir / "runner.log"
                print(
                    f"run {plan.model}/{plan.method}/{dataset_name}: "
                    + " ".join(command),
                    flush=True,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=repo_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                exit_code = int(process.returncode)
                row = collect_attempt_result(
                    attempt_dir,
                    plan=plan,
                    dataset_name=dataset_name,
                    exit_code=exit_code,
                )
                execution = "executed"
                if (
                    plan.model == "tabiclv2"
                    and not is_oom_row(row)
                    and is_sigterm_attempt(row, attempt_dir)
                ):
                    first_attempt_dir = attempt_dir
                    attempt_dir = next_attempt_dir(recovery_root, plan, dataset_name)
                    attempt_dir.mkdir(parents=True)
                    command = recovery_command(
                        source_command=source_command,
                        data_root=view,
                        out_dir=attempt_dir,
                        physical_gpu=args.physical_gpu,
                        max_feature_cells_per_chunk=1_000_000,
                        predict_batch_size=1_024,
                    )
                    log_path = attempt_dir / "runner.log"
                    print(
                        f"resource backoff {plan.model}/{plan.method}/{dataset_name}: "
                        + " ".join(command),
                        flush=True,
                    )
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.run(
                            command,
                            cwd=repo_root,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    exit_code = int(process.returncode)
                    row = collect_attempt_result(
                        attempt_dir,
                        plan=plan,
                        dataset_name=dataset_name,
                        exit_code=exit_code,
                    )
                    execution = "executed_resource_backoff"
                    recovery_manifest.setdefault("backoffs", []).append(
                        {
                            "model_family": plan.model,
                            "peft_method": plan.method,
                            "dataset_name": dataset_name,
                            "trigger_attempt_dir": str(first_attempt_dir),
                            "backoff_attempt_dir": str(attempt_dir),
                        }
                    )
            if (
                execution == "reused"
                and plan.model == "tabiclv2"
                and not is_oom_row(row)
                and is_sigterm_attempt(row, attempt_dir)
                and attempt_dir.name == "attempt_1"
                and not args.merge_only
            ):
                first_attempt_dir = attempt_dir
                attempt_dir = next_attempt_dir(recovery_root, plan, dataset_name)
                attempt_dir.mkdir(parents=True)
                command = recovery_command(
                    source_command=source_command,
                    data_root=view,
                    out_dir=attempt_dir,
                    physical_gpu=args.physical_gpu,
                    max_feature_cells_per_chunk=1_000_000,
                    predict_batch_size=1_024,
                )
                log_path = attempt_dir / "runner.log"
                print(
                    f"resume resource backoff {plan.model}/{plan.method}/{dataset_name}: "
                    + " ".join(command),
                    flush=True,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=repo_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                exit_code = int(process.returncode)
                row = collect_attempt_result(
                    attempt_dir,
                    plan=plan,
                    dataset_name=dataset_name,
                    exit_code=exit_code,
                )
                execution = "executed_resource_backoff_after_resume"
                recovery_manifest.setdefault("backoffs", []).append(
                    {
                        "model_family": plan.model,
                        "peft_method": plan.method,
                        "dataset_name": dataset_name,
                        "trigger_attempt_dir": str(first_attempt_dir),
                        "backoff_attempt_dir": str(attempt_dir),
                    }
                )
            recovered_by_cell[(plan.model, plan.method)][dataset_name] = row
            record = {
                "model_family": plan.model,
                "peft_method": plan.method,
                "dataset_name": dataset_name,
                "execution": execution,
                "attempt_dir": str(attempt_dir),
                "exit_code": exit_code,
                "result_status": row.get("status"),
                "eligible_success": is_eligible_success(row),
                "oom": is_oom_row(row),
                "finished_at": now(),
                "command": command,
            }
            attempt_records.append(record)
            recovery_manifest["attempts"] = attempt_records
            atomic_write_json(manifest_path, recovery_manifest)

        invalid = [
            record for record in attempt_records if not record["eligible_success"]
        ]
        if invalid:
            recovery_manifest.update(
                {
                    "status": "attempts_failed_no_commit",
                    "finished_at": now(),
                    "invalid_targets": invalid,
                    "formal_matrix_modified": False,
                }
            )
            atomic_write_json(manifest_path, recovery_manifest)
            print(f"recovery_manifest: {manifest_path}")
            print("recovery_status: attempts_failed_no_commit")
            return 2

        cell_results = apply_global_transaction(
            plans=plans,
            expected_names=expected_names,
            recovered_by_cell=recovered_by_cell,
            recovery_root=recovery_root,
            matrix_manifest=matrix_manifest,
            attempts=attempt_records,
        )
        recovery_manifest["cells"] = cell_results
        recovery_manifest["formal_matrix_modified"] = True
        atomic_write_json(manifest_path, recovery_manifest)
        audit_command = [
            sys.executable,
            str(repo_root / "scripts/summarize_openmlcc18_peft_matrix.py"),
            "--matrix-root",
            str(matrix_root),
            "--view-manifest",
            str(view_manifest),
            "--expected-datasets",
            "67",
        ]
        audit = subprocess.run(audit_command, cwd=repo_root, check=False)
        recovery_manifest.update(
            {
                "status": "complete" if audit.returncode == 0 else "audit_failed",
                "finished_at": now(),
                "audit_command": audit_command,
                "audit_exit_code": int(audit.returncode),
            }
        )
        atomic_write_json(manifest_path, recovery_manifest)
        print(f"recovery_manifest: {manifest_path}")
        print(f"recovery_status: {recovery_manifest['status']}")
        return int(audit.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
