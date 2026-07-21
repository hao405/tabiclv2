#!/usr/bin/env python3
"""Atomically merge a successful MICP recovery CSV into the canonical result."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


METRICS = (
    "accuracy",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "log_loss",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def atomic_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
    *,
    discovered_datasets: int,
) -> None:
    ok = [row for row in rows if row["status"] == "ok"]
    failed = [
        row for row in rows if row["status"] not in {"ok", "skip"}
    ]
    skipped = [row for row in rows if row["status"] == "skip"]

    def mean(column: str) -> str:
        values = [
            float(row[column])
            for row in ok
            if row.get(column, "") not in {"", "None", "nan"}
        ]
        return f"{sum(values) / len(values):.6f}" if values else "(none)"

    def names(items: list[dict[str, str]]) -> str:
        return (
            ", ".join(row["dataset_name"] for row in items)
            if items
            else "(none)"
        )

    wall_seconds = sum(
        float(row.get("total_seconds") or 0.0) for row in rows
    )
    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"processed_datasets: {len(rows)}",
        f"ok_count: {len(ok)}",
        f"failed_count: {len(failed)}",
        f"skipped_count: {len(skipped)}",
        "ft_oom_fallback_count: 0",
        *(f"avg_{metric}_ok: {mean(metric)}" for metric in METRICS),
        f"wall_seconds: {wall_seconds:.3f}",
        f"failed_datasets: {names(failed)}",
        f"skipped_datasets: {names(skipped)}",
        "ft_oom_fallback_datasets: (none)",
    ]
    atomic_text(path, "\n".join(lines) + "\n")


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    recovery_dir = Path(sys.argv[2]).resolve()
    root = repo / "results/PEFT_results/MICP/data184/seed42"
    cell = root / "tabiclv2/mixturepfn"
    canonical_csv = cell / "all_classification_results.csv"
    recovery_csv = recovery_dir / "all_classification_results.csv"
    fieldnames, canonical = read_csv(canonical_csv)
    recovery_fields, recovery = read_csv(recovery_csv)
    if fieldnames != recovery_fields:
        raise RuntimeError("canonical and recovery CSV schemas differ")
    if len(canonical) != 184:
        raise RuntimeError(f"canonical row count is {len(canonical)}, expected 184")
    if len(recovery) != 11:
        raise RuntimeError(f"recovery row count is {len(recovery)}, expected 11")
    if any(row["status"] != "ok" for row in recovery):
        raise RuntimeError("recovery contains non-ok rows")

    canonical_by_name = {row["dataset_name"]: row for row in canonical}
    recovery_by_name = {row["dataset_name"]: row for row in recovery}
    if len(canonical_by_name) != len(canonical):
        raise RuntimeError("canonical CSV contains duplicate dataset names")
    if len(recovery_by_name) != len(recovery):
        raise RuntimeError("recovery CSV contains duplicate dataset names")
    missing = sorted(set(recovery_by_name) - set(canonical_by_name))
    if missing:
        raise RuntimeError(f"recovery datasets absent from canonical CSV: {missing}")
    not_failed = sorted(
        name
        for name in recovery_by_name
        if canonical_by_name[name]["status"] == "ok"
    )
    if not_failed:
        raise RuntimeError(f"recovery would replace existing ok rows: {not_failed}")

    run_id = recovery_dir.name
    backup = cell / "recovery_backups" / run_id
    backup.mkdir(parents=True, exist_ok=False)
    backup_files = {
        canonical_csv: "cell_all_classification_results.csv",
        cell / "summary.txt": "cell_summary.txt",
        cell / "run_manifest.json": "cell_run_manifest.json",
        root / "all_classification_results.csv": "root_all_classification_results.csv",
        root / "summary.txt": "root_summary.txt",
        root / "matrix_manifest.json": "root_matrix_manifest.json",
    }
    for path, backup_name in backup_files.items():
        if path.exists():
            shutil.copy2(path, backup / backup_name)

    merged: list[dict[str, str]] = []
    for original in canonical:
        replacement = recovery_by_name.get(original["dataset_name"])
        if replacement is None:
            merged.append(original)
            continue
        replacement = replacement.copy()
        replacement["config_fingerprint"] = original["config_fingerprint"]
        merged.append(replacement)
    if len(merged) != 184 or any(row["status"] != "ok" for row in merged):
        raise RuntimeError("merged TabICLv2 surface is not 184/184 ok")

    atomic_csv(canonical_csv, fieldnames, merged)
    write_summary(cell / "summary.txt", merged, discovered_datasets=184)

    script_path = repo / "PEFT_Tabicl/MICP.py"
    script_hash = sha256(script_path)
    run_manifest_path = cell / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.exists()
        else {}
    )
    run_manifest.update(
        {
            "script_sha256": script_hash,
            "dataset_count": 184,
            "ok_count": 184,
            "error_count": 0,
            "skip_count": 0,
            "status": "ok",
            "result_csv": str(canonical_csv),
            "recovery_run_id": run_id,
            "recovered_datasets": sorted(recovery_by_name),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
    )
    atomic_json(run_manifest_path, run_manifest)

    combined: list[dict[str, str]] = []
    for model in ("tabiclv2", "tabpfnv3"):
        model_csv = root / model / "mixturepfn/all_classification_results.csv"
        model_fields, model_rows = read_csv(model_csv)
        if model_fields != fieldnames:
            raise RuntimeError(f"{model} CSV schema differs")
        combined.extend(model_rows)
    atomic_csv(root / "all_classification_results.csv", fieldnames, combined)
    write_summary(root / "summary.txt", combined, discovered_datasets=184)

    matrix_path = root / "matrix_manifest.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for trial in matrix.get("trials", []):
        model = trial["model_family"]
        _, model_rows = read_csv(
            root / model / "mixturepfn/all_classification_results.csv"
        )
        ok_count = sum(row["status"] == "ok" for row in model_rows)
        trial.update(
            {
                "row_count": len(model_rows),
                "ok_count": ok_count,
                "status": "ok" if ok_count == len(model_rows) else "partial",
            }
        )
    matrix["script_sha256"] = script_hash
    matrix["status"] = (
        "ok"
        if all(trial.get("status") == "ok" for trial in matrix["trials"])
        else "partial"
    )
    matrix["updated_at"] = datetime.now().astimezone().isoformat()
    atomic_json(matrix_path, matrix)

    merge_manifest = {
        "recovery_run_id": run_id,
        "recovery_csv": str(recovery_csv),
        "canonical_csv": str(canonical_csv),
        "backup_dir": str(backup),
        "recovered_datasets": sorted(recovery_by_name),
        "before_ok_count": 173,
        "after_ok_count": 184,
        "script_sha256": script_hash,
        "canonical_csv_sha256": sha256(canonical_csv),
        "merged_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(cell / "recovery_merge_manifest.json", merge_manifest)
    atomic_json(backup / "recovery_merge_manifest.json", merge_manifest)
    print(json.dumps(merge_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
