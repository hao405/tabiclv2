#!/usr/bin/env python3
"""Aggregate one seed of the LimiX-2M method matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.limix import METHODS


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def summarize(root: Path, methods: list[str], expected_datasets: int) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    successful_sets: list[set[str]] = []
    for method in methods:
        method_dir = root / method
        manifest_path = method_dir / "run_manifest.json"
        csv_path = method_dir / "all_classification_results.csv"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()
        if len(frame):
            frame = frame.copy()
            frame["method"] = method
            frames.append(frame)
            ok_names = set(
                frame.loc[
                    frame["status"].astype(str).str.lower() == "ok",
                    "dataset_name",
                ].astype(str)
            )
        else:
            ok_names = set()
        successful_sets.append(ok_names)
        status_counts = (
            frame["status"].astype(str).value_counts().to_dict()
            if len(frame) and "status" in frame
            else {}
        )
        cells.append(
            {
                "method": method,
                "output_dir": str(method_dir),
                "manifest_status": manifest.get("status", "missing"),
                "processed_datasets": int(len(frame)),
                "status_counts": {
                    str(key): int(value) for key, value in status_counts.items()
                },
                "config_hash": manifest.get("config_hash"),
            }
        )
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["dataset_name", "method", "status"])
    )
    atomic_csv(root / "all_methods_results.csv", combined)
    shared = set.intersection(*successful_sets) if successful_sets else set()
    (root / "shared_status_ok_datasets.txt").write_text(
        "".join(f"{name}\n" for name in sorted(shared)),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "methods": methods,
        "expected_methods": len(methods),
        "expected_datasets_per_method": int(expected_datasets),
        "expected_dataset_method_tasks": len(methods) * int(expected_datasets),
        "combined_rows": int(len(combined)),
        "shared_status_ok_count": len(shared),
        "cells": cells,
        "status": (
            "complete"
            if len(cells) == len(methods)
            and all(
                cell["processed_datasets"] == expected_datasets
                for cell in cells
            )
            else "incomplete"
        ),
    }
    atomic_json(root / "matrix_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--expected-datasets", type=int, default=184)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    payload = summarize(
        Path(args.root).expanduser().resolve(),
        methods,
        args.expected_datasets,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
