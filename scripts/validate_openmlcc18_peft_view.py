#!/usr/bin/env python3
"""Validate the read-only OpenML-CC18 dataset view used by PEFT runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


MANIFEST_NAME = "dataset_view_manifest.json"
EXPECTED_MAX_CLASSES = 10
EXPECTED_SOURCE_ROOT_NAME = "openml_cc18"
REQUIRED_BENCHMARK_FILES = (
    "info.json",
    "N_train.npy",
    "C_train.npy",
    "y_train.npy",
    "N_val.npy",
    "C_val.npy",
    "y_val.npy",
    "N_test.npy",
    "C_test.npy",
    "y_test.npy",
)


def _resolve_manifest_path(raw_path: object, *, data_root: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def audit_openmlcc18_peft_view(
    data_root: Path | str,
    *,
    expected_datasets: int = 67,
) -> dict[str, Any]:
    """Return a machine-readable audit without raising for contract failures."""

    root = Path(data_root).expanduser().resolve()
    errors: list[str] = []
    missing_links: list[str] = []
    invalid_links: list[str] = []
    missing_files: dict[str, list[str]] = {}
    duplicate_names: list[str] = []
    extra_directories: list[str] = []
    names: list[str] = []
    source_root: Path | None = None

    if expected_datasets < 0:
        errors.append("expected_datasets must be non-negative")

    if not root.is_dir():
        errors.append(f"data root is not an existing directory: {root}")

    manifest_path = root / MANIFEST_NAME
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read manifest: {exc}")
        else:
            if not isinstance(loaded, dict):
                errors.append("manifest root must be a JSON object")
            else:
                manifest = loaded

    if manifest:
        if manifest.get("max_classes") != EXPECTED_MAX_CLASSES:
            errors.append(
                "manifest max_classes must be "
                f"{EXPECTED_MAX_CLASSES}, got {manifest.get('max_classes')!r}"
            )

        source_root = _resolve_manifest_path(
            manifest.get("source_root"),
            data_root=root,
        )
        if source_root is None:
            errors.append("manifest source_root must be a non-empty path")
        else:
            if source_root.name != EXPECTED_SOURCE_ROOT_NAME:
                errors.append(
                    "manifest source_root must name the OpenML benchmark root "
                    f"{EXPECTED_SOURCE_ROOT_NAME!r}, got {source_root}"
                )
            if not source_root.is_dir():
                errors.append(
                    f"manifest source_root is not an existing directory: {source_root}"
                )

        included = manifest.get("included")
        if not isinstance(included, list):
            errors.append("manifest included must be a list")
            included = []

        for index, entry in enumerate(included):
            if not isinstance(entry, dict):
                errors.append(f"manifest included[{index}] must be an object")
                continue
            name = entry.get("dataset_name")
            if not isinstance(name, str) or not name:
                errors.append(
                    f"manifest included[{index}].dataset_name must be a non-empty string"
                )
                continue
            if Path(name).name != name or name in {".", ".."}:
                errors.append(f"invalid dataset name in manifest: {name!r}")
                continue
            names.append(name)

        duplicate_names = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        if duplicate_names:
            errors.append(
                "manifest contains duplicate dataset names: "
                + ", ".join(duplicate_names)
            )

        if len(included) != expected_datasets:
            errors.append(
                f"manifest included task count is {len(included)}, "
                f"expected {expected_datasets}"
            )

        included_count = manifest.get("included_count")
        if included_count != len(included):
            errors.append(
                f"manifest included_count is {included_count!r}, "
                f"but included contains {len(included)} entries"
            )

    unique_names = sorted(set(names))
    if root.is_dir():
        represented = set(unique_names)
        extra_directories = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.name not in represented and entry.is_dir()
        )
        if extra_directories:
            errors.append(
                "dataset view contains extra directories: "
                + ", ".join(extra_directories)
            )

        for name in unique_names:
            link = root / name
            if not link.is_symlink():
                missing_links.append(name)
                continue
            if not link.exists() or not link.is_dir():
                invalid_links.append(name)
                continue

            if source_root is not None:
                expected_target = (source_root / name).resolve()
                if link.resolve() != expected_target:
                    invalid_links.append(name)
                    continue

            missing = [
                filename
                for filename in REQUIRED_BENCHMARK_FILES
                if not (link / filename).is_file()
            ]
            if missing:
                missing_files[name] = missing

    if missing_links:
        errors.append(
            "view entries are not dataset-directory symlinks: "
            + ", ".join(missing_links)
        )
    if invalid_links:
        errors.append(
            "view entries have missing or unexpected symlink targets: "
            + ", ".join(invalid_links)
        )
    if missing_files:
        errors.append(
            "view entries are missing required benchmark files: "
            + "; ".join(
                f"{name}=[{','.join(files)}]"
                for name, files in sorted(missing_files.items())
            )
        )

    return {
        "valid": not errors,
        "data_root": str(root),
        "manifest": str(manifest_path),
        "source_root": str(source_root) if source_root is not None else None,
        "expected_datasets": expected_datasets,
        "included_datasets": len(names),
        "unique_datasets": len(unique_names),
        "validated_datasets": len(unique_names)
        - len(missing_links)
        - len(invalid_links)
        - len(missing_files),
        "duplicate_names": duplicate_names,
        "missing_links": missing_links,
        "invalid_links": invalid_links,
        "missing_files": missing_files,
        "extra_directories": extra_directories,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the OpenML-CC18 PEFT dataset view contract."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("results/dataset_views/openml_cc18_max10"),
        help="Dataset view root containing dataset_view_manifest.json.",
    )
    parser.add_argument(
        "--expected-datasets",
        type=int,
        default=67,
        help="Exact number of included datasets required.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_openmlcc18_peft_view(
        args.data_root,
        expected_datasets=args.expected_datasets,
    )
    print(json.dumps(audit, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
