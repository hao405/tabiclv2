#!/usr/bin/env python3
"""Build a symlink-based data184 stratification by table cell count.

For each dataset:

    n_features = n_num_features + n_cat_features
    n_rows = train_size + val_size + test_size
    n_cells = n_features * n_rows

The default bands are:

    small_lt20000_cells: n_cells < 20_000
    medium_20000_80000_cells: 20_000 <= n_cells <= 80_000
    large_gt80000_cells: n_cells > 80_000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


SMALL_GROUP = "small_lt20000_cells"
MEDIUM_GROUP = "medium_20000_80000_cells"
LARGE_GROUP = "large_gt80000_cells"
GROUPS = (SMALL_GROUP, MEDIUM_GROUP, LARGE_GROUP)


@dataclass(frozen=True)
class DatasetRecord:
    dataset: str
    cell_group: str
    n_features: int
    n_rows: int
    n_cells: int
    source_path: Path
    stratified_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify data184 by n_features * total rows using symlinks."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data184"),
        help="Source dataset root (default: data184).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data184_by_cells"),
        help="Output stratification root (default: data184_by_cells).",
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=20_000,
        help="Exclusive upper bound for the small group (default: 20000).",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=80_000,
        help="Inclusive upper bound for the medium group (default: 80000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output root.",
    )
    return parser.parse_args()


def require_nonnegative_int(info: dict[str, object], key: str, info_path: Path) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{info_path}: {key} must be a non-negative integer, got {value!r}")
    return value


def assign_group(n_cells: int, small_threshold: int, large_threshold: int) -> str:
    if n_cells < small_threshold:
        return SMALL_GROUP
    if n_cells <= large_threshold:
        return MEDIUM_GROUP
    return LARGE_GROUP


def display_path(path: Path, repo_root: Path) -> str:
    absolute_path = path if path.is_absolute() else repo_root / path
    try:
        return absolute_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def collect_records(
    source_root: Path,
    output_root: Path,
    small_threshold: int,
    large_threshold: int,
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    dataset_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found under {source_root}")

    for dataset_dir in dataset_dirs:
        info_path = dataset_dir / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {info_path}")
        with info_path.open(encoding="utf-8") as handle:
            info = json.load(handle)

        n_features = require_nonnegative_int(info, "n_num_features", info_path)
        n_features += require_nonnegative_int(info, "n_cat_features", info_path)
        n_rows = sum(
            require_nonnegative_int(info, key, info_path)
            for key in ("train_size", "val_size", "test_size")
        )
        if n_features == 0:
            raise ValueError(f"{info_path}: total feature count must be positive")
        if n_rows == 0:
            raise ValueError(f"{info_path}: total row count must be positive")

        n_cells = n_features * n_rows
        cell_group = assign_group(n_cells, small_threshold, large_threshold)
        records.append(
            DatasetRecord(
                dataset=dataset_dir.name,
                cell_group=cell_group,
                n_features=n_features,
                n_rows=n_rows,
                n_cells=n_cells,
                source_path=dataset_dir,
                stratified_path=output_root / cell_group / dataset_dir.name,
            )
        )
    return records


def prepare_output(output_root: Path, force: bool) -> None:
    if output_root.exists() or output_root.is_symlink():
        if not force:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --force to replace it."
            )
        if output_root.is_symlink() or output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    for group in GROUPS:
        (output_root / group).mkdir(parents=True, exist_ok=True)


def write_outputs(
    records: list[DatasetRecord],
    source_root: Path,
    output_root: Path,
    small_threshold: int,
    large_threshold: int,
) -> None:
    repo_root = Path.cwd()
    for record in records:
        relative_target = Path(
            os.path.relpath(
                record.source_path.resolve(), start=record.stratified_path.parent.resolve()
            )
        )
        record.stratified_path.symlink_to(relative_target, target_is_directory=True)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "cell_group",
                "n_features",
                "n_rows",
                "n_cells",
                "source_path",
                "stratified_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.dataset,
                    record.cell_group,
                    record.n_features,
                    record.n_rows,
                    record.n_cells,
                    display_path(record.source_path, repo_root),
                    display_path(record.stratified_path, repo_root),
                ]
            )

    counts = {group: 0 for group in GROUPS}
    for record in records:
        counts[record.cell_group] += 1
    summary_lines = [
        "data184 cell-count strata",
        f"data_root={display_path(source_root, repo_root)}",
        "cell_definition=(n_num_features+n_cat_features)*(train_size+val_size+test_size)",
        f"small_rule=n_cells<{small_threshold}",
        f"medium_rule={small_threshold}<=n_cells<={large_threshold}",
        f"large_rule=n_cells>{large_threshold}",
        "layout=symlink",
        f"total={len(records)}",
        *(f"{group}={counts[group]}" for group in GROUPS),
    ]
    (output_root / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def validate_thresholds(small_threshold: int, large_threshold: int) -> None:
    if small_threshold <= 0:
        raise ValueError("--small-threshold must be positive")
    if large_threshold < small_threshold:
        raise ValueError("--large-threshold must be >= --small-threshold")


def main() -> None:
    args = parse_args()
    validate_thresholds(args.small_threshold, args.large_threshold)

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root does not exist: {source_root}")
    if source_root == output_root:
        raise ValueError("Source root and output root must be different")

    records = collect_records(
        source_root,
        output_root,
        args.small_threshold,
        args.large_threshold,
    )
    prepare_output(output_root, args.force)
    write_outputs(
        records,
        source_root,
        output_root,
        args.small_threshold,
        args.large_threshold,
    )
    print((output_root / "summary.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
