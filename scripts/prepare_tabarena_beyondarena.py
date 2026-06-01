#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
PROBLEM_TYPE_TO_TASK_TYPE = {
    "binary": "binclass",
    "binary_classification": "binclass",
    "binclass": "binclass",
    "multiclass": "multiclass",
    "multiclass_classification": "multiclass",
}


def clear_unusable_socks_proxy() -> None:
    try:
        import socksio  # noqa: F401

        return
    except Exception:
        pass

    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key, "")
        if value.lower().startswith(("socks4://", "socks5://")):
            os.environ.pop(key, None)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot_dataset(repo_id: str, dataset_name: str, raw_root: Path) -> Path:
    clear_unusable_socks_proxy()
    from huggingface_hub import snapshot_download

    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{dataset_name}/**"],
        local_dir=raw_root,
    )
    return raw_root / dataset_name


def find_container_dir(dataset_root: Path) -> Path:
    matches = sorted(path.parent for path in dataset_root.rglob("dataset.parquet"))
    if not matches:
        raise FileNotFoundError(f"No dataset.parquet found under {dataset_root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple dataset.parquet files found under {dataset_root}: {matches}")
    return matches[0]


def read_parquet(path: Path):
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception as pandas_exc:
        try:
            import polars as pl

            return pl.read_parquet(path).to_pandas()
        except Exception as polars_exc:
            raise RuntimeError(
                f"Failed to read {path} with pandas/pyarrow ({pandas_exc}) "
                f"or polars ({polars_exc}). Install polars or upgrade pyarrow."
            ) from polars_exc


def split_indices(split_metadata: dict[str, Any], outer_split: str, inner_split: str) -> tuple[list[int], list[int]]:
    splits = split_metadata["splits"]
    try:
        split = splits[str(outer_split)][str(inner_split)]
    except KeyError as exc:
        available_outer = ",".join(sorted(splits))
        raise KeyError(
            f"Split {outer_split}/{inner_split} not available. "
            f"Available outer splits: {available_outer}"
        ) from exc

    if not isinstance(split, list) or len(split) != 2:
        raise ValueError(
            "Expected TabArena split metadata to contain [train_indices, test_indices] "
            f"for split {outer_split}/{inner_split}, got {type(split).__name__} length={len(split) if isinstance(split, list) else 'n/a'}"
        )
    return list(map(int, split[0])), list(map(int, split[1]))


def is_categorical_dtype(dtype_name: str) -> bool:
    normalized = dtype_name.lower()
    return normalized in {"category", "categorical", "object", "string", "str", "bool", "boolean"}


def normalize_categorical_frame(frame):
    import pandas as pd

    return frame.apply(
        lambda series: series.astype("string").fillna(CATEGORICAL_MISSING_TOKEN).astype(str)
    ).astype(object)


def save_split_arrays(
    out_dir: Path,
    split_name: str,
    frame,
    y,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    if numeric_columns:
        numeric_values = frame[numeric_columns].apply(lambda col: col.astype("float64")).to_numpy()
        np.save(out_dir / f"N_{split_name}.npy", numeric_values)
    if categorical_columns:
        categorical_values = normalize_categorical_frame(frame[categorical_columns]).to_numpy(dtype=object)
        np.save(out_dir / f"C_{split_name}.npy", categorical_values)
    np.save(out_dir / f"y_{split_name}.npy", np.asarray(y, dtype=object))


def convert_dataset(
    *,
    dataset_name: str,
    raw_root: Path,
    out_root: Path,
    repo_id: str,
    outer_split: str,
    inner_split: str,
    skip_download: bool,
) -> dict[str, Any]:
    if skip_download:
        dataset_root = raw_root / dataset_name
    else:
        dataset_root = snapshot_dataset(repo_id, dataset_name, raw_root)

    container_dir = find_container_dir(dataset_root)
    task_metadata = read_json(container_dir / "task_metadata.predictive-ml-task-mold-v1.json")
    split_metadata = read_json(container_dir / "experiment_metadata.predictive-ml-splits-mold-v1.json")
    dataset_metadata = read_json(container_dir / "dataset_metadata.dataset-mold-v1.json")
    dtypes = read_json(container_dir / "dtypes.json")

    target_column = task_metadata["target_column_name"]
    problem_type = str(task_metadata.get("problem_type", "")).lower()
    task_type = PROBLEM_TYPE_TO_TASK_TYPE.get(problem_type)
    if task_type is None:
        raise ValueError(f"{dataset_name} has unsupported problem_type={problem_type!r}")

    frame = read_parquet(container_dir / "dataset.parquet")
    if target_column not in frame.columns:
        raise KeyError(f"Target column {target_column!r} not found in {dataset_name}")

    train_idx, test_idx = split_indices(split_metadata, outer_split, inner_split)
    feature_columns = [column for column in frame.columns if column != target_column]
    categorical_columns = [
        column
        for column in feature_columns
        if is_categorical_dtype(str(dtypes.get(column, frame[column].dtype)))
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    y_train = train_frame[target_column].astype("string").fillna(CATEGORICAL_MISSING_TOKEN).astype(str)
    y_test = test_frame[target_column].astype("string").fillna(CATEGORICAL_MISSING_TOKEN).astype(str)

    save_split_arrays(out_dir, "train", train_frame, y_train, numeric_columns, categorical_columns)
    save_split_arrays(out_dir, "test", test_frame, y_test, numeric_columns, categorical_columns)

    info = {
        "name": dataset_name,
        "task_type": task_type,
        "problem_type": problem_type,
        "target_column": target_column,
        "objective_metric_name": task_metadata.get("objective_metric_name"),
        "n_num_features": len(numeric_columns),
        "n_cat_features": len(categorical_columns),
        "num_feature_intro": {column: column for column in numeric_columns},
        "cat_feature_intro": {column: column for column in categorical_columns},
        "train_size": len(train_idx),
        "val_size": 0,
        "test_size": len(test_idx),
        "n_classes": int(len(set(y_train.tolist()) | set(y_test.tolist()))),
        "source": "https://huggingface.co/datasets/TabArena/BeyondArena",
        "tabarena_repo_id": repo_id,
        "tabarena_unique_name": dataset_metadata.get("unique_name", dataset_name),
        "tabarena_container_dir": str(container_dir),
        "tabarena_outer_split": str(outer_split),
        "tabarena_inner_split": str(inner_split),
        "tabarena_dataset_source": dataset_metadata.get("dataset_source"),
        "tabarena_original_download_link": dataset_metadata.get("original_dataset_source_download_link"),
    }
    write_json(out_dir / "info.json", info)
    return info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TabArena/BeyondArena datasets and convert them to data178-style numpy splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("datasets", nargs="*", default=["churn"])
    parser.add_argument("--repo-id", default="TabArena/BeyondArena")
    parser.add_argument("--raw-root", default="data_tabarena_raw")
    parser.add_argument("--out-root", default="data_tabarena")
    parser.add_argument("--outer-split", default="0")
    parser.add_argument("--inner-split", default="0")
    parser.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_root = Path(args.raw_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset_name in args.datasets:
        info = convert_dataset(
            dataset_name=dataset_name,
            raw_root=raw_root,
            out_root=out_root,
            repo_id=args.repo_id,
            outer_split=args.outer_split,
            inner_split=args.inner_split,
            skip_download=args.skip_download,
        )
        records.append(info)
        print(
            f"[tabarena] {dataset_name}: task={info['task_type']} "
            f"train={info['train_size']} test={info['test_size']} "
            f"num={info['n_num_features']} cat={info['n_cat_features']}"
        )

    write_json(
        out_root / "_tabarena_manifest.json",
        {
            "repo_id": args.repo_id,
            "outer_split": str(args.outer_split),
            "inner_split": str(args.inner_split),
            "datasets": records,
        },
    )
    print(f"[tabarena] wrote {out_root / '_tabarena_manifest.json'}")


if __name__ == "__main__":
    main()
