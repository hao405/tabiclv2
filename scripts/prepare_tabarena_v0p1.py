#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABARENA_VERSION = "TabArena-v0.1"
DEFAULT_METADATA_CSV_URL = (
    "https://raw.githubusercontent.com/autogluon/tabarena/main/"
    "tabarena/tabarena/benchmark/task/metadata/sources/data/"
    "TabArena-v0.1_tasks_metadata.csv"
)
DEFAULT_OPENML_SERVER = "https://api.openml.org/api/v1/xml"
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
PROBLEM_TYPE_TO_TASK_TYPE = {
    "binary": "binclass",
    "multiclass": "multiclass",
    "regression": "regression",
}
EXPECTED_FULL_DATASETS = 51
EXPECTED_FULL_SPLITS = 816
SPLIT_INDEX_PATTERN = re.compile(r"^r(?P<repeat>\d+)f(?P<fold>\d+)$")
KNOWN_OUTPUT_FILES = {
    "N_train.npy",
    "C_train.npy",
    "y_train.npy",
    "N_test.npy",
    "C_test.npy",
    "y_test.npy",
    "info.json",
}


@dataclass(frozen=True)
class LoadedOpenMLTask:
    task_id: str
    openml_dataset_id: int | None
    openml_dataset_name: str | None
    target_name: str
    X: pd.DataFrame
    y: pd.Series
    categorical_indicator: list[bool] | None
    task: Any


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def metadata_cache_path(raw_root: Path) -> Path:
    return raw_root / "metadata" / "TabArena-v0.1_tasks_metadata.csv"


def resolve_metadata_csv(metadata_csv: str, raw_root: Path) -> tuple[Path, bool]:
    if not is_url(metadata_csv):
        path = Path(metadata_csv).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Metadata CSV does not exist: {path}")
        return path, False

    out_path = metadata_cache_path(raw_root)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path, True
    if out_path.exists():
        out_path.unlink()

    clear_unusable_socks_proxy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tabarena-v0.1] downloading metadata CSV: {metadata_csv}", flush=True)
    last_exc: Exception | None = None
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(metadata_csv, tmp_path)
            tmp_path.replace(out_path)
            break
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 3:
                raise
            print(
                f"[tabarena-v0.1] metadata download failed on attempt {attempt}/3: "
                f"{type(exc).__name__}: {exc}; retrying",
                flush=True,
            )
            time.sleep(float(attempt))
    if last_exc is not None and not out_path.exists():
        raise last_exc
    return out_path, True


def load_metadata_table(metadata_csv: str, raw_root: Path) -> tuple[pd.DataFrame, Path, bool]:
    path, cached = resolve_metadata_csv(metadata_csv, raw_root)
    dtype = {
        "dataset_name": "string",
        "problem_type": "string",
        "target_name": "string",
        "eval_metric": "string",
        "task_id_str": "string",
        "split_index": "string",
    }
    frame = pd.read_csv(path, dtype=dtype)
    required = {
        "dataset_name",
        "problem_type",
        "target_name",
        "eval_metric",
        "task_id_str",
        "repeat",
        "fold",
        "num_instances_train",
        "num_instances_test",
        "split_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    frame["dataset_name"] = frame["dataset_name"].astype(str)
    frame["problem_type"] = frame["problem_type"].astype(str).str.lower()
    frame["target_name"] = frame["target_name"].astype(str)
    frame["eval_metric"] = frame["eval_metric"].astype(str)
    frame["task_id_str"] = frame["task_id_str"].astype(str)
    frame["split_index"] = frame["split_index"].astype(str)
    frame["repeat"] = frame["repeat"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame, path, cached


def parse_split_index(split_index: str) -> tuple[int, int]:
    match = SPLIT_INDEX_PATTERN.match(split_index)
    if match is None:
        raise ValueError(f"Invalid split index {split_index!r}; expected r{{repeat}}f{{fold}}")
    return int(match.group("repeat")), int(match.group("fold"))


def validate_split_indices(split_indices: list[str] | None) -> None:
    for split_index in split_indices or []:
        parse_split_index(split_index)


def filter_metadata(
    frame: pd.DataFrame,
    *,
    datasets: list[str] | None,
    problem_types: list[str],
    split_mode: str,
    split_indices: list[str] | None,
) -> pd.DataFrame:
    problem_types = [value.lower() for value in problem_types]
    unsupported = sorted(set(problem_types) - set(PROBLEM_TYPE_TO_TASK_TYPE))
    if unsupported:
        raise ValueError(f"Unsupported problem types: {unsupported}")

    selected = frame[frame["problem_type"].isin(problem_types)].copy()

    if datasets:
        available = set(frame["dataset_name"].unique())
        requested = set(datasets)
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"Requested dataset names not found in TabArena v0.1 metadata: {missing}"
            )
        selected = selected[selected["dataset_name"].isin(datasets)].copy()

    validate_split_indices(split_indices)
    if split_indices:
        selected = selected[selected["split_index"].isin(split_indices)].copy()
    elif split_mode == "lite":
        selected = selected[selected["split_index"] == "r0f0"].copy()
    elif split_mode != "all":
        raise ValueError(f"Unsupported split mode {split_mode!r}")

    selected = selected.sort_values(["dataset_name", "repeat", "fold"]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No TabArena v0.1 metadata rows matched the requested filters.")
    return selected


def safe_dataset_dir_name(dataset_name: str, split_index: str) -> str:
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    safe_name = "".join("_" if ord(ch) < 32 else ch for ch in safe_name)
    return f"{safe_name}__{split_index}"


def is_categorical_dtype(dtype: Any) -> bool:
    try:
        import pandas.api.types as pdt

        return bool(isinstance(dtype, pd.CategoricalDtype) or pdt.is_bool_dtype(dtype))
    except Exception:
        normalized = str(dtype).lower()
        return normalized in {"category", "categorical", "bool", "boolean"}


def infer_feature_columns(
    X: pd.DataFrame,
    categorical_indicator: list[bool] | None,
) -> tuple[list[str], list[str]]:
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    indicator = categorical_indicator if categorical_indicator and len(categorical_indicator) == len(X.columns) else None

    for idx, column in enumerate(X.columns):
        series = X[column]
        marked_categorical = bool(indicator[idx]) if indicator is not None else False
        if marked_categorical or is_categorical_dtype(series.dtype):
            categorical = True
        else:
            converted = pd.to_numeric(series, errors="coerce")
            introduced_nan = converted.isna() & series.notna()
            categorical = bool(introduced_nan.any())

        if categorical:
            categorical_columns.append(str(column))
        else:
            numeric_columns.append(str(column))

    return numeric_columns, categorical_columns


def normalize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    normalized = X.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    return normalized


def numeric_values(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = []
    for column in columns:
        series = frame[column]
        converted = pd.to_numeric(series, errors="coerce")
        introduced_nan = converted.isna() & series.notna()
        if introduced_nan.any():
            raise ValueError(f"Column {column!r} was selected as numeric but contains non-numeric values")
        values.append(converted.astype("float64").to_numpy())
    return np.column_stack(values).astype("float64", copy=False)


def categorical_values(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    return (
        frame[columns]
        .apply(lambda series: series.astype("string").fillna(CATEGORICAL_MISSING_TOKEN).astype(str))
        .to_numpy(dtype=object)
    )


def target_values(y: pd.Series, problem_type: str) -> np.ndarray:
    if problem_type == "regression":
        converted = pd.to_numeric(y, errors="coerce")
        introduced_nan = converted.isna() & y.notna()
        if introduced_nan.any():
            raise ValueError("Regression target contains non-numeric values")
        return converted.astype("float64").to_numpy()
    return y.astype("string").fillna(CATEGORICAL_MISSING_TOKEN).astype(str).to_numpy(dtype=object)


def clean_known_output_files(out_dir: Path) -> None:
    for file_name in KNOWN_OUTPUT_FILES:
        path = out_dir / file_name
        if path.exists():
            path.unlink()


def output_is_complete(out_dir: Path, row: pd.Series) -> bool:
    info_path = out_dir / "info.json"
    if not info_path.exists() or not (out_dir / "y_train.npy").exists() or not (out_dir / "y_test.npy").exists():
        return False
    if not ((out_dir / "N_train.npy").exists() or (out_dir / "C_train.npy").exists()):
        return False
    if not ((out_dir / "N_test.npy").exists() or (out_dir / "C_test.npy").exists()):
        return False

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(info.get("tabarena_split_index")) == str(row["split_index"])
        and str(info.get("openml_task_id")) == str(row["task_id_str"])
        and str(info.get("tabarena_dataset_name")) == str(row["dataset_name"])
    )


def skipped_existing_record(row: pd.Series, out_root: Path) -> dict[str, Any]:
    dataset_name = str(row["dataset_name"])
    problem_type = str(row["problem_type"]).lower()
    task_type = PROBLEM_TYPE_TO_TASK_TYPE[problem_type]
    split_index = str(row["split_index"])
    out_name = safe_dataset_dir_name(dataset_name, split_index)
    out_dir = out_root / out_name
    info = json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
    return {
        "dataset_name": dataset_name,
        "output_name": out_name,
        "out_dir": str(out_dir),
        "problem_type": problem_type,
        "task_type": task_type,
        "openml_task_id": str(row["task_id_str"]),
        "openml_dataset_id": info.get("openml_dataset_id"),
        "openml_dataset_name": info.get("openml_dataset_name"),
        "tabarena_repeat": int(row["repeat"]),
        "tabarena_fold": int(row["fold"]),
        "tabarena_split_index": split_index,
        "status": "skipped_existing",
        "error": None,
        "train_size": info.get("train_size"),
        "test_size": info.get("test_size"),
        "n_num_features": info.get("n_num_features"),
        "n_cat_features": info.get("n_cat_features"),
        "n_classes": info.get("n_classes"),
    }


def expected_count(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(round(float(value)))


def validate_indices(
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
    expected_train: int | None,
    expected_test: int | None,
) -> None:
    if train_idx.ndim != 1 or test_idx.ndim != 1:
        raise ValueError("OpenML split indices must be one-dimensional")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("OpenML split produced an empty train or test partition")
    if train_idx.min() < 0 or test_idx.min() < 0 or train_idx.max() >= n_samples or test_idx.max() >= n_samples:
        raise ValueError("OpenML split indices are out of bounds for the dataset")
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap):
        raise ValueError(f"OpenML train/test split overlap has {len(overlap)} rows")
    if expected_train is not None and len(train_idx) != expected_train:
        raise ValueError(f"Train size mismatch: metadata={expected_train} actual={len(train_idx)}")
    if expected_test is not None and len(test_idx) != expected_test:
        raise ValueError(f"Test size mismatch: metadata={expected_test} actual={len(test_idx)}")


def load_openml_task(task_id: str) -> LoadedOpenMLTask:
    clear_unusable_socks_proxy()
    import openml

    task = openml.tasks.get_task(
        int(task_id),
        download_data=True,
        download_qualities=True,
        download_splits=True,
    )
    dataset = task.get_dataset()
    target_name = str(getattr(task, "target_name", ""))
    X, y, categorical_indicator, _attribute_names = dataset.get_data(
        target=target_name,
        dataset_format="dataframe",
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name=target_name)

    return LoadedOpenMLTask(
        task_id=str(task_id),
        openml_dataset_id=getattr(dataset, "id", None),
        openml_dataset_name=getattr(dataset, "name", None),
        target_name=target_name,
        X=normalize_feature_frame(X),
        y=y.reset_index(drop=True),
        categorical_indicator=[bool(value) for value in categorical_indicator]
        if categorical_indicator is not None
        else None,
        task=task,
    )


def convert_split(
    *,
    loaded: LoadedOpenMLTask,
    row: pd.Series,
    out_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    dataset_name = str(row["dataset_name"])
    problem_type = str(row["problem_type"]).lower()
    task_type = PROBLEM_TYPE_TO_TASK_TYPE[problem_type]
    split_index = str(row["split_index"])
    repeat = int(row["repeat"])
    fold = int(row["fold"])
    out_name = safe_dataset_dir_name(dataset_name, split_index)
    out_dir = out_root / out_name

    base_record: dict[str, Any] = {
        "dataset_name": dataset_name,
        "output_name": out_name,
        "out_dir": str(out_dir),
        "problem_type": problem_type,
        "task_type": task_type,
        "openml_task_id": str(row["task_id_str"]),
        "openml_dataset_id": loaded.openml_dataset_id,
        "openml_dataset_name": loaded.openml_dataset_name,
        "tabarena_repeat": repeat,
        "tabarena_fold": fold,
        "tabarena_split_index": split_index,
    }

    if output_is_complete(out_dir, row) and not overwrite:
        return {**base_record, "status": "skipped_existing", "error": None}

    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clean_known_output_files(out_dir)

    train_idx_raw, test_idx_raw = loaded.task.get_train_test_split_indices(fold=fold, repeat=repeat)
    train_idx = np.asarray(train_idx_raw, dtype=np.int64)
    test_idx = np.asarray(test_idx_raw, dtype=np.int64)
    validate_indices(
        train_idx=train_idx,
        test_idx=test_idx,
        n_samples=len(loaded.y),
        expected_train=expected_count(row["num_instances_train"]),
        expected_test=expected_count(row["num_instances_test"]),
    )

    X = loaded.X.reset_index(drop=True)
    y = loaded.y.reset_index(drop=True)
    numeric_columns, categorical_columns = infer_feature_columns(X, loaded.categorical_indicator)
    if not numeric_columns and not categorical_columns:
        raise ValueError("Dataset has no feature columns after removing target")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    if numeric_columns:
        np.save(out_dir / "N_train.npy", numeric_values(X_train, numeric_columns))
        np.save(out_dir / "N_test.npy", numeric_values(X_test, numeric_columns))
    if categorical_columns:
        np.save(out_dir / "C_train.npy", categorical_values(X_train, categorical_columns))
        np.save(out_dir / "C_test.npy", categorical_values(X_test, categorical_columns))

    y_train_values = target_values(y_train, problem_type)
    y_test_values = target_values(y_test, problem_type)
    np.save(out_dir / "y_train.npy", y_train_values)
    np.save(out_dir / "y_test.npy", y_test_values)

    if problem_type == "regression":
        n_classes = -1
    else:
        n_classes = int(len(set(y_train_values.tolist()) | set(y_test_values.tolist())))

    info = {
        "name": out_name,
        "task_type": task_type,
        "problem_type": problem_type,
        "target_column": str(row["target_name"]),
        "objective_metric_name": str(row["eval_metric"]),
        "eval_metric": str(row["eval_metric"]),
        "n_num_features": len(numeric_columns),
        "n_cat_features": len(categorical_columns),
        "num_feature_intro": {column: column for column in numeric_columns},
        "cat_feature_intro": {column: column for column in categorical_columns},
        "train_size": int(len(train_idx)),
        "val_size": 0,
        "test_size": int(len(test_idx)),
        "n_classes": n_classes,
        "source": "OpenML",
        "tabarena_version": TABARENA_VERSION,
        "tabarena_dataset_name": dataset_name,
        "tabarena_repeat": repeat,
        "tabarena_fold": fold,
        "tabarena_split_index": split_index,
        "openml_task_id": str(row["task_id_str"]),
        "openml_dataset_id": loaded.openml_dataset_id,
        "openml_dataset_name": loaded.openml_dataset_name,
    }
    write_json(out_dir / "info.json", info)

    return {
        **base_record,
        "status": "ok",
        "error": None,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "n_num_features": len(numeric_columns),
        "n_cat_features": len(categorical_columns),
        "n_classes": n_classes,
    }


def convert_selected_splits(
    *,
    selected: pd.DataFrame,
    out_root: Path,
    raw_root: Path,
    overwrite: bool,
    openml_server: str,
) -> list[dict[str, Any]]:
    clear_unusable_socks_proxy()
    import openml

    openml_cache_dir = raw_root / "openml_cache"
    openml_cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.server = openml_server
    openml.config.set_root_cache_directory(str(openml_cache_dir))

    records: list[dict[str, Any]] = []
    groups = list(selected.groupby("task_id_str", sort=False))
    print(
        f"[tabarena-v0.1] selected_splits={len(selected)} unique_openml_tasks={len(groups)} "
        f"openml_cache={openml_cache_dir} openml_server={openml_server}",
        flush=True,
    )

    for task_pos, (task_id, task_rows) in enumerate(groups, start=1):
        dataset_names = ",".join(sorted(task_rows["dataset_name"].unique()))
        print(
            f"[tabarena-v0.1] task {task_pos}/{len(groups)} openml_task_id={task_id} "
            f"datasets={dataset_names} splits={len(task_rows)}",
            flush=True,
        )
        rows_to_convert: list[pd.Series] = []
        if overwrite:
            rows_to_convert = [row for _, row in task_rows.iterrows()]
        else:
            for _, row in task_rows.iterrows():
                out_dir = out_root / safe_dataset_dir_name(str(row["dataset_name"]), str(row["split_index"]))
                if output_is_complete(out_dir, row):
                    records.append(skipped_existing_record(row, out_root))
                else:
                    rows_to_convert.append(row)

        if not rows_to_convert:
            print(
                f"[tabarena-v0.1] task {task_pos}/{len(groups)} openml_task_id={task_id} "
                "all requested splits already exist; skipping OpenML download",
                flush=True,
            )
            continue

        try:
            loaded = load_openml_task(str(task_id))
        except Exception as exc:
            error = " ".join(f"{type(exc).__name__}: {exc}".split())
            print(
                f"[tabarena-v0.1] task {task_pos}/{len(groups)} openml_task_id={task_id} "
                f"failed to load from OpenML: {error}",
                flush=True,
            )
            for row in rows_to_convert:
                records.append(
                    {
                        "dataset_name": str(row["dataset_name"]),
                        "output_name": safe_dataset_dir_name(str(row["dataset_name"]), str(row["split_index"])),
                        "problem_type": str(row["problem_type"]),
                        "task_type": PROBLEM_TYPE_TO_TASK_TYPE.get(str(row["problem_type"]), ""),
                        "openml_task_id": str(row["task_id_str"]),
                        "tabarena_repeat": int(row["repeat"]),
                        "tabarena_fold": int(row["fold"]),
                        "tabarena_split_index": str(row["split_index"]),
                        "status": "fail",
                        "error": error,
                    }
                )
            continue

        for row in rows_to_convert:
            try:
                record = convert_split(
                    loaded=loaded,
                    row=row,
                    out_root=out_root,
                    overwrite=overwrite,
                )
            except Exception as exc:
                error = " ".join(f"{type(exc).__name__}: {exc}".split())
                print(
                    f"[tabarena-v0.1] split failed dataset={row['dataset_name']} "
                    f"split={row['split_index']} openml_task_id={row['task_id_str']}: {error}",
                    flush=True,
                )
                record = {
                    "dataset_name": str(row["dataset_name"]),
                    "output_name": safe_dataset_dir_name(str(row["dataset_name"]), str(row["split_index"])),
                    "problem_type": str(row["problem_type"]),
                    "task_type": PROBLEM_TYPE_TO_TASK_TYPE.get(str(row["problem_type"]), ""),
                    "openml_task_id": str(row["task_id_str"]),
                    "openml_dataset_id": loaded.openml_dataset_id,
                    "openml_dataset_name": loaded.openml_dataset_name,
                    "tabarena_repeat": int(row["repeat"]),
                    "tabarena_fold": int(row["fold"]),
                    "tabarena_split_index": str(row["split_index"]),
                    "status": "fail",
                    "error": error,
                }
            records.append(record)
    return records


def build_manifest(
    *,
    args: argparse.Namespace,
    metadata_path: Path,
    metadata_cached: bool,
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = pd.Series([record["status"] for record in records], dtype="object").value_counts().to_dict()
    selected_dataset_counts = selected[["dataset_name", "problem_type"]].drop_duplicates()["problem_type"].value_counts()
    selected_split_counts = selected["problem_type"].value_counts()
    return {
        "tabarena_version": TABARENA_VERSION,
        "metadata_csv": str(args.metadata_csv),
        "metadata_path": str(metadata_path),
        "metadata_was_cached": metadata_cached,
        "openml_server": str(args.openml_server),
        "out_root": str(Path(args.out_root).expanduser()),
        "raw_root": str(Path(args.raw_root).expanduser()),
        "openml_cache_dir": str(Path(args.raw_root).expanduser() / "openml_cache"),
        "split_mode": args.split_mode,
        "split_indices": args.split_indices,
        "datasets_filter": args.datasets,
        "problem_types_filter": args.problem_types,
        "expected_full_tabarena_v0p1": {
            "n_datasets": EXPECTED_FULL_DATASETS,
            "n_splits": EXPECTED_FULL_SPLITS,
        },
        "metadata_summary": {
            "n_rows": int(len(metadata)),
            "n_datasets": int(metadata["dataset_name"].nunique()),
            "n_unique_openml_tasks": int(metadata["task_id_str"].nunique()),
            "problem_type_dataset_counts": metadata[["dataset_name", "problem_type"]]
            .drop_duplicates()["problem_type"]
            .value_counts()
            .to_dict(),
            "problem_type_split_counts": metadata["problem_type"].value_counts().to_dict(),
        },
        "selected_summary": {
            "n_splits": int(len(selected)),
            "n_datasets": int(selected["dataset_name"].nunique()),
            "n_unique_openml_tasks": int(selected["task_id_str"].nunique()),
            "problem_type_dataset_counts": selected_dataset_counts.to_dict(),
            "problem_type_split_counts": selected_split_counts.to_dict(),
        },
        "output_summary": {
            "n_records": int(len(records)),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "n_outputs_ok": int(status_counts.get("ok", 0)),
            "n_outputs_skipped_existing": int(status_counts.get("skipped_existing", 0)),
            "n_outputs_failed": int(status_counts.get("fail", 0)),
        },
        "records": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TabArena v0.1 OpenML tasks and convert official repeat/fold splits to data178-style arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-root", default="data_tabarena_v0p1")
    parser.add_argument("--raw-root", default="data_tabarena_v0p1_raw")
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV_URL)
    parser.add_argument("--openml-server", default=DEFAULT_OPENML_SERVER)
    parser.add_argument("--split-mode", choices=["all", "lite"], default="all")
    parser.add_argument("--split-indices", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--problem-types",
        nargs="+",
        default=["binary", "multiclass", "regression"],
        choices=sorted(PROBLEM_TYPE_TO_TASK_TYPE),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw_root = Path(args.raw_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata, metadata_path, metadata_cached = load_metadata_table(args.metadata_csv, raw_root)
    selected = filter_metadata(
        metadata,
        datasets=args.datasets,
        problem_types=args.problem_types,
        split_mode=args.split_mode,
        split_indices=args.split_indices,
    )
    records = convert_selected_splits(
        selected=selected,
        out_root=out_root,
        raw_root=raw_root,
        overwrite=args.overwrite,
        openml_server=args.openml_server,
    )
    manifest = build_manifest(
        args=args,
        metadata_path=metadata_path,
        metadata_cached=metadata_cached,
        metadata=metadata,
        selected=selected,
        records=records,
    )
    manifest_path = out_root / "_tabarena_v0p1_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[tabarena-v0.1] wrote manifest: {manifest_path}", flush=True)
    print(
        "[tabarena-v0.1] "
        f"ok={manifest['output_summary']['n_outputs_ok']} "
        f"skipped_existing={manifest['output_summary']['n_outputs_skipped_existing']} "
        f"failed={manifest['output_summary']['n_outputs_failed']}",
        flush=True,
    )
    return 1 if manifest["output_summary"]["n_outputs_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
