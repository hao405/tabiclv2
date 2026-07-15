#!/usr/bin/env python3
"""Export TabICLv2 Graph-SCM priors as benchmark-format classification tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PRIOR_SRC_ROOT = Path(os.environ.get("TABICLV2_PRIOR_SRC_ROOT", SRC_ROOT)).expanduser().resolve()
if str(PRIOR_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PRIOR_SRC_ROOT))
if PRIOR_SRC_ROOT != SRC_ROOT:
    isolated_tabicl = types.ModuleType("tabicl")
    isolated_tabicl.__path__ = [str(PRIOR_SRC_ROOT / "tabicl")]
    isolated_tabicl.__package__ = "tabicl"
    sys.modules["tabicl"] = isolated_tabicl

# ``tabicl.prior._dataset`` imports TreeSCM eagerly even when only graph_scm is
# requested. Keep this graph-only exporter usable in environments without the
# optional xgboost dependency; attempting to instantiate the unused XGB path
# still fails explicitly.
try:
    import xgboost as _xgboost  # noqa: F401
except ModuleNotFoundError:
    xgboost_stub = types.ModuleType("xgboost")

    class _UnavailableXGBRegressor:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "xgboost is required for TreeSCM but not for this graph_scm exporter"
            )

    xgboost_stub.XGBRegressor = _UnavailableXGBRegressor
    sys.modules["xgboost"] = xgboost_stub

from tabicl.prior import PriorDataset  # noqa: E402
from tabicl.prior.graph_lib._config import PriorConfig  # noqa: E402


DEFAULT_STAGE_COUNTS = (171, 171, 170)
MAX_GENERATION_ATTEMPTS = 20
MIN_TRAIN_CLASS_COUNT = 10
MIN_TEST_CLASS_COUNT = 2
REQUIRED_ARRAYS = ("N_train.npy", "N_test.npy", "y_train.npy", "y_test.npy")
MANIFEST_COLUMNS = (
    "dataset_name",
    "dataset_dir",
    "stage",
    "stage_id",
    "stage_index",
    "group_id",
    "group_position",
    "base_seed",
    "numpy_seed",
    "torch_seed",
    "generation_attempt",
    "prior_type",
    "n_total",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "min_seq_len",
    "max_seq_len",
    "log_seq_len",
    "min_train_size",
    "max_train_size",
    "batch_size_per_gp",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    stage_id: int
    count: int
    min_seq_len: int | None
    max_seq_len: int
    log_seq_len: bool
    min_train_size: float
    max_train_size: float
    batch_size_per_gp: int


@dataclass(frozen=True)
class GroupJob:
    data_root: str
    base_seed: int
    stage: StageSpec
    group_id: int
    stage_indices: tuple[int, ...]


def stage_specs(counts: Sequence[int]) -> tuple[StageSpec, ...]:
    if len(counts) != 3:
        raise ValueError("--stage-counts requires exactly three integers")
    if any(int(value) < 0 for value in counts):
        raise ValueError("--stage-counts values must be non-negative")
    return (
        StageSpec("stage1", 1, int(counts[0]), None, 1024, False, 0.3, 0.9, 4),
        StageSpec("stage2", 2, int(counts[1]), 400, 10240, True, 0.79, 0.81, 1),
        StageSpec("stage3", 3, int(counts[2]), 400, 60000, True, 0.79, 0.81, 1),
    )


def dataset_name(stage: StageSpec, stage_index: int, base_seed: int) -> str:
    return f"{stage.name}_graph_scm_{stage_index:04d}_seed{base_seed}"


def build_group_jobs(data_root: Path, counts: Sequence[int], base_seed: int) -> list[GroupJob]:
    jobs: list[GroupJob] = []
    for stage in stage_specs(counts):
        group_size = stage.batch_size_per_gp
        for group_id, start in enumerate(range(0, stage.count, group_size)):
            stop = min(start + group_size, stage.count)
            jobs.append(
                GroupJob(
                    data_root=str(data_root),
                    base_seed=int(base_seed),
                    stage=stage,
                    group_id=group_id,
                    stage_indices=tuple(range(start, stop)),
                )
            )
    return jobs


def derived_seeds(base_seed: int, stage_id: int, group_id: int, attempt: int) -> tuple[int, int]:
    state = np.random.SeedSequence(
        [int(base_seed), int(stage_id), int(group_id), int(attempt)]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]), int(state[1])


def prior_config() -> PriorConfig:
    return PriorConfig(
        add_gaussian_noise=False,
        allow_act_warping=False,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        min_n_nodes=2,
        max_n_nodes=32,
        cauchy_dag_offset=0.0,
    )


def build_prior(stage: StageSpec, batch_size: int) -> PriorDataset:
    return PriorDataset(
        regression=False,
        batch_size=batch_size,
        batch_size_per_gp=stage.batch_size_per_gp,
        min_features=1,
        max_features=100,
        max_classes=10,
        min_seq_len=stage.min_seq_len,
        max_seq_len=stage.max_seq_len,
        log_seq_len=stage.log_seq_len,
        log_n_features=False,
        seq_len_per_gp=True,
        min_train_size=stage.min_train_size,
        max_train_size=stage.max_train_size,
        replay_small=False,
        prior_type="graph_scm",
        config=prior_config(),
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )


def relabel_contiguous(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([y_train.reshape(-1), y_test.reshape(-1)])
    classes = np.unique(combined)
    mapping = {value.item() if hasattr(value, "item") else value: idx for idx, value in enumerate(classes)}
    mapped_train = np.asarray([mapping[value.item() if hasattr(value, "item") else value] for value in y_train], dtype=np.int64)
    mapped_test = np.asarray([mapping[value.item() if hasattr(value, "item") else value] for value in y_test], dtype=np.int64)
    return mapped_train, mapped_test


def validate_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("label arrays must be one-dimensional")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("feature/label row counts do not match")
    if not len(y_train) or not len(y_test):
        raise ValueError("train and test splits must both be non-empty")
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[1] < 1:
        raise ValueError("train/test feature counts must match and be positive")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("features contain NaN or infinite values")
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("labels contain NaN or infinite values")
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    all_classes = np.unique(np.concatenate([y_train, y_test]))
    if len(all_classes) < 2:
        raise ValueError("classification tasks must contain at least two classes")
    if not np.array_equal(all_classes, np.arange(len(all_classes))):
        raise ValueError("labels must be contiguous integers starting at zero")
    if not np.array_equal(train_classes, all_classes) or not np.array_equal(test_classes, all_classes):
        raise ValueError("every class must occur in both train and test splits")
    train_counts = np.unique(y_train, return_counts=True)[1]
    test_counts = np.unique(y_test, return_counts=True)[1]
    if int(train_counts.min()) < MIN_TRAIN_CLASS_COUNT:
        raise ValueError(
            f"every class needs at least {MIN_TRAIN_CLASS_COUNT} training rows for nested TTT splits"
        )
    if int(test_counts.min()) < MIN_TEST_CLASS_COUNT:
        raise ValueError(
            f"every class needs at least {MIN_TEST_CLASS_COUNT} test rows for stable evaluation"
        )


def row_from_info(dataset_dir: Path, info: dict[str, object]) -> dict[str, object]:
    return {column: info.get(column, str(dataset_dir) if column == "dataset_dir" else None) for column in MANIFEST_COLUMNS}


def validate_dataset_dir(dataset_dir: Path) -> dict[str, object]:
    missing = [name for name in (*REQUIRED_ARRAYS, "info.json") if not (dataset_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing files: {', '.join(missing)}")
    arrays = [np.load(dataset_dir / name, allow_pickle=False) for name in REQUIRED_ARRAYS]
    validate_arrays(*arrays)
    info = json.loads((dataset_dir / "info.json").read_text(encoding="utf-8"))
    if info.get("dataset_name") != dataset_dir.name:
        raise ValueError("info.json dataset_name does not match directory name")
    expected = {
        "n_train": len(arrays[2]),
        "n_test": len(arrays[3]),
        "n_total": len(arrays[2]) + len(arrays[3]),
        "n_features": arrays[0].shape[1],
        "n_classes": len(np.unique(np.concatenate([arrays[2], arrays[3]]))),
    }
    for key, value in expected.items():
        if int(info.get(key, -1)) != int(value):
            raise ValueError(f"info.json {key} mismatch")
    expected_task_type = "binclass" if expected["n_classes"] == 2 else "multiclass"
    if info.get("task_type") != expected_task_type:
        raise ValueError(f"info.json task_type must be {expected_task_type!r}")
    return row_from_info(dataset_dir, info)


def write_dataset_atomic(
    data_root: Path,
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    info: dict[str, object],
) -> dict[str, object]:
    validate_arrays(X_train, X_test, y_train, y_test)
    data_root.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{data_root.name}.{name}.", dir=data_root.parent))
    final_dir = data_root / name
    try:
        np.save(temp_dir / "N_train.npy", X_train.astype(np.float32, copy=False))
        np.save(temp_dir / "N_test.npy", X_test.astype(np.float32, copy=False))
        np.save(temp_dir / "y_train.npy", y_train.astype(np.int64, copy=False))
        np.save(temp_dir / "y_test.npy", y_test.astype(np.int64, copy=False))
        (temp_dir / "info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temp_dir, final_dir)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    return validate_dataset_dir(final_dir)


def generate_group_job(job: GroupJob) -> list[dict[str, object]]:
    data_root = Path(job.data_root)
    last_error: Exception | None = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        numpy_seed, torch_seed = derived_seeds(
            job.base_seed, job.stage.stage_id, job.group_id, attempt
        )
        random.seed(numpy_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(torch_seed)
        try:
            prior = build_prior(job.stage, len(job.stage_indices))
            X, y, d, seq_lens, train_sizes = prior.get_batch()
            prepared: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
            for group_position, stage_index in enumerate(job.stage_indices):
                n_features = int(d[group_position].item())
                n_total = int(seq_lens[group_position].item())
                n_train = int(train_sizes[group_position].item())
                values = X[group_position].detach().cpu().numpy()[:n_total, :n_features]
                labels = y[group_position].detach().cpu().numpy()[:n_total].reshape(-1)
                X_train = np.asarray(values[:n_train], dtype=np.float32)
                X_test = np.asarray(values[n_train:], dtype=np.float32)
                y_train, y_test = relabel_contiguous(labels[:n_train], labels[n_train:])
                validate_arrays(X_train, X_test, y_train, y_test)
                name = dataset_name(job.stage, stage_index, job.base_seed)
                n_classes = len(np.unique(np.concatenate([y_train, y_test])))
                info = {
                    "name": name,
                    "dataset_name": name,
                    "source": "tabiclv2_graph_scm_prior",
                    "task_type": "binclass" if n_classes == 2 else "multiclass",
                    "stage": job.stage.name,
                    "stage_id": job.stage.stage_id,
                    "stage_index": stage_index,
                    "group_id": job.group_id,
                    "group_position": group_position,
                    "base_seed": job.base_seed,
                    "numpy_seed": numpy_seed,
                    "torch_seed": torch_seed,
                    "generation_attempt": attempt,
                    "prior_type": "graph_scm",
                    "n_total": n_total,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_features": n_features,
                    "n_num_features": n_features,
                    "n_cat_features": 0,
                    "n_classes": n_classes,
                    "classes": list(range(n_classes)),
                    "min_seq_len": job.stage.min_seq_len,
                    "max_seq_len": job.stage.max_seq_len,
                    "log_seq_len": job.stage.log_seq_len,
                    "min_train_size": job.stage.min_train_size,
                    "max_train_size": job.stage.max_train_size,
                    "batch_size_per_gp": job.stage.batch_size_per_gp,
                    "graph_config": asdict(prior_config()),
                }
                info["dataset_dir"] = str(data_root / name)
                prepared.append((name, X_train, X_test, y_train, y_test, info))
            return [write_dataset_atomic(data_root, *item) for item in prepared]
        except Exception as exc:  # retry the whole shared-prior group deterministically
            last_error = exc
    raise RuntimeError(
        f"failed to generate {job.stage.name} group {job.group_id} after "
        f"{MAX_GENERATION_ATTEMPTS} attempts: {last_error}"
    )


def reusable_rows(job: GroupJob) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    for stage_index in job.stage_indices:
        path = Path(job.data_root) / dataset_name(job.stage, stage_index, job.base_seed)
        try:
            rows.append(validate_dataset_dir(path))
        except Exception:
            return None
    return rows


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["stage_id"]), int(row["stage_index"])))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(temp_path, path)


def generate(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    specs = stage_specs(args.stage_counts)
    target_count = sum(stage.count for stage in specs)
    if target_count < 1:
        raise ValueError("at least one dataset must be requested")
    if args.generation_workers < 1:
        raise ValueError("--generation-workers must be >= 1")
    if data_root.exists() and any(data_root.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty data root requires --resume: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)

    jobs = build_group_jobs(data_root, args.stage_counts, args.seed)
    rows: list[dict[str, object]] = []
    pending: list[GroupJob] = []
    for job in jobs:
        existing = reusable_rows(job) if args.resume else None
        if existing is None:
            pending.append(job)
        else:
            rows.extend(existing)

    print(f"target_datasets: {target_count}", flush=True)
    print(f"reused_datasets: {len(rows)}", flush=True)
    print(f"pending_groups: {len(pending)}", flush=True)
    if args.generation_workers == 1:
        for index, job in enumerate(pending, start=1):
            rows.extend(generate_group_job(job))
            print(f"generated_group: {index}/{len(pending)} {job.stage.name}/{job.group_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.generation_workers) as executor:
            futures = {executor.submit(generate_group_job, job): job for job in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                rows.extend(future.result())
                print(f"generated_group: {index}/{len(pending)} {job.stage.name}/{job.group_id}", flush=True)

    if len(rows) != target_count:
        raise RuntimeError(f"generated/reused {len(rows)} datasets; expected {target_count}")
    names = [str(row["dataset_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate dataset names in generated suite")

    manifest_path = data_root / "prior_manifest.csv"
    write_manifest(manifest_path, rows)
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "data_root": str(data_root),
        "seed": args.seed,
        "generation_workers": args.generation_workers,
        "stage_specs": [asdict(spec) for spec in specs],
        "target_datasets": target_count,
        "prior_type": "graph_scm",
        "graph_config": asdict(prior_config()),
    }
    (data_root / "generation_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"generated_datasets: {len(rows)}", flush=True)
    print(f"prior_manifest: {manifest_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the TabICLv2 three-stage Graph-SCM prior as benchmark datasets."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stage-counts", nargs=3, type=int, default=DEFAULT_STAGE_COUNTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(generate(build_parser().parse_args()))
