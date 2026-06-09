from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tabarena_v0p1.py"
SPEC = importlib.util.spec_from_file_location("prepare_tabarena_v0p1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_tabarena_v0p1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_tabarena_v0p1
SPEC.loader.exec_module(prepare_tabarena_v0p1)


class FakeOpenMLTask:
    def __init__(self, splits: dict[tuple[int, int], tuple[list[int], list[int]]]):
        self._splits = splits

    def get_train_test_split_indices(self, *, fold: int, repeat: int):
        return self._splits[(repeat, fold)]


def make_loaded_task(*, X: pd.DataFrame, y: pd.Series, categorical_indicator: list[bool], splits):
    return prepare_tabarena_v0p1.LoadedOpenMLTask(
        task_id="363623",
        openml_dataset_id=46915,
        openml_dataset_name="fake-openml-dataset",
        target_name="target",
        X=prepare_tabarena_v0p1.normalize_feature_frame(X),
        y=y,
        categorical_indicator=categorical_indicator,
        task=FakeOpenMLTask(splits),
    )


def make_row(**overrides) -> pd.Series:
    values = {
        "dataset_name": "churn",
        "problem_type": "binary",
        "target_name": "target",
        "eval_metric": "roc_auc",
        "task_id_str": "363623",
        "repeat": 0,
        "fold": 0,
        "num_instances_train": 4,
        "num_instances_test": 2,
        "split_index": "r0f0",
    }
    values.update(overrides)
    return pd.Series(values)


def test_convert_binary_split_writes_data178_style_arrays_and_info(tmp_path: Path):
    X = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", None, "a", "b", "c"],
            "num_as_string": ["1", "2", "3", "4", "5", "6"],
        }
    )
    y = pd.Series(["no", "yes", "no", "yes", "no", "yes"], name="target")
    loaded = make_loaded_task(
        X=X,
        y=y,
        categorical_indicator=[False, True, False],
        splits={(0, 0): ([0, 1, 2, 3], [4, 5])},
    )

    record = prepare_tabarena_v0p1.convert_split(
        loaded=loaded,
        row=make_row(),
        out_root=tmp_path,
        overwrite=True,
    )

    out_dir = tmp_path / "churn__r0f0"
    assert record["status"] == "ok"
    assert out_dir.exists()
    assert np.load(out_dir / "N_train.npy").shape == (4, 2)
    assert np.load(out_dir / "N_test.npy").shape == (2, 2)
    c_train = np.load(out_dir / "C_train.npy", allow_pickle=True)
    assert c_train.shape == (4, 1)
    assert c_train[2, 0] == prepare_tabarena_v0p1.CATEGORICAL_MISSING_TOKEN
    assert np.load(out_dir / "y_train.npy", allow_pickle=True).tolist() == ["no", "yes", "no", "yes"]

    info = prepare_tabarena_v0p1.json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
    assert info["name"] == "churn__r0f0"
    assert info["task_type"] == "binclass"
    assert info["problem_type"] == "binary"
    assert info["tabarena_dataset_name"] == "churn"
    assert info["tabarena_split_index"] == "r0f0"
    assert info["train_size"] == 4
    assert info["test_size"] == 2
    assert info["n_num_features"] == 2
    assert info["n_cat_features"] == 1


def test_convert_complete_split_skips_without_overwrite(tmp_path: Path):
    X = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
    y = pd.Series(["no", "yes", "no"], name="target")
    loaded = make_loaded_task(
        X=X,
        y=y,
        categorical_indicator=[False, True],
        splits={(0, 0): ([0, 1], [2])},
    )
    row = make_row(num_instances_train=2, num_instances_test=1)

    prepare_tabarena_v0p1.convert_split(
        loaded=loaded,
        row=row,
        out_root=tmp_path,
        overwrite=True,
    )
    record = prepare_tabarena_v0p1.convert_split(
        loaded=loaded,
        row=row,
        out_root=tmp_path,
        overwrite=False,
    )

    assert record["status"] == "skipped_existing"
    assert record["error"] is None


def test_convert_selected_splits_skips_existing_before_openml_download(tmp_path: Path, monkeypatch):
    X = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
    y = pd.Series(["no", "yes", "no"], name="target")
    loaded = make_loaded_task(
        X=X,
        y=y,
        categorical_indicator=[False, True],
        splits={(0, 0): ([0, 1], [2])},
    )
    row = make_row(num_instances_train=2, num_instances_test=1)
    prepare_tabarena_v0p1.convert_split(
        loaded=loaded,
        row=row,
        out_root=tmp_path,
        overwrite=True,
    )

    def fail_if_called(task_id: str):
        raise AssertionError(f"OpenML download should not be called for {task_id}")

    monkeypatch.setattr(prepare_tabarena_v0p1, "load_openml_task", fail_if_called)
    records = prepare_tabarena_v0p1.convert_selected_splits(
        selected=pd.DataFrame([row]),
        out_root=tmp_path,
        raw_root=tmp_path / "raw",
        overwrite=False,
        openml_server=prepare_tabarena_v0p1.DEFAULT_OPENML_SERVER,
    )

    assert records[0]["status"] == "skipped_existing"
    assert records[0]["train_size"] == 2
    assert records[0]["test_size"] == 1


def test_convert_regression_split_writes_float_targets(tmp_path: Path):
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": ["lo", "hi", "lo"]})
    y = pd.Series([1.5, 2.5, 3.5], name="target")
    loaded = make_loaded_task(
        X=X,
        y=y,
        categorical_indicator=[False, True],
        splits={(0, 0): ([0, 2], [1])},
    )
    row = make_row(
        dataset_name="airfoil_self_noise",
        problem_type="regression",
        eval_metric="rmse",
        task_id_str="363612",
        num_instances_train=2,
        num_instances_test=1,
    )

    record = prepare_tabarena_v0p1.convert_split(
        loaded=loaded,
        row=row,
        out_root=tmp_path,
        overwrite=True,
    )

    out_dir = tmp_path / "airfoil_self_noise__r0f0"
    y_train = np.load(out_dir / "y_train.npy")
    info = prepare_tabarena_v0p1.json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert y_train.dtype.kind == "f"
    assert y_train.tolist() == [1.5, 3.5]
    assert info["task_type"] == "regression"
    assert info["problem_type"] == "regression"
    assert info["n_classes"] == -1


def test_filter_metadata_supports_lite_and_explicit_split_indices():
    frame = pd.DataFrame(
        [
            make_row(dataset_name="churn", split_index="r0f0", repeat=0, fold=0),
            make_row(dataset_name="churn", split_index="r0f1", repeat=0, fold=1),
            make_row(
                dataset_name="airfoil_self_noise",
                problem_type="regression",
                split_index="r0f0",
                repeat=0,
                fold=0,
            ),
        ]
    )

    lite = prepare_tabarena_v0p1.filter_metadata(
        frame,
        datasets=None,
        problem_types=["binary", "regression"],
        split_mode="lite",
        split_indices=None,
    )
    explicit = prepare_tabarena_v0p1.filter_metadata(
        frame,
        datasets=["churn"],
        problem_types=["binary"],
        split_mode="all",
        split_indices=["r0f1"],
    )

    assert set(lite["split_index"]) == {"r0f0"}
    assert set(lite["dataset_name"]) == {"airfoil_self_noise", "churn"}
    assert explicit["dataset_name"].tolist() == ["churn"]
    assert explicit["split_index"].tolist() == ["r0f1"]
