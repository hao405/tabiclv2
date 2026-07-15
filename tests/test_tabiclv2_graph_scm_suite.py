from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_tabiclv2_graph_scm_suite.py"
SUMMARY_PATH = REPO_ROOT / "scripts" / "summarize_tabiclv2_graph_scm_matrix.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return load_module(GENERATOR_PATH, "tabiclv2_graph_scm_generator_for_tests")


@pytest.fixture(scope="module")
def summarizer():
    return load_module(SUMMARY_PATH, "tabiclv2_graph_scm_summary_for_tests")


def test_default_stage_contract_and_grouping(generator, tmp_path: Path):
    specs = generator.stage_specs((171, 171, 170))
    assert [spec.name for spec in specs] == ["stage1", "stage2", "stage3"]
    assert [spec.count for spec in specs] == [171, 171, 170]
    assert sum(spec.count for spec in specs) == 512
    assert specs[0].min_seq_len is None and specs[0].max_seq_len == 1024
    assert specs[0].batch_size_per_gp == 4
    assert (specs[1].min_seq_len, specs[1].max_seq_len, specs[1].log_seq_len) == (
        400,
        10240,
        True,
    )
    assert (specs[2].min_seq_len, specs[2].max_seq_len, specs[2].log_seq_len) == (
        400,
        60000,
        True,
    )

    jobs = generator.build_group_jobs(tmp_path, (171, 171, 170), 42)
    stage1_jobs = [job for job in jobs if job.stage.name == "stage1"]
    stage2_jobs = [job for job in jobs if job.stage.name == "stage2"]
    stage3_jobs = [job for job in jobs if job.stage.name == "stage3"]
    assert len(stage1_jobs) == 43
    assert [len(job.stage_indices) for job in stage1_jobs[:-1]] == [4] * 42
    assert len(stage1_jobs[-1].stage_indices) == 3
    assert len(stage2_jobs) == 171
    assert len(stage3_jobs) == 170


def test_group_seeds_are_deterministic_and_stage_specific(generator):
    left = generator.derived_seeds(42, 1, 0, 0)
    right = generator.derived_seeds(42, 1, 0, 0)
    assert left == right
    assert left != generator.derived_seeds(42, 2, 0, 0)
    assert left != generator.derived_seeds(42, 1, 1, 0)
    assert left != generator.derived_seeds(42, 1, 0, 1)


def valid_info(generator, root: Path, name: str) -> dict[str, object]:
    stage = generator.stage_specs((1, 0, 0))[0]
    np_seed, torch_seed = generator.derived_seeds(42, 1, 0, 0)
    return {
        "name": name,
        "dataset_name": name,
        "dataset_dir": str(root / name),
        "source": "tabiclv2_graph_scm_prior",
        "task_type": "binclass",
        "stage": "stage1",
        "stage_id": 1,
        "stage_index": 0,
        "group_id": 0,
        "group_position": 0,
        "base_seed": 42,
        "numpy_seed": np_seed,
        "torch_seed": torch_seed,
        "generation_attempt": 0,
        "prior_type": "graph_scm",
        "n_total": 24,
        "n_train": 20,
        "n_test": 4,
        "n_features": 3,
        "n_num_features": 3,
        "n_cat_features": 0,
        "n_classes": 2,
        "classes": [0, 1],
        "min_seq_len": stage.min_seq_len,
        "max_seq_len": stage.max_seq_len,
        "log_seq_len": stage.log_seq_len,
        "min_train_size": stage.min_train_size,
        "max_train_size": stage.max_train_size,
        "batch_size_per_gp": stage.batch_size_per_gp,
        "graph_config": {},
    }


def test_atomic_benchmark_export_and_corruption_detection(generator, tmp_path: Path):
    name = "stage1_graph_scm_0000_seed42"
    X_train = np.arange(60, dtype=np.float32).reshape(20, 3)
    X_test = np.arange(60, 72, dtype=np.float32).reshape(4, 3)
    y_train = np.array([0, 1] * 10, dtype=np.int64)
    y_test = np.array([1, 0, 1, 0], dtype=np.int64)
    row = generator.write_dataset_atomic(
        tmp_path,
        name,
        X_train,
        X_test,
        y_train,
        y_test,
        valid_info(generator, tmp_path, name),
    )
    assert row["dataset_name"] == name
    for filename in (*generator.REQUIRED_ARRAYS, "info.json"):
        assert (tmp_path / name / filename).is_file()
    np.testing.assert_array_equal(np.load(tmp_path / name / "N_train.npy"), X_train)

    np.save(tmp_path / name / "N_test.npy", np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="feature/label|feature counts"):
        generator.validate_dataset_dir(tmp_path / name)
    job = generator.build_group_jobs(tmp_path, (1, 0, 0), 42)[0]
    assert generator.reusable_rows(job) is None


def write_result(
    matrix_root: Path,
    model: str,
    method: str,
    rows: list[dict[str, object]],
) -> None:
    result_dir = matrix_root / model / method / "seed42"
    result_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(result_dir / "all_classification_results.csv", index=False)


def test_stage_summary_uses_only_shared_ok_rows(summarizer, tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {"dataset_name": "s1", "stage": "stage1", "stage_id": 1, "stage_index": 0},
            {"dataset_name": "s2", "stage": "stage2", "stage_id": 2, "stage_index": 0},
            {"dataset_name": "s3", "stage": "stage3", "stage_id": 3, "stage_index": 0},
        ]
    )
    manifest_path = tmp_path / "prior_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    matrix_root = tmp_path / "matrix"

    write_result(
        matrix_root,
        "tabicl-v2",
        "infer",
        [
            {"dataset_name": "s1", "status": "ok", "accuracy": 0.5, "balanced_accuracy": 0.5},
            {"dataset_name": "s2", "status": "ok", "accuracy": 0.6, "balanced_accuracy": 0.6},
            {"dataset_name": "s3", "status": "ok", "accuracy": 0.7, "balanced_accuracy": 0.7},
        ],
    )
    write_result(
        matrix_root,
        "tabicl-v2",
        "ft",
        [
            {"dataset_name": "s1", "status": "ok", "accuracy": 0.6, "balanced_accuracy": 0.6, "ttt_applied": True, "ttt_c_selection": "random"},
            {"dataset_name": "s2", "status": "fail", "accuracy": np.nan, "balanced_accuracy": np.nan},
            {"dataset_name": "s3", "status": "ok", "accuracy": 0.65, "balanced_accuracy": 0.65, "ttt_applied": True, "ttt_c_selection": "random"},
        ],
    )
    write_result(
        matrix_root,
        "tabicl-v2",
        "faware_ft",
        [
            {"dataset_name": "s1", "status": "ok", "accuracy": 0.65, "balanced_accuracy": 0.65, "ttt_applied": True, "ttt_c_selection": "f_mmd"},
            {"dataset_name": "s2", "status": "ok", "accuracy": 0.7, "balanced_accuracy": 0.7, "ttt_applied": True, "ttt_c_selection": "f_mmd"},
            {"dataset_name": "s3", "status": "ok", "accuracy": 0.7, "balanced_accuracy": 0.7, "ttt_applied": True, "ttt_c_selection": "f_mmd"},
        ],
    )

    joined = summarizer.load_joined(manifest_path, matrix_root, 42)
    coverage = summarizer.coverage_summary(joined)
    detail, summary = summarizer.paired_method_deltas(joined)
    ft_accuracy = summary[
        summary["model"].eq("tabicl-v2")
        & summary["comparison"].eq("ft_minus_infer")
        & summary["scope"].eq("overall")
        & summary["metric"].eq("accuracy")
    ].iloc[0]
    assert int(ft_accuracy["paired_rows"]) == 2
    assert int(ft_accuracy["wins"]) == 1
    assert int(ft_accuracy["losses"]) == 1
    assert int(ft_accuracy["ties"]) == 0

    faware_accuracy = summary[
        summary["model"].eq("tabicl-v2")
        & summary["comparison"].eq("faware_ft_minus_ft")
        & summary["scope"].eq("overall")
        & summary["metric"].eq("accuracy")
    ].iloc[0]
    assert int(faware_accuracy["paired_rows"]) == 2
    assert int(faware_accuracy["wins"]) == 2
    assert set(detail[detail["comparison"].eq("ft_minus_infer")]["dataset_name"]) == {
        "s1",
        "s3",
    }
    missing_cell = coverage[
        coverage["model"].eq("tabpfn-v3")
        & coverage["method"].eq("infer")
        & coverage["scope"].eq("overall")
    ].iloc[0]
    assert int(missing_cell["missing_rows"]) == 3
