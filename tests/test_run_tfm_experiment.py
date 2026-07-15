from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_tfm_experiment.py"


def load_launcher_module():
    module_name = "run_tfm_experiment_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launcher():
    return load_launcher_module()


def parse_args(launcher, *values: str) -> argparse.Namespace:
    return launcher.build_arg_parser().parse_args(list(values))


def value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("model", ["tabicl-v1.1", "tabicl-v2"])
@pytest.mark.parametrize("method", ["infer", "ft", "faware_ft"])
def test_tabicl_routes(launcher, model: str, method: str, tmp_path: Path):
    args = parse_args(
        launcher,
        "--model",
        model,
        "--method",
        method,
        "--out-dir",
        str(tmp_path / model / method),
        "--run-id",
        "test",
    )
    spec = launcher.build_launch_spec(args, repo_root=REPO_ROOT)

    expected_runner = "benchmark.py" if method == "infer" else "1C_Chunk_FT_Faware_C.py"
    assert Path(spec.runner).name == expected_runner
    assert spec.model_family == "tabicl"
    assert value_after(spec.command, "--model-path").endswith(
        launcher.TABICL_CHECKPOINTS[model]
    )
    if method == "infer":
        assert spec.selection is None
        assert "--gpus" in spec.command
        assert "--ttt-c-selection" not in spec.command
    elif method == "ft":
        assert spec.selection == "random"
        assert spec.source == "none"
        assert value_after(spec.command, "--ttt-c-selection") == "random"
        assert value_after(spec.command, "--ttt-c-source") == "none"
        assert "--gpu-groups" in spec.command
    else:
        assert spec.selection == "f_mmd"
        assert spec.source == "test"
        assert value_after(spec.command, "--ttt-c-selection") == "f_mmd"
        assert value_after(spec.command, "--ttt-c-source") == "test"
        assert value_after(spec.command, "--ttt-c-metric") == "standardized_l2"


@pytest.mark.parametrize("model", ["tabpfn-v2", "tabpfn-v3"])
@pytest.mark.parametrize("method", ["infer", "ft", "faware_ft"])
def test_tabpfn_routes(launcher, model: str, method: str, tmp_path: Path):
    args = parse_args(
        launcher,
        "--model",
        model,
        "--method",
        method,
        "--out-dir",
        str(tmp_path / model / method),
        "--run-id",
        "test",
    )
    spec = launcher.build_launch_spec(args, repo_root=REPO_ROOT)

    expected_runner = "benchmark_infer.py" if method == "infer" else "Tabpfn_1c_ttt_faware_c.py"
    assert Path(spec.runner).name == expected_runner
    assert spec.model_family == "tabpfn"
    assert value_after(spec.command, "--model-version") == model.removeprefix("tabpfn-")
    assert "--gpus" in spec.command
    if model == "tabpfn-v2":
        assert value_after(spec.command, "--model-path").endswith(
            "tabpfn-v2-classifier-v2_default.ckpt"
        )
    else:
        assert "--v3-binary-model-path" in spec.command
        assert "--v3-multiclass-model-path" in spec.command
    if method == "infer":
        assert spec.selection is None
        assert "--ttt-c-selection" not in spec.command
    elif method == "ft":
        assert spec.selection == "random"
        assert spec.source is None
        assert value_after(spec.command, "--ttt-c-selection") == "random"
        assert value_after(spec.command, "--ttt-eval-metric") == "acc"
    else:
        assert spec.selection == "f_mmd"
        assert value_after(spec.command, "--ttt-c-selection") == "f_mmd"
        assert value_after(spec.command, "--ttt-c-metric") == "standardized_l2"
        assert value_after(spec.command, "--ttt-eval-metric") == "acc"


def test_devices_are_translated_and_worker_count_is_checked(launcher, tmp_path: Path):
    args = parse_args(
        launcher,
        "--model",
        "tabicl-v2",
        "--method",
        "ft",
        "--workers",
        "2",
        "--devices",
        "2,3",
        "--out-dir",
        str(tmp_path),
    )
    spec = launcher.build_launch_spec(args, repo_root=REPO_ROOT)
    assert value_after(spec.command, "--gpu-groups") == "2;3"

    mismatch = parse_args(
        launcher,
        "--model",
        "tabpfn-v2",
        "--method",
        "infer",
        "--workers",
        "1",
        "--devices",
        "2,3",
    )
    with pytest.raises(ValueError, match="exactly one GPU id per worker"):
        launcher.build_launch_spec(mismatch, repo_root=REPO_ROOT)


def test_model_native_metric_and_passthrough(launcher, tmp_path: Path):
    args = parse_args(
        launcher,
        "--model",
        "tabpfn-v2",
        "--method",
        "faware_ft",
        "--ttt-c-metric",
        "model_native_l2",
        "--out-dir",
        str(tmp_path),
    )
    spec = launcher.build_launch_spec(
        args,
        repo_root=REPO_ROOT,
        passthrough_args=["--many-class", "off"],
    )
    assert value_after(spec.command, "--ttt-c-metric") == "raw_l2"
    assert spec.command[-2:] == ["--many-class", "off"]


def test_automatic_output_path(launcher):
    args = parse_args(
        launcher,
        "--model",
        "tabicl-v1.1",
        "--method",
        "ft",
        "--run-id",
        "unit-test",
        "--random-state",
        "7",
    )
    spec = launcher.build_launch_spec(args, repo_root=REPO_ROOT)
    assert Path(spec.out_dir) == (
        REPO_ROOT
        / "results"
        / "managed_experiments"
        / "unit-test"
        / "tabicl-v1.1"
        / "ft"
        / "seed7"
    )


def test_invalid_query_and_reserve_ratio_is_rejected(launcher):
    args = parse_args(
        launcher,
        "--model",
        "tabicl-v2",
        "--method",
        "faware_ft",
        "--ttt-query-ratio",
        "0.8",
        "--ttt-c-reserve-ratio",
        "0.2",
    )
    with pytest.raises(ValueError, match="must be < 1"):
        launcher.build_launch_spec(args, repo_root=REPO_ROOT)


def test_real_launch_writes_manifest_and_propagates_exit_code(
    launcher, monkeypatch, tmp_path: Path
):
    out_dir = tmp_path / "successful_run"
    seen = {}

    def fake_run(command, *, cwd, check):
        seen["command"] = command
        seen["cwd"] = cwd
        seen["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    exit_code = launcher.run(
        [
            "--model",
            "tabicl-v2",
            "--method",
            "infer",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "manifest-test",
        ]
    )

    assert exit_code == 0
    assert seen["cwd"] == REPO_ROOT
    assert seen["check"] is False
    manifest = json.loads((out_dir / "manager_manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["exit_code"] == 0
    assert manifest["model"] == "tabicl-v2"
    assert manifest["method"] == "infer"
    assert manifest["command"] == seen["command"]


def test_dry_run_does_not_create_output(launcher, tmp_path: Path):
    out_dir = tmp_path / "dry_run"
    exit_code = launcher.run(
        [
            "--model",
            "tabpfn-v2",
            "--method",
            "ft",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "dry-run-test",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not out_dir.exists()


def test_launch_mode_requires_scalar_pair_or_matrix(launcher):
    missing_method = parse_args(launcher, "--model", "tabicl-v2")
    with pytest.raises(ValueError, match="requires both"):
        launcher.validate_launch_mode(missing_method)

    mixed = parse_args(
        launcher,
        "--matrix",
        "--model",
        "tabicl-v2",
        "--method",
        "infer",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        launcher.validate_launch_mode(mixed)


def test_build_matrix_specs_preserves_fixed_axes_and_shared_run_id(launcher, tmp_path: Path):
    args = parse_args(
        launcher,
        "--matrix",
        "--data-root",
        "data184",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "matrix-test",
    )
    launcher.validate_launch_mode(args)
    specs = launcher.build_matrix_specs(args, repo_root=REPO_ROOT)

    assert [(spec.model, spec.method) for spec in specs] == [
        (model, method)
        for model in launcher.MODEL_CHOICES
        for method in launcher.METHOD_CHOICES
    ]
    assert len(specs) == 12
    assert {spec.run_id for spec in specs} == {"matrix-test"}
    assert len({spec.out_dir for spec in specs}) == 12
    assert {spec.n_estimators for spec in specs if spec.model_family == "tabicl"} == {32}
    assert {spec.n_estimators for spec in specs if spec.model_family == "tabpfn"} == {8}


def test_matrix_subset_and_model_native_metric(launcher, tmp_path: Path):
    args = parse_args(
        launcher,
        "--matrix",
        "--matrix-models",
        "tabicl-v2",
        "tabpfn-v3",
        "--matrix-methods",
        "faware_ft",
        "--ttt-c-metric",
        "model_native_l2",
        "--data-root",
        "data184",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "native-subset",
    )
    launcher.validate_launch_mode(args)
    specs = launcher.build_matrix_specs(args, repo_root=REPO_ROOT)

    assert [(spec.model, spec.method) for spec in specs] == [
        ("tabicl-v2", "faware_ft"),
        ("tabpfn-v3", "faware_ft"),
    ]
    assert [spec.metric for spec in specs] == ["tabicl_encoded_l2", "raw_l2"]
    assert [value_after(spec.command, "--ttt-c-metric") for spec in specs] == [
        "tabicl_encoded_l2",
        "raw_l2",
    ]
    assert [spec.n_estimators for spec in specs] == [32, 8]
    assert len({spec.out_dir for spec in specs}) == 2


def test_matrix_subset_flags_require_matrix_and_reject_duplicates(launcher):
    scalar = parse_args(
        launcher,
        "--model",
        "tabicl-v2",
        "--method",
        "faware_ft",
        "--matrix-models",
        "tabicl-v2",
    )
    with pytest.raises(ValueError, match="require --matrix"):
        launcher.validate_launch_mode(scalar)

    duplicate = parse_args(
        launcher,
        "--matrix",
        "--matrix-models",
        "tabicl-v2",
        "tabicl-v2",
        "--matrix-methods",
        "faware_ft",
    )
    with pytest.raises(ValueError, match="duplicate"):
        launcher.validate_launch_mode(duplicate)


def test_matrix_dry_run_does_not_execute_or_create_output(
    launcher, monkeypatch, tmp_path: Path
):
    def unexpected_execute(*args, **kwargs):
        raise AssertionError("dry-run must not execute a matrix cell")

    monkeypatch.setattr(launcher, "execute_launch_spec", unexpected_execute)
    exit_code = launcher.run(
        [
            "--matrix",
            "--data-root",
            "data184",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "matrix-dry-run",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not (tmp_path / "matrix-dry-run").exists()


def test_matrix_continues_after_failure_and_writes_aggregate(
    launcher, monkeypatch, tmp_path: Path
):
    args = parse_args(
        launcher,
        "--matrix",
        "--data-root",
        "data184",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "matrix-failure",
    )
    launcher.validate_launch_mode(args)
    seen = []

    def fake_execute(spec, repo_root):
        seen.append((spec.model, spec.method))
        return 7 if len(seen) == 2 else 0

    monkeypatch.setattr(launcher, "execute_launch_spec", fake_execute)
    exit_code = launcher.run_matrix(args, repo_root=REPO_ROOT, passthrough_args=[])

    assert exit_code == 7
    assert len(seen) == 12
    aggregate = json.loads(
        (tmp_path / "matrix-failure" / "matrix_manifest.json").read_text()
    )
    assert aggregate["status"] == "fail"
    assert [cell["status"] for cell in aggregate["cells"]].count("fail") == 1


def test_subset_matrix_manifest_records_actual_axes(
    launcher, monkeypatch, tmp_path: Path
):
    args = parse_args(
        launcher,
        "--matrix",
        "--matrix-models",
        "tabicl-v2",
        "tabpfn-v3",
        "--matrix-methods",
        "faware_ft",
        "--data-root",
        "data184",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "subset-manifest",
    )
    launcher.validate_launch_mode(args)
    monkeypatch.setattr(launcher, "execute_launch_spec", lambda spec, repo_root: 0)

    assert launcher.run_matrix(args, repo_root=REPO_ROOT, passthrough_args=[]) == 0
    aggregate = json.loads(
        (tmp_path / "subset-manifest" / "matrix_manifest.json").read_text()
    )
    assert aggregate["models"] == ["tabicl-v2", "tabpfn-v3"]
    assert aggregate["methods"] == ["faware_ft"]
    assert [(cell["model"], cell["method"]) for cell in aggregate["cells"]] == [
        ("tabicl-v2", "faware_ft"),
        ("tabpfn-v3", "faware_ft"),
    ]
