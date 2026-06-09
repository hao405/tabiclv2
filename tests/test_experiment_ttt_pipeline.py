from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

import benchmark as tabicl_benchmark
from Experiment_TTT_pipeline.pipeline import PipelineConfig, build_command, expand_models, run_one_model, run_pipeline
from Experiment_TTT_pipeline.registry import MODEL_REGISTRY, ModelSpec, StrategyScript


def test_expand_all_ttt_excludes_inference_only_tabr():
    model_keys = expand_models(["all_ttt"], "ttt")

    assert "tabicl" in model_keys
    assert "tabpfnv25" in model_keys
    assert "tabr" not in model_keys


def test_tabicl_ttt_command_clears_default_gpu_groups(tmp_path: Path):
    config = PipelineConfig(strategy="ttt", out_root=tmp_path, gpus="auto", workers=1)

    command, out_dir = build_command(MODEL_REGISTRY["tabicl"], "ttt", config)

    assert "--gpu-groups" in command
    assert command[command.index("--gpu-groups") + 1] == ""
    assert command[command.index("--gpus") + 1] == "auto"
    assert out_dir == tmp_path / "ttt" / "tabicl"


def test_dry_run_both_records_tabr_ttt_skip(tmp_path: Path):
    config = PipelineConfig(strategy="both", out_root=tmp_path, dry_run=True, gpus="0", workers=1)

    results = run_pipeline(["tabr"], config)

    assert [result.strategy for result in results] == ["inference", "ttt"]
    assert results[0].status == "dry_run"
    assert results[1].status == "skip"
    assert "no ttt script" in (results[1].error or "")


def test_pipeline_rejects_empty_successful_backend_csv(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "dataset_a").mkdir(parents=True)
    backend = tmp_path / "fake_backend.py"
    backend.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--data-root')",
                "parser.add_argument('--out-dir')",
                "parser.add_argument('--workers')",
                "parser.add_argument('--random-state')",
                "parser.add_argument('--gpus')",
                "args = parser.parse_args()",
                "out_dir = Path(args.out_dir)",
                "out_dir.mkdir(parents=True, exist_ok=True)",
                "(out_dir / 'all_classification_results.csv').write_text('dataset_name,status\\n')",
            ]
        ),
        encoding="utf-8",
    )
    model = ModelSpec(
        key="fake_empty",
        display_name="Fake Empty",
        aliases=(),
        inference=StrategyScript(script=backend),
    )
    config = PipelineConfig(
        strategy="inference",
        data_root=data_root,
        out_root=tmp_path / "out",
        workers=1,
        gpus="0",
        python_executable=sys.executable,
    )

    result = run_one_model(model, "inference", config)

    assert result.status == "fail"
    assert result.returncode == 0
    assert "empty backend results CSV" in (result.error or "")


def test_tabicl_worker_output_integrity_rows_cover_missing_exit_and_empty(tmp_path: Path):
    dataset_dir = tmp_path / "dataset_a"
    dataset_dir.mkdir()

    frames, errors = tabicl_benchmark.collect_worker_output_frames(
        worker_csv_paths=[tmp_path / "missing.csv", tmp_path / "also_missing.csv"],
        worker_assigned_counts=[1, 2],
        processes=[SimpleNamespace(exitcode=0), SimpleNamespace(exitcode=9)],
        dataset_dirs=[dataset_dir],
    )
    frame = pd.concat(frames, ignore_index=True)

    assert any("missing worker CSV" in error for error in errors)
    assert any("exitcode=9" in error for error in errors)
    assert "__WORKER_MISSING_CSV__0" in set(frame["dataset_name"])
    assert "__WORKER_EXIT__1" in set(frame["dataset_name"])

    frames, errors = tabicl_benchmark.collect_worker_output_frames(
        worker_csv_paths=[],
        worker_assigned_counts=[],
        processes=[],
        dataset_dirs=[dataset_dir],
    )
    frame = pd.concat(frames, ignore_index=True)

    assert errors == ["no worker results were produced"]
    assert frame.loc[0, "dataset_name"] == "__EMPTY_RESULT__all"


def test_tabicl_ttt_worker_failure_row_preserves_ttt_schema():
    module_path = Path(__file__).resolve().parent.parent / "1C_Chunk_TTT.py"
    spec = importlib.util.spec_from_file_location("tabicl_chunk_ttt_for_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    row = module.make_worker_failure_row(
        "WORKER_MISSING_CSV",
        0,
        assigned_count=1,
        error="missing worker CSV",
    )
    row_dict = asdict(row)

    assert row.dataset_name == "__WORKER_MISSING_CSV__0"
    assert row.status == "fail"
    assert row_dict["ttt_applied"] is False
    assert row_dict["ttt_steps"] == 0
    assert "ttt_val_best_metric" in row_dict
