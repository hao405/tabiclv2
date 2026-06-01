from __future__ import annotations

from pathlib import Path

from Experiment_TTT_pipeline.pipeline import PipelineConfig, build_command, expand_models, run_pipeline
from Experiment_TTT_pipeline.registry import MODEL_REGISTRY


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
