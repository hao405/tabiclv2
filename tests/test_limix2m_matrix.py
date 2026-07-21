from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from PEFT_Tabicl.MICP import PAPER_PROTOCOL
from runners.limix import METHODS
from runners.limix.methods import (
    MixtureContextBuilder,
    RetrievalPreprocessor,
    configure_last_block,
    configure_lora,
    configure_micp_adapters,
    configure_prefix,
    ensure_label_coverage,
    faware_reserved_indices,
    split_context_query,
)
from runners.limix.run_experiment import (
    DatasetResult,
    LimiXMethodRunner,
    atomic_write_json,
    build_arg_parser,
    load_existing_result,
    preflight_environment,
)
from runners.limix.summarize_matrix import summarize


REPO_ROOT = Path(__file__).resolve().parents[1]


def _torch():
    return pytest.importorskip("torch")


def _fake_model():
    torch = _torch()

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 4
            self.num_heads = 2
            self.head_dim = 2
            self.dropout = 0.0
            self.qkv_proj_weight = torch.nn.Parameter(torch.randn(3, 2, 2, 4))
            self.out_proj_weight = torch.nn.Parameter(torch.randn(2, 2, 4))

    class FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence_attentions = torch.nn.ModuleList([FakeAttention()])
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, value, **_kwargs):
            return self.linear(value), None, None

    class FakeStack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)[0]
            return value

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 4
            self.encoder_x = torch.nn.Linear(4, 4)
            self.transformer_encoder = FakeStack()
            self.cls_y_decoder = torch.nn.Linear(4, 3)

        def forward(self, value):
            return self.cls_y_decoder(
                self.transformer_encoder(self.encoder_x(value))
            )

    return FakeModel()


def _result(dataset: str, method: str, status: str = "ok") -> DatasetResult:
    return DatasetResult(
        dataset_name=dataset,
        dataset_dir=f"/tmp/{dataset}",
        task_type="binclass",
        method=method,
        seed=42,
        config_hash="hash",
        n_train=10,
        n_val=2,
        n_test=3,
        n_features=4,
        n_classes=2,
        accuracy=0.5 if status == "ok" else None,
        f1=0.5 if status == "ok" else None,
        balanced_accuracy=0.5 if status == "ok" else None,
        roc_auc=0.5 if status == "ok" else None,
        log_loss=0.7 if status == "ok" else None,
        fit_seconds=1.0,
        predict_seconds=1.0,
        wall_seconds=2.0,
        peak_vram_mb=None,
        status=status,
        error=None if status == "ok" else "failed",
        ttt_applied=method != "infer" and status == "ok",
        ttt_steps=1,
        ttt_loss=0.1,
        ttt_lr=1e-5,
        trainable_params=4,
        total_params=10,
        trainable_ratio=0.4,
        best_epoch=1,
        baseline_val_accuracy=0.4,
        best_val_accuracy=0.5,
        stopped_early=False,
        split_strategy=None,
        selection_metric=None,
        reserve_ratio=None,
        retrieval_backend=None,
        context_size=None,
        support_size=None,
        n_routes=None,
        micp_gamma=None,
        micp_train_batch_size=None,
        micp_inference_batch_size=None,
        micp_implementation=None,
        fallback_count=0,
    )


def test_method_axis_exposes_paper_micp_and_full_mixturepfn() -> None:
    assert METHODS == (
        "infer",
        "ft",
        "faware_ft",
        "lora",
        "prefix",
        "last_block",
        "micp",
        "mixturepfn",
    )


def test_parser_rejects_removed_localpfn_method() -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--method", "localpfn", "--out-dir", "/tmp/out"]
        )


def test_lora_is_initially_equivalent_and_only_lora_is_trainable() -> None:
    torch = _torch()
    model = _fake_model()
    value = torch.randn(3, 4)
    expected = model(value).detach().clone()
    trainable, installed = configure_lora(model, rank=2, alpha=4, dropout=0)
    actual = model(value).detach()
    assert installed >= 7
    assert torch.allclose(expected, actual)
    assert trainable
    assert all(
        ".lora_A" in name or ".lora_B" in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_last_block_selects_only_final_layer_and_decoder() -> None:
    model = _fake_model()
    configure_last_block(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith("transformer_encoder.layers.1.")
        or name.startswith("cls_y_decoder.")
        for name in trainable_names
    )
    assert not any(name.startswith("encoder_x.") for name in trainable_names)


def test_prefix_wraps_every_sequence_attention_and_freezes_backbone() -> None:
    torch = _torch()
    model = _fake_model()
    trainable, installed = configure_prefix(model, prefix_length=3)
    assert installed == 2
    assert trainable
    assert all(
        name.endswith("prefix_key") or name.endswith("prefix_value")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    attention = model.transformer_encoder.layers[0].sequence_attentions[0]
    x = torch.randn(1, 2, 5, 4)
    output, _, _ = attention(x, x_kv=x, copy_first_head_kv=True)
    assert output.shape == x.shape


def test_mixturepfn_adapter_is_identity_at_install_time() -> None:
    torch = _torch()
    model = _fake_model()
    value = torch.randn(2, 3, 4)
    expected = model.transformer_encoder.layers[0](value)[0].detach()
    trainable, installed = configure_micp_adapters(model, bottleneck=2)
    actual = model.transformer_encoder.layers[0](value)[0].detach()
    assert installed == 2
    assert torch.allclose(expected, actual)
    assert trainable
    assert all(
        ".adapter." in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_adaptation_moves_model_before_baseline_inference() -> None:
    torch = _torch()

    class StopAfterMove(RuntimeError):
        pass

    class TrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.moved = False

        def to(self, *args, **kwargs):
            assert not torch.is_inference_mode_enabled()
            self.moved = True
            return super().to(*args, **kwargs)

    model = TrackingModel()
    predictor = SimpleNamespace(model=model, device=torch.device("cpu"))
    runner = object.__new__(LimiXMethodRunner)
    runner.method = "ft"
    runner.args = SimpleNamespace(seed=42, validation_fraction=0.1)
    runner._load_predictor = lambda: predictor

    def stop_at_fit(*_args, **_kwargs):
        assert model.moved
        raise StopAfterMove

    runner._fit_standard = stop_at_fit
    loaded = SimpleNamespace(
        X_train_merged=pd.DataFrame({"x": np.arange(20, dtype=np.float32)}),
        y_train_merged=np.asarray([0, 1] * 10),
        X_test=pd.DataFrame({"x": [0.0]}),
    )
    with pytest.raises(StopAfterMove):
        runner.run_loaded(loaded)


def test_retrieval_preprocessor_fits_only_supplied_train_rows() -> None:
    train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "c": ["a", "b", None]})
    test = pd.DataFrame({"n": [100.0], "c": ["unseen"]})
    preprocessor = RetrievalPreprocessor([1]).fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_test = preprocessor.transform(test)
    assert preprocessor.fitted_rows == 3
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()


def test_faware_reserves_rows_farthest_from_unlabeled_test_mean() -> None:
    train = np.asarray([[0.0], [1.0], [10.0], [20.0]], dtype=np.float32)
    test = np.asarray([[0.0], [0.5]], dtype=np.float32)
    reserved, scores = faware_reserved_indices(train, test, reserve_ratio=0.25)
    assert reserved.tolist() == [3]
    assert int(np.argmax(scores)) == 3


def test_split_context_query_forces_reserved_rows_and_covers_query_labels() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    context, query, strategy = split_context_query(
        labels,
        query_ratio=0.3,
        seed=42,
        forced_context=[0, 3],
    )
    assert {0, 3}.issubset(set(context.tolist()))
    assert set(labels[query]).issubset(set(labels[context]))
    assert strategy


def test_label_coverage_repairs_missing_classes() -> None:
    train_space = np.asarray([[0.0], [0.1], [0.2], [10.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1])
    repaired, fallback, _ = ensure_label_coverage(
        [0, 1],
        y_train=labels,
        train_space=train_space,
        query_vector=np.asarray([0.0]),
        target_size=2,
        seed=42,
    )
    assert set(labels[repaired]) == {0, 1}
    assert fallback is not None


def test_micp_routes_share_fixed_support() -> None:
    space = np.asarray(
        [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 0, 1, 1])
    builder = MixtureContextBuilder(
        space,
        labels,
        support_size=3,
        n_clusters=2,
        seed=42,
        backend="sklearn",
    )
    episodes = builder.route(np.asarray([[0.05], [0.15]], dtype=np.float32))
    assert episodes[0].route_id == episodes[1].route_id
    assert np.array_equal(
        episodes[0].context_indices,
        episodes[1].context_indices,
    )


def test_micp_gamma_controls_paper_route_count() -> None:
    space = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.asarray([0, 1] * 6)
    builder = MixtureContextBuilder(
        space,
        labels,
        support_size=3,
        n_clusters=None,
        seed=42,
        backend="sklearn",
        gamma=1.5,
    )
    assert builder.n_clusters == 6
    assert builder.gamma == 1.5


def test_paper_protocol_constants_match_micp_appendix() -> None:
    assert PAPER_PROTOCOL.support_size == 3000
    assert PAPER_PROTOCOL.gamma_candidates == (5.0, 1.0)
    assert PAPER_PROTOCOL.capfn_steps == 128
    assert PAPER_PROTOCOL.capfn_learning_rate == 1e-3
    assert PAPER_PROTOCOL.train_batch_size == 64
    assert PAPER_PROTOCOL.inference_batch_size == 1024
    assert PAPER_PROTOCOL.n_estimators == 16
    assert PAPER_PROTOCOL.n_prompters(6000, 5.0) == 10


def test_micp_bootstrap_matches_large_b_then_batch_and_small_90_10() -> None:
    large_space = np.arange(40, dtype=np.float32).reshape(20, 2)
    large_labels = np.asarray([0, 1] * 10)
    large = MixtureContextBuilder(
        large_space,
        large_labels,
        support_size=10,
        n_clusters=2,
        seed=42,
        backend="sklearn",
    ).bootstrap_episode(0, query_batch_size=2)
    assert len(large.context_indices) + len(large.query_indices) == 10
    assert len(large.query_indices) == 2
    assert large.fallback_reason == "large_knn_b_then_batch64"

    small_space = np.arange(20, dtype=np.float32).reshape(10, 2)
    small_labels = np.asarray([0, 1] * 5)
    small = MixtureContextBuilder(
        small_space,
        small_labels,
        support_size=10,
        n_clusters=1,
        seed=42,
        backend="sklearn",
    ).bootstrap_episode(0, query_batch_size=64)
    assert len(small.context_indices) == 9
    assert len(small.query_indices) == 1
    assert small.fallback_reason == "small_random_90_10"


def test_atomic_dataset_result_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    row = _result("adult", "ft")
    atomic_write_json(path, row.__dict__)
    loaded = load_existing_result(path)
    assert loaded == row
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_summary_tracks_shared_ok_intersection(tmp_path: Path) -> None:
    for method in METHODS:
        method_dir = tmp_path / method
        method_dir.mkdir()
        frame = pd.DataFrame(
            [
                _result("shared", method).__dict__,
                _result("only_some", method, "ok" if method == "infer" else "error").__dict__,
            ]
        )
        frame.to_csv(method_dir / "all_classification_results.csv", index=False)
        (method_dir / "run_manifest.json").write_text(
            json.dumps({"status": "complete", "config_hash": method}),
            encoding="utf-8",
        )
    payload = summarize(tmp_path, list(METHODS), expected_datasets=2)
    assert payload["combined_rows"] == 16
    assert payload["shared_status_ok_count"] == 1
    assert (tmp_path / "matrix_manifest.json").is_file()


def test_parser_defaults_match_fixed_protocol() -> None:
    args = build_arg_parser().parse_args(
        ["--method", "faware_ft", "--out-dir", "/tmp/out"]
    )
    assert args.seed == 42
    assert args.lr == 1e-5
    assert args.epochs == 30
    assert args.chunk_size == 200
    assert args.reserve_ratio == 0.05
    assert args.lora_rank == 4
    assert args.prefix_length == 8
    assert args.micp_steps == 128
    assert args.micp_gammas == (5.0, 1.0)
    assert args.micp_train_batch_size == 64
    assert args.micp_inference_batch_size == 1024
    assert args.micp_n_estimators == 16


def test_runtime_preflight_reports_missing_dependencies(monkeypatch) -> None:
    import runners.limix.run_experiment as runner

    real_find_spec = runner.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "kditransform":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(runner.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="requirements_benchmark.txt"):
        preflight_environment()


def test_data184_has_184_supported_classification_tasks() -> None:
    data_root = REPO_ROOT / "data184"
    dataset_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    assert len(dataset_dirs) == 184
    audit = pd.read_csv(data_root / "_kept_ttt_successful.csv")
    assert set(audit["dataset_name"].astype(str)) == {
        path.name for path in dataset_dirs
    }
    for dataset_dir in dataset_dirs:
        info = json.loads((dataset_dir / "info.json").read_text(encoding="utf-8"))
        assert info["task_type"] in {"binclass", "multiclass"}
        if info.get("n_classes") is not None:
            assert int(info["n_classes"]) <= 10


def test_shell_dry_run_expands_exact_seed42_matrix(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path / "results"),
            "SEEDS": "42",
            "GPUS": "0,1,2,3",
        }
    )
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_limix2m_data184_matrix.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("command:") == 8
    assert "planned_cells: 8" in completed.stdout
    assert "datasets_per_cell: 184" in completed.stdout
    assert "planned_dataset_method_tasks: 1472" in completed.stdout
    for method in METHODS:
        assert f"seed42/{method}" in completed.stdout
    assert not (tmp_path / "results").exists()
