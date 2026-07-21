from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "PEFT_Tabicl/MICP.py"
SPEC = importlib.util.spec_from_file_location("micp_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_builder(
    *,
    n_rows: int,
    support_size: int,
    gamma: float = 5.0,
    require_all_classes: bool = False,
) -> MODULE.MICPContextBuilder:
    train = np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2)
    labels = np.asarray([index % 2 for index in range(n_rows)])
    return MODULE.MICPContextBuilder(
        train,
        labels,
        support_size=support_size,
        gamma=gamma,
        seed=42,
        retrieval_backend="sklearn",
        require_all_classes=require_all_classes,
    )


def test_prompter_count_matches_paper_formula() -> None:
    builder = make_builder(n_rows=31, support_size=10, gamma=5.0)

    assert builder.micp_enabled is True
    assert builder.n_clusters == 16


def test_small_dataset_disables_routing() -> None:
    builder = make_builder(n_rows=10, support_size=10)
    query = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    episodes = builder.episodes_for_external(query, query_batch_size=1)

    assert builder.micp_enabled is False
    assert builder.n_clusters == 1
    assert len(episodes) == 2
    np.testing.assert_array_equal(
        episodes[0].context_indices,
        np.arange(10),
    )


def test_every_large_dataset_support_has_exact_budget() -> None:
    builder = make_builder(n_rows=30, support_size=8, gamma=1.0)

    assert set(builder.supports) == set(range(builder.n_clusters))
    assert all(len(support) == 8 for support in builder.supports.values())
    assert all(len(np.unique(support)) == 8 for support in builder.supports.values())


def test_tabicl_supports_cover_every_training_class() -> None:
    train = np.concatenate(
        [
            np.arange(20, dtype=np.float32).reshape(10, 2),
            1000 + np.arange(20, dtype=np.float32).reshape(10, 2),
        ]
    )
    labels = np.asarray([0] * 10 + [1] * 10)
    builder = MODULE.MICPContextBuilder(
        train,
        labels,
        support_size=6,
        gamma=1.0,
        seed=42,
        retrieval_backend="sklearn",
        require_all_classes=True,
    )

    assert all(
        set(labels[support].tolist()) == {0, 1}
        for support in builder.supports.values()
    )


def test_tabicl_bootstrap_covers_every_training_class() -> None:
    train = np.concatenate(
        [
            np.arange(20, dtype=np.float32).reshape(10, 2),
            1000 + np.arange(20, dtype=np.float32).reshape(10, 2),
        ]
    )
    labels = np.asarray([0] * 10 + [1] * 10)
    builder = MODULE.MICPContextBuilder(
        train,
        labels,
        support_size=6,
        gamma=1.0,
        seed=42,
        retrieval_backend="sklearn",
        require_all_classes=True,
    )

    episode = builder.bootstrap_episode(0, query_size=2)
    covered = np.concatenate([episode.context_indices, episode.query_indices])
    assert set(labels[covered].tolist()) == {0, 1}


def test_support_repair_preserves_feature_variation() -> None:
    train = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    builder = MODULE.MICPContextBuilder(
        train,
        labels,
        support_size=4,
        gamma=1.0,
        seed=42,
        retrieval_backend="sklearn",
        require_all_classes=True,
    )

    repaired = builder._repair_support(
        np.asarray([0, 1, 2, 3]),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    assert np.any(np.ptp(train[repaired], axis=0) > 0)
    assert set(labels[repaired].tolist()) == {0, 1}


def test_external_routing_preserves_each_query_exactly_once() -> None:
    builder = make_builder(n_rows=30, support_size=8, gamma=1.0)
    query = np.arange(26, dtype=np.float32).reshape(13, 2)
    episodes = builder.episodes_for_external(query, query_batch_size=3)
    observed = sorted(
        int(index)
        for episode in episodes
        for index in episode.query_indices
    )

    assert observed == list(range(13))
    assert all(len(episode.query_indices) <= 3 for episode in episodes)
    assert all(
        np.array_equal(
            episode.context_indices,
            builder.supports[int(episode.route_id)],
        )
        for episode in episodes
    )


def test_large_bootstrap_is_b_nearest_with_label_covered_query() -> None:
    builder = make_builder(n_rows=30, support_size=10, gamma=1.0)
    episode = builder.bootstrap_episode(4, query_size=4)

    assert episode.bootstrap_policy == "large_knn_b"
    assert len(episode.context_indices) + len(episode.query_indices) == 10
    assert len(episode.query_indices) == 4
    assert set(builder.y_train[episode.query_indices]).issubset(
        set(builder.y_train[episode.context_indices])
    )


def test_small_bootstrap_uses_paper_90_10_split() -> None:
    builder = make_builder(n_rows=20, support_size=20)
    episode = builder.bootstrap_episode(3, query_size=64)

    assert episode.bootstrap_policy == "small_random_90_10"
    assert len(episode.context_indices) == 18
    assert len(episode.query_indices) == 2


def test_preprocessor_is_fit_only_on_training_rows() -> None:
    train = pd.DataFrame(
        {
            "number": [0.0, 1.0, np.nan],
            "category": ["a", "b", "a"],
        }
    )
    test = pd.DataFrame(
        {
            "number": [100.0],
            "category": ["test_only"],
        }
    )
    preprocessor = MODULE.PaperRetrievalPreprocessor([1]).fit(train)

    transformed_train = preprocessor.transform(train)
    transformed_test = preprocessor.transform(test)
    assert preprocessor.fitted_rows == 3
    assert transformed_train.shape == (3, 2)
    assert transformed_test.shape == (1, 2)
    assert np.isfinite(transformed_test).all()


def test_adapter_installation_freezes_backbone_and_starts_as_identity() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(16)
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, value):
            return self.linear(self.norm(value))

    class ICL(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

    class Predictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tf_icl = ICL()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.icl_predictor = Predictor()

    model = Model()
    value = torch.randn(2, 3, 16)
    expected = model.icl_predictor.tf_icl.blocks[0](value)
    trainable, telemetry = MODULE.install_micp_adapters(model)
    actual = model.icl_predictor.tf_icl.blocks[0](value)

    torch.testing.assert_close(actual, expected)
    assert telemetry == {"adapter_layers": 2, "adapter_bottleneck": 8}
    assert trainable
    assert all(
        parameter.requires_grad == (".micp_adapter." in name)
        for name, parameter in model.named_parameters()
    )


def test_adapter_width_prefers_block_residual_layernorm() -> None:
    torch = pytest.importorskip("torch")

    class CustomRMSNorm(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.normalized_shape = width

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.helper = torch.nn.Sequential(torch.nn.LayerNorm(64))
            self.layernorm = CustomRMSNorm(512)

    assert MODULE._module_width(Block()) == 512


def test_adapter_installation_prefers_tabpfn_top_level_blocks() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.layernorm = torch.nn.LayerNorm(width)

        def forward(self, value):
            return value, None

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.icl_blocks = torch.nn.ModuleList([Block(512)])
            self.icl_predictor = torch.nn.Module()
            self.icl_predictor.tf_icl = torch.nn.Module()
            self.icl_predictor.tf_icl.blocks = torch.nn.ModuleList([Block(64)])

    model = Model()
    _trainable, telemetry = MODULE.install_micp_adapters(model)

    assert telemetry == {"adapter_layers": 1, "adapter_bottleneck": 32}
    assert hasattr(model.icl_blocks[0], "micp_adapter")
    assert not hasattr(model.icl_predictor.tf_icl.blocks[0], "micp_adapter")


def test_matrix_expands_to_fixed_two_by_two() -> None:
    assert MODULE.matrix_trials("all", "all") == [
        ("tabiclv2", "infer"),
        ("tabiclv2", "mixturepfn"),
        ("tabpfnv3", "infer"),
        ("tabpfnv3", "mixturepfn"),
    ]


def test_resume_policy_reuses_ok_and_optionally_retries_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dirs = [tmp_path / "a", tmp_path / "b"]
    for path in dataset_dirs:
        path.mkdir()
    args = SimpleNamespace(
        model_family="tabiclv2",
        method="infer",
        out_dir=str(tmp_path / "out"),
        resume=True,
        retry_failed=False,
        gpus="cpu",
        config_fingerprint="",
    )
    for key, value in {
        "data_root": str(tmp_path),
        "dataset": [],
        "max_datasets": None,
        "seed": 42,
        "support_size": 3000,
        "gamma": 5.0,
        "ca_steps": 2,
        "ca_query_size": 2,
        "ca_lr": 1e-3,
        "adapter_bottleneck": None,
        "inference_batch_size": 8,
        "n_estimators": 2,
        "train_n_estimators": 1,
        "retrieval_backend": "sklearn",
        "tabicl_model_path": "tabicl.ckpt",
        "tabpfn_v3_binary_model_path": "binary.ckpt",
        "tabpfn_v3_multiclass_model_path": "multi.ckpt",
        "use_amp": False,
        "dry_run": False,
    }.items():
        setattr(args, key, value)
    monkeypatch.setattr(MODULE, "discover_datasets", lambda _args: dataset_dirs)
    calls: list[str] = []

    def fake_evaluate(path, *, args, device):
        del device
        calls.append(path.name)
        return MODULE.ResultRow(
            dataset_name=path.name,
            dataset_dir=str(path),
            task_type="binclass",
            model_family=args.model_family,
            method=args.method,
            status="ok" if path.name == "a" else "error",
            error=None if path.name == "a" else "expected",
            config_fingerprint=args.config_fingerprint,
            seed=42,
        )

    monkeypatch.setattr(MODULE, "evaluate_dataset", fake_evaluate)
    MODULE.run_cell(args, tmp_path / "out")
    assert calls == ["a", "b"]
    summary = (tmp_path / "out" / "summary.txt").read_text(encoding="utf-8")
    assert "discovered_datasets: 2" in summary
    assert "processed_datasets: 2" in summary
    assert "ok_count: 1" in summary
    assert "failed_count: 1" in summary

    calls.clear()
    MODULE.run_cell(args, tmp_path / "out")
    assert calls == []

    args.retry_failed = True
    MODULE.run_cell(args, tmp_path / "out")
    assert calls == ["b"]


def test_write_summary_matches_ft_result_format(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset_name": "ok_a",
                "status": "ok",
                "accuracy": 0.8,
                "f1": 0.7,
                "balanced_accuracy": 0.6,
                "roc_auc": 0.9,
                "log_loss": 0.2,
                "total_seconds": 3.25,
            },
            {
                "dataset_name": "failed_b",
                "status": "error",
                "total_seconds": 1.5,
            },
            {
                "dataset_name": "skipped_c",
                "status": "skip",
                "total_seconds": 0.25,
            },
        ]
    )
    path = tmp_path / "summary.txt"

    MODULE.write_summary(path, frame, discovered_datasets=184)

    assert path.read_text(encoding="utf-8") == (
        "discovered_datasets: 184\n"
        "processed_datasets: 3\n"
        "ok_count: 1\n"
        "failed_count: 1\n"
        "skipped_count: 1\n"
        "ft_oom_fallback_count: 0\n"
        "avg_accuracy_ok: 0.800000\n"
        "avg_f1_ok: 0.700000\n"
        "avg_balanced_accuracy_ok: 0.600000\n"
        "avg_roc_auc_ok: 0.900000\n"
        "avg_log_loss_ok: 0.200000\n"
        "wall_seconds: 5.000\n"
        "failed_datasets: failed_b\n"
        "skipped_datasets: skipped_c\n"
        "ft_oom_fallback_datasets: (none)\n"
    )
