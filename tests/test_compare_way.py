from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "PEFT_Tabicl/compare_way.py"
SPEC = importlib.util.spec_from_file_location("compare_way_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_loaded(
    *,
    y_train: np.ndarray | None = None,
) -> MODULE.LoadedDataset:
    return MODULE.LoadedDataset(
        name="toy",
        path=Path("toy"),
        task_type="multiclass",
        X_train=pd.DataFrame(
            {
                "number": [0.0, 1.0, 2.0, 3.0],
                "category": ["a", "a", "b", "b"],
            }
        ),
        y_train=(
            np.asarray([0, 1, 0, 1])
            if y_train is None
            else np.asarray(y_train)
        ),
        X_val=pd.DataFrame(
            {
                "number": [4.0],
                "category": ["validation_only"],
            }
        ),
        y_val=np.asarray([0]),
        X_test=pd.DataFrame(
            {
                "number": [5.0],
                "category": ["test_only"],
            }
        ),
        y_test=np.asarray([1]),
        categorical_indices=[1],
    )


def make_preprocessor(**overrides):
    values = {
        "clipping_value": 10.0,
        "disable_normalize_data": False,
        "use_one_hot_emb": False,
        "onehot_retrieval": False,
    }
    values.update(overrides)
    return MODULE.OfficialLoCalPFNPreprocessor(**values)


def test_official_preprocessor_uses_all_unlabeled_x_vocab_and_train_scaler() -> None:
    loaded = make_loaded()
    preprocessor = make_preprocessor(onehot_retrieval=True)
    prepared = preprocessor.fit_transform(loaded)

    assert preprocessor.vocabulary_rows == 6
    assert preprocessor.fitted_rows == 4
    assert preprocessor.category_values_[1] == [
        "a",
        "b",
        "test_only",
        "validation_only",
    ]
    assert preprocessor.raw_scaler_.mean_[0] == pytest.approx(1.5)
    assert prepared.retrieval_train.shape[1] == 5
    assert prepared.retrieval_val.shape[1] == prepared.retrieval_test.shape[1]
    assert np.max(np.abs(prepared.retrieval_test)) <= 10.0


def test_preprocessing_does_not_depend_on_any_labels() -> None:
    first = make_preprocessor(onehot_retrieval=True).fit_transform(make_loaded())
    second = make_preprocessor(onehot_retrieval=True).fit_transform(
        make_loaded(y_train=np.asarray([9, 8, 7, 6]))
    )

    np.testing.assert_array_equal(first.retrieval_train, second.retrieval_train)
    np.testing.assert_array_equal(first.retrieval_val, second.retrieval_val)
    np.testing.assert_array_equal(first.retrieval_test, second.retrieval_test)


def test_one_hot_model_embedding_obeys_official_feature_budget() -> None:
    loaded = make_loaded()
    prepared = make_preprocessor(use_one_hot_emb=True).fit_transform(loaded)

    assert prepared.X_train.shape[1] == 5
    assert prepared.categorical_indices == []


def test_external_knn_is_query_local_and_has_no_label_repair() -> None:
    train_space = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.asarray([0] * 5 + [1] * 5)
    builder = MODULE.LocalContextBuilder(
        train_space,
        y,
        context_size=3,
        train_query_length=2,
        backend="sklearn",
        seed=42,
    )
    episodes = builder.episodes_for_external(
        np.asarray([[0.1, 1.1], [18.1, 19.1]], dtype=np.float32)
    )

    assert set(y[episodes[0].context_indices]) == {0}
    assert set(y[episodes[1].context_indices]) == {1}
    assert tuple(episodes[0].context_indices) != tuple(
        episodes[1].context_indices
    )
    assert set(MODULE.ContextEpisode.__dataclass_fields__) == {
        "context_indices",
        "query_indices",
    }


def test_training_batch_retrieves_k_plus_q_and_shares_permutation() -> None:
    train_space = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.asarray([0, 1] * 6)
    builder = MODULE.LocalContextBuilder(
        train_space,
        y,
        context_size=3,
        train_query_length=2,
        backend="sklearn",
        seed=42,
    )
    batches = builder.training_batches(
        epoch=0,
        max_steps=1,
        batch_size=2,
        better_selection=False,
        exact_knn=False,
    )

    assert len(batches) == 1
    assert len(batches[0]) == 2
    rng = np.random.default_rng(42)
    anchors = rng.permutation(len(y))[:2]
    _, raw_neighbors = builder.index.query(train_space[anchors], 5)
    observed_permutations = []
    for anchor, raw, episode in zip(anchors, raw_neighbors, batches[0], strict=True):
        sequence = np.concatenate(
            [episode.context_indices, episode.query_indices]
        )
        observed_permutations.append(
            [int(np.flatnonzero(raw == value)[0]) for value in sequence]
        )
        assert len(episode.context_indices) == 3
        assert len(episode.query_indices) == 2
        assert int(anchor) in sequence
    assert observed_permutations[0] == observed_permutations[1]


def test_small_dataset_keeps_at_least_one_training_query() -> None:
    train_space = np.arange(4, dtype=np.float32).reshape(-1, 1)
    builder = MODULE.LocalContextBuilder(
        train_space,
        np.asarray([0, 1, 0, 1]),
        context_size=1000,
        train_query_length=1000,
        backend="sklearn",
        seed=3,
    )
    episode = builder.training_batches(
        epoch=0,
        max_steps=1,
        batch_size=2,
        better_selection=False,
        exact_knn=False,
    )[0][0]

    assert len(episode.context_indices) == 3
    assert len(episode.query_indices) == 1


def test_exact_knn_reverses_k_plus_one_sequence() -> None:
    train_space = np.arange(8, dtype=np.float32).reshape(-1, 1)
    builder = MODULE.LocalContextBuilder(
        train_space,
        np.asarray([0, 1] * 4),
        context_size=3,
        train_query_length=1,
        backend="sklearn",
        seed=9,
    )
    episode = builder.training_batches(
        epoch=0,
        max_steps=1,
        batch_size=2,
        better_selection=False,
        exact_knn=True,
    )[0][0]
    anchor = np.random.default_rng(9).permutation(8)[0]
    _, nearest = builder.index.query(train_space[[anchor]], 4)

    np.testing.assert_array_equal(
        np.concatenate([episode.context_indices, episode.query_indices]),
        nearest[0][::-1],
    )


def test_tabpfn_prefix_split_never_resplits_episode() -> None:
    X = np.arange(15).reshape(5, 3)
    y = np.arange(5)
    X_context, X_query, y_context, y_query = (
        MODULE.prefix_context_query_split(X, y, context_size=3)
    )

    np.testing.assert_array_equal(X_context, X[:3])
    np.testing.assert_array_equal(X_query, X[3:])
    np.testing.assert_array_equal(y_context, y[:3])
    np.testing.assert_array_equal(y_query, y[3:])


def test_tabicl_training_stack_preserves_two_episodes_and_all_queries() -> None:
    torch = pytest.importorskip("torch")
    first = (
        torch.zeros(2, 5, 3),
        torch.zeros(2, 3),
        torch.tensor([[0, 1], [0, 1]]),
    )
    second = (
        torch.zeros(2, 5, 5),
        torch.ones(2, 3),
        torch.tensor([[1, 2], [1, 2]]),
    )

    X, y_context, y_query = MODULE.TabICLNativeEngine._stack_training_batch(
        [first, second]
    )

    assert X.shape == (4, 5, 5)
    assert y_context.shape == (4, 3)
    assert y_query.shape == (4, 2)


def test_tabpfn_batched_prediction_aligns_missing_context_classes() -> None:
    class FakeClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba_batched(self, contexts, labels, queries):
            self.calls += 1
            classes = np.unique(labels[0])
            probabilities = np.arange(1, len(classes) + 1, dtype=float)
            probabilities /= probabilities.sum()
            return np.tile(probabilities, (len(contexts), 1, 1))

    loaded = MODULE.LoadedDataset(
        name="fake",
        path=Path("fake"),
        task_type="multiclass",
        X_train=pd.DataFrame({"x": np.arange(6)}),
        y_train=np.asarray([0, 1, 0, 2, 1, 2]),
        X_val=None,
        y_val=None,
        X_test=pd.DataFrame({"x": [10, 11]}),
        y_test=np.asarray([0, 2]),
    )
    prepared = MODULE.PreparedLoCalData(
        X_train=loaded.X_train.astype(float),
        X_val=None,
        X_test=loaded.X_test.astype(float),
        retrieval_train=np.arange(6, dtype=np.float32).reshape(-1, 1),
        retrieval_val=None,
        retrieval_test=np.asarray([[10.0], [11.0]], dtype=np.float32),
        categorical_indices=[],
    )
    episodes = [
        MODULE.ContextEpisode(np.asarray([0, 1, 2]), np.asarray([0])),
        MODULE.ContextEpisode(np.asarray([3, 4, 5]), np.asarray([1])),
    ]
    classifier = FakeClassifier()
    engine = MODULE.TabPFNv3Engine.__new__(MODULE.TabPFNv3Engine)
    probabilities = engine._predict_with_classifier(
        classifier,
        loaded,
        prepared,
        episodes,
        prepared.X_test,
        args=SimpleNamespace(local_inference_batch_size=512),
    )

    assert classifier.calls == 2
    assert probabilities.shape == (2, 3)
    assert probabilities[0, 2] == 0.0
    assert probabilities[1, 0] == 0.0


def test_single_class_context_is_explicit_failure() -> None:
    loaded = MODULE.LoadedDataset(
        name="fake",
        path=Path("fake"),
        task_type="binclass",
        X_train=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        y_train=np.asarray([0, 0, 1]),
        X_val=None,
        y_val=None,
        X_test=pd.DataFrame({"x": [3.0]}),
        y_test=np.asarray([1]),
    )
    prepared = MODULE.PreparedLoCalData(
        X_train=loaded.X_train,
        X_val=None,
        X_test=loaded.X_test,
        retrieval_train=np.arange(3, dtype=np.float32).reshape(-1, 1),
        retrieval_val=None,
        retrieval_test=np.asarray([[3.0]], dtype=np.float32),
        categorical_indices=[],
    )
    engine = MODULE.TabPFNv3Engine.__new__(MODULE.TabPFNv3Engine)

    with pytest.raises(ValueError, match="single-class"):
        engine._predict_with_classifier(
            object(),
            loaded,
            prepared,
            [MODULE.ContextEpisode(np.asarray([0, 1]), np.asarray([0]))],
            prepared.X_test,
            args=SimpleNamespace(local_inference_batch_size=512),
        )


def test_matrix_expands_exactly_two_cells_and_micp_is_absent() -> None:
    assert MODULE.expand_matrix_trials("all", "localpfn") == [
        ("tabiclv2", "localpfn"),
        ("tabpfnv3", "localpfn"),
    ]
    parser = MODULE.build_arg_parser()
    method_action = next(
        action for action in parser._actions if action.dest == "method"
    )
    assert set(method_action.choices) == {"localpfn"}
    assert not hasattr(MODULE, "MixtureContextBuilder")
    assert not hasattr(MODULE, "install_residual_adapters")
    with pytest.raises(SystemExit):
        parser.parse_args(["--method", "mixturepfn"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--method", "all"])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("negloss", -0.2),
        ("acc", 0.7),
        ("f1", 0.6),
        ("auc", 0.8),
    ],
)
def test_official_early_stopping_metric_selection(
    metric: str,
    expected: float,
) -> None:
    metrics = {
        "log_loss": 0.2,
        "accuracy": 0.7,
        "f1": 0.6,
        "roc_auc": 0.8,
    }
    assert MODULE._early_stopping_value(metric, metrics) == pytest.approx(expected)


def make_result(path: Path, fingerprint: str) -> MODULE.DatasetResult:
    return MODULE.DatasetResult(
        dataset_name=path.name,
        dataset_dir=str(path),
        task_type="binclass",
        model_family="tabiclv2",
        method="localpfn",
        config_fingerprint=fingerprint,
        status="ok",
        error=None,
    )


def test_dataset_checkpoint_resume_requires_matching_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_paths = [tmp_path / "first", tmp_path / "second"]
    for path in dataset_paths:
        path.mkdir()
    out_dir = tmp_path / "results"
    args = SimpleNamespace(
        resume=True,
        model_family="tabiclv2",
        method="localpfn",
        config_fingerprint="new",
    )
    pd.DataFrame(
        [
            {
                **MODULE.asdict(make_result(dataset_paths[0], "legacy")),
            }
        ]
    ).to_csv(out_dir.with_suffix(".csv"), index=False)
    calls: list[str] = []

    def evaluate(path: Path, **_kwargs):
        calls.append(path.name)
        return make_result(path, "new")

    monkeypatch.setattr(MODULE, "evaluate_dataset", evaluate)
    rows = MODULE._evaluate_with_checkpoints(
        dataset_paths,
        args=args,
        device="cpu",
        checkpoint_csv=out_dir.with_suffix(".csv"),
        result_out_dir=None,
    )

    assert calls == ["first", "second"]
    assert [row.config_fingerprint for row in rows] == ["new", "new"]


def test_cell_completion_requires_exact_coverage_and_fingerprint(
    tmp_path: Path,
) -> None:
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "run_manifest.json").write_text(
        json.dumps({"config_fingerprint": "correct"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"dataset_name": "first", "status": "ok"}]
    ).to_csv(cell_dir / "all_classification_results.csv", index=False)

    assert not MODULE._cell_is_complete(
        cell_dir,
        expected_dataset_names={"first", "second"},
        expected_fingerprint="correct",
    )
    assert not MODULE._cell_is_complete(
        cell_dir,
        expected_dataset_names={"first"},
        expected_fingerprint="wrong",
    )
    assert MODULE._cell_is_complete(
        cell_dir,
        expected_dataset_names={"first"},
        expected_fingerprint="correct",
    )
