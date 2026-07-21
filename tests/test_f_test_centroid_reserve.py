from __future__ import annotations

import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tabicl():
    return load_module(
        "tabicl_test_centroid_reserve_for_tests",
        REPO_ROOT / "1C_Chunk_FT" / "1C_Chunk_FT_Faware_C.py",
    )


@pytest.fixture(scope="module")
def tabpfn():
    return load_module(
        "tabpfn_test_centroid_reserve_for_tests",
        REPO_ROOT
        / "baseline_compare"
        / "TabPFN-main"
        / "Tabpfn_1c_ttt_faware_c.py",
    )


def test_tabicl_score_is_exact_centroid_squared_l2(tabicl):
    context = tabicl.CSelectionContext(
        reference=None,
        metric="tabicl_encoded_l2",
        target_mean=np.array([1.0, 2.0]),
    )
    rows = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]])
    expected = np.mean((rows - context.target_mean) ** 2, axis=1)

    np.testing.assert_allclose(
        tabicl._test_centroid_distance_scores(rows, context),
        expected,
    )
    assert "target_var" not in tabicl.CSelectionContext.__dataclass_fields__
    assert not hasattr(tabicl, "_mmd_greedy_order")


def test_tabicl_global_reserve_never_enters_query(tabicl):
    rows = np.arange(8, dtype=np.float64).reshape(-1, 1)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    context = tabicl.CSelectionContext(
        reference=None,
        metric="tabicl_encoded_l2",
        target_mean=np.array([0.0]),
    )
    tabicl.build_test_centroid_reserved_context(
        rows,
        context,
        reserve_ratio=0.25,
    )
    assert context.reserved_context_indices.tolist() == [7, 6]

    result = tabicl._select_ctx_query_for_chunk(
        labels,
        rows,
        global_indices=np.arange(len(rows)),
        query_size=2,
        seed=42,
        config=tabicl.TTTConfig(
            c_selection=tabicl.TEST_CENTROID_RESERVE_SELECTION,
            c_source="test",
            c_reserve_ratio=0.25,
        ),
        selection_context=context,
    )
    assert len(result.qry_idx) == 2
    assert not (
        set(context.reserved_context_indices.tolist())
        & set(result.qry_idx.tolist())
    )
    assert result.label_coverage_ok


def test_tabpfn_global_score_uses_train_fitted_mixed_feature_space(tabpfn):
    train = pd.DataFrame(
        {
            "numeric": [0.0, 2.0, np.nan],
            "category": ["a", "b", "a"],
        }
    )
    test = pd.DataFrame({"numeric": [1.0], "category": ["unseen"]})

    scores = tabpfn._test_centroid_distance_scores(
        train,
        test,
        categorical_feature_indices=[1],
        metric="raw_l2",
    )
    np.testing.assert_allclose(scores, np.array([1.0, 2.5, 0.5]))

    reservation = tabpfn._build_global_test_centroid_reservation(
        train,
        test,
        categorical_feature_indices=[1],
        metric="raw_l2",
        reserve_ratio=1 / 3,
    )
    assert reservation.reserved_row_ids.tolist() == [1]


def test_tabpfn_global_ids_are_reused_with_chunk_capacity_adjustment(tabpfn):
    reservation = tabpfn.GlobalCentroidReservation(
        reserved_row_ids=np.array([0, 1, 2, 3, 4]),
        scores=np.array([10.0, 9.0, 8.0, 7.0, 6.0, 0.0]),
        requested_count=5,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    row_ids = np.array([4, 3, 2, 1, 0, 5])
    split_fn = partial(train_test_split, test_size=2, random_state=42)

    first = tabpfn._global_test_centroid_reserve_indices(
        labels,
        row_ids,
        reservation,
        split_fn,
        labels,
    )
    second = tabpfn._global_test_centroid_reserve_indices(
        labels,
        row_ids,
        reservation,
        split_fn,
        labels,
    )

    np.testing.assert_array_equal(first.reserved_idx, second.reserved_idx)
    np.testing.assert_array_equal(first.qry_idx, second.qry_idx)
    assert len(first.reserved_idx) == 3
    assert "chunk capacity released 2" in str(first.fallback_reason)
    assert not set(first.reserved_idx.tolist()) & set(first.qry_idx.tolist())
    assert first.label_coverage_ok


def test_old_f_mmd_cli_value_is_rejected(tabicl, tabpfn):
    assert (
        tabicl.build_arg_parser().parse_args([]).ttt_c_selection
        == "f_test_centroid_reserve"
    )
    assert (
        tabpfn.build_arg_parser().parse_args([]).ttt_c_selection
        == "f_test_centroid_reserve"
    )
    with pytest.raises(SystemExit):
        tabicl.build_arg_parser().parse_args(
            ["--ttt-c-selection", "f_mmd"]
        )
    with pytest.raises(SystemExit):
        tabpfn.build_arg_parser().parse_args(
            ["--ttt-c-selection", "f_mmd"]
        )
