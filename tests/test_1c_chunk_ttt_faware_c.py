from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent.parent / "1C_Chunk_TTT_Faware_C.py"
SPEC = importlib.util.spec_from_file_location("chunk_ttt_faware_c_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
chunk_ttt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = chunk_ttt
SPEC.loader.exec_module(chunk_ttt)
chunk_ttt.ensure_runtime_deps()


def make_config(**overrides):
    values = dict(
        enabled=True,
        c_selection="f_nearest",
        c_source="test",
        c_metric="tabicl_encoded_l2",
        c_candidate_multiplier=5,
        c_class_balance=True,
        random_state=42,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_f_nearest_selects_requested_query_size_and_keeps_context_nonempty():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.array([[0.0], [0.2], [0.4], [9.6], [9.8], [10.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[10.1], [9.9]]),
        metric="tabicl_encoded_l2",
        target_mean=np.array([10.0]),
        target_var=np.array([0.01]),
    )

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=2,
        seed=123,
        config=make_config(),
        selection_context=context,
    )

    assert result.qry_idx.shape == (2,)
    assert result.ctx_idx.shape == (4,)
    assert set(result.qry_idx.tolist()) == {4, 5}
    assert result.label_coverage_ok is True
    assert result.distance_count == 2
    assert result.fallback_reason is None


def test_faware_selection_falls_back_when_label_coverage_would_fail():
    y = np.array([0, 0, 0, 1])
    X = np.array([[0.0], [0.1], [0.2], [10.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[10.0]]),
        metric="tabicl_encoded_l2",
        target_mean=np.array([10.0]),
        target_var=np.array([0.0]),
    )

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=1,
        seed=123,
        config=make_config(c_class_balance=False),
        selection_context=context,
    )

    assert result.qry_idx.shape == (1,)
    assert result.ctx_idx.shape == (3,)
    assert "fallback_from_f_nearest" in result.split_strategy
    assert result.fallback_reason == "f_nearest query labels absent from context"
    assert result.label_coverage_ok is True


def test_random_c_selection_matches_original_split_helper():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.arange(6).reshape(-1, 1)
    config = make_config(c_selection="random", c_source="none")

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=2,
        seed=77,
        config=config,
        selection_context=None,
    )
    ctx_idx, qry_idx, split_strategy = chunk_ttt._split_ctx_query(y, query_size=2, seed=77)

    np.testing.assert_array_equal(result.ctx_idx, ctx_idx)
    np.testing.assert_array_equal(result.qry_idx, qry_idx)
    assert result.split_strategy == split_strategy


def test_f_mmd_returns_only_requested_query_size():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    X = np.array([[0.0], [0.2], [0.4], [7.0], [8.0], [9.0], [10.0], [11.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[8.5], [9.5]]),
        metric="tabicl_encoded_l2",
        target_mean=np.array([9.0]),
        target_var=np.array([0.25]),
    )

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=3,
        seed=123,
        config=make_config(c_selection="f_mmd"),
        selection_context=context,
    )

    assert result.qry_idx.shape == (3,)
    assert result.ctx_idx.shape == (5,)
    assert result.label_coverage_ok is True


def test_build_ttt_config_rejects_faware_without_reference_source(tmp_path):
    parser = chunk_ttt.build_arg_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--ttt-c-selection",
            "f_nearest",
            "--ttt-c-source",
            "none",
        ]
    )

    try:
        chunk_ttt.build_ttt_config(args)
    except ValueError as exc:
        assert "requires --ttt-c-source validation or test" in str(exc)
    else:
        raise AssertionError("expected ValueError")
