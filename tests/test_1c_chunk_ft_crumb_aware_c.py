from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "1C_Chunk_FT"
    / "CRUMB_aware_C.py"
)
SPEC = importlib.util.spec_from_file_location("crumb_aware_c_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
crumb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = crumb
SPEC.loader.exec_module(crumb)
crumb.ensure_runtime_deps()


def make_context(reference: np.ndarray) -> object:
    return crumb.CSelectionContext(
        reference=np.asarray(reference, dtype=np.float64),
        metric="standardized_l2",
    )


def test_parser_defaults_match_requested_crumb_rff_early_stop(tmp_path):
    parser = crumb.build_arg_parser()
    args = parser.parse_args(["--out-dir", str(tmp_path)])
    config = crumb.build_ttt_config(args)

    assert config.c_selection == "crumb_mmd"
    assert config.crumb_rff_dim == 64
    assert config.crumb_greedy_batch_size == 500
    assert config.crumb_mmd_check_interval == 500
    assert config.crumb_early_stop_epsilon == pytest.approx(1e-4)
    assert config.crumb_early_stop_patience == 5


def test_global_pool_uses_rff_early_stop_after_five_low_improvement_checks():
    X_train = np.zeros((3501, 3), dtype=np.float64)
    X_test = np.zeros((80, 3), dtype=np.float64)
    context = make_context(X_test)
    config = crumb.TTTConfig(
        random_state=42,
        crumb_bandwidth="1.0",
        crumb_rff_dim=64,
        crumb_greedy_batch_size=500,
        crumb_mmd_check_interval=500,
        crumb_early_stop_epsilon=1e-4,
        crumb_early_stop_patience=5,
    )

    crumb.build_crumb_mmd_query_candidate_pool(
        X_train,
        context,
        config=config,
    )

    assert context.crumb_max_candidate_count == 3500
    assert context.crumb_early_stopped is True
    assert context.crumb_early_stop_checks == 5
    assert context.crumb_candidate_count == 3000
    assert context.crumb_final_mmd2 == pytest.approx(0.0, abs=1e-12)
    assert len(context.context_only_indices) == 501


def test_global_pool_is_deterministic_and_partitions_full_training_set():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(180, 4))
    X_test = rng.normal(size=(40, 4))
    config = crumb.TTTConfig(
        random_state=19,
        crumb_bandwidth="median",
        crumb_rff_dim=32,
        crumb_greedy_batch_size=50,
        crumb_mmd_check_interval=25,
        crumb_early_stop_epsilon=0.0,
        crumb_early_stop_patience=5,
    )
    context_a = make_context(X_test)
    context_b = make_context(X_test)

    crumb.build_crumb_mmd_query_candidate_pool(X_train, context_a, config=config)
    crumb.build_crumb_mmd_query_candidate_pool(X_train, context_b, config=config)

    np.testing.assert_array_equal(
        context_a.query_candidate_indices,
        context_b.query_candidate_indices,
    )
    candidates = set(context_a.query_candidate_indices.tolist())
    context_only = set(context_a.context_only_indices.tolist())
    assert candidates.isdisjoint(context_only)
    assert candidates | context_only == set(range(len(X_train)))
    assert len(candidates) <= len(X_train) - 1


def test_chunk_queries_only_come_from_global_pool_and_change_with_seed():
    y = np.array([0] * 12 + [1] * 12)
    X = np.arange(24, dtype=np.float64).reshape(-1, 1)
    context = make_context(np.zeros((4, 1)))
    context.query_candidate_indices = np.array(
        [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
        dtype=int,
    )
    context.context_only_indices = np.array(
        [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23],
        dtype=int,
    )
    config = crumb.TTTConfig(c_selection="crumb_mmd", c_class_balance=True)

    result_a = crumb._select_ctx_query_for_chunk(
        y,
        X,
        global_indices=np.arange(24),
        query_size=6,
        seed=41,
        config=config,
        selection_context=context,
    )
    result_b = crumb._select_ctx_query_for_chunk(
        y,
        X,
        global_indices=np.arange(24),
        query_size=6,
        seed=42,
        config=config,
        selection_context=context,
    )

    candidate_set = set(context.query_candidate_indices.tolist())
    context_only_set = set(context.context_only_indices.tolist())
    assert set(result_a.qry_idx.tolist()).issubset(candidate_set)
    assert set(result_b.qry_idx.tolist()).issubset(candidate_set)
    assert context_only_set.issubset(set(result_a.ctx_idx.tolist()))
    assert context_only_set.issubset(set(result_b.ctx_idx.tolist()))
    assert not np.array_equal(
        np.sort(result_a.qry_idx),
        np.sort(result_b.qry_idx),
    )
    assert len(result_a.qry_idx) == 6
    assert np.bincount(y[result_a.qry_idx], minlength=2).tolist() == [3, 3]
    assert result_a.label_coverage_ok is True


def test_candidate_shortage_safely_truncates_without_context_only_leakage():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    context = make_context(np.zeros((2, 1)))
    context.query_candidate_indices = np.array([0, 3], dtype=int)
    context.context_only_indices = np.array([1, 2, 4, 5], dtype=int)
    config = crumb.TTTConfig(c_selection="crumb_mmd", c_class_balance=True)

    result = crumb._select_ctx_query_for_chunk(
        y,
        X,
        global_indices=np.arange(6),
        query_size=4,
        seed=9,
        config=config,
        selection_context=context,
    )

    assert set(result.qry_idx.tolist()) == {0, 3}
    assert set(context.context_only_indices.tolist()).issubset(
        set(result.ctx_idx.tolist())
    )
    assert result.fallback_reason == (
        "CRUMB candidate pool safely truncated query rows from 4 to 2"
    )
    assert result.label_coverage_ok is True


def test_missing_pool_never_falls_back_to_unrestricted_random_query():
    y = np.array([0, 0, 1, 1])
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    config = crumb.TTTConfig(c_selection="crumb_mmd")

    result = crumb._select_ctx_query_for_chunk(
        y,
        X,
        global_indices=np.arange(4),
        query_size=2,
        seed=3,
        config=config,
        selection_context=make_context(np.zeros((2, 1))),
    )

    assert result.ctx_idx.size == 0
    assert result.qry_idx.size == 0
    assert result.split_strategy == "crumb_mmd_global_candidates:fallback"
    assert result.fallback_reason == "missing global CRUMB query-candidate pool"
