from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent.parent / "col_attention_TTT.py"
SPEC = importlib.util.spec_from_file_location("col_attention_ttt_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
col_ttt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = col_ttt
SPEC.loader.exec_module(col_ttt)
col_ttt.ensure_runtime_deps()


def test_icl_attention_mix_uses_sixty_forty_counts_and_top_scores():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    scores = np.array([0.01, 0.02, 0.03, 0.04, 0.90, 0.20, 0.10, 0.05, 0.99, 0.98, 0.30, 0.25])

    result = col_ttt._select_icl_attention_mix_query_indices(
        y,
        scores,
        query_size=5,
        seed=123,
        icl_attn_ratio=0.6,
        random_ratio=0.4,
    )

    assert result.qry_idx.shape == (5,)
    assert result.ctx_idx.shape == (7,)
    assert result.selected_attention_count == 3
    assert result.selected_random_count == 2
    assert {4, 8, 9}.issubset(set(result.qry_idx.tolist()))
    assert result.label_coverage_ok is True
    assert result.fallback_reason is None


def test_icl_attention_mix_skips_unique_label_that_would_break_context_coverage():
    y = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 100.0])

    result = col_ttt._select_icl_attention_mix_query_indices(
        y,
        scores,
        query_size=1,
        seed=7,
        icl_attn_ratio=1.0,
        random_ratio=0.0,
    )

    assert result.qry_idx.shape == (1,)
    assert 3 not in result.qry_idx.tolist()
    assert result.label_coverage_ok is True
    assert result.fallback_reason is None


def test_icl_attention_mix_random_part_changes_with_seed():
    y = np.array([0] * 8 + [1] * 8 + [2] * 8)
    scores = np.linspace(0.0, 1.0, y.shape[0])
    random_parts = set()

    for seed in range(10):
        result = col_ttt._select_icl_attention_mix_query_indices(
            y,
            scores,
            query_size=5,
            seed=seed,
            icl_attn_ratio=0.6,
            random_ratio=0.4,
        )
        attention_top = set(np.argsort(-scores, kind="mergesort")[:3].tolist())
        random_parts.add(tuple(sorted(set(result.qry_idx.tolist()) - attention_top)))

    assert len(random_parts) > 1


def test_icl_attention_mix_falls_back_on_score_shape_mismatch():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2])

    result = col_ttt._select_icl_attention_mix_query_indices(
        y,
        scores,
        query_size=2,
        seed=123,
        icl_attn_ratio=0.6,
        random_ratio=0.4,
    )

    assert result.qry_idx.shape == (2,)
    assert result.ctx_idx.shape == (4,)
    assert result.split_strategy.startswith("fallback_from_icl_attention_mix:")
    assert "icl_attention_score_shape_mismatch" in result.fallback_reason
    assert result.label_coverage_ok is True


def test_build_ttt_config_icl_attention_defaults(tmp_path):
    parser = col_ttt.build_arg_parser()
    args = parser.parse_args(["--out-dir", str(tmp_path)])

    config = col_ttt.build_ttt_config(args)

    assert config.c_selection == "icl_attention_mix"
    assert config.icl_attn_ratio == 0.6
    assert config.random_ratio == 0.4
    assert config.icl_attn_source == "test"
    assert config.icl_attn_layer == 4


def test_build_ttt_config_rejects_bad_ratio_sum(tmp_path):
    parser = col_ttt.build_arg_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--ttt-icl-attn-ratio",
            "0.7",
            "--ttt-random-ratio",
            "0.4",
        ]
    )

    try:
        col_ttt.build_ttt_config(args)
    except ValueError as exc:
        assert "must sum to 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_get_icl_attention_module_uses_one_based_fourth_block():
    attn_modules = [object() for _ in range(5)]
    blocks = [SimpleNamespace(attn=attn) for attn in attn_modules]
    classifier = SimpleNamespace(
        model_=SimpleNamespace(
            icl_predictor=SimpleNamespace(tf_icl=SimpleNamespace(blocks=blocks))
        )
    )

    assert col_ttt._get_icl_attention_module(classifier, layer=4) is attn_modules[3]


def test_get_icl_attention_module_rejects_out_of_range_layer():
    blocks = [SimpleNamespace(attn=object()) for _ in range(3)]
    classifier = SimpleNamespace(
        model_=SimpleNamespace(
            icl_predictor=SimpleNamespace(tf_icl=SimpleNamespace(blocks=blocks))
        )
    )

    try:
        col_ttt._get_icl_attention_module(classifier, layer=4)
    except RuntimeError as exc:
        assert "exceeds available ICL blocks" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
