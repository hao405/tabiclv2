from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parent.parent / "1C_Chunk_TTT_Faware_C_Upper.py"
SPEC = importlib.util.spec_from_file_location("chunk_ttt_faware_c_upper_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
chunk_ttt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = chunk_ttt
SPEC.loader.exec_module(chunk_ttt)
chunk_ttt.ensure_runtime_deps()


def make_config(**overrides):
    values = dict(
        enabled=True,
        c_selection="f_hybrid_upper",
        c_source="test",
        c_metric="standardized_l2",
        c_candidate_multiplier=5,
        c_class_balance=True,
        random_state=42,
        n_estimators_finetune=2,
        upper_final_epochs=10,
        upper_selection="f_hybrid_upper",
        upper_alpha_grid="0,0.25,0.5,0.75,1.0",
        upper_pool_ratio=0.5,
        upper_density_weight=0.25,
        upper_mmd_weight=0.35,
        upper_distance_weight=0.40,
        rollback_gate="validation",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hybrid_upper_selects_from_target_pool_without_test_labels():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.array([[0.0], [0.1], [0.2], [8.0], [9.0], [10.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[9.2], [9.8]]),
        metric="standardized_l2",
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        target_mean=np.array([9.5]),
        target_var=np.array([0.09]),
        train_reference=X,
        candidate_scores=np.array([0.9, 0.8, 0.7, 0.3, 0.1, 0.2]),
        candidate_pool=np.array([4, 5, 3]),
        shift_score=0.4,
        upper_pool_size=3,
    )

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=2,
        seed=123,
        config=make_config(),
        selection_context=context,
        global_indices=np.arange(len(y)),
        use_upper=True,
    )

    assert result.qry_idx.shape == (2,)
    assert set(result.qry_idx.tolist()).issubset({3, 4, 5})
    assert result.label_coverage_ok is True
    assert result.used_upper is True


def test_stage1_default_hybrid_uses_f_mmd_not_upper_pool():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.array([[0.0], [0.1], [0.2], [8.0], [9.0], [10.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[9.2], [9.8]]),
        metric="standardized_l2",
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        target_mean=np.array([9.5]),
        target_var=np.array([0.09]),
        train_reference=X,
        candidate_scores=np.array([0.9, 0.8, 0.7, 0.3, 0.1, 0.2]),
        candidate_pool=np.array([4, 5, 3]),
    )

    result = chunk_ttt._build_classification_meta_batch(
        classifier=SimpleNamespace(
            norm_methods=None,
            feat_shuffle_method="latin",
            class_shuffle_method="shift",
            outlier_threshold=4.0,
        ),
        X_chunk=X,
        y_chunk=y,
        config=make_config(),
        query_size=2,
        epoch_seed=123,
        chunk_idx=0,
        selection_context=context,
        global_indices=np.arange(len(y)),
        use_upper=False,
    )

    assert result.skip_reason is None
    assert result.used_upper is False


def test_hybrid_upper_falls_back_to_f_mmd_then_records_reason():
    y = np.array([0, 0, 0, 1])
    X = np.array([[0.0], [0.1], [0.2], [10.0]])
    context = chunk_ttt.CSelectionContext(
        reference=np.array([[10.0]]),
        metric="standardized_l2",
        target_mean=np.array([10.0]),
        target_var=np.array([0.0]),
        train_reference=X,
        candidate_scores=np.array([0.9, 0.8, 0.7, 0.1]),
        candidate_pool=np.array([3]),
    )

    result = chunk_ttt._select_ctx_query_for_chunk(
        y,
        X,
        query_size=1,
        seed=123,
        config=make_config(c_class_balance=False),
        selection_context=context,
        global_indices=np.arange(len(y)),
        use_upper=True,
    )

    assert result.qry_idx.shape == (1,)
    assert result.ctx_idx.shape == (3,)
    assert result.label_coverage_ok is True
    assert "fallback" in result.split_strategy
    assert result.fallback_reason is not None


def test_interpolate_state_dict_preserves_keys_and_integer_buffers():
    theta0 = {
        "weight": torch.tensor([0.0, 2.0]),
        "counter": torch.tensor([1], dtype=torch.int64),
    }
    theta1 = {
        "weight": torch.tensor([2.0, 4.0]),
        "counter": torch.tensor([3], dtype=torch.int64),
    }

    out = chunk_ttt._interpolate_state_dict(theta0, theta1, 0.5)

    assert set(out) == {"weight", "counter"}
    torch.testing.assert_close(out["weight"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(out["counter"], torch.tensor([3], dtype=torch.int64))


def test_upper_result_row_schema_defaults_are_present():
    row = chunk_ttt.ResultRow(
        dataset_name="dummy",
        dataset_dir="dummy",
        task_type="binclass",
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status="fail",
        error="x",
    )

    data = chunk_ttt.asdict(row)
    assert data["ttt_upper_enabled"] is False
    assert data["ttt_upper_pool_size"] == 0
    assert data["ttt_upper_stage2_steps"] == 0
    assert data["ttt_upper_used_test_features"] is False


def test_upper_default_config_is_test_feature_upper_bound(tmp_path):
    parser = chunk_ttt.build_arg_parser()
    args = parser.parse_args(["--out-dir", str(tmp_path)])
    config = chunk_ttt.build_ttt_config(args)

    assert config.c_selection == "f_hybrid_upper"
    assert config.c_source == "test"
    assert config.c_metric == "standardized_l2"
    assert config.query_ratio == 0.3
    assert config.upper_final_epochs == 10
    assert config.rollback_gate == "validation"
