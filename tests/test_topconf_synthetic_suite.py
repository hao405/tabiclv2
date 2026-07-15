from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_topconf_synthetic_suite.py"
SUMMARY_PATH = REPO_ROOT / "scripts" / "summarize_topconf_synthetic_matrix.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_smoke(generator, root: Path) -> None:
    args = generator.build_parser().parse_args(
        [
            "--data-root",
            str(root),
            "--preset",
            "smoke",
            "--seeds",
            "42",
            "--n-train",
            "64",
            "--n-test",
            "32",
            "--n-features",
            "6",
        ]
    )
    assert generator.generate(args) == 0


def test_smoke_suite_count_contract_and_paired_invariants(tmp_path: Path):
    generator = load_module(GENERATOR_PATH, "topconf_generator_for_contract_test")
    root = tmp_path / "suite"
    generate_smoke(generator, root)

    manifest = pd.read_csv(root / "synthetic_manifest.csv")
    assert len(manifest) == 8
    assert set(manifest["generator_family"]) == set(generator.GENERATOR_CHOICES)
    assert set(manifest["variant"]) == {"clean", "covariate_mild"}
    assert manifest["dataset_name"].is_unique

    for _, base_group in manifest.groupby("base_task_id"):
        clean_dir = Path(base_group.loc[base_group["variant"] == "clean", "dataset_dir"].iloc[0])
        stress_dir = Path(
            base_group.loc[base_group["variant"] == "covariate_mild", "dataset_dir"].iloc[0]
        )
        for filename in ("N_train.npy", "y_train.npy", "y_test.npy"):
            np.testing.assert_array_equal(np.load(clean_dir / filename), np.load(stress_dir / filename))
        assert not np.array_equal(
            np.load(clean_dir / "N_test.npy"), np.load(stress_dir / "N_test.npy")
        )
        assert (clean_dir / "info.json").exists()


def test_suite_generation_is_reproducible(tmp_path: Path):
    generator = load_module(GENERATOR_PATH, "topconf_generator_for_repro_test")
    left = tmp_path / "left"
    right = tmp_path / "right"
    generate_smoke(generator, left)
    generate_smoke(generator, right)

    left_manifest = pd.read_csv(left / "synthetic_manifest.csv")
    right_manifest = pd.read_csv(right / "synthetic_manifest.csv")
    assert left_manifest["dataset_name"].tolist() == right_manifest["dataset_name"].tolist()
    for dataset_name in left_manifest["dataset_name"]:
        for filename in ("N_train.npy", "N_test.npy", "y_train.npy", "y_test.npy"):
            np.testing.assert_array_equal(
                np.load(left / dataset_name / filename),
                np.load(right / dataset_name / filename),
            )


def test_paper_variant_stress_is_nested_and_train_unchanged():
    generator = load_module(GENERATOR_PATH, "topconf_generator_for_stress_test")
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(100, 8))
    X_test = rng.normal(size=(200, 8))
    mild = generator.apply_test_variant(
        X_train, X_test, variant="missing_mild", stress_seed=99
    )
    severe = generator.apply_test_variant(
        X_train, X_test, variant="missing_severe", stress_seed=99
    )
    assert np.all(~np.isnan(mild) | np.isnan(severe))
    np.testing.assert_array_equal(X_train, X_train.copy())

    rotated = generator.apply_test_variant(
        X_train, X_test, variant="rotation_severe", stress_seed=99
    )
    assert rotated.shape == X_test.shape
    assert np.isfinite(rotated).all()
    assert not np.allclose(rotated, X_test)


def test_paper512_resolves_to_exactly_512_datasets():
    generator = load_module(GENERATOR_PATH, "topconf_generator_for_512_test")
    args = generator.build_parser().parse_args(
        [
            "--data-root",
            "/tmp/not-created",
            "--preset",
            "paper512",
            "--seeds",
            "42 43 44 45 46 47 48 49",
        ]
    )
    specs = generator.build_base_specs(args)
    variants = generator.variants_for_preset(args.preset)
    assert len(specs) == 64
    assert len(variants) == 8
    assert len(specs) * len(variants) == 512


def test_matrix_summary_keeps_missing_cells_visible(tmp_path: Path):
    summary = load_module(SUMMARY_PATH, "topconf_summary_for_test")
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "task_clean",
                "base_task_id": "base",
                "variant": "clean",
                "stress_axis": "none",
                "stress_level": "clean",
                "generator_family": "scm_linear",
            }
        ]
    )
    manifest_path = tmp_path / "synthetic_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    matrix_root = tmp_path / "matrix"
    result_dir = matrix_root / "tabicl-v1.1" / "infer" / "seed42"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_name": "task_clean",
                "status": "ok",
                "accuracy": 0.75,
                "balanced_accuracy": 0.7,
            }
        ]
    ).to_csv(result_dir / "all_classification_results.csv", index=False)

    joined = summary.load_joined(manifest_path, matrix_root, 42)
    coverage = summary.coverage_summary(joined)
    assert len(joined) == 12
    assert int(coverage["ok_rows"].sum()) == 1
    assert int(coverage["missing_rows"].sum()) == 11


def test_selected_faware_root_joins_reference_and_computes_direct_stage_delta(
    tmp_path: Path,
):
    summary = load_module(SUMMARY_PATH, "topconf_summary_for_reference_test")
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "task_clean",
                "base_task_id": "base_a",
                "variant": "clean",
                "stress_axis": "none",
                "stress_level": "clean",
                "generator_family": "scm_linear",
            },
            {
                "dataset_name": "task_shift",
                "base_task_id": "base_a",
                "variant": "covariate_mild",
                "stress_axis": "covariate",
                "stress_level": "mild",
                "generator_family": "scm_linear",
            },
        ]
    )
    manifest_path = tmp_path / "synthetic_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    primary_root = tmp_path / "primary"
    reference_root = tmp_path / "reference"

    values = {
        "tabicl-v2": {
            "infer": [0.50, 0.60],
            "ft": [0.60, 0.55],
            "faware_ft": [0.70, 0.65],
        },
        "tabpfn-v3": {
            "infer": [0.40, 0.50],
            "ft": [0.45, 0.55],
            "faware_ft": [0.48, 0.60],
        },
    }
    for model, methods in values.items():
        for method, scores in methods.items():
            root = primary_root if method == "faware_ft" else reference_root
            result_dir = root / model / method / "seed42"
            result_dir.mkdir(parents=True)
            rows = []
            for dataset_name, score in zip(manifest["dataset_name"], scores):
                row = {
                    "dataset_name": dataset_name,
                    "status": "ok",
                    "accuracy": score,
                    "balanced_accuracy": score,
                    "roc_auc": score,
                    "log_loss": 1.0 - score,
                    "fit_seconds": 1.0,
                    "predict_seconds": 0.1,
                }
                if method == "faware_ft":
                    row.update(
                        {
                            "ttt_applied": True,
                            "ttt_oom_fallback": False,
                            "ttt_fallback_reason": "",
                            "ttt_c_selection": "f_mmd",
                            "ttt_c_source": "test",
                            "ttt_c_metric": (
                                "tabicl_encoded_l2" if model == "tabicl-v2" else "raw_l2"
                            ),
                            "ttt_c_fallback_reason": "",
                        }
                    )
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                result_dir / "all_classification_results.csv", index=False
            )

    joined = summary.load_joined(
        manifest_path,
        primary_root,
        42,
        models=("tabicl-v2", "tabpfn-v3"),
        methods=("infer", "ft", "faware_ft"),
        reference_matrix_root=reference_root,
        reference_methods=("infer", "ft"),
    )
    assert len(joined) == 12
    assert set(joined.loc[joined["method"] == "faware_ft", "result_source_role"]) == {
        "primary"
    }
    assert set(joined.loc[joined["method"] != "faware_ft", "result_source_role"]) == {
        "reference"
    }
    summary.validate_model_native_faware(joined)

    detail, stage_summary = summary.stage_transitions(
        joined, bootstrap_samples=100, bootstrap_seed=7
    )
    assert len(detail) == 8
    row = detail[
        detail["model"].eq("tabicl-v2")
        & detail["dataset_name"].eq("task_clean")
        & detail["comparison"].eq("ft_to_faware")
    ].iloc[0]
    assert row["accuracy_delta"] == pytest.approx(0.10)
    all_row = stage_summary[
        stage_summary["model"].eq("tabicl-v2")
        & stage_summary["comparison"].eq("ft_to_faware")
        & stage_summary["variant"].eq("all")
    ].iloc[0]
    assert int(all_row["accuracy_paired_rows"]) == 2
    assert int(all_row["accuracy_wins"]) == 2
    assert int(all_row["accuracy_losses"]) == 0

    out_dir = tmp_path / "analysis"
    args = summary.build_parser().parse_args(
        [
            "--manifest-csv",
            str(manifest_path),
            "--matrix-root",
            str(primary_root),
            "--out-dir",
            str(out_dir),
            "--models",
            "tabicl-v2",
            "tabpfn-v3",
            "--methods",
            "faware_ft",
            "--reference-matrix-root",
            str(reference_root),
            "--reference-methods",
            "infer",
            "ft",
            "--bootstrap-samples",
            "100",
            "--validate-model-native-faware",
        ]
    )
    assert summary.summarize(args) == 0
    assert (out_dir / "stage_transition_detail.csv").exists()
    assert (out_dir / "stage_transition_summary.csv").exists()
    assert (out_dir / "telemetry_audit.csv").exists()
