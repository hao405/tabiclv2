from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_openmlcc18_peft_matrix.py"


def load_module():
    name = "summarize_openmlcc18_peft_matrix_for_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_fixture(root: Path, names: list[str]) -> tuple[Path, Path]:
    view = root / "dataset_view_manifest.json"
    view.write_text(
        json.dumps(
            {
                "included_count": len(names),
                "included": [{"dataset_name": name} for name in names],
            }
        ),
        encoding="utf-8",
    )
    trials = []
    for model in ("tabiclv2", "tabpfnv3"):
        for method in ("lora", "last_layers", "ln_head_embedding"):
            cell = root / "outputs" / model / method
            cell.mkdir(parents=True)
            rows = []
            for index, name in enumerate(names):
                rows.append(
                    {
                        "dataset_name": name,
                        "task_type": "binclass" if index == 0 else "multiclass",
                        "n_classes": 2 if index == 0 else 3,
                        "status": "ok",
                        "error": "",
                        "accuracy": 0.7 + index / 100,
                        "balanced_accuracy": 0.68 + index / 100,
                        "roc_auc": 0.8,
                        "log_loss": 0.4,
                        "fit_seconds": 2.0,
                        "predict_seconds": 1.0,
                        "ttt_applied": True,
                        "ttt_oom_fallback": False,
                        "ttt_fallback_reason": "",
                        "model_family": model,
                        "peft_method": method,
                        "peft_trainable_params": 10,
                        "peft_trainable_ratio": 0.01,
                    }
                )
            pd.DataFrame(rows).to_csv(
                cell / "all_classification_results.csv", index=False
            )
            config_args = {}
            if model == "tabpfnv3":
                config_args = {
                    "tabpfn_v3_binary_model_path": "checkpoints/v3_binary.ckpt",
                    "tabpfn_v3_multiclass_model_path": "checkpoints/v3_multiclass.ckpt",
                }
            (cell / "run_config.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "model_family": model,
                        "peft_method": method,
                        "args": config_args,
                    }
                ),
                encoding="utf-8",
            )
            trials.append(
                {
                    "model_family": model,
                    "peft_method": method,
                    "status": "success",
                    "output_dir": str(cell),
                }
            )
    matrix_root = root / "matrix"
    matrix_root.mkdir()
    (matrix_root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "models": ["tabiclv2", "tabpfnv3"],
                "methods": ["lora", "last_layers", "ln_head_embedding"],
                "trials": trials,
            }
        ),
        encoding="utf-8",
    )
    return matrix_root, view


def rewrite_cell(matrix_root: Path, model: str, method: str, mutate) -> None:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text())
    trial = next(
        row
        for row in manifest["trials"]
        if row["model_family"] == model and row["peft_method"] == method
    )
    csv_path = Path(trial["output_dir"]) / "all_classification_results.csv"
    frame = pd.read_csv(csv_path)
    mutate(frame)
    frame.to_csv(csv_path, index=False)


def test_complete_matrix_writes_all_artifacts(tmp_path: Path):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])
    assert module.main(
        [
            "--matrix-root",
            str(matrix),
            "--view-manifest",
            str(view),
            "--expected-datasets",
            "2",
            "--require-all-success",
        ]
    ) == 0
    out = matrix / "openmlcc18_peft_audit"
    assert {
        "coverage.csv",
        "failures.csv",
        "eligible_results.csv",
        "metric_summary.csv",
        "audit_summary.md",
    } <= {path.name for path in out.iterdir()}
    coverage = pd.read_csv(out / "coverage.csv")
    assert len(coverage) == 6
    assert coverage["eligible_rows"].tolist() == [2] * 6
    eligible = pd.read_csv(out / "eligible_results.csv")
    assert len(eligible) == 12
    assert eligible["shared_eligible"].all()
    metrics = pd.read_csv(out / "metric_summary.csv")
    assert set(metrics["scope"]) == {"eligible", "shared_eligible"}


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda frame: frame.__setitem__("status", ["fail", "ok"]), "status_not_ok"),
        (lambda frame: frame.__setitem__("ttt_applied", [False, True]), "ttt_not_applied"),
        (lambda frame: frame.__setitem__("ttt_oom_fallback", [True, False]), "oom_fallback"),
        (lambda frame: frame.__setitem__("ttt_fallback_reason", ["fallback", ""]), "fallback"),
        (lambda frame: frame.__setitem__("model_family", ["wrong", "tabiclv2"]), "wrong_model"),
        (lambda frame: frame.__setitem__("peft_method", ["wrong", "lora"]), "wrong_peft_method"),
        (
            lambda frame: frame.__setitem__("peft_trainable_params", [0, 10]),
            "nonpositive_trainable_params",
        ),
    ],
)
def test_contract_failures_are_excluded_and_strict_fails(
    tmp_path: Path, mutation, reason: str
):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])
    rewrite_cell(matrix, "tabiclv2", "lora", mutation)
    assert module.main(
        [
            "--matrix-root",
            str(matrix),
            "--view-manifest",
            str(view),
            "--require-all-success",
        ]
    ) == 1
    failures = pd.read_csv(
        matrix / "openmlcc18_peft_audit" / "failures.csv"
    )
    assert failures["audit_reason"].str.contains(reason).any()
    eligible = pd.read_csv(
        matrix / "openmlcc18_peft_audit" / "eligible_results.csv"
    )
    assert (
        eligible[
            eligible["model_family_expected"].eq("tabiclv2")
            & eligible["peft_method_expected"].eq("lora")
        ]["shared_eligible"].sum()
        == 1
    )


def test_missing_and_skipped_rows_are_reported(tmp_path: Path):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])

    def mutate(frame):
        frame.loc[0, "status"] = "skip"
        frame.drop(index=1, inplace=True)

    rewrite_cell(matrix, "tabiclv2", "last_layers", mutate)
    assert module.main(
        ["--matrix-root", str(matrix), "--view-manifest", str(view)]
    ) == 0
    coverage = pd.read_csv(
        matrix / "openmlcc18_peft_audit" / "coverage.csv"
    )
    cell = coverage[
        coverage["model_family"].eq("tabiclv2")
        & coverage["peft_method"].eq("last_layers")
    ].iloc[0]
    assert cell["skipped_rows"] == 1
    assert cell["missing_rows"] == 1
    assert cell["eligible_rows"] == 0


@pytest.mark.parametrize("kind", ["duplicate", "unexpected", "sentinel"])
def test_structural_result_corruption_is_rejected(tmp_path: Path, kind: str):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])

    def mutate(frame):
        if kind == "duplicate":
            frame.loc[1, "dataset_name"] = "binary"
        elif kind == "unexpected":
            frame.loc[1, "dataset_name"] = "alien"
        else:
            frame.loc[1, "dataset_name"] = "__WORKER_EXIT__0"

    rewrite_cell(matrix, "tabpfnv3", "lora", mutate)
    with pytest.raises(ValueError, match=kind if kind != "sentinel" else "worker sentinel"):
        module.load_matrix(matrix_root=matrix, expected_names=["binary", "multi"])


def test_wrong_matrix_order_and_expected_count_are_rejected(tmp_path: Path):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])
    payload = json.loads((matrix / "matrix_manifest.json").read_text())
    payload["trials"][0], payload["trials"][1] = payload["trials"][1], payload["trials"][0]
    (matrix / "matrix_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be exactly"):
        module.load_matrix(matrix_root=matrix, expected_names=["binary", "multi"])
    with pytest.raises(ValueError, match="expected 3"):
        module.load_expected_names(view, 3)


def test_tabpfn_checkpoint_configuration_and_route_are_audited(tmp_path: Path):
    module = load_module()
    matrix, view = write_fixture(tmp_path, ["binary", "multi"])
    manifest = json.loads((matrix / "matrix_manifest.json").read_text())
    trial = next(
        row
        for row in manifest["trials"]
        if row["model_family"] == "tabpfnv3" and row["peft_method"] == "lora"
    )
    config_path = Path(trial["output_dir"]) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["args"]["tabpfn_v3_multiclass_model_path"] = config["args"][
        "tabpfn_v3_binary_model_path"
    ]
    config_path.write_text(json.dumps(config))
    assert module.main(
        [
            "--matrix-root",
            str(matrix),
            "--view-manifest",
            str(view),
            "--require-all-success",
        ]
    ) == 1
    failures = pd.read_csv(
        matrix / "openmlcc18_peft_audit" / "failures.csv"
    )
    bad = failures[
        failures["model_family_expected"].eq("tabpfnv3")
        & failures["peft_method_expected"].eq("lora")
    ]
    assert len(bad) == 2
    assert bad["audit_reason"].str.contains("checkpoint_config").all()
