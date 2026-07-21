from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "recover_openmlcc18_peft_matrix.py"


@pytest.fixture(scope="module")
def recovery():
    spec = importlib.util.spec_from_file_location("recover_openmlcc18_peft_matrix", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIELDS = [
    "dataset_name",
    "status",
    "error",
    "model_family",
    "peft_method",
    "ttt_applied",
    "ttt_oom_fallback",
    "peft_trainable_params",
    "accuracy",
    "balanced_accuracy",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ok_row(name: str, model: str, method: str) -> dict[str, object]:
    return {
        "dataset_name": name,
        "status": "ok",
        "error": "",
        "model_family": model,
        "peft_method": method,
        "ttt_applied": "True",
        "ttt_oom_fallback": "False",
        "peft_trainable_params": "10",
        "accuracy": "0.5",
        "balanced_accuracy": "0.5",
    }


def build_fixture(tmp_path: Path, recovery):
    names = [f"OpenML-ID-{index}" for index in range(67)]
    missing = names[-7:]
    view = tmp_path / "view"
    view.mkdir()
    (view / "dataset_view_manifest.json").write_text(
        json.dumps({"included_count": 67, "included": [{"dataset_name": n} for n in names]})
    )
    matrix_root = tmp_path / "results" / "PEFT" / "_matrix" / "run"
    matrix_root.mkdir(parents=True)
    trials = []
    for model, method in recovery.EXPECTED_CELLS:
        cell = tmp_path / "results" / "PEFT" / model / method / "run"
        rows = [ok_row(name, model, method) for name in names]
        if model == "tabiclv2":
            rows = [row for row in rows if row["dataset_name"] not in missing]
            rows.append(
                {
                    **ok_row("__WORKER_EXIT__0", model, method),
                    "status": "fail",
                    "error": "worker exited with exitcode=-15",
                    "ttt_applied": "False",
                }
            )
        if model == "tabpfnv3" and method == "last_layers":
            for row in rows:
                if row["dataset_name"] in names[-9:-7]:
                    row.update(
                        {
                            "status": "fail",
                            "error": "AssertionError: ",
                            "ttt_applied": "False",
                            "peft_trainable_params": "2567808",
                        }
                    )
        write_csv(cell / "all_classification_results.csv", rows)
        (cell / "summary.txt").write_text("old\n")
        (cell / "run_config.json").write_text(
            json.dumps({"status": "fail", "model_family": model, "peft_method": method})
        )
        command = [
            sys.executable,
            str(SCRIPT),
            "--model-family",
            model,
            "--ttt-peft-method",
            method,
            "--data-root",
            "old-view",
            "--out-dir",
            str(cell),
            "--workers",
            "2",
            "--gpu-groups",
            "1;2",
        ]
        trials.append(
            {
                "model_family": model,
                "peft_method": method,
                "output_dir": str(cell),
                "command": command,
                "status": "fail",
            }
        )
    manifest = matrix_root / "matrix_manifest.json"
    manifest.write_text(json.dumps({"status": "fail", "trials": trials}))
    return names, missing, view, matrix_root, manifest


def test_oom_classifier_uses_flag_and_exception_text(recovery):
    assert recovery.is_oom_row({"ttt_oom_fallback": "True", "error": ""})
    assert recovery.is_oom_row(
        {"ttt_oom_fallback": "False", "error": "TabPFNCUDAOutOfMemoryError: CUDA out of memory"}
    )
    assert not recovery.is_oom_row(
        {"ttt_oom_fallback": "False", "error": "AssertionError: "}
    )
def test_current_snapshot_expands_to_23_targets(tmp_path: Path, recovery):
    names, missing, _, _, manifest = build_fixture(tmp_path, recovery)
    _, _, plans = recovery.build_plans(
        matrix_manifest=manifest, expected_names=names, repo_root=tmp_path
    )
    assert sum(len(plan.retry_names) for plan in plans) == 23
    for plan in plans[:3]:
        assert plan.retry_names == missing
    assert plans[3].retry_names == []
    assert plans[4].retry_names == names[-9:-7]
    assert plans[5].retry_names == []


def test_skip_40996_leaves_only_two_tabpfn_targets(tmp_path: Path, recovery):
    names, _, _, _, manifest = build_fixture(tmp_path, recovery)
    _, _, plans = recovery.build_plans(
        matrix_manifest=manifest, expected_names=names, repo_root=tmp_path
    )
    # The synthetic fixture uses different task names, so first map its three
    # TabICL missing targets to the production skip name contract.
    production_skip = set(plans[0].retry_names)
    filtered = recovery.exclude_datasets_from_plans(plans, production_skip)
    assert all(not plan.retry_names for plan in filtered[:3])
    assert sum(len(plan.retry_names) for plan in filtered) == 2


def test_command_rewrites_only_execution_scope(tmp_path: Path, recovery):
    command = [
        "python",
        "runner.py",
        "--data-root",
        "old",
        "--out-dir",
        "old-out",
        "--workers",
        "2",
        "--gpu-groups",
        "1;2",
        "--ttt-epochs",
        "30",
        "--n-estimators",
        "32",
    ]
    rewritten = recovery.recovery_command(
        source_command=command,
        data_root=tmp_path / "view",
        out_dir=tmp_path / "out",
        physical_gpu=1,
    )
    assert rewritten[rewritten.index("--workers") + 1] == "1"
    assert rewritten[rewritten.index("--gpu-groups") + 1] == "1"
    assert rewritten[rewritten.index("--ttt-epochs") + 1] == "30"
    assert rewritten[rewritten.index("--n-estimators") + 1] == "32"


def test_merge_adds_missing_and_replaces_only_success(tmp_path: Path, recovery):
    names, missing, _, _, manifest = build_fixture(tmp_path, recovery)
    _, _, plans = recovery.build_plans(
        matrix_manifest=manifest, expected_names=names, repo_root=tmp_path
    )
    tabicl = plans[0]
    recovered_missing = {
        name: ok_row(name, tabicl.model, tabicl.method) for name in tabicl.retry_names
    }
    recovered_missing[missing[0]].update(
        {"status": "fail", "error": "OutOfMemoryError", "ttt_applied": "False"}
    )
    _, rows, replaced, retained = recovery.prepare_merged_rows(
        plan=tabicl, expected_names=names, recovered=recovered_missing
    )
    assert len(rows) == 67
    assert set(replaced) == set(missing)
    assert retained == []
    assert next(row for row in rows if row["dataset_name"] == missing[0])["status"] == "fail"

    tabpfn = plans[4]
    recovered_failed = {
        name: {
            **ok_row(name, tabpfn.model, tabpfn.method),
            "status": "fail",
            "error": "AssertionError: ",
            "ttt_applied": "False",
        }
        for name in tabpfn.retry_names
    }
    _, rows, replaced, retained = recovery.prepare_merged_rows(
        plan=tabpfn, expected_names=names, recovered=recovered_failed
    )
    assert replaced == []
    assert retained == tabpfn.retry_names
    assert all(
        next(row for row in rows if row["dataset_name"] == name)["error"]
        == "AssertionError: "
        for name in tabpfn.retry_names
    )


def test_transaction_is_atomic_and_preserves_backup(tmp_path: Path, recovery):
    names, _, _, _, manifest = build_fixture(tmp_path, recovery)
    _, _, plans = recovery.build_plans(
        matrix_manifest=manifest, expected_names=names, repo_root=tmp_path
    )
    plan = plans[0]
    recovered = {
        name: ok_row(name, plan.model, plan.method) for name in plan.retry_names
    }
    result = recovery.apply_cell_transaction(
        plan=plan,
        expected_names=names,
        recovered=recovered,
        recovery_root=tmp_path / "recovery",
    )
    assert result["rows"] == 67
    assert plan.result_csv.with_name(
        "all_classification_results.csv.before_targeted_recovery"
    ).is_file()
    _, rows = recovery.read_csv(plan.result_csv)
    assert len(rows) == 67
    journal = json.loads(
        (
            tmp_path
            / "recovery"
            / "transactions"
            / plan.model
            / plan.method
            / "journal.json"
        ).read_text()
    )
    assert journal["status"] == "committed"
