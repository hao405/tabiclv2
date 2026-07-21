from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "resume_openmlcc18_tabicl_v2_after_gpu_drop.py"


def load_module():
    name = "resume_openmlcc18_for_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(name: str, *, status: str = "ok", method: str = "infer") -> dict[str, object]:
    adapted = method != "infer"
    return {
        "dataset_name": name,
        "status": status,
        "accuracy": 0.5,
        "balanced_accuracy": 0.4,
        "ttt_applied": adapted,
        "ttt_oom_fallback": False,
        "ttt_fallback_reason": "",
        "ttt_c_fallback_reason": "",
        "ttt_c_selection": (
            "random"
            if method == "ft"
            else ("f_test_centroid_reserve" if adapted else "")
        ),
        "ttt_c_metric": "tabicl_encoded_l2" if adapted else "",
    }


def test_cell_plan_retries_missing_failed_and_ignores_worker_sentinel(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "ft" / "seed42" / "all_classification_results.csv"
    write_csv(
        result,
        [
            row("a", method="ft"),
            row("b", status="fail", method="ft"),
            {"dataset_name": "__WORKER_EXIT__0", "status": "fail"},
        ],
    )
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="ft",
        seed=42,
        expected_names=["a", "b", "c"],
    )
    assert list(plan.retained_rows) == ["a"]
    assert plan.recovery_names == ["b", "c"]


def test_merge_replaces_failed_rows_and_preserves_manifest_order(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "faware_ft" / "seed42" / "all_classification_results.csv"
    write_csv(
        result,
        [
            row("b", status="fail", method="faware_ft"),
            row("a", method="faware_ft"),
        ],
    )
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="faware_ft",
        seed=42,
        expected_names=["a", "b", "c"],
    )
    recovery = [row("c", method="faware_ft"), row("b", method="faware_ft")]
    _, merged = module.merge_rows(
        plan=plan,
        recovery_rows=recovery,
        expected_names=["a", "b", "c"],
        recovery_source=tmp_path / "recovery.csv",
    )
    assert [item["dataset_name"] for item in merged] == ["a", "b", "c"]
    assert all(item["status"] == "ok" for item in merged)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate dataset_name"),
        (lambda rows: rows.__setitem__(0, row("unexpected", method="ft")), "coverage mismatch"),
        (lambda rows: rows[0].__setitem__("ttt_applied", False), "ttt_applied"),
        (lambda rows: rows[0].__setitem__("ttt_oom_fallback", True), "OOM fallback"),
    ],
)
def test_merge_rejects_invalid_recovery(tmp_path: Path, mutator, match: str):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "ft" / "seed42" / "all_classification_results.csv"
    write_csv(result, [row("a", method="ft")])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="ft",
        seed=42,
        expected_names=["a", "b"],
    )
    recovery = [row("b", method="ft")]
    mutator(recovery)
    with pytest.raises(ValueError, match=match):
        module.merge_rows(
            plan=plan,
            recovery_rows=recovery,
            expected_names=["a", "b"],
            recovery_source=tmp_path / "recovery.csv",
        )


def test_backup_once_is_idempotent(tmp_path: Path):
    module = load_module()
    source = tmp_path / "result.csv"
    source.write_text("first\n", encoding="utf-8")
    backup = module.backup_once(source)
    assert backup is not None
    source.write_text("second\n", encoding="utf-8")
    assert module.backup_once(source) == backup
    assert backup.read_text(encoding="utf-8") == "first\n"


def test_merge_keeps_explicit_persistent_failure(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "infer" / "seed42" / "all_classification_results.csv"
    write_csv(result, [row("a")])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="infer",
        seed=42,
        expected_names=["a", "b"],
    )
    failed = row("b", status="fail")
    failed["error"] = "OutOfMemoryError: persistent"
    _, merged = module.merge_rows(
        plan=plan,
        recovery_rows=[failed],
        expected_names=["a", "b"],
        recovery_source=tmp_path / "recovery.csv",
    )
    assert [item["dataset_name"] for item in merged] == ["a", "b"]
    assert merged[1]["status"] == "fail"


def test_merge_accepts_declared_oom_fallback_but_keeps_it_auditable(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "ft" / "seed42" / "all_classification_results.csv"
    fallback = row("a", method="ft")
    fallback.update(
        {
            "ttt_applied": False,
            "ttt_oom_fallback": True,
            "ttt_fallback_reason": "TTT OOM; used original model parameters",
        }
    )
    write_csv(result, [fallback])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="ft",
        seed=42,
        expected_names=["a", "b"],
    )
    _, merged = module.merge_rows(
        plan=plan,
        recovery_rows=[row("b", method="ft")],
        expected_names=["a", "b"],
        recovery_source=tmp_path / "recovery.csv",
    )
    assert merged[0]["status"] == "ok"
    assert module.is_truthy(merged[0]["ttt_oom_fallback"])


def test_complete_cell_has_empty_recovery_set(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "infer" / "seed42" / "all_classification_results.csv"
    write_csv(result, [row("a"), row("b")])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="infer",
        seed=42,
        expected_names=["a", "b"],
    )
    assert plan.recovery_names == []
    module.validate_complete_tabicl_rows(
        method="infer",
        rows=list(plan.retained_rows.values()),
        expected_names=["a", "b"],
        source=result,
    )


def test_merge_rejects_worker_sentinel_in_recovery(tmp_path: Path):
    module = load_module()
    result = tmp_path / "tabicl-v2" / "infer" / "seed42" / "all_classification_results.csv"
    write_csv(result, [row("a")])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="infer",
        seed=42,
        expected_names=["a", "b"],
    )
    with pytest.raises(ValueError, match="worker sentinels"):
        module.merge_rows(
            plan=plan,
            recovery_rows=[row("b"), {"dataset_name": "__WORKER_EXIT__0", "status": "fail"}],
            expected_names=["a", "b"],
            recovery_source=tmp_path / "recovery.csv",
        )


def test_merge_cell_rejects_source_csv_changed_during_recovery(tmp_path: Path):
    module = load_module()
    cell_dir = tmp_path / "tabicl-v2" / "infer" / "seed42"
    result = cell_dir / "all_classification_results.csv"
    write_csv(result, [row("a")])
    plan = module.build_cell_plan(
        matrix_root=tmp_path,
        method="infer",
        seed=42,
        expected_names=["a", "b"],
    )
    write_csv(result, [row("a"), row("changed")])
    recovery_csv = tmp_path / "attempt" / "all_classification_results.csv"
    write_csv(recovery_csv, [row("b")])
    with pytest.raises(RuntimeError, match="source CSV changed"):
        module.merge_cell(
            plan=plan,
            recovery_csv=recovery_csv,
            expected_names=["a", "b"],
            attempt_dir=recovery_csv.parent,
        )


def test_interrupted_merge_transaction_can_be_reapplied(tmp_path: Path, monkeypatch):
    module = load_module()
    transaction = tmp_path / "merge_transaction"
    transaction.mkdir()
    artifacts = {}
    for name, content in (("a.txt", b"new-a"), ("b.txt", b"new-b")):
        stage = transaction / name
        stage.write_bytes(content)
        destination = tmp_path / "destination" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(b"old")
        artifacts[name] = {
            "stage": str(stage),
            "destination": str(destination),
            "sha256": module.sha256(stage),
        }
    module.atomic_write_json(
        transaction / "journal.json",
        {"status": "prepared", "completed": [], "artifacts": artifacts},
    )
    original = module.copy_stage_atomically
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected failure")
        original(source, destination)

    monkeypatch.setattr(module, "copy_stage_atomically", fail_second)
    with pytest.raises(OSError, match="injected failure"):
        module.apply_merge_transaction(transaction)
    monkeypatch.setattr(module, "copy_stage_atomically", original)
    module.apply_merge_transaction(transaction)
    assert (tmp_path / "destination" / "a.txt").read_bytes() == b"new-a"
    assert (tmp_path / "destination" / "b.txt").read_bytes() == b"new-b"
    journal = json.loads((transaction / "journal.json").read_text(encoding="utf-8"))
    assert journal["status"] == "committed"


def test_summary_manifest_is_reconstructed_without_overwriting_missing_view(tmp_path: Path):
    module = load_module()
    missing_view = tmp_path / "view" / "dataset_view_manifest.json"
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    result = module.ensure_summary_manifest(
        view_manifest=missing_view,
        recovery_root=recovery_root,
        data_root=tmp_path / "openml_cc18",
        expected_names=["a", "b"],
    )
    assert result == recovery_root / "reconstructed_dataset_view_manifest.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["included_count"] == 2
    assert [row["dataset_name"] for row in payload["included"]] == ["a", "b"]
    assert not missing_view.exists()


def test_recovery_command_keeps_32_estimators_and_original_batch_default(tmp_path: Path):
    module = load_module()
    manager = tmp_path / "matrix" / "tabicl-v2" / "infer" / "seed42" / "manager_manifest.json"
    manager.parent.mkdir(parents=True)
    manager.write_text(json.dumps({"checkpoint": "/checkpoint.ckpt"}), encoding="utf-8")
    args = Namespace(
        data_root=tmp_path / "openml_cc18",
        devices="2",
        seed=42,
        matrix_root=tmp_path / "matrix",
    )
    command = module.build_recovery_command(
        repo_root=tmp_path,
        method="infer",
        names=["a"],
        out_dir=tmp_path / "recovery",
        args=args,
    )
    assert command[command.index("--n-estimators") + 1] == "32"
    assert "--batch-size" not in command
    assert "--" not in command


def test_task_schedule_runs_small_common_preflight_across_methods_first(tmp_path: Path):
    module = load_module()
    for name, size in (("small", 1), ("large", 100)):
        dataset = tmp_path / name
        dataset.mkdir()
        (dataset / "payload").write_bytes(b"x" * size)
    plans = [
        module.CellPlan(
            method=method,
            cell_dir=tmp_path / method,
            fieldnames=["dataset_name", "status"],
            retained_rows={},
            recovery_names=["large", "small"],
            source_csv_hash="hash",
        )
        for method in module.METHODS
    ]
    preflight, schedule = module.build_task_schedule(plans=plans, data_root=tmp_path)
    assert preflight == "small"
    assert schedule[:3] == [
        ("infer", "small"),
        ("ft", "small"),
        ("faware_ft", "small"),
    ]
    assert len(schedule) == 6


def test_reusable_task_artifact_and_allocator_environment(tmp_path: Path):
    module = load_module()
    attempt = (
        tmp_path
        / module.MODEL
        / "infer"
        / "seed42"
        / "dataset-a"
        / "attempt_1"
    )
    write_csv(attempt / "validated_task_result.csv", [row("dataset-a")])
    artifact = module.find_reusable_success(
        recovery_root=tmp_path,
        method="infer",
        seed=42,
        dataset_name="dataset-a",
    )
    assert artifact is not None and artifact.reused
    assert artifact.row["status"] == "ok"
    assert module.recovery_environment()["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:True"
    )


def test_latest_task_artifact_prefers_raw_declared_fallback(tmp_path: Path):
    module = load_module()
    attempt = (
        tmp_path
        / module.MODEL
        / "ft"
        / "seed42"
        / "dataset-a"
        / "attempt_1"
    )
    fallback = row("dataset-a", method="ft")
    fallback.update(
        {
            "ttt_applied": False,
            "ttt_oom_fallback": True,
            "ttt_fallback_reason": "TTT OOM; used original model parameters",
        }
    )
    write_csv(attempt / "all_classification_results.csv", [fallback])
    failed = row("dataset-a", status="fail", method="ft")
    failed["error"] = "old strict validator rejected fallback"
    write_csv(attempt / "validated_task_result.csv", [failed])
    artifact = module.find_latest_task_artifact(
        recovery_root=tmp_path,
        method="ft",
        seed=42,
        dataset_name="dataset-a",
    )
    assert artifact is not None
    assert artifact.result_csv.name == "all_classification_results.csv"
    assert artifact.row["status"] == "ok"
    assert module.is_truthy(artifact.row["ttt_oom_fallback"])


def test_expected_initial_inventory_is_14_15_14():
    module = load_module()
    assert module.EXPECTED_INITIAL_INVENTORY == {
        "infer": 14,
        "ft": 15,
        "faware_ft": 14,
    }
