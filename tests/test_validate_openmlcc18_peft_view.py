from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_openmlcc18_peft_view.py"


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location(
        "validate_openmlcc18_peft_view",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "info.json").write_text(
        json.dumps({"name": dataset_dir.name, "task_type": "classification"}),
        encoding="utf-8",
    )
    for split, rows in (("train", 6), ("val", 2), ("test", 2)):
        np.save(dataset_dir / f"N_{split}.npy", np.ones((rows, 2)))
        np.save(dataset_dir / f"C_{split}.npy", np.empty((rows, 0)))
        np.save(dataset_dir / f"y_{split}.npy", np.arange(rows) % 2)


def _make_view(
    tmp_path: Path,
    names: tuple[str, ...] = ("OpenML-ID-1", "OpenML-ID-2"),
) -> tuple[Path, Path]:
    source = tmp_path / "openml_cc18"
    view = tmp_path / "results" / "dataset_views" / "openml_cc18_max10"
    source.mkdir()
    view.mkdir(parents=True)
    for name in names:
        _write_dataset(source / name)
        (view / name).symlink_to(source / name, target_is_directory=True)
    manifest = {
        "source_root": str(source),
        "effective_root": str(view),
        "max_classes": 10,
        "included_count": len(names),
        "included": [{"dataset_name": name} for name in names],
    }
    (view / "dataset_view_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return source, view


def _audit(validator, view: Path, expected: int = 2):
    return validator.audit_openmlcc18_peft_view(
        view,
        expected_datasets=expected,
    )


def test_valid_view_returns_compact_success_audit(validator, tmp_path: Path):
    source, view = _make_view(tmp_path)

    audit = _audit(validator, view)

    assert audit["valid"] is True
    assert audit["source_root"] == str(source.resolve())
    assert audit["included_datasets"] == 2
    assert audit["unique_datasets"] == 2
    assert audit["validated_datasets"] == 2
    assert audit["errors"] == []


def test_duplicate_dataset_names_are_rejected(validator, tmp_path: Path):
    _, view = _make_view(tmp_path)
    manifest_path = view / "dataset_view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["included"][1]["dataset_name"] = manifest["included"][0]["dataset_name"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert audit["duplicate_names"] == ["OpenML-ID-1"]


def test_missing_symlink_is_rejected(validator, tmp_path: Path):
    _, view = _make_view(tmp_path)
    (view / "OpenML-ID-1").unlink()

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert audit["missing_links"] == ["OpenML-ID-1"]


def test_missing_required_array_is_rejected(validator, tmp_path: Path):
    source, view = _make_view(tmp_path)
    (source / "OpenML-ID-1" / "y_val.npy").unlink()

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert audit["missing_files"] == {"OpenML-ID-1": ["y_val.npy"]}


def test_wrong_expected_count_is_rejected(validator, tmp_path: Path):
    _, view = _make_view(tmp_path)

    audit = _audit(validator, view, expected=67)

    assert audit["valid"] is False
    assert any("expected 67" in error for error in audit["errors"])


def test_wrong_class_limit_is_rejected(validator, tmp_path: Path):
    _, view = _make_view(tmp_path)
    manifest_path = view / "dataset_view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["max_classes"] = 11
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert any("max_classes" in error for error in audit["errors"])


def test_extra_dataset_directory_is_rejected(validator, tmp_path: Path):
    _, view = _make_view(tmp_path)
    (view / "OpenML-ID-extra").mkdir()

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert audit["extra_directories"] == ["OpenML-ID-extra"]


def test_wrong_symlink_target_is_rejected(validator, tmp_path: Path):
    source, view = _make_view(tmp_path)
    link = view / "OpenML-ID-1"
    link.unlink()
    link.symlink_to(source / "OpenML-ID-2", target_is_directory=True)

    audit = _audit(validator, view)

    assert audit["valid"] is False
    assert audit["invalid_links"] == ["OpenML-ID-1"]


def test_cli_prints_json_and_returns_contract_status(
    validator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    _, view = _make_view(tmp_path)

    status = validator.main(
        ["--data-root", str(view), "--expected-datasets", "2"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["valid"] is True
