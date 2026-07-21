from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "summarize_openmlcc18_tfm_matrix.py"


def load_module():
    name = "summarize_openmlcc18_for_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_view_manifest(path: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "included_count": len(names),
                "included": [{"dataset_name": name} for name in names],
            }
        ),
        encoding="utf-8",
    )


def write_cell(
    root: Path,
    *,
    model: str,
    method: str,
    rows: list[dict[str, object]],
) -> None:
    out = root / model / method / "seed42"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "all_classification_results.csv", index=False)


def result_row(
    name: str,
    *,
    accuracy: float,
    balanced_accuracy: float,
    method: str,
    model: str,
    oom: bool = False,
) -> dict[str, object]:
    adapted = method != "infer"
    return {
        "dataset_name": name,
        "status": "ok",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "ttt_applied": adapted,
        "ttt_oom_fallback": oom,
        "ttt_fallback_reason": "oom" if oom else "",
        "ttt_c_fallback_reason": "",
        "ttt_c_selection": (
            "random"
            if method == "ft"
            else ("f_test_centroid_reserve" if adapted else "")
        ),
        "ttt_c_metric": (
            "tabicl_encoded_l2" if model == "tabicl-v2" else "raw_l2"
        ) if method == "faware_ft" else "",
    }


def populate_matrix(root: Path, names: list[str], *, with_oom: bool) -> None:
    for model in ("tabicl-v2", "tabpfn-v3"):
        for method, offset in (("infer", 0.0), ("ft", 0.1), ("faware_ft", 0.15)):
            rows = [
                result_row(
                    name,
                    accuracy=0.5 + offset + index * 0.01,
                    balanced_accuracy=0.45 + offset + index * 0.01,
                    method=method,
                    model=model,
                    oom=with_oom and model == "tabicl-v2" and method == "ft" and index == 1,
                )
                for index, name in enumerate(names)
            ]
            write_cell(root, model=model, method=method, rows=rows)
    (root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "models": ["tabicl-v2", "tabpfn-v3"],
                "methods": ["infer", "ft", "faware_ft"],
            }
        ),
        encoding="utf-8",
    )


def test_paired_summary_excludes_fallback_rows(tmp_path: Path):
    module = load_module()
    names = ["a", "b"]
    populate_matrix(tmp_path, names, with_oom=True)
    joined = module.load_joined(matrix_root=tmp_path, expected_names=names, random_state=42)
    coverage = module.coverage_summary(joined)
    tabicl_ft = coverage[
        coverage["model"].eq("tabicl-v2") & coverage["method"].eq("ft")
    ].iloc[0]
    assert tabicl_ft["ok_rows"] == 2
    assert tabicl_ft["eligible_rows"] == 1
    assert tabicl_ft["ttt_oom_fallback_true"] == 1

    detail, summary = module.paired_results(
        joined,
        random_state=42,
        bootstrap_resamples=200,
    )
    row = summary[
        summary["model"].eq("tabicl-v2")
        & summary["comparison"].eq("ft_minus_infer")
        & summary["metric"].eq("accuracy")
    ].iloc[0]
    assert row["paired_rows"] == 1
    assert abs(row["delta_mean"] - 0.1) < 1e-12
    assert set(detail[detail["model"].eq("tabicl-v2") & detail["comparison"].eq("ft_minus_infer")]["dataset_name"]) == {"a"}
    failures = module.failure_rows(joined)
    assert ((failures["model"] == "tabicl-v2") & (failures["method"] == "ft") & (failures["dataset_name"] == "b")).any()


def test_smoke_contract_accepts_complete_native_metric_matrix(tmp_path: Path):
    module = load_module()
    names = ["a", "b", "c"]
    populate_matrix(tmp_path, names, with_oom=False)
    joined = module.load_joined(matrix_root=tmp_path, expected_names=names, random_state=42)
    assert module.smoke_errors(joined=joined, matrix_root=tmp_path, expected_count=3) == []


def test_missing_result_file_is_not_paired(tmp_path: Path):
    module = load_module()
    names = ["a"]
    populate_matrix(tmp_path, names, with_oom=False)
    (tmp_path / "tabpfn-v3" / "faware_ft" / "seed42" / "all_classification_results.csv").unlink()
    joined = module.load_joined(matrix_root=tmp_path, expected_names=names, random_state=42)
    _, summary = module.paired_results(joined, random_state=42, bootstrap_resamples=10)
    row = summary[
        summary["model"].eq("tabpfn-v3")
        & summary["comparison"].eq("faware_ft_minus_ft")
        & summary["metric"].eq("accuracy")
    ].iloc[0]
    assert row["paired_rows"] == 0
