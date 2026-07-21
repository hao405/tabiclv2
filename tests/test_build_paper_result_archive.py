from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_paper_result_archive import (
    Candidate,
    LocalSourceBackend,
    MatrixCell,
    SourceObservation,
    audit_candidate,
    build_inventory,
    classify_status,
    comparison_rows,
    expected_cells,
    infer_method,
    is_metric_column,
    metric_markdown,
    metric_rows,
    ranking_rows,
    remote_candidate_relevant,
    replace_managed_output,
    select_candidates,
    sha256_bytes,
)


def candidate(
    cell: MatrixCell,
    rows: list[dict[str, object]],
    *,
    path: str = "/tmp/formal/all_classification_results.csv",
    active: bool = False,
) -> Candidate:
    frame = pd.DataFrame(rows)
    data = frame.to_csv(index=False).encode()
    ok = frame["status"].astype(str).str.lower().eq("ok") if "status" in frame else pd.Series(True, index=frame.index)
    accuracy = pd.to_numeric(frame.loc[ok, "accuracy"], errors="coerce").dropna()
    return Candidate(
        source_host="local",
        source_path=path,
        cell=cell,
        data=data,
        frame=frame,
        observation=SourceObservation(len(data), 1.0, sha256_bytes(data)),
        active=active,
        result_rows=len(frame),
        unique_datasets=frame["dataset_name"].nunique(),
        status_ok=int(ok.sum()),
        mean_accuracy=float(accuracy.mean()) if len(accuracy) else float("-inf"),
    )


def basic_rows(count: int, *, accuracy: float = 0.8, applied: bool | None = None):
    rows = []
    for index in range(count):
        row = {
            "dataset_name": f"d{index}",
            "status": "ok",
            "accuracy": accuracy,
            "balanced_accuracy": accuracy - 0.1,
        }
        if applied is not None:
            row["ttt_applied"] = applied
        rows.append(row)
    return rows


def test_expected_matrix_has_72_unique_cells():
    cells = expected_cells()
    assert len(cells) == 72
    assert len(set(cells)) == 72


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/x/infer_baseline/run/all_classification_results.csv", "Infer"),
        ("/x/ft/run/all_classification_results.csv", "FT"),
        ("/x/faware_ft/run/all_classification_results.csv", "F-aware FT"),
        ("/x/lora/run/all_classification_results.csv", "LoRA"),
        ("/x/ln_head_embedding/run/all_classification_results.csv", "ln_head_embedding"),
        ("/x/last_layers/run/all_classification_results.csv", "Last-block"),
        ("/x/localpfn/run/all_classification_results.csv", "LoCalPFN"),
        ("/x/micp/run/all_classification_results.csv", "MICP"),
    ],
)
def test_method_aliases(path, expected):
    assert infer_method(path, None) == expected


def test_ft_baseline_inside_faware_parent_is_still_ft():
    path = "/x/tabicl/v2/ft/faware_c_ft_baseline_seed_sweep/seed8/all_classification_results.csv"
    assert infer_method(path, pd.DataFrame(basic_rows(2, applied=True))) == "FT"


def test_persisted_selector_overrides_best_para_directory_name():
    frame = pd.DataFrame(
        [{"dataset_name": "a", "ttt_c_selection": "random", "ttt_applied": True}]
    )
    assert (
        infer_method(
            "/x/tabpfn/v3/tabpfnv3_best_para/all_classification_results.csv",
            frame,
        )
        == "FT"
    )


def test_tabiclv2_seed_policy_selects_top_faware_and_middle_ft():
    ft_cell = MatrixCell("TabICLv2", "data184", "FT")
    faware_cell = MatrixCell("TabICLv2", "data184", "F-aware FT")
    root = "/x/tabicl_v11_v220260702_151902/tabiclv2"
    ft = []
    faware = []
    for rank, seed in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 9)):
        ft.append(
            candidate(
                ft_cell,
                basic_rows(184, accuracy=1.0 - rank / 100, applied=True),
                path=f"{root}/ft/seed{seed}/all_classification_results.csv",
            )
        )
        faware.append(
            candidate(
                faware_cell,
                basic_rows(184, accuracy=1.0 - rank / 100, applied=True),
                path=f"{root}/faware_c/seed{seed}/all_classification_results.csv",
            )
        )
    selected, _ = select_candidates(ft + faware)
    assert selected[faware_cell].selected_seeds == (1, 2, 3)
    assert selected[faware_cell].selection_policy == "top3_avg_accuracy_ok"
    assert selected[ft_cell].selected_seeds == (4, 5, 6)
    assert selected[ft_cell].selection_policy == "middle3_avg_accuracy_ok"
    assert selected[ft_cell].result_rows == 184
    assert selected[ft_cell].status_ok == 184
    assert selected[ft_cell].frame["ttt_applied"].all()


def test_remote_candidate_filter_keeps_only_paper_relevant_tabpfn_tabicl_paths():
    assert remote_candidate_relevant("/x/results/tabpfn/tabpfn_v3data184/ttt_baseline/all_classification_results.csv")
    assert remote_candidate_relevant("/x/results/tabicl/v2/seed_sweep/run/all_classification_results.csv")
    assert not remote_candidate_relevant("/x/results/tabicl/tabiclv2_prior_synthetic_eval_full/all_classification_results.csv")
    assert not remote_candidate_relevant("/x/results/PEFT/tabiclv2/lora/full_peft_data184/all_classification_results.csv")
    assert remote_candidate_relevant("/x/results/PEFT/tabiclv2/lora/openmlcc18_full/all_classification_results.csv")


def test_selector_prefers_coverage_then_status_then_accuracy():
    cell = MatrixCell("TabICLv2", "data184", "FT")
    partial = candidate(cell, basic_rows(2, accuracy=0.99, applied=True), path="/partial/all_classification_results.csv")
    full_less_ok_rows = basic_rows(3, accuracy=0.7, applied=True)
    full_less_ok_rows[-1]["status"] = "fail"
    full_less_ok = candidate(cell, full_less_ok_rows, path="/full_less_ok/all_classification_results.csv")
    full = candidate(cell, basic_rows(3, accuracy=0.71, applied=True), path="/full/all_classification_results.csv")
    selected, rejected = select_candidates([partial, full_less_ok, full])
    assert selected[cell] is full
    assert {item.source_path for item in rejected} == {
        "/partial/all_classification_results.csv",
        "/full_less_ok/all_classification_results.csv",
    }


def test_selector_prefers_exact_target_membership_over_extra_high_mean_rows():
    cell = MatrixCell("TabICLv2", "data184", "F-aware FT")
    exact = candidate(cell, basic_rows(2, accuracy=0.7, applied=True), path="/exact/all_classification_results.csv")
    extra = candidate(cell, basic_rows(3, accuracy=0.99, applied=True), path="/extra/all_classification_results.csv")
    extra.unique_datasets = 2
    extra.unexpected_names = ("extra",)
    selected, _ = select_candidates([extra, exact])
    assert selected[cell] is exact


def test_audit_counts_only_target_status_and_mean(tmp_path):
    path = tmp_path / "results/tabicl/v2/faware_c/seed42/all_classification_results.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset_name": "a", "status": "ok", "accuracy": 0.8, "ttt_applied": True},
            {"dataset_name": "b", "status": "ok", "accuracy": 0.6, "ttt_applied": True},
            {"dataset_name": "extra", "status": "ok", "accuracy": 1.0, "ttt_applied": True},
        ]
    ).to_csv(path, index=False)
    backend = LocalSourceBackend(tmp_path / "results")
    audited = audit_candidate(
        backend,
        str(path),
        {"data184": {"a", "b"}, "OpenML-CC18": set(), "Graph-SCM": set()},
        "",
    )
    assert audited.status_ok == 2
    assert audited.unique_datasets == 2
    assert audited.mean_accuracy == pytest.approx(0.7)
    assert audited.unexpected_names == ("extra",)


def test_audit_rejects_duplicate_dataset_identity(tmp_path):
    path = tmp_path / "results/tabicl/v2/ft/seed42/all_classification_results.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset_name": "a", "status": "ok", "accuracy": 0.8},
            {"dataset_name": "a", "status": "ok", "accuracy": 0.9},
        ]
    ).to_csv(path, index=False)
    audited = audit_candidate(
        LocalSourceBackend(tmp_path / "results"),
        str(path),
        {"data184": {"a"}, "OpenML-CC18": set(), "Graph-SCM": set()},
        "",
    )
    assert audited.invalid
    assert audited.rejection_reason == "duplicate_dataset_identity"


@pytest.mark.parametrize(
    ("rows", "expected", "method", "active", "status"),
    [
        (basic_rows(2), 3, "Infer", False, "partial"),
        (basic_rows(2), 3, "Infer", True, "running"),
        (basic_rows(3), 3, "Infer", False, "complete"),
        (basic_rows(3, applied=False), 3, "FT", False, "complete_with_failures"),
        (basic_rows(3, applied=True), 3, "FT", False, "complete"),
    ],
)
def test_completeness_states(rows, expected, method, active, status):
    item = candidate(MatrixCell("TabICLv2", "data184", method), rows, active=active)
    item.missing_names = tuple(f"d{i}" for i in range(item.unique_datasets, expected))
    observed, *_ = classify_status(item, expected, method)
    assert observed == status


def test_inventory_is_complete_even_when_all_cells_missing():
    names = {
        "data184": {"a"},
        "OpenML-CC18": {"b"},
        "Graph-SCM": {"c"},
    }
    inventory = build_inventory({}, names, "2026-07-19T00:00:00+00:00")
    assert len(inventory) == 72
    assert {row["overall_status"] for row in inventory} == {"missing"}


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("accuracy", True),
        ("ttt_loss", True),
        ("ttt_update_seconds", True),
        ("peft_trainable_params", True),
        ("n_train", False),
        ("ttt_lr", False),
        ("ttt_epochs", False),
        ("dataset_name", False),
        ("ttt_split_reason", False),
    ],
)
def test_metric_column_classification(column, expected):
    assert is_metric_column(column) is expected


def test_metric_rows_include_all_outcomes_but_not_configuration():
    cell = MatrixCell("TabICLv2", "data184", "FT")
    rows = [
        {
            "dataset_name": "a",
            "status": "ok",
            "accuracy": 0.8,
            "ttt_loss": 0.2,
            "fit_seconds": 2.0,
            "ttt_lr": 1e-5,
            "ttt_applied": True,
        },
        {
            "dataset_name": "b",
            "status": "ok",
            "accuracy": 0.6,
            "ttt_loss": 0.4,
            "fit_seconds": 4.0,
            "ttt_lr": 1e-5,
            "ttt_applied": True,
        },
    ]
    result = metric_rows({cell: candidate(cell, rows)})
    names = {row["metric"] for row in result}
    assert {"accuracy", "ttt_loss", "fit_seconds", "coverage_status_ok"} <= names
    assert "ttt_lr" not in names
    accuracy = next(row for row in result if row["metric"] == "accuracy")
    assert accuracy["mean"] == pytest.approx(0.7)
    assert accuracy["std"] == pytest.approx(0.1414213562)


def test_main_markdown_has_no_comparison_columns():
    rendered = metric_markdown(
        [
            {
                "dataset": "data184",
                "model": "TabICLv2",
                "method": "FT",
                "metric": "accuracy",
                "n": 2,
                "mean": 0.7,
                "std": 0.1,
                "min": 0.6,
                "max": 0.8,
                "source_path": "/x",
            }
        ]
    )
    assert "W/L/T" not in rendered
    assert "delta" not in rendered.lower()
    assert "reference_method" not in rendered


def test_comparison_uses_shared_status_ok_intersection():
    infer_cell = MatrixCell("TabICLv2", "data184", "Infer")
    ft_cell = MatrixCell("TabICLv2", "data184", "FT")
    infer = candidate(
        infer_cell,
        [
            {"dataset_name": "a", "status": "ok", "accuracy": 0.5},
            {"dataset_name": "b", "status": "ok", "accuracy": 0.7},
            {"dataset_name": "c", "status": "fail", "accuracy": 0.9},
        ],
    )
    ft = candidate(
        ft_cell,
        [
            {"dataset_name": "a", "status": "ok", "accuracy": 0.6},
            {"dataset_name": "b", "status": "fail", "accuracy": 0.9},
            {"dataset_name": "c", "status": "ok", "accuracy": 1.0},
        ],
    )
    result = comparison_rows({infer_cell: infer, ft_cell: ft})
    accuracy = next(row for row in result if row["metric"] == "accuracy")
    assert accuracy["shared_status_ok"] == 1
    assert accuracy["delta"] == pytest.approx(0.1)
    assert (accuracy["wins"], accuracy["losses"], accuracy["ties"]) == (1, 0, 0)


def test_ranking_uses_common_dataset_set():
    infer_cell = MatrixCell("TabICLv2", "data184", "Infer")
    ft_cell = MatrixCell("TabICLv2", "data184", "FT")
    infer = candidate(infer_cell, basic_rows(2, accuracy=0.5))
    ft = candidate(ft_cell, basic_rows(2, accuracy=0.7, applied=True))
    result = ranking_rows({infer_cell: infer, ft_cell: ft})
    accuracy = [row for row in result if row["metric"] == "accuracy"]
    assert len(accuracy) == 2
    ranks = {row["method"]: row["average_rank"] for row in accuracy}
    assert ranks == {"FT": 1.0, "Infer": 2.0}


def test_managed_output_refuses_unknown_files(tmp_path):
    out = tmp_path / "result"
    out.mkdir()
    (out / "user-note.txt").write_text("keep")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "README.md").write_text("generated")
    with pytest.raises(RuntimeError, match="unmanaged"):
        replace_managed_output(stage, out)
    assert (out / "user-note.txt").read_text() == "keep"
