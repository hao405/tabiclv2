#!/usr/bin/env python3
"""Audit and summarize the OpenML-CC18 TabICLv2/TabPFNv3 PEFT matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


MODELS = ("tabiclv2", "tabpfnv3")
METHODS = ("lora", "last_layers", "ln_head_embedding")
EXPECTED_CELLS = tuple((model, method) for model in MODELS for method in METHODS)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "log_loss",
    "fit_seconds",
    "predict_seconds",
)
TRUTHY = {"1", "true", "yes", "y", "on"}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin(TRUTHY)


def nonempty(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def load_expected_names(view_manifest: Path, expected_datasets: int | None) -> list[str]:
    payload = _read_json(view_manifest, "view manifest")
    included = payload.get("included")
    if not isinstance(included, list):
        raise ValueError("view manifest has no included list")
    try:
        names = [str(row["dataset_name"]) for row in included]
    except (KeyError, TypeError) as exc:
        raise ValueError("every included row must have dataset_name") from exc
    if not names or any(not name.strip() for name in names):
        raise ValueError("view manifest contains an empty dataset name")
    if len(names) != len(set(names)):
        raise ValueError("view manifest contains duplicate dataset names")
    if "included_count" in payload and int(payload["included_count"]) != len(names):
        raise ValueError("view manifest included_count mismatch")
    if expected_datasets is not None and len(names) != expected_datasets:
        raise ValueError(
            f"expected {expected_datasets} datasets, view manifest contains {len(names)}"
        )
    return names


def _manifest_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise ValueError("matrix manifest has no trials list")
    cells: list[dict[str, Any]] = []
    observed: list[tuple[str, str]] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"matrix trial {index} is not an object")
        key = (str(trial.get("model_family", "")), str(trial.get("peft_method", "")))
        observed.append(key)
        cells.append(trial)
    if tuple(observed) != EXPECTED_CELLS:
        raise ValueError(
            "matrix trials must be exactly "
            f"{list(EXPECTED_CELLS)}, observed {observed}"
        )
    if "models" in payload and tuple(payload["models"]) != MODELS:
        raise ValueError(f"unexpected matrix models: {payload['models']}")
    if "methods" in payload and tuple(payload["methods"]) != METHODS:
        raise ValueError(f"unexpected matrix methods: {payload['methods']}")
    return cells


def _resolve_output_dir(manifest_path: Path, value: Any) -> Path:
    if value in (None, ""):
        raise ValueError("matrix trial has no output_dir")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent.parent.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _cell_config(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    return _read_json(path, "run config") if path.is_file() else {}


def _configured_checkpoint_audit(
    *, model: str, config: dict[str, Any]
) -> tuple[bool, str, str]:
    if model != "tabpfnv3":
        return True, "", ""
    args = config.get("args", {})
    if not isinstance(args, dict):
        args = {}
    binary = str(args.get("tabpfn_v3_binary_model_path") or "")
    multiclass = str(args.get("tabpfn_v3_multiclass_model_path") or "")
    valid = (
        bool(binary)
        and bool(multiclass)
        and binary != multiclass
        and "binary" in Path(binary).name.lower()
        and "multiclass" in Path(multiclass).name.lower()
    )
    return valid, binary, multiclass


def _row_checkpoint_ok(frame: pd.DataFrame, model: str) -> pd.Series:
    result = pd.Series(True, index=frame.index, dtype=bool)
    if model != "tabpfnv3":
        return result
    task_type = frame.get("task_type", pd.Series("", index=frame.index)).astype("string")
    n_classes = pd.to_numeric(
        frame.get("n_classes", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    route = pd.Series("", index=frame.index, dtype="string")
    for column in ("checkpoint_route", "checkpoint_path", "model_path"):
        if column in frame:
            candidate = frame[column].astype("string").fillna("").str.lower()
            route = route.mask(route.eq("") & candidate.ne(""), candidate)
    has_route = route.ne("")
    expected_binary = task_type.eq("binclass") | n_classes.eq(2)
    expected_multiclass = task_type.eq("multiclass") & n_classes.gt(2)
    valid_task = expected_binary | expected_multiclass
    result &= valid_task
    result &= ~has_route | (
        (expected_binary & route.str.contains("binary", regex=False))
        | (expected_multiclass & route.str.contains("multiclass", regex=False))
    )
    return result


def _fallback_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in frame.columns:
        lowered = column.lower()
        if lowered.endswith("fallback_reason") or lowered == "fallback_reason":
            mask |= nonempty(frame[column])
        elif lowered.endswith("_fallback") and lowered != "ttt_oom_fallback":
            mask |= truthy(frame[column])
    return mask


def _reason_strings(frame: pd.DataFrame, checks: dict[str, pd.Series]) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype="string")
    for label, bad in checks.items():
        reasons = reasons.mask(bad & reasons.eq(""), label)
        reasons = reasons.mask(bad & reasons.ne("") & ~reasons.str.contains(label), reasons + "|" + label)
    return reasons


def load_matrix(
    *, matrix_root: Path, expected_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = _read_json(manifest_path, "matrix manifest")
    trials = _manifest_cells(manifest)
    expected = pd.DataFrame(
        {"dataset_name": expected_names, "dataset_order": range(len(expected_names))}
    )
    expected_set = set(expected_names)
    joined_frames: list[pd.DataFrame] = []
    config_rows: list[dict[str, Any]] = []

    for trial, (model, method) in zip(trials, EXPECTED_CELLS):
        output_dir = _resolve_output_dir(manifest_path, trial.get("output_dir"))
        result_path = output_dir / "all_classification_results.csv"
        if result_path.is_file():
            result = pd.read_csv(result_path)
            if "dataset_name" not in result:
                raise ValueError(f"missing dataset_name column: {result_path}")
            result["dataset_name"] = result["dataset_name"].astype(str)
            sentinels = sorted(
                name for name in result["dataset_name"] if name.startswith("__WORKER_")
            )
            if sentinels:
                raise ValueError(f"worker sentinel rows in {result_path}: {sentinels}")
            duplicates = sorted(
                result.loc[result["dataset_name"].duplicated(False), "dataset_name"].unique()
            )
            if duplicates:
                raise ValueError(f"duplicate datasets in {result_path}: {duplicates}")
            unexpected = sorted(set(result["dataset_name"]) - expected_set)
            if unexpected:
                raise ValueError(f"unexpected datasets in {result_path}: {unexpected}")
            cell = expected.merge(result, on="dataset_name", how="left", validate="one_to_one")
            cell["result_present"] = cell["status"].notna() if "status" in cell else False
        else:
            cell = expected.copy()
            cell["status"] = "missing_result_file"
            cell["result_present"] = False
        cell["model_family_expected"] = model
        cell["peft_method_expected"] = method
        cell["result_csv"] = str(result_path)
        config = _cell_config(output_dir)
        config_identity_ok = (
            bool(config)
            and config.get("model_family") == model
            and config.get("peft_method") == method
        )
        configured_ok, binary_path, multiclass_path = _configured_checkpoint_audit(
            model=model, config=config
        )
        cell["cell_config_identity_ok"] = config_identity_ok
        cell["checkpoint_config_ok"] = configured_ok
        cell["checkpoint_row_ok"] = _row_checkpoint_ok(cell, model)
        joined_frames.append(cell)
        config_rows.append(
            {
                "model_family": model,
                "peft_method": method,
                "manifest_status": trial.get("status", ""),
                "output_dir": str(output_dir),
                "cell_config_identity_ok": config_identity_ok,
                "cell_config_status": config.get("status", ""),
                "checkpoint_config_ok": configured_ok,
                "binary_checkpoint": binary_path,
                "multiclass_checkpoint": multiclass_path,
            }
        )
    return pd.concat(joined_frames, ignore_index=True, sort=False), pd.DataFrame(config_rows)


def classify_rows(joined: pd.DataFrame) -> pd.DataFrame:
    frame = joined.copy()
    present = frame["result_present"].fillna(False).astype(bool)
    status = frame.get("status", pd.Series("", index=frame.index)).astype("string").fillna("")
    applied = (
        truthy(frame["ttt_applied"])
        if "ttt_applied" in frame
        else pd.Series(False, index=frame.index)
    )
    oom = (
        truthy(frame["ttt_oom_fallback"])
        if "ttt_oom_fallback" in frame
        else pd.Series(False, index=frame.index)
    )
    fallback = _fallback_mask(frame)
    actual_model = frame.get("model_family", pd.Series("", index=frame.index)).astype("string")
    actual_method = frame.get("peft_method", pd.Series("", index=frame.index)).astype("string")
    trainable = pd.to_numeric(
        frame.get("peft_trainable_params", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    ratio = pd.to_numeric(
        frame.get("peft_trainable_ratio", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    checks = {
        "missing": ~present,
        "status_not_ok": present & ~status.eq("ok"),
        "ttt_not_applied": present & ~applied,
        "oom_fallback": present & oom,
        "fallback": present & fallback,
        "wrong_model": present & actual_model.ne(frame["model_family_expected"]),
        "wrong_peft_method": present & actual_method.ne(frame["peft_method_expected"]),
        "nonpositive_trainable_params": present & ~(trainable > 0),
        "invalid_trainable_ratio": (
            (present & ~(ratio > 0)) | (present & ~ratio.map(math.isfinite))
        ),
        "cell_config_identity": present & ~frame["cell_config_identity_ok"].astype(bool),
        "checkpoint_config": present & ~frame["checkpoint_config_ok"].astype(bool),
        "checkpoint_route": present & ~frame["checkpoint_row_ok"].astype(bool),
    }
    frame["audit_reason"] = _reason_strings(frame, checks)
    frame["eligible"] = frame["audit_reason"].eq("")
    frame["oom_fallback"] = oom
    frame["fallback_detected"] = fallback
    return frame


def coverage_summary(classified: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method), cell in classified.groupby(
        ["model_family_expected", "peft_method_expected"], sort=False
    ):
        status = cell["status"].astype("string").fillna("")
        row = {
            "model_family": model,
            "peft_method": method,
            "expected_rows": len(cell),
            "present_rows": int(cell["result_present"].fillna(False).sum()),
            "ok_rows": int(status.eq("ok").sum()),
            "failed_rows": int(status.eq("fail").sum()),
            "skipped_rows": int(status.eq("skip").sum()),
            "missing_rows": int((~cell["result_present"].fillna(False)).sum()),
            "eligible_rows": int(cell["eligible"].sum()),
            "ttt_applied_rows": int(
                truthy(cell["ttt_applied"]).sum() if "ttt_applied" in cell else 0
            ),
            "oom_fallback_rows": int(cell["oom_fallback"].sum()),
            "fallback_rows": int(cell["fallback_detected"].sum()),
            "wrong_model_rows": int(cell["audit_reason"].str.contains("wrong_model").sum()),
            "wrong_method_rows": int(
                cell["audit_reason"].str.contains("wrong_peft_method").sum()
            ),
            "zero_or_missing_trainable_rows": int(
                cell["audit_reason"].str.contains("nonpositive_trainable_params").sum()
            ),
            "invalid_trainable_ratio_rows": int(
                cell["audit_reason"].str.contains("invalid_trainable_ratio").sum()
            ),
            "checkpoint_error_rows": int(
                cell["audit_reason"].str.contains("checkpoint_").sum()
            ),
            "cell_config_identity_error_rows": int(
                cell["audit_reason"].str.contains("cell_config_identity").sum()
            ),
        }
        config = configs[
            configs["model_family"].eq(model) & configs["peft_method"].eq(method)
        ].iloc[0]
        row["manifest_status"] = config["manifest_status"]
        row["checkpoint_config_ok"] = config["checkpoint_config_ok"]
        rows.append(row)
    return pd.DataFrame(rows)


def metric_summary(eligible: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    rows: list[dict[str, Any]] = []
    shared = set.intersection(
        *[
            set(cell["dataset_name"])
            for _, cell in eligible.groupby(
                ["model_family_expected", "peft_method_expected"], sort=False
            )
        ]
    ) if not eligible.empty and eligible.groupby(
        ["model_family_expected", "peft_method_expected"]
    ).ngroups == len(EXPECTED_CELLS) else set()
    for model, method in EXPECTED_CELLS:
        cell = eligible[
            eligible["model_family_expected"].eq(model)
            & eligible["peft_method_expected"].eq(method)
        ]
        for scope, scoped in (
            ("eligible", cell),
            ("shared_eligible", cell[cell["dataset_name"].isin(shared)]),
        ):
            for metric in METRICS:
                values = pd.to_numeric(
                    scoped.get(metric, pd.Series(dtype=float)), errors="coerce"
                ).dropna()
                rows.append(
                    {
                        "model_family": model,
                        "peft_method": method,
                        "scope": scope,
                        "metric": metric,
                        "n": len(values),
                        "mean": values.mean() if len(values) else float("nan"),
                        "std": values.std(ddof=1) if len(values) > 1 else float("nan"),
                    }
                )
    return pd.DataFrame(rows), shared


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    display = frame.fillna("")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [
            str(row[column]).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(
    *, output_dir: Path, classified: pd.DataFrame, configs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage_summary(classified, configs)
    failures = classified[~classified["eligible"]].copy()
    eligible = classified[classified["eligible"]].copy()
    metrics, shared = metric_summary(eligible)
    eligible["shared_eligible"] = eligible["dataset_name"].isin(shared)
    failure_columns = [
        column
        for column in (
            "model_family_expected",
            "peft_method_expected",
            "dataset_name",
            "status",
            "error",
            "audit_reason",
            "ttt_applied",
            "ttt_oom_fallback",
            "ttt_fallback_reason",
            "peft_trainable_params",
            "peft_trainable_ratio",
            "model_family",
            "peft_method",
            "result_csv",
        )
        if column in failures
    ]
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    failures[failure_columns].to_csv(output_dir / "failures.csv", index=False)
    eligible.to_csv(output_dir / "eligible_results.csv", index=False)
    metrics.to_csv(output_dir / "metric_summary.csv", index=False)
    summary = [
        "# OpenML-CC18 PEFT 2×3 Matrix Audit",
        "",
        f"Shared eligible datasets: {len(shared)}",
        f"Excluded or missing rows: {len(failures)}",
        "",
        "## Coverage and PEFT contract",
        "",
        _markdown_table(coverage),
        "",
        "## Checkpoint configuration",
        "",
        _markdown_table(configs),
        "",
        "## Metric means",
        "",
        _markdown_table(metrics),
    ]
    (output_dir / "audit_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    return coverage, failures, shared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-datasets", type=int)
    parser.add_argument("--require-all-success", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_datasets is not None and args.expected_datasets < 1:
        raise ValueError("--expected-datasets must be >= 1")
    expected_names = load_expected_names(args.view_manifest, args.expected_datasets)
    joined, configs = load_matrix(
        matrix_root=args.matrix_root, expected_names=expected_names
    )
    classified = classify_rows(joined)
    output_dir = args.output_dir or (args.matrix_root / "openmlcc18_peft_audit")
    coverage, failures, shared = write_outputs(
        output_dir=output_dir, classified=classified, configs=configs
    )
    print(f"audit_dir: {output_dir}")
    print(f"shared_eligible_datasets: {len(shared)}")
    print(f"excluded_or_missing_rows: {len(failures)}")
    if args.require_all_success:
        expected_rows = len(expected_names)
        strict_ok = (
            len(coverage) == len(EXPECTED_CELLS)
            and coverage["present_rows"].eq(expected_rows).all()
            and coverage["eligible_rows"].eq(expected_rows).all()
            and coverage["manifest_status"].eq("success").all()
            and not len(failures)
            and len(shared) == expected_rows
        )
        if not strict_ok:
            print("strict_audit: failed")
            return 1
    print("strict_audit: passed" if args.require_all_success else "audit: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
