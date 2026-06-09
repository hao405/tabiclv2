#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

BASELINE_COMPARE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = BASELINE_COMPARE_ROOT / "results"


def sanitize_component(value: Any) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def dataset_label(data_root: Path, dataset_dirs: Sequence[Path]) -> str:
    if len(dataset_dirs) == 1:
        return Path(dataset_dirs[0]).name
    root_name = Path(data_root).name or "datasets"
    return f"{root_name}_{len(dataset_dirs)}datasets"


def checkpoint_stem(value: Any) -> str | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if raw_value.lower() in {"", "auto", "none", "null"}:
        return None
    return Path(raw_value).stem or None


def infer_estimator_label(args: Any, *, attr_name: str = "n_estimators", all_if_zero: bool = False) -> str:
    value = getattr(args, attr_name)
    if all_if_zero and int(value) == 0:
        return "infer_estall"
    return f"infer_est{value}"


def tabdpt_infer_label(args: Any) -> str:
    return f"infer_ens{args.n_ensembles}_ctx{args.context_size}"


def tabr_infer_label(args: Any) -> str:
    return f"infer_epoch{args.max_epochs}_ctx{args.context_size}_bs{args.batch_size}"


def seed_label(args: Any) -> str:
    seed_value = getattr(args, "random_state", None)
    if seed_value is None:
        seed_value = getattr(args, "seed")
    return f"seed{seed_value}"


def no_ttt_label() -> str:
    return "no_ttt"


def ttt_label(args: Any) -> str:
    if not bool(getattr(args, "ttt_enabled", False)):
        return no_ttt_label()

    parts = [
        f"ttt_eval-{getattr(args, 'ttt_eval_metric')}",
        f"ep{getattr(args, 'ttt_epochs')}",
        f"chunk{getattr(args, 'ttt_max_chunk_size')}",
        f"lr{getattr(args, 'ttt_lr')}",
        f"q{getattr(args, 'ttt_query_ratio')}",
    ]
    if hasattr(args, "ttt_accumulation_batch_size"):
        parts.append(f"accum{getattr(args, 'ttt_accumulation_batch_size')}")
    if hasattr(args, "ttt_n_estimators_finetune"):
        parts.append(f"finetuneest{getattr(args, 'ttt_n_estimators_finetune')}")
    if hasattr(args, "ttt_validation_n_estimators"):
        parts.append(f"valest{getattr(args, 'ttt_validation_n_estimators')}")
    if hasattr(args, "ttt_validation_n_ensembles"):
        parts.append(f"valens{getattr(args, 'ttt_validation_n_ensembles')}")
    return "_".join(parts)


def auto_out_dir(
    *,
    model_label: str,
    data_root: Path,
    dataset_dirs: Sequence[Path],
    infer_label: str,
    ttt_label: str,
    seed_label: str,
) -> Path:
    name_parts = [
        model_label,
        dataset_label(data_root, dataset_dirs),
        infer_label,
        ttt_label,
        seed_label,
    ]
    auto_name = "__".join(sanitize_component(part) for part in name_parts)
    return RESULTS_ROOT / auto_name
