#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Experiment_TTT_pipeline.pipeline import (
        PipelineConfig,
        expand_models,
        parse_model_extra_args,
        print_model_table,
        run_pipeline,
    )
    from Experiment_TTT_pipeline.registry import PACKAGE_ROOT, REPO_ROOT
else:
    from .pipeline import (
        PipelineConfig,
        expand_models,
        parse_model_extra_args,
        print_model_table,
        run_pipeline,
    )
    from .registry import PACKAGE_ROOT, REPO_ROOT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment_TTT_pipeline: unified runner for TFM inference and "
            "1C-style chunk TTT over baseline_compare plus TabICL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        help=(
            "Comma-separated model keys or aliases. Use all, all_inference, or all_ttt. "
            "Examples: tabicl,tabpfnv25,limix or all_ttt."
        ),
    )
    parser.add_argument("--strategy", choices=["inference", "ttt", "both"], default="both")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data178"))
    parser.add_argument("--out-root", default=str(PACKAGE_ROOT / "results"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument(
        "--gpu-groups",
        default=None,
        help=(
            "Only used by TabICL 1C TTT. If omitted, --gpus is passed and "
            "1C_Chunk_TTT.py's default gpu-groups value is cleared."
        ),
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--fail-on-unsupported",
        dest="skip_unsupported",
        action="store_false",
        help="Fail instead of recording a skip when a selected model lacks a strategy script.",
    )
    parser.set_defaults(skip_unsupported=True)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed to every selected backend script. Use "
            "--extra-arg=--flag=value when ARG starts with '--'."
        ),
    )
    parser.add_argument(
        "--model-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed to one model script, formatted as MODEL:ARG. "
            "Example: --model-extra-arg=tabpfnv3:--ttt-epochs=8"
        ),
    )
    parser.add_argument("--list-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print_model_table()
        return

    model_keys = expand_models(args.models, "inference" if args.strategy == "both" else args.strategy)
    model_extra_args = parse_model_extra_args(args.model_extra_arg)
    config = PipelineConfig(
        strategy=args.strategy,
        data_root=args.data_root,
        out_root=args.out_root,
        workers=args.workers,
        gpus=args.gpus,
        gpu_groups=args.gpu_groups,
        max_datasets=args.max_datasets,
        random_state=args.random_state,
        python_executable=args.python_executable,
        verbose=args.verbose,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
        skip_unsupported=args.skip_unsupported,
        extra_args=tuple(args.extra_arg),
        model_extra_args=model_extra_args,
    )
    results = run_pipeline(model_keys, config)
    failed = [result for result in results if result.status == "fail"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
