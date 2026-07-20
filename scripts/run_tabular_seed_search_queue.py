#!/usr/bin/env python3
"""Method-level mixed-seed queue for TabICLv2 and TabPFNv3 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG_SEEDS = (3, 10, 16, 42, *range(2025, 2036))
DEFAULT_SHORT_SEEDS = (3, 10, 16)
MODELS = ("tabiclv2", "tabpfnv3")
METHODS = (
    "infer",
    "ft",
    "faware_ft",
    "lora",
    "last_layers",
    "ln_head_embedding",
    "micp",
)
PEFT_METHODS = {"lora", "last_layers", "ln_head_embedding"}
TFM_METHODS = {"infer", "ft", "faware_ft"}
SHORT_SEED_METHODS = PEFT_METHODS | {"micp"}

TFM_RUNNER = Path("scripts/run_tfm_experiment.py")
PEFT_RUNNER = Path("PEFT_Tabicl/1C_Chunk_PEFT.py")
MICP_RUNNER = Path("PEFT_Tabicl/MICP.py")

TABICL_CHECKPOINT = Path("tabicl-classifier-v2-20260212.ckpt")
TABPFN_BINARY_CHECKPOINT = Path(
    "baseline_compare/TabPFN-main/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
)
TABPFN_MULTICLASS_CHECKPOINT = Path(
    "baseline_compare/TabPFN-main/"
    "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
)

DEFAULT_TABPFN_FAWARE_BEST = {
    "data184": Path(
        "results/tabpfn/v3/data184_lr_reserve_ratio/"
        "tabpfn_v3_optuna_faware_c_lr_reserve_ratio_1/best_summary.json"
    ),
    "openmlcc18": Path(
        "results/tabpfn/v3/optuna_lr_mlcc18/"
        "tabpfn_v3_optuna_faware_c_lr_reserve_ratio_3/best_summary.json"
    ),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_int_list(value: str) -> tuple[int, ...]:
    output: list[int] = []
    for item in value.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number not in output:
            output.append(number)
    if not output:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(output)


def dataset_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() or path.is_symlink()
    )


def dataset_identity(
    label: str,
    root: Path,
    manifest_path: Path | None,
) -> dict[str, Any]:
    names = [path.name for path in dataset_directories(root)]
    payload: dict[str, Any] = {
        "label": label,
        "root": relative_or_absolute(root),
        "count": len(names),
        "names_sha256": sha256_json(names),
    }
    if manifest_path is not None:
        payload["manifest"] = relative_or_absolute(manifest_path)
        payload["manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
    return payload


def command_option(tokens: Sequence[str], flag: str) -> str | None:
    try:
        index = list(tokens).index(flag)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def load_tabpfn_best_summary(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, f"missing best summary: {relative_or_absolute(path)}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid best summary {relative_or_absolute(path)}: {exc}"
    best = payload.get("best_trial")
    if not isinstance(best, dict):
        return None, f"best_trial is not complete in {relative_or_absolute(path)}"
    state = str(best.get("state", "COMPLETE")).upper()
    if state != "COMPLETE":
        return None, f"best trial state is {state!r} in {relative_or_absolute(path)}"
    try:
        values: dict[str, Any] = {
            "ttt_lr": float(best["ttt_lr"]),
            "ttt_c_reserve_ratio": float(best["ttt_c_reserve_ratio"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"best trial parameters are incomplete: {exc}"
    command_text = str(best.get("command", "")).strip()
    tokens = shlex.split(command_text) if command_text else []
    typed_options: tuple[tuple[str, str, type], ...] = (
        ("ttt_c_metric", "--ttt-c-metric", str),
        ("ttt_query_ratio", "--ttt-query-ratio", float),
        ("ttt_epochs", "--ttt-epochs", int),
        ("n_estimators", "--n-estimators", int),
        ("ttt_patience", "--ttt-patience", int),
        ("ttt_validation_fraction", "--ttt-validation-fraction", float),
        (
            "ttt_n_estimators_finetune",
            "--ttt-n-estimators-finetune",
            int,
        ),
        (
            "ttt_validation_n_estimators",
            "--ttt-validation-n-estimators",
            int,
        ),
        ("ttt_weight_decay", "--ttt-weight-decay", float),
        ("ttt_min_delta", "--ttt-min-delta", float),
        (
            "ttt_grad_accumulation_steps",
            "--ttt-grad-accumulation-steps",
            int,
        ),
        ("ttt_eval_metric_runner", "--ttt-eval-metric", str),
    )
    for key, flag, caster in typed_options:
        raw_value = command_option(tokens, flag)
        if raw_value is not None:
            values[key] = caster(raw_value)
    values["source_command"] = command_text
    return values, relative_or_absolute(path)


def wait_for_required_configs(args: argparse.Namespace) -> None:
    if args.mode != "full" or not args.wait_for_config:
        return
    paths = [
        resolve_path(args.tabpfn_faware_data184_best),
        resolve_path(args.tabpfn_faware_openml_best),
    ]
    while True:
        missing: list[str] = []
        for path in paths:
            params, reason = load_tabpfn_best_summary(path)
            if params is None:
                missing.append(reason)
        if not missing:
            print(
                f"{now_iso()} all required TabPFNv3 F-aware configs are ready",
                flush=True,
            )
            return
        print(
            f"{now_iso()} waiting_for_config: {'; '.join(missing)}",
            flush=True,
        )
        time.sleep(args.config_poll_seconds)


def write_prequeue_manifest(
    args: argparse.Namespace,
    specs: Sequence["CellSpec"],
    run_root: Path,
) -> Path:
    path = run_root / "prequeue_manifest.json"
    cells = [
        {
            "cell_id": spec.cell_id,
            "order": spec.order,
            "dataset": spec.dataset,
            "model": spec.model,
            "method": spec.method,
            "seed_policy": spec.seed_policy,
            "seeds": spec.seeds,
            "config_status": spec.config_status,
            "config_source": spec.config_source,
            "config_error": spec.config_error,
        }
        for spec in specs
    ]
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at": now_iso(),
        "status": (
            "waiting_for_config"
            if any(spec.config_status != "ready" for spec in specs)
            else "ready"
        ),
        "planned_cells": len(specs),
        "planned_seed_slots": sum(len(spec.seeds) for spec in specs),
        "long_seeds": list(args.long_seeds),
        "short_seeds": list(args.short_seeds),
        "gpus": list(args.gpus),
        "cells": cells,
        "matrix_sha256": sha256_json(cells),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("matrix_sha256") != payload["matrix_sha256"]:
            identity_fields = (
                "cell_id",
                "order",
                "dataset",
                "model",
                "method",
                "seeds",
            )
            existing_identity = [
                {field: cell.get(field) for field in identity_fields}
                for cell in existing.get("cells", [])
            ]
            payload_identity = [
                {field: cell.get(field) for field in identity_fields}
                for cell in payload["cells"]
            ]
            config_gate_opened = (
                existing_identity == payload_identity
                and existing.get("status") == "waiting_for_config"
                and payload["status"] == "ready"
            )
            if not config_gate_opened:
                raise ValueError(f"existing prequeue manifest differs: {path}")
            payload["created_at"] = existing.get("created_at", payload["created_at"])
            payload["updated_at"] = now_iso()
            payload["config_transition"] = "waiting_for_config -> ready"
            atomic_json(path, payload)
            return path
        if existing.get("gpus") != payload["gpus"]:
            existing["gpus"] = payload["gpus"]
            existing["updated_at"] = now_iso()
            existing["resource_update"] = (
                "Use every physical GPU after the blocking Optuna search exits."
            )
            atomic_json(path, existing)
    else:
        atomic_json(path, payload)
    return path


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    order: int
    dataset: str
    model: str
    method: str
    runner_family: str
    runner: str
    data_root: str
    dataset_identity: dict[str, Any]
    config_source: str
    config_status: str
    config_error: str | None
    static_args: list[str]
    config_fingerprint: str
    seed_policy: str
    seeds: list[int]


@dataclass
class SeedState:
    seed: int
    status: str = "pending"
    execution: str | None = None
    worker: str | None = None
    gpu: int | None = None
    output_dir: str | None = None
    command: list[str] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    dataset_count: int | None = None
    ok_count: int | None = None
    failed_count: int | None = None
    error: str | None = None


@dataclass
class CellState:
    cell_id: str
    status: str
    claimed_by: str | None = None
    gpu: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    seeds: list[SeedState] = field(default_factory=list)


def full_matrix() -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for dataset in ("data184", "openmlcc18"):
        for model in MODELS:
            for method in METHODS:
                if (
                    dataset == "data184"
                    and model == "tabiclv2"
                    and method in TFM_METHODS
                ):
                    continue
                cells.append((dataset, model, method))
    return cells


def smoke_matrix() -> list[tuple[str, str, str]]:
    return [
        ("data184", "tabpfnv3", "infer"),
        ("data184", "tabiclv2", "lora"),
        ("data184", "tabiclv2", "micp"),
    ]


def seed_policy_for_method(method: str) -> str:
    if method in TFM_METHODS:
        return "long"
    if method in SHORT_SEED_METHODS:
        return "short"
    raise ValueError(f"unknown method for seed policy: {method}")


def tfm_static_args(
    *,
    model: str,
    method: str,
    best_params: dict[str, Any] | None,
    smoke: bool,
) -> list[str]:
    command_model = "tabicl-v2" if model == "tabiclv2" else "tabpfn-v3"
    args = [
        "--model",
        command_model,
        "--method",
        method,
        "--workers",
        "1",
        "--n-estimators",
        str(
            2
            if smoke
            else (best_params or {}).get(
                "n_estimators",
                32 if model == "tabiclv2" else 8,
            )
        ),
        "--ttt-epochs",
        str(1 if smoke else (best_params or {}).get("ttt_epochs", 30)),
        "--ttt-lr",
        str((best_params or {}).get("ttt_lr", 1e-5)),
        "--ttt-query-ratio",
        str((best_params or {}).get("ttt_query_ratio", 0.2)),
        "--ttt-eval-metric",
        "accuracy",
        "--ttt-c-metric",
        str((best_params or {}).get("ttt_c_metric", "model_native_l2")),
        "--ttt-c-reserve-ratio",
        str((best_params or {}).get("ttt_c_reserve_ratio", 0.05)),
        "--ttt-validation-fraction",
        str((best_params or {}).get("ttt_validation_fraction", 0.1)),
        "--ttt-n-estimators-finetune",
        str((best_params or {}).get("ttt_n_estimators_finetune", 2)),
        "--ttt-validation-n-estimators",
        str((best_params or {}).get("ttt_validation_n_estimators", 2)),
        "--ttt-patience",
        str(1 if smoke else (best_params or {}).get("ttt_patience", 8)),
    ]
    if smoke:
        args.extend(["--max-datasets", "1"])
    if best_params and not smoke:
        passthrough: list[str] = []
        for key, flag in (
            ("ttt_weight_decay", "--ttt-weight-decay"),
            ("ttt_min_delta", "--ttt-min-delta"),
            (
                "ttt_grad_accumulation_steps",
                "--ttt-grad-accumulation-steps",
            ),
            ("ttt_eval_metric_runner", "--ttt-eval-metric"),
        ):
            if key in best_params:
                passthrough.extend([flag, str(best_params[key])])
        if passthrough:
            args.extend(["--", *passthrough])
    return args


def peft_static_args(*, model: str, method: str, smoke: bool) -> list[str]:
    args = [
        "--model-family",
        model,
        "--ttt-peft-method",
        method,
        "--workers",
        "1",
        "--n-estimators",
        "2" if smoke else "32",
        "--ttt-lr",
        "1e-5",
        "--ttt-weight-decay",
        "0.01",
        "--ttt-epochs",
        "1" if smoke else "30",
        "--ttt-max-chunk-size",
        "10000",
        "--ttt-min-chunk-size",
        "50",
        "--ttt-query-ratio",
        "0.2",
        "--ttt-n-estimators-finetune",
        "2",
        "--ttt-validation-n-estimators",
        "2",
        "--ttt-validation-fraction",
        "0.1",
        "--ttt-eval-metric",
        "accuracy",
        "--ttt-patience",
        "1" if smoke else "8",
        "--ttt-min-delta",
        "0.0001",
        "--ttt-peft-targets",
        "col,row,icl",
        "--ttt-lora-rank",
        "4",
        "--ttt-lora-alpha",
        "8.0",
        "--ttt-lora-dropout",
        "0.0",
        "--ttt-last-n-icl-blocks",
        "1",
        "--tabicl-model-path",
        str(TABICL_CHECKPOINT),
        "--tabpfn-v3-binary-model-path",
        str(TABPFN_BINARY_CHECKPOINT),
        "--tabpfn-v3-multiclass-model-path",
        str(TABPFN_MULTICLASS_CHECKPOINT),
        "--checkpoint-version",
        str(TABICL_CHECKPOINT),
        "--ttt-early-stopping",
        "True",
        "--ttt-save-ckpt",
        "False",
    ]
    if smoke:
        args.extend(["--max-datasets", "1"])
    return args


def micp_static_args(*, model: str, smoke: bool) -> list[str]:
    args = [
        "--model-family",
        model,
        "--method",
        "mixturepfn",
        "--support-size",
        "3000",
        "--gamma",
        "5.0",
        "--ca-steps",
        "1" if smoke else "128",
        "--ca-query-size",
        "64",
        "--ca-lr",
        "1e-3",
        "--inference-batch-size",
        "1024",
        "--n-estimators",
        "2" if smoke else "16",
        "--train-n-estimators",
        "2",
        "--retrieval-backend",
        "faiss",
        "--tabicl-model-path",
        str(TABICL_CHECKPOINT),
        "--tabpfn-v3-binary-model-path",
        str(TABPFN_BINARY_CHECKPOINT),
        "--tabpfn-v3-multiclass-model-path",
        str(TABPFN_MULTICLASS_CHECKPOINT),
        "--use-amp",
        "auto",
        "--resume",
    ]
    if smoke:
        args.extend(["--max-datasets", "1"])
    return args


def build_specs(args: argparse.Namespace) -> list[CellSpec]:
    data_roots = {
        "data184": resolve_path(args.data184_root),
        "openmlcc18": resolve_path(args.openml_root),
    }
    openml_manifest = resolve_path(args.openml_manifest)
    identities = {
        "data184": dataset_identity("data184", data_roots["data184"], None),
        "openmlcc18": dataset_identity(
            "openmlcc18",
            data_roots["openmlcc18"],
            openml_manifest,
        ),
    }
    matrix = smoke_matrix() if args.mode == "smoke" else full_matrix()
    specs: list[CellSpec] = []
    for order, (dataset, model, method) in enumerate(matrix):
        seed_policy = seed_policy_for_method(method)
        policy_seeds = (
            list(args.long_seeds)
            if seed_policy == "long"
            else list(args.short_seeds)
        )
        seeds = policy_seeds[:1] if args.mode == "smoke" else policy_seeds
        config_status = "ready"
        config_error: str | None = None
        best_params: dict[str, Any] | None = None
        if method == "faware_ft" and model == "tabpfnv3":
            summary_path = resolve_path(
                args.tabpfn_faware_data184_best
                if dataset == "data184"
                else args.tabpfn_faware_openml_best
            )
            best_params, source = load_tabpfn_best_summary(summary_path)
            if best_params is None:
                config_status = "blocked_config"
                config_error = source
                config_source = relative_or_absolute(summary_path)
            else:
                config_source = source
        elif method in PEFT_METHODS:
            config_source = "formal_peft_seed42_profile"
        elif method == "micp":
            config_source = "formal_micp_seed42_profile"
        else:
            config_source = "run_tfm_experiment_defaults"

        if method in TFM_METHODS:
            runner_family = "tfm"
            runner = TFM_RUNNER
            static_args = tfm_static_args(
                model=model,
                method=method,
                best_params=best_params,
                smoke=args.mode == "smoke",
            )
        elif method in PEFT_METHODS:
            runner_family = "peft"
            runner = PEFT_RUNNER
            static_args = peft_static_args(
                model=model,
                method=method,
                smoke=args.mode == "smoke",
            )
        else:
            runner_family = "micp"
            runner = MICP_RUNNER
            static_args = micp_static_args(
                model=model,
                smoke=args.mode == "smoke",
            )
        fingerprint_payload = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "runner": str(runner),
            "runner_sha256": sha256_bytes(resolve_path(runner).read_bytes()),
            "data_identity": identities[dataset],
            "static_args": static_args,
            "config_source": config_source,
        }
        cell_id = f"{dataset}__{model}__{method}"
        specs.append(
            CellSpec(
                cell_id=cell_id,
                order=order,
                dataset=dataset,
                model=model,
                method=method,
                runner_family=runner_family,
                runner=str(runner),
                data_root=relative_or_absolute(data_roots[dataset]),
                dataset_identity=identities[dataset],
                config_source=config_source,
                config_status=config_status,
                config_error=config_error,
                static_args=static_args,
                config_fingerprint=sha256_json(fingerprint_payload),
                seed_policy=seed_policy,
                seeds=seeds,
            )
        )
    return specs


def build_command(
    spec: CellSpec,
    *,
    seed: int,
    gpu: int,
    output_dir: Path,
    python_bin: str,
    retry_failed: bool,
) -> list[str]:
    static_args = list(spec.static_args)
    passthrough: list[str] = []
    if spec.runner_family == "tfm" and "--" in static_args:
        marker = static_args.index("--")
        passthrough = static_args[marker:]
        static_args = static_args[:marker]
    command = [
        python_bin,
        str(resolve_path(spec.runner)),
        *static_args,
        "--data-root",
        str(resolve_path(spec.data_root)),
        "--out-dir",
        str(output_dir),
    ]
    if spec.runner_family == "tfm":
        command.extend(["--devices", str(gpu), "--random-state", str(seed)])
        command.extend(passthrough)
    elif spec.runner_family == "peft":
        command.extend(["--gpu-groups", str(gpu), "--random-state", str(seed)])
        if output_dir.exists():
            command.append("--resume")
    else:
        command.extend(["--gpus", str(gpu), "--seed", str(seed)])
        if retry_failed:
            command.append("--retry-failed")
    return command


def read_result_counts(output_dir: Path) -> tuple[int | None, int | None, int | None]:
    csv_path = output_dir / "all_classification_results.csv"
    if not csv_path.is_file():
        return None, None, None
    total = 0
    ok = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if str(row.get("status", "")).strip().lower() == "ok":
                ok += 1
    return total, ok, total - ok


class QueueManager:
    def __init__(
        self,
        args: argparse.Namespace,
        specs: list[CellSpec],
        run_root: Path,
    ) -> None:
        self.args = args
        self.specs = specs
        self.spec_by_id = {spec.cell_id: spec for spec in specs}
        self.run_root = run_root
        self.manifest_path = run_root / "queue_manifest.json"
        self.state_path = run_root / "queue_state.json"
        self.manager_log = run_root / "manager.log"
        self.lock = threading.Lock()
        self.work: queue.Queue[str] = queue.Queue()
        self.states = self._load_or_create_state()

    def _load_or_create_state(self) -> dict[str, CellState]:
        if self.args.resume and self.state_path.is_file():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            states: dict[str, CellState] = {}
            for item in payload["cells"]:
                seed_states = [SeedState(**seed) for seed in item.pop("seeds")]
                state = CellState(**item, seeds=seed_states)
                if state.status == "running":
                    state.status = "pending"
                    state.claimed_by = None
                    state.gpu = None
                for seed in state.seeds:
                    if seed.status == "running":
                        seed.status = "pending"
                        seed.worker = None
                        seed.gpu = None
                states[state.cell_id] = state
            return states
        return {
            spec.cell_id: CellState(
                cell_id=spec.cell_id,
                status=(
                    "blocked_config"
                    if spec.config_status != "ready"
                    else "pending"
                ),
                seeds=[SeedState(seed=seed) for seed in spec.seeds],
            )
            for spec in self.specs
        }

    def log(self, message: str) -> None:
        line = f"{now_iso()} {message}"
        print(line, flush=True)
        self.manager_log.parent.mkdir(parents=True, exist_ok=True)
        with self.manager_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save_state(self) -> None:
        payload = {
            "run_id": self.args.run_id,
            "updated_at": now_iso(),
            "cells": [asdict(self.states[spec.cell_id]) for spec in self.specs],
        }
        atomic_json(self.state_path, payload)
        self.write_summaries()

    def write_summaries(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        cell_path = self.run_root / "queue_summary.csv"
        seed_path = self.run_root / "seed_summary.csv"
        with cell_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cell_id",
                    "dataset",
                    "model",
                    "method",
                    "seed_policy",
                    "status",
                    "claimed_by",
                    "gpu",
                    "completed_seeds",
                    "failed_seeds",
                    "reused_seeds",
                ],
            )
            writer.writeheader()
            for spec in self.specs:
                state = self.states[spec.cell_id]
                writer.writerow(
                    {
                        "cell_id": spec.cell_id,
                        "dataset": spec.dataset,
                        "model": spec.model,
                        "method": spec.method,
                        "seed_policy": spec.seed_policy,
                        "status": state.status,
                        "claimed_by": state.claimed_by,
                        "gpu": state.gpu,
                        "completed_seeds": sum(
                            seed.status in {"success", "failed"} for seed in state.seeds
                        ),
                        "failed_seeds": sum(
                            seed.status == "failed"
                            or (seed.failed_count or 0) > 0
                            for seed in state.seeds
                        ),
                        "reused_seeds": sum(
                            seed.execution == "reused" for seed in state.seeds
                        ),
                    }
                )
        seed_fields = [
            "cell_id",
            "dataset",
            "model",
            "method",
            "seed_policy",
            *SeedState.__dataclass_fields__.keys(),
        ]
        with seed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=seed_fields)
            writer.writeheader()
            for spec in self.specs:
                for seed in self.states[spec.cell_id].seeds:
                    writer.writerow(
                        {
                            "cell_id": spec.cell_id,
                            "dataset": spec.dataset,
                            "model": spec.model,
                            "method": spec.method,
                            "seed_policy": spec.seed_policy,
                            **asdict(seed),
                        }
                    )

    def prepare(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "mode": self.args.mode,
            "created_at": now_iso(),
            "long_seeds": list(self.args.long_seeds),
            "short_seeds": list(self.args.short_seeds),
            "seed_policy_methods": {
                "long": sorted(TFM_METHODS),
                "short": sorted(SHORT_SEED_METHODS),
            },
            "long_seed_cells": sum(
                spec.seed_policy == "long" for spec in self.specs
            ),
            "short_seed_cells": sum(
                spec.seed_policy == "short" for spec in self.specs
            ),
            "planned_cells": len(self.specs),
            "planned_seed_slots": sum(len(spec.seeds) for spec in self.specs),
            "gpus": list(self.args.gpus),
            "schedule": "method_cell_atomic_dynamic_gpu_workers",
            "cells": [asdict(spec) for spec in self.specs],
        }
        manifest_payload["spec_sha256"] = sha256_json(manifest_payload["cells"])
        if self.manifest_path.exists():
            if not self.args.resume:
                raise FileExistsError(
                    f"queue run already exists: {self.run_root}; use --resume"
                )
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("spec_sha256") != manifest_payload["spec_sha256"]:
                raise ValueError(
                    f"existing queue manifest differs: {self.manifest_path}"
                )
        else:
            atomic_json(self.manifest_path, manifest_payload)
        self.save_state()

    def seed_terminal_path(self, output_dir: Path) -> Path:
        return output_dir / "queue_seed_terminal.json"

    def maybe_reuse_seed(
        self,
        spec: CellSpec,
        seed_state: SeedState,
        output_dir: Path,
    ) -> bool:
        candidates = [self.seed_terminal_path(output_dir)]
        for root_value in self.args.reuse_root:
            root = resolve_path(root_value)
            candidates.append(
                root
                / spec.dataset
                / spec.model
                / spec.method
                / f"seed{seed_state.seed}"
                / "queue_seed_terminal.json"
            )
        for terminal_path in candidates:
            if not terminal_path.is_file():
                continue
            try:
                payload = json.loads(terminal_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                payload.get("config_fingerprint") != spec.config_fingerprint
                or int(payload.get("seed", -1)) != seed_state.seed
                or payload.get("dataset") != spec.dataset
                or payload.get("model") != spec.model
                or payload.get("method") != spec.method
            ):
                continue
            if payload.get("status") == "failed" and self.args.retry_failed:
                continue
            seed_state.status = str(payload.get("status"))
            seed_state.execution = "reused"
            seed_state.output_dir = str(payload.get("output_dir", terminal_path.parent))
            seed_state.command = payload.get("command")
            seed_state.started_at = payload.get("started_at")
            seed_state.finished_at = payload.get("finished_at")
            seed_state.exit_code = payload.get("exit_code")
            seed_state.dataset_count = payload.get("dataset_count")
            seed_state.ok_count = payload.get("ok_count")
            seed_state.failed_count = payload.get("failed_count")
            seed_state.error = payload.get("error")
            return True
        return False

    def wait_for_gpu(self, gpu: int, worker: str) -> None:
        if self.args.skip_gpu_wait:
            return
        consecutive = 0
        while consecutive < self.args.gpu_ready_checks:
            command = [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            except FileNotFoundError as exc:
                completed = subprocess.CompletedProcess(
                    command,
                    127,
                    stdout="",
                    stderr=f"{type(exc).__name__}: {exc}",
                )
            ready = False
            audit = completed.stdout.strip() or completed.stderr.strip()
            if completed.returncode == 0:
                try:
                    used, total, utilization = [
                        float(value.strip())
                        for value in completed.stdout.strip().split(",")
                    ]
                    free_ratio = (total - used) / total
                    ready = (
                        free_ratio >= self.args.gpu_min_free_ratio
                        and utilization <= self.args.gpu_max_utilization
                    )
                    audit = (
                        f"used={used:.0f} total={total:.0f} "
                        f"free_ratio={free_ratio:.4f} util={utilization:.0f}"
                    )
                except Exception:
                    ready = False
            else:
                torch_check = subprocess.run(
                    [
                        self.args.python_bin,
                        "-c",
                        (
                            "import json,torch,sys;"
                            "i=int(sys.argv[1]);"
                            "free,total=torch.cuda.mem_get_info(i);"
                            "print(json.dumps({'free':free,'total':total,"
                            "'name':torch.cuda.get_device_name(i)}))"
                        ),
                        str(gpu),
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                audit = torch_check.stdout.strip() or torch_check.stderr.strip()
                if torch_check.returncode == 0:
                    try:
                        payload = json.loads(torch_check.stdout)
                        free_ratio = float(payload["free"]) / float(payload["total"])
                        ready = free_ratio >= self.args.gpu_min_free_ratio
                        audit = (
                            f"torch name={payload['name']} "
                            f"free_ratio={free_ratio:.4f}"
                        )
                    except Exception:
                        ready = False
            consecutive = consecutive + 1 if ready else 0
            self.log(
                f"{worker} gpu={gpu} ready={ready} "
                f"checks={consecutive}/{self.args.gpu_ready_checks} {audit}"
            )
            if consecutive < self.args.gpu_ready_checks:
                time.sleep(self.args.gpu_poll_seconds)

    def run_seed(
        self,
        spec: CellSpec,
        seed_state: SeedState,
        *,
        worker: str,
        gpu: int,
    ) -> None:
        output_dir = (
            self.run_root
            / spec.dataset
            / spec.model
            / spec.method
            / f"seed{seed_state.seed}"
        )
        if self.maybe_reuse_seed(spec, seed_state, output_dir):
            self.log(f"{worker} reused {spec.cell_id} seed={seed_state.seed}")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "runner.log"
        command = build_command(
            spec,
            seed=seed_state.seed,
            gpu=gpu,
            output_dir=output_dir,
            python_bin=self.args.python_bin,
            retry_failed=self.args.retry_failed,
        )
        seed_state.status = "running"
        seed_state.execution = "executed"
        seed_state.worker = worker
        seed_state.gpu = gpu
        seed_state.output_dir = str(output_dir)
        seed_state.command = command
        seed_state.started_at = now_iso()
        with self.lock:
            self.save_state()
        self.log(
            f"{worker} start {spec.cell_id} seed={seed_state.seed}: "
            f"{shlex.join(command)}"
        )
        error: str | None = None
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            exit_code = int(completed.returncode)
        except Exception as exc:
            exit_code = 127
            error = f"{type(exc).__name__}: {exc}"
        total, ok, failed = read_result_counts(output_dir)
        seed_state.exit_code = exit_code
        seed_state.dataset_count = total
        seed_state.ok_count = ok
        seed_state.failed_count = failed
        seed_state.error = error
        seed_state.finished_at = now_iso()
        seed_state.status = "success" if exit_code == 0 else "failed"
        terminal = {
            "dataset": spec.dataset,
            "model": spec.model,
            "method": spec.method,
            "seed": seed_state.seed,
            "status": seed_state.status,
            "execution": seed_state.execution,
            "config_fingerprint": spec.config_fingerprint,
            "output_dir": str(output_dir),
            "command": command,
            "started_at": seed_state.started_at,
            "finished_at": seed_state.finished_at,
            "exit_code": exit_code,
            "dataset_count": total,
            "ok_count": ok,
            "failed_count": failed,
            "error": error,
        }
        atomic_json(self.seed_terminal_path(output_dir), terminal)
        self.log(
            f"{worker} finish {spec.cell_id} seed={seed_state.seed} "
            f"status={seed_state.status} exit={exit_code} ok={ok} failed={failed}"
        )

    def run_cell(self, cell_id: str, *, worker: str, gpu: int) -> None:
        spec = self.spec_by_id[cell_id]
        state = self.states[cell_id]
        with self.lock:
            state.status = "running"
            state.claimed_by = worker
            state.gpu = gpu
            state.started_at = state.started_at or now_iso()
            self.save_state()
        for seed_state in state.seeds:
            if seed_state.status == "success":
                continue
            if seed_state.status == "failed" and not self.args.retry_failed:
                continue
            self.run_seed(
                spec,
                seed_state,
                worker=worker,
                gpu=gpu,
            )
            with self.lock:
                self.save_state()
        any_failure = any(
            seed.status == "failed" or (seed.failed_count or 0) > 0
            for seed in state.seeds
        )
        with self.lock:
            state.status = "complete_with_failures" if any_failure else "complete"
            state.finished_at = now_iso()
            self.save_state()

    def worker(self, gpu: int, index: int) -> None:
        worker = f"worker{index}"
        self.log(f"{worker} assigned gpu={gpu}")
        while True:
            try:
                cell_id = self.work.get_nowait()
            except queue.Empty:
                self.log(f"{worker} queue empty")
                return
            try:
                self.wait_for_gpu(gpu, worker)
                self.run_cell(cell_id, worker=worker, gpu=gpu)
            except Exception as exc:
                with self.lock:
                    state = self.states[cell_id]
                    state.status = "complete_with_failures"
                    state.finished_at = now_iso()
                    self.save_state()
                self.log(f"{worker} cell={cell_id} error={type(exc).__name__}: {exc}")
            finally:
                self.work.task_done()

    def run(self) -> None:
        blocked = [
            spec for spec in self.specs if spec.config_status != "ready"
        ]
        if blocked:
            details = "; ".join(
                f"{spec.cell_id}: {spec.config_error}" for spec in blocked
            )
            raise RuntimeError(f"blocked configuration cells: {details}")
        for spec in self.specs:
            if self.states[spec.cell_id].status not in {
                "complete",
                "complete_with_failures",
            }:
                self.work.put(spec.cell_id)
        threads = [
            threading.Thread(
                target=self.worker,
                args=(gpu, index),
                daemon=False,
                name=f"seed-search-gpu{gpu}",
            )
            for index, gpu in enumerate(self.args.gpus)
        ]
        self.log(
            f"manager start cells={len(self.specs)} "
            f"seed_slots={sum(len(spec.seeds) for spec in self.specs)} "
            f"gpus={list(self.args.gpus)}"
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.write_shared_coverage()
        if self.args.mode == "smoke":
            failed = [
                state.cell_id
                for state in self.states.values()
                if state.status != "complete"
            ]
            if failed:
                self.log(f"smoke gate failed cells={failed}")
                raise RuntimeError(f"smoke gate failed: {failed}")
        self.log("manager complete")

    def write_shared_coverage(self) -> None:
        groups: dict[tuple[str, str, int], list[set[str]]] = {}
        expected_method_counts: dict[tuple[str, str, int], int] = {}
        for spec in self.specs:
            for seed in spec.seeds:
                key = (spec.dataset, spec.model, seed)
                expected_method_counts[key] = expected_method_counts.get(key, 0) + 1
        for spec in self.specs:
            for seed_state in self.states[spec.cell_id].seeds:
                if not seed_state.output_dir:
                    continue
                csv_path = Path(seed_state.output_dir) / "all_classification_results.csv"
                if not csv_path.is_file():
                    continue
                ok_names: set[str] = set()
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if str(row.get("status", "")).strip().lower() == "ok":
                            name = str(row.get("dataset_name", "")).strip()
                            if name:
                                ok_names.add(name)
                groups.setdefault(
                    (spec.dataset, spec.model, seed_state.seed),
                    [],
                ).append(ok_names)
        rows = []
        for (dataset, model, seed), method_sets in sorted(groups.items()):
            intersection = set.intersection(*method_sets) if method_sets else set()
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "seed_policy": (
                        "shared_short_long"
                        if seed in self.args.short_seeds
                        else "long_only"
                    ),
                    "expected_method_count": expected_method_counts[
                        (dataset, model, seed)
                    ],
                    "method_count": len(method_sets),
                    "shared_status_ok_count": len(intersection),
                    "shared_dataset_names": sorted(intersection),
                }
            )
        atomic_json(self.run_root / "shared_success_coverage.json", rows)


def preflight(args: argparse.Namespace, specs: Sequence[CellSpec]) -> None:
    expected = 3 if args.mode == "smoke" else 25
    expected_slots = 3 if args.mode == "smoke" else 183
    expected_long_cells = 1 if args.mode == "smoke" else 9
    expected_short_cells = 2 if args.mode == "smoke" else 16
    actual_slots = sum(len(spec.seeds) for spec in specs)
    if len(specs) != expected:
        raise ValueError(f"expected {expected} cells, found {len(specs)}")
    if actual_slots != expected_slots:
        raise ValueError(f"expected {expected_slots} seed slots, found {actual_slots}")
    long_cells = [spec for spec in specs if spec.seed_policy == "long"]
    short_cells = [spec for spec in specs if spec.seed_policy == "short"]
    if len(long_cells) != expected_long_cells:
        raise ValueError(
            f"expected {expected_long_cells} long-seed cells, found {len(long_cells)}"
        )
    if len(short_cells) != expected_short_cells:
        raise ValueError(
            f"expected {expected_short_cells} short-seed cells, found "
            f"{len(short_cells)}"
        )
    if len(set(args.long_seeds)) != len(args.long_seeds):
        raise ValueError("long seeds must be unique")
    if len(set(args.short_seeds)) != len(args.short_seeds):
        raise ValueError("short seeds must be unique")
    if not set(args.short_seeds).issubset(args.long_seeds):
        raise ValueError("short seeds must be a subset of long seeds")
    for spec in specs:
        expected_policy = seed_policy_for_method(spec.method)
        if spec.seed_policy != expected_policy:
            raise ValueError(
                f"{spec.cell_id} must use {expected_policy} seed policy"
            )
        seed = spec.seeds[0]
        command = build_command(
            spec,
            seed=seed,
            gpu=args.gpus[spec.order % len(args.gpus)],
            output_dir=resolve_path(args.output_root) / "_preflight",
            python_bin=args.python_bin,
            retry_failed=args.retry_failed,
        )
        expected_flag = "--seed" if spec.runner_family == "micp" else "--random-state"
        forbidden_flag = "--random-state" if expected_flag == "--seed" else "--seed"
        if expected_flag not in command or forbidden_flag in command:
            raise ValueError(
                f"{spec.cell_id} must use {expected_flag}, not {forbidden_flag}"
            )
    identities = {spec.dataset: spec.dataset_identity for spec in specs}
    if identities["data184"]["count"] != 184:
        raise ValueError(
            f"data184 must contain 184 datasets, found {identities['data184']['count']}"
        )
    if args.mode == "full" and identities["openmlcc18"]["count"] != 67:
        raise ValueError(
            "OpenML-CC18 view must contain 67 datasets, found "
            f"{identities['openmlcc18']['count']}"
        )
    required = {
        resolve_path(TFM_RUNNER),
        resolve_path(PEFT_RUNNER),
        resolve_path(MICP_RUNNER),
        resolve_path(TABICL_CHECKPOINT),
        resolve_path(TABPFN_BINARY_CHECKPOINT),
        resolve_path(TABPFN_MULTICLASS_CHECKPOINT),
    }
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required files are missing: {missing}")


def dry_run_payload(
    args: argparse.Namespace,
    specs: Sequence[CellSpec],
    run_root: Path,
) -> dict[str, Any]:
    commands = []
    for spec in specs:
        seed = spec.seeds[0]
        command = build_command(
            spec,
            seed=seed,
            gpu=args.gpus[spec.order % len(args.gpus)],
            output_dir=(
                run_root
                / spec.dataset
                / spec.model
                / spec.method
                / f"seed{seed}"
            ),
            python_bin=args.python_bin,
            retry_failed=args.retry_failed,
        )
        commands.append(
            {
                "cell_id": spec.cell_id,
                "seed_policy": spec.seed_policy,
                "seeds": spec.seeds,
                "seed_slots": len(spec.seeds),
                "config_status": spec.config_status,
                "config_error": spec.config_error,
                "config_fingerprint": spec.config_fingerprint,
                "command": command,
                "command_shell": shlex.join(command),
            }
        )
    return {
        "mode": args.mode,
        "planned_cells": len(specs),
        "planned_seed_slots": sum(len(spec.seeds) for spec in specs),
        "long_seed_cells": sum(spec.seed_policy == "long" for spec in specs),
        "short_seed_cells": sum(spec.seed_policy == "short" for spec in specs),
        "long_seeds": list(args.long_seeds),
        "short_seeds": list(args.short_seeds),
        "blocked_config_cells": sum(
            spec.config_status != "ready" for spec in specs
        ),
        "commands": commands,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable method-level seed queue for "
            "TabICLv2/TabPFNv3 on data184/OpenML-CC18."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--output-root", default="results/seed_search")
    parser.add_argument(
        "--long-seeds",
        type=parse_int_list,
        default=DEFAULT_LONG_SEEDS,
        help="Seeds for infer, ft, and faware_ft cells.",
    )
    parser.add_argument(
        "--short-seeds",
        type=parse_int_list,
        default=DEFAULT_SHORT_SEEDS,
        help="Seeds for PEFT and MICP cells.",
    )
    parser.add_argument(
        "--gpus",
        type=parse_int_list,
        default=(0, 1, 2),
        help="Physical GPU ids, one method-level worker per id.",
    )
    parser.add_argument("--data184-root", default="data184")
    parser.add_argument(
        "--openml-root",
        default="results/dataset_views/openml_cc18_max10",
    )
    parser.add_argument(
        "--openml-manifest",
        default=(
            "results/dataset_views/openml_cc18_max10/"
            "dataset_view_manifest.json"
        ),
    )
    parser.add_argument(
        "--tabpfn-faware-data184-best",
        default=str(DEFAULT_TABPFN_FAWARE_BEST["data184"]),
    )
    parser.add_argument(
        "--tabpfn-faware-openml-best",
        default=str(DEFAULT_TABPFN_FAWARE_BEST["openmlcc18"]),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reuse-root", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--wait-for-config",
        action="store_true",
        help="Wait until both required TabPFNv3 F-aware best summaries are complete.",
    )
    parser.add_argument("--config-poll-seconds", type=float, default=60.0)
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument("--gpu-min-free-ratio", type=float, default=0.90)
    parser.add_argument("--gpu-max-utilization", type=float, default=5.0)
    parser.add_argument("--gpu-ready-checks", type=int, default=3)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_for_config:
        preliminary_specs = build_specs(args)
        preflight(args, preliminary_specs)
        preliminary_root = resolve_path(args.output_root) / args.run_id
        prequeue_path = write_prequeue_manifest(
            args,
            preliminary_specs,
            preliminary_root,
        )
        print(f"prequeue_manifest: {prequeue_path}", flush=True)
    wait_for_required_configs(args)
    specs = build_specs(args)
    preflight(args, specs)
    run_root = resolve_path(args.output_root) / args.run_id
    if args.dry_run:
        print(json.dumps(dry_run_payload(args, specs, run_root), indent=2))
        return 0
    manager = QueueManager(args, specs, run_root)
    manager.prepare()
    print(f"queue_manifest: {manager.manifest_path}")
    print(f"queue_state: {manager.state_path}")
    print(f"manager_log: {manager.manager_log}")
    if args.prepare_only:
        return 0
    manager.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
