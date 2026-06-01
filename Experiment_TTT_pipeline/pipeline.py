from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from .registry import MODEL_REGISTRY, PACKAGE_ROOT, REPO_ROOT, ModelSpec, StrategyScript, resolve_model_key


@dataclass
class PipelineConfig:
    strategy: str = "inference"
    data_root: str | Path = REPO_ROOT / "data178"
    out_root: str | Path = PACKAGE_ROOT / "results"
    workers: int = 1
    gpus: str | None = "auto"
    gpu_groups: str | None = None
    max_datasets: int | None = None
    random_state: int = 42
    python_executable: str = sys.executable
    verbose: bool = False
    dry_run: bool = False
    stop_on_failure: bool = False
    skip_unsupported: bool = True
    extra_args: tuple[str, ...] = ()
    model_extra_args: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    model_key: str
    display_name: str
    strategy: str
    status: str
    returncode: int | None
    command: str
    out_dir: str
    elapsed_seconds: float
    error: str | None = None


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def strategies_for(config: PipelineConfig) -> tuple[str, ...]:
    if config.strategy == "both":
        return ("inference", "ttt")
    if config.strategy in {"inference", "ttt"}:
        return (config.strategy,)
    raise ValueError("--strategy must be one of: inference, ttt, both")


def expand_models(values: Sequence[str], strategy: str) -> list[str]:
    if not values:
        values = ("all",)

    expanded: list[str] = []
    for raw_value in values:
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered == "all":
                if strategy == "ttt":
                    candidates = [key for key, spec in MODEL_REGISTRY.items() if spec.ttt is not None]
                else:
                    candidates = [key for key, spec in MODEL_REGISTRY.items() if spec.inference is not None]
            elif lowered in {"all_inference", "inference_all"}:
                candidates = [key for key, spec in MODEL_REGISTRY.items() if spec.inference is not None]
            elif lowered in {"all_ttt", "ttt_all"}:
                candidates = [key for key, spec in MODEL_REGISTRY.items() if spec.ttt is not None]
            else:
                candidates = [resolve_model_key(token)]

            for key in candidates:
                if key not in expanded:
                    expanded.append(key)
    return expanded


def output_dir_for(model: ModelSpec, strategy: str, config: PipelineConfig) -> Path:
    return resolve_repo_path(config.out_root) / strategy / model.key


def add_common_args(
    command: list[str],
    script: StrategyScript,
    strategy: str,
    config: PipelineConfig,
    out_dir: Path,
) -> None:
    command.extend(["--data-root", str(resolve_repo_path(config.data_root))])
    command.extend(["--out-dir", str(out_dir)])

    if config.workers is not None:
        command.extend(["--workers", str(config.workers)])
    if config.max_datasets is not None:
        command.extend(["--max-datasets", str(config.max_datasets)])
    if script.seed_arg:
        command.extend([script.seed_arg, str(config.random_state)])
    if config.verbose:
        command.append("--verbose")

    if script.gpu_mode == "gpus":
        if config.gpus:
            command.extend(["--gpus", str(config.gpus)])
    elif script.gpu_mode == "tabicl_ttt":
        if config.gpu_groups:
            command.extend(["--gpu-groups", str(config.gpu_groups)])
        elif config.gpus:
            command.extend(["--gpu-groups", "", "--gpus", str(config.gpus)])
    else:
        raise ValueError(f"Unknown gpu_mode={script.gpu_mode!r} for strategy={strategy!r}")


def build_command(model: ModelSpec, strategy: str, config: PipelineConfig) -> tuple[list[str], Path]:
    script = model.script_for(strategy)
    if script is None:
        raise ValueError(f"{model.display_name} does not support strategy={strategy!r}")
    if not script.script.exists():
        raise FileNotFoundError(f"Pipeline script does not exist: {script.script}")

    out_dir = output_dir_for(model, strategy, config)
    command = [config.python_executable, str(script.script)]
    add_common_args(command, script, strategy, config, out_dir)
    command.extend(script.default_args)
    command.extend(config.extra_args)
    command.extend(config.model_extra_args.get(model.key, ()))
    return command, out_dir


def unsupported_result(model: ModelSpec, strategy: str, config: PipelineConfig) -> PipelineRunResult:
    return PipelineRunResult(
        model_key=model.key,
        display_name=model.display_name,
        strategy=strategy,
        status="skip",
        returncode=None,
        command="",
        out_dir=str(output_dir_for(model, strategy, config)),
        elapsed_seconds=0.0,
        error=f"{model.display_name} has no {strategy} script registered",
    )


def run_one_model(model: ModelSpec, strategy: str, config: PipelineConfig) -> PipelineRunResult:
    started = time.time()
    try:
        if model.script_for(strategy) is None:
            if config.skip_unsupported:
                result = unsupported_result(model, strategy, config)
                print(f"[pipeline] skip {model.display_name} ({strategy}): {result.error}")
                return result
            raise ValueError(f"{model.display_name} does not support strategy={strategy!r}")

        command, out_dir = build_command(model, strategy, config)
        command_preview = shlex.join(command)
        print(f"[pipeline] {model.display_name} ({strategy})")
        print(command_preview)
        if config.dry_run:
            return PipelineRunResult(
                model_key=model.key,
                display_name=model.display_name,
                strategy=strategy,
                status="dry_run",
                returncode=None,
                command=command_preview,
                out_dir=str(out_dir),
                elapsed_seconds=0.0,
                error=None,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        elapsed = time.time() - started
        status = "ok" if completed.returncode == 0 else "fail"
        return PipelineRunResult(
            model_key=model.key,
            display_name=model.display_name,
            strategy=strategy,
            status=status,
            returncode=int(completed.returncode),
            command=command_preview,
            out_dir=str(out_dir),
            elapsed_seconds=elapsed,
            error=None if status == "ok" else f"returncode={completed.returncode}",
        )
    except Exception as exc:
        elapsed = time.time() - started
        error = f"{type(exc).__name__}: {exc}"
        print(f"[pipeline] {model.display_name} failed before launch: {error}")
        return PipelineRunResult(
            model_key=model.key,
            display_name=model.display_name,
            strategy=strategy,
            status="fail",
            returncode=None,
            command="",
            out_dir=str(output_dir_for(model, strategy, config)),
            elapsed_seconds=elapsed,
            error=error,
        )


def run_pipeline(model_keys: Sequence[str], config: PipelineConfig) -> list[PipelineRunResult]:
    results: list[PipelineRunResult] = []
    out_root = resolve_repo_path(config.out_root)
    stage_names = strategies_for(config)

    if not config.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    for strategy in stage_names:
        stage_config = replace(config, strategy=strategy)
        stage_results: list[PipelineRunResult] = []
        for model_key in model_keys:
            model = MODEL_REGISTRY[model_key]
            result = run_one_model(model, strategy, stage_config)
            stage_results.append(result)
            results.append(result)
            if result.status == "fail" and config.stop_on_failure:
                if not config.dry_run:
                    write_pipeline_manifest(out_root / strategy, stage_results)
                    write_pipeline_manifest(out_root / "manifests", results, stem="pipeline_runs_all")
                return results
        if not config.dry_run:
            write_pipeline_manifest(out_root / strategy, stage_results)

    if not config.dry_run:
        write_pipeline_manifest(out_root / "manifests", results, stem="pipeline_runs_all")
    return results


def write_pipeline_manifest(
    output_dir: Path,
    results: Sequence[PipelineRunResult],
    *,
    stem: str = "pipeline_runs",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    records = [asdict(result) for result in results]
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(PipelineRunResult.__annotations__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[pipeline] saved_manifest_json: {json_path}")
    print(f"[pipeline] saved_manifest_csv: {csv_path}")


def parse_model_extra_args(values: Sequence[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(
                "--model-extra-arg must use MODEL:ARG, for example "
                "--model-extra-arg=tabpfnv3:--ttt-epochs=8"
            )
        model_name, extra_arg = value.split(":", 1)
        model_key = resolve_model_key(model_name)
        if not extra_arg:
            raise ValueError("--model-extra-arg received an empty ARG part")
        parsed.setdefault(model_key, []).append(extra_arg)
    return parsed


def print_model_table() -> None:
    rows = []
    for spec in MODEL_REGISTRY.values():
        rows.append(
            {
                "model": spec.key,
                "display_name": spec.display_name,
                "inference": "yes" if spec.inference else "no",
                "ttt": "yes" if spec.ttt else "no",
                "aliases": ",".join(spec.aliases),
            }
        )
    widths = {
        key: max(len(key), *(len(row[key]) for row in rows))
        for key in ("model", "display_name", "inference", "ttt", "aliases")
    }
    header = "  ".join(key.ljust(widths[key]) for key in widths)
    print(header)
    print("  ".join("-" * widths[key] for key in widths))
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key in widths))
