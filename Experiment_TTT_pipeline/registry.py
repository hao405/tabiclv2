from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class StrategyScript:
    script: Path
    default_args: tuple[str, ...] = ()
    gpu_mode: str = "gpus"
    seed_arg: str | None = "--random-state"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    inference: StrategyScript | None
    ttt: StrategyScript | None = None

    def script_for(self, strategy: str) -> StrategyScript | None:
        if strategy == "inference":
            return self.inference
        if strategy == "ttt":
            return self.ttt
        raise ValueError(f"Unsupported strategy: {strategy!r}")


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tabicl": ModelSpec(
        key="tabicl",
        display_name="TabICL",
        aliases=("tabicl", "tabicl_v11", "tabiclv1.1"),
        inference=StrategyScript(
            script=repo_path("benchmark.py"),
            gpu_mode="gpus",
        ),
        ttt=StrategyScript(
            script=repo_path("1C_Chunk_TTT.py"),
            gpu_mode="tabicl_ttt",
        ),
    ),
    "tabpfnv2": ModelSpec(
        key="tabpfnv2",
        display_name="TabPFN v2",
        aliases=("tabpfnv2", "tabpfn_v2", "tabpfn2"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "benchmark_infer.py"),
            default_args=("--model-version", "v2"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "Tabpfn_1c_ttt.py"),
            default_args=("--model-version", "v2", "--ttt"),
        ),
    ),
    "tabpfnv25": ModelSpec(
        key="tabpfnv25",
        display_name="TabPFN v2.5",
        aliases=("tabpfnv25", "tabpfn_v25", "tabpfnv2.5", "tabpfn25", "tabpfn"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "benchmark_infer.py"),
            default_args=("--model-version", "v2.5"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "Tabpfn_1c_ttt.py"),
            default_args=("--model-version", "v2.5", "--ttt"),
        ),
    ),
    "tabpfnv26": ModelSpec(
        key="tabpfnv26",
        display_name="TabPFN v2.6",
        aliases=("tabpfnv26", "tabpfn_v26", "tabpfnv2.6", "tabpfn26"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "benchmark_infer.py"),
            default_args=("--model-version", "v2.6"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "Tabpfn_1c_ttt.py"),
            default_args=("--model-version", "v2.6", "--ttt"),
        ),
    ),
    "tabpfnv3": ModelSpec(
        key="tabpfnv3",
        display_name="TabPFN v3",
        aliases=("tabpfnv3", "tabpfn_v3", "tabpfn3"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "benchmark_infer.py"),
            default_args=("--model-version", "v3"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "TabPFN-main", "Tabpfn_1c_ttt.py"),
            default_args=("--model-version", "v3", "--ttt"),
        ),
    ),
    "limix": ModelSpec(
        key="limix",
        display_name="LimiX",
        aliases=("limix", "limix16m", "limix-16m"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "LimiX", "benchmark_infer.py"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "LimiX", "limix_ttt.py"),
            default_args=("--ttt-holdout",),
        ),
    ),
    "tabr": ModelSpec(
        key="tabr",
        display_name="TaBR",
        aliases=("tabr", "tabular-dl-tabr"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "tabular-dl-tabr", "benchmark_infer.py"),
        ),
        ttt=None,
    ),
    "tabdpt": ModelSpec(
        key="tabdpt",
        display_name="TabDPT",
        aliases=("tabdpt", "tabdpt-inference"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "TabDPT-inference", "tabdpt_ttt.py"),
            default_args=("--no-ttt",),
            seed_arg="--seed",
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "TabDPT-inference", "tabdpt_ttt.py"),
            seed_arg="--seed",
        ),
    ),
    "orion_msp": ModelSpec(
        key="orion_msp",
        display_name="Orion-MSP",
        aliases=("orion_msp", "orion", "orion-msp"),
        inference=StrategyScript(
            script=repo_path("baseline_compare", "Orion-MSP", "benchmark_infer.py"),
        ),
        ttt=StrategyScript(
            script=repo_path("baseline_compare", "Orion-MSP", "orion_msp_ttt.py"),
            default_args=("--ttt",),
        ),
    ),
}


def alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, spec in MODEL_REGISTRY.items():
        aliases[key] = key
        for alias in spec.aliases:
            aliases[alias.lower()] = key
    return aliases


def resolve_model_key(value: str) -> str:
    normalized = value.strip().lower()
    aliases = alias_map()
    if normalized not in aliases:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {value!r}. Known models: {known}")
    return aliases[normalized]
