#!/usr/bin/env python3
"""Generate a paired, benchmark-format synthetic suite for TFM experiments.

The suite combines two random SCM teachers and two sklearn tree teachers.  Every
base task has one clean test split and controlled test-only OOD siblings, so all
model/method cells can be compared on exactly the same arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


GENERATOR_CHOICES = ("scm_linear", "scm_nonlinear", "tree_dt", "tree_extra")
PAPER_VARIANTS = (
    "clean",
    "covariate_mild",
    "covariate_severe",
    "missing_mild",
    "missing_severe",
    "outlier_mild",
    "outlier_severe",
)
PAPER512_VARIANTS = (*PAPER_VARIANTS, "rotation_severe")
SMOKE_VARIANTS = ("clean", "covariate_mild")
MANIFEST_COLUMNS = (
    "dataset_name",
    "dataset_dir",
    "suite_version",
    "preset",
    "base_task_id",
    "generator_family",
    "teacher_type",
    "variant",
    "stress_axis",
    "stress_level",
    "stress_seed",
    "seed",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "task_type",
)


@dataclass(frozen=True)
class BaseSpec:
    generator_family: str
    n_classes: int
    seed: int

    @property
    def base_task_id(self) -> str:
        return f"{self.generator_family}_k{self.n_classes}_seed{self.seed}"


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    tokens = value.replace(",", " ").split() if isinstance(value, str) else value
    values: list[int] = []
    for token in tokens:
        number = int(token)
        if number not in values:
            values.append(number)
    return values


def stable_seed(*parts: int) -> int:
    value = 17
    for part in parts:
        value = (value * 1_000_003 + int(part) * 97_409) % (2**32 - 1)
    return int(value)


def quantile_labels(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
    thresholds = np.unique(np.quantile(train_scores, quantiles))
    if len(thresholds) != n_classes - 1:
        raise ValueError("Teacher scores do not support the requested number of classes")
    y_train = np.digitize(train_scores, thresholds).astype("int64")
    y_test = np.digitize(test_scores, thresholds).astype("int64")
    if len(np.unique(y_train)) != n_classes:
        raise ValueError("Training split is missing a requested class")
    return y_train, y_test


def make_scm_task(
    spec: BaseSpec,
    *,
    n_train: int,
    n_test: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(spec.seed, spec.n_classes, 11))
    total = n_train + n_test
    roots = rng.normal(size=(total, n_features))
    X = np.zeros_like(roots)
    X[:, 0] = roots[:, 0]
    for feature_idx in range(1, n_features):
        n_parents = min(3, feature_idx)
        parents = rng.choice(feature_idx, size=n_parents, replace=False)
        weights = rng.normal(scale=0.8, size=n_parents)
        signal = X[:, parents] @ weights
        if spec.generator_family == "scm_nonlinear":
            signal = np.sin(signal) + 0.35 * np.square(X[:, parents[0]])
        X[:, feature_idx] = signal + rng.normal(scale=0.5, size=total)

    target_weights = rng.normal(size=min(6, n_features))
    scores = X[:, -len(target_weights) :] @ target_weights
    if spec.generator_family == "scm_nonlinear":
        scores = scores + 1.2 * X[:, 0] * X[:, 1] + np.sin(2.0 * X[:, 2])
    scores = scores + rng.normal(scale=0.25, size=total)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = quantile_labels(
        scores[:n_train], scores[n_train:], spec.n_classes
    )
    return X_train, y_train, X_test, y_test


def latent_teacher_target(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = rng.normal(size=min(8, X.shape[1]))
    return (
        X[:, : len(weights)] @ weights
        + 1.5 * X[:, 0] * X[:, 1]
        + np.sin(2.0 * X[:, 2])
        + 0.4 * np.square(X[:, 3])
        + rng.normal(scale=0.2, size=len(X))
    )


def make_tree_task(
    spec: BaseSpec,
    *,
    n_train: int,
    n_test: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = stable_seed(spec.seed, spec.n_classes, 23)
    rng = np.random.default_rng(seed)
    teacher_X = rng.normal(size=(4096, n_features))
    teacher_y = latent_teacher_target(teacher_X, rng)
    if spec.generator_family == "tree_dt":
        teacher = DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=8,
            random_state=seed,
        )
        teacher_type = "DecisionTreeRegressor"
    else:
        teacher = ExtraTreesRegressor(
            n_estimators=64,
            max_depth=8,
            min_samples_leaf=4,
            n_jobs=1,
            random_state=seed,
        )
        teacher_type = "ExtraTreesRegressor"
    teacher.fit(teacher_X, teacher_y)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train, y_test = quantile_labels(
        teacher.predict(X_train), teacher.predict(X_test), spec.n_classes
    )
    setattr(make_tree_task, "last_teacher_type", teacher_type)
    return X_train, y_train, X_test, y_test


def variant_metadata(variant: str) -> tuple[str, str]:
    if variant == "clean":
        return "none", "clean"
    axis, level = variant.rsplit("_", 1)
    return axis, level


def apply_test_variant(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    variant: str,
    stress_seed: int,
) -> np.ndarray:
    result = X_test.copy()
    if variant == "clean":
        return result

    axis, level = variant_metadata(variant)
    severe = level == "severe"
    rng = np.random.default_rng(stress_seed)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    if axis == "covariate":
        count = max(1, result.shape[1] // 3)
        magnitude = 1.5 if severe else 0.75
        result[:, :count] += magnitude * scale[:count]
    elif axis == "missing":
        probability = 0.30 if severe else 0.10
        mask = rng.random(result.shape) < probability
        result[mask] = np.nan
    elif axis == "outlier":
        probability = 0.05 if severe else 0.01
        magnitude = 12.0 if severe else 6.0
        mask = rng.random(result.shape) < probability
        signs = np.where(rng.random(result.shape) < 0.5, -1.0, 1.0)
        result += mask * signs * magnitude * scale
    elif axis == "rotation":
        mean = np.nanmean(X_train, axis=0)
        standardized = (result - mean) / scale
        random_matrix = rng.normal(size=(result.shape[1], result.shape[1]))
        orthogonal, _ = np.linalg.qr(random_matrix)
        result = (standardized @ orthogonal) * scale + mean
    else:
        raise ValueError(f"Unsupported stress axis: {axis}")
    return result


def build_base_specs(args: argparse.Namespace) -> list[BaseSpec]:
    seeds = parse_int_list(args.seeds)
    class_counts = [2] if args.preset == "smoke" else [2, 5]
    return [
        BaseSpec(generator_family=generator, n_classes=n_classes, seed=seed)
        for seed in seeds
        for generator in GENERATOR_CHOICES
        for n_classes in class_counts
    ]


def variants_for_preset(preset: str) -> tuple[str, ...]:
    if preset == "smoke":
        return SMOKE_VARIANTS
    if preset == "paper512":
        return PAPER512_VARIANTS
    return PAPER_VARIANTS


def ensure_output_root(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"{path} is not empty; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def generate(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    specs = build_base_specs(args)
    variants = variants_for_preset(args.preset)
    target_count = len(specs) * len(variants)
    if args.preset == "paper512" and target_count != 512:
        raise ValueError(
            "paper512 requires exactly eight generation seeds and resolves to 512 datasets"
        )
    print(f"preset: {args.preset}")
    print(f"base_tasks: {len(specs)}")
    print(f"variants_per_base: {len(variants)}")
    print(f"target_datasets: {target_count}")
    print(f"data_root: {data_root}")
    if args.dry_run:
        print("dry_run: true")
        return 0

    ensure_output_root(data_root, args.force)
    rows: list[dict[str, Any]] = []
    for base_index, spec in enumerate(specs):
        if spec.generator_family.startswith("scm_"):
            X_train, y_train, X_test, y_test = make_scm_task(
                spec,
                n_train=args.n_train,
                n_test=args.n_test,
                n_features=args.n_features,
            )
            teacher_type = "random_causal_dag"
        else:
            X_train, y_train, X_test, y_test = make_tree_task(
                spec,
                n_train=args.n_train,
                n_test=args.n_test,
                n_features=args.n_features,
            )
            teacher_type = str(getattr(make_tree_task, "last_teacher_type"))

        stress_seed = stable_seed(spec.seed, spec.n_classes, base_index, 71)
        for variant in variants:
            stress_axis, stress_level = variant_metadata(variant)
            X_variant = apply_test_variant(
                X_train, X_test, variant=variant, stress_seed=stress_seed
            )
            dataset_name = f"topsyn_{spec.base_task_id}_{variant}"
            dataset_dir = data_root / dataset_name
            dataset_dir.mkdir()
            np.save(dataset_dir / "N_train.npy", X_train.astype("float32"))
            np.save(dataset_dir / "N_test.npy", X_variant.astype("float32"))
            np.save(dataset_dir / "y_train.npy", y_train.astype("int64"))
            np.save(dataset_dir / "y_test.npy", y_test.astype("int64"))
            task_type = "binclass" if spec.n_classes == 2 else "multiclass"
            common = {
                "suite_version": "topconf_synthetic_v1",
                "preset": args.preset,
                "base_task_id": spec.base_task_id,
                "generator_family": spec.generator_family,
                "teacher_type": teacher_type,
                "variant": variant,
                "stress_axis": stress_axis,
                "stress_level": stress_level,
                "stress_seed": stress_seed,
                "seed": spec.seed,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "n_features": args.n_features,
                "n_classes": spec.n_classes,
                "task_type": task_type,
            }
            write_json(
                dataset_dir / "info.json",
                {
                    **common,
                    "source": "topconf_synthetic_suite",
                    "n_num_features": args.n_features,
                    "n_cat_features": 0,
                    "classes": list(range(spec.n_classes)),
                    "paper_inspiration": (
                        "TabPFN/TabICL SCM priors; MITRA-style tree priors; "
                        "controlled tabular inductive-bias stress tests"
                    ),
                    "claim_boundary": (
                        "clean is prior-ID; non-clean variants are paired controlled OOD. "
                        "This suite is inspired by, not an exact reproduction of, prior papers."
                    ),
                },
            )
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_dir": str(dataset_dir),
                    **common,
                }
            )

    manifest_path = data_root / "synthetic_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        data_root / "generation_config.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "data_root": str(data_root),
            "target_datasets": target_count,
            "generators": list(GENERATOR_CHOICES),
            "variants": list(variants),
            "base_specs": [asdict(spec) for spec in specs],
            "args": vars(args),
        },
    )
    print(f"generated_datasets: {len(rows)}")
    print(f"synthetic_manifest: {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired SCM/tree-prior synthetic benchmark tasks."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--preset", choices=("smoke", "paper", "paper512"), default="paper"
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_train < 20 or args.n_test < 20 or args.n_features < 4:
        raise SystemExit("n-train/n-test must be >=20 and n-features must be >=4")
    raise SystemExit(generate(args))


if __name__ == "__main__":
    main()
