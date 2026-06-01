#!/usr/bin/env python3
"""Rank data178 train/test shift and plot the most shifted datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_data178_tsne import (  # noqa: E402
    clean_numeric,
    encode_categorical,
    label_codes,
    load_optional_array,
    reduce_for_tsne,
    stratified_sample_indices,
)
from plot_ttt_decrease_train_test_shift import compute_embedding, plot_class_facets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute train/test shift metrics for all data178 datasets, then draw t-SNE "
            "plots for the most shifted datasets."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data178"))
    parser.add_argument("--out-dir", type=Path, default=Path("data178_tsne_plots/top_train_test_shift"))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-samples-per-split", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--max-categories", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def label_tvd(y_train: np.ndarray, y_test: np.ndarray) -> float:
    train_labels = np.asarray([str(x) for x in y_train])
    test_labels = np.asarray([str(x) for x in y_test])
    labels = sorted(set(train_labels) | set(test_labels))
    if not labels:
        return float("nan")
    train_props = np.asarray([(train_labels == label).mean() for label in labels])
    test_props = np.asarray([(test_labels == label).mean() for label in labels])
    return float(0.5 * np.abs(train_props - test_props).sum())


def load_feature_matrix(dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)

    stable_offset = int(hashlib.md5(dataset_dir.name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.random_state + stable_offset % 1_000_000)
    train_idx = stratified_sample_indices(y_train, args.max_samples_per_split, rng)
    test_idx = stratified_sample_indices(y_test, args.max_samples_per_split, rng)

    n_train = load_optional_array(dataset_dir, "N_train.npy")
    n_test = load_optional_array(dataset_dir, "N_test.npy")
    c_train = load_optional_array(dataset_dir, "C_train.npy")
    c_test = load_optional_array(dataset_dir, "C_test.npy")

    numeric = None
    categorical = None
    feature_count = 0
    if n_train is not None and n_test is not None:
        numeric = np.vstack([n_train[train_idx], n_test[test_idx]])
        feature_count += numeric.shape[1] if numeric.ndim > 1 else 1
    if c_train is not None and c_test is not None:
        categorical = np.vstack([c_train[train_idx], c_test[test_idx]])
        feature_count += categorical.shape[1] if categorical.ndim > 1 else 1

    blocks = []
    numeric_block = clean_numeric(numeric)
    if numeric_block is not None:
        blocks.append(numeric_block)
    categorical_block = encode_categorical(categorical, args.max_categories)
    if categorical_block is not None:
        blocks.append(categorical_block)
    if not blocks:
        raise ValueError("missing both numeric and categorical features")

    x = sparse.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    domain = np.asarray([0] * len(train_idx) + [1] * len(test_idx))
    return {
        "x": x,
        "domain": domain,
        "y_train": y_train,
        "y_test": y_test,
        "sampled_train": len(train_idx),
        "sampled_test": len(test_idx),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features_raw": feature_count,
        "n_features_encoded": x.shape[1],
    }


def domain_auc(x: sparse.csr_matrix, domain: np.ndarray, random_state: int) -> float:
    _, counts = np.unique(domain, return_counts=True)
    min_count = int(counts.min())
    if min_count < 3:
        return float("nan")
    cv = StratifiedKFold(
        n_splits=min(5, min_count),
        shuffle=True,
        random_state=random_state,
    )
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    scores = cross_val_predict(clf, x, domain, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(domain, scores))


def centroid_gap(x: sparse.csr_matrix, domain: np.ndarray, args: argparse.Namespace) -> float:
    reduced = reduce_for_tsne(x, min(args.pca_dims, 32), args.random_state)
    train = reduced[domain == 0]
    test = reduced[domain == 1]
    if len(train) == 0 or len(test) == 0:
        return float("nan")
    center_dist = float(np.linalg.norm(train.mean(axis=0) - test.mean(axis=0)))
    pooled_radius = float(np.sqrt(np.mean(np.var(reduced, axis=0)))) + 1e-12
    return center_dist / pooled_radius


def score_row(dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_matrix(dataset_dir, args)
    auc = domain_auc(payload["x"], payload["domain"], args.random_state)
    gap = centroid_gap(payload["x"], payload["domain"], args)
    tvd = label_tvd(payload["y_train"], payload["y_test"])
    auc_component = max(0.0, 2.0 * (auc - 0.5)) if np.isfinite(auc) else 0.0
    gap_component = gap / (1.0 + gap) if np.isfinite(gap) else 0.0
    shift_score = 0.75 * auc_component + 0.20 * gap_component + 0.05 * tvd
    _, classes = label_codes(np.concatenate([payload["y_train"], payload["y_test"]]))
    return {
        "dataset_name": dataset_dir.name,
        "status": "ok",
        "shift_score": shift_score,
        "domain_auc": auc,
        "norm_centroid_gap": gap,
        "label_tvd": tvd,
        "n_classes": len(classes),
        "sampled_train": payload["sampled_train"],
        "sampled_test": payload["sampled_test"],
        "n_train": payload["n_train"],
        "n_test": payload["n_test"],
        "n_features_raw": payload["n_features_raw"],
        "n_features_encoded": payload["n_features_encoded"],
        "png": "",
        "error": "",
    }


def write_index(out_dir: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    ok_rows = [row for row in rows if row.get("png")]
    cards = []
    for rank, row in enumerate(ok_rows, start=1):
        img_name = Path(row["png"]).name
        cards.append(
            f'<a class="card" href="{html.escape(img_name)}">'
            f'<img src="{html.escape(img_name)}" loading="lazy">'
            f'<span>#{rank} {html.escape(row["dataset_name"])}</span>'
            f'<small>score={float(row["shift_score"]):.3f}; '
            f'AUC={float(row["domain_auc"]):.3f}; '
            f'gap={float(row["norm_centroid_gap"]):.3f}</small></a>'
        )
    failures = "".join(
        f"<li>{html.escape(row['dataset_name'])}: {html.escape(row['error'])}</li>"
        for row in rows
        if row["status"] != "ok"
    )
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top data178 Train/Test Shift t-SNE</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; color: #111; text-decoration: none; }}
    .card img {{ width: 100%; display: block; }}
    .card span {{ display: block; margin-top: 6px; font-size: 13px; overflow-wrap: anywhere; }}
    .card small {{ display: block; color: #555; margin-top: 3px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Top {top_k} data178 Train/Test Shift t-SNE</h1>
  <p>Ranking uses a composite shift score dominated by train/test domain-classifier AUC, plus centroid gap and label-distribution TVD. Blue points are train; orange points are test.</p>
  <div class="grid">{''.join(cards)}</div>
  {"<h2>Failed</h2><ul>" + failures + "</ul>" if failures else ""}
</body>
</html>
"""
    (out_dir / "index.html").write_text(content)


def write_contact_sheet(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("png")]
    if not ok_rows:
        return
    cols = min(4, len(ok_rows))
    rows_n = math.ceil(len(ok_rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 3.6), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for rank, (ax, row) in enumerate(zip(axes_arr, ok_rows), start=1):
        img = plt.imread(row["png"])
        ax.imshow(img)
        ax.set_title(
            f"#{rank} {row['dataset_name']}\nAUC={float(row['domain_auc']):.3f}, "
            f"score={float(row['shift_score']):.3f}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes_arr[len(ok_rows) :]:
        ax.axis("off")
    fig.savefig(out_dir / "contact_sheet.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in args.out_dir.glob("[0-9][0-9]_*_train_test_shift.png"):
        old_png.unlink()

    datasets = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        datasets = [p for p in datasets if p.name in wanted]

    metric_rows: list[dict[str, Any]] = []
    for i, dataset_dir in enumerate(datasets, start=1):
        print(f"[metric {i}/{len(datasets)}] {dataset_dir.name}", flush=True)
        try:
            metric_rows.append(score_row(dataset_dir, args))
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            metric_rows.append(
                {
                    "dataset_name": dataset_dir.name,
                    "status": "failed",
                    "shift_score": "",
                    "domain_auc": "",
                    "norm_centroid_gap": "",
                    "label_tvd": "",
                    "n_classes": "",
                    "sampled_train": "",
                    "sampled_test": "",
                    "n_train": "",
                    "n_test": "",
                    "n_features_raw": "",
                    "n_features_encoded": "",
                    "png": "",
                    "error": repr(exc),
                }
            )

    ok_metric_rows = [row for row in metric_rows if row["status"] == "ok"]
    top_rows = sorted(ok_metric_rows, key=lambda row: row["shift_score"], reverse=True)[: args.top_k]
    top_names = {row["dataset_name"] for row in top_rows}

    plot_rows: list[dict[str, Any]] = []
    for i, row in enumerate(top_rows, start=1):
        dataset_name = row["dataset_name"]
        out_path = args.out_dir / f"{i:02d}_{dataset_name}_train_test_shift.png"
        print(f"[plot {i}/{len(top_rows)}] {dataset_name}", flush=True)
        try:
            payload = compute_embedding(args.data_root / dataset_name, args)
            plot_class_facets(
                dataset_name=dataset_name,
                result_row=None,
                payload=payload,
                out_path=out_path,
                dpi=args.dpi,
            )
            row = dict(row)
            row["png"] = str(out_path)
            plot_rows.append(row)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            row = dict(row)
            row["status"] = "failed"
            row["error"] = repr(exc)
            plot_rows.append(row)

    for row in metric_rows:
        if row["dataset_name"] not in top_names:
            plot_rows.append(row)

    fieldnames = [
        "dataset_name",
        "status",
        "shift_score",
        "domain_auc",
        "norm_centroid_gap",
        "label_tvd",
        "n_classes",
        "sampled_train",
        "sampled_test",
        "n_train",
        "n_test",
        "n_features_raw",
        "n_features_encoded",
        "png",
        "error",
    ]
    ranked_rows = sorted(
        plot_rows,
        key=lambda row: float(row["shift_score"]) if row["shift_score"] != "" else -1.0,
        reverse=True,
    )
    with (args.out_dir / "train_test_shift_ranking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    pd.DataFrame(ranked_rows).head(args.top_k).to_markdown(
        args.out_dir / "top_train_test_shift.md",
        index=False,
        floatfmt=".4f",
    )
    write_index(args.out_dir, ranked_rows[: args.top_k], args.top_k)
    write_contact_sheet(args.out_dir, ranked_rows[: args.top_k])

    ok_plots = sum(bool(row.get("png")) and row["status"] == "ok" for row in ranked_rows[: args.top_k])
    print(
        f"done: ranked={len(ok_metric_rows)}, plotted={ok_plots}/{args.top_k}, "
        f"out_dir={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
