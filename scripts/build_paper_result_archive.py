#!/usr/bin/env python3
"""Build the curated paper-result archive from local and jiqun artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import pandas as pd


MODELS = ("TabICLv2", "TabPFNv3", "LimiX-2M")
DATASETS = ("data184", "OpenML-CC18", "Graph-SCM")
METHODS = (
    "Infer",
    "FT",
    "F-aware FT",
    "LoRA",
    "ln_head_embedding",
    "Last-block",
    "LoCalPFN",
    "MICP",
)
ADAPTIVE_METHODS = frozenset(METHODS[1:])
PERFORMANCE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "log_loss",
)
LOWER_IS_BETTER = frozenset({"log_loss"})
SUPPORT_FILENAMES = (
    "summary.txt",
    "summary.md",
    "manager_manifest.json",
    "matrix_manifest.json",
    "run_manifest.json",
    "best_params.json",
    "best_trial_summary.json",
)
EXCLUDED_PATH_TAGS = (
    "/smoke",
    "_smoke",
    "/debug",
    "_debug",
    "interrupted",
    "/_legacy/",
    "targeted_recovery",
    "_recovery/",
    "_recovery_attempt",
)
REMOTE_SCAN_ROOTS = (
    "managed_experiments/openmlcc18_2x3_full_20260715_223032",
    "PEFT/tabiclv2",
    "PEFT/tabpfnv3",
    "PEFT_compare",
    "limix/data184",
    "tabpfn",
    "tabicl",
)
LOCAL_SCAN_ROOTS = (
    "tabicl/v2",
    "tabpfn/v3",
    "PEFT_results",
    "合成数据实验",
    "PEFT_compare",
    "limix",
)
MANAGED_MARKER = ".paper_result_archive_managed.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def falsey(value: object) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "n"}


def safe_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


@dataclass(frozen=True, order=True)
class MatrixCell:
    model: str
    dataset: str
    method: str


@dataclass
class SourceObservation:
    size: int
    mtime: float
    sha256: str


@dataclass
class Candidate:
    source_host: str
    source_path: str
    cell: MatrixCell | None
    data: bytes | None = None
    frame: pd.DataFrame | None = None
    observation: SourceObservation | None = None
    eligible: bool = True
    invalid: bool = False
    unstable: bool = False
    active: bool = False
    rejection_reason: str = ""
    dataset_column: str = "dataset_name"
    result_rows: int = 0
    unique_datasets: int = 0
    status_ok: int = 0
    mean_accuracy: float = float("-inf")
    selected_on_eval: bool = False
    missing_names: tuple[str, ...] = ()
    unexpected_names: tuple[str, ...] = ()
    members: tuple["Candidate", ...] = ()
    selection_policy: str = ""
    selected_seeds: tuple[int, ...] = ()

    def selection_key(self) -> tuple[int, int, int, int, float, float]:
        formal = int(self.eligible and not self.invalid and not self.unstable)
        exact_membership = int(not self.missing_names and not self.unexpected_names)
        mtime = self.observation.mtime if self.observation else 0.0
        return (
            formal,
            exact_membership,
            self.unique_datasets,
            self.status_ok,
            self.mean_accuracy,
            mtime,
        )


class SourceBackend:
    host_label: str

    def list_result_csvs(self) -> list[str]:
        raise NotImplementedError

    def observe(self, path: str) -> SourceObservation:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def stable_read(self, path: str, retries: int = 3) -> tuple[bytes | None, SourceObservation, bool]:
        last_observation: SourceObservation | None = None
        for _ in range(retries):
            before = self.observe(path)
            data = self.read_bytes(path)
            after = self.observe(path)
            last_observation = after
            if before == after and sha256_bytes(data) == after.sha256:
                return data, after, True
            time.sleep(0.05)
        assert last_observation is not None
        return None, last_observation, False

    def optional_support(self, csv_path: str) -> list[str]:
        parent = str(PurePosixPath(csv_path).parent)
        found: list[str] = []
        for name in SUPPORT_FILENAMES:
            candidate = str(PurePosixPath(parent) / name)
            try:
                self.observe(candidate)
            except (FileNotFoundError, RuntimeError):
                continue
            found.append(candidate)
        return found

    def active_text(self) -> str:
        return ""


class LocalSourceBackend(SourceBackend):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.host_label = "local"

    def list_result_csvs(self) -> list[str]:
        paths: list[str] = []
        for relative in LOCAL_SCAN_ROOTS:
            root = self.root / relative
            if root.exists():
                paths.extend(str(path.resolve()) for path in root.rglob("all_classification_results.csv"))
        return sorted(set(paths))

    def observe(self, path: str) -> SourceObservation:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(path)
        stat = target.stat()
        return SourceObservation(stat.st_size, stat.st_mtime, sha256_path(target))

    def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()


class SSHSourceBackend(SourceBackend):
    def __init__(self, host: str, root: str):
        self.host = host
        self.root = root.rstrip("/")
        self.host_label = host
        self._prefetched: dict[str, tuple[bytes | None, SourceObservation, bool]] = {}

    def _run(self, script: str, *, binary: bool = False) -> bytes | str:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=60",
                "-o",
                "ControlPath=/tmp/codex-paper-archive-%r@%h:%p",
                self.host,
                f"bash -lc {shlex.quote(script)}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")

    def list_result_csvs(self) -> list[str]:
        roots = " ".join(json.dumps(f"{self.root}/{relative}") for relative in REMOTE_SCAN_ROOTS)
        script = (
            f"for root in {roots}; do "
            '[ -d "$root" ] && find "$root" -type f -name all_classification_results.csv; '
            "done"
        )
        output = str(self._run(script))
        return sorted(set(line.strip() for line in output.splitlines() if line.strip()))

    def observe(self, path: str) -> SourceObservation:
        quoted = json.dumps(path)
        script = (
            f'test -f {quoted} || exit 44; '
            f'stat -c "%s|%Y" {quoted}; '
            f'sha256sum {quoted} | cut -d" " -f1'
        )
        try:
            lines = str(self._run(script)).strip().splitlines()
        except RuntimeError as exc:
            if "exit" in str(exc).lower():
                raise FileNotFoundError(path) from exc
            raise
        if len(lines) < 2:
            raise FileNotFoundError(path)
        size_text, mtime_text = lines[0].split("|", 1)
        return SourceObservation(int(size_text), float(mtime_text), lines[1].strip())

    def read_bytes(self, path: str) -> bytes:
        return bytes(self._run(f"cat {json.dumps(path)}", binary=True))

    def prefetch(self, paths: Sequence[str]) -> None:
        if not paths:
            return
        probe = (
            "import base64,hashlib,json,os,sys;"
            "paths=json.loads(sys.argv[1]);out={};"
            "\nfor p in paths:\n"
            " try:\n"
            "  a=os.stat(p);d=open(p,'rb').read();b=os.stat(p);h=hashlib.sha256(d).hexdigest();"
            "out[p]={'size':b.st_size,'mtime':b.st_mtime,'sha256':h,"
            "'stable':(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns),"
            "'data':base64.b64encode(d).decode('ascii')}\n"
            " except Exception as e: out[p]={'error':type(e).__name__+':'+str(e)}\n"
            "print(json.dumps(out,separators=(',',':')))"
        )
        payload = str(
            self._run(
                f"python -c {shlex.quote(probe)} {shlex.quote(json.dumps(list(paths)))}"
            )
        )
        decoded = json.loads(payload)
        for path, meta in decoded.items():
            if "error" in meta:
                continue
            observation = SourceObservation(
                int(meta["size"]),
                float(meta["mtime"]),
                str(meta["sha256"]),
            )
            data = base64.b64decode(meta["data"])
            stable = bool(meta["stable"]) and len(data) == observation.size
            self._prefetched[path] = (data if stable else None, observation, stable)

    def stable_read(self, path: str, retries: int = 3) -> tuple[bytes | None, SourceObservation, bool]:
        if path in self._prefetched:
            return self._prefetched[path]
        probe = (
            "import hashlib,json,os,sys;"
            "p=sys.argv[1];"
            "a=os.stat(p);"
            "d=open(p,'rb').read();"
            "b=os.stat(p);"
            "h=hashlib.sha256(d).hexdigest();"
            "meta={'size':b.st_size,'mtime':b.st_mtime,'sha256':h,"
            "'stable':(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)};"
            "sys.stdout.buffer.write((json.dumps(meta)+'\\n').encode()+d)"
        )
        last_observation: SourceObservation | None = None
        for _ in range(retries):
            payload = bytes(
                self._run(
                    f"python -c {shlex.quote(probe)} {shlex.quote(path)}",
                    binary=True,
                )
            )
            header, data = payload.split(b"\n", 1)
            meta = json.loads(header)
            last_observation = SourceObservation(
                int(meta["size"]),
                float(meta["mtime"]),
                str(meta["sha256"]),
            )
            if bool(meta["stable"]) and len(data) == last_observation.size:
                return data, last_observation, True
            time.sleep(0.05)
        assert last_observation is not None
        return None, last_observation, False

    def active_text(self) -> str:
        script = "ps -eo pid,ppid,args; tmux list-windows -t zh -F '#{window_index}|#{pane_current_command}' 2>/dev/null"
        try:
            return str(self._run(script))
        except RuntimeError:
            return ""

    def optional_support(self, csv_path: str) -> list[str]:
        parent = str(PurePosixPath(csv_path).parent)
        candidates = [str(PurePosixPath(parent) / name) for name in SUPPORT_FILENAMES]
        return [path for path in candidates if path in self._prefetched]


def expected_cells() -> list[MatrixCell]:
    return [
        MatrixCell(model, dataset, method)
        for model in MODELS
        for dataset in DATASETS
        for method in METHODS
    ]


def excluded_path(path: str) -> str:
    lower = path.lower()
    for tag in EXCLUDED_PATH_TAGS:
        if tag in lower:
            return f"excluded_path_tag:{tag}"
    if "/optuna/" in lower and "/best_matrix/" not in lower:
        return "noncanonical_optuna_trial"
    return ""


def remote_candidate_relevant(path: str) -> bool:
    lower = path.lower()
    if "/results/peft/" in lower and "openmlcc18" not in lower:
        return False
    if "/results/tabpfn/" in lower or "/results/tabicl/" in lower:
        return any(
            token in lower
            for token in ("data184", "seed_sweep", "best_para", "tabpfn_v3data184")
        )
    return True


def infer_dataset(path: str, frame: pd.DataFrame | None) -> str | None:
    lower = path.lower()
    if "graph_scm" in lower or "合成数据实验" in path:
        return "Graph-SCM"
    if "openmlcc18" in lower or "openml_cc18" in lower:
        return "OpenML-CC18"
    if any(
        token in lower
        for token in ("data184", "talent184", "seed_sweep", "tabpfn_v3data184", "tabpfnv3_best_para")
    ):
        return "data184"
    if frame is not None and "dataset_name" in frame:
        names = frame["dataset_name"].astype(str)
        if names.str.startswith("stage").mean() > 0.8:
            return "Graph-SCM"
        if names.str.startswith("OpenML-ID-").mean() > 0.8:
            return "OpenML-CC18"
    return None


def infer_model(path: str, frame: pd.DataFrame | None) -> str | None:
    lower = path.lower().replace("_", "-")
    if "limix" in lower:
        return "LimiX-2M"
    if any(token in lower for token in ("tabpfn-v3", "tabpfnv3", "/tabpfn/v3/")):
        return "TabPFNv3"
    if any(token in lower for token in ("tabicl-v2", "tabiclv2", "/tabicl/v2/")):
        return "TabICLv2"
    if frame is not None and "model_family" in frame:
        values = set(frame["model_family"].dropna().astype(str).str.lower())
        if any("tabpfnv3" in value or "tabpfn-v3" in value for value in values):
            return "TabPFNv3"
        if any("tabiclv2" in value or "tabicl-v2" in value for value in values):
            return "TabICLv2"
    return None


def infer_method(path: str, frame: pd.DataFrame | None) -> str | None:
    lower = path.lower()
    if "localpfn" in lower:
        return "LoCalPFN"
    if re.search(r"(^|[/_-])micp([/_-]|$)", lower):
        return "MICP"
    if "ln_head_embedding" in lower:
        return "ln_head_embedding"
    if "last_layers" in lower or "last_block" in lower:
        return "Last-block"
    if re.search(r"(^|[/_-])lora([/_-]|$)", lower):
        return "LoRA"
    # Persisted selector telemetry is stronger evidence than folder names.  In
    # particular, tabpfnv3_best_para is a random-C FT run despite its generic
    # "best" directory name.
    if frame is not None and "ttt_c_selection" in frame:
        values = set(frame["ttt_c_selection"].dropna().astype(str).str.lower())
        if values & {"f_mmd", "f_test_centroid_reserve"}:
            return "F-aware FT"
        if "random" in values:
            return "FT"
    if "faware_c_ft_baseline" in lower and "/ft/" in lower:
        return "FT"
    if any(
        token in lower
        for token in (
            "faware_ft",
            "faware_c",
            "/f_mmd/",
            "/f_test_centroid_reserve/",
            "best_para",
        )
    ):
        return "F-aware FT"
    if any(token in lower for token in ("infer_baseline", "/infer/", "base_infer")):
        return "Infer"
    if any(token in lower for token in ("/ft/", "/random/", "ttt_baseline")):
        return "FT"
    if frame is not None:
        if "peft_method" in frame:
            values = set(frame["peft_method"].dropna().astype(str).str.lower())
            mapping = {
                "lora": "LoRA",
                "ln_head_embedding": "ln_head_embedding",
                "last_layers": "Last-block",
                "last_block": "Last-block",
            }
            if len(values) == 1 and next(iter(values)) in mapping:
                return mapping[next(iter(values))]
        if "ttt_applied" in frame:
            return "FT"
    return None


def parse_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def dataset_column(frame: pd.DataFrame) -> str | None:
    for name in ("dataset_name", "dataset", "task_name"):
        if name in frame.columns:
            return name
    return None


def ok_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series([True] * len(frame), index=frame.index)
    return frame["status"].fillna("").astype(str).str.lower().eq("ok")


def audit_candidate(
    backend: SourceBackend,
    path: str,
    expected_names: Mapping[str, set[str]],
    active_text: str,
) -> Candidate:
    candidate = Candidate(backend.host_label, path, None)
    reason = excluded_path(path)
    if reason:
        candidate.eligible = False
        candidate.rejection_reason = reason
        return candidate
    try:
        data, observation, stable = backend.stable_read(path)
        candidate.observation = observation
        candidate.unstable = not stable
        if not stable or data is None:
            candidate.rejection_reason = "unstable_source"
            candidate.active = source_path_active(path, active_text)
            return candidate
        candidate.data = data
        frame = parse_csv(data)
        candidate.frame = frame
    except Exception as exc:
        candidate.invalid = True
        candidate.eligible = False
        candidate.rejection_reason = f"read_error:{type(exc).__name__}:{exc}"
        return candidate

    column = dataset_column(frame)
    if not column:
        candidate.invalid = True
        candidate.eligible = False
        candidate.rejection_reason = "missing_dataset_identity"
        return candidate
    candidate.dataset_column = column
    names = frame[column].dropna().astype(str)
    candidate.result_rows = len(frame)
    candidate.unique_datasets = names.nunique()
    if names.duplicated().any():
        candidate.invalid = True
        candidate.eligible = False
        candidate.rejection_reason = "duplicate_dataset_identity"
    if names.eq("__WORKER_EXIT__").any():
        candidate.invalid = True
        candidate.eligible = False
        candidate.rejection_reason = "worker_exit_pseudorow"

    dataset = infer_dataset(path, frame)
    if dataset is None:
        present_names = set(names)
        overlaps = sorted(
            (
                (len(present_names & target_names), dataset_name)
                for dataset_name, target_names in expected_names.items()
            ),
            reverse=True,
        )
        if overlaps and overlaps[0][0] > 0 and (
            len(overlaps) == 1 or overlaps[0][0] > overlaps[1][0]
        ):
            dataset = overlaps[0][1]
    model = infer_model(path, frame)
    method = infer_method(path, frame)
    if dataset and model and method:
        candidate.cell = MatrixCell(model, dataset, method)
    else:
        candidate.eligible = False
        candidate.rejection_reason = candidate.rejection_reason or (
            f"unclassified:model={model},dataset={dataset},method={method}"
        )
        return candidate

    target = expected_names.get(dataset, set())
    present = set(names)
    mask = ok_mask(frame)
    if target:
        target_mask = names.isin(target)
        mask = mask & target_mask
        candidate.missing_names = tuple(sorted(target - present))
        candidate.unexpected_names = tuple(sorted(present - target))
        candidate.unique_datasets = len(target & present)
    candidate.status_ok = int(mask.sum())
    if "accuracy" in frame:
        values = pd.to_numeric(frame.loc[mask, "accuracy"], errors="coerce").dropna()
        if len(values):
            candidate.mean_accuracy = float(values.mean())
    lower = path.lower()
    candidate.selected_on_eval = "optuna" in lower or "best_para" in lower or "best_matrix" in lower
    recently_written_remote = (
        backend.host_label != "local"
        and candidate.observation is not None
        and time.time() - candidate.observation.mtime < 600
        and bool(candidate.missing_names)
    )
    candidate.active = source_path_active(path, active_text) or recently_written_remote
    return candidate


def source_path_active(path: str, active_text: str) -> bool:
    parent = str(PurePosixPath(path).parent)
    if parent in active_text:
        return True
    if "/results/" in parent:
        relative = "results/" + parent.split("/results/", 1)[1]
        return relative in active_text
    return False


def select_candidates(
    candidates: Sequence[Candidate],
) -> tuple[dict[MatrixCell, Candidate], list[Candidate]]:
    groups: dict[MatrixCell, list[Candidate]] = defaultdict(list)
    rejected: list[Candidate] = []
    for candidate in candidates:
        if candidate.cell is None or not candidate.eligible or candidate.invalid or candidate.unstable:
            rejected.append(candidate)
            continue
        groups[candidate.cell].append(candidate)
    selected: dict[MatrixCell, Candidate] = {}
    for cell, group in groups.items():
        seed_bundle = select_tabiclv2_seed_bundle(cell, group)
        if seed_bundle is not None:
            selected[cell] = seed_bundle
            selected_member_ids = {id(item) for item in seed_bundle.members}
            for item in group:
                if id(item) not in selected_member_ids:
                    item.rejection_reason = (
                        f"not_selected_by_policy:{seed_bundle.selection_policy}"
                    )
                    rejected.append(item)
            continue
        ordered = sorted(group, key=lambda item: (item.selection_key(), item.source_path), reverse=True)
        winner = ordered[0]
        selected[cell] = winner
        for item in ordered[1:]:
            item.rejection_reason = (
                "lower_selection_key:"
                f"coverage={item.unique_datasets},ok={item.status_ok},mean={item.mean_accuracy}"
            )
            rejected.append(item)
    return selected, rejected


TABICLV2_SEED_SWEEP_TAGS = (
    "/tabicl/v2/",
    "tabicl_v11_v220260702_151902/tabiclv2",
)
TABICLV2_SEED_SWEEP_RUN = "seed_sweep_20260702_151902"


def seed_from_path(path: str) -> int | None:
    match = re.search(r"/seed(\d+)/all_classification_results\.csv$", path)
    return int(match.group(1)) if match else None


def select_tabiclv2_seed_bundle(
    cell: MatrixCell,
    group: Sequence[Candidate],
) -> Candidate | None:
    if cell not in {
        MatrixCell("TabICLv2", "data184", "FT"),
        MatrixCell("TabICLv2", "data184", "F-aware FT"),
    }:
        return None
    method_dir = "ft" if cell.method == "FT" else "faware_c"
    eligible = [
        item
        for item in group
        if any(tag in item.source_path for tag in TABICLV2_SEED_SWEEP_TAGS)
        and (
            TABICLV2_SEED_SWEEP_RUN in item.source_path
            or "tabicl_v11_v220260702_151902" in item.source_path
        )
        and f"/{method_dir}/seed" in item.source_path
        and seed_from_path(item.source_path) is not None
        and item.unique_datasets == 184
        and item.status_ok == 184
        and not item.missing_names
        and not item.unexpected_names
    ]
    by_seed: dict[int, Candidate] = {}
    for item in eligible:
        seed = seed_from_path(item.source_path)
        assert seed is not None
        incumbent = by_seed.get(seed)
        if incumbent is None or (item.selection_key(), item.source_path) > (
            incumbent.selection_key(),
            incumbent.source_path,
        ):
            by_seed[seed] = item
    ordered = sorted(
        by_seed.values(),
        key=lambda item: (item.mean_accuracy, -(seed_from_path(item.source_path) or 0)),
        reverse=True,
    )
    if len(ordered) < 3:
        return None
    if cell.method == "F-aware FT":
        members = ordered[:3]
        policy = "top3_avg_accuracy_ok"
    else:
        start = (len(ordered) - 3) // 2
        members = ordered[start : start + 3]
        policy = "middle3_avg_accuracy_ok"
    return aggregate_seed_candidates(cell, members, policy)


def aggregate_seed_candidates(
    cell: MatrixCell,
    members: Sequence[Candidate],
    policy: str,
) -> Candidate:
    if not members or any(item.frame is None for item in members):
        raise ValueError("seed bundle requires parsed member frames")
    seeds = tuple(seed_from_path(item.source_path) for item in members)
    if any(seed is None for seed in seeds):
        raise ValueError("seed bundle member lacks seed path")
    frames: list[pd.DataFrame] = []
    for item in members:
        assert item.frame is not None
        indexed = item.frame.copy()
        indexed[item.dataset_column] = indexed[item.dataset_column].astype(str)
        indexed = indexed.set_index(item.dataset_column, drop=False)
        frames.append(indexed)
    shared_names = sorted(set.intersection(*(set(frame.index) for frame in frames)))
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    aggregated_rows: list[dict[str, object]] = []
    for name in shared_names:
        row: dict[str, object] = {}
        for column in columns:
            values = [
                frame.at[name, column]
                for frame in frames
                if column in frame.columns and pd.notna(frame.at[name, column])
            ]
            if not values:
                row[column] = ""
            elif column == members[0].dataset_column:
                row[column] = name
            elif column == "status":
                row[column] = "ok" if all(str(value).lower() == "ok" for value in values) else "fail"
            elif column.endswith("_applied"):
                row[column] = all(truthy(value) for value in values)
            elif column.endswith("_fallback"):
                row[column] = any(truthy(value) for value in values)
            elif is_metric_column(column):
                numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
                row[column] = float(numeric.mean()) if len(numeric) else values[0]
            else:
                row[column] = values[0]
        row["selected_seed_count"] = len(members)
        row["selected_seeds"] = ",".join(str(seed) for seed in seeds)
        aggregated_rows.append(row)
    frame = pd.DataFrame(aggregated_rows)
    data = frame.to_csv(index=False).encode("utf-8")
    observation = SourceObservation(len(data), max(item.observation.mtime for item in members if item.observation), sha256_bytes(data))
    mask = ok_mask(frame)
    accuracy = pd.to_numeric(frame.loc[mask, "accuracy"], errors="coerce").dropna()
    paths = ";".join(item.source_path for item in members)
    return Candidate(
        source_host="local",
        source_path=paths,
        cell=cell,
        data=data,
        frame=frame,
        observation=observation,
        result_rows=len(frame),
        unique_datasets=frame[members[0].dataset_column].nunique(),
        status_ok=int(mask.sum()),
        mean_accuracy=float(accuracy.mean()) if len(accuracy) else float("-inf"),
        selected_on_eval=True,
        members=tuple(members),
        selection_policy=policy,
        selected_seeds=tuple(int(seed) for seed in seeds if seed is not None),
    )


def count_failures(frame: pd.DataFrame) -> int:
    return int((~ok_mask(frame)).sum())


def count_fallback(frame: pd.DataFrame) -> int:
    masks: list[pd.Series] = []
    for column in frame.columns:
        lower = column.lower()
        if lower.endswith("fallback") or lower.endswith("_fallback"):
            masks.append(frame[column].map(truthy))
        elif "fallback_reason" in lower:
            masks.append(frame[column].fillna("").astype(str).str.strip().ne(""))
    if not masks:
        return 0
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined |= mask
    return int(combined.sum())


def adaptation_applied(frame: pd.DataFrame) -> int:
    for column in ("ttt_applied", "ft_applied", "peft_applied"):
        if column in frame:
            return int(frame[column].map(truthy).sum())
    return 0


def classify_status(
    candidate: Candidate | None,
    expected_count: int,
    method: str,
) -> tuple[str, bool, int, int, int]:
    if candidate is None:
        return "missing", False, 0, 0, 0
    if candidate.invalid:
        return "invalid", False, candidate.status_ok, 0, 0
    frame = candidate.frame
    if frame is None:
        return ("running" if candidate.active or candidate.unstable else "invalid"), False, 0, 0, 0
    artifact_complete = (
        candidate.unique_datasets == expected_count
        and not candidate.missing_names
        and not candidate.unexpected_names
    )
    failed = count_failures(frame)
    fallback = count_fallback(frame)
    applied = adaptation_applied(frame)
    if candidate.active and not artifact_complete:
        status = "running"
    elif not artifact_complete:
        status = "partial"
    elif (
        failed
        or fallback
        or candidate.status_ok != expected_count
        or (method in ADAPTIVE_METHODS and applied != expected_count)
    ):
        status = "complete_with_failures"
    else:
        status = "complete"
    return status, artifact_complete, candidate.status_ok, applied, fallback


NON_METRIC_EXACT = {
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_classes",
    "n_train_a",
    "n_train_b",
    "n_holdout_c",
    "n_test_d",
}
NON_METRIC_PATTERNS = (
    r"(^|_)seed($|_)",
    r"random_state",
    r"(^|_)(lr|learning_rate)($|_)",
    r"(^|_)(epoch|epochs|steps|batch|estimators|patience|rank)($|_)",
    r"(query|reserve|context).*ratio",
    r"(query|context).*size",
    r"chunks_per_epoch",
    r"best_epoch",
)
IDENTIFIER_PATTERNS = (
    "dataset",
    "model",
    "method",
    "path",
    "dir",
    "worker",
    "trial",
    "run",
    "status",
    "error",
    "reason",
    "strategy",
    "source",
    "metric",
    "mode",
    "targets",
)


def canonical_metric_name(name: str) -> str:
    aliases = {"acc": "accuracy", "balanced_acc": "balanced_accuracy", "auc": "roc_auc"}
    return aliases.get(name.lower(), name.lower())


def is_metric_column(name: str) -> bool:
    lower = name.lower()
    if lower in NON_METRIC_EXACT:
        return False
    if any(token in lower for token in IDENTIFIER_PATTERNS):
        return False
    if any(re.search(pattern, lower) for pattern in NON_METRIC_PATTERNS):
        return False
    return True


def metric_rows(selected: Mapping[MatrixCell, Candidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in expected_cells():
        candidate = selected.get(cell)
        if not candidate or candidate.frame is None:
            continue
        frame = candidate.frame.loc[ok_mask(candidate.frame)].copy()
        metrics: dict[str, pd.Series] = {}
        for column in frame.columns:
            if not is_metric_column(column):
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            if numeric.empty:
                continue
            canonical = canonical_metric_name(column)
            if canonical in metrics:
                continue
            metrics[canonical] = numeric.astype(float)
        coverage = {
            "coverage_result_rows": float(candidate.result_rows),
            "coverage_status_ok": float(candidate.status_ok),
            "coverage_failed": float(count_failures(candidate.frame)),
            "coverage_fallback": float(count_fallback(candidate.frame)),
            "coverage_adaptation_applied": float(adaptation_applied(candidate.frame)),
        }
        for name, value in coverage.items():
            metrics[name] = pd.Series([value], dtype=float)
        for name, values in sorted(metrics.items()):
            rows.append(
                {
                    "dataset": cell.dataset,
                    "model": cell.model,
                    "method": cell.method,
                    "metric": name,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "source_path": candidate.source_path,
                }
            )
    return rows


def metric_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = ["# Method Metrics", "", "Direct comparisons are intentionally excluded from this display.", ""]
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["model"]), str(row["method"]))].append(row)
    for dataset in DATASETS:
        lines.extend([f"## {dataset}", ""])
        for model in MODELS:
            lines.extend([f"### {model}", ""])
            for method in METHODS:
                group = grouped.get((dataset, model, method))
                if not group:
                    lines.extend([f"#### {method}", "", "—", ""])
                    continue
                lines.extend(
                    [
                        f"#### {method}",
                        "",
                        "| Metric | N | Mean | Std | Min | Max |",
                        "|---|---:|---:|---:|---:|---:|",
                    ]
                )
                for row in sorted(group, key=lambda item: str(item["metric"])):
                    lines.append(
                        f"| {row['metric']} | {row['n']} | {float(row['mean']):.10g} | "
                        f"{float(row['std']):.10g} | {float(row['min']):.10g} | "
                        f"{float(row['max']):.10g} |"
                    )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def comparison_rows(selected: Mapping[MatrixCell, Candidate]) -> list[dict[str, object]]:
    pairs = (
        ("FT", "Infer"),
        ("F-aware FT", "FT"),
        ("LoRA", "Infer"),
        ("LoRA", "FT"),
        ("ln_head_embedding", "Infer"),
        ("ln_head_embedding", "FT"),
        ("Last-block", "Infer"),
        ("Last-block", "FT"),
        ("LoCalPFN", "FT"),
        ("MICP", "FT"),
    )
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in DATASETS:
            for method, reference in pairs:
                left = selected.get(MatrixCell(model, dataset, method))
                right = selected.get(MatrixCell(model, dataset, reference))
                if not left or not right or left.frame is None or right.frame is None:
                    continue
                left_frame = left.frame.loc[ok_mask(left.frame)].copy()
                right_frame = right.frame.loc[ok_mask(right.frame)].copy()
                left_col, right_col = left.dataset_column, right.dataset_column
                merged = left_frame.merge(
                    right_frame,
                    left_on=left_col,
                    right_on=right_col,
                    suffixes=("_method", "_reference"),
                    how="inner",
                )
                for metric in PERFORMANCE_METRICS:
                    a, b = f"{metric}_method", f"{metric}_reference"
                    if a not in merged or b not in merged:
                        continue
                    method_values = pd.to_numeric(merged[a], errors="coerce")
                    reference_values = pd.to_numeric(merged[b], errors="coerce")
                    valid = method_values.notna() & reference_values.notna()
                    if not valid.any():
                        continue
                    method_values = method_values[valid]
                    reference_values = reference_values[valid]
                    raw_delta = method_values - reference_values
                    oriented = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "reference_method": reference,
                            "metric": metric,
                            "shared_status_ok": int(valid.sum()),
                            "method_mean": float(method_values.mean()),
                            "reference_mean": float(reference_values.mean()),
                            "delta": float(raw_delta.mean()),
                            "wins": int((oriented > 1e-12).sum()),
                            "losses": int((oriented < -1e-12).sum()),
                            "ties": int((oriented.abs() <= 1e-12).sum()),
                        }
                    )
    return rows


def ranking_rows(selected: Mapping[MatrixCell, Candidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in DATASETS:
            available = {
                method: selected.get(MatrixCell(model, dataset, method)) for method in METHODS
            }
            available = {
                method: candidate
                for method, candidate in available.items()
                if candidate is not None and candidate.frame is not None
            }
            if len(available) < 2:
                continue
            for metric in PERFORMANCE_METRICS:
                series: dict[str, pd.Series] = {}
                for method, candidate in available.items():
                    assert candidate.frame is not None
                    if metric not in candidate.frame:
                        continue
                    frame = candidate.frame.loc[ok_mask(candidate.frame)]
                    values = pd.to_numeric(frame[metric], errors="coerce")
                    indexed = pd.Series(values.values, index=frame[candidate.dataset_column].astype(str))
                    series[method] = indexed.dropna()
                if len(series) < 2:
                    continue
                shared = set.intersection(*(set(values.index) for values in series.values()))
                if not shared:
                    continue
                matrix = pd.DataFrame(
                    {method: values.loc[sorted(shared)] for method, values in series.items()}
                )
                ranks = matrix.rank(
                    axis=1,
                    ascending=metric in LOWER_IS_BETTER,
                    method="average",
                )
                for method in sorted(ranks):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "metric": metric,
                            "method": method,
                            "shared_status_ok": len(shared),
                            "average_rank": float(ranks[method].mean()),
                        }
                    )
    return rows


INVENTORY_FIELDS = (
    "model",
    "dataset",
    "method",
    "overall_status",
    "artifact_complete",
    "source_host",
    "source_path",
    "selected_run",
    "selection_policy",
    "selected_seeds",
    "selected_seed_count",
    "selected_on_eval",
    "expected_datasets",
    "result_rows",
    "unique_datasets",
    "status_ok",
    "adaptation_expected",
    "adaptation_applied",
    "fallback_count",
    "failed_count",
    "selection_note",
    "snapshot_timestamp",
)
PROVENANCE_FIELDS = (
    "model",
    "dataset",
    "method",
    "selected",
    "source_host",
    "source_path",
    "source_mtime",
    "size_bytes",
    "sha256",
    "rejection_reason",
    "snapshot_timestamp",
)


def build_inventory(
    selected: Mapping[MatrixCell, Candidate],
    expected_names: Mapping[str, set[str]],
    snapshot_time: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in expected_cells():
        candidate = selected.get(cell)
        expected_count = len(expected_names[cell.dataset])
        status, complete, ok, applied, fallback = classify_status(candidate, expected_count, cell.method)
        frame = candidate.frame if candidate else None
        notes: list[str] = []
        if candidate:
            if candidate.missing_names:
                notes.append(f"missing={len(candidate.missing_names)}")
            if candidate.unexpected_names:
                notes.append(f"unexpected={len(candidate.unexpected_names)}")
            if candidate.active:
                notes.append("active_remote_evidence")
            if candidate.selected_on_eval:
                notes.append("selected_on_eval")
            if candidate.selection_policy:
                notes.append(candidate.selection_policy)
        rows.append(
            {
                "model": cell.model,
                "dataset": cell.dataset,
                "method": cell.method,
                "overall_status": status,
                "artifact_complete": complete,
                "source_host": candidate.source_host if candidate else "",
                "source_path": candidate.source_path if candidate else "",
                "selected_run": (
                    "+".join(f"seed{seed}" for seed in candidate.selected_seeds)
                    if candidate and candidate.selected_seeds
                    else PurePosixPath(candidate.source_path).parent.name if candidate else ""
                ),
                "selection_policy": candidate.selection_policy if candidate else "",
                "selected_seeds": (
                    ",".join(str(seed) for seed in candidate.selected_seeds) if candidate else ""
                ),
                "selected_seed_count": len(candidate.selected_seeds) if candidate else 0,
                "selected_on_eval": candidate.selected_on_eval if candidate else False,
                "expected_datasets": expected_count,
                "result_rows": candidate.result_rows if candidate else 0,
                "unique_datasets": candidate.unique_datasets if candidate else 0,
                "status_ok": ok,
                "adaptation_expected": expected_count if cell.method in ADAPTIVE_METHODS else 0,
                "adaptation_applied": applied,
                "fallback_count": fallback,
                "failed_count": count_failures(frame) if frame is not None else 0,
                "selection_note": ";".join(notes),
                "snapshot_timestamp": snapshot_time,
            }
        )
    return rows


def build_provenance(
    selected: Mapping[MatrixCell, Candidate],
    rejected: Sequence[Candidate],
    snapshot_time: str,
) -> list[dict[str, object]]:
    selected_ids = {
        id(member)
        for candidate in selected.values()
        for member in (candidate.members or (candidate,))
    }
    selected_candidates = [
        member
        for candidate in selected.values()
        for member in (candidate.members or (candidate,))
    ]
    all_candidates = selected_candidates + list(rejected)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(all_candidates, key=lambda item: (item.source_host, item.source_path)):
        key = (candidate.source_host, candidate.source_path)
        if key in seen:
            continue
        seen.add(key)
        observation = candidate.observation
        rows.append(
            {
                "model": candidate.cell.model if candidate.cell else "",
                "dataset": candidate.cell.dataset if candidate.cell else "",
                "method": candidate.cell.method if candidate.cell else "",
                "selected": id(candidate) in selected_ids,
                "source_host": candidate.source_host,
                "source_path": candidate.source_path,
                "source_mtime": observation.mtime if observation else "",
                "size_bytes": observation.size if observation else "",
                "sha256": observation.sha256 if observation else "",
                "rejection_reason": candidate.rejection_reason,
                "snapshot_timestamp": snapshot_time,
            }
        )
    return rows


def completeness_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Completeness Matrix",
        "",
        "| Dataset | Model | Method | Status | Coverage | status=ok | Adaptation | Note |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['method']} | "
            f"{row['overall_status']} | {row['unique_datasets']}/{row['expected_datasets']} | "
            f"{row['status_ok']} | {row['adaptation_applied']}/{row['adaptation_expected']} | "
            f"{row['selection_note'] or '—'} |"
        )
    for status in ("missing", "running", "partial", "complete_with_failures", "invalid"):
        subset = [row for row in rows if row["overall_status"] == status]
        lines.extend([f"", f"## {status}", ""])
        if not subset:
            lines.append("—")
            continue
        for row in subset:
            path = row["source_path"] or "no qualifying artifact"
            lines.append(
                f"- `{row['dataset']} / {row['model']} / {row['method']}`: "
                f"{row['unique_datasets']}/{row['expected_datasets']}, `{path}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def readme_text(inventory: Sequence[Mapping[str, object]], snapshot_time: str) -> str:
    counts = Counter(str(row["overall_status"]) for row in inventory)
    lines = [
        "# Paper Result Archive",
        "",
        f"Snapshot: `{snapshot_time}`",
        "",
        "Sources: local `results/` and read-only `jiqun:~/zh/tabiclv2/results/`.",
        "",
        "The inventory covers 72 cells: 3 models × 3 datasets × 8 methods.",
        "`ln_head_embedding` is kept under that exact name. FT and F-aware FT use",
        "formal results after coverage and status=ok precedence. For TabICLv2/data184,",
        "F-aware FT uses the top three seeds by status=ok mean accuracy, while FT uses",
        "the middle three seeds; their per-dataset numeric metrics are seed-averaged.",
        "",
        "Main Markdown tables show each method's own metrics. Direct deltas, W/L/T,",
        "and rankings are reference-only CSV artifacts.",
        "",
        "## Status Counts",
        "",
    ]
    for status in sorted(counts):
        lines.append(f"- `{status}`: {counts[status]}")
    flagged = [row for row in inventory if truthy(row["selected_on_eval"])]
    lines.extend(["", "## Selection-on-evaluation Warning", ""])
    if flagged:
        lines.append(
            "The following cells use a best/Optuna result selected on the reported "
            "evaluation tasks and must not be described as unbiased held-out estimates:"
        )
        for row in flagged:
            lines.append(f"- `{row['dataset']} / {row['model']} / {row['method']}`")
    else:
        lines.append("No selected cell was identified as selected on the reporting set.")
    lines.extend(
        [
            "",
            "## Completeness Semantics",
            "",
            "- `complete`: full target membership, all status=ok, and full required adaptation.",
            "- `complete_with_failures`: full membership with failures, fallback, or unapplied adaptation.",
            "- `partial`: stable formal artifact with missing target rows.",
            "- `running`: active or unstable partial remote artifact.",
            "- `missing`: no qualifying artifact.",
            "- `invalid`: malformed or duplicate-identity artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_inventory(rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) != 72:
        raise RuntimeError(f"inventory must contain 72 rows, found {len(rows)}")
    keys = {(row["model"], row["dataset"], row["method"]) for row in rows}
    if len(keys) != 72:
        raise RuntimeError("inventory contains duplicate cells")


def copy_selected_raw(
    stage: Path,
    selected: Mapping[MatrixCell, Candidate],
    backends: Mapping[str, SourceBackend],
) -> None:
    copied_hashes: dict[str, Path] = {}
    for cell, candidate in selected.items():
        if candidate.data is None or candidate.observation is None:
            continue
        target_dir = stage / "raw" / slug(cell.dataset) / slug(cell.model) / slug(cell.method)
        target = target_dir / "all_classification_results.csv"
        digest = candidate.observation.sha256
        if digest in copied_hashes:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copied_hashes[digest], target)
        else:
            atomic_write(target, candidate.data)
            copied_hashes[digest] = target
        source_candidates = candidate.members or (candidate,)
        for member in source_candidates:
            if candidate.members and member.data is not None:
                seed = seed_from_path(member.source_path)
                seed_target = target_dir / f"seed{seed}" / "all_classification_results.csv"
                atomic_write(seed_target, member.data)
            backend = backends[member.source_host]
            for support_path in backend.optional_support(member.source_path):
                try:
                    support_data, support_observation, stable = backend.stable_read(support_path)
                except Exception:
                    continue
                if not stable or support_data is None:
                    continue
                support_name = PurePosixPath(support_path).name
                support_dir = target_dir / f"seed{seed_from_path(member.source_path)}" if candidate.members else target_dir
                support_target = support_dir / support_name
                if support_target.exists() and sha256_path(support_target) != support_observation.sha256:
                    support_target = support_dir / (
                        f"{PurePosixPath(support_name).stem}-{support_observation.sha256[:8]}"
                        f"{PurePosixPath(support_name).suffix}"
                    )
                atomic_write(support_target, support_data)


def replace_managed_output(stage: Path, out: Path) -> None:
    files = sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file())
    atomic_write_text(
        stage / MANAGED_MARKER,
        json.dumps({"managed_files": files + [MANAGED_MARKER]}, indent=2, ensure_ascii=False) + "\n",
    )
    if not out.exists():
        os.replace(stage, out)
        return
    marker = out / MANAGED_MARKER
    if not marker.exists():
        if any(out.iterdir()):
            raise RuntimeError(f"refusing to replace unmanaged non-empty output: {out}")
        out.rmdir()
        os.replace(stage, out)
        return
    managed = set(json.loads(marker.read_text())["managed_files"])
    existing = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()}
    unknown = existing - managed
    if unknown:
        raise RuntimeError(f"refusing to remove unmanaged files in {out}: {sorted(unknown)}")
    backup = out.with_name(f".{out.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(out, backup)
    try:
        os.replace(stage, out)
    except Exception:
        os.replace(backup, out)
        raise
    shutil.rmtree(backup)


def load_expected_names(repo_root: Path) -> dict[str, set[str]]:
    data184_root = repo_root / "data184"
    data184 = {path.name for path in data184_root.iterdir() if path.is_dir()}
    openml_path = repo_root / "results/dataset_views/openml_cc18_max10/dataset_view_manifest.json"
    openml_manifest = json.loads(openml_path.read_text())
    openml = {str(item["dataset_name"]) for item in openml_manifest["included"]}
    graph_path = repo_root / "results/合成数据实验/data_manifest/prior_manifest.csv"
    with graph_path.open(newline="", encoding="utf-8") as handle:
        graph = {str(row["dataset_name"]) for row in csv.DictReader(handle)}
    expected = {"data184": data184, "OpenML-CC18": openml, "Graph-SCM": graph}
    required = {"data184": 184, "OpenML-CC18": 67, "Graph-SCM": 512}
    mismatches = {name: (len(expected[name]), count) for name, count in required.items() if len(expected[name]) != count}
    if mismatches:
        raise RuntimeError(f"manifest count mismatch: {mismatches}")
    return expected


def write_archive(
    stage: Path,
    script_path: Path,
    inventory: list[dict[str, object]],
    provenance: list[dict[str, object]],
    metrics: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    rankings: list[dict[str, object]],
    selected: Mapping[MatrixCell, Candidate],
    backends: Mapping[str, SourceBackend],
    snapshot_time: str,
) -> None:
    copy_selected_raw(stage, selected, backends)
    atomic_write(stage / "inventory.csv", csv_bytes(inventory, INVENTORY_FIELDS))
    atomic_write(stage / "provenance.csv", csv_bytes(provenance, PROVENANCE_FIELDS))
    metric_fields = ("dataset", "model", "method", "metric", "n", "mean", "std", "min", "max", "source_path")
    comparison_fields = (
        "dataset",
        "model",
        "method",
        "reference_method",
        "metric",
        "shared_status_ok",
        "method_mean",
        "reference_mean",
        "delta",
        "wins",
        "losses",
        "ties",
    )
    ranking_fields = ("dataset", "model", "metric", "method", "shared_status_ok", "average_rank")
    atomic_write(stage / "tables/method_metrics.csv", csv_bytes(metrics, metric_fields))
    atomic_write_text(stage / "tables/method_metrics.md", metric_markdown(metrics))
    atomic_write(stage / "tables/completeness_matrix.csv", csv_bytes(inventory, INVENTORY_FIELDS))
    atomic_write_text(stage / "tables/completeness_matrix.md", completeness_markdown(inventory))
    atomic_write(stage / "tables/reference_comparisons.csv", csv_bytes(comparisons, comparison_fields))
    atomic_write(stage / "tables/reference_rankings.csv", csv_bytes(rankings, ranking_fields))
    atomic_write_text(stage / "README.md", readme_text(inventory, snapshot_time))
    atomic_write(stage / "scripts/build_result_archive.py", script_path.read_bytes())


def summarize_dry_run(
    inventory: Sequence[Mapping[str, object]],
    selected: Mapping[MatrixCell, Candidate],
) -> None:
    counts = Counter(str(row["overall_status"]) for row in inventory)
    print(json.dumps({"cells": len(inventory), "status_counts": dict(sorted(counts.items()))}, indent=2))
    for cell in expected_cells():
        candidate = selected.get(cell)
        if candidate:
            print(
                f"{cell.dataset}|{cell.model}|{cell.method}|"
                f"{candidate.unique_datasets}|{candidate.status_ok}|{candidate.source_host}:{candidate.source_path}"
            )


def build(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    expected_names = load_expected_names(repo_root)
    snapshot_time = args.snapshot_time or utc_now()
    local_backend = LocalSourceBackend(Path(args.local_results))
    backends: list[SourceBackend] = [local_backend]
    if not args.local_only:
        backends.append(SSHSourceBackend(args.remote_host, args.remote_results))
    backend_map = {backend.host_label: backend for backend in backends}

    candidates: list[Candidate] = []
    for backend in backends:
        active_text = backend.active_text()
        paths = backend.list_result_csvs()
        paths = [
            path
            for path in paths
            if backend.host_label == "local" or remote_candidate_relevant(path)
        ]
        audit_paths = [path for path in paths if not excluded_path(path)]
        if isinstance(backend, SSHSourceBackend):
            backend.prefetch(audit_paths)
        for path in paths:
            candidates.append(audit_candidate(backend, path, expected_names, active_text))
    selected, rejected = select_candidates(candidates)
    for backend in backends:
        if not isinstance(backend, SSHSourceBackend):
            continue
        support_paths: list[str] = []
        for candidate in selected.values():
            if candidate.source_host != backend.host_label:
                continue
            parent = str(PurePosixPath(candidate.source_path).parent)
            support_paths.extend(str(PurePosixPath(parent) / name) for name in SUPPORT_FILENAMES)
        backend.prefetch(support_paths)

    inventory = build_inventory(selected, expected_names, snapshot_time)
    validate_inventory(inventory)
    provenance = build_provenance(selected, rejected, snapshot_time)
    metrics = metric_rows(selected)
    comparisons = comparison_rows(selected)
    rankings = ranking_rows(selected)
    summarize_dry_run(inventory, selected)
    if args.dry_run:
        return 0

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage.", dir=out.parent))
    try:
        write_archive(
            stage,
            script_path,
            inventory,
            provenance,
            metrics,
            comparisons,
            rankings,
            selected,
            backend_map,
            snapshot_time,
        )
        replace_managed_output(stage, out)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", default="results")
    parser.add_argument("--remote-host", default="jiqun")
    parser.add_argument(
        "--remote-results",
        default="/home/crc00006699/zh/tabiclv2/results",
    )
    parser.add_argument("--out", default="result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--snapshot-time")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return build(parse_args(argv))
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
