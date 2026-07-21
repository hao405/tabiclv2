#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESET="${1:-${PRESET:-smoke-then-full}}"
case "${PRESET}" in
  smoke|full|smoke-then-full) ;;
  *)
    echo "usage: $0 [smoke|full|smoke-then-full]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
PEFT_RUNNER="${PEFT_RUNNER:-PEFT_Tabicl/1C_Chunk_PEFT.py}"
VIEW_VALIDATOR="${VIEW_VALIDATOR:-scripts/validate_openmlcc18_peft_view.py}"
MATRIX_AUDITOR="${MATRIX_AUDITOR:-scripts/summarize_openmlcc18_peft_matrix.py}"
DATA_ROOT="${DATA_ROOT:-results/dataset_views/openml_cc18_max10}"
SMOKE_DATA_ROOT="${SMOKE_DATA_ROOT:-results/dataset_views/openml_cc18_max10_peft_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/PEFT}"
LOG_ROOT="${LOG_ROOT:-logs/openmlcc18_peft_matrix}"
WORKERS="${WORKERS:-2}"
GPU_GROUPS="${GPU_GROUPS:-1;2}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

if [[ "${WORKERS}" != "2" ]]; then
  echo "OpenML PEFT matrix requires WORKERS=2 (got ${WORKERS})" >&2
  exit 2
fi
if [[ ! "${GPU_GROUPS}" =~ ^[0-9]+(,[0-9]+)*\;[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_GROUPS must define exactly two non-empty numeric groups (got ${GPU_GROUPS})" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-openmlcc18_peft_2x3_smoke_${STAMP}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-openmlcc18_peft_2x3_full_${STAMP}}"
if [[ "${SMOKE_RUN_NAME}" == "${FULL_RUN_NAME}" ]]; then
  echo "SMOKE_RUN_NAME and FULL_RUN_NAME must be different" >&2
  exit 2
fi

SMOKE_DATASETS=(
  OpenML-ID-1063
  OpenML-ID-23381
  OpenML-ID-188
)

mkdir -p "${LOG_ROOT}"

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  set +e
  "$@" > >(tee -a "${log_file}") 2>&1
  local exit_code=$?
  set -e
  return "${exit_code}"
}

validate_full_view() {
  local command=(
    "${PYTHON_BIN}" -u "${VIEW_VALIDATOR}"
    --data-root "${DATA_ROOT}"
    --expected-datasets 67
  )
  print_command "${command[@]}"
  run_logged "${LOG_ROOT}/view_validation.log" "${command[@]}"
}

materialize_smoke_view() {
  "${PYTHON_BIN}" - "${DATA_ROOT}" "${SMOKE_DATA_ROOT}" "${SMOKE_DATASETS[@]}" <<'PY'
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
names = sys.argv[3:]
manifest_path = source / "dataset_view_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
records = {
    str(item["dataset_name"]): item
    for item in manifest.get("included", [])
}
missing = [name for name in names if name not in records]
if missing:
    raise SystemExit(f"smoke datasets absent from full manifest: {missing}")

def valid_existing() -> bool:
    smoke_manifest_path = destination / "dataset_view_manifest.json"
    if not smoke_manifest_path.is_file():
        return False
    try:
        smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    actual_names = [
        str(item.get("dataset_name"))
        for item in smoke_manifest.get("included", [])
    ]
    if actual_names != names or smoke_manifest.get("included_count") != len(names):
        return False
    return all(
        (destination / name).is_symlink()
        and (destination / name).resolve() == (source / name).resolve()
        and (destination / name).resolve().is_dir()
        for name in names
    )

if destination.exists():
    if not valid_existing():
        raise SystemExit(
            f"existing smoke view is not the expected generated view: {destination}"
        )
    print(json.dumps({"smoke_view": str(destination), "status": "reused"}))
    raise SystemExit(0)

destination.parent.mkdir(parents=True, exist_ok=True)
temporary = Path(
    tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
)
try:
    for name in names:
        target = (source / name).resolve(strict=True)
        if not target.is_dir():
            raise RuntimeError(f"smoke dataset target is not a directory: {target}")
        (temporary / name).symlink_to(target, target_is_directory=True)
    smoke_manifest = dict(manifest)
    smoke_manifest.update(
        {
            "source_view_root": str(source),
            "effective_root": str(destination),
            "requested_dataset_names": names,
            "source_count": len(names),
            "included_count": len(names),
            "excluded_count": 0,
            "included": [records[name] for name in names],
            "excluded": [],
        }
    )
    (temporary / "dataset_view_manifest.json").write_text(
        json.dumps(smoke_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps({"smoke_view": str(destination), "status": "created"}))
finally:
    if temporary.exists():
        shutil.rmtree(temporary)
PY
}

run_preset() {
  local mode="$1"
  local data_root expected run_name epochs patience log_file view_manifest
  local use_smoke_settings=0

  if [[ "${mode}" == "smoke" ]]; then
    materialize_smoke_view
    data_root="${SMOKE_DATA_ROOT}"
    expected=3
    run_name="${SMOKE_RUN_NAME}"
    epochs=1
    patience=1
    use_smoke_settings=1
  else
    data_root="${DATA_ROOT}"
    expected=67
    run_name="${FULL_RUN_NAME}"
    epochs=30
    patience=8
  fi

  view_manifest="${data_root}/dataset_view_manifest.json"
  log_file="${LOG_ROOT}/${run_name}.log"
  local runner_command=(
    "${PYTHON_BIN}" -u "${PEFT_RUNNER}"
    --model-family all
    --ttt-peft-method all
    --data-root "${data_root}"
    --output-root "${OUTPUT_ROOT}"
    --run-name "${run_name}"
    --workers "${WORKERS}"
    --gpu-groups "${GPU_GROUPS}"
    --random-state "${SEED}"
    --ttt-epochs "${epochs}"
    --ttt-lr 1e-5
    --ttt-weight-decay 0.01
    --ttt-query-ratio 0.2
    --ttt-patience "${patience}"
    --ttt-lora-rank 4
    --ttt-lora-alpha 8
    --ttt-lora-dropout 0
    --ttt-last-n-icl-blocks 1
  )
  if [[ "${use_smoke_settings}" == "1" ]]; then
    runner_command+=(
      --n-estimators 2
      --matrix-fail-fast
      --require-all-peft-success
    )
  fi
  if [[ "${RESUME}" == "1" ]]; then
    runner_command+=(--resume)
  fi

  local matrix_root="${OUTPUT_ROOT}/_matrix/${run_name}"
  local audit_command=(
    "${PYTHON_BIN}" -u "${MATRIX_AUDITOR}"
    --matrix-root "${matrix_root}"
    --view-manifest "${view_manifest}"
    --expected-datasets "${expected}"
  )
  if [[ "${use_smoke_settings}" == "1" ]]; then
    audit_command+=(--require-all-success)
  fi

  echo "preset: ${mode}"
  echo "run_name: ${run_name}"
  echo "matrix_root: ${matrix_root}"
  echo "log_file: ${log_file}"
  print_command "${runner_command[@]}"
  print_command "${audit_command[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local runner_exit=0
  local audit_exit=0
  run_logged "${log_file}" "${runner_command[@]}" || runner_exit=$?
  run_logged "${log_file}" "${audit_command[@]}" || audit_exit=$?
  echo "runner_exit_code: ${runner_exit}" | tee -a "${log_file}"
  echo "audit_exit_code: ${audit_exit}" | tee -a "${log_file}"
  if [[ "${runner_exit}" -ne 0 ]]; then
    return "${runner_exit}"
  fi
  return "${audit_exit}"
}

echo "wrapper_pid: $$"
echo "preset: ${PRESET}"
echo "data_root: ${DATA_ROOT}"
echo "gpu_groups: ${GPU_GROUPS}"
echo "dry_run: ${DRY_RUN}"
validate_full_view

case "${PRESET}" in
  smoke)
    run_preset smoke
    ;;
  full)
    run_preset full
    ;;
  smoke-then-full)
    run_preset smoke
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "smoke_gate: dry-run (not evaluated)"
    else
      echo "smoke_gate: passed"
    fi
    run_preset full
    ;;
esac
