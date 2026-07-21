#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESET="${1:-${PRESET:-full}}"
if [[ "${PRESET}" != "smoke" && "${PRESET}" != "full" ]]; then
  echo "usage: $0 [smoke|full]" >&2
  exit 2
fi

SEED="${SEED:-42}"
WORKERS="${WORKERS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/managed_experiments}"
LOG_DIR="${LOG_DIR:-logs/openmlcc18_tfm_matrix}"
DRY_RUN="${DRY_RUN:-0}"
DEVICES="${DEVICES:-auto}"
GPU_CANDIDATES="${GPU_CANDIDATES:-1,2}"
GPU_MIN_FREE_RATIO="${GPU_MIN_FREE_RATIO:-0.90}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

if [[ "${PRESET}" == "smoke" ]]; then
  EXPECTED_DATASETS=3
  RUN_ID="${RUN_ID:-openmlcc18_2x3_smoke_$(date +%Y%m%d_%H%M%S)}"
  TTT_EPOCHS="${TTT_EPOCHS:-1}"
  TTT_PATIENCE="${TTT_PATIENCE:-1}"
  N_ESTIMATORS="${N_ESTIMATORS:-2}"
  DATASET_NAMES=(OpenML-ID-1063 OpenML-ID-23381 OpenML-ID-188)
else
  EXPECTED_DATASETS=67
  RUN_ID="${RUN_ID:-openmlcc18_2x3_full_$(date +%Y%m%d_%H%M%S)}"
  TTT_EPOCHS="${TTT_EPOCHS:-30}"
  TTT_PATIENCE="${TTT_PATIENCE:-8}"
  N_ESTIMATORS="${N_ESTIMATORS:-}"
  DATASET_NAMES=()
fi

mkdir -p "${LOG_DIR}"
MATRIX_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

select_free_gpu() {
  GPU_CANDIDATES="${GPU_CANDIDATES}" GPU_MIN_FREE_RATIO="${GPU_MIN_FREE_RATIO}" python - <<'PY'
import os
import torch

candidates = [int(value) for value in os.environ["GPU_CANDIDATES"].split(",") if value.strip()]
threshold = float(os.environ["GPU_MIN_FREE_RATIO"])
available = []
for index in candidates:
    if index < 0 or index >= torch.cuda.device_count():
        continue
    free, total = torch.cuda.mem_get_info(index)
    ratio = free / total
    if ratio >= threshold:
        available.append((free, ratio, index, total))
if available:
    free, ratio, index, total = max(available)
    print(index)
PY
}

if [[ "${DRY_RUN}" == "1" && "${DEVICES}" == "auto" ]]; then
  DEVICES=0
elif [[ "${DEVICES}" == "auto" ]]; then
  while true; do
    DEVICES="$(select_free_gpu)"
    if [[ -n "${DEVICES}" ]]; then
      break
    fi
    echo "waiting_for_free_gpu: candidates=${GPU_CANDIDATES} min_free_ratio=${GPU_MIN_FREE_RATIO}"
    sleep "${GPU_WAIT_SECONDS}"
  done
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  GPU_AUDIT="dry_run index=${DEVICES}"
else
  GPU_AUDIT="$(DEVICE_INDEX="${DEVICES}" python - <<'PY'
import os
import torch
index = int(os.environ["DEVICE_INDEX"])
free, total = torch.cuda.mem_get_info(index)
print(f"index={index} name={torch.cuda.get_device_name(index)} free={free} total={total} ratio={free/total:.6f}")
PY
)"
fi

MATRIX_COMMAND=(
  python -u scripts/run_tfm_experiment.py
  --matrix
  --matrix-resume
  --data-root openml_cc18
  --dataset-max-classes 10
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --workers "${WORKERS}"
  --devices "${DEVICES}"
  --random-state "${SEED}"
  --ttt-epochs "${TTT_EPOCHS}"
  --ttt-lr 1e-5
  --ttt-patience "${TTT_PATIENCE}"
  --ttt-c-metric model_native_l2
)
if [[ -n "${N_ESTIMATORS}" ]]; then
  MATRIX_COMMAND+=(--n-estimators "${N_ESTIMATORS}")
fi
if [[ "${#DATASET_NAMES[@]}" -gt 0 ]]; then
  MATRIX_COMMAND+=(--dataset-names "${DATASET_NAMES[@]}")
fi
if [[ "${PRESET}" == "smoke" ]]; then
  MATRIX_COMMAND+=(--matrix-fail-fast)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  MATRIX_COMMAND+=(--dry-run)
fi

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

echo "preset: ${PRESET}"
echo "wrapper_pid: $$"
echo "run_id: ${RUN_ID}"
echo "matrix_root: ${MATRIX_ROOT}"
echo "log_file: ${LOG_FILE}"
echo "selected_gpu: ${GPU_AUDIT}"
print_command "${MATRIX_COMMAND[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${MATRIX_COMMAND[@]}"
  exit $?
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "started_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

set +e
"${MATRIX_COMMAND[@]}"
MATRIX_EXIT_CODE=$?
set -e

VIEW_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_view_manifest"])' "${MATRIX_ROOT}/matrix_manifest.json")"
SUMMARY_COMMAND=(
  python -u scripts/summarize_openmlcc18_tfm_matrix.py
  --matrix-root "${MATRIX_ROOT}"
  --view-manifest "${VIEW_MANIFEST}"
  --random-state "${SEED}"
  --expected-datasets "${EXPECTED_DATASETS}"
)
if [[ "${PRESET}" == "smoke" ]]; then
  SUMMARY_COMMAND+=(--require-smoke)
fi
print_command "${SUMMARY_COMMAND[@]}"

set +e
"${SUMMARY_COMMAND[@]}"
SUMMARY_EXIT_CODE=$?
set -e

echo "matrix_exit_code: ${MATRIX_EXIT_CODE}"
echo "summary_exit_code: ${SUMMARY_EXIT_CODE}"
echo "completed_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
if [[ "${MATRIX_EXIT_CODE}" -ne 0 ]]; then
  exit "${MATRIX_EXIT_CODE}"
fi
exit "${SUMMARY_EXIT_CODE}"
