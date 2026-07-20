#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-tabular_mixedseed_$(date +%Y%m%d_%H%M%S)}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${RUN_ID}_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2}"
LONG_SEEDS="${LONG_SEEDS:-3,10,16,42,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035}"
SHORT_SEEDS="${SHORT_SEEDS:-3,10,16}"
REUSE_ROOTS="${REUSE_ROOTS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/seed_search}"
LOG_ROOT="${LOG_ROOT:-logs/seed_search}"
RESUME="${RESUME:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
CONFIG_POLL_SECONDS="${CONFIG_POLL_SECONDS:-60}"

mkdir -p "${LOG_ROOT}"
WRAPPER_LOG="${LOG_ROOT}/${RUN_ID}.log"

COMMON_ARGS=(
  --output-root "${OUTPUT_ROOT}"
  --gpus "${GPUS}"
  --long-seeds "${LONG_SEEDS}"
  --short-seeds "${SHORT_SEEDS}"
  --python-bin "${PYTHON_BIN}"
  --gpu-poll-seconds "${GPU_POLL_SECONDS}"
)

for reuse_root in ${REUSE_ROOTS}; do
  COMMON_ARGS+=(--reuse-root "${reuse_root}")
done

if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

{
  echo "run_id: ${RUN_ID}"
  echo "smoke_run_id: ${SMOKE_RUN_ID}"
  echo "repo_root: ${REPO_ROOT}"
  echo "python_bin: ${PYTHON_BIN}"
  echo "gpus: ${GPUS}"
  echo "long_seeds: ${LONG_SEEDS}"
  echo "short_seeds: ${SHORT_SEEDS}"
  echo "reuse_roots: ${REUSE_ROOTS}"
  echo "output_root: ${OUTPUT_ROOT}"
  echo "started_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

  "${PYTHON_BIN}" scripts/run_tabular_seed_search_queue.py \
    --run-id "${SMOKE_RUN_ID}" \
    --mode smoke \
    "${COMMON_ARGS[@]}"

  "${PYTHON_BIN}" scripts/run_tabular_seed_search_queue.py \
    --run-id "${RUN_ID}" \
    --mode full \
    --wait-for-config \
    --config-poll-seconds "${CONFIG_POLL_SECONDS}" \
    "${COMMON_ARGS[@]}"

  echo "completed_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
} 2>&1 | tee -a "${WRAPPER_LOG}"
