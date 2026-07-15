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
DEVICES="${DEVICES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/managed_experiments}"
LOG_DIR="${LOG_DIR:-logs/tabiclv2_graph_scm_matrix}"
DRY_RUN="${DRY_RUN:-0}"
PRIOR_SRC_ROOT="${PRIOR_SRC_ROOT:-}"

if [[ "${PRESET}" == "smoke" ]]; then
  STAGE_COUNTS=(1 1 1)
  EXPECTED_DATASETS=3
  GENERATION_WORKERS="${GENERATION_WORKERS:-1}"
  DATA_ROOT="${DATA_ROOT:-results/tabiclv2_prior_data/graph_scm_3stage_smoke_seed${SEED}}"
  RUN_ID="${RUN_ID:-tabiclv2_graph_scm_3stage_smoke_$(date +%Y%m%d_%H%M%S)}"
  TTT_EPOCHS="${TTT_EPOCHS:-1}"
  TTT_PATIENCE="${TTT_PATIENCE:-1}"
  N_ESTIMATORS="${N_ESTIMATORS:-2}"
else
  STAGE_COUNTS=(171 171 170)
  EXPECTED_DATASETS=512
  GENERATION_WORKERS="${GENERATION_WORKERS:-4}"
  DATA_ROOT="${DATA_ROOT:-results/tabiclv2_prior_data/graph_scm_3stage_512_seed${SEED}}"
  RUN_ID="${RUN_ID:-tabiclv2_graph_scm_3stage_512_full_$(date +%Y%m%d_%H%M%S)}"
  TTT_EPOCHS="${TTT_EPOCHS:-30}"
  TTT_PATIENCE="${TTT_PATIENCE:-8}"
  N_ESTIMATORS="${N_ESTIMATORS:-}"
fi

MATRIX_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

GENERATOR_COMMAND=(
  python -u scripts/generate_tabiclv2_graph_scm_suite.py
  --data-root "${DATA_ROOT}"
  --stage-counts "${STAGE_COUNTS[@]}"
  --seed "${SEED}"
  --generation-workers "${GENERATION_WORKERS}"
  --resume
)
if [[ -n "${PRIOR_SRC_ROOT}" ]]; then
  GENERATOR_COMMAND=(env TABICLV2_PRIOR_SRC_ROOT="${PRIOR_SRC_ROOT}" "${GENERATOR_COMMAND[@]}")
fi

MATRIX_COMMAND=(
  python -u scripts/run_tfm_experiment.py
  --matrix
  --matrix-resume
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --workers "${WORKERS}"
  --devices "${DEVICES}"
  --random-state "${SEED}"
  --ttt-epochs "${TTT_EPOCHS}"
  --ttt-lr 1e-5
  --ttt-patience "${TTT_PATIENCE}"
  --ttt-c-metric standardized_l2
)
if [[ -n "${N_ESTIMATORS}" ]]; then
  MATRIX_COMMAND+=(--n-estimators "${N_ESTIMATORS}")
fi

SUMMARY_COMMAND=(
  python -u scripts/summarize_tabiclv2_graph_scm_matrix.py
  --manifest-csv "${DATA_ROOT}/prior_manifest.csv"
  --matrix-root "${MATRIX_ROOT}"
  --random-state "${SEED}"
)

echo "preset: ${PRESET}"
echo "wrapper_pid: $$"
echo "run_id: ${RUN_ID}"
echo "data_root: ${DATA_ROOT}"
echo "matrix_root: ${MATRIX_ROOT}"
echo "log_file: ${LOG_FILE}"
print_command "${GENERATOR_COMMAND[@]}"
print_command "${MATRIX_COMMAND[@]}"
print_command "${SUMMARY_COMMAND[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "started_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

"${GENERATOR_COMMAND[@]}"

ACTUAL_DATASETS="$(python -c 'import pandas as pd, sys; print(len(pd.read_csv(sys.argv[1])))' "${DATA_ROOT}/prior_manifest.csv")"
if [[ "${ACTUAL_DATASETS}" != "${EXPECTED_DATASETS}" ]]; then
  echo "dataset_count_mismatch: expected=${EXPECTED_DATASETS} actual=${ACTUAL_DATASETS}" >&2
  exit 2
fi
echo "verified_dataset_count: ${ACTUAL_DATASETS}"

set +e
"${MATRIX_COMMAND[@]}"
MATRIX_EXIT_CODE=$?
set -e

"${SUMMARY_COMMAND[@]}"
echo "matrix_exit_code: ${MATRIX_EXIT_CODE}"
echo "completed_at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
exit "${MATRIX_EXIT_CODE}"
