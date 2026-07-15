#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-topconf_synthetic_v1_$(date +%Y%m%d_%H%M%S)}"
PRESET="${PRESET:-paper}"
if [[ -z "${SUITE_ID+x}" ]]; then
  if [[ "${PRESET}" == "paper512" ]]; then
    SUITE_ID="topconf_synthetic_v1_512_seed42_49"
  else
    SUITE_ID="topconf_synthetic_v1_seed42"
  fi
fi
if [[ -z "${SEEDS+x}" ]]; then
  if [[ "${PRESET}" == "paper512" ]]; then
    SEEDS="42 43 44 45 46 47 48 49"
  else
    SEEDS="42"
  fi
fi
DATA_ROOT="${DATA_ROOT:-results/topconf_synthetic_data/${SUITE_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topconf_synthetic_matrix}"
MATRIX_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${LOG_DIR:-logs/topconf_synthetic_matrix}"
WORKERS="${WORKERS:-1}"
DEVICES="${DEVICES:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
TTT_EPOCHS="${TTT_EPOCHS:-30}"
MAX_DATASETS="${MAX_DATASETS:-}"
N_ESTIMATORS="${N_ESTIMATORS:-}"
MATRIX_MODELS="${MATRIX_MODELS:-}"
MATRIX_METHODS="${MATRIX_METHODS:-}"
TTT_C_METRIC="${TTT_C_METRIC:-standardized_l2}"
REFERENCE_MATRIX_ROOT="${REFERENCE_MATRIX_ROOT:-}"
REFERENCE_METHODS="${REFERENCE_METHODS:-infer ft}"
VALIDATE_MODEL_NATIVE_FAWARE="${VALIDATE_MODEL_NATIVE_FAWARE:-0}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-}"
if [[ -z "${EXPECTED_DATASETS}" && "${PRESET}" == "paper512" ]]; then
  EXPECTED_DATASETS="512"
fi
REGENERATE="${REGENERATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

GENERATOR_COMMAND=(
  python -u scripts/generate_topconf_synthetic_suite.py
  --data-root "${DATA_ROOT}"
  --preset "${PRESET}"
  --seeds "${SEEDS}"
)
if [[ "${REGENERATE}" == "1" ]]; then
  GENERATOR_COMMAND+=(--force)
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
  --random-state "${RANDOM_STATE}"
  --ttt-epochs "${TTT_EPOCHS}"
  --ttt-c-metric "${TTT_C_METRIC}"
)
if [[ -n "${MATRIX_MODELS}" ]]; then
  read -r -a MATRIX_MODEL_ARGS <<< "${MATRIX_MODELS}"
  MATRIX_COMMAND+=(--matrix-models "${MATRIX_MODEL_ARGS[@]}")
fi
if [[ -n "${MATRIX_METHODS}" ]]; then
  read -r -a MATRIX_METHOD_ARGS <<< "${MATRIX_METHODS}"
  MATRIX_COMMAND+=(--matrix-methods "${MATRIX_METHOD_ARGS[@]}")
fi
if [[ -n "${MAX_DATASETS}" ]]; then
  MATRIX_COMMAND+=(--max-datasets "${MAX_DATASETS}")
fi
if [[ -n "${N_ESTIMATORS}" ]]; then
  MATRIX_COMMAND+=(--n-estimators "${N_ESTIMATORS}")
fi

SUMMARY_COMMAND=(
  python -u scripts/summarize_topconf_synthetic_matrix.py
  --manifest-csv "${DATA_ROOT}/synthetic_manifest.csv"
  --matrix-root "${MATRIX_ROOT}"
  --random-state "${RANDOM_STATE}"
)
if [[ -n "${MATRIX_MODELS}" ]]; then
  SUMMARY_COMMAND+=(--models "${MATRIX_MODEL_ARGS[@]}")
fi
if [[ -n "${MATRIX_METHODS}" ]]; then
  SUMMARY_COMMAND+=(--methods "${MATRIX_METHOD_ARGS[@]}")
fi
if [[ -n "${REFERENCE_MATRIX_ROOT}" ]]; then
  read -r -a REFERENCE_METHOD_ARGS <<< "${REFERENCE_METHODS}"
  SUMMARY_COMMAND+=(
    --reference-matrix-root "${REFERENCE_MATRIX_ROOT}"
    --reference-methods "${REFERENCE_METHOD_ARGS[@]}"
  )
fi
if [[ "${VALIDATE_MODEL_NATIVE_FAWARE}" == "1" ]]; then
  SUMMARY_COMMAND+=(--validate-model-native-faware)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "run_id: ${RUN_ID}"
  echo "data_root: ${DATA_ROOT}"
  echo "matrix_root: ${MATRIX_ROOT}"
  print_command "${GENERATOR_COMMAND[@]}"
  print_command "${MATRIX_COMMAND[@]}"
  print_command "${SUMMARY_COMMAND[@]}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "run_id: ${RUN_ID}"
echo "data_root: ${DATA_ROOT}"
echo "matrix_root: ${MATRIX_ROOT}"
echo "log_file: ${LOG_FILE}"

if [[ -f "${DATA_ROOT}/synthetic_manifest.csv" && "${REGENERATE}" != "1" ]]; then
  echo "reuse_existing_suite: ${DATA_ROOT}"
else
  run_command "${GENERATOR_COMMAND[@]}"
fi

if [[ -n "${EXPECTED_DATASETS}" ]]; then
  ACTUAL_DATASETS="$(awk 'END { print NR - 1 }' "${DATA_ROOT}/synthetic_manifest.csv")"
  if [[ "${ACTUAL_DATASETS}" != "${EXPECTED_DATASETS}" ]]; then
    echo "dataset_count_mismatch: expected=${EXPECTED_DATASETS} actual=${ACTUAL_DATASETS}" >&2
    exit 2
  fi
  echo "verified_dataset_count: ${ACTUAL_DATASETS}"
fi

set +e
run_command "${MATRIX_COMMAND[@]}"
MATRIX_EXIT_CODE=$?
set -e

run_command "${SUMMARY_COMMAND[@]}"
echo "matrix_exit_code: ${MATRIX_EXIT_CODE}"
exit "${MATRIX_EXIT_CODE}"
