#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data184}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/limix/data184/limix2m}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/baseline_compare/LimiX/LimiX-2M.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/baseline_compare/LimiX/config/cls_default_noretrieval.json}"
SEEDS="${SEEDS:-42}"
METHODS="${METHODS:-infer,ft,faware_ft,lora,prefix,last_block,micp,mixturepfn}"
GPUS="${GPUS:-0}"
RESUME="${RESUME:-1}"
RETRY_FAILED="${RETRY_FAILED:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DATASETS="${MAX_DATASETS:-}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "${MAX_PARALLEL}" -lt 1 || "${MAX_PARALLEL}" -gt "${#GPU_IDS[@]}" ]]; then
  echo "MAX_PARALLEL must be between 1 and the number of GPUS" >&2
  exit 2
fi

NORMALIZED_SEEDS="${SEEDS//,/ }"
NORMALIZED_METHODS="${METHODS//,/ }"
read -r -a SEED_VALUES <<< "${NORMALIZED_SEEDS}"
read -r -a METHOD_VALUES <<< "${NORMALIZED_METHODS}"
METHODS_CSV="$(IFS=,; echo "${METHOD_VALUES[*]}")"

EXPECTED_DATASETS=184
if [[ -n "${MAX_DATASETS}" ]]; then
  EXPECTED_DATASETS="${MAX_DATASETS}"
fi
PLANNED_CELLS=$(( ${#SEED_VALUES[@]} * ${#METHOD_VALUES[@]} ))
PLANNED_TASKS=$(( PLANNED_CELLS * EXPECTED_DATASETS ))

build_command() {
  local seed="$1"
  local method="$2"
  local gpu="$3"
  local out_dir="${RESULT_ROOT}/seed${seed}/${method}"
  COMMAND=(
    "${PYTHON}" -u "${REPO_ROOT}/runners/limix/run_experiment.py"
    --method "${method}"
    --data-root "${DATA_ROOT}"
    --out-dir "${out_dir}"
    --model-path "${MODEL_PATH}"
    --config-path "${CONFIG_PATH}"
    --seed "${seed}"
    --gpu "${gpu}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    COMMAND+=(--resume)
  else
    COMMAND+=(--no-resume)
  fi
  if [[ "${RETRY_FAILED}" == "1" ]]; then
    COMMAND+=(--retry-failed)
  else
    COMMAND+=(--no-retry-failed)
  fi
  if [[ -n "${MAX_DATASETS}" ]]; then
    COMMAND+=(--max-datasets "${MAX_DATASETS}")
  fi
}

print_command() {
  printf 'command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
}

echo "planned_cells: ${PLANNED_CELLS}"
echo "datasets_per_cell: ${EXPECTED_DATASETS}"
echo "planned_dataset_method_tasks: ${PLANNED_TASKS}"
echo "result_root: ${RESULT_ROOT}"

trial_index=0
batch_pids=()
batch_labels=()
batch_failed=0

wait_batch() {
  local index
  local exit_code
  for ((index=0; index<${#batch_pids[@]}; index++)); do
    exit_code=0
    wait "${batch_pids[$index]}" || exit_code=$?
    if [[ "${exit_code}" -ne 0 ]]; then
      echo "cell_failed: ${batch_labels[$index]} exit_code=${exit_code}" >&2
      batch_failed=1
    fi
  done
  batch_pids=()
  batch_labels=()
}

for seed in "${SEED_VALUES[@]}"; do
  for method in "${METHOD_VALUES[@]}"; do
    gpu_index=$(( trial_index % MAX_PARALLEL ))
    gpu="${GPU_IDS[$gpu_index]}"
    build_command "${seed}" "${method}" "${gpu}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      print_command
    else
      out_dir="${RESULT_ROOT}/seed${seed}/${method}"
      mkdir -p "${out_dir}"
      print_command
      (
        "${COMMAND[@]}" 2>&1 | tee "${out_dir}/run.log"
      ) &
      batch_pids+=("$!")
      batch_labels+=("seed${seed}/${method}/gpu${gpu}")
      if [[ "${#batch_pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
        wait_batch
      fi
    fi
    trial_index=$(( trial_index + 1 ))
  done
done

if [[ "${DRY_RUN}" != "1" && "${#batch_pids[@]}" -gt 0 ]]; then
  wait_batch
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for seed in "${SEED_VALUES[@]}"; do
  "${PYTHON}" "${REPO_ROOT}/runners/limix/summarize_matrix.py" \
    --root "${RESULT_ROOT}/seed${seed}" \
    --methods "${METHODS_CSV}" \
    --expected-datasets "${EXPECTED_DATASETS}"
done

if [[ "${batch_failed}" -ne 0 ]]; then
  exit 1
fi
