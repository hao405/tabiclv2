#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit these knobs before launching a full benchmark.
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATA_ROOTS="${DATA_ROOTS:-openml_cc18 tabzilla}"
OUT_ROOT="${OUT_ROOT:-baseline_compare/compare_results_openml_cc18_tabzilla}"
WORKERS="${WORKERS:-1}"
GPUS="${GPUS:-3}"
TABICL_TTT_GPU_GROUPS="${TABICL_TTT_GPU_GROUPS:-}"
MAX_DATASETS="${MAX_DATASETS:-}"
RANDOM_STATE="${RANDOM_STATE:-42}"
VERBOSE="${VERBOSE:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
LIVE_LOG="${LIVE_LOG:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONUNBUFFERED

TABICL_V1_CHECKPOINT="${TABICL_V1_CHECKPOINT:-tabicl-classifier-v1.1-20250506.ckpt}"
TABICL_V2_CHECKPOINT="${TABICL_V2_CHECKPOINT:-tabicl-classifier-v2-20260212.ckpt}"
TABICL_N_ESTIMATORS="${TABICL_N_ESTIMATORS:-32}"
TABICL_BATCH_SIZE="${TABICL_BATCH_SIZE:-8}"
TABICL_KV_CACHE="${TABICL_KV_CACHE:-False}"
TABICL_USE_AMP="${TABICL_USE_AMP:-auto}"
TABICL_USE_FA3="${TABICL_USE_FA3:-auto}"
TABICL_OFFLOAD_MODE="${TABICL_OFFLOAD_MODE:-auto}"
TABICL_PREFETCH_MODELS="${TABICL_PREFETCH_MODELS:-4}"

TABPFN_N_ESTIMATORS="${TABPFN_N_ESTIMATORS:-8}"
TABPFN_MANY_CLASS="${TABPFN_MANY_CLASS:-auto}"
TABPFN_MANY_CLASS_REDUNDANCY="${TABPFN_MANY_CLASS_REDUNDANCY:-4}"
TABPFN_MANY_CLASS_N_ESTIMATORS="${TABPFN_MANY_CLASS_N_ESTIMATORS:-16}"

LIMIX_CONFIG_PATH="${LIMIX_CONFIG_PATH:-config/cls_default_noretrieval.json}"
LIMIX_MODEL_CACHE_DIR="${LIMIX_MODEL_CACHE_DIR:-cache}"
LIMIX_HF_ENDPOINT="${LIMIX_HF_ENDPOINT:-https://hf-mirror.com}"
LIMIX_MAX_CLASSES="${LIMIX_MAX_CLASSES:-0}"
LIMIX_MAX_TRAIN_ROWS="${LIMIX_MAX_TRAIN_ROWS:-0}"
LIMIX_DIRECT_MAX_CLASSES="${LIMIX_DIRECT_MAX_CLASSES:-}"
LIMIX_TEST_BATCH_ROWS="${LIMIX_TEST_BATCH_ROWS:-0}"

TABR_MAX_EPOCHS="${TABR_MAX_EPOCHS:-1}"
TABR_PATIENCE="${TABR_PATIENCE:-4}"
TABR_BATCH_SIZE="${TABR_BATCH_SIZE:-256}"
TABR_CONTEXT_SIZE="${TABR_CONTEXT_SIZE:-96}"
TABR_D_MAIN="${TABR_D_MAIN:-128}"
TABR_D_MULTIPLIER="${TABR_D_MULTIPLIER:-2.0}"
TABR_ENCODER_N_BLOCKS="${TABR_ENCODER_N_BLOCKS:-0}"
TABR_PREDICTOR_N_BLOCKS="${TABR_PREDICTOR_N_BLOCKS:-1}"
TABR_CONTEXT_DROPOUT="${TABR_CONTEXT_DROPOUT:-0.0}"
TABR_DROPOUT0="${TABR_DROPOUT0:-0.0}"
TABR_DROPOUT1="${TABR_DROPOUT1:-0.0}"
TABR_LR="${TABR_LR:-1e-3}"
TABR_WEIGHT_DECAY="${TABR_WEIGHT_DECAY:-0.0}"
TABR_SYNTHETIC_VAL_FRACTION="${TABR_SYNTHETIC_VAL_FRACTION:-0.2}"

ORION_N_ESTIMATORS="${ORION_N_ESTIMATORS:-32}"
ORION_BATCH_SIZE="${ORION_BATCH_SIZE:-8}"
ORION_NORM_METHODS="${ORION_NORM_METHODS:-}"
ORION_FEAT_SHUFFLE_METHOD="${ORION_FEAT_SHUFFLE_METHOD:-latin}"
ORION_CHECKPOINT_VERSION="${ORION_CHECKPOINT_VERSION:-Orion-MSP-v1.0.ckpt}"
ORION_SOFTMAX_TEMPERATURE="${ORION_SOFTMAX_TEMPERATURE:-0.9}"
ORION_OUTLIER_THRESHOLD="${ORION_OUTLIER_THRESHOLD:-4.0}"

TTT_EPOCHS="${TTT_EPOCHS:-30}"
TTT_LR="${TTT_LR:-1e-5}"
TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-0.01}"
TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
TTT_MAX_CHUNK_SIZE="${TTT_MAX_CHUNK_SIZE:-10000}"
TTT_MIN_CHUNK_SIZE="${TTT_MIN_CHUNK_SIZE:-50}"
TTT_QUERY_RATIO="${TTT_QUERY_RATIO:-0.2}"
TTT_PATIENCE="${TTT_PATIENCE:-8}"
TTT_MIN_DELTA="${TTT_MIN_DELTA:-1e-4}"
TTT_EVAL_METRIC="${TTT_EVAL_METRIC:-accuracy}"
TTT_VALIDATION_FRACTION="${TTT_VALIDATION_FRACTION:-0.1}"
TTT_N_ESTIMATORS_FINETUNE="${TTT_N_ESTIMATORS_FINETUNE:-2}"
TTT_VALIDATION_N_ESTIMATORS="${TTT_VALIDATION_N_ESTIMATORS:-2}"
TTT_MICRO_BATCH_SIZE="${TTT_MICRO_BATCH_SIZE:-1}"
TTT_EARLY_STOPPING="${TTT_EARLY_STOPPING:-True}"

TABICL_TTT_SCHEDULER="${TABICL_TTT_SCHEDULER:-cosine_warmup}"
TABICL_TTT_WARMUP_PROPORTION="${TABICL_TTT_WARMUP_PROPORTION:-0.1}"
TABICL_TTT_DTYPE="${TABICL_TTT_DTYPE:-float32}"
TABICL_TTT_FREEZE_COL="${TABICL_TTT_FREEZE_COL:-False}"
TABICL_TTT_FREEZE_ROW="${TABICL_TTT_FREEZE_ROW:-False}"
TABICL_TTT_FREEZE_ICL="${TABICL_TTT_FREEZE_ICL:-False}"
TABICL_TTT_SAVE_CKPT="${TABICL_TTT_SAVE_CKPT:-False}"

TABPFN_TTT_MAX_CHUNK_SIZE="${TABPFN_TTT_MAX_CHUNK_SIZE:-2000}"
TABPFN_TTT_EVAL_METRIC="${TABPFN_TTT_EVAL_METRIC:-roc_auc}"

LIMIX_TTT_LR="${LIMIX_TTT_LR:-5e-6}"
LIMIX_TTT_WEIGHT_DECAY="${LIMIX_TTT_WEIGHT_DECAY:-0.0}"
LIMIX_TTT_MAX_CHUNK_SIZE="${LIMIX_TTT_MAX_CHUNK_SIZE:-200}"
LIMIX_TTT_MIN_CHUNK_SIZE="${LIMIX_TTT_MIN_CHUNK_SIZE:-10}"
LIMIX_TTT_ACCUMULATION_BATCH_SIZE="${LIMIX_TTT_ACCUMULATION_BATCH_SIZE:-10000}"

if [[ "${OUT_ROOT}" = /* ]]; then
    OUT_ROOT_ABS="${OUT_ROOT}"
else
    OUT_ROOT_ABS="${REPO_ROOT}/${OUT_ROOT}"
fi
LOG_ROOT="${OUT_ROOT_ABS}/logs"
FAILURES=()

MAX_DATASETS_ARGS=()
if [[ -n "${MAX_DATASETS}" ]]; then
    MAX_DATASETS_ARGS=(--max-datasets "${MAX_DATASETS}")
fi

VERBOSE_ARGS=()
if [[ "${VERBOSE}" == "1" || "${VERBOSE}" == "true" || "${VERBOSE}" == "True" ]]; then
    VERBOSE_ARGS=(--verbose)
fi

LIMIX_DIRECT_MAX_CLASSES_ARGS=()
if [[ -n "${LIMIX_DIRECT_MAX_CLASSES}" ]]; then
    LIMIX_DIRECT_MAX_CLASSES_ARGS=(--direct-max-classes "${LIMIX_DIRECT_MAX_CLASSES}")
fi

ORION_NORM_METHODS_ARGS=()
if [[ -n "${ORION_NORM_METHODS}" ]]; then
    ORION_NORM_METHODS_ARGS=(--norm-methods "${ORION_NORM_METHODS}")
fi

sanitize_label() {
    printf '%s' "$1" | tr '/ :' '___' | tr -cd 'A-Za-z0-9_.-'
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    local label="$1"
    shift
    local log_label
    log_label="$(sanitize_label "${label}")"
    local log_path="${LOG_ROOT}/${log_label}.log"
    local status=0
    local tee_status=0
    local -a pipe_status=()
    mkdir -p "$(dirname "${log_path}")"
    echo
    echo "[run] ${label}"
    print_command "$@"
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "True" ]]; then
        echo "[dry-run] log would be: ${log_path}"
        return 0
    fi
    if [[ "${LIVE_LOG}" == "0" || "${LIVE_LOG}" == "false" || "${LIVE_LOG}" == "False" ]]; then
        "$@" >"${log_path}" 2>&1
        status=$?
    else
        "$@" 2>&1 | tee "${log_path}"
        pipe_status=("${PIPESTATUS[@]}")
        status="${pipe_status[0]}"
        tee_status="${pipe_status[1]:-0}"
        if [[ "${status}" -eq 0 && "${tee_status}" -ne 0 ]]; then
            status="${tee_status}"
        fi
    fi
    if [[ "${status}" -eq 0 ]]; then
        echo "[ok] ${label}"
        echo "[log] ${log_path}"
        return 0
    else
        echo "[fail] ${label} returncode=${status}" >&2
        echo "[log] ${log_path}" >&2
        FAILURES+=("${label}:returncode=${status}:log=${log_path}")
        if [[ "${STOP_ON_FAILURE}" == "1" || "${STOP_ON_FAILURE}" == "true" || "${STOP_ON_FAILURE}" == "True" ]]; then
            exit "${status}"
        fi
    fi
    return 0
}

benchmark_name_for_root() {
    basename "$1"
}

run_tabicl_ttt() {
    local label="$1"
    local data_root="$2"
    local out_dir="$3"
    local checkpoint="$4"
    if [[ -n "${TABICL_TTT_GPU_GROUPS}" ]]; then
        run_cmd "${label}" "${PYTHON_BIN}" 1C_Chunk_TTT.py --data-root "${data_root}" --out-dir "${out_dir}" --workers "${WORKERS}" --gpu-groups "${TABICL_TTT_GPU_GROUPS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --checkpoint-version "${checkpoint}" --n-estimators "${TABICL_N_ESTIMATORS}" --batch-size "${TABICL_BATCH_SIZE}" --kv-cache "${TABICL_KV_CACHE}" --use-amp "${TABICL_USE_AMP}" --use-fa3 "${TABICL_USE_FA3}" --offload-mode "${TABICL_OFFLOAD_MODE}" --prefetch-models "${TABICL_PREFETCH_MODELS}" --ttt-lr "${TTT_LR}" --ttt-scheduler "${TABICL_TTT_SCHEDULER}" --ttt-warmup-proportion "${TABICL_TTT_WARMUP_PROPORTION}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-dtype "${TABICL_TTT_DTYPE}" --ttt-micro-batch-size "${TTT_MICRO_BATCH_SIZE}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TTT_MAX_CHUNK_SIZE}" --ttt-min-chunk-size "${TTT_MIN_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-early-stopping "${TTT_EARLY_STOPPING}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" --ttt-freeze-col "${TABICL_TTT_FREEZE_COL}" --ttt-freeze-row "${TABICL_TTT_FREEZE_ROW}" --ttt-freeze-icl "${TABICL_TTT_FREEZE_ICL}" --ttt-save-ckpt "${TABICL_TTT_SAVE_CKPT}" "${VERBOSE_ARGS[@]}"
    else
        run_cmd "${label}" "${PYTHON_BIN}" 1C_Chunk_TTT.py --data-root "${data_root}" --out-dir "${out_dir}" --workers "${WORKERS}" --gpu-groups "" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --checkpoint-version "${checkpoint}" --n-estimators "${TABICL_N_ESTIMATORS}" --batch-size "${TABICL_BATCH_SIZE}" --kv-cache "${TABICL_KV_CACHE}" --use-amp "${TABICL_USE_AMP}" --use-fa3 "${TABICL_USE_FA3}" --offload-mode "${TABICL_OFFLOAD_MODE}" --prefetch-models "${TABICL_PREFETCH_MODELS}" --ttt-lr "${TTT_LR}" --ttt-scheduler "${TABICL_TTT_SCHEDULER}" --ttt-warmup-proportion "${TABICL_TTT_WARMUP_PROPORTION}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-dtype "${TABICL_TTT_DTYPE}" --ttt-micro-batch-size "${TTT_MICRO_BATCH_SIZE}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TTT_MAX_CHUNK_SIZE}" --ttt-min-chunk-size "${TTT_MIN_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-early-stopping "${TTT_EARLY_STOPPING}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" --ttt-freeze-col "${TABICL_TTT_FREEZE_COL}" --ttt-freeze-row "${TABICL_TTT_FREEZE_ROW}" --ttt-freeze-icl "${TABICL_TTT_FREEZE_ICL}" --ttt-save-ckpt "${TABICL_TTT_SAVE_CKPT}" "${VERBOSE_ARGS[@]}"
    fi
}

echo "run_id: ${RUN_ID}"
echo "repo_root: ${REPO_ROOT}"
echo "data_roots: ${DATA_ROOTS}"
echo "out_root: ${OUT_ROOT}"
echo "out_root_abs: ${OUT_ROOT_ABS}"
echo "workers: ${WORKERS}"
echo "gpus: ${GPUS}"
echo "tabicl_ttt_gpu_groups: ${TABICL_TTT_GPU_GROUPS:-<use GPUS>}"
echo "max_datasets: ${MAX_DATASETS:-<all>}"
echo "dry_run: ${DRY_RUN}"
echo "stop_on_failure: ${STOP_ON_FAILURE}"
echo "live_log: ${LIVE_LOG}"
echo "pythonunbuffered: ${PYTHONUNBUFFERED}"

read -r -a DATA_ROOT_LIST <<< "${DATA_ROOTS}"
for data_root in "${DATA_ROOT_LIST[@]}"; do
    benchmark_name="$(benchmark_name_for_root "${data_root}")"
    if [[ ! -d "${data_root}" ]]; then
        echo "[skip] data root does not exist: ${data_root}" >&2
        FAILURES+=("${benchmark_name}:missing_data_root=${data_root}")
        if [[ "${STOP_ON_FAILURE}" == "1" || "${STOP_ON_FAILURE}" == "true" || "${STOP_ON_FAILURE}" == "True" ]]; then
            exit 1
        fi
        continue
    fi
    data_root_abs="$(cd "${data_root}" && pwd)"
    benchmark_out_root="${OUT_ROOT_ABS}/${benchmark_name}"

    echo
    echo "========== benchmark: ${benchmark_name} (${data_root_abs}) =========="

    run_cmd "${benchmark_name}/inference/tabicl" "${PYTHON_BIN}" benchmark.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabicl" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --checkpoint-version "${TABICL_V1_CHECKPOINT}" --n-estimators "${TABICL_N_ESTIMATORS}" --batch-size "${TABICL_BATCH_SIZE}" --kv-cache "${TABICL_KV_CACHE}" --use-amp "${TABICL_USE_AMP}" --use-fa3 "${TABICL_USE_FA3}" --offload-mode "${TABICL_OFFLOAD_MODE}" --prefetch-models "${TABICL_PREFETCH_MODELS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/tabiclv2" "${PYTHON_BIN}" benchmark.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabiclv2" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --checkpoint-version "${TABICL_V2_CHECKPOINT}" --n-estimators "${TABICL_N_ESTIMATORS}" --batch-size "${TABICL_BATCH_SIZE}" --kv-cache "${TABICL_KV_CACHE}" --use-amp "${TABICL_USE_AMP}" --use-fa3 "${TABICL_USE_FA3}" --offload-mode "${TABICL_OFFLOAD_MODE}" --prefetch-models "${TABICL_PREFETCH_MODELS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/tabpfnv2" "${PYTHON_BIN}" baseline_compare/TabPFN-main/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabpfnv2" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v2 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/tabpfnv25" "${PYTHON_BIN}" baseline_compare/TabPFN-main/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabpfnv25" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v2.5 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/tabpfnv3" "${PYTHON_BIN}" baseline_compare/TabPFN-main/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabpfnv3" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v3 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/limix" "${PYTHON_BIN}" baseline_compare/LimiX/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/limix" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --config-path "${LIMIX_CONFIG_PATH}" --model-cache-dir "${LIMIX_MODEL_CACHE_DIR}" --hf-endpoint "${LIMIX_HF_ENDPOINT}" --max-classes "${LIMIX_MAX_CLASSES}" --max-train-rows "${LIMIX_MAX_TRAIN_ROWS}" "${LIMIX_DIRECT_MAX_CLASSES_ARGS[@]}" --test-batch-rows "${LIMIX_TEST_BATCH_ROWS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/tabr" "${PYTHON_BIN}" baseline_compare/tabular-dl-tabr/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/tabr" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --max-epochs "${TABR_MAX_EPOCHS}" --patience "${TABR_PATIENCE}" --batch-size "${TABR_BATCH_SIZE}" --context-size "${TABR_CONTEXT_SIZE}" --d-main "${TABR_D_MAIN}" --d-multiplier "${TABR_D_MULTIPLIER}" --encoder-n-blocks "${TABR_ENCODER_N_BLOCKS}" --predictor-n-blocks "${TABR_PREDICTOR_N_BLOCKS}" --context-dropout "${TABR_CONTEXT_DROPOUT}" --dropout0 "${TABR_DROPOUT0}" --dropout1 "${TABR_DROPOUT1}" --lr "${TABR_LR}" --weight-decay "${TABR_WEIGHT_DECAY}" --synthetic-val-fraction "${TABR_SYNTHETIC_VAL_FRACTION}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/inference/orion_msp" "${PYTHON_BIN}" baseline_compare/Orion-MSP/benchmark_infer.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/inference/orion_msp" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --n-estimators "${ORION_N_ESTIMATORS}" --batch-size "${ORION_BATCH_SIZE}" "${ORION_NORM_METHODS_ARGS[@]}" --feat-shuffle-method "${ORION_FEAT_SHUFFLE_METHOD}" --checkpoint-version "${ORION_CHECKPOINT_VERSION}" --softmax-temperature "${ORION_SOFTMAX_TEMPERATURE}" --outlier-threshold "${ORION_OUTLIER_THRESHOLD}" "${VERBOSE_ARGS[@]}"

    run_tabicl_ttt "${benchmark_name}/ttt/tabicl" "${data_root_abs}" "${benchmark_out_root}/ttt/tabicl" "${TABICL_V1_CHECKPOINT}"
    run_tabicl_ttt "${benchmark_name}/ttt/tabiclv2" "${data_root_abs}" "${benchmark_out_root}/ttt/tabiclv2" "${TABICL_V2_CHECKPOINT}"
    run_cmd "${benchmark_name}/ttt/tabpfnv2" "${PYTHON_BIN}" baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/ttt/tabpfnv2" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v2 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" --ttt --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TABPFN_TTT_MAX_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-lr "${TTT_LR}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TABPFN_TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/ttt/tabpfnv25" "${PYTHON_BIN}" baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/ttt/tabpfnv25" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v2.5 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" --ttt --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TABPFN_TTT_MAX_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-lr "${TTT_LR}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TABPFN_TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/ttt/tabpfnv3" "${PYTHON_BIN}" baseline_compare/TabPFN-main/Tabpfn_1c_ttt.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/ttt/tabpfnv3" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --model-version v3 --n-estimators "${TABPFN_N_ESTIMATORS}" --many-class "${TABPFN_MANY_CLASS}" --many-class-redundancy "${TABPFN_MANY_CLASS_REDUNDANCY}" --many-class-n-estimators "${TABPFN_MANY_CLASS_N_ESTIMATORS}" --ttt --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TABPFN_TTT_MAX_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-lr "${TTT_LR}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TABPFN_TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" "${VERBOSE_ARGS[@]}"
    run_cmd "${benchmark_name}/ttt/limix" "${PYTHON_BIN}" baseline_compare/LimiX/limix_ttt.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/ttt/limix" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --config-path "${LIMIX_CONFIG_PATH}" --model-cache-dir "${LIMIX_MODEL_CACHE_DIR}" --hf-endpoint "${LIMIX_HF_ENDPOINT}" --max-classes "${LIMIX_MAX_CLASSES}" --max-train-rows "${LIMIX_MAX_TRAIN_ROWS}" "${LIMIX_DIRECT_MAX_CLASSES_ARGS[@]}" --test-batch-rows "${LIMIX_TEST_BATCH_ROWS}" --ttt-holdout --ttt-lr "${LIMIX_TTT_LR}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-weight-decay "${LIMIX_TTT_WEIGHT_DECAY}" --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${LIMIX_TTT_MAX_CHUNK_SIZE}" --ttt-accumulation-batch-size "${LIMIX_TTT_ACCUMULATION_BATCH_SIZE}" --ttt-min-chunk-size "${LIMIX_TTT_MIN_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-early-stopping "${TTT_EARLY_STOPPING}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric accuracy --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-validation-n-estimators 0 --ttt-n-estimators-finetune 0 "${VERBOSE_ARGS[@]}"
    echo "[skip] ${benchmark_name}/ttt/tabr: TaBR has no direct TTT backend in baseline_compare"
    run_cmd "${benchmark_name}/ttt/orion_msp" "${PYTHON_BIN}" baseline_compare/Orion-MSP/orion_msp_ttt.py --data-root "${data_root_abs}" --out-dir "${benchmark_out_root}/ttt/orion_msp" --workers "${WORKERS}" --gpus "${GPUS}" "${MAX_DATASETS_ARGS[@]}" --random-state "${RANDOM_STATE}" --n-estimators "${ORION_N_ESTIMATORS}" --batch-size "${ORION_BATCH_SIZE}" "${ORION_NORM_METHODS_ARGS[@]}" --feat-shuffle-method "${ORION_FEAT_SHUFFLE_METHOD}" --checkpoint-version "${ORION_CHECKPOINT_VERSION}" --softmax-temperature "${ORION_SOFTMAX_TEMPERATURE}" --outlier-threshold "${ORION_OUTLIER_THRESHOLD}" --ttt --ttt-epochs "${TTT_EPOCHS}" --ttt-max-chunk-size "${TTT_MAX_CHUNK_SIZE}" --ttt-query-ratio "${TTT_QUERY_RATIO}" --ttt-lr "${TTT_LR}" --ttt-weight-decay "${TTT_WEIGHT_DECAY}" --ttt-grad-clip "${TTT_GRAD_CLIP}" --ttt-patience "${TTT_PATIENCE}" --ttt-min-delta "${TTT_MIN_DELTA}" --ttt-eval-metric "${TTT_EVAL_METRIC}" --ttt-validation-fraction "${TTT_VALIDATION_FRACTION}" --ttt-n-estimators-finetune "${TTT_N_ESTIMATORS_FINETUNE}" --ttt-validation-n-estimators "${TTT_VALIDATION_N_ESTIMATORS}" --ttt-micro-batch-size "${TTT_MICRO_BATCH_SIZE}" "${VERBOSE_ARGS[@]}"
done

echo
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
    echo "completed_with_failures: ${#FAILURES[@]}" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo "completed_ok"
echo "out_root: ${OUT_ROOT}"
