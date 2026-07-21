#!/usr/bin/env bash
set -uo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate tabicl
cd "$HOME/zh/tabiclv2"

RUN_ID="micp_tabiclv2_failure11_recovery_gpu0_20260720_151300"
OUT_ROOT="results/PEFT_results/MICP/recovery/${RUN_ID}"
LOG_PATH="logs/micp/${RUN_ID}.log"
mkdir -p logs/micp

python PEFT_Tabicl/MICP_candidate.py \
  --model-family tabiclv2 \
  --method mixturepfn \
  --data-root data184 \
  --dataset 'BNG(breast-w)' \
  --dataset Click_prediction_small \
  --dataset SDSS17 \
  --dataset accelerometer \
  --dataset artificial-characters \
  --dataset customer_satisfaction_in_airline \
  --dataset dry_bean_dataset \
  --dataset internet_firewall \
  --dataset page-blocks \
  --dataset pendigits \
  --dataset shuttle \
  --out-dir "$OUT_ROOT" \
  --gpus 0 \
  --seed 42 \
  --support-size 3000 \
  --gamma 5 \
  --ca-steps 128 \
  --ca-query-size 64 \
  --ca-lr 0.001 \
  --inference-batch-size 1024 \
  --n-estimators 16 \
  --train-n-estimators 2 \
  --retrieval-backend faiss \
  --no-resume \
  >"$LOG_PATH" 2>&1
rc=$?

printf '[done] run_id=%s rc=%s out=%s log=%s\n' \
  "$RUN_ID" "$rc" "$OUT_ROOT" "$LOG_PATH" | tee -a "$LOG_PATH"
exit "$rc"
