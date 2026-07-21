#!/usr/bin/env bash
set -uo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate tabicl
cd "$HOME/zh/tabiclv2"

RUN_ID="micp_smoke_wilt_gpu01_final_20260720_105000"
OUT_ROOT="results/PEFT_results/MICP/smoke/${RUN_ID}"
LOG_PATH="logs/micp/${RUN_ID}.log"
mkdir -p logs/micp

python PEFT_Tabicl/MICP.py \
  --model-family all \
  --method all \
  --data-root data184 \
  --dataset Wilt \
  --out-dir "$OUT_ROOT" \
  --gpus 0,1 \
  --seed 42 \
  --support-size 3000 \
  --gamma 5 \
  --ca-steps 2 \
  --ca-query-size 64 \
  --ca-lr 0.001 \
  --inference-batch-size 128 \
  --n-estimators 2 \
  --train-n-estimators 1 \
  --retrieval-backend faiss \
  --no-resume \
  >"$LOG_PATH" 2>&1
rc=$?

printf '[done] run_id=%s rc=%s out=%s log=%s\n' \
  "$RUN_ID" "$rc" "$OUT_ROOT" "$LOG_PATH" | tee -a "$LOG_PATH"
exit "$rc"
