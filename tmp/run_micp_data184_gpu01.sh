#!/usr/bin/env bash
set -uo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate tabicl
cd "$HOME/zh/tabiclv2"

RUN_ID="micp_data184_seed42_2x1_gpu01_20260720_110000"
OUT_ROOT="results/PEFT_results/MICP/data184/seed42"
LOG_PATH="logs/micp/${RUN_ID}.log"
mkdir -p logs/micp

python PEFT_Tabicl/MICP.py \
  --model-family all \
  --method mixturepfn \
  --data-root data184 \
  --out-dir "$OUT_ROOT" \
  --gpus 0,1 \
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
  --resume \
  --retry-failed \
  >"$LOG_PATH" 2>&1
rc=$?

printf '[done] run_id=%s rc=%s out=%s log=%s\n' \
  "$RUN_ID" "$rc" "$OUT_ROOT" "$LOG_PATH" | tee -a "$LOG_PATH"
exit "$rc"
