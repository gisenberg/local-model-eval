#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
SWEAGENT=/home/gisenberg/.micromamba/envs/cuda/bin/sweagent
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python

SERVED_NAME=${SERVED_NAME:-qwen38-27b-fp8-sglang-mtp4}
PORT=${PORT:-8092}
NUM_WORKERS=${NUM_WORKERS:-15}
EVAL_WORKERS=${EVAL_WORKERS:-4}
OUT_DIR=${OUT_DIR:-experiments/sweagent_lite_qwen38_27b_fp8_sglang_mtp4_c15}
RUN_ID=${RUN_ID:-qwen38-27b-fp8-sglang-mtp4-medium-budget4k-c15-full300}

cd "$REPO"
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_swebench_lite.log") 2>&1

echo "=== Qwen3.8-27B FP8 SGLang MTP-4 SWE-bench Lite run ==="
date -Is
echo "out_dir=$OUT_DIR"
echo "run_id=$RUN_ID"
echo "num_workers=$NUM_WORKERS reasoning_effort=medium reasoning_budget=4096"

MODELS_JSON=$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models")
if ! jq -e --arg model "$SERVED_NAME" '.data[] | select(.id == $model)' <<<"$MODELS_JSON" >/dev/null; then
  echo "Expected model $SERVED_NAME is not ready on port $PORT"
  exit 1
fi
echo "Model server ready"

curl -fsS -X POST "http://127.0.0.1:$PORT/flush_cache"

echo "Starting SWE-agent batch"
"$SWEAGENT" run-batch \
  --config tools/sweagent-rtxpro6000-qwen38-sglang.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --output_dir "$OUT_DIR" \
  --num_workers "$NUM_WORKERS"

echo "SWE-agent batch complete"
date -Is

if [[ ! -f "$OUT_DIR/preds.json" ]]; then
  echo "No preds.json found; skipping evaluation"
  exit 1
fi

PRED_COUNT=$(jq 'length' "$OUT_DIR/preds.json")
if [[ "$PRED_COUNT" != 300 ]]; then
  echo "Expected 300 predictions but found $PRED_COUNT"
  exit 1
fi

mkdir -p "$OUT_DIR/eval"
echo "Starting official SWE-bench harness evaluation"
"$PYTHON" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --split test \
  --predictions_path "$OUT_DIR/preds.json" \
  --run_id "$RUN_ID" \
  --max_workers "$EVAL_WORKERS" \
  --cache_level instance \
  --report_dir "$OUT_DIR/eval"

REPORT="$OUT_DIR/eval/$RUN_ID.json"
ROOT_REPORT="$REPO/${OUT_DIR##*/}.$RUN_ID.json"
if [[ ! -f "$REPORT" && -f "$ROOT_REPORT" ]]; then
  cp "$ROOT_REPORT" "$REPORT"
fi
if [[ ! -f "$REPORT" ]]; then
  echo "Official evaluation finished without expected report $REPORT"
  exit 1
fi

RESOLVED=$(jq '.resolved_ids | length' "$REPORT")
echo "Official evaluation complete: $RESOLVED/$PRED_COUNT resolved"
if command -v wmux-notify >/dev/null 2>&1; then
  wmux-notify \
    --title "Qwen3.8 SGLang SWE-bench Lite complete" \
    --subtitle "Official evaluation finished" \
    --body "Resolved $RESOLVED/$PRED_COUNT SWE-bench Lite cases."
fi
echo "Done"
date -Is
