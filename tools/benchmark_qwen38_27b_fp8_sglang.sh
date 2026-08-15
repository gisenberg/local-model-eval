#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8092}
SERVED_NAME=${SERVED_NAME:-qwen38-27b-fp8-sglang-mtp4}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/qwen38_27b_fp8_sglang_mtp4}
PYTHON=${PYTHON:-/home/gisenberg/.micromamba/envs/cuda/bin/python}
REASONING_BUDGET_MESSAGE=${REASONING_BUDGET_MESSAGE:-$'\n\nWait, I\x27m overthinking this. Thinking budget complete. Let\x27s answer now.'}

mkdir -p "$OUTPUT_DIR"

"$PYTHON" tools/mimo_v25_api_smoke.py \
  --base-url "$BASE_URL/v1" \
  --model "$SERVED_NAME" \
  --reasoning required \
  --reasoning-effort medium \
  --max-tokens 8192 \
  --output "$OUTPUT_DIR/api_smoke.json"

"$PYTHON" tools/muse_glimmer_fp8_bench.py \
  --base-url "$BASE_URL/v1" \
  --model "$SERVED_NAME" \
  --reasoning-strength medium \
  --reasoning-effort medium \
  --reasoning-budget-tokens 4096 \
  --reasoning-budget-message "$REASONING_BUDGET_MESSAGE" \
  --top-k 20 \
  --presence-penalty 0.0 \
  --repetition-penalty 1.0 \
  --thinking-mode on \
  --warmups 1 \
  --runs 3 \
  --throughput-tokens 512 \
  --coding-tokens 16384 \
  --output "$OUTPUT_DIR/lightweight"

"$PYTHON" tools/qwen38_sglang_concurrency_bench.py \
  --base-url "$BASE_URL/v1" \
  --model "$SERVED_NAME" \
  --reasoning-effort medium \
  --reasoning-budget-tokens 4096 \
  --reasoning-budget-message "$REASONING_BUDGET_MESSAGE" \
  --concurrency 1,2,4,8 \
  --trials 2 \
  --max-tokens 512 \
  --output "$OUTPUT_DIR/concurrency.json"

"$PYTHON" tools/mimo_v25_long_context_smoke.py \
  --base-url "$BASE_URL" \
  --model "$SERVED_NAME" \
  --tokenizer-api vllm \
  --target-tokens 250000 \
  --max-tokens 256 \
  --output "$OUTPUT_DIR/context_250k.json"
