#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
LLAMA_DIR=/home/gisenberg/llama-build/src-deepseek-v4-0731/build/bin
LLAMA_SERVER="$LLAMA_DIR/llama-server"
CUDA_LIB=/home/gisenberg/.micromamba/envs/cuda/lib
SWEAGENT=/home/gisenberg/.micromamba/envs/cuda/bin/sweagent
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python

MODEL=${MODEL:-/mnt/extended/gisenberg/models/deepseek-v4-flash-antirez-imatrix-0731-1cd7b564/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}
SERVED_NAME=${SERVED_NAME:-deepseek-v4-flash-antirez-imatrix-0731}
PORT=${PORT:-8091}
TOTAL_CONTEXT=${TOTAL_CONTEXT:-262144}
NUM_WORKERS=${NUM_WORKERS:-1}
EVAL_WORKERS=${EVAL_WORKERS:-4}
OUT_DIR=${OUT_DIR:-$REPO/experiments/sweagent_lite_deepseek_v4_flash_antirez_imatrix_0731}
RUN_ID=${RUN_ID:-deepseek-v4-flash-antirez-imatrix-0731-full}
RUN_PROTOCOL_SMOKE=${RUN_PROTOCOL_SMOKE:-1}
PER_INSTANCE_CALL_LIMIT=${PER_INSTANCE_CALL_LIMIT:-}

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_swebench_lite.log") 2>&1

echo "=== DeepSeek-V4-Flash-0731 antirez IQ2_XXS imatrix SWE-bench Lite run ==="
date -Is
echo "out_dir=$OUT_DIR"
echo "run_id=$RUN_ID"
echo "num_workers=$NUM_WORKERS total_context=$TOTAL_CONTEXT"
echo "per_instance_call_limit=${PER_INSTANCE_CALL_LIMIT:-config-default}"

SERVE_CMD=(
  env
  "LD_LIBRARY_PATH=$LLAMA_DIR:$CUDA_LIB"
  proxy
  "$LLAMA_SERVER"
  -m "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  -c "$TOTAL_CONTEXT"
  -np 1
  -ngl auto
  -fa on
  --jinja
  --fit on
  --fit-target 4096
  --fit-ctx "$TOTAL_CONTEXT"
  -b 1024
  -ub 128
  -ctk f16
  -ctv f16
  --threads 16
  --threads-batch 32
  --ctx-checkpoints 0
  --reasoning-format deepseek
  --reasoning off
  --reasoning-budget 0
  --metrics
)

printf '%q ' "${SERVE_CMD[@]}" > "$OUT_DIR/serve_cmd.txt"
printf '\n' >> "$OUT_DIR/serve_cmd.txt"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping llama-server pid $SERVER_PID"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

echo "Starting llama-server on port $PORT"
setsid "${SERVE_CMD[@]}" > "$OUT_DIR/llama-server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$OUT_DIR/llama-server.pid"

echo "Waiting for llama-server readiness"
READY=0
for _ in {1..600}; do
  MODELS_JSON=""
  if MODELS_JSON=$(curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null) &&
    [[ "$MODELS_JSON" == *"$MODEL"* ]]; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "llama-server exited during startup"
    tail -n 200 "$OUT_DIR/llama-server.log" || true
    exit 1
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "llama-server did not become ready"
  tail -n 200 "$OUT_DIR/llama-server.log" || true
  exit 1
fi
echo "llama-server ready"

if [[ "$RUN_PROTOCOL_SMOKE" == 1 ]]; then
  echo "Running API and tool-call protocol smoke test"
  "$PYTHON" "$REPO/tools/mimo_v25_api_smoke.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$SERVED_NAME" \
    --output "$OUT_DIR/api_smoke.json" \
    --max-tokens 8192 \
    --reasoning forbidden
fi

SLICE_ARGS=()
if [[ -n "${SWE_SLICE:-}" ]]; then
  SLICE_ARGS=(--instances.slice "$SWE_SLICE")
  echo "Using SWE slice: $SWE_SLICE"
fi

MODEL_ARGS=()
if [[ -n "$PER_INSTANCE_CALL_LIMIT" ]]; then
  MODEL_ARGS=(--agent.model.per_instance_call_limit "$PER_INSTANCE_CALL_LIMIT")
fi

echo "Starting SWE-agent batch"
"$SWEAGENT" run-batch \
  --config "$REPO/tools/sweagent-rtxpro6000-deepseek-v4-flash-0731-iq2-xxs.yaml" \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  "${SLICE_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --output_dir "$OUT_DIR" \
  --num_workers "$NUM_WORKERS"

echo "SWE-agent batch complete"
date -Is

if [[ -f "$OUT_DIR/preds.json" ]]; then
  stop_server
  trap - EXIT
  echo "Starting SWE-bench harness evaluation"
  "$PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --predictions_path "$OUT_DIR/preds.json" \
    --run_id "$RUN_ID" \
    --max_workers "$EVAL_WORKERS" \
    --cache_level instance \
    --report_dir "$OUT_DIR/eval"

  HARNESS_REPORT="$REPO/${OUT_DIR##*/}.$RUN_ID.json"
  if [[ -f "$HARNESS_REPORT" ]]; then
    mkdir -p "$OUT_DIR/eval"
    cp "$HARNESS_REPORT" "$OUT_DIR/eval/$RUN_ID.json"
  fi
else
  echo "No preds.json found; skipping evaluation"
fi

echo "Done"
date -Is
