#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
LLAMA_DIR=/home/gisenberg/llama-build/src-mimo-v25-ea63b4d/build/bin
LLAMA_SERVER="$LLAMA_DIR/llama-server"
CUDA_LIB=/home/gisenberg/.micromamba/envs/cuda/lib
SWEAGENT=/home/gisenberg/.micromamba/envs/cuda/bin/sweagent
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python

MODEL=${MODEL:-/mnt/extended/gisenberg/models/mimo-v2.5-ud-iq2-xxs-f7aff786/UD-IQ2_XXS/MiMo-V2.5-UD-IQ2_XXS-00001-of-00003.gguf}
SERVED_NAME=${SERVED_NAME:-mimo-v2.5-iq2-xxs}
PORT=${PORT:-8091}
TOTAL_CONTEXT=${TOTAL_CONTEXT:-262144}
PARALLEL_SLOTS=${PARALLEL_SLOTS:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
REASONING_BUDGET=${REASONING_BUDGET:-4096}
OUT_DIR=${OUT_DIR:-experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k}
RUN_ID=${RUN_ID:-sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k.mimo-v2.5-iq2-xxs-r4k-full}

cd "$REPO"
rtk mkdir -p "$OUT_DIR"
exec > >(rtk tee -a "$OUT_DIR/run_swebench_lite.log") 2>&1

rtk echo "=== MiMo-V2.5 UD-IQ2_XXS SWE-bench Lite run ==="
rtk date -Is
rtk echo "out_dir=$OUT_DIR"
rtk echo "run_id=$RUN_ID"
rtk echo "num_workers=$NUM_WORKERS parallel_slots=$PARALLEL_SLOTS total_context=$TOTAL_CONTEXT"
rtk echo "reasoning_budget=$REASONING_BUDGET"

SERVE_CMD=(
  rtk proxy env
  "LD_LIBRARY_PATH=$LLAMA_DIR:$CUDA_LIB"
  "$LLAMA_SERVER"
  -m "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  -c "$TOTAL_CONTEXT"
  -np "$PARALLEL_SLOTS"
  -ngl auto
  -fa on
  --jinja
  --fit on
  --fit-target 4096
  --fit-ctx "$TOTAL_CONTEXT"
  -b 1024
  -ub 512
  -ctk q8_0
  -ctv q8_0
  --threads 16
  --threads-batch 32
  --reasoning-format deepseek
  --reasoning-preserve
  --reasoning-budget "$REASONING_BUDGET"
  --metrics
)

rtk proxy printf '%q ' "${SERVE_CMD[@]}" > "$OUT_DIR/serve_cmd.txt"
rtk proxy printf '\n' >> "$OUT_DIR/serve_cmd.txt"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && rtk proxy kill -0 "$SERVER_PID" 2>/dev/null; then
    rtk echo "Stopping llama-server pid $SERVER_PID"
    rtk proxy kill -TERM -- "-$SERVER_PID" 2>/dev/null || rtk proxy kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      if ! rtk proxy kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      rtk sleep 2
    done
    rtk proxy kill -KILL -- "-$SERVER_PID" 2>/dev/null || rtk proxy kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

rtk echo "Starting llama-server on port $PORT"
rtk proxy setsid "${SERVE_CMD[@]}" > "$OUT_DIR/llama-server.log" 2>&1 &
SERVER_PID=$!
rtk proxy printf '%s\n' "$SERVER_PID" > "$OUT_DIR/llama-server.pid"

rtk echo "Waiting for llama-server readiness"
READY=0
for _ in {1..600}; do
  MODELS_JSON=""
  if MODELS_JSON=$(rtk curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null) &&
    [[ "$MODELS_JSON" == *"$MODEL"* ]]; then
    READY=1
    break
  fi
  if ! rtk proxy kill -0 "$SERVER_PID" 2>/dev/null; then
    rtk echo "llama-server exited during startup"
    rtk tail -n 200 "$OUT_DIR/llama-server.log" || true
    exit 1
  fi
  rtk sleep 2
done
if [[ "$READY" != 1 ]]; then
  rtk echo "llama-server did not become ready"
  rtk tail -n 200 "$OUT_DIR/llama-server.log" || true
  exit 1
fi
rtk echo "llama-server ready"

SLICE_ARGS=()
if [[ -n "${SWE_SLICE:-}" ]]; then
  SLICE_ARGS=(--instances.slice "$SWE_SLICE")
  rtk echo "Using SWE slice: $SWE_SLICE"
fi

rtk echo "Starting SWE-agent batch"
rtk "$SWEAGENT" run-batch \
  --config tools/sweagent-rtxpro6000-mimo-v2.5-iq2-xxs.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  "${SLICE_ARGS[@]}" \
  --output_dir "$OUT_DIR" \
  --num_workers "$NUM_WORKERS"

rtk echo "SWE-agent batch complete"
rtk date -Is

if [[ -f "$OUT_DIR/preds.json" ]]; then
  stop_server
  trap - EXIT
  rtk echo "Starting SWE-bench harness evaluation"
  rtk "$PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --predictions_path "$OUT_DIR/preds.json" \
    --run_id "$RUN_ID" \
    --max_workers "$NUM_WORKERS" \
    --cache_level instance \
    --report_dir "$OUT_DIR/eval"

  HARNESS_REPORT="$REPO/${OUT_DIR##*/}.$RUN_ID.json"
  if [[ -f "$HARNESS_REPORT" ]]; then
    rtk mkdir -p "$OUT_DIR/eval"
    rtk cp "$HARNESS_REPORT" "$OUT_DIR/eval/$RUN_ID.json"
    rtk cp "$HARNESS_REPORT" "$REPO/$RUN_ID.json"
  fi
else
  rtk echo "No preds.json found; skipping evaluation"
fi

rtk echo "Done"
rtk date -Is
