#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
VENV=/home/gisenberg/venvs/dflash-pr40898
CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
SWEAGENT=/home/gisenberg/.micromamba/envs/cuda/bin/sweagent
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python

MODEL=${MODEL:-/mnt/extended/gisenberg/models/nemotron3-puzzle-75b-fp8}
SERVED_NAME=${SERVED_NAME:-nemotron3-puzzle-75b-fp8-mtp}
PORT=${PORT:-8091}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.94}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
NUM_WORKERS=${NUM_WORKERS:-4}
OUT_DIR=${OUT_DIR:-experiments/sweagent_lite_nemotron_puzzle}
RUN_ID=${RUN_ID:-sweagent_lite_nemotron_puzzle.nemotron3-puzzle-75b-fp8-mtp-full}

cd "$REPO"
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_swebench_lite.log") 2>&1

echo "=== Nemotron Puzzle SWE-bench Lite run ==="
date -Is
echo "out_dir=$OUT_DIR"
echo "run_id=$RUN_ID"
echo "num_workers=$NUM_WORKERS max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"

export PATH="$VENV/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export CUDA_HOME
export MAX_JOBS=${MAX_JOBS:-8}

VLLM_LOG="$OUT_DIR/vllm.log"
SERVE_CMD=(
  "$VENV/bin/vllm" serve "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --moe-backend auto
  --enable-expert-parallel
  --async-scheduling
  --trust-remote-code
  --mamba-backend flashinfer
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"
)

printf '%q ' "${SERVE_CMD[@]}" > "$OUT_DIR/serve_cmd.txt"
printf '\n' >> "$OUT_DIR/serve_cmd.txt"

VLLM_PID=""
stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM pid $VLLM_PID"
    kill -TERM "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    kill -KILL "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -KILL -f "$SERVED_NAME" 2>/dev/null || true
  pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null || true
}
trap stop_vllm EXIT

echo "Starting vLLM on port $PORT"
setsid "${SERVE_CMD[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$OUT_DIR/vllm.pid"

echo "Waiting for vLLM readiness..."
READY=0
for _ in {1..1800}; do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED_NAME"; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "vLLM did not become ready"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi
echo "vLLM ready"

SLICE_ARGS=()
if [[ -n "${SWE_SLICE:-}" ]]; then
  SLICE_ARGS=(--instances.slice "$SWE_SLICE")
  echo "Using SWE slice: $SWE_SLICE"
fi

echo "Starting SWE-agent batch"
"$SWEAGENT" run-batch \
  --config tools/sweagent-rtxpro6000-nemotron-puzzle.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  "${SLICE_ARGS[@]}" \
  --output_dir "$OUT_DIR" \
  --num_workers "$NUM_WORKERS"

echo "SWE-agent batch complete"
date -Is

if [[ -f "$OUT_DIR/preds.json" ]]; then
  stop_vllm
  trap - EXIT
  echo "Starting SWE-bench harness evaluation"
  "$PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --predictions_path "$OUT_DIR/preds.json" \
    --run_id "$RUN_ID" \
    --max_workers 4 \
    --cache_level instance \
    --report_dir "$OUT_DIR/eval"
  if [[ -f "$OUT_DIR/eval/$RUN_ID.json" ]]; then
    cp "$OUT_DIR/eval/$RUN_ID.json" "$REPO/$RUN_ID.json"
    "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$OUT_DIR/eval/$RUN_ID.json")
d = json.loads(p.read_text())
print(json.dumps({
    "total_instances": d.get("total_instances"),
    "submitted_instances": d.get("submitted_instances"),
    "resolved_instances": d.get("resolved_instances"),
    "unresolved_instances": d.get("unresolved_instances"),
    "empty_patch_instances": d.get("empty_patch_instances"),
}, indent=2))
PY
  fi
else
  echo "No preds.json found; skipping evaluation"
fi

echo "Done"
date -Is
