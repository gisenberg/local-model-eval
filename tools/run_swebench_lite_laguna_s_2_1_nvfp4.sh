#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
VENV=${VENV:-/home/gisenberg/venvs/laguna-s-2.1-07614121-vllm026}
CUDA_HOME=/usr/local/cuda-13.0
SWEAGENT=/home/gisenberg/.micromamba/envs/cuda/bin/sweagent
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python

MODEL=${MODEL:-/mnt/extended/gisenberg/models/laguna-s-2.1-nvfp4-07614121}
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/extended/gisenberg/models/laguna-s-2.1-dflash-nvfp4-4cdcc6e9}
SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-7}
TOKENIZER=${TOKENIZER:-/mnt/extended/gisenberg/models/laguna-s-2.1-tokenizer-07614121-fix-regex}
SERVED_NAME=${SERVED_NAME:-poolside/Laguna-S-2.1-NVFP4}
PORT=${PORT:-8091}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
CPU_OFFLOAD_PARAMS=${CPU_OFFLOAD_PARAMS:-experts}
OFFLOAD_GROUP_SIZE=${OFFLOAD_GROUP_SIZE:-0}
OFFLOAD_NUM_IN_GROUP=${OFFLOAD_NUM_IN_GROUP:-1}
OFFLOAD_PREFETCH_STEP=${OFFLOAD_PREFETCH_STEP:-1}
ENABLE_DFLASH=${ENABLE_DFLASH:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
OUT_DIR=${OUT_DIR:-experiments/sweagent_lite_laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026}
RUN_ID=${RUN_ID:-sweagent_lite_laguna_s_2_1_nvfp4_07614121.laguna-s-2.1-nvfp4-dflash-4cdcc6e9-vllm026}
RUN_LIGHTWEIGHT_BENCH=${RUN_LIGHTWEIGHT_BENCH:-0}
LIGHTWEIGHT_OUT_DIR=${LIGHTWEIGHT_OUT_DIR:-experiments/laguna_s_2_1_nvfp4_reasoning_preserved}
LIGHTWEIGHT_RUNS=${LIGHTWEIGHT_RUNS:-3}
LIGHTWEIGHT_MAX_TOKENS=${LIGHTWEIGHT_MAX_TOKENS:-15000}

cd "$REPO"
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_swebench_lite.log") 2>&1

echo "=== Laguna-S-2.1-NVFP4 SWE-bench Lite run ==="
date -Is
echo "out_dir=$OUT_DIR"
echo "run_id=$RUN_ID"
echo "num_workers=$NUM_WORKERS max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"
echo "cpu_offload_gb=$CPU_OFFLOAD_GB offload_group_size=$OFFLOAD_GROUP_SIZE"
echo "cpu_offload_params=$CPU_OFFLOAD_PARAMS"
echo "enable_dflash=$ENABLE_DFLASH speculative_tokens=$SPECULATIVE_TOKENS"

SERVE_CMD=(
  env
  "PATH=$VENV/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "CUDA_HOME=$CUDA_HOME"
  "CUTE_DSL_ARCH=sm_120a"
  "CUTE_DSL_CACHE_DIR=$VENV/cache/cute-dsl"
  "FLASHINFER_WORKSPACE_BASE=$VENV/cache/flashinfer"
  "VLLM_CACHE_ROOT=$VENV/cache/vllm"
  "MAX_JOBS=${MAX_JOBS:-4}"
  "$VENV/bin/vllm" serve "$MODEL"
  --tokenizer "$TOKENIZER"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --reasoning-parser poolside_v1
  --tool-call-parser poolside_v1
  --enable-auto-tool-choice
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
  --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":20}'
  --enable-prefix-caching
)

if [[ "$ENABLE_DFLASH" == 1 ]]; then
  SERVE_CMD+=(
    --speculative-config "{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS,\"method\":\"dflash\"}"
  )
fi

if [[ "$CPU_OFFLOAD_GB" != 0 ]]; then
  SERVE_CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
  if [[ -n "$CPU_OFFLOAD_PARAMS" ]]; then
    SERVE_CMD+=(--cpu-offload-params "$CPU_OFFLOAD_PARAMS")
  fi
fi

if [[ "$OFFLOAD_GROUP_SIZE" != 0 ]]; then
  SERVE_CMD+=(
    --offload-group-size "$OFFLOAD_GROUP_SIZE"
    --offload-num-in-group "$OFFLOAD_NUM_IN_GROUP"
    --offload-prefetch-step "$OFFLOAD_PREFETCH_STEP"
  )
fi

printf '%q ' "${SERVE_CMD[@]}" > "$OUT_DIR/serve_cmd.txt"
printf '\n' >> "$OUT_DIR/serve_cmd.txt"

VLLM_PID=""
stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM pid $VLLM_PID"
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
}
trap stop_vllm EXIT

echo "Starting vLLM on port $PORT"
setsid "${SERVE_CMD[@]}" > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
printf '%s\n' "$VLLM_PID" > "$OUT_DIR/vllm.pid"

echo "Waiting for vLLM readiness..."
READY=0
for _ in {1..1800}; do
  MODELS_JSON=""
  if MODELS_JSON=$(curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null) &&
    [[ "$MODELS_JSON" == *"\"id\":\"$SERVED_NAME\""* ]]; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup"
    tail -n 200 "$OUT_DIR/vllm.log" || true
    exit 1
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "vLLM did not become ready"
  tail -n 200 "$OUT_DIR/vllm.log" || true
  exit 1
fi
echo "vLLM ready"

if [[ "$RUN_LIGHTWEIGHT_BENCH" == 1 ]]; then
  echo "Running Laguna agent protocol smoke test"
  "$PYTHON" tools/laguna_agent_protocol_smoke.py \
    --port "$PORT" \
    --served-name "$SERVED_NAME"

  echo "Running lightweight coding benchmark"
  "$PYTHON" tools/nvfp4_qwen36_27b_bench.py \
    --port "$PORT" \
    --served-name "$SERVED_NAME" \
    --output-dir "$LIGHTWEIGHT_OUT_DIR" \
    --temp 1.0 \
    --runs "$LIGHTWEIGHT_RUNS" \
    --max-tokens "$LIGHTWEIGHT_MAX_TOKENS"
fi

SLICE_ARGS=()
if [[ -n "${SWE_SLICE:-}" ]]; then
  SLICE_ARGS=(--instances.slice "$SWE_SLICE")
  echo "Using SWE slice: $SWE_SLICE"
fi

echo "Starting SWE-agent batch"
"$SWEAGENT" run-batch \
  --config tools/sweagent-rtxpro6000-laguna-s-2.1-nvfp4.yaml \
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

  HARNESS_REPORT="$REPO/${OUT_DIR##*/}.$RUN_ID.json"
  if [[ -f "$HARNESS_REPORT" ]]; then
    mkdir -p "$OUT_DIR/eval"
    cp "$HARNESS_REPORT" "$OUT_DIR/eval/$RUN_ID.json"
    cp "$HARNESS_REPORT" "$REPO/$RUN_ID.json"
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
