#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
VENV=${VENV:-/home/gisenberg/venvs/laguna-s-2.1-07614121-vllm026}
CLIENT_PYTHON=${CLIENT_PYTHON:-/home/gisenberg/.micromamba/envs/cuda/bin/python}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}

TARGET_REPO=poolside/Laguna-S-2.1-NVFP4
TARGET_REVISION=07614121b31898586430f189d27a25a0be310843
TARGET_DIR=${TARGET_DIR:-/mnt/extended/gisenberg/models/laguna-s-2.1-nvfp4-07614121}
DRAFT_REPO=poolside/Laguna-S-2.1-DFlash-NVFP4
DRAFT_REVISION=4cdcc6e9b29105e8ff5790885cadccbeb4f33f54
DRAFT_DIR=${DRAFT_DIR:-/mnt/extended/gisenberg/models/laguna-s-2.1-dflash-nvfp4-4cdcc6e9}
SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-7}
TOKENIZER_DIR=${TOKENIZER_DIR:-/mnt/extended/gisenberg/models/laguna-s-2.1-tokenizer-07614121-fix-regex}

SERVED_NAME=poolside/Laguna-S-2.1-NVFP4
PORT=${PORT:-8091}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
CPU_OFFLOAD_PARAMS=${CPU_OFFLOAD_PARAMS:-experts}
OFFLOAD_GROUP_SIZE=${OFFLOAD_GROUP_SIZE:-0}
OFFLOAD_NUM_IN_GROUP=${OFFLOAD_NUM_IN_GROUP:-1}
OFFLOAD_PREFETCH_STEP=${OFFLOAD_PREFETCH_STEP:-1}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_DFLASH=${RUN_DFLASH:-1}
RUN_QUALITY=${RUN_QUALITY:-1}
PROTOCOL_REASONING_MAX_TOKENS=${PROTOCOL_REASONING_MAX_TOKENS:-4096}
PROTOCOL_TOOL_MAX_TOKENS=${PROTOCOL_TOOL_MAX_TOKENS:-2048}
PERF_MAX_TOKENS=${PERF_MAX_TOKENS:-1024}
QUALITY_RUNS=${QUALITY_RUNS:-3}
QUALITY_MAX_TOKENS=${QUALITY_MAX_TOKENS:-15000}
OUT_DIR=${OUT_DIR:-experiments/laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026}

cd "$REPO"
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

date -Is
echo "target_repo=$TARGET_REPO"
echo "target_revision=$TARGET_REVISION"
echo "target_dir=$TARGET_DIR"
echo "draft_repo=$DRAFT_REPO"
echo "draft_revision=$DRAFT_REVISION"
echo "draft_dir=$DRAFT_DIR"
echo "tokenizer_dir=$TOKENIZER_DIR"
echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
echo "cpu_offload_params=$CPU_OFFLOAD_PARAMS"
echo "offload_group_size=$OFFLOAD_GROUP_SIZE"
echo "offload_num_in_group=$OFFLOAD_NUM_IN_GROUP"

nvidia-smi -q > "$OUT_DIR/nvidia-smi-q.txt"
uv pip freeze -p "$VENV/bin/python" > "$OUT_DIR/environment.txt"
if [[ ! -s "$OUT_DIR/artifact-sha256.txt" ]]; then
  sha256sum \
    "$TARGET_DIR/config.json" \
    "$TARGET_DIR/generation_config.json" \
    "$TARGET_DIR/chat_template.jinja" \
    "$TARGET_DIR/tokenizer.json" \
    "$TARGET_DIR/model.safetensors.index.json" \
    "$TARGET_DIR"/model-*.safetensors \
    "$DRAFT_DIR/config.json" \
    "$DRAFT_DIR/model.safetensors" \
    "$TOKENIZER_DIR/tokenizer.json" \
    "$TOKENIZER_DIR/tokenizer_config.json" \
    "$TOKENIZER_DIR/special_tokens_map.json" \
    "$TOKENIZER_DIR/chat_template.jinja" > "$OUT_DIR/artifact-sha256.txt"
fi
find "$TARGET_DIR" "$DRAFT_DIR" "$TOKENIZER_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%p\t%s\n' > "$OUT_DIR/artifact-files.tsv"

VLLM_PID=""
stop_server() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
    wait "$VLLM_PID" || true
  fi
  VLLM_PID=""
}
trap stop_server EXIT

wait_for_server() {
  local log_file=$1
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      tail -n 200 "$log_file"
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM"
  tail -n 200 "$log_file"
  return 1
}

start_server() {
  local mode=$1
  local log_file="$OUT_DIR/vllm-$mode.log"
  local -a serve_cmd=(
    env
    "PATH=$VENV/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "CUDA_HOME=$CUDA_HOME"
    "CUTE_DSL_ARCH=sm_120a"
    "CUTE_DSL_CACHE_DIR=$VENV/cache/cute-dsl"
    "FLASHINFER_WORKSPACE_BASE=$VENV/cache/flashinfer"
    "VLLM_CACHE_ROOT=$VENV/cache/vllm"
    "MAX_JOBS=4"
    "$VENV/bin/vllm" serve "$TARGET_DIR"
    --tokenizer "$TOKENIZER_DIR"
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

  if [[ "$mode" == "dflash" ]]; then
    serve_cmd+=(
      --speculative-config
      "{\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS,\"method\":\"dflash\"}"
    )
  fi

  if [[ "$CPU_OFFLOAD_GB" != 0 ]]; then
    serve_cmd+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
    if [[ -n "$CPU_OFFLOAD_PARAMS" ]]; then
      serve_cmd+=(--cpu-offload-params "$CPU_OFFLOAD_PARAMS")
    fi
  fi

  if [[ "$OFFLOAD_GROUP_SIZE" != 0 ]]; then
    serve_cmd+=(
      --offload-group-size "$OFFLOAD_GROUP_SIZE"
      --offload-num-in-group "$OFFLOAD_NUM_IN_GROUP"
      --offload-prefetch-step "$OFFLOAD_PREFETCH_STEP"
    )
  fi

  printf '%q ' "${serve_cmd[@]}" > "$OUT_DIR/serve-$mode.txt"
  printf '\n' >> "$OUT_DIR/serve-$mode.txt"
  "${serve_cmd[@]}" > "$log_file" 2>&1 &
  VLLM_PID=$!
  wait_for_server "$log_file"
  curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUT_DIR/models-$mode.json"
  nvidia-smi \
    --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader > "$OUT_DIR/gpu-idle-$mode.csv"
}

run_protocol_and_perf() {
  local mode=$1
  "$VENV/bin/python" tools/laguna_agent_protocol_smoke.py \
    --port "$PORT" \
    --served-name "$SERVED_NAME" \
    --reasoning-max-tokens "$PROTOCOL_REASONING_MAX_TOKENS" \
    --tool-max-tokens "$PROTOCOL_TOOL_MAX_TOKENS" \
    --raw-output "$OUT_DIR/protocol-$mode-raw.json" \
    > "$OUT_DIR/protocol-$mode.json"
  "$VENV/bin/python" tools/laguna_api_perf.py \
    --port "$PORT" \
    --served-name "$SERVED_NAME" \
    --mode "$mode" \
    --max-tokens "$PERF_MAX_TOKENS" \
    --output "$OUT_DIR/perf-$mode.json"
}

if [[ "$RUN_BASELINE" == 1 ]]; then
  start_server baseline
  run_protocol_and_perf baseline
  stop_server
fi

if [[ "$RUN_DFLASH" == 1 ]]; then
  start_server dflash
  run_protocol_and_perf dflash
  if [[ "$RUN_QUALITY" == 1 ]]; then
    "$CLIENT_PYTHON" tools/nvfp4_qwen36_27b_bench.py \
      --port "$PORT" \
      --served-name "$SERVED_NAME" \
      --output-dir "$OUT_DIR/quality-dflash" \
      --temp 1.0 \
      --runs "$QUALITY_RUNS" \
      --max-tokens "$QUALITY_MAX_TOKENS"
  fi
  stop_server
fi

date -Is
echo "Evaluation complete: $OUT_DIR"
