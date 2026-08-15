#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:qwen38}
MODEL_DIR=${MODEL_DIR:-/mnt/extended/gisenberg/models/qwen3.8-27b-fp8-017b9c7a}
VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-/mnt/extended/gisenberg/models/.vllm-cache-qwen38}
CONTAINER_NAME=${CONTAINER_NAME:-qwen38-27b-fp8}
SERVED_NAME=${SERVED_NAME:-qwen38-27b-fp8-mtp3}
PORT=${PORT:-8092}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-3}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model checkpoint at $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$VLLM_CACHE_DIR"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc host \
  --network host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR:/model:ro" \
  -v "$VLLM_CACHE_DIR:/root/.cache/vllm" \
  "$IMAGE" \
  /model \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"reasoning_effort":"xhigh","preserve_thinking":true}' \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS}"
