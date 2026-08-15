#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:qwen38-27b}
MODEL_DIR=${MODEL_DIR:-/mnt/extended/gisenberg/models/qwen3.8-27b-fp8-017b9c7a}
CACHE_DIR=${CACHE_DIR:-/mnt/extended/gisenberg/models/.sglang-cache-qwen38}
CONTAINER_NAME=${CONTAINER_NAME:-qwen38-27b-fp8-sglang}
SERVED_NAME=${SERVED_NAME:-qwen38-27b-fp8-sglang-mtp4}
PORT=${PORT:-8092}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model checkpoint at $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc host \
  --network host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/root/.cache" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --context-length "$CONTEXT_LENGTH" \
  --language-only \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"reasoning_effort":"medium","preserve_thinking":true}' \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --stream-response-default-include-usage \
  --enable-metrics
