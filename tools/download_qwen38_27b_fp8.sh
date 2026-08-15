#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}
REVISION=${REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
MODEL_DIR=${MODEL_DIR:-/mnt/extended/gisenberg/models/qwen3.8-27b-fp8-017b9c7a}

mkdir -p "$MODEL_DIR"
exec hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$MODEL_DIR"
