#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO=poolside/Laguna-S-2.1-NVFP4
TARGET_REVISION=07614121b31898586430f189d27a25a0be310843
DRAFT_REPO=poolside/Laguna-S-2.1-DFlash-NVFP4
DRAFT_REVISION=4cdcc6e9b29105e8ff5790885cadccbeb4f33f54

MODEL_ROOT=${MODEL_ROOT:-/mnt/extended/gisenberg/models}
TARGET_DIR=${TARGET_DIR:-"$MODEL_ROOT/laguna-s-2.1-nvfp4-07614121"}
DRAFT_DIR=${DRAFT_DIR:-"$MODEL_ROOT/laguna-s-2.1-dflash-nvfp4-4cdcc6e9"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$MODEL_ROOT/laguna-s-2.1-tokenizer-07614121-fix-regex"}
VENV=${VENV:-/home/gisenberg/venvs/laguna-s-2.1-07614121-vllm026}

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" -p 3.12
fi

uv pip install -p "$VENV/bin/python" \
  "vllm==0.26.0" \
  --torch-backend=cu130

uv pip install -p "$VENV/bin/python" \
  "flashinfer-python==0.6.15.dev20260712" \
  "flashinfer-cubin==0.6.15.dev20260712" \
  "flashinfer-jit-cache==0.6.15.dev20260712" \
  --extra-index-url https://flashinfer.ai/whl/nightly/ \
  --extra-index-url https://flashinfer.ai/whl/nightly/cu130/ \
  --index-strategy unsafe-best-match

hf download "$TARGET_REPO" \
  --revision "$TARGET_REVISION" \
  --local-dir "$TARGET_DIR"

hf download "$DRAFT_REPO" \
  --revision "$DRAFT_REVISION" \
  --local-dir "$DRAFT_DIR"

mkdir -p "$TOKENIZER_DIR"
cp \
  "$TARGET_DIR/tokenizer.json" \
  "$TARGET_DIR/special_tokens_map.json" \
  "$TARGET_DIR/chat_template.jinja" \
  "$TOKENIZER_DIR/"
jq '.fix_mistral_regex = true' \
  "$TARGET_DIR/tokenizer_config.json" > "$TOKENIZER_DIR/tokenizer_config.json.tmp"
mv \
  "$TOKENIZER_DIR/tokenizer_config.json.tmp" \
  "$TOKENIZER_DIR/tokenizer_config.json"

echo "target=$TARGET_DIR"
echo "target_revision=$TARGET_REVISION"
echo "draft=$DRAFT_DIR"
echo "draft_revision=$DRAFT_REVISION"
echo "tokenizer=$TOKENIZER_DIR"
echo "venv=$VENV"
