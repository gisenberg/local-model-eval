#!/usr/bin/env bash
set -euo pipefail

speculative_tokens=${MUSE_DFLASH_TOKENS:-15}
speculative_config=$(printf \
    '{"method":"dflash","model":"/models/muse-glimmer-30b-assistant","num_speculative_tokens":%d}' \
    "$speculative_tokens")

docker run \
    --name muse-glimmer-fp8-dflash \
    --gpus all \
    --ipc host \
    --publish 8092:8000 \
    --volume /home/gisenberg/models-vllm:/models:ro \
    local/vllm-muse-glimmer:dflash-native \
    /models/muse-glimmer-30b-fp8 \
    --served-model-name muse-glimmer-30b-fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.97 \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer \
    --generation-config auto \
    --speculative-config "$speculative_config"
