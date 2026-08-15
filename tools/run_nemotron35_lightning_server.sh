#!/usr/bin/env bash
set -euo pipefail

mode=${1:-dspark}
speculative_tokens=${2:-3}

model_root=${MODEL_ROOT:-/mnt/extended/gisenberg/models}
cache_root=${VLLM_CACHE_ROOT:-/mnt/extended/gisenberg/models/.vllm-cache-nemotron35}
target_model=${TARGET_MODEL:-$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-0dcd680e}
dspark_model=${DSPARK_MODEL:-$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-dspark-d10c6ff4}
dflash_model=${DFLASH_MODEL:-$model_root/nemotron-3.5-lightning-30b-a3b-nvfp4-dflash-7fc1f1ff}

image=${VLLM_IMAGE:-local/vllm-nemotron35:v0.27.1}
container_name=${CONTAINER_NAME:-nemotron35-lightning}
served_name=${SERVED_NAME:-nemotron-3.5-lightning-30b-a3b-nvfp4}
port=${PORT:-8092}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.94}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_seqs=${MAX_NUM_SEQS:-16}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}

speculative_args=()
case "$mode" in
    base)
        ;;
    mtp)
        speculative_args=(
            --speculative-config
            "{\"method\":\"mtp\",\"num_speculative_tokens\":$speculative_tokens}"
        )
        ;;
    dspark)
        speculative_args=(
            --speculative-config
            "{\"method\":\"dspark\",\"model\":\"/models/$(basename "$dspark_model")\",\"num_speculative_tokens\":$speculative_tokens}"
        )
        ;;
    dflash)
        speculative_args=(
            --speculative-config
            "{\"method\":\"dflash\",\"model\":\"/models/$(basename "$dflash_model")\",\"num_speculative_tokens\":$speculative_tokens}"
        )
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        exit 2
        ;;
esac

for path in "$target_model"; do
    if [[ ! -f "$path/config.json" ]]; then
        echo "Model is incomplete or missing: $path" >&2
        exit 1
    fi
done
if [[ "$mode" == dspark && ! -f "$dspark_model/config.json" ]]; then
    echo "DSpark model is incomplete or missing: $dspark_model" >&2
    exit 1
fi
if [[ "$mode" == dflash && ! -f "$dflash_model/config.json" ]]; then
    echo "DFlash model is incomplete or missing: $dflash_model" >&2
    exit 1
fi

docker rm -f "$container_name" >/dev/null 2>&1 || true
mkdir -p "$cache_root"

command=(
    docker run --detach
    --name "$container_name"
    --gpus all
    --ipc host
    --publish "$port:8000"
    --volume "$model_root:/models:ro"
    --volume "$cache_root:/root/.cache/vllm"
    "$image"
    "/models/$(basename "$target_model")"
    --served-model-name "$served_name"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
    --max-num-seqs "$max_num_seqs"
    --max-num-batched-tokens "$max_num_batched_tokens"
    --kv-cache-dtype fp8
    --enable-prefix-caching
    --async-scheduling
    --moe-backend humming
    --mamba-backend flashinfer
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float16
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --reasoning-parser nemotron_v3
    --tool-call-parser qwen3_coder
    --enable-auto-tool-choice
    --generation-config auto
    "${speculative_args[@]}"
)

printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
