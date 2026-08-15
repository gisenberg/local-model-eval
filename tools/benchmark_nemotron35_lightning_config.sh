#!/usr/bin/env bash
set -euo pipefail

mode=${1:-dspark}
speculative_tokens=${2:-3}
run_quality=${RUN_QUALITY:-0}

repo=/home/gisenberg/git/gisenberg/local-model-eval
served_name=${SERVED_NAME:-nemotron-3.5-lightning-30b-a3b-nvfp4}
port=${PORT:-8092}
container_name=${CONTAINER_NAME:-nemotron35-lightning}
python_bin=${PYTHON_BIN:-/home/gisenberg/.micromamba/envs/cuda/bin/python}

if [[ "$mode" == base ]]; then
    slug=base
else
    slug="${mode}${speculative_tokens}"
fi
output_dir=${OUTPUT_DIR:-$repo/experiments/nemotron35_lightning_nvfp4/$slug}
mkdir -p "$output_dir"

cd "$repo"
date -Is | tee "$output_dir/started_at.txt"
tools/run_nemotron35_lightning_server.sh "$mode" "$speculative_tokens" \
    | tee "$output_dir/launch.txt"

ready=0
for attempt in $(seq 1 90); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" \
        | jq -e --arg model "$served_name" \
            '.data[] | select(.id == $model)' >/dev/null; then
        ready=1
        break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != true ]]; then
        docker logs "$container_name" >"$output_dir/server.log" 2>&1 || true
        echo "Server container exited during startup" >&2
        exit 1
    fi
    sleep 10
done
if [[ "$ready" != 1 ]]; then
    docker logs "$container_name" >"$output_dir/server.log" 2>&1 || true
    echo "Server did not become ready in 15 minutes" >&2
    exit 1
fi

docker logs "$container_name" >"$output_dir/server.log" 2>&1
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,power.draw \
    --format=csv,noheader >"$output_dir/gpu_ready.csv"

"$python_bin" tools/mimo_v25_api_smoke.py \
    --base-url "http://127.0.0.1:$port/v1" \
    --model "$served_name" \
    --output "$output_dir/api_smoke.json" \
    --max-tokens 4096 \
    --reasoning optional

"$python_bin" tools/muse_glimmer_fp8_bench.py \
    --base-url "http://127.0.0.1:$port/v1" \
    --model "$served_name" \
    --reasoning-strength high \
    --top-k -1 \
    --warmups 1 \
    --runs 3 \
    --throughput-tokens 512 \
    --skip-coding \
    --output "$output_dir/throughput"

"$python_bin" tools/muse_glimmer_concurrency_bench.py \
    --base-url "http://127.0.0.1:$port/v1" \
    --model "$served_name" \
    --reasoning-strength high \
    --top-k -1 \
    --concurrency 1,2,4,8,16 \
    --trials 2 \
    --max-tokens 512 \
    --output "$output_dir/concurrency.json"

if [[ "$run_quality" == 1 ]]; then
    "$python_bin" tools/muse_glimmer_fp8_bench.py \
        --base-url "http://127.0.0.1:$port/v1" \
        --model "$served_name" \
        --reasoning-strength high \
        --top-k -1 \
        --coding-tokens 16384 \
        --skip-throughput \
        --output "$output_dir/quality"
fi

docker logs "$container_name" >"$output_dir/server.log" 2>&1
date -Is | tee "$output_dir/completed_at.txt"
