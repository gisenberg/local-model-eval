#!/usr/bin/env bash
set -uo pipefail

REPO=/home/gisenberg/git/gisenberg/local-model-eval
LLAMA_SRC=/home/gisenberg/llama-build/src-mimo-v25-ea63b4d
LLAMA_DIR="$LLAMA_SRC/build/bin"
LLAMA_SERVER="$LLAMA_DIR/llama-server"
CUDA_LIB=/home/gisenberg/.micromamba/envs/cuda/lib
PYTHON=/home/gisenberg/.micromamba/envs/cuda/bin/python
MODEL=/mnt/extended/gisenberg/models/mimo-v2.5-ud-iq2-xxs-f7aff786/UD-IQ2_XXS/MiMo-V2.5-UD-IQ2_XXS-00001-of-00003.gguf
SERVED_NAME=mimo-v2.5-iq2-xxs
PORT=${PORT:-8092}
TOTAL_CONTEXT=${TOTAL_CONTEXT:-262144}
TARGET_PREFIX_TOKENS=${TARGET_PREFIX_TOKENS:-100000}
REQUESTS=${REQUESTS:-10}
MAX_TOKENS=${MAX_TOKENS:-4096}
OUT_DIR=${OUT_DIR:-experiments/mimo_v25_cuda_isolation_59584_ea63b4d}
VARIANT_SET=${VARIANT_SET:-all}
CACHE_TYPE_OVERRIDE=${CACHE_TYPE_OVERRIDE:-}

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

echo "=== MiMo V2.5 CUDA isolation matrix ==="
date -Is
echo "out_dir=$OUT_DIR"
echo "target_prefix_tokens=$TARGET_PREFIX_TOKENS requests=$REQUESTS max_tokens=$MAX_TOKENS"
echo "variant_set=$VARIANT_SET"
nvidia-smi
uname -a
git -C "$LLAMA_SRC" rev-parse HEAD
cat /proc/driver/nvidia/version

SERVER_PID=""
TELEMETRY_PID=""

stop_processes() {
  if [[ -n "$TELEMETRY_PID" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""

  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_processes EXIT INT TERM

run_variant() {
  local variant=$1
  local flash_attn=$2
  local ubatch=$3
  local cache_type=q8_0
  local variant_dir="$OUT_DIR/$variant"
  local started
  local ready=0
  local client_rc
  local server_alive
  local xid_count

  if [[ "$flash_attn" == "off" ]]; then
    cache_type=f16
  fi
  if [[ -n "$CACHE_TYPE_OVERRIDE" ]]; then
    cache_type="$CACHE_TYPE_OVERRIDE"
  fi

  mkdir -p "$variant_dir"
  started=$(date -Is)
  echo "--- variant=$variant flash_attn=$flash_attn ubatch=$ubatch cache_type=$cache_type ---"
  echo "$started" > "$variant_dir/started_at.txt"

  local serve_cmd=(
    env
    "LD_LIBRARY_PATH=$LLAMA_DIR:$CUDA_LIB"
    "$LLAMA_SERVER"
    -m "$MODEL"
    --host 127.0.0.1
    --port "$PORT"
    -c "$TOTAL_CONTEXT"
    -np 1
    -ngl auto
    -fa "$flash_attn"
    --jinja
    --fit on
    --fit-target 4096
    --fit-ctx "$TOTAL_CONTEXT"
    -b 1024
    -ub "$ubatch"
    -ctk "$cache_type"
    -ctv "$cache_type"
    --threads 16
    --threads-batch 32
    --reasoning-format deepseek
    --reasoning-preserve
    --reasoning-budget 4096
    --metrics
  )

  printf '%q ' "${serve_cmd[@]}" > "$variant_dir/serve_cmd.txt"
  printf '\n' >> "$variant_dir/serve_cmd.txt"

  setsid "${serve_cmd[@]}" > "$variant_dir/server.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$variant_dir/server.pid"

  (
    echo "timestamp,power_w,temperature_c,gpu_util_pct,memory_used_mib,graphics_clock_mhz,memory_clock_mhz"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.current.graphics,clocks.current.memory \
        --format=csv,noheader,nounits || break
      sleep 1
    done
  ) > "$variant_dir/gpu_telemetry.csv" 2>&1 &
  TELEMETRY_PID=$!

  for _ in {1..180}; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 2
  done

  if [[ "$ready" != 1 ]]; then
    echo "variant=$variant server_start_failed"
    echo "server_start_failed" > "$variant_dir/result.txt"
    journalctl -k --since "$started" --no-pager > "$variant_dir/kernel.log"
    stop_processes
    return 0
  fi

  echo "variant=$variant server_ready"
  "$PYTHON" "$REPO/tools/mimo_v25_cuda_decode_probe.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED_NAME" \
    --target-prefix-tokens "$TARGET_PREFIX_TOKENS" \
    --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --variant "$variant" \
    --output "$variant_dir/probe.json"
  client_rc=$?

  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    server_alive=1
  else
    server_alive=0
  fi

  journalctl -k --since "$started" --no-pager > "$variant_dir/kernel.log"
  xid_count=$(grep -Ec 'NVRM:.*Xid' "$variant_dir/kernel.log" || true)
  printf 'client_rc=%s\nserver_alive=%s\nxid_count=%s\n' \
    "$client_rc" "$server_alive" "${xid_count:-0}" > "$variant_dir/result.txt"
  echo "variant=$variant client_rc=$client_rc server_alive=$server_alive xid_count=${xid_count:-0}"

  stop_processes
  sleep 5
  if ! nvidia-smi > "$variant_dir/nvidia_smi_after.txt" 2>&1; then
    echo "variant=$variant gpu_unavailable_stopping_matrix"
    return 1
  fi
  return 0
}

case "$VARIANT_SET" in
  all)
    run_variant fa_on_ub512 on 512 || exit 1
    run_variant fa_on_ub128 on 128 || exit 1
    run_variant fa_off_f16kv_ub512 off 512 || exit 1
    run_variant fa_off_f16kv_ub128 off 128 || exit 1
    ;;
  flash_on)
    run_variant fa_on_ub512 on 512 || exit 1
    run_variant fa_on_ub128 on 128 || exit 1
    ;;
  flash_off)
    run_variant fa_off_f16kv_ub512 off 512 || exit 1
    run_variant fa_off_f16kv_ub128 off 128 || exit 1
    ;;
  flash_on_f16)
    CACHE_TYPE_OVERRIDE=f16
    run_variant fa_on_f16kv_ub128 on 128 || exit 1
    ;;
  *)
    echo "Unknown VARIANT_SET=$VARIANT_SET"
    exit 2
    ;;
esac

echo "=== matrix complete ==="
for result in "$OUT_DIR"/*/result.txt; do
  echo "$result"
  cat "$result"
done
date -Is
