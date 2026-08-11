#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/gisenberg/git/gisenberg/local-model-eval}
OUT_DIR=${OUT_DIR:-experiments/sweagent_lite_muse_glimmer_30b_fp8_dflash15_c8}
RUN_ID=${RUN_ID:-muse-glimmer-30b-fp8-dflash15-c8-full300}
RUN_MATCH=${RUN_MATCH:-run_swebench_lite_muse_glimmer_fp8_dflash.sh}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:8092/v1/models}
EXPECTED_PREDICTIONS=${EXPECTED_PREDICTIONS:-300}
POLL_SECONDS=${POLL_SECONDS:-60}
STALL_SECONDS=${STALL_SECONDS:-900}
MILESTONE_SIZE=${MILESTONE_SIZE:-25}

cd "$REPO"
mkdir -p "$OUT_DIR"

STATUS_FILE="$OUT_DIR/monitor_status.json"
LOG_FILE="$OUT_DIR/monitor.log"
REPORT="$OUT_DIR/eval/$RUN_ID.json"
ROOT_REPORT="$REPO/${OUT_DIR##*/}.$RUN_ID.json"

last_prediction_count=-1
last_trajectory_bytes=-1
last_progress_epoch=$(date +%s)
last_milestone=-1
stall_notified=0
server_failure_notified=0

notify() {
  local title=$1
  local subtitle=$2
  local body=$3
  if command -v wmux-notify >/dev/null 2>&1; then
    wmux-notify --title "$title" --subtitle "$subtitle" --body "$body" || true
  fi
}

write_status() {
  local state=$1
  local instance_count=$2
  local prediction_count=$3
  local inflight_count=$4
  local trajectory_bytes=$5
  local server_ready=$6
  local runner_alive=$7
  local stalled_seconds=$8
  local temp_file="$STATUS_FILE.tmp"

  jq -n \
    --arg timestamp "$(date -Is)" \
    --arg state "$state" \
    --arg run_id "$RUN_ID" \
    --argjson expected_predictions "$EXPECTED_PREDICTIONS" \
    --argjson instance_count "$instance_count" \
    --argjson prediction_count "$prediction_count" \
    --argjson inflight_count "$inflight_count" \
    --argjson trajectory_bytes "$trajectory_bytes" \
    --argjson server_ready "$server_ready" \
    --argjson runner_alive "$runner_alive" \
    --argjson stalled_seconds "$stalled_seconds" \
    '{
      timestamp: $timestamp,
      state: $state,
      run_id: $run_id,
      expected_predictions: $expected_predictions,
      instance_directories: $instance_count,
      completed_predictions: $prediction_count,
      inflight_instances: $inflight_count,
      trajectory_bytes: $trajectory_bytes,
      server_ready: $server_ready,
      runner_alive: $runner_alive,
      seconds_since_progress: $stalled_seconds
    }' > "$temp_file"
  mv "$temp_file" "$STATUS_FILE"
}

notify \
  "SWE-bench Lite monitor started" \
  "Muse Glimmer FP8 DFlash, 8 workers" \
  "Watching prediction progress, server health, stalls, and official evaluation."

while true; do
  now=$(date +%s)
  instance_count=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d ! -name eval | wc -l)
  prediction_count=$(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.pred' | wc -l)
  if [[ -f "$OUT_DIR/preds.json" ]]; then
    prediction_count=$(jq 'length' "$OUT_DIR/preds.json")
  fi
  # Trajectories are rewritten while an instance is active, so they are not
  # guaranteed to be valid JSON at the instant this monitor reads them.
  # Aggregate bytes provide a stable progress signal without parsing a live
  # file or depending on its temporary serialization state.
  trajectory_bytes=$(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.traj' -printf '%s\n' |
    awk '{total += $1} END {print total + 0}')
  trajectory_bytes=${trajectory_bytes:-0}
  inflight_count=$((instance_count - prediction_count))
  if ((inflight_count < 0)); then
    inflight_count=0
  fi

  server_ready=false
  if curl -fsS --max-time 10 "$SERVER_URL" >/dev/null 2>&1; then
    server_ready=true
    server_failure_notified=0
  elif ((server_failure_notified == 0)); then
    notify \
      "SWE-bench model server unavailable" \
      "Muse Glimmer endpoint failed its health check" \
      "The monitor will continue polling while the run attempts recovery."
    server_failure_notified=1
  fi

  runner_alive=false
  if pgrep -f "$RUN_MATCH" >/dev/null 2>&1; then
    runner_alive=true
  fi

  if ((prediction_count > last_prediction_count || trajectory_bytes != last_trajectory_bytes)); then
    last_progress_epoch=$now
    stall_notified=0
  fi
  last_prediction_count=$prediction_count
  last_trajectory_bytes=$trajectory_bytes
  stalled_seconds=$((now - last_progress_epoch))

  state=generating
  if ((prediction_count >= EXPECTED_PREDICTIONS)); then
    state=evaluating
  fi
  if [[ -f "$REPORT" || -f "$ROOT_REPORT" ]]; then
    report_path=$REPORT
    if [[ ! -f "$report_path" ]]; then
      report_path=$ROOT_REPORT
    fi
    resolved=$(jq '.resolved_ids | length' "$report_path")
    write_status complete "$instance_count" "$prediction_count" "$inflight_count" \
      "$trajectory_bytes" "$server_ready" "$runner_alive" "$stalled_seconds"
    printf '%s state=complete predictions=%s resolved=%s trajectory_bytes=%s\n' \
      "$(date -Is)" "$prediction_count" "$resolved" "$trajectory_bytes" | tee -a "$LOG_FILE"
    notify \
      "SWE-bench Lite evaluation complete" \
      "Muse Glimmer resolved $resolved/$EXPECTED_PREDICTIONS" \
      "The official report is available in $OUT_DIR/eval."
    exit 0
  fi

  if ((prediction_count / MILESTONE_SIZE > last_milestone)); then
    last_milestone=$((prediction_count / MILESTONE_SIZE))
    if ((prediction_count > 0)); then
      notify \
        "SWE-bench Lite progress" \
        "$prediction_count/$EXPECTED_PREDICTIONS predictions complete" \
        "$inflight_count instances are currently in flight."
    fi
  fi

  if ((stalled_seconds >= STALL_SECONDS && stall_notified == 0)); then
    notify \
      "SWE-bench Lite may be stalled" \
      "No trajectory or prediction progress for $stalled_seconds seconds" \
      "The monitor is still running and will report recovery or termination."
    stall_notified=1
  fi

  if [[ "$runner_alive" == false ]]; then
    state=failed
    write_status "$state" "$instance_count" "$prediction_count" "$inflight_count" \
      "$trajectory_bytes" "$server_ready" "$runner_alive" "$stalled_seconds"
    printf '%s state=failed predictions=%s trajectory_bytes=%s reason=runner_exited\n' \
      "$(date -Is)" "$prediction_count" "$trajectory_bytes" | tee -a "$LOG_FILE"
    notify \
      "SWE-bench Lite run stopped" \
      "$prediction_count/$EXPECTED_PREDICTIONS predictions found" \
      "The benchmark process exited before an official evaluation report appeared."
    exit 1
  fi

  write_status "$state" "$instance_count" "$prediction_count" "$inflight_count" \
    "$trajectory_bytes" "$server_ready" "$runner_alive" "$stalled_seconds"
  printf '%s state=%s predictions=%s/%s inflight=%s trajectory_bytes=%s server_ready=%s stalled_seconds=%s\n' \
    "$(date -Is)" "$state" "$prediction_count" "$EXPECTED_PREDICTIONS" "$inflight_count" \
    "$trajectory_bytes" "$server_ready" "$stalled_seconds" | tee -a "$LOG_FILE"
  sleep "$POLL_SECONDS"
done
