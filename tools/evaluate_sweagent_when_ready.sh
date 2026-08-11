#!/usr/bin/env bash
set -euo pipefail

AGENT_PID="${1:?SWE-agent PID is required}"
OUT_DIR="${2:?Output directory is required}"
RUN_ID="${3:?Run id is required}"
EXPECTED_PREDICTIONS="${4:?Expected prediction count is required}"
SERVER_WORKSPACE="${5:?Server workspace id is required}"
AGENT_WORKSPACE="${6:?Agent workspace id is required}"
OUTPUT_NAME="$(basename "${OUT_DIR}")"
ROOT_REPORT="/home/gisenberg/${OUTPUT_NAME}.${RUN_ID}.json"

echo "Waiting for SWE-agent PID ${AGENT_PID} to finish"
while [[ -d "/proc/${AGENT_PID}" ]]; do
  sleep 60
done

if [[ ! -f "${OUT_DIR}/preds.json" ]]; then
  echo "SWE-agent exited without preds.json; leaving workspaces available for inspection"
  exit 1
fi

PRED_COUNT="$(jq 'length' "${OUT_DIR}/preds.json")"
if [[ "${PRED_COUNT}" != "${EXPECTED_PREDICTIONS}" ]]; then
  echo "Expected ${EXPECTED_PREDICTIONS} predictions but found ${PRED_COUNT}"
  exit 1
fi

mkdir -p "${OUT_DIR}/eval"
echo "Starting official SWE-bench evaluation for ${PRED_COUNT} predictions"
/home/gisenberg/.micromamba/envs/cuda/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --split test \
  --predictions_path "${OUT_DIR}/preds.json" \
  --run_id "${RUN_ID}" \
  --max_workers 4 \
  --cache_level instance \
  --report_dir "${OUT_DIR}/eval"

if [[ ! -f "${ROOT_REPORT}" ]]; then
  echo "Official evaluation finished without the expected report ${ROOT_REPORT}"
  exit 1
fi

cp "${ROOT_REPORT}" "${OUT_DIR}/eval/${RUN_ID}.json"
RESOLVED="$(jq '.resolved_ids | length' "${ROOT_REPORT}")"
echo "Official evaluation complete: ${RESOLVED}/${PRED_COUNT} resolved"
wmux-notify \
  --title "DeepSeek V4 EXL3 full SWE-bench complete" \
  --subtitle "Official evaluation finished" \
  --body "Resolved ${RESOLVED}/${PRED_COUNT} SWE-bench Lite cases."
python3 /home/gisenberg/.agents/skills/wmux/scripts/wmuxctl.py finish \
  --workspace "${SERVER_WORKSPACE}" \
  --agent codex \
  --status completed \
  --summary "Full SWE-bench Lite evaluation completed at ${RESOLVED}/${PRED_COUNT}; model server stopped" \
  --close
python3 /home/gisenberg/.agents/skills/wmux/scripts/wmuxctl.py finish \
  --workspace "${AGENT_WORKSPACE}" \
  --agent codex \
  --status completed \
  --summary "Generated ${PRED_COUNT} predictions and completed official evaluation at ${RESOLVED}/${PRED_COUNT}" \
  --close
