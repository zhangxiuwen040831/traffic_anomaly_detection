#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-$PWD}"
CONFIG_PATH="${2:-config/stage3/base.yaml}"
OVERRIDE_PATH="${3:-config/stage3/cloud_gpu.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${WORKDIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

MODEL_NAME="${MODEL_NAME:-enhanced_mlp_ae}"
DATASET_NAME="${DATASET_NAME:-unsw_nb15}"
RUN_GROUP="${RUN_GROUP:-future_gpu_plan}"
RUN_NAME="${RUN_NAME:-${DATASET_NAME}_${MODEL_NAME}_gpu}"
SESSION_NAME="${SESSION_NAME:-stage3_train}"
LAUNCH_MODE="${LAUNCH_MODE:-direct}"

mkdir -p "${WORKDIR}/outputs_stage3/${RUN_GROUP}/launcher_logs"
LOG_FILE="${WORKDIR}/outputs_stage3/${RUN_GROUP}/launcher_logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

CMD="cd ${WORKDIR} && ${PYTHON_BIN} scripts/stage3_cli.py --config ${CONFIG_PATH} --override ${OVERRIDE_PATH} train --model ${MODEL_NAME} --dataset ${DATASET_NAME} --run-group ${RUN_GROUP} --run-name ${RUN_NAME} --resume"

echo "launch_mode=${LAUNCH_MODE}"
echo "log_file=${LOG_FILE}"
echo "python_bin=${PYTHON_BIN}"
echo "command=${CMD}"

case "${LAUNCH_MODE}" in
  direct)
    bash -lc "${CMD}" 2>&1 | tee "${LOG_FILE}"
    ;;
  nohup)
    nohup bash -lc "${CMD}" >"${LOG_FILE}" 2>&1 &
    echo "nohup started with pid=$!"
    ;;
  tmux)
    tmux new -d -s "${SESSION_NAME}" "bash -lc '${CMD} 2>&1 | tee ${LOG_FILE}'"
    echo "tmux session=${SESSION_NAME}"
    ;;
  screen)
    screen -dmS "${SESSION_NAME}" bash -lc "${CMD} 2>&1 | tee '${LOG_FILE}'"
    echo "screen session=${SESSION_NAME}"
    ;;
  *)
    echo "Unsupported launch mode: ${LAUNCH_MODE}"
    exit 1
    ;;
esac
