#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-$PWD}"
VENV_DIR="${VENV_DIR:-${WORKDIR}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "[info] workdir=${WORKDIR}"
echo "[info] venv_dir=${VENV_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 is required but not found"
  exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
  echo "[info] pip3 missing, installing python3-pip and python3-venv"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
python -m pip install pandas scikit-learn matplotlib seaborn pyyaml tqdm requests paramiko

python - <<'PY'
import sys
try:
    import torch
    print(f"[info] python={sys.version.split()[0]}")
    print(f"[info] torch={torch.__version__}")
    print(f"[info] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] gpu={torch.cuda.get_device_name(0)}")
except Exception as exc:  # noqa: BLE001
    print(f"[warn] torch validation failed: {exc}")
PY
