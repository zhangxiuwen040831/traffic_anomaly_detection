#!/usr/bin/env bash
set -u

REPORT_DIR="${1:-outputs_stage3/cloud_prepare/reports}"
mkdir -p "${REPORT_DIR}"
REPORT_FILE="${REPORT_DIR}/environment_probe.txt"

{
  echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "hostname: $(hostname)"
  echo "pwd: $(pwd)"
  echo
  echo "[system]"
  uname -a 2>/dev/null || true
  echo
  echo "[python]"
  (python3 -V || python -V || echo "python_missing") 2>&1
  (pip3 -V || pip -V || echo "pip_missing") 2>&1
  echo
  echo "[cpu_memory_disk]"
  (nproc 2>/dev/null || echo "nproc_unavailable") 2>&1
  (free -h 2>/dev/null || head -n 5 /proc/meminfo || echo "free_unavailable") 2>&1
  (df -h 2>/dev/null || echo "df_unavailable") 2>&1
  echo
  echo "[cgroup_limits]"
  python3 - <<'PY'
from pathlib import Path

cpu_max = Path("/sys/fs/cgroup/cpu.max")
mem_max = Path("/sys/fs/cgroup/memory.max")

if cpu_max.exists():
    quota, period = cpu_max.read_text().strip().split()
    if quota == "max":
        print("cpu_limit: unlimited")
    else:
        cpu_limit = float(quota) / float(period)
        print(f"cpu_limit_cores: {cpu_limit:.2f}")
else:
    print("cpu_limit_cores: unavailable")

if mem_max.exists():
    raw = mem_max.read_text().strip()
    if raw == "max":
        print("memory_limit: unlimited")
    else:
        mem_gb = int(raw) / (1024 ** 3)
        print(f"memory_limit_gb: {mem_gb:.2f}")
else:
    print("memory_limit_gb: unavailable")
PY
  echo
  echo "[gpu]"
  (nvidia-smi 2>&1 || echo "nvidia-smi unavailable") 2>&1
  echo
  echo "[tools]"
  for tool in tmux screen nohup curl wget unzip tar git; do
    if command -v "${tool}" >/dev/null 2>&1; then
      echo "${tool}: $(command -v "${tool}")"
    else
      echo "${tool}: missing"
    fi
  done
  echo
  echo "[torch]"
  python3 - <<'PY'
try:
    import torch
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_probe_error: {exc}")
PY
} | tee "${REPORT_FILE}"
