#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"

mkdir -p "${PROJECT_ROOT}/data/raw/unsw_nb15"
mkdir -p "${PROJECT_ROOT}/data/raw/cic_ids2017"
mkdir -p "${PROJECT_ROOT}/data/processed"

mkdir -p "${PROJECT_ROOT}/outputs_stage3/cloud_prepare/logs"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/cloud_prepare/checkpoints"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/cloud_prepare/figures"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/cloud_prepare/reports"

mkdir -p "${PROJECT_ROOT}/outputs_stage3/future_gpu_plan/logs"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/future_gpu_plan/checkpoints"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/future_gpu_plan/figures"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/future_gpu_plan/reports"

mkdir -p "${PROJECT_ROOT}/outputs_stage3/comparison"
mkdir -p "${PROJECT_ROOT}/outputs_stage3/final_report"

echo "Prepared layout under ${PROJECT_ROOT}"
