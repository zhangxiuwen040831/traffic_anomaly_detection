#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
DATASET_NAME="${2:-all}"

RAW_ROOT="${PROJECT_ROOT}/data/raw"
mkdir -p "${RAW_ROOT}/unsw_nb15" "${RAW_ROOT}/cic_ids2017"

download_with_fallback() {
  local target="$1"
  shift
  local urls=("$@")

  if [[ -f "${target}" ]]; then
    echo "[skip] ${target}"
    return 0
  fi

  mkdir -p "$(dirname "${target}")"
  for url in "${urls[@]}"; do
    if [[ -z "${url}" ]]; then
      continue
    fi
    echo "[try] ${url}"
    rm -f "${target}.part"
    if command -v curl >/dev/null 2>&1; then
      if curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 30 --output "${target}.part" "${url}"; then
        mv "${target}.part" "${target}"
        echo "[ok] ${target}"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -c -O "${target}.part" "${url}"; then
        mv "${target}.part" "${target}"
        echo "[ok] ${target}"
        return 0
      fi
    else
      echo "curl/wget unavailable"
      return 1
    fi
    rm -f "${target}.part"
  done

  echo "[fail] ${target}"
  return 1
}

extract_and_cleanup() {
  local archive="$1"
  local destination="$2"
  if [[ ! -f "${archive}" ]]; then
    return 0
  fi
  mkdir -p "${destination}"
  case "${archive}" in
    *.zip)
      unzip -oq "${archive}" -d "${destination}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${destination}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${destination}"
      ;;
    *)
      echo "[warn] unsupported archive type: ${archive}"
      return 0
      ;;
  esac
  rm -f "${archive}"
  echo "[cleanup] removed ${archive}"
}

download_unsw() {
  local dest_dir="${RAW_ROOT}/unsw_nb15"
  download_with_fallback "${dest_dir}/UNSW_NB15_training-set.csv" \
    "${UNSW_TRAIN_MIRROR:-https://gitcode.net/mirrors/defcom17/UNSW_NB15/raw/master/UNSW_NB15_training-set.csv}" \
    "https://raw.githubusercontent.com/defcom17/UNSW_NB15/master/UNSW_NB15_training-set.csv" \
    "https://github.com/defcom17/UNSW_NB15/raw/master/UNSW_NB15_training-set.csv"

  download_with_fallback "${dest_dir}/UNSW_NB15_testing-set.csv" \
    "${UNSW_TEST_MIRROR:-https://gitcode.net/mirrors/defcom17/UNSW_NB15/raw/master/UNSW_NB15_testing-set.csv}" \
    "https://raw.githubusercontent.com/defcom17/UNSW_NB15/master/UNSW_NB15_testing-set.csv" \
    "https://github.com/defcom17/UNSW_NB15/raw/master/UNSW_NB15_testing-set.csv"
}

download_cic_ids2017() {
  local dest_dir="${RAW_ROOT}/cic_ids2017"
  local archive="${dest_dir}/MachineLearningCSV.zip"
  download_with_fallback "${archive}" \
    "${CIC_IDS2017_ML_MIRROR:-}" \
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
  extract_and_cleanup "${archive}" "${dest_dir}"
}

case "${DATASET_NAME}" in
  unsw_nb15|unsw)
    download_unsw
    ;;
  cic_ids2017|cic)
    download_cic_ids2017
    ;;
  all)
    download_unsw
    download_cic_ids2017
    ;;
  *)
    echo "Unsupported dataset target: ${DATASET_NAME}"
    exit 1
    ;;
esac
