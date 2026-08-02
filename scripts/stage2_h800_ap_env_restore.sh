#!/usr/bin/env bash
set -euo pipefail

# Restore the H800 AP finetune/eval environment after files have been synced.
# This script contains no credentials and is intended to run on H800.

ENV_ARCHIVE="${ENV_ARCHIVE:-/exdata/jichengzhi/conda_envs/UniV2X_2.0_h800.tar.gz}"
ENV_DIR="${ENV_DIR:-/exdata/jichengzhi/conda_envs/UniV2X_2.0}"
ENV_LINK="${ENV_LINK:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0}"
HEAL_DIR="${HEAL_DIR:-/exdata/jichengzhi/heal_research/HEAL}"
HEAL_LINK_ROOT="${HEAL_LINK_ROOT:-/home/jichengzhi/heal_research}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/exdata/jichengzhi/heal_research/checkpoints}"
REPO_ROOT="${REPO_ROOT:-/home/jichengzhi/V2X}"
DATASET_TARGET="${DATASET_TARGET:-/exdata/jichengzhi/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure}"

echo "[restore] env archive: ${ENV_ARCHIVE}"
test -f "${ENV_ARCHIVE}"

mkdir -p "$(dirname "${ENV_DIR}")"
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  rm -rf "${ENV_DIR}.tmp"
  mkdir -p "${ENV_DIR}.tmp"
  tar -xzf "${ENV_ARCHIVE}" -C "${ENV_DIR}.tmp"
  rm -rf "${ENV_DIR}"
  mv "${ENV_DIR}.tmp" "${ENV_DIR}"
else
  echo "[restore] env already exists: ${ENV_DIR}"
fi

mkdir -p "$(dirname "${ENV_LINK}")"
if [ -e "${ENV_LINK}" ] && [ ! -L "${ENV_LINK}" ]; then
  echo "[restore] refusing to replace non-symlink ${ENV_LINK}" >&2
  exit 2
fi
ln -sfn "${ENV_DIR}" "${ENV_LINK}"

if [ ! -d "${HEAL_DIR}" ]; then
  echo "[restore] missing HEAL_DIR=${HEAL_DIR}; sync HEAL before running restore" >&2
  exit 2
fi
mkdir -p "${HEAL_LINK_ROOT}"
if [ -e "${HEAL_LINK_ROOT}/HEAL" ] && [ ! -L "${HEAL_LINK_ROOT}/HEAL" ]; then
  echo "[restore] refusing to replace non-symlink ${HEAL_LINK_ROOT}/HEAL" >&2
  exit 2
fi
ln -sfn "${HEAL_DIR}" "${HEAL_LINK_ROOT}/HEAL"

if [ -d "${CHECKPOINTS_DIR}" ]; then
  if [ -e "${HEAL_LINK_ROOT}/checkpoints" ] && [ ! -L "${HEAL_LINK_ROOT}/checkpoints" ]; then
    echo "[restore] refusing to replace non-symlink ${HEAL_LINK_ROOT}/checkpoints" >&2
    exit 2
  fi
  ln -sfn "${CHECKPOINTS_DIR}" "${HEAL_LINK_ROOT}/checkpoints"
else
  echo "[restore] missing CHECKPOINTS_DIR=${CHECKPOINTS_DIR}; sync checkpoints before running restore" >&2
  exit 2
fi

mkdir -p "${HEAL_DIR}/dataset/my_dair_v2x/v2x_c"
if [ -e "${HEAL_DIR}/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure" ] \
   && [ ! -L "${HEAL_DIR}/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure" ]; then
  echo "[restore] refusing to replace non-symlink dataset cooperative-vehicle-infrastructure" >&2
  exit 2
fi
ln -sfn "${DATASET_TARGET}" \
  "${HEAL_DIR}/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

PYTHONPATH="${HEAL_DIR}" "${ENV_LINK}/bin/python" - <<'PY'
import torch, numpy, yaml, opencood
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("yaml", yaml.__version__)
print("opencood", opencood.__file__)
PY

python3 "${REPO_ROOT}/scripts/stage2_h800_ap_env_preflight.py" \
  --json-out "${REPO_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/exports/h800_ap_env_preflight_latest.json"

echo "[restore] done"
