#!/usr/bin/env bash
set -euo pipefail

# Argv-safe production entrypoint.  The Python worker owns only immutable
# Stage7 lineage admission, actual-v3 executor delegation, and barrier checks.
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKER="$SCRIPT_DIR/stage7_core_round_worker_v2.py"
readonly PYTHON="${STAGE7_CONFIG_PYTHON:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}"

[[ -f "$WORKER" ]] || {
  echo "stage7 core v2 controller: round worker is missing" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || {
  echo "stage7 core v2 controller: configured Python is not executable" >&2
  exit 2
}

exec "$PYTHON" "$WORKER" "$@"
