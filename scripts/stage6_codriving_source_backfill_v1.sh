#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
BACKEND=${BACKEND:-tvm}
GPU=${GPU:?GPU is required}
GROUPS=${GROUPS:?GROUPS is required, comma-separated group_ids}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}

cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller/source_backfill"
DONE="$FORMAL_ROOT/controller/source_backfill/gpu${GPU}.done"
FAILED="$FORMAL_ROOT/controller/source_backfill/gpu${GPU}.failed"
LOG="$FORMAL_ROOT/controller/source_backfill/gpu${GPU}.log"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$FAILED"; fi' EXIT

IFS=',' read -r -a ITEMS <<<"$GROUPS"
for group_id in "${ITEMS[@]}"; do
  [[ -n "$group_id" ]] || continue
  request=$(
    "$PY" - "$FORMAL_ROOT" "$BACKEND" "$group_id" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
backend = sys.argv[2]
group_id = sys.argv[3]
for path in sorted((root / backend / "compression_only").glob("batch_*/measurement_request.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(row.get("group_id") == group_id for row in data.get("rows") or []):
        print(path)
        raise SystemExit(0)
raise SystemExit(f"missing request for {group_id}")
PY
  )
  echo "$(date -Is) gpu=$GPU group=$group_id request=$request" >>"$LOG"
  "$REPO/scripts/stage5_materialize_round_sources_v1.sh" \
    --request "$request" --model codriving --group-id "$group_id" --gpu "$GPU" >>"$LOG" 2>&1
done

date -Is >"$DONE"
rm -f "$FAILED"
trap - EXIT
