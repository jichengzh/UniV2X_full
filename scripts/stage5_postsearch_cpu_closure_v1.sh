#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
COLDSTART_ROOT=${COLDSTART_ROOT:-$REPO/results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1}
PROFILES=${PROFILES:-$REPO/results/s1_profile_final_v3_20260711/capability_profiles_v3.json}
REGISTRY=${REGISTRY:-$ROOT/candidate_source_registry_full.json}
CLOSURE_ROOT=${CLOSURE_ROOT:-$ROOT/closure_v3}
GRAPH_ROOT=${GRAPH_ROOT:-$CLOSURE_ROOT/graph_feature_audit_v1}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-60}
ROWS_SHA=${ROWS_SHA:-9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19}
GRAPHS_SHA=${GRAPHS_SHA:-c5f03e19daba4779cb187f036d7c4bf612d3479a00453149f4eaf213412536cd}

TERMINAL=$ROOT/controller/full_budget_scheduler_terminal.json
DONE=$ROOT/controller/stage5_postsearch_cpu_closure_v1.done
FAILED=$ROOT/controller/stage5_postsearch_cpu_closure_v1.failed

trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$FAILED"; fi' EXIT

while ! jq -e '
  .schema_version == "stage5_full_budget_scheduler_terminal_v3" and
  .status == "budget_exhausted" and .task_count == 4 and
  .round_count == 16 and .formal_online_genomes == 64
' "$TERMINAL" >/dev/null 2>&1; do
  sleep "$WAIT_INTERVAL_S"
done

cd "$REPO"
mkdir -p "$CLOSURE_ROOT" "$GRAPH_ROOT"

"$PY" scripts/stage5_finalize_full_budget_v3.py \
  --formal-root "$ROOT" \
  --coldstart-rows-json "$COLDSTART_ROOT/gold176_final.json" \
  --coldstart-graph-features-json "$COLDSTART_ROOT/graph_features.json" \
  --expected-coldstart-rows-sha256 "$ROWS_SHA" \
  --expected-coldstart-graph-features-sha256 "$GRAPHS_SHA" \
  --output-dir "$CLOSURE_ROOT"

"$PY" scripts/stage5_backfill_actual_graph_features_v1.py \
  --formal-root "$ROOT" --output-dir "$GRAPH_ROOT"

"$PY" scripts/stage5_graph_feature_replay_v1.py \
  --formal-root "$ROOT" \
  --coldstart-rows-json "$COLDSTART_ROOT/gold176_final.json" \
  --coldstart-graph-features-json "$COLDSTART_ROOT/graph_features.json" \
  --profiles-json "$PROFILES" --source-registry-json "$REGISTRY" \
  --actual-feedback-view-json \
    "$GRAPH_ROOT/stage5_online_feedback_actual_graph_view_v1.json" \
  --output-json \
    "$GRAPH_ROOT/stage5_graph_feature_teacher_forced_replay_v1.json"

date -Is >"$DONE"
rm -f "$FAILED"
trap - EXIT
