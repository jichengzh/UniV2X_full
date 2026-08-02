#!/usr/bin/env bash
set -u

repo_root="/home/jichengzhi/V2X"
log_root="$repo_root/results/stage3_gold96_v3_20260711/performance_execution"
plan_root="$repo_root/results/stage3_gold96_performance_plans_v3_20260711"
python_bin="/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
gpu_pool="7"

mkdir -p "$log_root"
cd "$repo_root"

while true; do
  busy="$({ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || exit 1; } | awk -F, '$1 ~ /^[[:space:]]*7[[:space:]]*$/ {memory=$2+0; utilization=$3+0; if (memory>100 || utilization>10) n++} END{print n+0}')"
  printf '%s idle_wait busy=%s\n' "$(date -Is)" "$busy" >> "$log_root/orchestrator_progress.log"
  [[ "$busy" -eq 0 ]] && break
  sleep 30
done

for batch in 01 02 03 04 05 06; do
  while true; do
    printf '%s batch_%s_start\n' "$(date -Is)" "$batch" >> "$log_root/orchestrator_progress.log"
    "$python_bin" scripts/stage3_execute_performance_plan_v3.py \
      --jobs-jsonl "$plan_root/stage3_gold96_performance_batch_${batch}_jobs.jsonl" \
      --state-jsonl "$log_root/batch_${batch}_state.jsonl" \
      --gpus "$gpu_pool" \
      --max-workers 1 \
      > "$log_root/batch_${batch}_stdout.log" \
      2> "$log_root/batch_${batch}_stderr.log"
    rc=$?
    printf '%s batch_%s_end rc=%s\n' "$(date -Is)" "$batch" "$rc" >> "$log_root/orchestrator_progress.log"
    [[ "$rc" -eq 0 ]] && break
    printf '%s batch_%s_retry_after_environment_busy\n' "$(date -Is)" "$batch" >> "$log_root/orchestrator_progress.log"
    sleep 30
  done
done

printf '%s all_batches_end\n' "$(date -Is)" >> "$log_root/orchestrator_progress.log"
