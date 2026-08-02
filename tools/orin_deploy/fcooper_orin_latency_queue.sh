#!/usr/bin/env bash
set -euo pipefail

root=/home/jichengzhi/V2X/results/lane_c_orin_fcooper_stage6_five_config_20260724
runner=/home/jichengzhi/V2X/tools/orin_deploy/fcooper_orin_runner.py
python=/home/jichengzhi/uniad_venv/bin/python
heal=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL
export PYTHONPATH=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/V2X:/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL

arms=(
  compression_only
  schedule_only
  compress_then_tune
  joint_fp16_control
  original_default
)

wait_for_ap() {
  local arm
  for arm in "${arms[@]}"; do
    local marker=$root/05_full2170_ap/$arm/full.exit
    while [[ ! -f $marker ]]; do
      sleep 30
    done
    [[ $(<"$marker") == 0 ]]
  done
}

wait_for_power_permission() {
  local probe_dir=$root/04_energy/permission_probes
  mkdir -p "$probe_dir"
  while true; do
    set +e
    sudo -n /usr/bin/tegrastats --stop >/dev/null 2>&1
    sudo -n /usr/bin/tegrastats --interval 500 \
      > "$probe_dir/latest.log" 2>&1 &
    local probe_pid=$!
    sleep 2
    sudo -n /usr/bin/tegrastats --stop >/dev/null 2>&1
    local stop_status=$?
    wait "$probe_pid"
    set -e
    if [[ $stop_status == 0 ]] && grep -q 'VIN_SYS_5V0' "$probe_dir/latest.log"; then
      return
    fi
    sleep 60
  done
}

run_measurement() {
  local arm=$1
  local command_name=measure-trt
  local engine_arguments=(
    --engine "$root/02_engines/$arm/model.engine"
  )
  if [[ $arm == original_default ]]; then
    command_name=measure-native
    engine_arguments=()
  fi
  local latency_dir=$root/03_latency/$arm
  local power_dir=$root/04_energy/raw_tegrastats
  mkdir -p "$latency_dir" "$power_dir"
  local command=(
    "$python" -u "$runner" "$command_name"
    --manifest "$root/00_source_audit/canonical_manifest.json"
    --arm "$arm"
    --inputs-npy "$root/01_numeric_gate/heldout_inputs.npy"
    --power-log "$power_dir/$arm.log"
    --raw-samples-csv "$latency_dir/samples.csv"
    --output-json "$latency_dir/report.json"
    --warmup 20
    --iters 300
    --repeat 5
    "${engine_arguments[@]}"
  )
  printf '%q ' "${command[@]}" > "$latency_dir/measure-command.txt"
  printf '\n' >> "$latency_dir/measure-command.txt"
  set +e
  (
    cd "$heal"
    "${command[@]}"
  ) > "$latency_dir/measure.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$latency_dir/measure.exit"
  [[ $status == 0 ]]
}

wait_for_ap
wait_for_power_permission
run_measurement original_default
run_measurement compression_only
run_measurement schedule_only
run_measurement compress_then_tune
run_measurement joint_fp16_control
