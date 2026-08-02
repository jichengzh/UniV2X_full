# 3_7_1_交接文档_INT8_AP缺口补测启动与单ckpt修复

## 1. 本轮目标

本轮目标是补测当前三精度主表中唯一仍未补齐的数据集合空缺:

```text
INT8 AP: 52/60 -> 目标 60/60
```

当前 8 个缺失 label:

```text
frontier_16
frontier_18
frontier_25
frontier_26
frontier_27
frontier_31
s1_112
s2_096
```

本轮不重测 latency/energy。AP 补测口径继续沿用既有 Native INT8 full AP bridge:

- TVM native INT8 backbone/subnet + PyTorch head/postprocess。
- `full_network_claim=false`。
- full-val row 必须满足 `processed_samples >= 1789`。
- 通过 `scripts/stage2_import_native_int8_full_ap_row.py` gate 后才能写入 `rows/native_int8_original60_ap_rows_v1.jsonl`。

## 2. 启动前状态

主表路径:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
```

启动前覆盖:

| precision | latency | energy | AP |
|---|---:|---:|---:|
| FP32 | 60/60 | 60/60 | 60/60 |
| FP16 | 60/60 | 60/60 | 60/60 |
| INT8 | 60/60 | 60/60 | 52/60 |

启动前 row 文件:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

状态:

```text
rows: 52
unique measured labels: 52
missing_gap8:
frontier_16, frontier_18, frontier_25, frontier_26,
frontier_27, frontier_31, s1_112, s2_096
```

## 3. H800 启动信息

H800 连接方式固定使用:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

本轮使用 H800 服务器:

```text
host: <PRIVATE_HOST>
repo: ${V2X_ROOT}
```

启动时 GPU 状态:

- GPU0, GPU2, GPU3, GPU6 可用于本轮 AP 补测。
- GPU1, GPU4, GPU5 已有其他负载，不作为本轮 AP lane 主资源。

注意: AP full-val 不要求像 latency/energy 那样 GPU 完全空闲，但仍需要记录同机负载，后续解释 AP runtime 或异常时要保留该背景。

## 4. 已启动的补测任务

### 4.1 第一批 4 lane

启动脚本:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/launch_gap8_20260701_int8_ap_gap8/launch_gap8.sh
```

启动命令核心参数:

```bash
scripts/stage2_native_int8_ap_bulk_pipeline.py \
  --gpu-id <gpu> \
  --labels <labels> \
  --run-stamp 20260701_int8_ap_gap8 \
  --disable-default-quarantine \
  --quarantine-labels ""
```

必须使用 `--disable-default-quarantine`，因为历史默认 quarantine 集合包含多个当前缺失点；不禁用会直接跳过补测。

lane 分配:

| GPU | labels | PID | control dir |
|---:|---|---:|---|
| 0 | `frontier_16,frontier_18` | 957161 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu0_20260701_int8_ap_gap8` |
| 2 | `frontier_25,frontier_26` | 957164 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu2_20260701_int8_ap_gap8` |
| 3 | `frontier_27,frontier_31` | 957167 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu3_20260701_int8_ap_gap8` |
| 6 | `s1_112,s2_096` | 957170 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu6_20260701_int8_ap_gap8` |

### 4.2 第一批早期结果

已经确认:

- `frontier_25`: export/bootstrap/range_capture/numeric_sanity/AP smoke 已通过，进入 full-val 阶段。
- `s1_112`: export/bootstrap/range_capture/calibrated_route/numeric_sanity/AP smoke 已通过，进入 full-val 阶段。

失败:

- `frontier_16`: range_capture 失败。
- `frontier_18`: range_capture 失败。
- `frontier_27`: range_capture 失败。
- `frontier_31`: range_capture 失败。

失败原因一致:

```text
HEAL/opencood/tools/train_utils.py load_saved_model:
assert len(file_list) == 1
```

这些 checkpoint 目录中存在多个 `net_epoch*.pth`，而 `train_utils.load_saved_model` 要求 ckpt 目录里只能有一个 checkpoint 文件。

## 5. 已执行的修复

为失败的 4 个 label 创建了单 checkpoint symlink 目录:

```text
${V2X_HOME}/heal_research/checkpoints/stage1_single_ckpt_20260701
```

映射关系:

| label | single ckpt dir | selected checkpoint |
|---|---|---|
| frontier_16 | `Pyramid_DAIR_m1_stage2_ap_frontier_16_2026_06_99_single` | `net_epoch_bestval_at27.pth` |
| frontier_18 | `Pyramid_DAIR_m1_stage2_ap_frontier_18_2026_06_99_single` | `net_epoch_bestval_at27.pth` |
| frontier_27 | `Pyramid_DAIR_m1_stage2_ap_frontier_27_2026_06_99_single` | `net_epoch_bestval_at25.pth` |
| frontier_31 | `Pyramid_DAIR_m1_stage2_ap_frontier_31_2026_06_99_single` | `net_epoch_bestval_at25.pth` |

为避免复用旧的多 checkpoint 导出产物，已将这 4 个 label 本轮生成的 export/bootstrap/range_capture 目录改名备份，后缀:

```text
backup_before_single_ckpt_rerun_20260701
```

未删除历史数据。

## 6. 单 ckpt 修复重跑

启动脚本:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/launch_gap8_singleckpt_20260701/launch_singleckpt.sh
```

启动命令核心参数:

```bash
scripts/stage2_native_int8_ap_bulk_pipeline.py \
  --gpu-id <gpu> \
  --labels <labels> \
  --run-stamp 20260701_int8_ap_gap8_singleckpt \
  --ckpt-root ${V2X_HOME}/heal_research/checkpoints/stage1_single_ckpt_20260701 \
  --disable-default-quarantine \
  --quarantine-labels ""
```

lane 分配:

| GPU | labels | PID | control dir |
|---:|---|---:|---|
| 0 | `frontier_16,frontier_18` | 987708 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu0_20260701_int8_ap_gap8_singleckpt` |
| 3 | `frontier_27,frontier_31` | 987711 | `raw/int8_native_route/20260629_native_int8_ap_bulk_pipeline_gpu3_20260701_int8_ap_gap8_singleckpt` |

当前确认进展:

- `frontier_16`: export -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity 均已通过。
- `frontier_27`: export -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity 均已通过。

这说明单 checkpoint 修复有效。

## 7. 当前仍在后台运行的任务

截至本文写入时，以下 PID 仍在 H800 后台:

```text
957164: GPU2, frontier_25/frontier_26
957170: GPU6, s1_112/s2_096
987708: GPU0, frontier_16/frontier_18, single ckpt rerun
987711: GPU3, frontier_27/frontier_31, single ckpt rerun
```

当前主 row 文件仍未出现新增 gap8 row:

```text
rows/native_int8_original60_ap_rows_v1.jsonl: 52 rows, 52 unique labels
missing_gap8:
frontier_16, frontier_18, frontier_25, frontier_26,
frontier_27, frontier_31, s1_112, s2_096
```

这是正常状态，因为 full-val 1789-frame 评估尚未完成并通过 importer。

## 8. 巡检命令

查看进程:

```bash
ps -fp 957164,957170,987708,987711 || true
```

查看 GPU:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

查看 events:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
for d in \
  $BASE/20260629_native_int8_ap_bulk_pipeline_gpu{2,6}_20260701_int8_ap_gap8 \
  $BASE/20260629_native_int8_ap_bulk_pipeline_gpu{0,3}_20260701_int8_ap_gap8_singleckpt
do
  echo "===$d==="
  tail -n 40 "$d/events.tsv" 2>/dev/null || true
  cat "$d/latest_results.json" 2>/dev/null || true
done
```

查看 row 是否补齐:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
labels = {r.get("label") for r in rows if r.get("measurement_status") == "measured"}
gap8 = ["frontier_16","frontier_18","frontier_25","frontier_26","frontier_27","frontier_31","s1_112","s2_096"]
print("rows", len(rows), "unique", len(labels), "missing_gap8", [x for x in gap8 if x not in labels])
PY
```

## 9. 完成后的收口步骤

当 row 文件新增并覆盖 8 个缺失 label 后，需要执行:

1. 从 H800 回传更新后的 row 文件和 raw artifact 索引到 4090 本地。
2. 重新生成三指标主表:

```bash
cd ${V2X_ROOT}
OUT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627
python scripts/stage2_generate_original60_quant_state_coverage.py \
  --output-root "$OUT" \
  --fp32-latency-smoke-rows "$OUT/rows/fp32_latency_strict_direct_rows_v1.jsonl" \
  --fp16-latency-rows "$OUT/rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl" \
  --fp16-energy-rows "$OUT/rows/fp16_rewritten_tensorcore_full60_energy_rows_v1.jsonl"
cp -a "$OUT/exports/original60_quant_three_metric_summary_latest.md" \
  ${V2X_ROOT}/multi_agent/data/original60_quant_three_metric_summary_latest.md
```

3. 运行 validator:

```bash
python scripts/stage2_validate_original60_quant_summary.py --output-root "$OUT"
```

4. 运行自定义覆盖检查，目标为:

```text
FP32 latency/energy/AP = 60/60
FP16 latency/energy/AP = 60/60
INT8 latency/energy/AP = 60/60
```

## 10. 注意事项

- 不要把 full-val 未完成的中间 report 导入主表。
- 不要把 5-sample smoke AP 导入 `native_int8_original60_ap_rows_v1.jsonl`。
- 不要恢复默认 quarantine 后再跑 gap8，否则会跳过大部分缺失点。
- 如果某个 full-val 失败，保留 `failure_report.json`、`stdout.txt`、`stderr.txt`、`events.tsv`，并继续其他 label。
- 如果 importer 成功但主表仍显示 INT8 AP 未补齐，先检查 `native_int8_original60_ap_rows_v1.jsonl` 是否已回传到 4090，再重新生成 summary。
