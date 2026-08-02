# 1_7_2_交接文档_INT8_AP补齐与exdata输出规则

## 1. 本轮结论

截至 2026-07-02，本轮 INT8 AP 缺口补测已经完成。

最新三指标主表覆盖:

| precision | latency | energy | AP |
|---|---:|---:|---:|
| FP32 | 60/60 | 60/60 | 60/60 |
| FP16 | 60/60 | 60/60 | 60/60 |
| INT8 | 60/60 | 60/60 | 60/60 |

主表 validator:

```text
scripts/stage2_validate_original60_quant_summary.py --output-root .../original60_quant_20260627
status: PASS
int8_ap_measured: 60
```

## 2. 本轮补齐的 INT8 AP 点

本轮实际新增/补齐 8 个 INT8 AP full-val row:

| label | AP70 | samples | raw artifact |
|---|---:|---:|---|
| frontier_25 | 0.5963335599023177 | 1789 | `/home/.../frontier_25/ap_fullval_1789_bnaware_bias_bulk_v1_gpu2_20260701_int8_ap_gap8` |
| s1_112 | 0.5955747980889666 | 1789 | `/home/.../s1_112/ap_fullval_1789_bnaware_bias_bulk_v1_gpu6_20260701_int8_ap_gap8` |
| frontier_16 | 0.5901800247083223 | 1789 | `/exdata/.../fullval_gap6_20260702/frontier_16/ap_fullval_1789_bnaware_bias_bulk_v1_gpu0_20260702_int8_ap_gap6_home_exfull` |
| frontier_18 | 0.5967172632780710 | 1789 | `/exdata/.../fullval_gap6_20260702/frontier_18/ap_fullval_1789_bnaware_bias_bulk_v1_gpu0_20260702_int8_ap_gap6_home_exfull` |
| frontier_26 | 0.5922425880260059 | 1789 | `/exdata/.../fullval_gap6_20260702/frontier_26/ap_fullval_1789_bnaware_bias_bulk_v1_gpu0_20260702_int8_ap_gap6_home_exfull` |
| frontier_27 | 0.5954547060371977 | 1789 | `/exdata/.../fullval_gap6_20260702/frontier_27/ap_fullval_1789_bnaware_bias_bulk_v1_gpu6_20260702_int8_ap_gap6_home_exfull` |
| frontier_31 | 0.5640735418191275 | 1789 | `/exdata/.../fullval_gap6_20260702/frontier_31/ap_fullval_1789_bnaware_bias_bulk_v1_gpu6_20260702_int8_ap_gap6_home_exfull` |
| s2_096 | 0.5958442033152379 | 1789 | `/exdata/.../fullval_gap6_20260702/s2_096/ap_fullval_1789_bnaware_bias_bulk_v1_gpu6_20260702_int8_ap_gap6_home_exfull` |

其中 `frontier_31` AP70 明显低于同批多数点，应进入后续 suspect list，但它是合法 1789-frame full-val measured row，不是缺失。

## 3. row 文件位置规则

H800 远端从本轮后半段开始固定:

```text
${V2X_DATA_ROOT}/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

`/home` 下对应文件已经改为 symlink:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
  -> ${V2X_DATA_ROOT}/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

本机 4090 没有可用 `/exdata` 挂载，因此已将远端 `/exdata` row 同步回本机项目 rows 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

本机同步前备份:

```text
rows/native_int8_original60_ap_rows_v1.jsonl.before_remote_exdata_ap60_sync_20260702
```

## 4. pipeline 修改

已给 `scripts/stage2_native_int8_ap_bulk_pipeline.py` 增加参数:

```text
--full-ap-raw-root
```

用途:

- `--output-root` 仍指向 `/home/.../original60_quant_20260627`，复用已有 route/range/calibrated artifacts。
- full-val AP 的大 raw 输出写到 `/exdata`，避免 `/home` 根分区爆掉。
- importer 的 row 写入继续走默认 `/home` 路径，但 H800 上该路径已 symlink 到 `/exdata` row 文件。

本轮成功命令形态:

```bash
scripts/stage2_native_int8_ap_bulk_pipeline.py \
  --gpu-id 0 \
  --labels frontier_16,frontier_18,frontier_26 \
  --run-stamp 20260702_int8_ap_gap6_home_exfull \
  --output-root ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627 \
  --full-ap-raw-root ${V2X_DATA_ROOT}/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/fullval_gap6_20260702 \
  --ckpt-root ${V2X_HOME}/heal_research/checkpoints/stage1_single_ckpt_20260701 \
  --disable-default-quarantine \
  --quarantine-labels ""
```

GPU6 同理:

```text
frontier_27,frontier_31,s2_096
```

## 5. 踩坑与修复

### 5.1 78G 可用但仍触发 No space

`/home` 当时还有约 76-78G，但 4 lane full-val 并发为每个 sample 写 `bridge_call_xxx` 中间目录，峰值写入和已有 raw 累积导致 `OSError: [Errno 28] No space left on device`。

修复:

- 不再让 full-val raw 写 `/home`。
- 新增 `--full-ap-raw-root`。
- full-val raw 写 `/exdata`。

### 5.2 直接把整个 output-root 切到 `/exdata` 不可行

直接使用:

```text
--output-root /exdata/...
```

会导致 route helper 在 `/exdata/raw/int8_native_route` 下执行，出现 helper 依赖和 route 产物上下文问题，例如:

```text
ModuleNotFoundError: stage2_h800_native_int8_capability_probe
tvm_operator_inventory.json missing
```

最终采用的正确方案:

```text
output-root = /home
full-ap-raw-root = /exdata
row file = /exdata, /home symlink
```

### 5.3 多 checkpoint 目录问题

部分 label 原始 checkpoint 目录中有多个 `net_epoch*.pth`，`train_utils.load_saved_model` 要求只能有一个 checkpoint。

修复:

```text
${V2X_HOME}/heal_research/checkpoints/stage1_single_ckpt_20260701
```

该目录为每个问题 label 建立只包含 `config.yaml` 和一个 `net_epoch_bestval_at*.pth` 的 symlink checkpoint 目录。

## 6. 最新主表位置

刷新后的主表:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
```

根目录可读副本:

```text
${V2X_ROOT}/multi_agent/data/original60_quant_three_metric_summary_latest.md
```

刷新前备份:

```text
exports/original60_quant_three_metric_summary_latest.json.before_int8_ap60_20260702
exports/original60_quant_three_metric_summary_latest.csv.before_int8_ap60_20260702
exports/original60_quant_three_metric_summary_latest.md.before_int8_ap60_20260702
```

## 7. 下一步仍需关注的点

当前已没有三指标覆盖缺口。下一步不是补缺失点，而是做异常点复核:

1. `frontier_01` FP16/INT8 AP 仍为 0，需要复测或隔离。
2. `frontier_31` INT8 AP70 = 0.5640735418191275，明显低于同批点，应审查是否真实退化。
3. INT8 energy 高于 FP16 的若干点仍需单独审查，不能因 AP 补齐而忽略。
4. H800 `/home` 只剩约 55G，可继续跑 AP，但不建议继续把 full-val raw 写 `/home`。
