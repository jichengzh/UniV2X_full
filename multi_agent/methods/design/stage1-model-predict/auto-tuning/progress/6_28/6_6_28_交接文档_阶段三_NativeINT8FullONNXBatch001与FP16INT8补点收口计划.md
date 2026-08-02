# 12_6_28_交接文档_阶段三_NativeINT8FullONNXBatch001与FP16INT8补点收口计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/10_6_28_交接文档_阶段三_FP16INT8_Original60补点收口计划.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/11_6_28_交接文档_阶段三_Original60补点队列与H800启动计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.md`

本文件 supersede `11_...H800启动计划.md` 中的 H800 未启动状态: H800 访问已经打通, original60 native INT8 full-ONNX batch001 已完成并拉回本地。

## 0. 两个口径问题

### 0.1 INT8 backbone/subnet 是否已经打通

结论: 是, 但必须限定为 `backbone/subnet` 范围, 不是完整感知网络。

当前已经有两层证据:

1. 三个 anchor 的 native INT8 smoke route 已完成: `base/s0_024/s1_048`。
2. original60 batch001 的 full-ONNX-topology native INT8 route 已完成: `frontier_01/frontier_02/frontier_03`。

batch001 使用:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

batch001 没有使用 TRT, 没有使用旧的 `static_qdq_synthetic_minmax` QDQ route。dtype gate 显示 `float32=0`, `quantize=0`, `dequantize=0`。

batch001 实测结果:

| label | latency_ms | energy_J / inference | op coverage | dtype/QDQ gate |
|---|---:|---:|---|---|
| `frontier_01` | 3.477088 | 0.668574 | `Conv=51, Relu=48, Add=16, Identity=46` | `float32=0, quantize=0, dequantize=0` |
| `frontier_02` | 3.901205 | 0.818304 | `Conv=51, Relu=48, Add=16, Identity=47` | `float32=0, quantize=0, dequantize=0` |
| `frontier_03` | 3.494091 | 0.690632 | `Conv=51, Relu=48, Add=16, Identity=45` | `float32=0, quantize=0, dequantize=0` |

本地产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001_remote_run_log/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

远端 H800 run:

```text
${V2X_DATA_ROOT}/s2_tvm/native_int8_pack/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
```

### 0.2 当前 latency 是否是 backbone 端到端推理速度

结论: 是 backbone/subnet compiled module 的端到端实测; 不是完整感知网络端到端。

统一口径:

```text
latency_ms = H800 + TVM measured backbone/subnet end-to-end module latency.
full_network_claim = false.
```

解释:

- 如果“端到端”指从 `spatial_features` 输入到 backbone/subnet 输出, 当前 latency 是 compiled module 的真实端到端推理, 不是 per-op 估算。
- 如果“端到端”指完整 V2X 感知链路, 包括 head、postprocess、dataset eval、通信/IO/CPU 调度, 当前 latency 不是完整链路端到端。
- 如果“RSU 边缘段”指 RSU-side backbone/dense-core workload, 当前 H800 + TVM latency 可以作为这一段 GPU 计算速度 proxy。
- 如果“RSU 边缘段设备”指真实 RSU 物理边缘硬件的绝对推理速度, 当前不能这样声明, 因为实测硬件是 H800 Hopper。

格式注意:

- 旧 row schema 中原始字段仍常见 `latency_p50_us` / `latency_unit=us`。
- 对外总表和审阅 MD 必须统一显示 `latency_ms = latency_p50_us / 1000`。
- 下一阶段写总表时不允许再把 us 字段直接暴露为最终 latency 指标。

## 1. 当前补点状态

Completion queue:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

队列规模:

| item | count |
|---|---:|
| total jobs | 120 |
| FP16 jobs | 60 |
| INT8 jobs | 60 |
| unique labels | 60 |

目前已完成但尚未合并进 canonical 总表的关键补点:

| precision | metric | measured rows | row file |
|---|---:|---:|---|
| INT8 native full ONNX | latency | 3 / 60 | `rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl` |
| INT8 native full ONNX | energy | 3 / 60 | `rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl` |
| INT8 native full ONNX | AP70 | 0 / 60 | not produced |
| FP16 true route | latency | 2 / 60 in canonical smoke state | `rows/latency_original60_quant_rows_v1.jsonl` |
| FP16 true route | energy | 2 / 60 in canonical smoke state | `rows/energy_original60_quant_rows_v1.jsonl` |
| FP16 true route | AP70 | 0 / 60 | not produced |

当前 canonical 三表仍是状态覆盖表, 不是最终完成表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/latency_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/energy_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/ap_original60_quant_rows_v1.jsonl
```

## 2. 下一阶段最终目标

收口目标必须具体到行数:

```text
完成 original60 现有 60 个配置的 FP16 和 INT8 补点。
```

最终验收为 120 个 config-precision cell:

```text
60 configs x {fp16, int8} = 120 cells
```

每个 cell 必须有:

| metric | requirement |
|---|---|
| `latency_ms` | H800 + TVM measured backbone/subnet module latency; 总表统一 ms |
| `energy_J / inference` | H800 telemetry measured energy; 必须有 idle/active samples |
| `AP70` | 合规 real eval/import measured source; 禁止 predicted/interpolated/model-fit |
| raw evidence | latency/energy/AP runner 输出、命令、日志、row source |
| precision evidence | FP16 必须证明 true FP16; INT8 必须证明 native INT8 或标清 route |
| claim gate | `measurement_status=measured`, `full_network_claim=false` |

目标输出行数:

| output family | target measured rows |
|---|---:|
| FP16 latency | 60 |
| FP16 energy | 60 |
| FP16 AP70 | 60 |
| native INT8 latency | 60 |
| native INT8 energy | 60 |
| INT8 AP70 | 60 |

## 3. 执行顺序

### 3.1 INT8 native full-ONNX 大范围补点

从已经成功的 batch001 继续扩展:

1. 先把 `frontier_01/02/03` 的 row 合入下一版 review/gap 状态, 并在审阅 MD 中用 ms 展示。
2. 按 queue 分批跑剩余 57 个 label。
3. 每批建议 3 到 6 个 label, 避免单次失败影响整批。
4. 如果出现新 op 或 shape, 先扩展 route 或写最小复现, 不直接停止。
5. 每批完成后拉回 raw 和 rows, 校验 `measurement_status=measured`、`full_network_claim=false`、`energy_J>0`、`latency_ms>0`、无 TRT/QDQ。

核心输出:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

### 3.2 FP16 latency 和 energy 补点

FP16 路线要避免历史问题: 不能把 FP32 测量误标为 FP16。

要求:

- runner 必须显式证明 input/initializer/compiled module 为 true FP16。
- raw 中保留 layer precision summary 或 dtype manifest。
- energy 采用与 INT8 相同的 idle-subtracted telemetry burst 方法。
- `s0_024/s1_048` 已有 smoke measured, 但仍应纳入最终 consistency audit。

建议输出:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl
rows/fp16_true_original60_energy_rows_v1.jsonl
```

### 3.3 FP16 和 INT8 AP70 补点

AP 不能从 backbone latency/energy 推断。

FP16 AP acceptance:

- 使用 true FP16 eval source 或明确 precision/eval marker 的合规 import。
- 记录 dataset split、ckpt、eval command、AP30/AP50/AP70、raw eval output。
- 禁止把 TRT FP16、FP32 或 predicted AP 写成 H800 TVM FP16 measured AP。

INT8 AP acceptance:

- 使用真实 INT8 eval source, 并记录 route、quant policy、calibration/eval backend。
- 如果当前 native INT8 backbone 不能直接接 eval pipeline, 先实现 artifact 接口或写清 precise blocker。
- 禁止把 QDQ-only、TRT INT8、simulated INT8 或 predicted AP 写成 measured AP。

建议输出:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 3.4 总表刷新

完成补点后刷新:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
```

总表审阅版必须只展示 `latency_ms`, 不展示裸 `latency_p50_us` 作为最终指标。

## 4. 失败处理策略

遇到任何问题不得直接停止整阶段。必须先完成反思、审查和问题解决闭环:

1. 保存 raw stdout/stderr、runner command、GPU id、env、artifact path、failure reason。
2. 分类失败: artifact 缺失、ONNX op/shape 不支持、TVM build 失败、runtime 失败、power telemetry 失败、AP eval source 缺失、AP OOM、AP trend anomaly。
3. 为失败点构造最小复现或 op-level blocker。
4. 修改 runner / queue / route 后重试。
5. 只有在重试和审查后仍无法解决时, 才写 quarantine/no-claim row。

blocker 必须精确到:

```text
label, precision, metric, op name, op type, input shape, group, stride, pad,
build_status, run_status, traceback, raw log path, retry count, final decision
```

## 5. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 补点收口。不要启动 agent team, 单 agent 执行即可。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/12_6_28_交接文档_阶段三_NativeINT8FullONNXBatch001与FP16INT8补点收口计划.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl

当前已完成:
- H800 访问已打通。
- original60 native INT8 full-ONNX batch001 已完成 frontier_01/frontier_02/frontier_03。
- batch001 raw 已拉回:
  multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
- batch001 rows 已拉回:
  rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
  rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
- batch001 结果:
  frontier_01 latency_ms=3.477088, energy_J=0.668574
  frontier_02 latency_ms=3.901205, energy_J=0.818304
  frontier_03 latency_ms=3.494091, energy_J=0.690632
- 所有 batch001 row 均为 backbone/subnet scope, full_network_claim=false。

下一阶段目标:
- 完成 original60 现有 60 个配置的 FP16 和 INT8 补点。
- 最终目标行数:
  FP16 latency 60, FP16 energy 60, FP16 AP70 60,
  native INT8 latency 60, native INT8 energy 60, INT8 AP70 60。
- latency 对外统一为 latency_ms; 如果 raw 是 latency_p50_us, 总表写入和 MD 审阅必须转换为 ms。
- INT8 从 batch001 继续, 分批完成剩余 57 个 native full-ONNX latency/energy。
- FP16 补齐 true-FP16 latency/energy, 并证明不是 FP32 误标。
- AP70 必须来自合规 measured eval/import source, 不得用 predicted/interpolated/model-fit/TRT reference 冒充。
- 遇到任何问题不要直接停止: 保存日志, 分类失败, 做最小复现或 op-level blocker, 修改后重试。只有确认无法解决时才写 quarantine/no-claim row。

最终交付:
- rows/fp16_true_original60_latency_rows_v1.jsonl
- rows/fp16_true_original60_energy_rows_v1.jsonl
- rows/fp16_true_original60_ap_rows_v1.jsonl
- rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
- rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
- rows/native_int8_original60_ap_rows_v1.jsonl
- rows/latency_original60_quant_rows_v1.jsonl
- rows/energy_original60_quant_rows_v1.jsonl
- rows/ap_original60_quant_rows_v1.jsonl
- exports/original60_quant_three_metric_summary_latest.md/.json
- exports/fp16_int8_original60_completion_review_latest.md/.json
- exports/fp16_int8_original60_gap_report_latest.md/.json
```
