# 10_6_28_交接文档_阶段三_FP16INT8_Original60补点收口计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/7_6_27_交接文档_阶段二_FP16INT8补测数据位置与INT8根因解释.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/9_6_28_交接文档_阶段二_NativeINT8Smoke实测结果与完整ONNXRoute下一步.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`

## 0. 两个口径问题先回答清楚

### 0.1 INT8 backbone 是否已经打通

结论: 已经打通 `base/s0_024/s1_048` 三个 anchor 的 H800 + TVM native INT8 backbone/subnet 路线。

已完成的是:

- 从 H800 `${V2X_DATA_ROOT}/s2_tvm/models/{base,s0_024,s1_048}_backbone.onnx` 读取真实 backbone ONNX 拓扑。
- 支持 `Conv / Relu / Add / Identity`。
- 覆盖 1x1 conv、group=32 的 3x3 conv、residual Add、Identity initializer alias、stage width changes。
- 使用 TVM TE/TOPI native INT8 graph executor, 不使用 TRT, 不使用 static QDQ route。
- 三个 anchor 均已产出 latency 和 energy raw evidence。

当前三点 full ONNX topology native INT8 结果:

| label | latency_ms | energy_J / inference | op coverage | dtype/QDQ gate |
|---|---:|---:|---|---|
| `base` | 11.095775 | 2.853151923181 | `Conv=50`, `Relu=48`, `Add=16` | `float32=0`, `quantize=0`, `dequantize=0` |
| `s0_024` | 9.235392 | 2.290300810919 | `Conv=51`, `Relu=48`, `Add=16`, `Identity=46` | `float32=0`, `quantize=0`, `dequantize=0` |
| `s1_048` | 8.109461 | 2.118255342182 | `Conv=50`, `Relu=48`, `Add=16`, `Identity=44` | `float32=0`, `quantize=0`, `dequantize=0` |

证据位置:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_h800_v1/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.md
```

必须保留的边界:

- 这不是 full perception network, 所有 row 都是 `full_network_claim=false`。
- 这不是 AP measured, 不能把它写成检测精度完成。
- 这只证明 native INT8 full ONNX backbone route 在三个 anchor 上可 build/run, 还没有覆盖 original60 的 60 个配置。

### 0.2 当前 latency 是否都是 backbone 端到端推理速度

结论要分两层写:

1. 如果“端到端”指的是 backbone/subnet compiled module 内部, 即从 `spatial_features` 输入跑到 backbone 输出, 那么当前 latency rows 是 backbone/subnet scope 的端到端实测, 不是 per-op 估算。
2. 如果“端到端”指完整感知模型, 包括 head、postprocess、dataset eval、V2X 全链路 IO, 那么当前结果不是完整模型端到端。

RSU 边缘段速度口径也要拆开:

- 如果“RSU 边缘段”指 RSU-side backbone/dense-core 这一段模型 workload, 当前 H800 + TVM latency 可以作为该段计算速度的 proxy。
- 如果“RSU 边缘段设备”指真实 RSU 边缘硬件的绝对推理速度, 当前结果不能这样声明, 因为实测硬件是 H800 Hopper, 不是 RSU 物理边缘设备。

因此后续文档统一写:

```text
latency_ms = H800 + TVM measured backbone/subnet end-to-end module latency.
full_network_claim = false.
```

## 1. 当前 original60 三指标状态

当前 canonical 状态表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/latency_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/energy_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/ap_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
```

当前行数:

| table | rows | meaning |
|---|---:|---|
| `latency_original60_quant_rows_v1.jsonl` | 180 | 60 config x 3 precision state rows |
| `energy_original60_quant_rows_v1.jsonl` | 180 | 60 config x 3 precision state rows |
| `ap_original60_quant_rows_v1.jsonl` | 180 | 60 config x 3 precision state rows |

这些是状态覆盖表, 不是 FP16/INT8 已完成表。当前 measured 覆盖如下:

| metric | fp16 measured | fp16 no_claim | int8 measured | int8 no_claim | notes |
|---|---:|---:|---:|---:|---|
| latency | 2 | 58 | 2 | 58 | measured 的两个点是 `s0_024/s1_048`; INT8 还是旧 TVM VM/QDQ smoke, 不是 native full ONNX route |
| energy | 2 | 58 | 2 | 58 | measured 的两个点是 `s0_024/s1_048`; INT8 还是旧 energy smoke |
| AP70 | 0 | 60 | 0 | 60 | 没有合规 FP16/INT8 AP measured source |

特别注意:

- `base` 是 quant anchor, 不属于 original60 的 60 个配置。
- 新的 native INT8 full ONNX 三点结果还没有合入 canonical original60 三指标总表。
- `s0_024/s1_048` 在 original60 中已有旧 INT8 latency/energy smoke, 但下一阶段应优先替换为 native INT8 full ONNX route 的 measured row, 并保留旧 QDQ/TVM VM route 作为历史对照。

## 2. 下一阶段具体目标

最终收口目标:

```text
完成 original60 现有 60 个配置的 fp16 与 int8 补点。
```

验收单位是 120 个 config-precision cell:

```text
60 configs x {fp16, int8} = 120 cells
```

每个 cell 的完整数据必须包含:

| field | requirement |
|---|---|
| `latency_ms` | H800 + TVM measured backbone/subnet module latency, 对外统一 ms |
| `energy_J / inference` | H800 power telemetry measured energy, 带 idle/active samples |
| `AP70` | 合规 measured AP source, 不能用 predicted/model-fit/interpolated AP |
| `raw_artifact` | latency/energy/AP 原始日志、runner 输出、row source |
| `precision evidence` | FP16 必须证明 input/initializer/runner 是 true FP16; INT8 必须证明 native INT8 或明确 route |
| `claim gate` | `measurement_status=measured`, `claim_status=claimable_*`, `full_network_claim=false` |

目标行数:

| output | target measured rows |
|---|---:|
| FP16 latency | 60 |
| FP16 energy | 60 |
| FP16 AP70 | 60 |
| native INT8 latency | 60 |
| native INT8 energy | 60 |
| INT8 AP70 | 60 |

如果某个 cell 失败, 不允许直接停止整批任务。必须先完成问题解决闭环:

1. 保存 raw stdout/stderr、runner command、GPU id、env、artifact path、failure reason。
2. 判断失败类型: artifact 缺失、ONNX shape/op 不支持、TVM build 失败、runtime 失败、power telemetry 失败、AP eval source 缺失、AP eval OOM、AP trend anomaly。
3. 为失败点构造最小复现或 op-level blocker。
4. 修改 runner 或 job queue 后重试。
5. 只有在重试和审查后仍不能解决时, 才写 quarantine/no-claim row, 并保留逐点 blocker。

## 3. 执行路线

### 3.1 Inventory 和队列

先生成 original60 x precision 队列:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

每条 job 至少包含:

```text
label
width
precision
onnx_backbone_path
workdir
latency_status
energy_status
ap_status
artifact_ready
last_failure_reason
```

队列必须从 canonical original60 表反推缺口, 不重新发明 60 个配置。

### 3.2 FP16 补点

FP16 路线目标:

- 使用 true FP16 backbone ONNX 或 runner 中显式 half input/half initializer。
- 禁止继续使用历史 suspect FP16-tagged row 作为 true-FP16 证据。
- 对 60 个配置补 `latency_ms` 和 `energy_J / inference`。
- 每个 raw 目录必须包含:

```text
latency_result.json
energy_result.json
idle_power_samples.csv
active_power_samples.csv
telemetry_payload.json
layer_precision_summary.json
```

建议新 row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
```

### 3.3 Native INT8 补点

INT8 路线目标:

- 以 `stage2_h800_native_int8_full_onnx_route.py` 为基础推广到 original60 的全部 `{label}_backbone.onnx`。
- 优先保持 `route_spec=full_onnx_topology_conv_relu_add_identity_v1`。
- 如果发现新 op, 先扩展 route; 不能直接写 TVM 不支持 INT8。
- 失败 blocker 必须精确到 op name、op type、shape、group、stride、pad、build/run status、traceback。
- 对 60 个配置补 native INT8 `latency_ms` 和 `energy_J / inference`。

建议新 row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

### 3.4 AP70 补点

AP70 不能从 backbone latency/energy 推断。AP measured row 必须来自合规 eval/import source。

FP16 AP acceptance:

- 使用 true FP16/TVM eval source, 或有明确 precision/eval marker 的合规 import。
- 记录 dataset split、ckpt、eval command、AP30/AP50/AP70、raw eval output。
- 禁止把 TRT FP16 AP 写成 H800 TVM FP16 measured AP。

INT8 AP acceptance:

- 使用 real INT8 eval source, 并记录 INT8 route、quant recipe、calibration evidence、eval backend。
- 如果当前 eval pipeline 无法直接跑 native INT8 backbone, 先实现 backbone quantized artifact 与下游 eval pipeline 的连接。
- 禁止把 TRT INT8、simulated INT8、QDQ-only artifact 或 predicted AP 写成 measured AP。

建议新 row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

### 3.5 总表收口

补点完成后刷新:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/latency_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/energy_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/ap_original60_quant_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
```

审阅版还应新增:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.json
```

## 4. 不能再犯的口径错误

1. latency 对外统一 `ms`, 原始 `latency_p50_us` 必须转换后再进入审阅表。
2. `full_network_claim=false` 不能改成 true。
3. backbone/subnet end-to-end 不能写成 full perception network end-to-end。
4. H800 + TVM 结果不能写成真实 RSU 边缘硬件绝对速度。
5. FP16 必须防止 FP32 backbone 被误当作 FP16 测量。
6. INT8 必须区分旧 QDQ/TVM VM smoke 和新的 native INT8 full ONNX route。
7. AP 只能来自 true eval/import evidence, 不能使用 predicted、interpolated、model-fit、TRT reference 冒充 measured。

## 5. 验证命令

补点阶段每轮至少运行:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
python -m py_compile framework/stage2/lut_productization.py framework/stage2/native_int8_route.py framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py
```

行数检查:

```text
wc -l multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/*original60*rows_v1.jsonl
jq -r '.precision + "\t" + (.measurement_status // "") + "\t" + (.claim_status // "") + "\t" + (.quality_gate_status // "")' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/latency_original60_quant_rows_v1.jsonl | sort | uniq -c
jq -r '.precision + "\t" + (.measurement_status // "") + "\t" + (.claim_status // "") + "\t" + (.quality_gate_status // "")' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/energy_original60_quant_rows_v1.jsonl | sort | uniq -c
jq -r '.precision + "\t" + (.measurement_status // "") + "\t" + (.claim_status // "") + "\t" + (.quality_gate_status // "")' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/ap_original60_quant_rows_v1.jsonl | sort | uniq -c
```

最终验收检查:

```text
FP16 latency measured = 60
FP16 energy measured = 60
FP16 AP70 measured = 60
native INT8 latency measured = 60
native INT8 energy measured = 60
INT8 AP70 measured = 60
所有 measured row 均有 raw_artifact/source_files/digest/precision evidence
所有失败 cell 均有逐点 blocker/quarantine, 且不影响其他 cell 继续运行
```

## 6. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 补点收口。不要启动 agent team, 单 agent 执行即可。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/10_6_28_交接文档_阶段三_FP16INT8_Original60补点收口计划.md
- multi_agent/methods/design/auto-tuning/progress/6_27/9_6_28_交接文档_阶段二_NativeINT8Smoke实测结果与完整ONNXRoute下一步.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md

H800 服务器密码是12345678

当前事实:
- H800 + TVM native INT8 full ONNX backbone route 已在 base/s0_024/s1_048 三个 anchor 打通。
- 当前 latency 是 H800 + TVM backbone/subnet module end-to-end latency, 不是 full perception network latency; full_network_claim 必须保持 false。
- original60 canonical 三指标表已有 60 x 3 precision 状态行, 但 FP16/INT8 大多数还是 no_claim。
- original60 当前 FP16 latency/energy measured 只有 s0_024/s1_048 两点, INT8 latency/energy measured 也只有 s0_024/s1_048 两点且是旧 QDQ/TVM VM smoke; AP70 对 FP16/INT8 仍为 0 measured。

硬目标:
- 完成 original60 60 个配置 x {fp16,int8} 的补点, 共 120 个 config-precision cell。
- 每个 cell 都要产出 latency_ms、energy_J / inference、AP70 measured, 或在确实无法解决后写逐点 blocker/quarantine。
- 最终目标行数: FP16 latency 60 measured, FP16 energy 60 measured, FP16 AP70 60 measured, native INT8 latency 60 measured, native INT8 energy 60 measured, INT8 AP70 60 measured。

执行要求:
- 先生成 original60 x {fp16,int8} completion queue, 从现有 canonical original60 表反推缺口。
- FP16 禁止复用历史 suspect FP16-tagged row 作为 true-FP16 证据; 必须证明 input/initializer/runner 是 true FP16。
- INT8 必须优先推广 native INT8 full ONNX route, 不再把旧 QDQ/float32-heavy route 当作最终 INT8 加速结果。
- AP70 必须来自合规 true eval/import source; 禁止 predicted/model-fit/interpolated AP, 禁止 TRT reference 冒充 H800 TVM measured。
- latency 对外统一 ms; energy 统一 J / inference; 所有 measured row 必须带 raw_artifact/source_files/digest/precision evidence。

失败处理:
- 遇到任一配置失败, 不得直接停止整批任务。
- 必须保存 stdout/stderr、runner command、GPU id、raw artifact、failure reason。
- 必须判断失败类型, 构造最小复现或 op-level blocker, 修改 runner/job queue 后重试。
- 只有经过反思、审查和问题解决仍不可解时, 才对该 cell 写 quarantine/no-claim; 其他 cell 继续运行。

收口产物:
- rows/fp16_true_original60_latency_rows_v1.jsonl
- rows/fp16_true_original60_energy_rows_v1.jsonl
- rows/fp16_true_original60_ap_rows_v1.jsonl
- rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
- rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
- rows/native_int8_original60_ap_rows_v1.jsonl
- 刷新 rows/latency_original60_quant_rows_v1.jsonl、rows/energy_original60_quant_rows_v1.jsonl、rows/ap_original60_quant_rows_v1.jsonl
- 刷新 exports/original60_quant_three_metric_summary_latest.md/.json
- 新增 exports/fp16_int8_original60_completion_review_latest.md/.json
- 新增 exports/fp16_int8_original60_gap_report_latest.md/.json
- 写下一份中文交接文档, 记录 measured 覆盖、失败 blocker、剩余缺口和下一步。
```
