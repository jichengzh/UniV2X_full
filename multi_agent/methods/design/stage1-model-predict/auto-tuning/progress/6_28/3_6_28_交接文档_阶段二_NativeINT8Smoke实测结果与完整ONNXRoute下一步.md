# 9_6_28_交接文档_阶段二_NativeINT8Smoke实测结果与完整ONNXRoute下一步

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/8_6_27_交接文档_阶段二_NativeINT8BackboneSubnet加速计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_lowering_audit_latest.md`

## 0. 最新结果摘要

已在 H800 + TVM 路线上完成 native INT8 smoke 闭环。最终采用 run:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_exdata_v3/
```

三类 attempt 均为 success:

| attempt | build_status | run_status | 结论 |
|---|---|---|---|
| `tiny_int8_conv` | success | success | TVM/H800 可 build/run 最小 native INT8 conv |
| `resnet_style_int8_block` | success | success | TVM/H800 可 build/run 残差式 INT8 block |
| `target_backbone_subnet_route` | success | success | 三个 label 的 width-matched native INT8 smoke route 完成 latency + energy |

三点 H800 native INT8 smoke 实测:

| label | width | latency_ms | energy_J / inference | dtype/QDQ gate |
|---|---:|---:|---:|---|
| `base` | `[64,128,256]` | 0.508514 | 0.144209023342 | `float32=0`, `quantize=0`, `dequantize=0` |
| `s0_024` | `[24,128,256]` | 0.450200 | 0.128370079340 | `float32=0`, `quantize=0`, `dequantize=0` |
| `s1_048` | `[64,48,256]` | 0.189394 | 0.046026392511 | `float32=0`, `quantize=0`, `dequantize=0` |

行表已写入:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_energy_smoke_rows_v1.jsonl
```

当前行数:

- `native_int8_latency_smoke_rows_v1.jsonl`: 3 行
- `native_int8_energy_smoke_rows_v1.jsonl`: 3 行

审阅版 summary:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_lowering_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_lowering_audit_latest.json
```

## 0.1 完整 ONNX topology native INT8 实测结果

已完成下一阶段 full ONNX topology route。最终采用 run:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_h800_v1/
```

新的 full ONNX row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_energy_rows_v1.jsonl
```

当前行数:

- `native_int8_full_onnx_latency_rows_v1.jsonl`: 3 行
- `native_int8_full_onnx_energy_rows_v1.jsonl`: 3 行

审阅版 summary:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_full_onnx_audit_latest.json
```

三点 H800 full ONNX topology native INT8 实测:

| label | latency_ms | energy_J / inference | op coverage | dtype/QDQ gate |
|---|---:|---:|---|---|
| `base` | 11.095775 | 2.853151923181 | `Conv=50`, `Relu=48`, `Add=16` | `float32=0`, `quantize=0`, `dequantize=0` |
| `s0_024` | 9.235392 | 2.290300810919 | `Conv=51`, `Relu=48`, `Add=16`, `Identity=46` | `float32=0`, `quantize=0`, `dequantize=0` |
| `s1_048` | 8.109461 | 2.118255342182 | `Conv=50`, `Relu=48`, `Add=16`, `Identity=44` | `float32=0`, `quantize=0`, `dequantize=0` |

实现边界:

- route spec: `full_onnx_topology_conv_relu_add_identity_v1`
- 从 H800 远端 `${V2X_DATA_ROOT}/s2_tvm/models/{base,s0_024,s1_048}_backbone.onnx` 读取真实 ONNX 拓扑。
- 支持 1x1 conv、group=32 的 3x3 conv、Relu、residual Add、Identity initializer alias、stage width changes。
- 所有 row 均为 `full_network_claim=false`。
- 没有写 AP measured, 没有使用 TRT, 没有使用 `static_qdq_synthetic_minmax` route。

## 1. 关键边界

这次解决的是旧 INT8 smoke 的核心实现问题: 不再走 `static_qdq_synthetic_minmax`, 不再把 QDQ-heavy / float32-heavy lowering 写成 native INT8。

但是必须保留一个重要边界:

- 本轮 route spec 是 `width_matched_qnn_conv_chain_v1`。
- 它是 native INT8 H800 smoke route, 不是完整 ONNX backbone graph lowering。
- 所有新增 row 都是 `full_network_claim=false`。
- 不要把这些结果写成 AP measured, full-network measured, 或完整 ONNX backbone measured。

真实 ONNX backbone 结构探测显示:

| label | Conv | Relu | Add | Identity | 主要难点 |
|---|---:|---:|---:|---:|---|
| `base` | 50 | 48 | 16 | 0 | 1x1 conv + group=32 3x3 conv + residual Add |
| `s0_024` | 51 | 48 | 16 | 46 | 额外 Identity, s0 第一层更窄 |
| `s1_048` | 50 | 48 | 16 | 44 | stage1 宽度变化 |

H800 TVM native INT8 group conv probe 已成功:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_group_conv_probe_v1/
```

这说明目前不是 TVM/H800 完全不能 build INT8 route。下一阶段应该进入完整 ONNX topology lowering, 而不是再解释 TVM 是否支持 INT8。

## 2. 原始产物位置

每个 label 均已产出以下文件:

```text
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/latency_result.json
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/energy_result.json
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/idle_power_samples.csv
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/active_power_samples.csv
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/telemetry_payload.json
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/tvm_operator_inventory.json
raw/int8_native_route/20260627_native_int8_h800_exdata_v3/{base,s0_024,s1_048}/native_int8_route_manifest.json
```

示例:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_exdata_v3/base/latency_result.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_exdata_v3/base/energy_result.json
```

执行日志与同步证据:

```text
raw/int8_native_route/20260627_native_int8_h800_exdata_v3_remote_run_log/remote_run_result.json
raw/int8_native_route/20260627_native_int8_h800_exdata_v3_pull_log/pull_result.json
raw/int8_native_route/20260627_native_int8_h800_exdata_pack_rsync_v4/rsync_result.json
```

## 3. 代码变更

新增:

```text
framework/stage2/native_int8_route.py
framework/tests/test_stage2_native_int8_route.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py
```

更新:

```text
framework/stage2/lut_productization.py
```

`lut_productization.py` 的更新点:

- 为 measured LUT row 补齐量化合同字段校验。
- 保持旧 QDQ INT8 仍要求 `engine_kind=tvm_vm`。
- 仅允许新的 `quant_method=h800_tvm_native_int8_backbone_subnet` 且 `quant_scope=backbone_subnet_native_int8` 使用 `engine_kind=tvm_graph_executor`。
- 继续禁止 TRT 或 non-H800 backend 冒充 H800 TVM measured。

## 4. 验证命令与结果

已执行:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
python -m py_compile framework/stage2/lut_productization.py framework/stage2/native_int8_route.py framework/tests/test_stage2_native_int8_route.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py
```

结果:

```text
Ran 7 tests in 0.001s
OK
```

额外 completion check 已验证:

- `capability_probe.json.status == success`
- 三类 attempt 均 `status == success`
- 三个 label 的五类 raw 文件均存在且非空
- 三个 label 的 `latency_ms > 0`
- 三个 label 的 `energy_J == joule_per_inference > 0`
- 两个 native INT8 row 文件各 3 行
- 所有 row 通过 `validate_lut_row`
- 所有 row 均为 `measurement_status=measured`, `precision=int8`, `full_network_claim=false`
- 新产物未写入 H800 密码或 `password-based SSH (disabled; use an SSH key)` 的 secret value

## 5. 为什么 v2 不采用

`20260627_native_int8_h800_exdata_v2` 也完成了 latency 与 energy 文件, 但当时 energy loop 对短 kernel 使用逐次同步 + `nvidia-smi` 采样, `base` 和 `s0_024` 的 idle-subtracted energy 被压成 0。

v3 修正为:

- power sampler 独立线程采样
- 至少 5 秒连续 inference burst
- 每 50 次 sync 一次, 避免把 GPU 活跃窗口采掉

因此最终采用 v3。

## 6. 下一阶段目标

下一阶段的主目标不应该再停留在 smoke route。建议目标:

1. 从 ONNX 读取真实 backbone 拓扑, 生成 native INT8 TE/TOPI graph。
2. 支持:
   - 1x1 conv
   - group=32 的 3x3 conv
   - Relu
   - residual Add
   - s0/s1 的 Identity 和宽度变化
3. 对 `base/s0_024/s1_048` 完成完整 ONNX topology native INT8 route build/run。
4. 复用 v3 的 H800 energy burst 采样方法。
5. 成功后写新的 row 文件, 不覆盖 smoke row:

```text
rows/native_int8_full_onnx_latency_rows_v1.jsonl
rows/native_int8_full_onnx_energy_rows_v1.jsonl
```

6. 如果完整 ONNX topology route 失败, blocker 必须精确到具体 op/shape/group/stride/pad, 不能再笼统写 TVM 不支持 INT8。

## 7. 下一轮 /goal 提示词

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 native INT8 工作。不要启动其他 agent。

先阅读:
multi_agent/methods/design/auto-tuning/progress/6_27/9_6_28_交接文档_阶段二_NativeINT8Smoke实测结果与完整ONNXRoute下一步.md

当前已完成:
- H800 + TVM native INT8 smoke route: 20260627_native_int8_h800_exdata_v3
- base/s0_024/s1_048 均已有 latency_ms 与 energy_J
- rows/native_int8_latency_smoke_rows_v1.jsonl 和 rows/native_int8_energy_smoke_rows_v1.jsonl 均为 3 行

下一阶段目标:
- 不再重复 smoke route。
- 实现完整 ONNX backbone topology 的 native INT8 route。
- 从 ${V2X_DATA_ROOT}/s2_tvm/models/{base,s0_024,s1_048}_backbone.onnx 读取真实 Conv/Relu/Add/Identity 拓扑。
- 支持 1x1 conv、group=32 3x3 conv、Relu、residual Add、stage width changes。
- 在 H800 + TVM 路线完成 base/s0_024/s1_048 的完整 ONNX topology native INT8 latency_ms 与 energy_J。
- 成功时写 raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/latency_result.json、energy_result.json、idle_power_samples.csv、active_power_samples.csv、telemetry_payload.json、tvm_operator_inventory.json。
- 成功时写 rows/native_int8_full_onnx_latency_rows_v1.jsonl 与 rows/native_int8_full_onnx_energy_rows_v1.jsonl。
- 所有 row 必须 full_network_claim=false, 不得写 AP measured, 不得使用 TRT, 不得使用 static QDQ route。
- 如果失败, 必须保存具体 op-level blocker: op name、op type、shape、group、stride、pad、stdout/stderr/traceback、build/run status、failure_reason。
```

## 8. 交接结论

本阶段已经证明并实测: H800 + TVM 可以执行 native INT8 route, 且 smoke hot path 不再是 QDQ-heavy / float32-heavy lowering。

当前已完成: 完整 ONNX backbone topology native INT8 lowering 的 H800 latency + energy smoke。

下一步应进入结果审计与总表集成: 将 full ONNX native INT8 rows 与 smoke rows 分开导入, 并与 FP32/FP16/QDQ INT8 做同口径比较。不要把这些 backbone/subnet rows 写成 AP measured 或 full network measured。
