# 8_6_27_交接文档_阶段二_NativeINT8BackboneSubnet加速计划

日期: 2026-06-27

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/7_6_27_交接文档_阶段二_FP16INT8补测数据位置与INT8根因解释.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_int8_next_step_review_latest.md`

实施计划:

```text
docs/superpowers/plans/2026-06-27-stage2-native-int8-backbone-subnet-acceleration.md
```

## 0. 目标重定向

下一阶段的第一目标不再是继续解释“INT8 为什么不快”, 而是解决当前实现路径的问题: QDQ-heavy / float32-heavy lowering。

本阶段只使用一个实验 agent。目标必须具体到可验收文件, 不允许只给分析性结论。

具体目标只有两个允许出口:

1. **成功出口:** 为 `base/s0_024/s1_048` 构建 native INT8 backbone/subnet route, 在 H800 上完成真实 `latency_ms` 和 `energy_J` 实测, 并把原始产物写到指定目录。
2. **阻塞出口:** 如果不能完成, 必须证明 TVM/H800 当前环境无法 build/run native INT8 route。这个证明必须包含 tiny INT8 conv、ResNet-style INT8 block、以及目标 backbone/subnet route 三类 build/run 证据, 不能只凭完整模型失败下结论。

指定产出目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/<run_id>/
```

成功出口下必须看到:

```text
raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/latency_result.json
raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/energy_result.json
raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/idle_power_samples.csv
raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/active_power_samples.csv
raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/telemetry_payload.json
```

行表输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_energy_smoke_rows_v1.jsonl
```

AP70 不作为本阶段主验收。AP 没有 real TVM/TVM VM eval source 时继续写 blocker, 不得声明 measured。

## 0.1 本轮落地状态

已新增 native INT8 证据契约和能力探测脚本:

```text
framework/stage2/native_int8_route.py
framework/tests/test_stage2_native_int8_route.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py
```

本机调试 run:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_local_target_debug_v2/
```

本机环境是 RTX 4090, 不是 H800。因此这些文件只能证明脚本和 TVM native INT8 build/run 路径可执行, 不能作为目标表的 measured latency/energy:

| attempt | local status | local latency_ms | dtype/QDQ 结论 |
|---|---:|---:|---|
| tiny INT8 conv | success | 0.008141 | `float32=0`, `quantize=0`, `dequantize=0` |
| ResNet-style INT8 block | success | 0.023757 | `float32=0`, `quantize=0`, `dequantize=0` |
| target route/base | debug only, non-H800 | 0.059392 | `float32=0`, `quantize=0`, `dequantize=0` |
| target route/s0_024 | debug only, non-H800 | 0.050429 | `float32=0`, `quantize=0`, `dequantize=0` |
| target route/s1_048 | debug only, non-H800 | 0.046085 | `float32=0`, `quantize=0`, `dequantize=0` |

当前没有写入 `rows/native_int8_latency_smoke_rows_v1.jsonl` 或 `rows/native_int8_energy_smoke_rows_v1.jsonl`, 因为真实验收要求必须在 H800 上完成。脚本内置 H800 guard: 非 H800 只能生成 debug artifact, 不允许写 measured rows, 不允许生成 `energy_J`。

H800 非交互式访问探测:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_access_probe_v1/h800_access_probe.json
```

当前返回:

```text
${V2X_REMOTE_USER}@<PRIVATE_HOST>: Permission denied (publickey,password).
```

这不是 TVM INT8 能力 blocker, 只说明当前会话没有可用的非交互式 H800 登录通道。拿到 H800 会话后必须继续执行真实测量。

H800 运行命令已写入:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/h800_native_int8_run_command.json
```

## 0.2 H800 native INT8 smoke 实测结果

已完成单实验 agent 直接执行, 没有启动其他 agent。由于 H800 远端 `/home` 根分区 100% 满, 新脚本和输出先放在 `${V2X_DATA_ROOT}/s2_tvm/native_int8_pack` 与 `${V2X_DATA_ROOT}/s2_tvm/native_int8_results/`, 完成后已 rsync 回本地指定目录。

最终采用 run:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260627_native_int8_h800_exdata_v3/
```

行表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_energy_smoke_rows_v1.jsonl
```

审阅摘要:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_lowering_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_lowering_audit_latest.json
```

结果:

| label | latency_ms | energy_J / J-per-inference | dtype/QDQ |
|---|---:|---:|---|
| base | 0.508514 | 0.144209023342 | `float32=0`, `quantize=0`, `dequantize=0` |
| s0_024 | 0.450200 | 0.128370079340 | `float32=0`, `quantize=0`, `dequantize=0` |
| s1_048 | 0.189394 | 0.046026392511 | `float32=0`, `quantize=0`, `dequantize=0` |

三类 attempt 均成功:

- `tiny_int8_conv`
- `resnet_style_int8_block`
- `target_backbone_subnet_route`

重要边界:

- 这些 row 是 native TVM INT8 H800 smoke, `quant_method=h800_tvm_native_int8_backbone_subnet`, `engine_kind=tvm_graph_executor`, `full_network_claim=false`。
- 本轮 route spec 是 `width_matched_qnn_conv_chain_v1`, 不是完整 ONNX backbone graph lowering。不要把它写成 full ONNX backbone/AP coverage。
- 额外 ONNX structure probe 显示真实 backbone 约有 `Conv=50/51`, `Relu=48`, `Add=16`; 真实完整 ONNX native INT8 route 的下一步是覆盖 1x1 conv、group=32 3x3 conv、Relu、Add 拓扑。
- H800 TVM native INT8 group conv probe 已成功, 因此目前不是 TVM/H800 无法 build INT8 route 的 blocker。

## 1. 当前问题基线

当前 INT8 smoke 的问题不是“量化天然对速度不敏感”, 而是实现路径没有证明命中 native INT8 kernel coverage。

已知证据:

| evidence | location | meaning |
|---|---|---|
| QDQ route script | `raw/int8_route/stage2_h800_int8_route_attempt.py` | 使用 ONNXRuntime static QDQ + synthetic calibration |
| route recipe | `raw/int8_route/20260627_083401/{base,s0_024,s1_048}/quant_recipe.json` | `quant_method=static_qdq_synthetic_minmax` |
| precision summary | `raw/int8_route/20260627_083401/{base,s0_024,s1_048}/layer_precision_summary.json` | 有大量 QDQ 节点和 int8 initializer, 但这不等于 native INT8 kernel |
| lowered inventory | `raw/quant_speed_root_cause/20260627_quant_speed_profile_v1/{base,s0_024,s1_048}/int8/tvm_operator_inventory.json` | lowered graph 仍有大量 float32 token |
| manual lowered evidence | `exports/quant_speed_manual_lowered_evidence_int8_qdq_v1.json` | 看到 `dequantize*` 和 `fused_conv*_quantize*_dequantize*` 函数 |

代表性 `s0_024` inventory 显示:

- ONNX op: `QuantizeLinear=68`, `DequantizeLinear=124`, `Conv=51`
- lowered dtype token: `float32=882`, `int8=28`, `uint8=136`, `int32=106`
- 这说明当前 INT8 更像 QDQ artifact + float32-heavy lowered route, 不是干净的 int8 conv hot path。

## 2. 新阶段验收标准

| gate | pass 条件 |
|---|---|
| route identity | 新 route 标记为 `quant_method=h800_tvm_native_int8_backbone_subnet`, 不复用旧 `static_qdq_synthetic_minmax` |
| QDQ gate | hot path 不再由 `QuantizeLinear` / `DequantizeLinear` 主导; boundary Q/DQ 必须单独标注 |
| dtype gate | conv-heavy path 保持 int8/uint8 tensor + int32 accumulation; float32 只允许在 scale/final dequant/unsupported boundary |
| schedule gate | FP32/true-FP16/native INT8 比较必须写清 schedule policy 和 artifact digest |
| latency gate | `base/s0_024/s1_048` 都必须有 `latency_result.json`; `latency_ms` 单位必须是 ms; `s0_024/s1_048` 应低于当前 QDQ INT8 和 FP32 |
| energy gate | `base/s0_024/s1_048` 都必须有 `energy_result.json`、`idle_power_samples.csv`、`active_power_samples.csv` 和 `telemetry_payload.json`; `energy_J` 必须来自 H800 telemetry |
| blocker gate | 如果 build/run 失败, 必须给出 tiny INT8 conv、ResNet-style INT8 block、目标 backbone/subnet route 的 build/run logs 和 traceback |
| scope gate | 全部写 `full_network_claim=false`, scope 只写 backbone/subnet |
| AP gate | AP70 不作为本阶段主验收; 只有 real TVM/TVM VM eval source 才能 measured, 否则保留 no-claim/blocker |

## 3. 技术路线

### 3.1 P0: 能力探测

新增:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py
```

探测内容:

- TVM 是否可 build/run tiny CUDA INT8/QNN conv。
- TVM 是否可 build/run ResNet-style INT8 block, 例如 conv + add/relu/requant 的残差块。
- 是否支持 int8/uint8 input/weight + int32 accumulation。
- QNN path 是否可用; 不可用时是否需要 TensorIR fallback。
- 输出 `capability_probe.json`, 结论只能是:
  - `qnn_native_int8_available`
  - `tensorir_int8_required`
  - `blocked_by_tvm_cuda_int8_capability`

若输出 `blocked_by_tvm_cuda_int8_capability`, 必须同时写:

```text
raw/int8_native_route/<run_id>/tiny_int8_conv_attempt.json
raw/int8_native_route/<run_id>/resnet_style_int8_block_attempt.json
raw/int8_native_route/<run_id>/capability_probe_stdout.txt
raw/int8_native_route/<run_id>/capability_probe_stderr.txt
```

### 3.2 P1: native INT8 route 原型

新增:

```text
raw/int8_native_route/stage2_h800_native_int8_route_attempt.py
raw/int8_native_route/stage2_h800_native_int8_latency_smoke.py
raw/int8_native_route/stage2_h800_native_int8_profile.py
```

优先顺序:

1. 先做 `s0_024`, 因为当前 FP32/FP16/QDQ INT8 的 latency 接近, 最适合判断 route 是否真正变快。
2. 如果 QNN route 可行, 保持中间 conv-heavy tensor 量化, 只在 unsupported boundary dequant。
3. 如果 QNN route 不能生成 H800 CUDA INT8 kernel, 改做 TensorIR conv-heavy prototype, 并把 `quant_scope` 写成 `backbone_subnet_partial_native_int8`。
4. route build 成功但 latency 不快时, 先查 lowered dtype、layout conversion、VM overhead 和 schedule, 不扩展到 60 行。

### 3.3 P2: latency 与 energy 实测

成功 route 的实测顺序:

1. `s0_024` 先完成 native INT8 build + latency + energy。
2. `s0_024` 成功后扩展 `s1_048`。
3. `base` 也必须测 latency + energy, 但 speedup 因果声明要保留 schedule fairness audit, 因为当前 `base` FP32 与 FP16/INT8 差距异常大。
4. 每个 label 成功后都要在指定目录写 `latency_result.json`、`energy_result.json`、`idle_power_samples.csv`、`active_power_samples.csv`、`telemetry_payload.json`。
5. 三个 label 都完成后, 才写 `rows/native_int8_latency_smoke_rows_v1.jsonl` 和 `rows/native_int8_energy_smoke_rows_v1.jsonl`。
6. 旧 QDQ INT8 rows 保留为 diagnostic/baseline, 不能覆盖成 native INT8。

## 4. 下一阶段产物

必须新增或更新:

| artifact | purpose |
|---|---|
| `docs/superpowers/plans/2026-06-27-stage2-native-int8-backbone-subnet-acceleration.md` | 可执行实施计划 |
| `raw/int8_native_route/<run_id>/capability_probe.json` | H800/TVM native INT8 能力证据 |
| `raw/int8_native_route/<run_id>/tiny_int8_conv_attempt.json` | 最小 INT8 conv build/run 证据 |
| `raw/int8_native_route/<run_id>/resnet_style_int8_block_attempt.json` | ResNet-style INT8 block build/run 证据 |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/native_int8_route_manifest.json` | route identity 和 digest |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/latency_result.json` | native INT8 H800 latency 实测 |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/energy_result.json` | native INT8 H800 energy 实测 |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/idle_power_samples.csv` | idle baseline telemetry |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/active_power_samples.csv` | active inference telemetry |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/telemetry_payload.json` | energy measurement command/env payload |
| `raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/tvm_operator_inventory.json` | lowered dtype/QDQ 证据 |
| `exports/native_int8_lowering_audit_latest.md/.json` | 审阅用 lowering 审计 |
| `rows/native_int8_latency_smoke_rows_v1.jsonl` | 三点 native INT8 latency_ms 行表 |
| `rows/native_int8_energy_smoke_rows_v1.jsonl` | 三点 native INT8 energy_J 行表 |

## 5. 单实验 agent 职责

只启动一个实验 agent。该 agent 负责从 capability probe 到 latency/energy 产物的完整闭环:

1. 实现并运行 `stage2_h800_native_int8_capability_probe.py`。
2. 若 TVM/H800 能 build/run tiny INT8 conv 或 ResNet-style INT8 block, 继续实现 `stage2_h800_native_int8_route_attempt.py`。
3. 优先完成 `s0_024`, 再扩展 `s1_048`, 最后完成 `base`。
4. 对三个 label 都写出 native INT8 latency 和 energy 原始文件。
5. 更新 `rows/native_int8_latency_smoke_rows_v1.jsonl` 与 `rows/native_int8_energy_smoke_rows_v1.jsonl`。
6. 如果不能完成, 必须输出可以审计的 TVM INT8 build/run blocker, 包括 stdout/stderr、traceback、attempt json 和 lowered inventory。

硬性规则:

- 不允许把旧 `static_qdq_synthetic_minmax` route 标成 native INT8。
- 不允许只产出 latency 不产出 energy。
- 不允许用非 H800 telemetry 或推算值冒充 `energy_J`。
- 不允许把 AP blocker 写成 AP measured。

## 6. 下一轮 /goal 提示词

```text
/goal 启动一个实验 agent 执行 Stage2 native INT8 backbone/subnet latency+energy 闭环。不要启动其他 agent。

工作目录: ${V2X_ROOT}

主目标:
- 解决当前 INT8 的 QDQ-heavy / float32-heavy lowering 问题。
- 为 base/s0_024/s1_048 构建 native INT8 backbone/subnet route。
- 在 H800 上完成 native INT8 latency_ms 与 energy_J 实测。
- 成功时必须在 ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/<run_id>/ 下产出三点 latency_result.json、energy_result.json、idle_power_samples.csv、active_power_samples.csv、telemetry_payload.json。
- 同步写入 rows/native_int8_latency_smoke_rows_v1.jsonl 与 rows/native_int8_energy_smoke_rows_v1.jsonl。
- 如果失败, 必须证明 TVM/H800 无法 build/run native INT8 route: tiny INT8 conv、ResNet-style INT8 block、目标 backbone/subnet route 三类 attempt 都要有 json、stdout/stderr 和 traceback。

启动命令模板:

multi_agent_v1.spawn_agent(
  agent_type="worker",
  message="实验 agent: 在 ${V2X_ROOT} 中执行 multi_agent/methods/design/auto-tuning/progress/6_27/8_6_27_交接文档_阶段二_NativeINT8BackboneSubnet加速计划.md。只启动本 agent, 不要再派生其他 agent。优先实现 raw/int8_native_route 的 TVM CUDA INT8 capability probe、native INT8 route attempt、native INT8 latency smoke、native INT8 energy telemetry。成功时必须在 raw/int8_native_route/<run_id>/{base,s0_024,s1_048}/ 写 latency_result.json、energy_result.json、idle_power_samples.csv、active_power_samples.csv、telemetry_payload.json, 并写 rows/native_int8_latency_smoke_rows_v1.jsonl 与 rows/native_int8_energy_smoke_rows_v1.jsonl。失败时必须用 tiny INT8 conv、ResNet-style INT8 block、目标 backbone/subnet route 三类 attempt 证明 TVM/H800 不能 build/run native INT8。不要把 static QDQ route 当 native INT8; 所有 latency 对外统一写 ms; energy 使用 J/inference; full_network_claim=false。"
)
```

## 7. 停止条件

成功:

- `base/s0_024/s1_048` 都有 native INT8 `latency_result.json` 和 `energy_result.json`。
- `base/s0_024/s1_048` 都有 `idle_power_samples.csv`、`active_power_samples.csv`、`telemetry_payload.json`。
- `rows/native_int8_latency_smoke_rows_v1.jsonl` 有三条 native INT8 latency_ms row。
- `rows/native_int8_energy_smoke_rows_v1.jsonl` 有三条 native INT8 energy_J row。
- `s0_024` 和 `s1_048` native INT8 latency_ms 低于 FP32 和当前 QDQ INT8。
- `base` 有 native INT8 measured latency_ms/energy_J, 并有 schedule fairness audit。
- 旧 QDQ INT8 与新 native INT8 在表和 summary 中明确分开。

阻塞:

- TVM/H800 无法 build 或 run native INT8/QNN/TensorIR route。此时必须输出 capability-probe blocker, 包括 tiny INT8 conv、ResNet-style INT8 block、目标 backbone/subnet route 三类 attempt 的 json、stdout/stderr、traceback 和 lowered inventory。保留旧 QDQ INT8 为 diagnostic-only。

不可接受:

- lowered graph 仍是 QDQ-heavy / float32-heavy, 但 summary 写成 native INT8 acceleration。
- 只有 latency, 没有真实 H800 telemetry energy 产物。
- 只有完整模型 build 失败, 没有 tiny INT8 conv 和 ResNet-style INT8 block 失败证据。
- AP 无 real TVM/TVM VM eval source, 但写成 AP70 measured。
