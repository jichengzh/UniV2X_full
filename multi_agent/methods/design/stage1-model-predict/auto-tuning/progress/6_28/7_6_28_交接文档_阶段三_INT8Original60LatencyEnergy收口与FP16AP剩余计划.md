# 13_6_28_交接文档_阶段三_INT8Original60LatencyEnergy收口与FP16AP剩余计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/12_6_28_交接文档_阶段三_NativeINT8FullONNXBatch001与FP16INT8补点收口计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`

## 0. 本轮结论

本轮完成了 original60 的 native INT8 full-ONNX backbone/subnet latency 和 energy 大范围补点收口:

| precision | metric | measured | target | status |
|---|---:|---:|---:|---|
| INT8 native full-ONNX | latency_ms | 60 | 60 | complete |
| INT8 native full-ONNX | energy_J / inference | 60 | 60 | complete |
| INT8 native full-ONNX | AP70 | 0 | 60 | pending |
| FP16 true route | latency_ms | 2 | 60 | pending 58 |
| FP16 true route | energy_J / inference | 2 | 60 | pending 58 |
| FP16 true route | AP70 | 0 | 60 | pending |

所有新增 INT8 measured row 均保持:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

统一口径不变:

```text
latency_ms = H800 + TVM measured backbone/subnet end-to-end module latency.
```

这些不是 full perception network latency, 也不是真实 RSU 物理设备绝对速度。

## 1. H800 执行记录

已完成两个 native INT8 full-ONNX 批次:

| run_id | labels | status |
|---|---:|---|
| `20260628_native_int8_full_onnx_original60_batch001` | 3 | success |
| `20260628_native_int8_full_onnx_original60_batch002_retry2_57labels` | 57 | success |

batch002 运行中有两个启动层面的非模型问题:

1. 第一次启动失败: 远端 `set -u` 下 `PYTHONPATH` 未设置, shell 展开失败, runner 未启动。
2. 第二次启动失败: 非交互 shell 中没有 `python` 命令, runner 未启动。

修正后使用:

```text
${V2X_DATA_ROOT}/tvm310/bin/python
PYTHONPATH=${V2X_DATA_ROOT}/s2_tvm/native_int8_pack:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages
LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib
```

batch002 retry2 结果:

```text
status_counts = {"success": 57}
returncode = 0
```

本地产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001_remote_run_log/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels_remote_run_log/
```

最终 INT8 row:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

行数:

```text
native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60
native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60
```

## 2. Canonical 表与 review 已刷新

已更新:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/original60_quant_three_metric_summary_latest.csv
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

刷新后的 canonical 计数:

| table | status |
|---|---|
| latency | `int8 measured=60`, `fp16 measured=2`, `fp16 no_claim=58` |
| energy | `int8 measured=60`, `fp16 measured=2`, `fp16 no_claim=58` |
| AP | `fp16 no_claim=60`, `int8 no_claim=60` |

Completion queue summary:

```json
{
  "latency": {"measured": 62, "no_claim": 58},
  "energy": {"measured": 62, "no_claim": 58},
  "ap": {"no_claim": 120}
}
```

解释:

- `latency/energy measured=62` 是 `INT8 60 + FP16 2`。
- FP16 的 2 个 measured 是既有 `s0_024/s1_048` true-FP16 smoke。
- AP 对 FP16/INT8 仍没有合规 measured source。

## 3. 代码变更

更新:

```text
scripts/stage2_generate_original60_quant_state_coverage.py
framework/tests/test_stage2_lut_productization.py
```

新增能力:

- `--native-int8-full-onnx-latency-rows`
- `--native-int8-full-onnx-energy-rows`
- canonical generator 现在能导入 native INT8 full-ONNX latency/energy rows。
- native INT8 full-ONNX rows 会覆盖旧 QDQ / TVM VM smoke rows。
- summary MD 说明改为按每行 status/claim gate 解读 measured, 不再沿用旧的 “不是 FP32/INT8 measured claim” 文案。

新增测试:

```text
test_original60_quant_state_coverage_prefers_native_int8_full_onnx_rows
```

该测试证明:

- 同 label 同时存在旧 QDQ row 和 native full-ONNX row 时, canonical latency 选择 native row。
- native energy row 可进入 canonical energy 表。
- `engine_kind=tvm_graph_executor`, `quant_scope=backbone_subnet_native_int8`, `full_network_claim=false` 保持不变。

## 4. 验证

已执行:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_lut_productization.Stage2Original60QuantCoverageCliTest framework.tests.test_stage2_original60_quant_completion
python -m py_compile scripts/stage2_generate_original60_quant_state_coverage.py framework/stage2/original60_quant_completion.py framework/tests/test_stage2_lut_productization.py framework/tests/test_stage2_original60_quant_completion.py
```

结果:

```text
Ran 12 tests in 1.403s
OK
py_compile exit 0
```

行级校验:

```text
native INT8 latency rows: 60 rows, 60 unique labels
native INT8 energy rows: 60 rows, 60 unique labels
missing latency labels: []
missing energy labels: []
contract failures: 0
missing raw required files: 0
latency_ms range: 1.722880 to 10.572736
energy_J range: 0.2598788607420836 to 2.796973770305826
```

Secret scan:

```text
No literal credential value found in touched docs, generated review MD, summary MD, or patched code.
```

## 5. 剩余缺口

### 5.1 FP16 latency/energy

当前:

```text
FP16 latency measured = 2 / 60
FP16 energy measured = 2 / 60
```

剩余 58 个 FP16 labels 可从 completion queue 提取。已有 runner:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/stage2_h800_fp16_true_smoke.py
```

注意:

- 该 runner 当前只产品化 latency row, 不产品化 energy row。
- 启动 H800 前要审查远端路径和 Python/TVM 环境, 避免重复 batch002 的 `PYTHONPATH` / `python` 非交互 shell 问题。
- FP16 必须保留 true-FP16 evidence: FP16 ONNX、layer precision summary、input/initializer dtype。

### 5.2 AP70

当前:

```text
FP16 AP70 measured = 0 / 60
INT8 AP70 measured = 0 / 60
```

AP 不能由 backbone latency/energy 推断。下一阶段必须选择一条合规路线:

1. 恢复/实现 true-FP16 eval/import route。
2. 连接 native INT8 backbone artifact 到 eval pipeline, 或写出逐点 precise blocker。
3. 任何 AP measured row 必须包含 dataset split、ckpt、eval command、AP30/AP50/AP70、raw eval output。
4. 禁止 predicted/model-fit/interpolated AP, 禁止 TRT reference 冒充 H800 TVM measured。

## 6. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 补点收口。不要启动 agent team, 单 agent 执行即可。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/13_6_28_交接文档_阶段三_INT8Original60LatencyEnergy收口与FP16AP剩余计划.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl

当前已完成:
- native INT8 full-ONNX original60 latency 60/60 measured。
- native INT8 full-ONNX original60 energy 60/60 measured。
- canonical latency/energy/AP 三表已刷新。
- completion review/gap 已刷新。
- 所有新增 INT8 measured row 均为 H800 + TVM backbone/subnet module, full_network_claim=false。

当前未完成:
- FP16 latency 58 个 label 待补。
- FP16 energy 58 个 label 待补。
- FP16 AP70 60 个 label 待补。
- INT8 AP70 60 个 label 待补。

下一步执行:
1. 先产品化或审查 FP16 true latency runner 的 H800 启动路径, 使用 ${V2X_DATA_ROOT}/tvm310/bin/python 和显式 PYTHONPATH/LD_LIBRARY_PATH。
2. 跑 FP16 latency 剩余 58 labels, 输出 rows/fp16_true_original60_latency_rows_v1.jsonl, 并回填 canonical latency 表。
3. 产品化 FP16 energy runner, 复用 INT8 的 idle-subtracted telemetry burst 方法, 输出 rows/fp16_true_original60_energy_rows_v1.jsonl。
4. 设计并实现 FP16/INT8 AP70 合规 eval/import route; AP 不得预测或用 TRT reference 冒充。
5. 每一轮都刷新 original60_quant_three_metric_summary_latest.* 和 fp16_int8_original60_completion_review/gap。
6. 遇到失败不要停止整批: 保存 stdout/stderr、runner command、GPU id、raw artifact、failure reason, 做最小复现或 op-level blocker, 修改后重试; 确认不可解后才写 quarantine/no_claim。
```
