# 57_6_29_交接文档_阶段三_FP32Energy异常复盘与冷启动防错计划

## 0. 本文用途

本文用于清理上下文后的冷启动续接, 重点不是继续堆数据, 而是先把 2026-06-29 暴露出的 FP32 energy 异常做复盘, 并把下一阶段的数据补点规则写清楚。

本轮已完成的纠偏:

```text
1. FP32 energy 已用 H800 TVM threaded_window power telemetry 全量重测 original60, 60/60 成功。
2. 旧 FP32 post_sync_per_iter energy row 已降级为历史审计证据, 不再作为 canonical FP32 energy。
3. 当前总表已经改为优先读取 rows/fp32_original60_energy_threaded60_rows_v1.jsonl。
4. 已抽查 5 个 FP16 TIR/workdir, 结论是 FP16 dtype 存在, 但没有 tensorcore/wmma 证据。
```

新窗口启动后的第一原则:

```text
不要只看 measured/no_claim 计数。任何跨 precision 比较都必须同时看:
metric value + measurement_source + evidence_scope + schedule_policy + sampling_mode/raw_artifact + quality_gate_status。
```

## 1. 启动初期必读材料

启动后先读本文, 再读以下审查文件, 再改代码或启动 GPU 任务:

```text
本文:
multi_agent/methods/design/auto-tuning/progress/6_27/57_6_29_交接文档_阶段三_FP32Energy异常复盘与冷启动防错计划.md

当前三精度总表:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json

当前最可信数据审查记录:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.json

FP32 energy threaded_window 全量重测报告:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.json

FP16 TIR 抽查报告:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.json

上一份总线交接:
multi_agent/methods/design/auto-tuning/progress/6_27/56_6_28_交接文档_阶段三_FP32FP16INT8五点核对与FullVal收口计划.md

FP16 AP/checkpoint 主线:
multi_agent/methods/design/auto-tuning/progress/6_27/54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md

INT8 AP/ScaleAware 主线:
multi_agent/methods/design/auto-tuning/progress/6_27/55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md
```

快速核对命令:

```bash
cd ${V2X_ROOT}
sed -n '1,120p' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md
sed -n '1,120p' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.md
sed -n '1,80p' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.md
```

## 2. 事故复盘

### 2.1 Failure Capture

```text
Session / task:
- original60 FP32/FP16/INT8 三指标总表补齐与口径审查。

Failure:
- 总表中旧 FP32 energy 被标为 measured, 但数值与 FP16/INT8 不可比。
- 旧 FP32 dynamic watt 大量低于 50W, 导致 FP32 energy 看起来异常偏低。
- 用户指出 FP16 energy 严格多于 FP32 energy 不合理后, 才触发 full remeasure。

Last successful step:
- 已定位旧 FP32 energy 使用 post_sync_per_iter 采样口径, 并新增 threaded_window 采样模式。
- 已完成 5 点 smoke 后扩展到 original60 全量 60/60 FP32 energy 重测。

Repeated pattern seen:
- 过度信任 measured status。
- 没有先把 sampling method/schedule/raw artifact 展开给用户审查。
- 对跨 precision 比较使用了不同测量协议的数据。
```

### 2.2 根因反思

这次异常不是单一脚本 bug, 而是数据治理口径失控:

```text
1. 测量协议不一致
   旧 FP32 energy 使用 post_sync_per_iter: 每次 VM 调用后同步并采样, 容易错过 GPU active power window。
   FP16/INT8 energy 使用更接近 active window 的采样口径。
   结果是 measured=true, 但跨 precision 不可比。

2. 采样窗口过短且位置错误
   旧 FP32 row 的 dynamic watt 大量低于 50W, 在 H800 上不符合真实 backbone/subnet 推理负载。
   这说明 telemetry 捕获到的主要是 idle/near-idle 区间, idle baseline subtraction 进一步放大了低能耗错觉。

3. status 字段表达力不足
   energy_status=measured 只能说明脚本产生了 measurement row, 不能说明它与其他 precision 同协议、可横向比较。
   缺少 sampling_mode、active_window、dynamic_watt gate、raw_artifact 完整性 gate 的硬性阻断。

4. 总表过早合并
   总表把 true measurement、historical remap、reference AP、no_claim 放在同一张表里是必要的,
   但如果不同时展示 source/schedule/sampling, 容易让读者误以为所有 measured 都是同等级证据。

5. FP16 性能解释也存在风险
   5 个 FP16 抽查点显示 ONNX/input/initializer 是 float16,
   但复用的 MetaSchedule workdir 没有 float16/wmma/tensorcore 证据。
   因此 FP16 energy/latency 不优于 FP32, 不能简单解释为硬件反常, 更可能是 lowering/schedule 没有真正走 tensor-core optimized route。

6. Agent 侧问题
   前序判断过度依赖交接记忆和 measured 计数, 没有第一时间按 raw artifact 和 schema 重新审计。
   后续任何结论必须先从当前文件系统重新读取, 不允许用上下文记忆替代证据。
```

### 2.3 已采取恢复动作

```text
1. 在 scripts/stage2_h800_run_measurement_job.py 增加 --energy-sampling-mode threaded_window。
2. 使用后台线程 50ms 采样、active window 至少 5s、每 50 iter sync 的方式重测 FP32 energy。
3. 先跑 5 个 label smoke, 确认旧低功耗异常消失, 再跑 original60 全量。
4. 在 scripts/stage2_generate_original60_quant_state_coverage.py 中将 FP32 energy 优先级切到 threaded60 row。
5. 增加/更新 unittest, 防止后续 summary generator 回退到旧 FP32 energy row。
6. 生成新的 source audit, 明确旧 post_sync_per_iter 只能保留为历史审计证据。
```

### 2.4 恢复结果

```text
FP32 energy threaded60:
- row_count: 60
- unique_label_count: 60
- finish_success: 60
- finish_fail: 0
- schedule: metaschedule_tuned 56, default 4
- dynamic watt < 50W count: 0

旧/新对比:
- old_fp32_dynamic_w_median: 22.842 W
- new_fp32_dynamic_w_median: 190.338 W
- fp16_dynamic_w_median: 211.023 W
- old_fp32_j_median: 4.321 J
- new_fp32_j_median: 6.731 J
- fp16_j_median: 7.717 J
```

## 3. 当前权威数据状态

权威总表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
```

当前覆盖计数:

| precision | latency measured | energy measured | AP measured |
|---|---:|---:|---:|
| fp32 | 58/60 | 60/60 | 5/60 |
| fp16 | 60/60 | 60/60 | 60/60 |
| int8 | 60/60 | 60/60 | 1/60 |

来源分解:

| precision | latency source | energy source | AP source |
|---|---|---|---|
| fp32 | historical_true_measurement_reclassified: 56; true_measurement: 2; no_claim: 2 | true_measurement: 60 | stage2_original60_ap_source_map_reference: 5; no_claim: 55 |
| fp16 | true_measurement_smoke: 60 | true_measurement_smoke: 60 | true_eval: 60 |
| int8 | true_measurement_smoke: 60 | true_measurement_smoke: 60 | true_eval: 1; no_claim: 59 |

重要解释:

```text
1. FP32 energy 现在可以作为当前 canonical energy 使用。
2. FP32 latency 仍是 direct smoke + historical reclassified 混合, 不是 60/60 新一轮直接同协议重测。
3. FP32 AP 只有 5/60 reference AP, 不是 60/60 precision-specific TVM FP32 AP。
4. FP16 三轴在总表中为 60/60, 但 FP16 TIR 没有 tensorcore/wmma 证据, 所以不能宣称已经完成 tensor-core optimized FP16 加速。
5. INT8 latency/energy 60/60, AP 只有 1/60 full eval measured; 之前 5-sample smoke 不能写成 measured AP。
```

## 4. 现有数据空白与风险

### 4.1 FP32

```text
已可信:
- energy: 60/60 H800 TVM threaded_window 全量重测。

仍有空白:
- latency: 58/60 measured, 其中 56 是 historical_true_measurement_reclassified, 2 是 direct true_measurement, 2 no_claim。
- AP: 5/60 reference AP, 55/60 no_claim。

风险:
- 如果要做严格三精度同协议 latency 对比, FP32 latency 应补做 60/60 direct same-protocol rerun, 或至少补齐 2 个 no_claim 并标注 56 个 historical reclassified。
- 如果要做严格三轴精度表, FP32 AP 不能靠 reference AP 直接冒充 60/60 measured。
```

### 4.2 FP16

```text
已可信:
- latency: 60/60 measured。
- energy: 60/60 measured。
- AP: 60/60 true_eval measured。

仍有空白:
- TIR/tuning 证据不足。5 个抽查点均为 fp16_dtype_present_no_tensorcore_evidence。

风险:
- 当前 FP16 数据可以用于“当前 TVM FP16 route 实测表现”,
  但不能用于证明“真正 tensor-core optimized FP16 已实现”。
- 如果后续修复 FP16 lowering/schedule, latency/energy 必须重新测, 不能沿用旧 FP16 latency/energy。
```

### 4.3 INT8

```text
已可信:
- latency: 60/60 measured。
- energy: 60/60 measured。
- AP: 1/60 full eval measured。

仍有空白:
- AP: 59/60 no_claim。
- 5-sample smoke 只能证明非塌缩趋势, 不能证明 full-val AP。
- scale-aware route 若仍包含 float32 requant/Add/ReLU 实值换算, 不能宣称 fully integer fixed-point INT8 acceleration 完成。

风险:
- INT8 AP smoke 高于 FP16 full-val 不能解释为 INT8 更准, 因为协议不一致。
- 后续必须以 processed_samples=1789、pred_nonempty_count 合理、AP30/AP50/AP70 趋势合理为 measured AP gate。
```

## 5. 防止同类错误的固定规则

后续任何总表更新必须遵守:

```text
1. latency 单位统一为 ms; energy 单位统一为 joule/inference。
2. measured 只能表示有实测 row, 不能单独表示可横向比较。
3. 每个 measured row 必须能追溯 raw_artifact、run_id、source_files、schedule_policy、measurement_source。
4. energy 跨 precision 比较前必须确认 sampling mode 一致或明确标注不同。
5. H800 energy 如果 dynamic_watt_avg < 50W, 默认进入 quarantine, 不能自动写 canonical。
6. 同一张 review 表必须展示 energy schedule, 不允许只展示 energy J。
7. AP measured 只能来自 full eval 或明确允许的 same-protocol eval; 5-sample smoke 禁止写正式 AP measured。
8. historical reclassified/remap row 必须在 measurement_source 中显式保留 reclassified/remap 字样。
9. JSON/CSV/MD 字段读取必须先看 schema, 不能假设字段名。例如当前 energy 字段是 energy_j_per_inference, 不是 energy_j。
10. 任何“FP16/INT8 已加速”的表述必须有 TIR/lowering 证据支持, 至少包括 dtype、workload、tensorization 或 integer route 证据。
11. 发现反直觉结果时, 先做 3-5 点 paired smoke 和 raw telemetry 审计, 再扩展全量。
12. 新窗口启动后必须先读 source audit, 不允许直接沿用旧交接结论。
```

建议新增或继续维护的自动 gate:

```text
scripts/stage2_generate_original60_quant_state_coverage.py:
- 保持 FP32 threaded60 row 优先级高于旧 remeasured/remap row。
- 输出 summary 时继续保留 schedule/source/status 列。

建议新增:
- scripts/stage2_validate_original60_quant_summary.py
  检查 180 rows schema、energy_j_per_inference 非空、source/status 一致、raw_artifact 存在、
  FP32 threaded60 dynamic_watt gate、AP smoke/full-val gate、latency 单位 gate。
```

## 6. 下一阶段计划

### P0: 冻结当前可信总表并加审计 gate

目标:

```text
确保 original60_quant_three_metric_summary_latest.* 不再因为旧 row 优先级、字段误读或 smoke/full-val 混淆而回退。
```

执行:

```text
1. 读取 summary json schema, 明确字段:
   latency_ms, energy_j_per_inference, ap70,
   latency_measurement_source, energy_measurement_source, ap_measurement_source,
   latency_schedule_policy, energy_schedule_policy, quality_gate_status。

2. 写或运行 summary validator:
   - 总行数必须是 180。
   - precision 分布必须是 fp32/fp16/int8 各 60。
   - FP32 energy 必须 60/60 measured 且 quality_gate_status 包含 fp32_energy_threaded_window_h800。
   - FP32 energy dynamic_watt < 50W count 必须为 0。
   - INT8 AP 不能把 5-sample smoke 写成 measured。
   - FP16 TIR audit 结果必须在 source audit 中保留限制说明。

3. 刷新并审阅:
   exports/original60_quant_measurement_source_audit_latest.md
   exports/original60_quant_three_metric_summary_latest.md
```

验收:

```text
有一条命令能重跑 validator 并输出 PASS/FAIL;
任何 FAIL 都不能继续启动大规模 GPU 任务。
```

### P1: FP16 TIR/lowering 问题收口

目标:

```text
回答 FP16 为什么没有明显快于 FP32:
是 TVM 没有生成 tensor-core FP16 route, 还是审计 workdir 没查到正确 artifact。
```

执行:

```text
1. 选 5 个代表 label:
   s0_024, lhc_07, lhc_20, s0_056, frontier_25。

2. 对每个 label 审查:
   - ONNX input dtype / initializer dtype。
   - Relax/TE/TIR lowered module 中是否出现 float16。
   - 是否出现 wmma/tensorcore/tensor_core。
   - MetaSchedule workload 是否仍主要是 float32。
   - 是否复用了 FP32 workdir 或未按 FP16 重新 tune。

3. 如果确认没有 tensorcore route:
   - 不直接否定 TVM FP16 能力。
   - 先尝试独立 FP16 workdir + 正确 target/cuda tensorization 配置。
   - 产出 1-3 个 label 的新 FP16 tensorcore smoke latency/energy。

4. 如果确认只是审计路径错误:
   - 更新 TIR audit 脚本, 指向真实 lowered artifact。
   - 重新出 fp16_tir_lowering_*_audit_latest.md。
```

验收:

```text
至少产出一个明确结论:
A. 当前 FP16 route 不是 tensor-core optimized, 需重新 tune/build;
B. 当前 FP16 route 有 tensor-core 证据, 旧审计路径错误;
C. TVM build/tune 阻断, 并有可复现 blocker。
```

### P2: INT8 AP full-val 补点

目标:

```text
把 INT8 AP 从 1/60 推进到至少 5 个 original60 label full-val measured, 再扩展大范围。
```

执行:

```text
1. 从 5 个 smoke label 开始:
   s0_024, s0_040, s0_056, s1_048, s2_160。

2. 对每个 label 执行 gate:
   numeric_sanity -> 5-sample AP smoke -> 1789-frame full-val AP。

3. full-val AP 通过条件:
   - processed_samples=1789。
   - pred_nonempty_count 合理。
   - AP30/AP50/AP70 有合理非零趋势。
   - head/postprocess 分布与 FP16 baseline 可解释。
   - importer 写入 native_int8_original60_ap_rows_v1.jsonl measured row。

4. 失败时:
   - 保存 raw artifact、command、stdout/stderr、GPU id、failure reason。
   - 写 per-label blocker。
   - 继续其他 label, 不因单点失败停止全局。
```

验收:

```text
native_int8_original60_ap_rows_v1.jsonl 中至少新增 5 个 full-val measured AP row,
或每个失败 label 都有明确 blocker 和下一步修复路径。
```

### P3: FP32 latency/AP 严格补齐

目标:

```text
如果下一阶段需要最终三精度同协议大表, FP32 不能只停在 energy 修复。
```

执行:

```text
1. FP32 latency:
   - 优先补齐 2 个 no_claim label。
   - 如果要做严格同协议比较, 重跑 60/60 direct H800 TVM latency, 替换 historical_true_measurement_reclassified。

2. FP32 AP:
   - 明确 5/60 reference AP 的来源和限制。
   - 若需要 60/60 FP32 AP, 使用同一 AP eval protocol 生成 full-val row。
   - 不把 reference AP 或历史 fp16-tagged engine 报告直接写成 precision-specific FP32 measured。
```

验收:

```text
FP32 latency/AP 每个 no_claim 都有 measured row 或 blocker;
总表中不再出现读者无法区分的 remap/reference/true_eval 混合口径。
```

## 7. 关键 artifact 位置

### 总表与审计

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.json
```

### FP32 energy threaded60

```text
rows:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_original60_energy_threaded60_rows_v1.jsonl

raw root:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp32_energy_remeasure/20260629_fp32_threaded_window_original60_60labels

review:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.json
```

### FP16 TIR audit

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.json
```

### FP16 rows

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
```

### INT8 rows

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

## 8. 建议冷启动验证命令

```bash
cd ${V2X_ROOT}
python - <<'PY'
import json, collections
from pathlib import Path
root=Path('multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627')
data=json.loads((root/'exports/original60_quant_three_metric_summary_latest.json').read_text())
rows=data['rows']
print('schema', data.get('schema'))
print('total rows', len(rows))
for precision in ['fp32','fp16','int8']:
    pr=[r for r in rows if r.get('precision')==precision]
    print('\\n'+precision, len(pr))
    print('latency', collections.Counter(r.get('latency_status') for r in pr))
    print('energy ', collections.Counter(r.get('energy_status') for r in pr))
    print('ap     ', collections.Counter(r.get('ap_status') for r in pr))
    print('energy source', collections.Counter(r.get('energy_measurement_source') for r in pr))
    print('energy schedule', collections.Counter(r.get('energy_schedule_policy') for r in pr))
PY
```

期望输出重点:

```text
total rows = 180
fp32 energy measured = 60
fp32 energy source true_measurement = 60
fp32 energy schedule = metaschedule_tuned 56, default 4
fp16 AP measured = 60
int8 AP measured = 1
```

## 9. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP32/FP16/INT8 三指标收口。启动后必须先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/57_6_29_交接文档_阶段三_FP32Energy异常复盘与冷启动防错计划.md, 再阅读 exports/original60_quant_measurement_source_audit_latest.md、exports/original60_quant_three_metric_summary_latest.md、exports/fp32_threaded_window_energy_60label_review_latest.md、exports/fp16_tir_lowering_5label_audit_latest.md、54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md、55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md。固定规则: latency 统一 ms; energy 统一 joule/inference; measured 不等于可横向比较; 每个 measured row 必须追溯 raw_artifact/run_id/source/schedule/sampling; FP32 energy 只使用 threaded60 canonical row, 旧 post_sync_per_iter row 仅为历史审计; dynamic_watt_avg < 50W 的 H800 energy 默认 quarantine; 5-sample AP smoke 禁止写正式 measured AP; FP16/INT8 加速声明必须有 TIR/lowering 证据。当前优先级: P0 冻结并验证 current summary/source audit, 增加或运行 summary validator, 防止旧 row 回退和 schema 误读; P1 收口 FP16 TIR/lowering 问题, 判定当前 route 是否没有 tensorcore/wmma, 如需则独立 FP16 workdir 重新 tune/build 并做 1-3 点 latency/energy smoke; P2 将 INT8 AP 从 1/60 推进到至少 5 个 original60 label 的 1789-frame full-val measured, 失败 label 必须写 blocker; P3 补齐 FP32 latency/AP 严格口径, 至少处理 2 个 latency no_claim 和 55 个 AP no_claim 的 measured/blocker。遇到任何异常不得直接停止或继续扩表, 必须先保存 raw artifact、命令、stdout/stderr、GPU 状态、failure reason, 做 paired smoke/source audit/schema audit 后再继续。
```

