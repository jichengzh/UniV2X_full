# HANDOFF — Stage2 真实 Smoke 与大规模 LUT 生成计划 v1

日期: 2026-06-25

本文继承:

- `multi_agent/methods/progress/HANDOFF_stage2_lut_productization_plan_v1_zh.md`
- `docs/stage2-evidence-registry.zh-CN.md`
- `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v2_zh.md`
- `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`

目标是把下一步从链路 smoke 推进到 **真实 H800 smoke**，并定义后续大规模 latency/AP/energy LUT 补点时的配置分布、scope 口径和必须遵守的实验规则。

---

## 0. 当前结论

当前已经完成的是产品化链路:

```text
job plan -> worker -> generate_latency/AP/energy_lut -> canonical rows -> registry update
```

但上一轮 `evidence_smoke_20260625_125003` 使用的是 stub command，只验证链路，不是实测。

真实 smoke 的启动条件:

1. 目标机器必须是 H800 或明确指定的 H800 backend 环境。
2. latency 与 energy 测试前 GPU 必须完全空闲。
3. 三类真实命令必须输出 generator 要求的 stdout JSON object。
4. 真实 smoke 只跑 `smoke` 级 1-3 个 config，确认 row schema、raw artifact、worker resume、registry update 全部闭环后，才能进入 calibration / paper 级补点。

当前本地环境检查结果显示:

```text
当前可见 GPU: NVIDIA GeForce RTX 4090 x 8
当前不是 H800
GPU 0/1 正在运行 python 负载
GPU 4/5/6/7 有常驻 python 进程占用显存
```

因此，当前机器不能生成 `backend=h800_tvm` 的真实 measured LUT。可以生成真实 smoke job plan，但不能在当前环境把结果标为 H800 measured。

---

## 1. Scope 重锚规则

### 1.1 主 scope

后续 Stage2 LUT 统一以:

```text
optimized_scope = rsu_dense_core
```

为主，即 RSU 侧 dense core subnet 的推理耗时、AP 和 energy 证据。

原因:

- 当前论文和 Stage2 搜索主 claim 是 dense-core 结构/硬件协同优化，不是 full-model speedup。
- RSU 路测设施推理耗时应优先锚定 RSU dense core 的可控计算子图。
- Stage1 manifest 负责标出 dense core / skipped sparse or fusion subgraph / coverage，不负责提供实测 latency。

### 1.2 backbone-only 是否可以

可以测 backbone 部分，但必须降级为更窄 scope:

```text
optimized_scope = backbone_only
```

或在 row 中明确:

```text
notes = "backbone-only; not full rsu_dense_core"
```

不能把 backbone-only 的速度收益写成:

- `rsu_dense_core` 全部收益。
- full-model speedup。
- RSU 端到端 latency 改进。

当前建议:

1. Pyramid 主线继续用 `rsu_dense_core`。
2. CoDriving 若当前可稳定测的只是 `backbone.resnet`，则先作为 `backbone_only` smoke，不能混入正式 `rsu_dense_core` coverage。
3. 当 backbone-only 是 dense core 的主要耗时代理时，可以作为筛选 proxy，但 `measurement_status` 不能写成 dense-core measured，除非 manifest scope 明确该 subnet 等价于当前 dense core measurement scope。

### 1.3 full-model claim

继续禁止:

```text
full_model_claim_allowed = false
```

dense-core evidence 不得外推 full-model speedup。

---

## 2. 必须遵守的规则回顾

### 2.1 Evidence / backend 规则

- Stage1 manifest 只定义结构、coverage 和硬件合法性，不伪装为 latency/AP/energy/DS 实测。
- 新 hardware measured latency evidence 只允许 `backend=h800_tvm`。
- energy measured evidence 只允许 `backend=h800_tvm_power_telemetry`。
- AP measured evidence 使用 `backend=model_eval`，并必须记录 dataset / split / ckpt / finetune protocol。
- TRT 只能标为 `historical_trt` 或独立 backend scope，不能写成 H800 measured。
- demo / proxy / historical / estimated evidence 不升级为论文正式 measured claim。
- failed row 必须写入 JSONL，不能静默跳过。

### 2.2 GPU 空闲规则

latency 与 energy 必须在 GPU 完全空闲时启动:

```text
gpu_util == 0
other_process_memory == 0 或低于显式 guard
无其他 compute process
P-state / clocks / power cap 状态记录到 raw artifact
```

启动前必须记录:

```bash
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

若 GPU 非空闲:

- latency job 不写 measured row。
- energy job 不写 measured row。
- 可以写 failed row，`failure_reason=gpu_not_idle`。

### 2.3 Energy 规则

- energy 与 latency 必须共享 `config_id`。
- 最好和 latency 同 run 采样；若不是同 run，必须记录 `latency_run_id` 和差异说明。
- 必须记录 `telemetry_source`。
- 必须记录 `idle_baseline_policy`。
- 不同 telemetry source 不混成同一 measured population。
- energy 缺失或 failed-only 时 registry 必须继续 no-claim。

### 2.4 AP 规则

- AP row 必须记录 dataset、eval split、ckpt path、ckpt digest 或 unknown、finetune protocol、training budget。
- 不同 finetune protocol 不能混在同一正式比较表。
- FP16 AP 不能直接复用为 INT8 AP，除非 quant policy 明确且有对应 evidence。
- AP predicted / fitted 只能 screening，不能替代 final measured AP。

### 2.5 DS 规则

- DS map 当前只对 CoDriving / Town05 / clean6 / full traffic scope 有效。
- AP x latency -> DS lookup 默认 report-only，除非 AP 和 latency 都是 measured。
- predicted AP + predicted latency -> DS 必须标注 `downstream_validation_required`。
- DS cliff 只支持 600-650 ms 之间的帧级边界，不声明 sub-frame 阈值。
- DS top-K closed-loop validation 是验证 gate，不是 Stage2 主 Pareto 目标。

---

## 3. 真实 Smoke 配置计划

真实 smoke 每个模型先跑最小 1 个 config。

### 3.1 Pyramid smoke

主 smoke config:

```text
model = pyramid_lidar
optimized_scope = rsu_dense_core
candidate_id = smoke_base
software_point_id = smoke_base:w64x128x256:fp16
width = [64, 128, 256]
quant_policy = fp16
schedule_policy = default
```

可选第二个 config:

```text
candidate_id = smoke_p50
software_point_id = smoke_p50:w32x64x128:fp16
width = [32, 64, 128]
quant_policy = fp16
schedule_policy = metaschedule_tuned
```

目的:

- 验证 H800 TVM latency command 可产出 p50/p90/mean/std/raw artifact。
- 验证 AP eval command 可产出 AP70 / dataset / split / ckpt。
- 验证 energy telemetry command 可产出 joule/frame / power / telemetry source。
- 验证 registry update 后三类 measured coverage 均大于 0。

### 3.2 CoDriving smoke

只有当 CoDriving 当前可测 scope 能明确映射为 `rsu_dense_core` 时，才写入 `rsu_dense_core`。

若只测 `backbone.resnet`:

```text
optimized_scope = backbone_only
measurement_status = measured
notes = "CoDriving backbone-only smoke; not rsu_dense_core coverage"
```

建议 config:

```text
model = codriving
candidate_id = smoke_backbone_base
software_point_id = smoke_backbone_base:w64x128x256:fp16
width = [64, 128, 256]
quant_policy = fp16
schedule_policy = default
```

目的:

- 只验证 CoDriving 真实命令链路和 row schema。
- 不把 backbone-only 结果并入 dense-core 正式 coverage。

---

## 4. 大规模 LUT 配置分布

大规模生成分三阶段。

### 4.1 Smoke 阶段

每个模型:

```text
1-3 configs
latency/AP/energy 各 1 row
```

必跑:

- base width。
- 一个预计高价值宽度点。
- 一个可能触发 cliff / schedule 差异的点。

产物:

```text
latency rows: 1-3
AP rows: 1-3
energy rows: 1-3
```

### 4.2 Calibration 阶段

Pyramid 推荐分布:

| 组 | 宽度点 | 目的 |
|---|---:|---|
| anchor | `[64,128,256]` | base anchor |
| single-axis s0 | `[16,128,256]`, `[32,128,256]`, `[48,128,256]` | P0 sensitivity |
| single-axis s1 | `[64,32,256]`, `[64,64,256]`, `[64,96,256]` | P1 sensitivity |
| single-axis s2 | `[64,128,64]`, `[64,128,128]`, `[64,128,192]` | P2 sensitivity |
| coupled combos | `[32,64,128]`, `[48,96,192]`, `[64,96,192]`, `[32,96,192]`, `[48,64,256]`, `[16,128,128]` | coupling / cliff validation |

Schedule distribution:

```text
default: all calibration points
metaschedule_tuned: base + single-axis + coupled combos
```

Quant distribution:

```text
fp16: all points
int8/mixed: only buildable and AP-evaluable subset
```

建议规模:

```text
latency: 15 width points x 2 schedules = 30 rows
AP: 15 width points x selected quant policies = 15-30 rows
energy: 5-8 representative points x 1-2 schedules = 5-16 rows
```

Energy 不需要覆盖所有点，优先:

- base。
- fastest latency candidate。
- Pareto AP-preserving candidate。
- high-risk/high-power tuned candidate。
- one failed/cliff candidate if safe to run。

### 4.3 Paper 阶段

Pyramid:

```text
latency: 30-60 measured rows
AP: 20-40 measured anchors
energy: 8-20 measured representative rows
Q evidence: independent H800 quant evidence rows, not historical TRT
```

CoDriving:

```text
latency: first remeasure currently estimated cells
AP: measured anchors only
energy: base + top serial candidates
DS: top-K closed-loop validation only
```

top-K validation:

- 从 Stage2 Pareto 或 recommendation 中选 K=3-5。
- AP 和 latency 都 measured 后，才允许 DS lookup 进入 constraint / validation。
- DS closed-loop result 与 DS LUT query 分开存证，不混写。

---

## 5. 真实 Smoke 启动命令模板

真实 smoke 不应使用 stub command。应把以下三个 command JSON 替换为真实命令。

```bash
SMOKE_ROOT=results/stage2/pyramid_lidar/evidence_real_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SMOKE_ROOT"/{jobs,latency/raw,ap/raw,energy/raw,logs}

PYTHONPATH=${V2X_ROOT} python scripts/stage2_prepare_evidence.py \
  --model pyramid_lidar \
  --out-json "$SMOKE_ROOT/evidence_registry.json"

PYTHONPATH=${V2X_ROOT} python scripts/stage2_plan_lut_jobs.py \
  --model pyramid_lidar \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --registry "$SMOKE_ROOT/evidence_registry.json" \
  --config-id pyramid_lidar__smoke_base__q_fp16__s_default \
  --candidate-id smoke_base \
  --software-point-id smoke_base:w64x128x256:fp16 \
  --dense-stage neck \
  --width 64,128,256 \
  --quant-policy fp16 \
  --schedule-policy default \
  --latency-measurement-command-json "$REAL_LATENCY_COMMAND_JSON" \
  --ap-eval-command-json "$REAL_AP_COMMAND_JSON" \
  --energy-telemetry-command-json "$REAL_ENERGY_COMMAND_JSON" \
  --out-jsonl "$SMOKE_ROOT/jobs/lut_job_plan_v1.jsonl"

PYTHONPATH=${V2X_ROOT} python scripts/stage2_lut_worker.py \
  --job-plan "$SMOKE_ROOT/jobs/lut_job_plan_v1.jsonl" \
  --job-state "$SMOKE_ROOT/jobs/lut_job_state_v1.jsonl" \
  --max-jobs 3 \
  --resume \
  --log-dir "$SMOKE_ROOT/logs/worker_jobs"

PYTHONPATH=${V2X_ROOT} python scripts/stage2_update_registry_from_luts.py \
  --registry "$SMOKE_ROOT/evidence_registry.json" \
  --latency-rows "$SMOKE_ROOT/latency/latency_lut_rows_v1.jsonl" \
  --ap-rows "$SMOKE_ROOT/ap/ap_anchor_rows_v1.jsonl" \
  --energy-rows "$SMOKE_ROOT/energy/energy_lut_rows_v1.jsonl" \
  --out-json "$SMOKE_ROOT/evidence_registry.updated.json"
```

真实命令 stdout 必须输出 generator contract:

```text
latency: latency_p50_us required
AP: metric_value required
energy: joule_per_inference + telemetry_source + raw_artifact required
```

---

## 6. 当前阻塞项

当前不能在本地启动真实 H800 smoke，原因:

1. 本地可见 GPU 是 RTX 4090，不是 H800。
2. GPU 0/1 正在运行 Python 计算负载。
3. GPU 4/5/6/7 有常驻 Python 进程占用显存。
4. 即便 GPU 2/3 空闲，生成的结果也只能是 4090 backend，不可写为 `h800_tvm measured`。

可立即执行的下一步:

- 在 H800 机器上运行第 5 节命令模板。
- 或提供 H800 上可调用的 latency/AP/energy 真实 command JSON，我再生成正式 `evidence_real_smoke_*` job plan。

本轮 preflight artifact:

```text
results/stage2/pyramid_lidar/real_smoke_preflight_20260625_152133/
  PRECHECK_SUMMARY.json
  nvidia_smi_query.csv
  nvidia_smi_pmon.txt
```

`PRECHECK_SUMMARY.json` 当前结论:

```text
status = blocked
can_run_h800_measured_smoke = false
reasons =
  - visible_gpus_are_not_h800
  - one_or_more_gpus_not_idle
  - h800_tvm_measured_rows_must_not_be_generated_on_rtx4090
```

---

## 7. 大规模启动判定

只有满足以下条件才开始大规模补点:

1. H800 真实 smoke 三类 job 都 succeeded。
2. 三张 row 表均有 measured rows。
3. worker resume 二次运行 `jobs_run=0`。
4. registry update 后 latency/AP measured coverage > 0。
5. energy claim gate 只在 measured telemetry row 存在时为 true。
6. latency/energy raw artifact 中包含 GPU idle preflight 记录。
7. AP raw artifact 中包含 dataset / split / ckpt / eval command。
8. CoDriving backbone-only 与 `rsu_dense_core` 不混表。

若任一条件不满足，只允许继续 smoke / calibration，不进入 paper 级大规模补点。
