# 交接文档 23：Gold Cold-start 96 v3 实施计划

**日期**：2026-07-11  
**状态**：阶段 2 已收口；阶段 3 进入 planner/manifest 实施  
**目标**：生成可进入 backend-conditioned cost model 的 96 行新 gold cold-start，不混入历史诊断数据或人工后端偏好。

## 1. 固定目标与完整组

阶段 3 的固定规模为：

```text
2 models x 12 widths x 2 q_mode x 2 compiler profiles = 96 rows
```

- `models`：Pyramid、CoDriving；
- 每个 model 选择 12 个 width 点；
- `q_mode`：FP16、INT8；
- `compiler profile`：当前正式 TVM profile、TensorRT profile；
- 一个 `(model, width)` 必须形成四行完整组：

```text
TVM FP16
TVM INT8
TensorRT FP16
TensorRT INT8
```

缺任一行时，该组只能标记为 incomplete/feasibility evidence，不得进入 backend gold。不得用历史行补齐四行组。

## 2. 3A：合同冻结

先冻结 schema、主键和验收门禁，再生成 manifest。

1. 主键至少包含 `model_id, width_id, q_mode, compiler_profile_id, hardware_id, run_id`。
2. 每行记录 build 状态、真实 latency/energy、数值检查、AP 来源、校准来源、realization、失败原因和 evidence trust。
3. backend/profile 是测量上下文，不进入逻辑搜索 genome；禁止写入“TVM 选 FP16、TRT 选 INT8”的人工规则。
4. gold 行必须来自同一冻结代码、模型权重、数据切分、硬件与正式测量协议。
5. manifest 校验必须证明总计 24 个完整组、96 个唯一行，且每组恰有四行。

## 3. 3B：12 个 width 选点

每个 model 独立选择 12 个 width，覆盖低、中、高复杂度和已有可行边界；选点依据只允许结构覆盖、算力覆盖及能力扫描结果，不按预期后端胜负挑点。

执行要求：

- width 必须能稳定重建并映射到唯一 `width_id`；
- 两个 profile 使用完全相同的 model/width/q_mode 语义；
- 先输出候选、去重和覆盖审计，再冻结最终 12 点；
- planner 生成 24 个组，manifest 展开为 96 行，并执行唯一性与四行完整性检查。

## 4. 3C：两组 8 行 pilot

pilot 只取两组：Pyramid 1 个 width、CoDriving 1 个 width，共 `2 groups x 4 rows = 8 rows`。

pilot 必须验证：

1. 两个 profile 的 FP16/INT8 均走正式 build/run 路径；
2. 行级 schema、日志、产物路径、重跑策略和聚合脚本可用；
3. latency/energy 协议一致，独立进程复测与离散度字段完整；
4. INT8 使用正式校准和 numerical gate；
5. 四行组可被自动判定为 complete、incomplete 或 feasibility-only；
6. AP 共享规则可执行且审计记录可追溯。

pilot 八行全部通过合同后才能进入 3D；pilot 数据通过正式门禁时直接计入 96 行，不重复测量。

## 5. AP 共享硬规则

AP 只允许在 **相同 `model/width/q_mode`、不同 backend/profile** 之间共享，并且必须先证明两端输出数值等价。

- 数值等价检查使用冻结输入集、相同预处理/后处理和预先冻结的误差阈值；
- 通过后，AP 只计算一次，另一行记录 `ap_shared_from`、等价性报告和阈值；
- 未验证、超阈值、build 失败或输出不可比时禁止共享 AP；
- 共享失败不伪造 AP，也不删除该测量，必须记录为 `feasibility`，附失败阶段和原因；
- FP16 与 INT8 之间不得共享 AP。

## 6. 3D：六批正式执行

pilot 覆盖 2 个组后，剩余 22 个组按六批执行：

| 批次 | 完整组数 | 行数 | 累计行数（含 pilot） |
|---|---:|---:|---:|
| Pilot | 2 | 8 | 8 |
| Batch 1 | 4 | 16 | 24 |
| Batch 2 | 4 | 16 | 40 |
| Batch 3 | 4 | 16 | 56 |
| Batch 4 | 4 | 16 | 72 |
| Batch 5 | 3 | 12 | 84 |
| Batch 6 | 3 | 12 | 96 |

每批执行顺序为：manifest 校验 -> build -> numerical/calibration gate -> 正式测量 -> AP 计算或合规共享 -> 四行组聚合 -> 批次审计。每批验收后再放行下一批；失败行按固定重试预算处理，不得以其他 width 或旧数据静默替换。

## 7. 3E：必须交付的产物

1. 冻结的 planner 配置、96 行 manifest 及其校验报告；
2. 行级原始 build/run、数值、校准、latency、energy 和 AP 证据；
3. 24 个四行组的 complete/incomplete/feasibility 状态表；
4. AP 等价性报告与 `ap_shared_from` 审计表；
5. 仅由合格新数据生成的 backend gold 表及数据字典；
6. 失败清单、重试记录、环境指纹和可复现命令；
7. 96 行规模、唯一性、覆盖度和无历史污染的最终验收报告。

旧 180 行仅用于 width 候选覆盖分析和后续消融，不进入 backend gold，不用于补行，也不与新 96 行混合训练主结果。

## 8. 3F：停止条件

出现以下任一情况时停止扩批，保留已完成证据并先修复合同或执行链：

1. pilot 任一行无法按正式路径产出，或两组无法形成可审计的四行组；
2. manifest 不是 24 组/96 唯一行，存在重复、漏组合或语义不一致；
3. 两个 profile 对同一 model/width/q_mode 的实现语义不可比；
4. AP 在未证明跨后端数值等价时被共享，或共享链不可追溯；
5. 正式校准、数值门禁、环境指纹或测量协议发生未冻结变更；
6. 同类 build/run 失败连续出现并超过既定重试预算，说明是系统性 feasibility 问题；
7. 需要用旧 180、手写后端偏好或人工挑选胜者才能凑齐 gold。

停止不等于删除失败点。所有失败均进入 feasibility 记录；只有满足合同和门禁的完整四行组进入 backend gold。

## 9. 当前执行状态

- 阶段 2：已收口。
- 阶段 3A planner/manifest：已完成并通过 7 项测试。
- 阶段 3C pilot：尚未发射，正在审计 8 行的 runner、checkpoint、calibration、AP 与 provenance 入口。

### 9.1 已冻结宽度与 split

Pyramid 和 CoDriving 使用同一个 12 点逻辑宽度网格，用于跨模型比较同一 shape 动作的效果：

```text
diagonal/base:
32x32x128, 48x96x192, 64x96x192, 64x128x256

alignment/group traps:
16x32x64, 32x64x128, 40x64x128, 48x64x128

off-diagonal/AP-sensitive:
24x32x96, 24x64x128, 56x112x224, 64x64x128
```

locked holdout 只根据预先冻结的结构分层选择，未读取本轮性能标签：

```text
pyramid|24x32x96
pyramid|64x128x256
codriving|24x32x96
codriving|64x128x256
```

pilot 组为：

```text
pyramid|32x64x128
codriving|32x64x128
```

### 9.2 已生成产物

- planner：`${V2X_ROOT}/scripts/stage2_gold_coldstart96_plan_v3.py`
- manifest：`${V2X_ROOT}/results/gold_coldstart96_v3_20260711/gold_coldstart96_manifest_v3-c2e0645a8f975340f53f8a29f8fed03447a46f6eb212d301351c2e71f1d2a880.json`
- split：`${V2X_ROOT}/results/gold_coldstart96_v3_20260711/gold_coldstart96_split_v3-ebb532ba05283bdd09d26d521d40c398f339485015d8fc92ae6db29c5e3d5b52.json`
- contract：`${V2X_ROOT}/results/gold_coldstart96_v3_20260711/gold_coldstart96_contract_v3-e54dbca5f15a5ceb94de3cac1b055cf7945ab899ab3ea8c5e33d952400a3086c.json`

当前审计结果：96 行、24 个完整 `2 q_mode x 2 profile` 组，80 train + 16 locked holdout，`mixed_policy_id=none`，行级 genome 与 `strategy_id` 不含 backend/compiler token。AP 跨 profile 共享必须绑定 checkpoint、ONNX/编译产物、calibration、dataset/eval protocol、输入、容差与输出比较报告的哈希。

### 9.3 Pilot 入口审计结果

8 行 pilot 已全部映射，但本轮**未发射真测**，原因是正式 artifact/provenance 门禁未满足：

1. CoDriving 已有可改造的 ONNX、TVM/TRT perf、AP 和 calibration runner，但旧结果引用的 `/exdata/...` checkpoint、ONNX、engine 和 calibration 产物不完整，不能直接登记为 v3 gold。
2. Pyramid 只有历史 TVM/TRT latency/energy 线索，缺少 Stage3 专用导出、统一 AP 和 INT8 calibration 入口；特别是 TVM-auto INT8 不得用 hand-rewrite、mixed 或 BYOC-TRT 结果代替。
3. planner 当前引用的两个 capability profile 仍是 S1 前的 bootstrap digest；在 pilot 前需冻结包含 S1-Q 的最终 profile，S1-P 的 group/per-group 信息作为候选图/shape 上下文，不压成人工 backend 规则。

因此下一个执行节点调整为：

```text
冻结 probe-conditioned capability profile
-> 建立 Pyramid/CoDriving 统一 source checkpoint + ONNX artifact staging
-> 产出 8 行 pilot dispatch request
-> H800 build/numerical/latency/energy/AP
```

只有前两项产物哈希齐全后才发射 pilot。这是 3C 合同门禁的正常触发，不代表删除 TVM INT8 行；若其正式自动路径无法 build，该行以 feasibility failure 收口。

## 10. 2026-07-12 六批性能闭环结果（覆盖并修正第 9 节旧状态）

第 9 节记录的是发射前状态，以下结果为当前有效状态。最终 manifest 使用模型各自具备 checkpoint/ONNX provenance 的 12 个宽度，不再使用第 9.1 节所写的共享宽度假设；pilot 为 `pyramid|32x32x128` 和 `codriving|32x64x128`。

### 10.1 执行结果

| 范围 | 计划行 | 测量成功 | 两次重试后 feasibility failure | 终态覆盖 |
|---|---:|---:|---:|---:|
| Pilot 性能 | 8 | 8 | 0 | 8/8 |
| Batch 1 | 16 | 16 | 0 | 16/16 |
| Batch 2 | 16 | 16 | 0 | 16/16 |
| Batch 3 | 16 | 16 | 0 | 16/16 |
| Batch 4 | 16 | 15 | 1 | 16/16 |
| Batch 5 | 12 | 11 | 1 | 12/12 |
| Batch 6 | 12 | 12 | 0 | 12/12 |
| 合计 | 96 | 94 | 2 | 96/96 |

六个正式批次覆盖 22/22 个非 pilot 四行组；加入 pilot 后覆盖 24/24 组。86 个非 pilot 成功行全部通过 finite latency、finite energy、build/status、correctness 和结果 SHA256 校验。状态日志位于 `results/stage3_gold96_v3_20260711/performance_execution/batch_{01..06}_state.jsonl`。

两个可信失败均来自 Pyramid 的 TVM-auto FP16：

- `pyramid|16x64x128|tvm_fp16`；
- `pyramid|32x32x256|tvm_fp16`。

两行均独立执行两次；build 完成后 measure 稳定触发 CUDA illegal memory access，MetaSchedule 日志同时记录 `Ramp lanes=8 > CUDA codegen limit 4`。因此它们是 shape-conditioned backend feasibility evidence，不是缺失值，也不使用旧数据替换。

### 10.2 执行器修复与环境审计

批量发射前后修复了 runner 相对路径、TVM CUDA runtime `LD_LIBRARY_PATH`、TVM `<out-dir>/<label>` 结果定位、真实 energy 字段，以及 GPU 动态重映射和断点恢复。执行器测试 14/14 通过。早期由 harness 字段不匹配产生的 attempt 与 GPU5/6 外部负载污染批次均单独归档为 `*_harness_invalid_*` / `*_environment_contaminated_*`，未进入上述统计。

正式统计最终只在空闲 GPU7 上顺序重跑。逐行启动前执行空闲门；环境暂忙只中止当前调度并从 state JSONL 恢复，不计为模型失败。

### 10.3 收口边界

本节只收口本轮要求的 **96/96 性能与 feasibility 终态生成**，不能据此宣称 96 行 backend Gold 已全部完成。manifest 的 `required_metrics` 仍包含 AP：

- Pyramid pilot 的 TRT-FP16、TRT-INT8、TVM-FP16 已完成 1789 样本 AP；AP70 分别为 `0.548483`、`0.535017`、`0.548560`；
- Pyramid pilot 的 TVM-INT8 数值 sanity 为 blocked，输出跨样本近似恒定且与参考相关系数接近 0，应按数值 feasibility 问题处理，不能伪造或共享 AP；
- 其余非 pilot 行和 CoDriving pilot 的 AP 尚未进入本节性能批次状态。

因此当前可以进入下一节点“AP/数值合同回填与最终 Gold 表生成”，但任何训练表在该节点完成前必须把本节状态标为 `performance_terminal`，不得标为 `measured_success_gold`。
