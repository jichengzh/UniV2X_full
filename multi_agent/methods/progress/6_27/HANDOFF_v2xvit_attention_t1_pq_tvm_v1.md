# HANDOFF — V2X-ViT Attention T1-P/T1-Q TVM 口径收尾 (v1, 2026-06-23)

> **最新清上下文入口是 [`HANDOFF_v2xvit_attention_t1_pq_tvm_v2.md`](HANDOFF_v2xvit_attention_t1_pq_tvm_v2.md)**。v1 保留为长版历史细节。
>
> **接手先读本页**。本页 = 当前 V2X-ViT attention fusion 接入框架前的 T1-P/T1-Q 状态、已补产物、真实 TVM 实验结果、阻塞点和下一步执行命令。
> 背景设计文档: [`plan_transformer_into_framework_v1.md`](../design/plan_transformer_into_framework_v1.md)。当前结论很明确: **T1 仍未完成, 不要启动 T2**。

---

## §0 一句话状态 + 接手第一步
**状态**: T1-S 已完成; T1-P 已补 HMSA/MSwin 手写结构化 scanner; T1-Q 已从旧 TensorRT 口径改为 **TVM-only**。H800 TVM direct INT8 子算子可行; MSwin full-attention TVM FP16/mixed INT8 已补但 mixed INT8 是负/中性证据; **HMSA minimal explicit 2-agent relation core** 已在 TVM FP16/mixed INT8 跑通, 但 mixed INT8 同样慢于 FP16。最新代码已补 HMSA fixed-type `[0,1]` static TVM target, 覆盖 q/k/v projection + relation core + output projection, 但这个 static target **尚未在 H800 重测**, 也仍不覆盖 dynamic `types` dispatch/e2e quantized AP。attention subnet TVM 表已补但仍基于旧 minimal-core v2 结果: p50 FP16 subnet `1.7783×`, p50 mixed INT8 subnet `1.7069×`, mixed INT8 相对同剪枝 FP16 是 `0.9598×`。full DAIR val 的全模型 FP16 表已补: p50 shortft `1.0784×`, ΔAP50=`-0.0128`, ΔAP70=`-0.0084`, 通过 AP guardrail。最终验收表已落 `results/attention_final_acceptance_v1.*`, 但 **完整 TVM mixed-INT8 e2e 行仍缺失**。因此当前 gate 仍是:

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

**★接手第一步**: 不要进 T2。下一步只剩最终 TVM quantization/e2e integration:
1. 先在 H800 重跑最新 `t1_attention_tvm_bench.py --full-attention --full-only`, 生成 static-HMSA v3 subnet 结果; 旧 `attention_subnet_accel_v1` 只能作为 minimal-core 历史证据。
2. 以 `attention-p50-shortft_steps100` 为当前剪枝+微调候选, 构建 `attention-p50-int8/mixed` 的完整 TVM e2e 或至少 fusion-subgraph runner。
3. 对 attention subnet 做 fused/calibrated TVM mixed INT8, 目标是超过 p50 FP16 subnet; 旧 v2 synthetic mixed INT8 是 `0.462336 ms`, 慢于同剪枝 FP16 `0.443765 ms`, 不合格。
4. 再接入全模型 e2e 表: baseline / p50-shortft-fp16 / p50-shortft-TVM-mixedINT8, full DAIR val AP 与 latency 同协议。
5. 只有完整 TVM mixed-INT8 e2e 行通过 speed/AP guardrail 后, 才能重新讨论 T2。

---

## §1 本轮用户反馈与修正原则
用户指出三点, 已按此修正:
- **Q 轴不应继续使用 TensorRT**: 项目已经全面迁移 TVM, T1-Q gate 必须 TVM-only。
- **P 轴必须有 HMSA/MSwin 手写结构化 scanner**: 不能只给 DepGraph/candidate。
- **先完善 T1, 不提前 T2**: 需要给 attention fusion prune+INT8 的速度/精度表, 但必须标明真测与 prior 的边界。

当前脚本已把非 TVM 引擎排除在 Q gate 外; QDQ-ONNX fake quant 也不能算真实 INT8。

---

## §2 已完成产物

### 2.1 T1-P — HMSA/MSwin manual scanner 已补
实现位置:
- [`scripts/phase2/t1_attention_pq_feasibility.py`](../../../scripts/phase2/t1_attention_pq_feasibility.py)
  - `scan_manual_attention_pruning_groups`
  - `gate_p_axis`

scanner 当前枚举:

| family | count | 含义 |
|---|---:|---|
| `hmsa_head` | 3 | 3 个 encoder layer, 每层 HMSA 8 heads × dim_head 32 |
| `mswin_head` | 9 | 3 层 × window_size {4,8,16}; heads 分别 {16,8,4} |
| `encoder_depth` | 3 | depth pruning 候选 |
| `embed_dim` | 1 | dim=256, round_to=32 的全局通道候选 |

P 轴 gate:

```text
P_axis_READY_WITH_MANUAL_SCANNER_DEPGRAPH_PARTIAL
```

解释: scanner 已有; isolated FeedForward DepGraph 可过; `mswin_bwa_depgraph` 仍失败, 因此 T2 时不能依赖通用 DepGraph 自动切 MSwin, 必须用 scanner 的手写 slice 规则。

### 2.2 T1-Q — TVM direct INT8 子算子实测已补
新增 TVM bench:
- [`scripts/phase2/t1_attention_tvm_bench.py`](../../../scripts/phase2/t1_attention_tvm_bench.py)
  - direct Relax matmul INT8。
  - MSwin BaseWindowAttention full-window FP16/mixed INT8。
  - HMSA v2 minimal relation core FP16/mixed INT8。
  - **新增但未重测**: HMSA fixed-type `[0,1]` static target, `make_hmsa_static_2agent_param_plan`, `build_relax_hmsa_static_2agent_{fp16,mixed_int8}`, 覆盖 q/k/v projection、relation core、per-type output projection。

H800 实测原始结果:
- [`results/attention_axis_tvm_bench_v1.json`](../../../results/attention_axis_tvm_bench_v1.json)

汇总报告:
- [`results/attention_axis_feasibility_v1.json`](../../../results/attention_axis_feasibility_v1.json)
- [`results/attention_axis_feasibility_v1.md`](../../../results/attention_axis_feasibility_v1.md)

Q 轴 gate:

```text
Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED
```

解释: TVM direct matmul 子算子 INT8 能跑并加速; MSwin full window 与 HMSA minimal relation core 的 mixed INT8 能跑但慢于 FP16。最新 HMSA static qkv/relation/out target 只完成了代码和单元测试, 还没有 H800 latency。HMSA dynamic dispatch/e2e quant 仍未集成, 不能说 T1-Q 完成。

### 2.3 Stop-B — attention-p50 真实剪枝 pilot + blocker 已落档
新增执行/验收产物:
- [`scripts/phase2/t1_attention_e2e_pq.py`](../../../scripts/phase2/t1_attention_e2e_pq.py): 真实 HMSA/MSwin p50 head surgery、pilot latency/AP、Stop-B blocker 生成。
- [`tests/phase2/test_t1_attention_e2e_pq.py`](../../../tests/phase2/test_t1_attention_e2e_pq.py): HMSA/MSwin surgery、HMSA blocker、Stop-B markdown 口径测试。
- [`results/attention_e2e_pq_blocker_v1.md`](../../../results/attention_e2e_pq_blocker_v1.md): reviewer 接受的 Stop-B blocker。
- [`results/attention_e2e_pq_partial_v1.json`](../../../results/attention_e2e_pq_partial_v1.json), [`results/attention_e2e_pq_partial_v1.csv`](../../../results/attention_e2e_pq_partial_v1.csv): baseline、attention-p50 no-ft、attention-p50 100-step shortft pilot。
- [`results/attention_p50_shortft_v1.json`](../../../results/attention_p50_shortft_v1.json): attention-p50 100-step short finetune pilot。
- [`models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`](../../../models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth): shortft checkpoint。
- [`models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`](../../../models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json): shortft protocol + prune manifest。
- [`results/attention_e2e_pq_review_v1.md`](../../../results/attention_e2e_pq_review_v1.md): reviewer verdict `ACCEPT`, 但只接受 Stop-B 记录质量, 不接受最终收益。
- [`models/v2xvit_attention_t1/attention_p50_manifest_v1.json`](../../../models/v2xvit_attention_t1/attention_p50_manifest_v1.json): 3 个 HMSA + 9 个 MSwin head group 的真实 p50 manifest。
- `models/v2xvit_attention_t1/attention_p50_noft_epoch17.pth`: no-finetune attention-p50 checkpoint。
- [`results/attention_full_tvm_bench_v1.json`](../../../results/attention_full_tvm_bench_v1.json): H800 TVM full MSwin report + 旧 HMSA `NOT_IMPLEMENTED` blocker。
- [`results/attention_full_tvm_bench_v2.json`](../../../results/attention_full_tvm_bench_v2.json): H800 TVM full MSwin + HMSA minimal explicit relation-core FP16/mixed INT8 report。
- [`logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log`](../../../logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log): v2 远端运行日志。
- [`scripts/phase2/attention_final_acceptance_report.py`](../../../scripts/phase2/attention_final_acceptance_report.py): Stop-A 最终验收表生成器; 当前生成 blocked 表, 等待真实 `attention-p50-int8/mixed` TVM e2e row。
- [`results/attention_final_acceptance_v1.json`](../../../results/attention_final_acceptance_v1.json), [`results/attention_final_acceptance_v1.csv`](../../../results/attention_final_acceptance_v1.csv), [`results/attention_final_acceptance_v1.md`](../../../results/attention_final_acceptance_v1.md): 当前最终验收候选表, `gate_status=BLOCKED_TVM_MIXED_INT8_E2E_MISSING`。

Stop-B reviewer 结论:

```text
Verdict: ACCEPT
Scope: accepted only as actionable blocker + negative/pilot partial evidence.
Unsupported: final attention prune+quant e2e speedup, final DAIR val AP, p50-int8/mixed benefit, dynamic HMSA TVM path/e2e quant integration.
```

---

## §3 H800 TVM 实测数值

环境: H800 GPU6, TVM `0.20.dev1070+gb628d91fa`, input `(B=1, L=2, H=64, W=128, C=256)`, direct Relax matmul, `int8 x int8 -> int32`。

### 3.1 base direct 子算子

| target | FP16 p50 ms | INT8 p50 ms | speedup |
|---|---:|---:|---:|
| `linear_256x256_direct` | 0.0336 | 0.0229 | 1.4672× |
| `linear_qkv_direct` | 0.0960 | 0.0631 | 1.5214× |
| `mswin_out_direct` | 0.0336 | 0.0229 | 1.4672× |
| `ffn_direct_pair_estimate` | 0.0672 | 0.0458 | 1.4672× |

### 3.2 p50 attention-head prune + INT8 direct 子算子
这里的 speedup 是 **base FP16 子算子 → p50 pruned INT8 子算子**。

| target | prune_rate | FP16 baseline ms | p50 INT8 ms | speedup |
|---|---:|---:|---:|---:|
| `hmsa_proj_head_p50_direct` | 50% | 0.0336 | 0.0128 | 2.6250× |
| `hmsa_out_head_p50_direct` | 50% | 0.0336 | 0.0158 | 2.1266× |
| `mswin_qkv_head_p50_direct` | 50% | 0.0960 | 0.0352 | 2.7273× |
| `mswin_out_head_p50_direct` | 50% | 0.0336 | 0.0158 | 2.1266× |

### 3.3 精度/AP 状态
当前 coupling table 中 AP 数值来自:
- [`results/coupling_map/C4_QgranxP_v2xvit.json`](../../../results/coupling_map/C4_QgranxP_v2xvit.json)

这些 AP 是 **full-model fake-quant prior**, 不是 attention-head pruning + TVM INT8 的真实 AP:

| config | prune_rate | quant | AP50 | AP70 | status |
|---|---:|---|---:|---:|---|
| base prior | 0% | int8 | 0.6988 | 0.4942 | `SIMULATED_PRIOR_NOT_TRUE_TVM` |
| p50 prior | 50% | int8 | 0.7062 | 0.5126 | `SIMULATED_GLOBAL_PRIOR_NOT_ATTENTION_SPECIFIC_NOT_TRUE_TVM` |

**写论文/报告时禁止表述为真实 TVM INT8 AP**。真实 AP 需要下一步模型 surgery/eval。

### 3.4 Stop-B pilot: attention-p50 no-ft/shortft model-forward/AP
以下是 **64-sample pilot**, 不是最终 DAIR val AP; latency scope 是 `pytorch_model_forward_e2e`, 计 `model(batch["ego"])`, 不含 dataloader/postprocess/NMS/eval loop。

| config | prune | quant_backend | finetune | latency scope | p50 ms | speedup | AP50 | AP70 | status |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| baseline | 0% | none | official ckpt | `pytorch_model_forward_e2e` | 34.2072 | 1.0000× | 0.58045 | 0.44731 | `REAL_EVAL_LOG_64_SAMPLE_PILOT` |
| attention-p50-fp16 no-ft | 50% | none | none | `pytorch_model_forward_e2e` | 32.8882 | 1.0401× | 0.54216 | 0.37612 | `REAL_EVAL_LOG_64_SAMPLE_PILOT` |
| attention-p50-shortft-fp16 | 50% | none | 100 steps, lr=1e-4, all params, batch=1 | `pytorch_model_forward_e2e` | 35.7510 | 0.9568× | 0.62761 | 0.47746 | `REAL_EVAL_LOG_64_SAMPLE_PILOT` |

pilot 结论:
- no-ft speedup = `1.0401×`, 小于 Stop-C 风险阈值 `1.05×`。
- ΔAP50 = `-0.0383`, ΔAP70 = `-0.0712`, no-ft 已显著超出 AP guardrail。
- 100-step shortft pilot 的 AP 在 64-sample 上恢复: ΔAP50=`+0.0472`, ΔAP70=`+0.0302`; 但 latency 变慢: speedup=`0.9568×`。
- 这不是最终 ACCEPT: AP 提升必须做 epoch/lr/seed/sample 协议审查和 larger/full DAIR val 复核; latency 未达到系统收益阈值。

### 3.5 H800 TVM full/MSwin + HMSA minimal-core evidence
`results/attention_full_tvm_bench_v2.json` 中:

| target | FP16 p50 ms | mixed INT8 p50 ms | mixed/FP16 speedup | status |
|---|---:|---:|---:|---|
| `mswin_bwa_full_attention` | 0.456245 | 0.465418 | 0.9803× | negative/neutral |
| `mswin_bwa_p50_full_attention` | 0.244672 | 0.257077 | 0.9517× | negative/neutral |
| `hmsa_full_attention` minimal relation core | 0.332906 | 0.348373 | 0.9556× | negative/neutral; minimal core only |
| `hmsa_p50_full_attention` minimal relation core | 0.199093 | 0.205259 | 0.9700× | negative/neutral; minimal core only |

这说明 MSwin full window attention 与 HMSA minimal relation core 都有 TVM runtime 证据, 但 mixed INT8 当前均比 FP16 慢, 必须写成负/中性证据, 不能写成收益。HMSA v2 只覆盖 explicit 2-agent relation core, 不覆盖 dynamic `types` dispatch、q/k/v input projection ModuleList lowering 和校准后的 e2e AP。

最新代码已把 HMSA TVM target 推进到 static fixed-type `[0,1]` q/k/v projection + relation core + output projection:
- `make_hmsa_static_2agent_param_plan`
- `build_relax_hmsa_static_2agent_fp16`
- `build_relax_hmsa_static_2agent_mixed_int8`
- `bench_hmsa_static_2agent_attention`

但这部分 **尚未在 H800 生成 latency JSON**。因此所有 `attention_subnet_accel_v1` 数字仍按旧 minimal relation-core 解释; 下一步必须重跑 v3 static subnet 后再比较 p50 mixed INT8 是否能超过 p50 FP16。

### 3.6 Attention subnet TVM acceleration
`results/attention_subnet_accel_v1.json` 中把 MSwin full-window + HMSA minimal relation-core 作为 attention subnet 汇总:

| config | prune | quant | subnet p50 ms | speedup vs base fp16 | speedup vs same-prune fp16 |
|---|---:|---|---:|---:|---:|
| `attention-subnet-base-fp16` | 0% | fp16 | 0.789151 | 1.0000× | 1.0000× |
| `attention-subnet-base-mixed-int8` | 0% | mixed_int8 | 0.813791 | 0.9697× | 0.9697× |
| `attention-subnet-p50-fp16` | 50% | fp16 | 0.443765 | 1.7783× | 1.0000× |
| `attention-subnet-p50-mixed-int8` | 50% | mixed_int8 | 0.462336 | 1.7069× | 0.9598× |

结论:
- 剪枝对子网有效: p50 FP16 subnet `1.7783×`。
- 当前 mixed INT8 不合格: p50 mixed INT8 虽相对 base FP16 是 `1.7069×`, 但相对同剪枝 FP16 只有 `0.9598×`, 说明 synthetic mixed INT8 抵消了部分剪枝收益。
- 下一步 TVM Q path 不能继续靠 direct matmul。先用新增 static-HMSA target 重跑 subnet; 如果 mixed INT8 仍慢于同剪枝 FP16, 必须做 fused/calibrated attention schedule, 目标至少超过 p50 FP16 subnet。

### 3.7 Full DAIR val e2e FP16 checkpoint evidence
`results/attention_e2e_checkpoint_eval_full_v1.json` 中:

| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | official ckpt | 38.6740 | 1.0000× | 0.71013 | 0.52161 | 0 | 0 |
| attention-p50-fp16 no-ft | 50% | none | 38.9024 | 0.9941× | 0.66884 | 0.46274 | -0.0413 | -0.0589 |
| attention-p50-shortft-fp16 | 50% | 100 steps, lr=1e-4 | 35.8636 | 1.0784× | 0.69729 | 0.51322 | -0.0128 | -0.0084 |

结论:
- no-ft p50 在 full val 下失败: 无速度收益且 AP70 下降 `-0.0589`。
- 100-step shortft p50 在 full val 下满足当前 FP16 剪枝 guardrail: speedup `1.0784×`, AP50/AP70 下降均小于 `0.02`。
- 这仍不是 Stop-A, 因为缺 `attention-p50-shortft-TVM-mixedINT8` 的完整 e2e 行。

### 3.8 Stop-A candidate/final acceptance table
`results/attention_final_acceptance_v1.json` 已把当前 full-val 证据转换成最终验收 schema:

| config | prune | quant | latency p50 ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0% | fp16 | 38.6740 | 1.0000× | 0.7101 | 0.5216 | 0 | 0 | real full-val |
| attention-p50-fp16 | 50% | fp16 | 35.8636 | 1.0784× | 0.6973 | 0.5132 | -0.0128 | -0.0084 | real full-val |
| attention-p50-int8/mixed | 50% | TVM mixed INT8 | missing | missing | missing | missing | missing | missing | `MISSING_TVM_E2E` |

`attention_e2e_pq_validator.py --mode stop-a --report-json results/attention_final_acceptance_v1.json` 当前应返回 `REJECT`, 且拒绝点只应集中在 `attention-p50-int8/mixed` 行缺少 e2e latency/AP/log/command。这是预期行为, 防止把 subnet 或逐算子结果冒充最终验收。

---

## §4 当前阻塞点

### 4.1 full MSwin BaseWindowAttention TVM 已有, 但 mixed INT8 是负/中性证据
当前报告中:

```text
mswin_bwa_full_attention: FP16 OK, mixed INT8 OK but slower than FP16
mswin_bwa_p50_full_attention: FP16 OK, mixed INT8 OK but slower than FP16
```

解释:
- MSwin full window Q/K/V projection、QK、relative-position add、softmax、AV、output projection 的 TVM runtime 证据已补。
- 当前 mixed INT8 使用合成 int8 cast/scale, 不是 AP-calibrated quant。
- v2 base: mixed INT8 `0.465418ms` vs FP16 `0.456245ms`; p50: mixed INT8 `0.257077ms` vs FP16 `0.244672ms`。
- 因此 MSwin mixed INT8 当前不能作为 latency benefit; 后续如继续 Q path, 必须检查 scale/requant 与 schedule, 不得只报告 direct matmul speedup。

### 4.2 HMSA TVM static target 已有代码, 但 H800 重测/dynamic/e2e quant 仍未完成
当前 v2 报告中:

```text
hmsa_full_attention: FP16 OK, mixed INT8 OK but slower than FP16
hmsa_p50_full_attention: FP16 OK, mixed INT8 OK but slower than FP16
remaining stage: DYNAMIC_DISPATCH_AND_E2E_QUANT_NOT_INTEGRATED
```

已完成:
- minimal explicit 2-agent relation core: `q * relation_att * k`, softmax over 2 agents, `relation_msg * v`, weighted aggregation, per-type output projection。
- base target: `B=1,L=2,H=64,W=128,C=256,heads=8,dim_head=32`。
- p50 target: `heads=4,inner_dim=128`, relation tensors `(4,4,32,32)`。
- 最新代码新增 fixed type order `[0,1]` static target, 将 q/k/v input projection 和 per-type output projection 纳入 TVM Relax 构图; 单元测试已覆盖参数 plan, 但还没有 H800 latency JSON。

仍缺:
- H800 v3 static-HMSA benchmark 数字。
- dynamic `types`/relation index dispatch, 而不是固定 `[0,1]` type order。
- calibrated TVM mixed INT8 e2e 集成与 AP eval。
- 若继续 Q path, 必须解释为什么当前 mixed INT8 full-attention 负证据还能被后续 scale/requant/schedule 改善; 否则应按负证据停止。

### 4.3 MSwin isolated DepGraph 仍失败
P 轴 scanner 已绕开这个问题, 但 T2 真剪枝时要遵守:
- MSwin head pruning 走手写 slice, 不要指望 torch-pruning 自动推完整关系。
- `relative_indices` 是 buffer, 不应被当参数剪。
- `pos_embedding` 不随 head pruning 改 shape。

---

## §4A 用户新增审查: 当前结果不能支撑哪些结论

用户最新反馈的核心要求: 最终需要的是 **针对 attention-designed fusion 模块的端到端实测表格**, 即剪枝+量化到底带来多少检测 pipeline 速度提升、最终检测 AP50/AP70 下降多少。当前 T1 结果还没达到这个标准。

### 4A.1 逐算子 latency 不是端到端 latency
当前 `linear_qkv_direct`、`hmsa_proj_head_p50_direct` 等结果只是 TVM direct matmul 子算子:
- 不包含 V2XTransformer 前后的 STTF/warping、mask、relative position、softmax、residual、FFN、detector head。
- 不包含 full model 的 tensor layout 转换、host/device 同步、数据搬运、kernel launch 组合开销。
- 不包含 AP 评估链路。

因此这些数只能回答: **attention projection 子算子在 TVM INT8 下有无加速潜力**。不能写成“fusion 模块端到端加速 2.7×”, 更不能写成“检测 pipeline 加速 2.7×”。

### 4A.2 当前 attention 剪枝方法到底是什么
当前已经执行了 **no-finetune attention-p50 真实模型 surgery pilot**: `models/v2xvit_attention_t1/attention_p50_noft_epoch17.pth` + `attention_p50_manifest_v1.json`。方法是 **结构化 head pruning**, 不是 unstructured mask, 也不是随便删参数。

真实剪枝方法:
- HMSA: 删除完整 attention head。同步切 `q/k/v Linear` 的输出通道、`a_linears` 的输入通道、`relation_att[:, head, :, :]` 和 `relation_msg[:, head, :, :]` 的 head 维。
- MSwin: 删除完整 window attention head。同步切 `to_qkv` 输出通道中 Q/K/V 对应 head 的片段, 以及 `to_out[0]` 输入通道。
- 保持 residual 输出维度 `dim=256` 不变, 否则后续 residual add 和 detector head 会断。
- `pos_embedding`、`relative_indices` 不跟 head pruning 改 shape。

仍缺:
- short finetune 后的 AP recovery。
- larger/full DAIR val AP。
- attention-p50-int8/mixed 真实 TVM e2e/AP 行。

### 4A.3 目前没有 attention-pruning finetune
当前 attention-head p50 数字已有 **no-finetune pilot**, 但还没有 short finetune。后续至少要比较:
- no-finetune immediate AP drop: 当前 64-sample pilot AP70 `-0.0712`, 已触发 Stop-C 风险。
- short finetune 后 AP recovery。
- larger/full DAIR val 下 baseline/no-ft/ft 三者是否同协议。

报告里必须写明 finetune setting: epoch 数、学习率、数据集 split、是否从原 ckpt 初始化、是否只 finetune fusion 或全模型。

### 4A.4 “精度为什么不降反升”当前不能归因给 attention pruning
当前 coupling table 里的 AP 来自 `results/coupling_map/C4_QgranxP_v2xvit.json`, 是 full-model fake-quant/global prior, 不是 attention-head pruning 真测。

AP 不降反升的可能原因只能作为待验证假设:
- DAIR/V2X-ViT 可能有过参数化, 剪枝后正则化效应或 finetune 使 AP 恢复。
- baseline ckpt/finetune epoch 不完全等价, 多训练 epoch 会造成“剪枝模型更高”的 confound。
- AP50/AP70 有评估波动, 需要同协议复跑或至少固定 seed/ckpt。
- 当前 prior 不是 TVM INT8, 也不是 attention-only pruning, 不能说明 attention 剪枝本身提高精度。

下一阶段验收时, 如果出现 AP 不降反升, reviewer agent 必须检查是否存在 epoch、seed、dataset、checkpoint、threshold、eval script 不一致。

### 4A.5 最终必须交付的端到端表格
下一阶段的核心产物不是逐算子表, 而是下面这种端到端表:

| config | attention prune | quant | finetune | e2e latency ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 | evidence |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0% | fp16/fp32 | official ckpt | measured | 1.00× | measured | measured | 0 | 0 | log/json |
| attention-p50 | 50% heads | fp16/fp32 | no-ft + ft | measured | measured | measured | measured | measured | measured | log/json |
| attention-p50-int8 | 50% heads | TVM int8/mixed | same as above | measured | measured | measured | measured | measured | measured | log/json |

没有这张表, 不能声称“attention fusion 剪枝+量化带来了端到端优化效果”。

---

## §5 复现命令

### 5.1 本地测试
```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile \
  scripts/phase2/t1_attention_pq_feasibility.py \
  scripts/phase2/t1_attention_tvm_bench.py \
  scripts/phase2/t1_attention_e2e_pq.py \
  scripts/phase2/t1_attention_p50_short_finetune.py \
  scripts/phase2/attention_subnet_accel_report.py \
  scripts/phase2/attention_e2e_checkpoint_eval.py \
  scripts/phase2/attention_final_acceptance_report.py \
  scripts/phase2/attention_e2e_pq_validator.py

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m pytest \
  tests/phase2/test_t1_attention_pq_feasibility.py \
  tests/phase2/test_t1_attention_tvm_bench.py \
  tests/phase2/test_t1_attention_e2e_pq.py \
  tests/phase2/test_t1_attention_p50_short_finetune.py \
  tests/phase2/test_attention_subnet_accel_report.py \
  tests/phase2/test_attention_e2e_checkpoint_eval.py \
  tests/phase2/test_attention_final_acceptance_report.py \
  tests/phase2/test_attention_e2e_pq_validator.py -q
```

当前验证结果:

```text
py_compile passed for all listed scripts
tests/phase2/test_t1_attention_tvm_bench.py: 6 passed
tests/phase2/test_attention_final_acceptance_report.py: 4 passed
prior phase2 set before latest static-HMSA/final-acceptance additions: 37 passed, 2 warnings
attention_final_acceptance_v1 stop-a validator: expected REJECT, missing only attention-p50-int8/mixed e2e metrics/logs
```

### 5.2 H800 TVM direct bench
本地没有 TVM; H800 可用。注意必须显式设置 `PATH` 和 `LD_LIBRARY_PATH`, 否则会遇到 `libcudart.so.12` 符号错误。

```bash
ssh -p 30001 -o StrictHostKeyChecking=accept-new \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  "export PATH=/usr/local/cuda-12.2/bin:\$PATH; \
   export LD_LIBRARY_PATH=\$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path); \
   cd ${V2X_DATA_ROOT}/v2x_t1_attention; \
   CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
     scripts/t1_attention_tvm_bench.py \
     --out-json results/attention_axis_tvm_bench_v1.json \
     --number 30 --repeat 5"
```

传回本地:

```bash
scp -P 30001 -o StrictHostKeyChecking=accept-new \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_DATA_ROOT}/v2x_t1_attention/results/attention_axis_tvm_bench_v1.json \
  ${V2X_ROOT}/results/attention_axis_tvm_bench_v1.json
```

### 5.3 H800 TVM full-attention static-HMSA v3 bench
```bash
scp -P 30001 -o StrictHostKeyChecking=accept-new \
  scripts/phase2/t1_attention_tvm_bench.py \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_DATA_ROOT}/v2x_t1_attention/scripts/t1_attention_tvm_bench.py

ssh -p 30001 -o StrictHostKeyChecking=accept-new \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> 'bash -s' <<'REMOTE'
set -o pipefail
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path)
cd ${V2X_DATA_ROOT}/v2x_t1_attention
mkdir -p results logs
CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/t1_attention_tvm_bench.py \
  --out-json results/attention_full_tvm_bench_v3_static.json \
  --full-attention --full-only --number 3 --repeat 3 \
  > logs/attention_full_tvm_bench_v3_static.log 2>&1
REMOTE

scp -P 30001 -o StrictHostKeyChecking=accept-new \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_DATA_ROOT}/v2x_t1_attention/results/attention_full_tvm_bench_v3_static.json \
  ${V2X_ROOT}/results/attention_full_tvm_bench_v3_static.json
scp -P 30001 -o StrictHostKeyChecking=accept-new \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_DATA_ROOT}/v2x_t1_attention/logs/attention_full_tvm_bench_v3_static.log \
  ${V2X_ROOT}/logs/attention_e2e_pq_v1/attention_full_tvm_bench_v3_static.log
```

注意: `results/attention_full_tvm_bench_v2.json` 是旧 minimal HMSA relation-core 历史结果, 不要覆盖; v3 static 结果应单独落盘, 然后生成 `attention_subnet_accel_v2_static.*`。

### 5.4 汇总 T1 报告
```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/t1_attention_pq_feasibility.py \
  --device cuda \
  --tvm-results results/attention_axis_tvm_bench_v1.json
```

断言:

```bash
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python - <<'PY'
import json
from pathlib import Path
data=json.loads(Path('results/attention_axis_feasibility_v1.json').read_text())
assert data['gate']['overall']['verdict']=='T1_PQ_INCOMPLETE_DO_NOT_START_T2'
assert data['gate']['q_axis']['verdict']=='Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED'
assert data['p_axis']['manual_scanner']['counts']['hmsa_head']==3
assert data['p_axis']['manual_scanner']['counts']['mswin_head']==9
assert any(row['prune_rate_pct']==50 and row['quant']=='int8' for row in data['coupling_table'])
print('gate_assertions_ok')
PY
```

### 5.5 attention-p50 100-step shortft pilot
```bash
cd ${V2X_ROOT}
CUDA_VISIBLE_DEVICES=6 ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/t1_attention_p50_short_finetune.py \
  --device cuda:0 \
  --steps 100 \
  --lr 1e-4 \
  --batch-size 1 \
  --num-workers 2 \
  --seed 20260623 \
  --max-train-samples 512 \
  --eval-samples 64 \
  --latency-warmup 5 \
  --latency-samples 30 \
  --eval-precision fp16
```

输出:
- `results/attention_p50_shortft_v1.json`
- `logs/attention_e2e_pq_v1/attention_p50_shortft_{train,latency,ap}_v1.json`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`

### 5.6 attention subnet TVM acceleration report
```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_subnet_accel_report.py \
  --tvm-report results/attention_full_tvm_bench_v3_static.json \
  --out-json results/attention_subnet_accel_v2_static.json \
  --out-csv results/attention_subnet_accel_v2_static.csv \
  --out-md results/attention_subnet_accel_v2_static.md
```

输出:
- `results/attention_subnet_accel_v2_static.json`
- `results/attention_subnet_accel_v2_static.csv`
- `results/attention_subnet_accel_v2_static.md`

### 5.7 full DAIR val checkpoint eval
```bash
cd ${V2X_ROOT}
CUDA_VISIBLE_DEVICES=6 ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_checkpoint_eval.py \
  --device cuda:0 \
  --precision fp16 \
  --eval-samples 1789 \
  --latency-warmup 5 \
  --latency-samples 30 \
  --num-workers 2 \
  --out-json results/attention_e2e_checkpoint_eval_full_v1.json \
  --out-csv results/attention_e2e_checkpoint_eval_full_v1.csv \
  --out-md results/attention_e2e_checkpoint_eval_full_v1.md
```

输出:
- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_e2e_checkpoint_eval_full_v1.csv`
- `results/attention_e2e_checkpoint_eval_full_v1.md`
- `logs/attention_e2e_pq_v1/*_full_{latency,ap}_v1.json`

### 5.8 Stop-A final acceptance staging table
当前没有真实 TVM e2e row 时:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v1.json \
  --out-json results/attention_final_acceptance_v1.json \
  --out-csv results/attention_final_acceptance_v1.csv \
  --out-md results/attention_final_acceptance_v1.md

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_final_acceptance_v1.json
```

当前 validator 应返回 `REJECT`, 这是预期结果。只有当 full-model TVM runner 生成完整 row 后, 才使用:

```bash
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md
```

---

## §6 关键文件清单

| 文件 | 用途 |
|---|---|
| `scripts/phase2/t1_attention_pq_feasibility.py` | T1-P/T1-Q 主汇总; scanner/gate/coupling table |
| `scripts/phase2/t1_attention_tvm_bench.py` | H800 TVM direct + MSwin full-attention + HMSA minimal relation-core + latest static-HMSA qkv/relation/out FP16/mixed INT8 bench code |
| `scripts/phase2/t1_attention_e2e_pq.py` | 真实 attention-p50 surgery、pilot latency/AP、Stop-B blocker 生成 |
| `scripts/phase2/t1_attention_p50_short_finetune.py` | attention-p50 short finetune runner; 100-step pilot 已跑通 |
| `scripts/phase2/attention_subnet_accel_report.py` | attention subnet TVM 加速汇总: pruning vs mixed INT8 |
| `scripts/phase2/attention_e2e_checkpoint_eval.py` | baseline/no-ft/shortft checkpoint larger/full DAIR val 对照 |
| `scripts/phase2/attention_final_acceptance_report.py` | Stop-A final acceptance staging table; 当前 blocked, 等待 TVM mixed INT8 e2e row |
| `scripts/phase2/attention_e2e_pq_validator.py` | Stop-A/Stop-B 结果口径验收脚本; 防止逐算子/fake AP/非 TVM 结果冒充 e2e |
| `tests/phase2/test_t1_attention_pq_feasibility.py` | TDD 约束: scanner、TVM-only gate、coupling table |
| `tests/phase2/test_t1_attention_tvm_bench.py` | TDD 约束: MSwin/HMSA TVM shape/blocker 元数据; static-HMSA qkv/relation/out 参数 plan |
| `tests/phase2/test_t1_attention_e2e_pq.py` | TDD 约束: HMSA/MSwin surgery、Stop-B 口径 |
| `tests/phase2/test_t1_attention_p50_short_finetune.py` | TDD 约束: shortft manifest 与 train-scope 冻结逻辑 |
| `tests/phase2/test_attention_subnet_accel_report.py` | TDD 约束: subnet TVM pruning/INT8 speedup 拆分 |
| `tests/phase2/test_attention_e2e_checkpoint_eval.py` | TDD 约束: checkpoint eval rows、speedup/delta、repo-relative 输出路径 |
| `tests/phase2/test_attention_final_acceptance_report.py` | TDD 约束: final acceptance 表必须 blocked until TVM e2e row; 未来完整 row 可过 validator |
| `tests/phase2/test_attention_e2e_pq_validator.py` | TDD 约束: e2e PQ 表、blocker 文档、TVM/AP/manifest 口径 |
| `results/attention_axis_feasibility_v1.json` | 当前 T1 总结 JSON |
| `results/attention_axis_feasibility_v1.md` | 当前 T1 人读报告 |
| `results/attention_axis_tvm_bench_v1.json` | H800 TVM 原始 direct 子算子结果 |
| `results/attention_full_tvm_bench_v1.json` | H800 TVM full MSwin report + 旧 HMSA NOT_IMPLEMENTED blocker |
| `results/attention_full_tvm_bench_v2.json` | H800 TVM full MSwin + HMSA minimal relation-core report; mixed INT8 均为负/中性 |
| `logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log` | H800 v2 远端运行日志 |
| `results/attention_e2e_pq_blocker_v1.md` | reviewer ACCEPT 的 Stop-B blocker/负证据记录 |
| `results/attention_e2e_pq_partial_v1.{json,csv}` | baseline vs attention-p50-fp16 no-ft vs 100-step shortft 64-sample pilot |
| `results/attention_p50_shortft_v1.json` | 100-step shortft train/eval/latency result |
| `results/attention_subnet_accel_v1.{json,csv,md}` | attention subnet TVM 加速表; p50 FP16 positive, mixed INT8 negative vs same prune |
| `results/attention_e2e_checkpoint_eval_v1.{json,csv,md}` | 256-sample checkpoint eval pilot |
| `results/attention_e2e_checkpoint_eval_full_v1.{json,csv,md}` | full DAIR val checkpoint eval; p50-shortft FP16 通过 AP/speed guardrail |
| `results/attention_final_acceptance_v1.{json,csv,md}` | 当前 final acceptance staging 表; Stop-A validator 预期 REJECT, 缺 TVM e2e row |
| `results/attention_e2e_pq_review_v1.md` | reviewer 对 Stop-B 包的最终验收, verdict `ACCEPT` |
| `models/v2xvit_attention_t1/attention_p50_manifest_v1.json` | 真实 attention-p50 manifest |
| `results/coupling_map/C4_QgranxP_v2xvit.json` | 仅作 fake-quant AP prior |
| `${V2X_HOME}/heal_research/HEAL/opencood/models/sub_modules/hmsa.py` | HMSA 实现参考 |
| `${V2X_HOME}/heal_research/HEAL/opencood/models/sub_modules/mswin.py` | MSwin/BWA 实现参考 |

---

## §7 下一步计划

### T1-QC — HMSA TVM feasibility
状态: **已完成 minimal target 实测 + static target 代码**, 但结论仍未到 Stop-A。
- `hmsa_full_attention.fp16/mixed_int8.status=OK`。
- `hmsa_p50_full_attention.fp16/mixed_int8.status=OK`。
- mixed INT8 慢于 FP16: base `0.9556×`, p50 `0.9700×`。
- 旧 v2 caveat: 仅覆盖 fixed B=1,L=2 explicit relation core。
- 最新代码 caveat: static `[0,1]` HMSA 已覆盖 q/k/v projection + relation core + output projection, 但尚未 H800 重测, 仍不覆盖 dynamic `types` dispatch/e2e AP。

### T1-QD — full attention Q 负证据/收益判定
目标: 不再只看 direct matmul。
- 已知 MSwin 和 HMSA minimal core mixed INT8 当前均慢于 FP16。
- subnet 汇总显示 p50 FP16 attention subnet `1.7783×`, p50 mixed INT8 `1.7069×`, 但 mixed INT8 相对同剪枝 FP16 是 `0.9598×`。
- 下一步先用 static-HMSA v3 重跑 subnet; 若 p50 mixed INT8 仍慢于同剪枝 FP16, 必须做 fused/calibrated attention TVM schedule 并复测。
- 默认判断: 当前 synthetic mixed INT8 不能作为 latency benefit。
- 如果没有 calibrated TVM mixed INT8 e2e row, 不允许写 `attention-p50-int8/mixed` 收益。

### T1-AP — finetune + larger/full split
目标: 处理 no-ft p50 pilot 的 Stop-C 风险。
- 100-step short finetune pilot 已完成: AP50/AP70 在 64-sample 上恢复, 但 latency 变慢。
- 256-sample larger pilot 已完成: p50-shortft speedup `1.2224×`, ΔAP50=`+0.0125`, ΔAP70=`+0.0073`。
- full DAIR val 已完成: p50-shortft speedup `1.0784×`, ΔAP50=`-0.0128`, ΔAP70=`-0.0084`; FP16 剪枝+微调通过当前 guardrail。
- 剩余关键缺口: `attention-p50-shortft-TVM-mixedINT8` 的完整 e2e/AP 行。
- `results/attention_final_acceptance_v1.*` 已经把缺口显式化; 只有填入完整 `attention-p50-int8/mixed` row 后才能生成 `results/attention_e2e_pq_v1.*` 并进入 Stop-A。

---

## §7A 下一阶段明确停止目标

下一阶段不是“继续探索到没有问题”, 而是到达下面任一 stop condition 即停止并汇报。

### Stop-A: 成功停止条件
产出一个可审计的端到端 attention fusion prune+quant 表格, 写入:
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.md`

表格至少包含 3 行:
- baseline: unpruned V2X-ViT, same eval split, measured e2e latency, AP50/AP70。
- attention-p50-fp16: HMSA/MSwin head prune 50%, measured e2e latency, AP50/AP70, no-ft 和/或 ft 必须标清。
- attention-p50-int8/mixed: 同一个 pruned checkpoint + TVM INT8/mixed attention path, measured e2e latency, AP50/AP70。

每一行必须包含:
- exact checkpoint path / manifest path。
- prune manifest: HMSA keep_heads、MSwin keep_heads、是否保留 dim=256。
- quant backend: TVM direct/mixed, 不能写非 TVM。
- e2e latency command 和原始 log。
- AP eval command 和原始 log。
- dataset split 和 sample 数。

### Stop-B: 阻塞停止条件
如果 full attention TVM INT8/mixed 在合理工程范围内无法完成, 必须产出 blocker 文档:
- `results/attention_e2e_pq_blocker_v1.md`
- 包含具体失败点: ONNX import、Relax lowering、TIR compile、runtime correctness、AP eval、或 model surgery shape mismatch。
- 仍需给 baseline 与 attention-p50-fp16 的 e2e latency/AP, 以便至少回答剪枝本身是否值得继续。

### Stop-C: 否定停止条件
如果端到端表显示:
- speedup < 1.05×, 或
- AP70 下降 > 0.02 且 finetune 后不能恢复, 或
- attention fusion 占比太小导致 Amdahl 稀释,

则停止推进该 attention fusion PQ 方向, 把结果写成负证据。不要继续为得到正结果更换口径。

---

## §7B 验收方案

### 必须通过的验收项
1. **端到端一致性**: baseline、pruned、pruned+int8 使用同一 DAIR val split、同一 eval script、同一 batch/agent/input shape。
2. **latency 是端到端**: 至少覆盖 V2X-ViT fusion 到 detection head 的实际 forward path; 如果只是 fusion subgraph, 表头必须写 `fusion-subgraph`, 不得写 e2e。
3. **AP 是真实评估**: AP50/AP70 必须来自 eval log/json, 不能用 C4 fake-quant prior 替代。
4. **剪枝是真的模型剪枝**: 必须有 pruned checkpoint + manifest, 不能只用 direct matmul shape 代表剪枝。
5. **量化是真的 TVM 路径**: INT8/mixed 需要 TVM 产物或 TVM runtime 证据; QDQ-ONNX fake path 不通过。
6. **correctness smoke**: pruned/quantized model forward 不 NaN, 输出 shape 与 baseline detection head 对齐。
7. **可复现**: 报告里必须有命令、日志路径、commit/diff 状态、环境路径。

机器验收脚本:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py --mode auto
```

若 Stop-A/Stop-B 产物仍不存在, 该脚本会返回 `MISSING_RESULT`; 若表格混入 `direct_matmul`、`SIMULATED_*`/fake AP、非 TVM INT8、或 pruned 行缺 checkpoint/manifest, 会返回 `REJECT`。

### 验收表的判定规则
建议初始通过阈值:
- speedup: `pruned+int8 e2e latency <= baseline latency / 1.10` 才算有系统收益。
- accuracy: `ΔAP70 >= -0.02` 且 `ΔAP50 >= -0.02` 才算检测精度可接受。
- 如果只达到 fusion-subgraph speedup, 但 e2e speedup 不达标, 结论写为 “operator/fusion 局部可加速, e2e 暂无系统收益”。

### reviewer 必查问题
- AP 不降反升时: 查 finetune epoch、lr、seed、eval sample、checkpoint、NMS/threshold 是否一致。
- speedup 很大时: 查是否漏掉 STTF/mask/softmax/layout conversion 或只测了 projection。
- INT8 很快但 AP 没有真实结果时: 不允许写成最终优化。
- full model 导出失败时: 把 blocker 定位到具体 op, 不接受 “TVM 不支持” 这种泛化表述。

---

## §7C 双 agent 分工要求

清上下文后启动下一阶段时, 明确拉起两个 agent, 不能只由执行者自证。

### Agent-1: `attention-pq-executor`
职责: 方案执行。
- 实现 attention head pruning surgery。
- 生成 pruned checkpoint 和 prune manifest。
- 跑 full attention/fusion/e2e latency。
- 跑 AP eval。
- 写 `results/attention_e2e_pq_v1.{csv,json,md}`。

交付物:
- pruned model artifact。
- TVM/latency logs。
- AP logs。
- 端到端表格。
- 失败时的 blocker 文档。

### Agent-2: `attention-pq-reviewer`
职责: 批判性验收, 不参与实现。
- 独立读取 executor 的 manifest/log/json。
- 按 §7B 逐项验收。
- 对不合理结果提出纠正要求: 例如逐算子冒充 e2e、fake AP 冒充真实 AP、finetune 协议不一致、量化路径不是 TVM。
- 必须输出 `results/attention_e2e_pq_review_v1.md`。

reviewer 的结论只允许三种:
- `ACCEPT`: 端到端 latency/AP 表真实可用。
- `REVISE`: 有口径或实验缺陷, executor 必须补跑。
- `REJECT`: 结果不能支持 attention PQ 方向, 停止推进并记录负证据。

---

## §8 纪律 / 诚实边界
- **不要启动 T2**: 当前 overall gate 明确是 `T1_PQ_INCOMPLETE_DO_NOT_START_T2`。
- **不要用非 TVM 引擎补 Q 轴**: 非 TVM engine 结果只能做历史参考, 不进入 T1-Q gate。
- **不要把 QDQ-ONNX 当真实 INT8**: direct TVM int8 op 才能算 Q 轴证据。
- **不要把 direct matmul speedup 写成 full attention speedup**: 当前 2.1–2.7× 是 projection 子算子层面的剪枝+INT8效果。
- **不要把 C4 AP prior 写成真实 attention-head pruning AP**: 当前 AP 是 full-model fake-quant prior。
- H800 TVM 必须用 fresh/明确环境: `PATH=/usr/local/cuda-12.2/bin:$PATH` + `LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path)`。
- 延迟测量要看 idle GPU, 本轮用 GPU6 空闲测得。

---

## §9 清上下文后的最短恢复流程
1. 读本文件。
2. 打开 [`results/attention_axis_feasibility_v1.md`](../../../results/attention_axis_feasibility_v1.md), [`results/attention_e2e_pq_blocker_v1.md`](../../../results/attention_e2e_pq_blocker_v1.md), [`results/attention_e2e_pq_review_v1.md`](../../../results/attention_e2e_pq_review_v1.md), 确认当前 gate 与 reviewer `ACCEPT` 仅限 Stop-B 记录质量。
3. 拉起两个 agent:
   - `attention-pq-executor`: 继续 §7 的 finetune/full-val gate 与 e2e quant integration 判定。
   - `attention-pq-reviewer`: 按 §7B 批判性验收, 特别检查 Stop-C 风险、MSwin negative INT8 evidence、HMSA trace 是否具体。
4. executor 不要重复实现 HMSA minimal TVM target; 直接读取 `results/attention_full_tvm_bench_v2.json` 与 `logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log`。
5. 不要重复跑 full-val FP16; 直接读取 `results/attention_e2e_checkpoint_eval_full_v1.json`。
6. 下一步补 `attention-p50-shortft-TVM-mixedINT8` e2e/AP: 必须先解决 dynamic HMSA dispatch/qkv projection + calibrated/fused TVM mixed INT8 integration; 否则把 Q path 写成负证据停止。
7. 跑 §5 的测试、gate 断言, `scripts/phase2/attention_e2e_pq_validator.py --mode auto/stop-b`, 以及 reviewer 指定的复核命令。
8. 只有 validator 不报 `REJECT`、reviewer 给出最终 `ACCEPT`, 且 HMSA+MSwin TVM INT8/mixed benchmark 和真实 AP 边界明确后, 才能重新讨论 T2。
