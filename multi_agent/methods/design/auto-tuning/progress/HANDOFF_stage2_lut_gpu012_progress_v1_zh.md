# HANDOFF: Stage2 LUT GPU0/1/2 本轮进展与补点计划 v1

日期: 2026-06-25  
范围: H800 + TVM, 以 Pyramid backbone-only / dense-core 过渡 LUT 为主, 同步记录 AP/energy smoke gate。

## 1. 本轮结论

本轮在 GPU0/1/2 可用后继续补测。实际执行中 GPU0 后续被其他用户进程占用, 因此有效补测主要使用 GPU1/GPU2。

当前不是全量 `GO`, 而是分层启动:

| 启动类型 | 判定 | 说明 |
|---|---|---|
| latency 后台补点 | `GO_CONDITIONAL` | 可继续补低风险 FP16 点, 但必须排除/隔离 `s1_64` tuned DB/kernel 问题 |
| AP generator/import | `GO` | 目前已有 6 条 true AP anchors, 可继续扩展 AP rows |
| energy 串行补点 | `GO_CONDITIONAL` | base/p50 已通过; pad64 tuned energy 暂停, 需先修复 tuned kernel/DB |
| full unattended tri-axis large LUT | `NO_GO` | 还不能无监督全量跑, 因为 latency `s1_64` 和 energy `pad64` 都暴露 tuned CUDA illegal address |

## 2. 数据产物位置

latency calibration:

```text
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/
```

smoke AP/energy:

```text
multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/
```

关键 summary:

```text
generated/calibration/latency_h800_tvm_v1/compare/latency_calibration_progress_summary_v1.json
generated/calibration/latency_h800_tvm_v1/compare/latency_calibration_valid_quick_review_v1.csv
generated/smoke/h800_tvm_v1/compare/smoke_summary_v2.json
generated/smoke/h800_tvm_v1/compare/energy_smoke_supplement_gate_v1.json
```

## 3. 当前实测配置总览

说明:

- `量化/数值精度` 指当前 LUT row 的 quant policy, 本轮主要是 `FP16`。
- `AP70` 是检测精度指标, 来源为 true AP anchor/imported eval evidence。
- latency 单位统一为 `ms`; energy 单位为 `J / inference`。
- 表内 `待测` 不代表失败, 只是该轴还没有 measured row。
- `s1_64` default 诊断在 failed raw 中有数值, 但 tuned job 失败, 因此不写 canonical measured row。

| model | config | width | 量化/数值精度 | AP70 | energy J/inf | default latency ms | tuned latency ms | 状态 |
|---|---|---|---|---:|---:|---:|---:|---|
| Pyramid | base | [64,128,256] | FP16 | 0.630864 | 2.050199 | 56.260 | 6.210 | 三轴 smoke/calibration 均有 measured row |
| Pyramid | p50 | [32,64,128] | FP16 | 0.564132 | 0.776318 | 26.231 | 3.012 | 三轴 smoke/calibration 均有 measured row |
| Pyramid | p75 | [16,32,64] | FP16 | 0.529988 | 待测 | 12.675 | 0.483 | latency/AP measured; energy 待测 |
| Pyramid | trap25 | [48,96,192] | FP16 | 0.590472 | 待测 | 42.327 | 21.605 | latency/AP measured; energy 待测 |
| Pyramid | pad64 | [64,96,192] | FP16 | 待测 | 失败阻塞 | 47.371 | 6.164 | latency measured; energy tuned kernel failed; AP 待测 |
| Pyramid | iso_s0 | [48,128,256] | FP16 | 待测 | 待测 | 51.192 | 21.723 | latency measured; AP/energy 待测 |
| Pyramid | iso_s1 | [64,96,256] | FP16 | 待测 | 待测 | 51.624 | 6.918 | latency measured; AP/energy 待测 |
| Pyramid | iso_s2 | [64,128,192] | FP16 | 待测 | 待测 | 51.957 | 7.252 | latency measured; AP/energy 待测 |
| Pyramid | s2_128 | [64,128,128] | FP16 | 待测 | 待测 | 61.399 | 7.314 | latency measured; AP/energy 待测 |
| Pyramid | s0_16 | [16,128,256] | FP16 | 待测 | 待测 | 39.468 | 6.898 | latency measured; AP/energy 待测 |
| Pyramid | s0_32 | [32,128,256] | FP16 | 待测 | 待测 | 44.634 | 5.968 | latency measured; AP/energy 待测 |
| Pyramid | s1_32 | [64,32,256] | FP16 | 待测 | 待测 | 41.887 | 3.761 | latency measured; AP/energy 待测 |
| Pyramid | s1_64 | [64,64,256] | FP16 | 待测 | 待测 | 诊断值存在 | 失败阻塞 | tuned VM 在 clean GPU1/GPU2 上复现 CUDA illegal memory access |
| Pyramid | s2_64 | [64,128,64] | FP16 | 待测 | 待测 | 43.568 | 5.033 | latency measured; AP/energy 待测 |
| Pyramid | mix_a | [32,96,192] | FP16 | 待测 | 待测 | 35.793 | 5.658 | latency measured; AP/energy 待测 |
| Pyramid | mix_b | [48,64,256] | FP16 | 0.636160 | 待测 | 41.446 | 19.390 | latency/AP measured; energy 待测 |
| Pyramid | mix_c | [16,128,128] | FP16 | 待测 | 待测 | 30.730 | 5.103 | latency measured; AP/energy 待测 |
| Pyramid | mix_d | [48,128,128] | FP16 | 0.636938 | 待测 | 42.409 | 21.044 | latency/AP measured; energy 待测 |
| CoDriving | base | [64,128,256] | FP16 | 待测 | 待测 | 待测 | 8.039 | latency smoke measured; AP/energy 待测 |

## 4. 本轮补测细节

### 4.1 latency calibration

当前 canonical latency rows:

| 类别 | row 数 | 说明 |
|---|---:|---|
| calibration small | 18 | 9 个 config x default/tuned, preflight-clean |
| calibration medium clean | 16 | 8 个 medium config x default/tuned, preflight-clean |
| total valid canonical | 34 | schema validator 通过 |
| medium quarantine | 18 | 旧污染批次, 只作诊断 |

medium clean 已完成:

```text
s0_16, s0_32, s1_32, s2_64, mix_a, mix_b, mix_c, mix_d
```

`s1_64` 阻塞:

- clean g012 批次中, 因上一批 GPU 上下文尚未释放, preflight 拒绝, 没有采信。
- env-fixed retry 后, TVM import 问题已修复。
- 后续在 clean GPU1/GPU2 上, default 阶段能跑, tuned VM 在 `dev.sync()` 处复现 `CUDA illegal memory access`。
- 不写 canonical measured row, 避免 default-only 诊断值污染主表。

最新 raw:

```text
generated/calibration/latency_h800_tvm_v1/raw/calib_h800_tvm_pyramid_s1_64_fp16_medium_clean_retry_g1_20260625_210525/
```

### 4.2 AP smoke

当前 AP rows:

| label | AP70 | 来源 |
|---|---:|---|
| base | 0.630864 | true AP anchor import |
| p50 | 0.564132 | true AP anchor import |
| trap25 | 0.590472 | true AP anchor import |
| p75 | 0.529988 | true AP anchor import |
| mix_b | 0.636160 | true AP anchor import |
| mix_d | 0.636938 | true AP anchor import |

AP 当前可继续扩展, 但必须保留 dataset/split/ckpt/protocol, 不能用 predicted AP 写 measured claim。

### 4.3 energy smoke

当前 energy rows:

| label | J / inference | idle W | active W avg | 状态 |
|---|---:|---:|---:|---|
| base | 2.050199 | 72.668 | 401.955 | measured |
| p50 | 0.776318 | 72.910 | 330.688 | measured |
| pad64 | 无 measured row | - | - | tuned kernel failed |

pad64 energy retry:

- GPU1 preflight clean。
- tuned TVM VM 运行触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。
- failure kernel: `fused_conv2d15_add8_relu6_kernel`。
- 未写 measured energy row。

失败 raw:

```text
generated/smoke/h800_tvm_v1/raw/smoke_h800_tvm_energy_pyramid_pad64_fp16_tuned_20260625_211139/
```

## 5. 当前阻塞项

| 阻塞项 | 影响 | 当前判断 | 下一步 |
|---|---|---|---|
| `s1_64` tuned latency illegal memory access | medium calibration 缺 1 个 config x 2 rows | MetaSchedule DB/kernel 稳定性问题 | 重建 DB 或加入 bad-DB quarantine |
| `pad64` tuned energy illegal address | energy smoke 缺 pad64 row | tuned kernel 在 energy path 下失败 | 复用 known-good latency path 或重建 DB 后重测 |
| SSH `Exceeded MaxStartups` | 状态轮询/rsync 偶发失败 | 远端未认证连接数上限 | 长任务减少轮询, 优先单连接/低频同步 |
| GPU0/GPU7 仍有其他用户进程 | 可用 GPU 不稳定 | 不能假设 GPU0/1/2 全部空闲 | 每个 job 启动前重新 preflight |

## 6. 下一阶段补点计划

### P0: 先补机制, 再补数据

1. 实现 bad-DB quarantine 机制:
   - 任意 config 发生 TVM/CUDA illegal memory access 时, 写入 bad DB 列表。
   - 后续 worker 自动跳过该 config 或只跑 default/probe, 不写 measured tuned claim。
   - `s1_64` 和 `pad64 energy tuned` 先进入 quarantine。

2. 强化 worker 进程隔离:
   - 每个 config 单独子进程。
   - 子进程崩溃只影响当前 config。
   - 父进程继续后续 allowlist job, 并写 failed job_state/raw artifact。

3. 固定 TVM runtime 环境:
   - `LD_LIBRARY_PATH` 必须优先包含 `tvm310` 自带 `nvidia/cuda_runtime/lib` 和 `tvm/lib`。
   - 否则会出现 `cudaGraphAddDependencies_v2` undefined symbol。

4. 继续执行 GPU idle 硬门槛:
   - target GPU util <= 5%。
   - target GPU memory <= 1024 MiB。
   - `nvidia-smi pmon` 无 compute process。
   - energy 默认串行。

### P1: 条件启动补点

latency:

| 批次 | 范围 | 策略 |
|---|---|---|
| L1 | 已成功 medium 周边继续扩展 | exclude `s1_64`; 每 GPU 1 job; 每点 default+tuned |
| L2 | `s1_64` 修复后回补 | 先单点复测, 通过后再写 canonical |
| L3 | CoDriving low-risk anchors | 先补 base/p50 之外的少量对照点, 避免直接大网格 |

AP:

| 批次 | 配置 | 说明 |
|---|---|---|
| A1 | pad64, iso_s1, s0_32, s1_32, s2_64 | 优先覆盖 latency 已测且 tuned 表现有代表性的点 |
| A2 | mix_a, mix_c, iso_s0, iso_s2, s2_128 | 扩展 AP 对 latency 前沿/异常点的解释能力 |
| A3 | CoDriving base/p50 | 若需要跨模型 DS 对照, 再补 |

energy:

| 批次 | 配置 | 说明 |
|---|---|---|
| E1 | base/p50 repeat | 先做 repeat, 确认 energy telemetry 稳定性 |
| E2 | p75, mix_b, mix_d | 避开已知 pad64 tuned failure, 先补安全点 |
| E3 | pad64 | 仅在 tuned kernel/DB 修复后重测 |

### P2: paper-grade 补点

paper-grade 只在 P0/P1 通过后启动:

1. 固定 GPU 与 clock/power policy。
2. 尽量整机低负载, 不和其它用户 GPU 任务并行。
3. latency/energy 每个关键配置至少 2 次 repeat raw artifact。
4. AP fresh eval 与 imported true anchor 必须区分 provenance。
5. 统一更新 registry/export/docs。

## 7. 大规模启动判定

当前可以启动的是“带 allowlist/quarantine 的后台补点”, 不是“全量无监督大规模 LUT”。

允许启动条件:

- 排除 `s1_64` tuned 与 pad64 energy tuned。
- worker 已支持 per-config failure isolation。
- 每个 job 运行前做 GPU idle preflight。
- measured rows 只从成功 raw 写入 canonical JSONL。

暂不允许启动条件:

- 不允许 full grid 直接无监督跑。
- 不允许把 failed/default-only 诊断值写作 measured tuned row。
- 不允许用 predicted AP/energy 填补 measured claim。

## 8. 2026-06-25 P0/P1/P2 启动更新

后续进展已记录在:

```text
multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_lut_p0_p1_p2_start_v1_zh.md
```

摘要:

- P0 已实现 bad-DB quarantine、worker per-job failure isolation、TVM runtime env 注入、GPU idle preflight。
- P1 已补 AP 5 条 true anchors, calibration energy 5 条 measured rows。
- P2 已补 latency repeat 10 条 rows: `base/p50/trap25/mix_b/mix_d` x default/tuned。
- `p50b2_136` 与 `mix_e_64_64_128` 的 H800 latency 新点因远端 ONNX artifact 缺失失败, 未写 measured row。
- `trap25` P2 repeat tuned 有 outlier, 需 clean GPU 单点复测后再作为 paper-grade。
