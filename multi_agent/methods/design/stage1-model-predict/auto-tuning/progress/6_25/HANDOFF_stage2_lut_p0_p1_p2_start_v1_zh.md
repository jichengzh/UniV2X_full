# HANDOFF: Stage2 LUT P0/P1/P2 启动进展 v1

日期: 2026-06-25  
范围: H800 + TVM, Pyramid backbone-only / dense-core 过渡 LUT。

## 1. 本轮结论

本轮已完成 P0 机制补强, 并启动/完成一批 P1/P2 补点:

| 项 | 状态 | 说明 |
|---|---|---|
| P0 bad-DB quarantine | 已实现 | 新增 `lut_bad_db_quarantine_row_v1`; seed 中包含 `s1_64 tuned latency` 与 `pad64 tuned energy` |
| P0 worker 进程隔离 | 已实现 | `stage2_lut_worker.py` 支持每 job 子进程失败隔离、quarantine skip |
| P0 TVM runtime env | 已实现 | worker 默认注入 `tvm310` CUDA runtime / TVM lib, 避免 `cudaGraphAddDependencies_v2` |
| P0 GPU idle preflight | 已实现 | worker 支持 `--require-gpu-idle`, busy GPU 写 `preflight_blocked` |
| P1 AP 补点 | 已完成 5 条 | `iso_s0/iso_s1/iso_s2/mix_a/p50b2_136` true AP anchors |
| P1 energy 补点 | 已完成 5 条 | `base/p50/p75/mix_b/mix_d` H800 telemetry rows |
| P2 latency repeat | 已完成 10 条 | `base/p50/trap25/mix_b/mix_d` x default/tuned |
| P1 new latency | 已尝试但阻塞 | `p50b2_136` 与 `mix_e_64_64_128` 远端缺 ONNX artifact |

当前仍不是 full unattended tri-axis 大规模启动。可以继续 allowlist 后台补点, 但 full grid 需要先补齐新 width 的 ONNX/MetaSchedule artifact, 并复测 trap25 P2 outlier。

## 2. 数据位置

P0 quarantine:

```text
multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl
```

P1/P2 runner 与状态:

```text
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/jobs/p1_p2_h800_gpu012_runner.py
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/jobs/job_state_p1_p2_gpu012.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/compare/p1_p2_h800_gpu012_summary_v1.json
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/compare/p1_p2_quick_review_v1.csv
```

Canonical rows 当前计数:

| 文件 | rows |
|---|---:|
| calibration latency `latency/latency_lut_rows_v1.jsonl` | 44 |
| calibration energy `energy/energy_lut_rows_v1.jsonl` | 5 |
| smoke AP `ap/ap_anchor_rows_v1.jsonl` | 11 |

## 3. 当前可读结果表

单位: latency 为 ms, energy 为 J/inference。

| label | AP70 | energy | default latency | tuned latency | 备注 |
|---|---:|---:|---:|---:|---|
| base | 0.630864 | 1.924400 | 56.323715 | 6.212762 | P2 latency repeat + P1 energy |
| p50 | 0.564132 | 0.695417 | 26.239023 | 3.010426 | P2 latency repeat + P1 energy |
| trap25 | 0.590472 | 待测 | 42.358457 | 30.169297 | P2 repeat 有 outlier, 需复测 |
| p75 | 0.529988 | 0.057155 | 12.674737 | 0.483032 | energy P1 已补 |
| mix_b | 0.636160 | 4.682046 | 41.429098 | 19.401178 | P2 latency repeat + P1 energy |
| mix_d | 0.636938 | 5.314299 | 42.436141 | 21.050262 | P2 latency repeat + P1 energy |
| iso_s0 | 0.629919 | 待测 | 51.191752 | 21.722840 | AP P1 已补 |
| iso_s1 | 0.633599 | 待测 | 51.623513 | 6.917596 | AP P1 已补 |
| iso_s2 | 0.633890 | 待测 | 51.956699 | 7.251650 | AP P1 已补 |
| mix_a | 0.628755 | 待测 | 35.793215 | 5.658444 | AP P1 已补 |
| p50b2_136 | 0.642530 | 待测 | 待测 | 待测 | AP P1 已补; H800 latency artifact 缺失 |

## 4. 异常和处理

1. energy runner 第一次失败不是测量失败, 而是 generator 路径/依赖问题:
   - 远端缺 `${V2X_HOME}/miniconda3/bin/python3`。
   - 改成 `GEN_PY = sys.executable` 后又遇到 TVM Python 缺 `yaml`。
   - 已修复 `framework/stage2/__init__.py`, 使 LUT-only import 不依赖 `contracts.py/yaml`。
   - 已用 raw 中的 `telemetry_payload.json` salvage 写入 5 条 canonical energy rows。

2. `trap25` P2 repeat tuned 出现 outlier:
   - repeats: `21621.723, 21622.115, 30169.297, 46872.504, 46430.875 us`。
   - p50 因 outlier 变成 `30.169 ms`, 不能作为 paper-grade 稳定结论。
   - 下一轮需要单独 clean GPU 复测 trap25 tuned。

3. P1 new latency 两个点缺 artifact:
   - `p50b2_136`: 远端缺 `p50b2_136_backbone.onnx` / `p50b2_136.onnx` / stage_a ONNX。
   - `mix_e_64_64_128`: 远端缺候选 ONNX。
   - 当前只写 failed job_state, 未写 measured latency row。

## 5. 下一步计划

P0 follow-up:

1. 给 runner 加 resume/skip-existing row 逻辑, 避免 generator 失败后整批重跑。
2. 把 energy salvage 写成正式脚本, 输出单独 job_state。
3. 将 `framework/stage2/__init__.py` 的 lightweight import 保留为产品化路径要求。

P1:

1. 补远端 ONNX / MetaSchedule DB:
   - `p50b2_136`
   - `mix_e_64_64_128`
2. 继续补 energy:
   - `iso_s1`, `iso_s2`, `mix_a`
   - 暂停 `pad64` tuned energy, 等 DB/kernel 修复。
3. AP 只继续导入 true eval anchors; `pad64` weight-identity 先不写 measured AP。

P2:

1. 复测 `trap25` tuned latency, 至少 2 次 clean repeat。
2. 对 `base/p50/mix_b/mix_d` 做第二轮 paper-grade repeat, 形成稳定性表。
3. 完成 registry update/export, 但 full unattended tri-axis 仍需等待 artifact 缺口关闭。

## 6. 验证

本地已验证:

```text
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_evidence_registry -q
Ran 27 tests OK
```

Schema validation 通过:

```text
latency rows: 44 ok
calibration energy rows: 5 ok
AP rows: 11 ok
quarantine rows: 2 ok
```
