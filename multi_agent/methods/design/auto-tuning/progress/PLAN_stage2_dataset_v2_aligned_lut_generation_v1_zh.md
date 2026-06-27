# Stage2 Dataset-v2 Aligned LUT Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 `multi_agent/data/dataset_v2.csv` 的 63 个 AP-valid 真实点为基线, 建立 Stage2 latency/AP/energy 产品化 LUT 生成计划, 先做同配置验证, 再进入 smoke、calibration、paper 三阶段补点。

**Architecture:** 已有 `dataset_v2.csv` 不被改写, 作为 historical measured seed。新生成数据统一进入 `multi_agent/data/stage2_lut_generation_v1/`, 使用 `generate_latency_lut` / `generate_ap_lut` / `generate_energy_lut` 主路径生成 canonical JSONL rows, 并通过 validation compare 把新链路和历史链路对齐。所有大规模任务通过 job plan + worker 长时间后台运行, 每个 row 保留 raw artifact 和 scope/backend/provenance。

**Tech Stack:** Python stdlib CSV/JSON, Stage2 LUT productization scripts, H800 TVM/Relax/MetaSchedule, 4090/Orin historical dataset_v2 evidence, NVML/telemetry energy rows, JSONL canonical tables.

---

## 章节 1: 前置知识

### 1.1 必须遵守的规则

1. **不改写历史事实。** `multi_agent/data/dataset_v2.csv` 是历史实测总表, 不在本计划中原地修改; 只复制和转写到新目录。
2. **Stage1 manifest 不提供实测。** Stage1 只给结构、硬件合法性和 coverage; latency/AP/energy 必须来自 LUT row 或明确标为 historical/proxy/failed。
3. **主路径是 `generate_*_lut`。** 新增实测必须走 `scripts/stage2_generate_latency_lut.py`, `scripts/stage2_generate_ap_lut.py`, `scripts/stage2_generate_energy_lut.py`; `import_existing_*` 只能用于历史 seed 或一次性迁移。
4. **backend 不混写。** H800 TVM 只写 `backend=h800_tvm`; H800 energy 只写 `backend=h800_tvm_power_telemetry`; 4090 TRT / Orin 历史数据不能伪装成 H800 measured。
5. **GPU 空闲是硬门槛。** latency 和 energy 启动前必须记录 `nvidia-smi --query-gpu ...` 与 `nvidia-smi pmon -c 1`; 目标 GPU 有计算进程或显存异常常驻时不写 measured row。
6. **energy no-claim。** 没有同 backend/scope 的 measured energy row 时, registry 继续 no-claim; 不能用 latency 或功率 proxy 替代。
7. **AP 不跨协议乱复用。** AP row 必须记录 dataset/split/ckpt/finetune protocol。FP16 AP 不自动等于 INT8 AP; Orin INT8 行若 `ap_valid=False` 不允许输出 AP claim。
8. **DS 仍是验证项。** AP x latency -> DS 只在 DS scope 有效时可用; predicted AP + predicted latency 的 DS lookup 默认 report-only, 不进入主 claim。
9. **失败也写证据。** build/eval/telemetry 失败必须写 failed row 或 job_state, 保留 `failure_reason`, 不能静默丢点。
10. **口径先于数值。** 任意表格合并前必须检查 `hardware`, `latency_kind`, `optimized_scope`, `regime`, `backend`, `ap_valid`, `source`。
11. **测试环境。** 统一使用H800和TVM作为新数据的生成环境 

### 1.2 Scope 规则

本计划同时管理三类 scope, 不允许混用:

| scope / latency_kind | 用途 | 可比较范围 | 禁止行为 |
|---|---|---|---|
| `body_subnet_collab2` | dataset_v2 4090 collab2 body subnet 主历史口径 | 4090 TRT collab2 内部 | 当成 H800 或 full-model latency |
| `engine_board_energy` | E4 board energy bench | E4 energy regime 内部 | 和 `body_subnet_collab2` 混成单一 latency 前沿 |
| `body_subnet_collab2_orin` | Orin body subnet | Orin 内部 | 和 4090 绝对能耗直接比较 |
| `dla_pipeline_e2e_single_frame` | Orin DLA pipeline e2e | Orin DLA pipeline 内部 | 和 body subnet latency 混用 |
| `backbone_only` | 当前 H800 smoke 的产品化链路验证 | H800 backbone-only 内部 | 写成 `rsu_dense_core` 或 full-model claim |
| `rsu_dense_core` | 最终 Stage2 H800 主目标 | 同硬件同 backend 同 dense-core scope | 用 historical TRT 或 backbone-only 代替 |

### 1.3 当前结论

1. `dataset_v2.csv` 有 66 行、93 列, 其中 63 行 `ap_valid=True`; 这 63 行已经作为 seed 数据导出到新数据目录。
2. dataset_v2 的主体是 4090/Orin 历史实测, 不是 H800 TVM 新实测。（对比的时候需要考虑这一点）
3. 当前 H800 产品化链路已经跑通一条真实 latency smoke: `backend=h800_tvm`, `optimized_scope=backbone_only`, `latency_p50_us=56129.53`。（之后的测试单位统一用ms）
4. 当前还没有完成 H800 AP generator 与 H800 energy telemetry generator 的真实 smoke。
5. 大规模启动前必须先做同配置 replay validation, 证明当前生成方案能在已有配置上复现或解释 dataset_v2 的数值差异。

### 1.4 最终目标

最终目标不是一次性生成无限网格, 而是构建可长期后台补点的证据仓:

| 类别 | 目标行数 | 说明 |
|---|---:|---|
| existing seed | 63 | `dataset_v2.csv` 中 `ap_valid=True` 的历史真实点 |
| validation replay | 12 | 4 个 dataset_v2 同配置 x latency/AP/energy, 生成对比结果 |
| H800/productized smoke | 9 | 3 个 config x latency/AP/energy |
| calibration | 106 | H800 dense-core/backbone 过渡校准: latency 60 + AP 26 + energy 20 |
| paper stage | 66 | 最终报告候选: 20 config x 3 axes + 6 DS validation rows |
| **合计目标** | **256** | 63 existing + 193 new/validation canonical evidence rows |

`256` 是行级证据目标, 不等于 raw repeat 数。latency/energy 的 raw repeat 保存在 raw artifacts 中, canonical row 写聚合统计。

---

## 章节 2: 已有数据基础

### 2.1 新数据目录中已经落地的 dataset_v2 seed

目录:

```text
/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/
```

已经生成的 seed 文件:

```text
existing/dataset_v2_full_66.csv
existing/dataset_v2_ap_valid_63.csv
existing/dataset_v2_ap_valid_63.jsonl
existing/dataset_v2_front_32.csv
existing/dataset_v2_summary.json
README.md
```

快速审查:

```bash
cd /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1
tail -n +2 existing/dataset_v2_ap_valid_63.csv | wc -l
python -m json.tool existing/dataset_v2_summary.json | sed -n '1,120p'
find . -maxdepth 2 -type f | sort
```

期望:

```text
63
full_rows = 66
ap_valid_rows = 63
front_rows = 32
duplicate_config_id = {"pyr_16-32-64_III": 2}
```

### 2.2 dataset_v2 分组

| dataset_src | n | 作用 |
|---|---:|---|
| `complete_points_v1` | 7 | 4090 Pyramid 主前沿完整点 |
| `perstage_AP_v2` | 22 | mixed precision 消融, 多数不作为主前沿默认候选 |
| `E4_energy_v1` | 13 | 4090 board energy / batch1-2 能耗锚 |
| `E6_orin_p03` | 6 | Orin body latency/energy 跨平台点 |
| `P0_1_p25_trap` | 2 | p25 trap / forced INT8 对照 |
| `P0_2_136` | 2 | p50b2_136 宽度对照 |
| `pathA_forced_int8` | 3 | 极端剪枝 forced INT8 guardrail |
| `pathB_head_int8` | 2 | head-only INT8 guardrail |
| `P12_collab2_throughput` | 3 | request 2/4/8 吞吐饱和 |
| `E3_orin_dla_pipe_v1` | 2 | Orin DLA pipeline e2e |
| `v2xvit_A1A2_ap` | 3 | V2X-ViT AP reference |
| `v2xvit_dair_real_timing` | 1 | V2X-ViT PyTorch timing |

### 2.3 最适合作为验证基线的 4 个同配置点

同配置 replay validation 第一批只选 4 个点, 覆盖 base/p50 与 FP16/INT8:

| config_id | planes | precision | latency_kind | lat ms | AP50 | AP70 | energy mJ |
|---|---|---|---|---:|---:|---:|---:|
| `pyr_64-128-256_FP16` | 64/128/256 | FP16/FP16/FP16 | body_subnet_collab2 | 1.2715 | 0.791041 | 0.630864 | 462.4518 |
| `pyr_64-128-256_INT8` | 64/128/256 | INT8/INT8/INT8 | body_subnet_collab2 | 0.8113 | 0.790498 | 0.622809 | 261.0653 |
| `pyr_32-64-128_FP16` | 32/64/128 | FP16/FP16/FP16 | body_subnet_collab2 | 1.0138 | 0.764428 | 0.564132 | 433.5189 |
| `pyr_32-64-128_INT8` | 32/64/128 | INT8/INT8/INT8 | body_subnet_collab2 | 0.7956 | 0.752232 | 0.554228 | 287.0130 |

对比容差:

| 指标 | 容差 | 说明 |
|---|---:|---|
| latency p50 | <= 5% 或 <= 0.05 ms, 取较大值 | 4090 TRT replay 必须严格; H800 不按数值比较 |
| AP50/AP70 | <= 0.003 absolute | 同 ckpt/split/protocol replay |
| energy | <= 8% | power telemetry 对环境敏感, 但必须同 idle-exclusive policy |
| engine size | <= 1% | 同 engine artifact 应近似固定 |

### 2.4 零星测试的关系

零星测试不作为新的事实源覆盖 dataset_v2, 只作为辅助 evidence:

- H800 smoke: `results/stage2/pyramid_lidar/evidence_real_smoke_20260625_154907/`
- 旧 H800 latency LUT: `results/latency_lut_pyramid.json`
- 4090 Q/backbone LUT: `results/latency_lut_pyramid_q.json`
- E5 energy verification: `results/E5_energy_verification.md`

---

## 章节 3: 数据生成计划

### 3.1 验证阶段: historical replay vs current generation

目的: 在扩展新 H800 LUT 前, 先证明“目前生成方案”和 `dataset_v2` 的历史同配置数据可对齐。

输入:

```text
existing/dataset_v2_ap_valid_63.csv
```

输出:

```text
validation/replay_job_plan_v1.jsonl
validation/replay_rows_v1.jsonl
validation/replay_compare_v1.csv
validation/replay_summary_v1.json
```

配置:

| config_id | latency | AP | energy |
|---|---|---|---|
| `pyr_64-128-256_FP16` | 4090 TRT collab2 replay | DAIR val 1789 replay or exact AP import | E5/NVML replay or exact energy import |
| `pyr_64-128-256_INT8` | 4090 TRT collab2 replay | DAIR val 1789 replay or exact AP import | E5/NVML replay or exact energy import |
| `pyr_32-64-128_FP16` | 4090 TRT collab2 replay | DAIR val 1789 replay or exact AP import | E5/NVML replay or exact energy import |
| `pyr_32-64-128_INT8` | 4090 TRT collab2 replay | DAIR val 1789 replay or exact AP import | E5/NVML replay or exact energy import |

判定:

- 4 个 config 都有 latency/AP/energy 对比行。
- 每个指标都给出 `old_value`, `new_value`, `abs_error`, `rel_error_pct`, `pass_fail`, `reason`。
- 若 replay 命令不可用, 先写 `comparison_mode=canonical_import_check`, 并禁止进入 calibration。

执行步骤:

- [ ] 读取 `existing/dataset_v2_ap_valid_63.csv`, 过滤上述 4 个 config。
- [ ] 生成 `validation/replay_job_plan_v1.jsonl`, 每个 config 三个 job。
- [ ] 运行 replay worker, 只占用空闲 GPU。
- [ ] 写 `validation/replay_compare_v1.csv`。
- [ ] 生成 `validation/replay_summary_v1.json`。
- [ ] 只有 `summary.status=pass` 才允许进入 smoke/calibration。

### 3.2 Smoke 阶段

目标: 验证产品化 `generate_*_lut` 在 H800/目标环境上能产出三类真实 row。

配置分布:

| config | width | precision | schedule | 目的 |
|---|---|---|---|---|
| smoke_base_fp16 | 64/128/256 | FP16 | default | base 可建性和 AP/energy 最小链路 |
| smoke_p50_fp16 | 32/64/128 | FP16 | metaschedule_tuned | 复用 dataset_v2 p50 语义, 检查 tuned route |
| smoke_p75_int8_or_failed | 16/32/64 | INT8 | default | 验证 INT8 成功或 failed row |

目标行数:

```text
3 config x 3 axes = 9 canonical rows
```

Smoke 通过条件:

- latency/AP/energy 三类 row schema 均 validate。
- GPU preflight raw artifact 存在。
- AP/energy 不写 placeholder row。
- INT8 若失败, 必须有 failed row 和 `failure_reason`。
- registry coverage summary 能读到新增 row。

### 3.3 Calibration 阶段

Calibration 的配置分布必须从已有数据基础延伸, 不是重新发明网格。

#### 3.3.1 width 分布

已有数据锚点:

```text
64/128/256  base
48/96/192   p25/trap
32/64/128   p50
16/32/64    p75
32/64/136   p50b2_136
```

新增 calibration width:

| 组 | widths | 目的 |
|---|---|---|
| base/prune anchors | 64/128/256, 48/96/192, 32/64/128, 16/32/64, 32/64/136 | 对齐 dataset_v2 |
| single-axis s0 | 16/128/256, 32/128/256, 48/128/256 | stage0 sensitivity |
| single-axis s1 | 64/32/256, 64/64/256, 64/96/256 | stage1 sensitivity |
| single-axis s2 | 64/128/64, 64/128/128, 64/128/192 | stage2 sensitivity |
| coupled/cliff | 64/96/192, 48/64/256, 48/128/128, 32/96/192, 16/128/128, 64/64/128 | Q/P/S coupling |

#### 3.3.2 行数目标

| evidence | 分布 | 行数 |
|---|---|---:|
| latency | 18 FP16 widths x 2 schedules | 36 |
| latency | 12 INT8/buildable widths x 2 schedules | 24 |
| AP | 18 FP16 + 8 INT8 selected anchors | 26 |
| energy | 12 FP16 + 8 INT8 latency-aligned subset | 20 |
| **合计** |  | **106** |

Calibration 通过条件:

- coverage summary 中 measured latency >= 50, AP >= 20, energy >= 16。
- 失败 row 占比 <= 20%, 且失败原因分布可解释。
- 至少包含 base/p25/p50/p75/p50b2_136 的 dataset_v2 对齐配置。

### 3.4 Paper 阶段

Paper 阶段只补最终报告需要的点, 不盲目扩网格。

配置分布:

| 组 | config 数 | 用途 |
|---|---:|---|
| Pareto/front candidates | 8 | 最终多目标前沿 |
| cliff / rank-flip evidence | 6 | 证明硬件耦合机制 |
| cross-hardware comparable anchors | 3 | 4090/Orin/H800 口径分报 |
| DS top-K validation candidates | 3 | downstream report-only 或 closed-loop validation |

行数目标:

```text
20 config x latency/AP/energy = 60 rows
DS validation = 6 rows
paper total = 66 rows
```

Paper 通过条件:

- 每个 final claim 都能追到 canonical row 和 raw artifact。
- 4090/Orin/H800 分别成表, 不混绝对 latency/energy。
- H800 energy claim 只有在 H800 telemetry measured rows 存在时才开启。
- DS claim 只有 closed-loop validation pass 后才写入正文; 否则只 report-only。

---

## 章节 4: 最终存储目录和快速审查

### 4.1 目录

最终目录:

```text
/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/
```

目录格式:

```text
stage2_lut_generation_v1/
  README.md
  existing/
    dataset_v2_full_66.csv
    dataset_v2_ap_valid_63.csv
    dataset_v2_ap_valid_63.jsonl
    dataset_v2_front_32.csv
    dataset_v2_summary.json
  validation/
    replay_job_plan_v1.jsonl
    replay_job_state_v1.jsonl
    replay_rows_v1.jsonl
    replay_compare_v1.csv
    replay_summary_v1.json
  generated/
    smoke/
      jobs/
      latency/
      ap/
      energy/
      raw/
      logs/
    calibration/
      jobs/
      latency/
      ap/
      energy/
      raw/
      logs/
    paper/
      jobs/
      latency/
      ap/
      energy/
      ds/
      raw/
      logs/
  registry/
    evidence_registry_v1.json
    coverage_summary_v1.json
    claim_gate_summary_v1.json
  raw/
    external_or_large_artifacts/
  logs/
```

### 4.2 文件作用

| 文件/目录 | 作用 |
|---|---|
| `existing/dataset_v2_full_66.csv` | 原始 dataset_v2 完整副本 |
| `existing/dataset_v2_ap_valid_63.csv` | 当前已有 63 个 AP-valid 实测点, 作为 seed baseline |
| `existing/dataset_v2_ap_valid_63.jsonl` | 63 点 JSONL, 便于 registry/import 脚本读取 |
| `existing/dataset_v2_front_32.csv` | `regime=front` 快速审查表 |
| `existing/dataset_v2_summary.json` | 行数、分组、重复 ID、invalid AP 摘要 |
| `validation/replay_compare_v1.csv` | 历史点 vs 当前生成方案的对比结果 |
| `generated/*/latency/latency_lut_rows_v1.jsonl` | 新 latency canonical rows |
| `generated/*/ap/ap_anchor_rows_v1.jsonl` | 新 AP canonical rows |
| `generated/*/energy/energy_lut_rows_v1.jsonl` | 新 energy canonical rows |
| `generated/*/jobs/*.jsonl` | job plan 与 job state |
| `generated/*/raw/` | bench/eval/telemetry 原始产物 |
| `registry/coverage_summary_v1.json` | 每阶段 coverage 和 claim gate 汇总 |

### 4.3 快速审查命令

确认 63 个已有点已经落地:

```bash
cd /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1
tail -n +2 existing/dataset_v2_ap_valid_63.csv | wc -l
```

期望:

```text
63
```

查看分组统计:

```bash
python -m json.tool existing/dataset_v2_summary.json | sed -n '1,160p'
```

检查新生成行数:

```bash
find generated -name '*_rows_v1.jsonl' -print -exec wc -l {} \;
```

检查 job 是否失败:

```bash
find generated -name '*job_state*.jsonl' -print -exec sh -c 'grep -H "\"status\": \"failed\"" "$1" || true' sh {} \;
```

检查 claim gate:

```bash
python -m json.tool registry/claim_gate_summary_v1.json | sed -n '1,160p'
```

### 4.4 文档位置

计划文件:

```text
/home/jichengzhi/V2X/multi_agent/methods/design/auto-tuning/progress/PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md
```

交接文件:

```text
/home/jichengzhi/V2X/multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_dataset_v2_aligned_lut_generation_v1_zh.md
```
