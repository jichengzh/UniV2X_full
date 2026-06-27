# Stage2 Dataset-v2 Aligned LUT Generation Handoff v1

更新时间: 2026-06-25

关联计划:

```text
/home/jichengzhi/V2X/multi_agent/methods/design/auto-tuning/progress/PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md
```

数据根目录:

```text
/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1
```

---

## 1. 实验目前进展

### 1.1 新计划已建立

已撰写新的 dataset_v2 对齐版 LUT 生成计划, 用来替代旧的 `HANDOFF_stage2_dataset_v2_inventory_v1_zh.md` 作为后续执行依据。

新计划明确了四个核心原则:

1. `dataset_v2.csv` 是 historical measured seed, 不原地改写。
2. 新增 latency/AP/energy 数据的主路径必须是 `generate_latency_lut` / `generate_ap_lut` / `generate_energy_lut`。
3. 先做同配置 validation replay, 再进入 smoke、calibration、paper 三阶段。
4. H800 dense-core / backbone-only / 4090 TRT / Orin DLA 等 scope 不混写、不互相伪装。

### 1.2 dataset_v2 已整理为新数据目录 seed

`multi_agent/data/dataset_v2.csv` 已复制并整理到:

```text
/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/existing
```

当前文件:

| 文件 | 作用 |
|---|---|
| `dataset_v2_full_66.csv` | dataset_v2 全量 66 行历史表复制 |
| `dataset_v2_ap_valid_63.csv` | `ap_valid=True` 的 63 行, 作为当前主要 historical seed |
| `dataset_v2_ap_valid_63.jsonl` | 63 行 JSONL 版本, 便于 worker/registry 后续读取 |
| `dataset_v2_front_32.csv` | 32 行 front regime 子集 |
| `dataset_v2_summary.json` | 数据分布摘要 |
| `README.md` | 数据目录说明和快速审查命令 |

摘要:

| 指标 | 当前值 |
|---|---:|
| full rows | 66 |
| AP-valid rows | 63 |
| front rows | 32 |
| columns | 93 |
| rows with latency | 63 |
| rows with energy | 60 |
| rows with engine size | 51 |
| `pyramid_fusion` rows | 62 |
| `v2x_vit` rows | 4 |
| RTX4090 rows | 58 |
| Orin AGX rows | 8 |

需要特别注意:

| 问题 | 当前状态 |
|---|---|
| duplicate config id | `pyr_16-32-64_III` 出现 2 次 |
| invalid AP rows | `pyr_64-128-256_INT8_orin_E6`, `pyr_32-64-128_INT8_orin_E6`, `pyr_16-32-64_INT8_orin_E6` |
| mixed latency kind | `body_subnet_collab2`, `engine_board_energy`, `body_subnet_collab2_orin`, `dla_pipeline_e2e_single_frame`, `forward_hook_pytorch_fp32`, `NA_pending_trt_build` 并存 |

### 1.3 H800 latency 产品化真实 smoke 已跑通过一次

已完成一条 H800 真实 latency smoke:

```text
/home/jichengzhi/V2X/results/stage2/pyramid_lidar/evidence_real_smoke_20260625_154907
```

关键事实:

| 字段 | 值 |
|---|---|
| backend | `h800_tvm` |
| measurement status | `measured` |
| optimized scope | `backbone_only` |
| latency p50 | `56129.53 us` |
| target host | `zs-nj-tap-gpu18` |

H800 TVM 环境需要显式设置:

```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}
```

当前 H800 smoke 只证明 `backbone_only` latency path 可运行, 还不能代表 `rsu_dense_core` 或 full-model latency。

### 1.4 产品化代码基础已具备

当前已经具备以下 Stage2 LUT 产品化基础:

| 能力 | 当前状态 |
|---|---|
| schema validator | 已有 |
| job plan generator | 已有 |
| latency generator | 已有最小主路径 |
| AP generator | 已有最小主路径, 但真实目标 AP eval 仍需验证 |
| energy generator | 已有 row schema 与 no-claim/claim validator, 但真实 H800 telemetry 仍需验证 |
| background worker | 已有 |
| registry update script | 已有 |
| H800 B1 latency adapter | 已加入真实 smoke 路径 |

最近一次本地验证曾通过:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest \
  framework.tests.test_stage2_h800_latency_adapter \
  framework.tests.test_stage2_lut_productization \
  framework.tests.test_stage2_evidence_registry \
  framework.tests.test_stage2_integration_contract \
  framework.tests.test_stage2_search_space_contract
```

结果为 34 个测试通过。

---

## 2. 当前阻塞项

### 2.1 不能直接进入完整大规模生成的阻塞

| 阻塞项 | 影响 | 解除条件 |
|---|---|---|
| validation replay 尚未执行 | 无法证明当前生成链路与 dataset_v2 历史口径一致 | 4 个同配置 x latency/AP/energy 对比完成并通过容差 |
| H800 AP generator 真实 eval 未完成 | 不能生成可 claim 的 AP LUT | AP row 输出 dataset/split/ckpt/protocol, schema validate 通过 |
| H800 energy telemetry 未完成 | 不能生成可 claim 的 H800 energy LUT | energy row 输出 telemetry source/idle baseline/raw artifact, claim validator 通过 |
| H800 当前只有 backbone-only latency smoke | 不能直接外推到 RSU dense-core | dense-core scope 命令和 row schema 明确, 或明确继续以 backbone-only 作为第一阶段目标 |
| GPU idle 约束需要自动化 preflight | latency/energy 数值可能受其他任务污染 | 每个 measured row 绑定 `nvidia-smi` 和 `pmon` raw artifact, 忙碌则写 failed/blocked |
| dataset_v2 存在重复和口径混杂 | registry ingest 可能出现重复 claim 或错误比较 | duplicate config 处理, latency_kind/scope/backend guard 生效 |

### 2.2 数据质量待审查项

1. `pyr_16-32-64_III` 在 dataset_v2 中重复出现, registry 导入前必须区分 trial/revision 或选定 canonical row。
2. Orin INT8 三行 `ap_valid=False`, 只能保留 latency/energy 事实, 不能参与 AP claim。
3. dataset_v2 包含多种 latency_kind, 后续前沿/DS/registry 比较必须按 scope 分表处理。
4. 若发现 p75 INT8 行复用了 FP16 engine_path 或 energy artifact, 必须在 validation 阶段写入 `audit_flag`, 不能静默合并。

### 2.3 当前需要用户或环境确认的事项

1. H800 上可长期占用的 GPU id 和空闲时间窗口。
2. 第一阶段是否正式把 `backbone_only` 作为 H800 LUT 主 scope, 还是等待 `rsu_dense_core` measurement command 完成后再扩展。
3. AP eval 使用的 checkpoint、dataset split、eval script 是否固定为 dataset_v2 历史协议, 或需要建立 H800 新协议。
4. energy telemetry 采用 NVML、板级功耗还是外部采样, 以及 idle baseline 采样时长。

---

## 3. 大规模启动判定

### 3.1 当前判定

当前结论:

```text
GO: dataset_v2 seed 审查与 validation replay 准备
GO: latency-only H800 backbone-only 小规模 smoke/calibration 扩展
NO-GO: latency/AP/energy 三轴完整大规模 calibration/paper 生成
```

原因:

1. 63 个 AP-valid historical seed 已经整理到新数据目录, 可以作为验证和 registry seed。
2. H800 latency 真实 smoke 已经跑通, 但 scope 仍是 `backbone_only`。
3. AP 与 energy 的真实产品化链路尚未完成 smoke。
4. validation replay 尚未完成, 不能证明当前生成链路和 dataset_v2 历史链路一致。

### 3.2 大规模启动硬门槛

只有同时满足以下条件, 才建议启动长期后台 LUT 补点:

| Gate | 判定条件 |
|---|---|
| G1 seed gate | `existing/dataset_v2_ap_valid_63.csv` 行数确认为 63 |
| G2 replay gate | `pyr_64-128-256_FP16`, `pyr_64-128-256_INT8`, `pyr_32-64-128_FP16`, `pyr_32-64-128_INT8` 的 latency/AP/energy replay 对比完成 |
| G3 smoke gate | H800 smoke 3 个 config x latency/AP/energy 生成真实 measured row 或明确 failed row |
| G4 AP gate | AP row 记录 dataset/split/ckpt/protocol, AP validator 通过 |
| G5 energy gate | energy row 记录 telemetry source、idle baseline、raw artifact, no-claim/claim validator 通过 |
| G6 idle gate | latency/energy 每个 job 启动前记录 GPU idle preflight, 忙碌 GPU 不写 measured |
| G7 registry gate | registry coverage/claim gate summary 通过, no-claim 状态明确 |
| G8 duplicate gate | `pyr_16-32-64_III` duplicate config id 已处理 |

### 3.3 建议下一步执行顺序

1. 先审查 `existing/dataset_v2_ap_valid_63.csv` 和 `dataset_v2_summary.json`, 确认 63 点 seed 满足预期。
2. 生成并运行 validation replay job plan, 第一批只跑 4 个同配置。
3. 如果 replay 通过, 启动 H800 smoke: 3 个 config x latency/AP/energy。
4. 如果 AP/energy smoke 仍未 ready, 只允许启动 latency-only 小规模 calibration, 并在 registry 中保持 AP/energy no-claim。
5. smoke 三轴通过后, 再启动 calibration 阶段的长期后台 worker。
6. calibration 达标后, 再进入 paper stage 的 20 config x 3 axes 与 6 个 DS validation row。

### 3.4 快速审查命令

```bash
cd /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1

tail -n +2 existing/dataset_v2_ap_valid_63.csv | wc -l
python -m json.tool existing/dataset_v2_summary.json | sed -n '1,120p'
find . -maxdepth 2 -type f | sort
```

期望最小结果:

```text
63
existing/dataset_v2_ap_valid_63.csv
existing/dataset_v2_ap_valid_63.jsonl
existing/dataset_v2_front_32.csv
existing/dataset_v2_full_66.csv
existing/dataset_v2_summary.json
README.md
```

