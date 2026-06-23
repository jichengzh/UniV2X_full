# 小规模高信息量 DoE 设计 v1.0

> 角色: 数据生成师交付物 2
> 代码: `tools/configurable/doe_anchors.py` (anchor 矩阵) + `tools/configurable/generate_dataset.py` (生成管线)
> 配套: 搜索空间规模 `search_space_size_v1.md`
> 生成日期: 2026-05-31

---

## 1. 设计目标

用**最少 anchor** 拿到训练 latency/AP 预测器所需的**最高信息量**, 解决已知瓶颈:
- latency 预测器卡在 R²~0.73 且**跨数量级崩塌** (MEMORY)。
- INT8 真 build 覆盖率只有 10.8% (硬件师 Stage 1 发现), 边角样本稀缺。

**不要纯随机**: searcher 纯随机在 Pyramid 单模块上 pass_rate 仅 0.05% [实测], 且过采样中间配置、欠采样边界 (selection bias)。

---

## 2. 采样策略 (4 类, 共 25 anchor)

| 策略 | anchor 数 (4090/Orin) | 作用 | 信息论依据 |
|------|----------------------|------|-----------|
| **基线锚 baseline** | 3 / 3 | FP32/FP16/INT8 不剪枝, 校准 latency/AP/size 绝对参照 | 没有它所有"加速倍率"无锚; 锚定响应面原点 |
| **敏感度分层 sens (OAT)** | 6 / 6 | 沿单轴扫 (剪枝率×3 + 校准器 + 粒度 + W-only), 其余锚 baseline | OAT 主效应: 最少点拿每轴边际效应, 喂 LGB 主效应项 |
| **Pareto 边界加密 pareto_edge** | 3 / 3 | 拐点交互锚 (p25/p50/p75 × INT8) | LGB 跨数量级崩塌最缺的剪枝×量化交互项样本 |
| **跨硬件镜像 cross_hw** | 0 / 1 | Orin DLA0 路由 (4090 无此轴) | latency 跨硬件外推配对样本 (带宽 5× 差距) |

> 总计 **25 anchor** = 12 (4090) + 13 (Orin)。基线/sens/pareto 在两硬件镜像复制 (跨硬件配对), Orin 额外 +1 DLA 锚。

### 2.1 为什么是"最少 anchor" (验收要点)

- **OAT 主效应** (6 轴) + **拐点交互** (3 点) 覆盖了 GBDT 最需要的低阶项 (主效应 + 二阶交互)。全笛卡尔积 4.5 万点 (Pyramid 4090) 里 99% 是内部插值点——LGB 能内插, 不需真测。
- **四角覆盖**: baseline(FP32) ↔ 最激进 (p75) ↔ 最低精度 (INT8) ↔ 交互角 (p75+INT8) 锚定响应面四角, 防止预测器边界外推无约束发散。
- **避免 selection bias**: 分层强制覆盖边界, 纠正纯随机过采样中间区的偏差。
- **每模型轴最少 anchor 理由**: 剪枝率轴需 ≥3 点 (拟合 params↔latency 的非线性)；量化轴每个旋钮 ≥1 OAT 点 (主效应)；交互需 ≥3 拐点 (剪枝×量化的二阶项)。低于此预测器主效应缺项。

---

## 3. anchor 清单 (每 anchor 的 Config + 要测标签)

完整清单见 `python tools/configurable/doe_anchors.py` 输出。核心 12 个 (4090, Orin 镜像同构):

| anchor_id | strategy | triplet | prune | bits | gran | obj | calib | route |
|-----------|----------|---------|-------|------|------|-----|-------|-------|
| base_fp32 | baseline | T1_base | 0.00 | FP32 | none | none | none | GPU |
| base_fp16 | baseline | T1_base | 0.00 | FP16 | per-tensor | W+A | none | GPU |
| base_int8 | baseline | T1_base | 0.00 | INT8 | per-channel | W+A | minmax | GPU |
| sens_prune_25/50/75 | sens_prune | T2/T4/T6 | 0.25/.50/.75 | FP16 | per-tensor | W+A | none | GPU |
| sens_calib_pct | sens_calib | T1_base | 0.00 | INT8 | per-channel | W+A | **percentile_99_99** | GPU |
| sens_gran_pt | sens_gran | T1_base | 0.00 | INT8 | **per-tensor** | W+A | minmax | GPU |
| sens_wonly | sens_wonly | T1_base | 0.00 | INT8 | per-channel | **W-only** | minmax | GPU |
| pareto_p25/p50/p75_int8 | pareto_edge | T2/T4/T6 | .25/.50/.75 | INT8 | per-channel | W+A | minmax | GPU |
| (Orin) cross_dla_fp16 | cross_hw | T1_base | 0.00 | FP16 | per-tensor | W+A | none | **DLA0** |

### 3.1 每 anchor 要测的标签 (对齐 e2e_bench_v1_schema.md)

| 标签类 | 列 | 数据源 | 本管线状态 |
|--------|----|--------|-----------|
| **latency** | `throughput_fps` (+ bench JSON 三段 `lat_trt/postproc/e2e`) | e2e bench (TRT + PyTorch wrapper) | ⏸ 需空闲 GPU + e2e timing (本管线 build_engine 不含 timing) |
| **AP** | `ap30/ap50/ap70` | DAIR val sweep (INT8 走 e2e engine 实测) | ⏸ 需 DAIR eval |
| **资源 (engine)** | `engine_size_mb` / `build_secs` / `int8_layer_pct` | `deploy_config.build_engine` report | ✅ 管线接通, GPU 空闲时真测 |
| **资源 (剪枝)** | `params_total_new` / `num_filters_new` / reduction% | `prune_config.execute_pyramid` manifest | ✅ **已真测** (CPU rebuild, 见 §5) |

---

## 4. 生成管线 (`generate_dataset.py`)

每 anchor Config 依次喂三工具, 收集标签, 写统一 schema (33 列 + 12 追踪列):

```
Config ─► prune_config.resolve_prune_plan + execute_pyramid   (B: 真实小模型, params/通道数)
       ─► quant_config.resolve_quant_plan                     (Q: effective_bits/calibrator/manifest)
       ─► deploy_config.build_plan + build_engine             (D: engine_size/build_secs/int8%)
```

每行标 `is_real_measured` (latency/AP/size 是否真测) + `source` (每标签来源: prune:real_rebuild / quant:plan / deploy:real_trt_build / deploy:dry_run / ...)。

运行模式:
- `--mode dry_run`: 只走 plan 决议 (无 GPU 可跑, 接通管线)。
- `--mode prune_real`: 真 rebuild 剪枝小模型 (CPU 可跑)。
- `--mode full`: + TRT build (需空闲 GPU)。

---

## 5. 真生成结果 (本次)

| 模式 | 跑通 | 真测标签 |
|------|------|---------|
| dry_run (25 anchor, 两硬件) | ✅ 全 25 行管线接通, 三工具 plan/manifest 全产出 | — (无真测, plan only) |
| prune_real (12 anchor, 4090) | ✅ 9 个剪枝 anchor 真 rebuild 小模型 (CPU) | `params_total_new` / `num_filters_new` 真实 |
| full (TRT build) | ❌ GPU 占用 (3×4090 均 76-83% util, 20-23GB used) → CUDA init error 2 (OOM) | deploy 标 dry_run |

**真测 params 数据点** (prune_real, 已写入 `output/doe_dataset_v1/doe_dataset_v1.csv`):
| anchor | prune_rate | round_to | num_filters_new | params_new | reduce% |
|--------|-----------|----------|-----------------|-----------|---------|
| pareto_p50_int8 | 0.50 | 32 | [32,64,128] | 2,808,311 | 48.6% |
| pareto_p75_int8 | 0.75 | 32 | [32,32,64] | 2,080,535 | 61.9% |
| sens_prune_50 (FP16) | 0.50 | 8 | [32,64,128] | 2,808,311 | 48.6% |
| sens_prune_75 (FP16) | 0.75 | 8 | [16,32,64]* | 2,063,079 | 62.3% |

> *FP16 round_to=8 与 INT8 round_to=32 在同剪率下通道数/params 不同 — 印证 D4 round_to 约束真实生效 (INT8 对齐更粗 → params 略高)。

---

## 6. 与三工具的接口调用情况

| 工具 | 接口 | 状态 |
|------|------|------|
| `prune_config` | `resolve_prune_plan` / `resolve_pyramid_num_filters` / `execute_pyramid` | ✅ 全跑通 (含真 CPU rebuild) |
| `quant_config` | `resolve_quant_plan` → manifest (effective_bits/calibrator/mixed_substr) | ✅ 全跑通 (entropy 自动回退 minmax, W-only 全局 flag warn 都正常触发) |
| `deploy_config` | `build_plan` (推导) ✅ / `build_engine` (真 build) | ✅ build_plan 全跑通; ❌ build_engine 受 GPU 占用阻塞 (CUDA OOM) |

---

## 7. 未完成项

1. **TRT 真 build (engine_size/int8%/build_secs)**: GPU 空闲后跑 `--mode full --hardware rtx4090`。管线已接通, 单 anchor 失败会标 `deploy:build_failed` 不污染其他行 (但 CUDA init OOM 是 C++ terminate, 当前会整进程崩——建议等 GPU 空闲或限定单卡 `CUDA_VISIBLE_DEVICES`)。
2. **latency (throughput_fps) + AP (ap30/50/70)**: 本管线只产 build 期标签; latency/AP 需复用 `scripts/phase2/e2e_bench_v1_orchestrator.py` (e2e timing) + DAIR eval。建议下一步把 generate_dataset 的真 build engine 喂给 orchestrator 拿 latency。
3. **Orin 真测**: Orin 无本地 TRT, DLA 锚点需在 Orin host (172.16.62.222) build。当前 Orin 行全 dry_run。
4. **UniV2X anchor**: 本 DoE 聚焦 Pyramid; UniV2X 10^15 空间留代理模型阶段。
