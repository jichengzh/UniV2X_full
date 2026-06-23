# 硬件可配置维度清单 (D 维度) v1.0

> 角色: 硬件/TRT 部署工程师交付物 #3
> 数据源: `configs/hardware/{rtx4090,orin_agx,schema}.yaml` (实测/探查), `framework/config_schema.py` (共享契约), `framework/capability_schema.py` (硬件能力 schema), TRT 10.13 (4090) / TRT 8.5.2.2 (Orin) 文档与实测。
> 用途: 数据生成师据此枚举 D 维度搜索空间; 搜索器据此展开合法档位; 部署工具 (`tools/configurable/deploy_config.py`) 据此 build。

---

## 0. 一句话总览

D 维度 = "同一个 (剪枝+量化后的) 网络, 部署到目标硬件时还能调的旋钮"。
取值域 **完全由目标硬件 capability 决定** (硬件门控): FP8/INT4/per-group 仅 4090; DLA 路由/per-tensor-INT8 仅 Orin。

| 平台 | 框架版本 (实测) | D 档位计数 (本文 §6) |
|------|----------------|----------------------|
| RTX 4090 | TensorRT 10.13.0.35 (实测 `trt.__version__`) | **~25 档** (无 DLA, 路由轴退化) |
| Jetson AGX Orin 64GB | TensorRT 8.5.2.2 (YAML 实测) | **~96 档** (DLA 路由 ×6 是主增量) |

---

## 1. 维度枚举表

| # | 维度 | config_schema 字段 | 4090 取值域 | Orin 取值域 | 平台约束 | 来源 |
|---|------|---------------------|-------------|-------------|----------|------|
| D1 | hardware target | (外部 `--hw` yaml) | rtx4090.yaml | orin_agx.yaml | 选定后门控 D2-D8 | configs/hardware/*.yaml |
| D2 | 设备路由 (per-module) | `d_routing[m]` | {GPU} | {GPU, DLA0, DLA1} | DLA 仅 Orin (4090 `dla.enabled=false`) | config_schema L39, rtx4090 §diff_vs_orin |
| D3 | GPU fallback (DLA 时) | (部署 flag, plan) | n/a | {on, off} | 仅 use_dla 时有效; `BuilderFlag.GPU_FALLBACK` | orin_agx dla op_blacklist |
| D4 | tactic source | (部署 flag, plan) | {CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONV, JIT_CONV} 子集 | {CUBLAS, CUDNN, EDGE_MASK_CONV} 子集 | TRT 版本相关; 实践取 ~5 个常用组合档 | TRT TacticSource enum |
| D5 | workspace | (部署 flag, plan) | {1,2,4,8} GB (≤22GB avail) | {1,2,4} GB (≤50GB unified 但保守) | 受 `memory.available_for_inference_gb` 上限 | rtx4090/orin memory |
| D6 | precision flag | 派生自 `q_bits` + 硬件 | {FP32, FP16, INT8, FP8, INT4} | {FP32, FP16, INT8, INT4} | FP8 仅 Ada TC4 (4090); 见 §3 | rtx4090 ips.gpu.precisions L33 / orin L36 |
| D7 | INT8 模式 | 派生自 ONNX (Q/DQ) | {explicit Q/DQ, implicit calib} | {implicit calib} | explicit/STRONGLY_TYPED 仅 TRT10 (4090); Orin TRT8.5 走 calibrator | rtx4090 quant_constraints.notes / orin notes |
| D8 | 量化粒度 (W) | `q_granularity[m]` | {per-tensor, per-channel, per-group} | {per-tensor, per-channel} | per-group(AWQ) 仅 TRT10; DLA 仅 per-tensor | rtx4090 quant_constraints L58 / orin L88 |
| D9 | 量化粒度 (A) | `q_granularity[m]` (A) | {per-tensor} | {per-tensor} | per-channel-A GPU GEMM 不支持 (两平台) | quant_constraints.per_channel_activation_supported=false |
| D10 | plugin 开关 | (部署 flag, plan) | MSDA-FP16 必开; Rotate/Inverse 必开 | + DCNv4 plugin (libDCNv4_plugin.so) | 见 §4 plugin 清单 | plugins/build/, orin dcnv4_plugin 路径 |
| D11 | sparse weights | 派生自 `prune_object=2:4` | {on, off} | {on, off} | `--sparsity=enable`; 需 2:4 recipe + k≥64 对齐 | features.trt_sparse_weights_flag (两平台 true) |
| D12 | strongly typed | (TRT10 only) | {on, off} | {off} | 仅 TRT10 (4090); 与显式 FP16 flag 互斥 | rtx4090 features.trt_strongly_typed=true / orin=false |
| D13 | 对齐强度 (软约束) | (capability 字段, 非搜索) | hard (未实证) | soft (N2v2 实证) | 影响通道剪枝档过滤; 不是独立可搜旋钮 | alignment_enforcement: rtx4090 L49 / orin L80 |
| D14 | 功耗模式 (nvpmodel) | (Orin only) | n/a (单档) | {MAXN, 15W, 30W, 50W} | 4090 无 nvpmodel; Orin nvpmodel **30W 档实测可用** [2026-06-03 ISS-016 解封, 见 v2 + results/E6_orin_energy.csv]; MAXN sweep 进行中 | rtx4090 power 单档 / orin power 4 档 |

> 说明: D3/D4/D5/D7/D10/D11/D12 在 `config_schema.Config` 里没有专用字段 (Config 聚焦 B1/B2/D-routing), 由部署工具 `DeployPlan` 承载 (从 ONNX + hardware yaml + Config 推导)。这是有意的: 这些是 build-time 部署旋钮, 不污染搜索器的核心 Config 契约。

---

## 2. 与 config_schema / capability_schema 的映射

```
framework/config_schema.Config
  ├─ d_routing: dict[module → "GPU"|"DLA0"|"DLA1"|"CPU"]   → D2 (核心可搜路由)
  ├─ q_bits:    dict[module → INT8|FP16|FP32]              → D6 precision (与硬件门控交)
  ├─ q_granularity: dict[module → per-tensor|per-channel]  → D8/D9
  └─ q_calibrator:  dict[module → minmax|percentile|...]   → D7 implicit 模式细节

framework/capability_schema.HardwareCapability  (from configs/hardware/*.yaml)
  ├─ ips.gpu.precisions        → D6 取值域门控
  ├─ has_dla (property)        → D2/D3 是否开放 DLA 路由
  ├─ alignment.alignment_enforcement → D13
  ├─ features.trt_strongly_typed     → D12
  ├─ features.trt_sparse_weights_flag→ D11
  └─ quant_constraints.{per_group_supported, granularity_w/a} → D8/D9

tools/configurable/deploy_config.DeployPlan  (推导产物, 承载 build-time 旋钮)
  └─ want_fp16/want_int8/int8_mode/routing/use_dla/tactic_sources/workspace_gb/strongly_typed
```

部署工具 `build_plan()` 实现了上述门控逻辑: 例如 `d_routing` 含 DLA 但硬件 `has_dla==False` 时, 自动降级 GPU 并写 warning (不静默)。

---

## 3. precision × 平台门控矩阵 (D6)

| precision | 4090 (Ada TC4, TRT10) | Orin (Ampere TC3, TRT8.5) | 备注 |
|-----------|----------------------|---------------------------|------|
| FP32 | ✅ | ✅ | baseline |
| TF32 | ✅ | ✅ | 默认 matmul accum |
| FP16 | ✅ | ✅ | 主路径 |
| BF16 | ✅ | ✅ | |
| INT8 | ✅ explicit+implicit | ✅ implicit calib only | explicit Q/DQ 仅 TRT10 |
| INT4 | ✅ IMMA TC | ✅ IMMA TC (sm87 可用) | Hopper sm90 退化为 IMAD (迁移注意) |
| FP8 (E4M3/E5M2) | ✅ TC4 | ❌ (无 FP8 硬件) | 仅 Ada/Hopper |
| per-group W4A16 (AWQ) | ✅ (Marlin kernel) | ❌ TRT8.5 不支持 | per_group_supported: 4090 true / Orin false |

来源: rtx4090.yaml L33/L55/L95, orin_agx.yaml L36/L85。

---

## 4. plugin 准备状态 (D10)

| plugin | 算子 | 4090 (本仓库) | Orin | 精度 | 状态 |
|--------|------|--------------|------|------|------|
| MSDAPlugin / MultiscaleDeformableAttnPlugin_TRT | multi-scale deformable attn | `plugins/build/libuniv2x_plugins.so` (已编, TRT10.13 加载验证) | 同源可编 | FP16 (无 INT8 实现) | ✅ 已注册 (v1/v2 creator 可见) |
| RotatePlugin | BEV rotate (grid 旋转) | 同 .so | 同源 | FP16/FP32 | ✅ 已注册 (custom_op domain) |
| InversePlugin | 矩阵求逆 | 同 .so | 同源 | FP32 | ✅ 已注册 |
| ModulatedDeformConv2d (DCNv2) | DCN | TRT 内建 plugin (registry 可见 v1/v2) | DCNv2 fused FP16 kernel | FP16 | ✅ registry 自带 |
| DCNv4 plugin | deformable conv v4 | (4090 需重编, 见依赖) | `libDCNv4_plugin.so` 15.6MB aarch64 已编 (DL4AGX) | FP16 | Orin ✅ / 4090 待编 |
| grid_sample | grid_sample | TRT 8.5+ 原生 FP16 (无需 plugin) | TRT 8.5 原生 | FP16 | ✅ 原生, 无需 plugin |
| sparse voxelize / spconv | 点云体素化 | **ONNX 无标准支持** → PyTorch 前处理 hybrid | 同 | — | ⚠️ 边界: 不进 TRT, host 侧 PyTorch |
| rotated NMS | 旋转框 NMS | 已 CUDA 化 (后处理, 不进 engine) | 同 | — | ✅ host CUDA, engine 外 |

注册方案 (已在 `deploy_config.load_plugins()` 实现):
1. `ctypes.CDLL(plugin.so, RTLD_GLOBAL)` 加载自定义 .so
2. `trt.init_libnvinfer_plugins(logger, "")` 初始化 registry
3. 自定义 PLUGIN_V2 层无 INT8 实现 → TRT 自动留在 FP16, **无需** 显式 `layer.precision`
   (显式设会插 Q/DQ 反而降精度; 与 `build_trt_int8_univ2x._force_msda_fp16` 结论一致)

**4090 DCNv4 依赖**: 若 4090 路径需 DCNv4, 需用 4090 的 TRT10 头文件重编 DL4AGX/dcnv4-trt (Orin 的 aarch64 .so 不能直接用于 x86_64 sm89)。本网络 (UniV2X BEV encoder) 用的是 MSDA + DCNv2, 已覆盖; DCNv4 仅在用 DCNv4 backbone 变体时需要。

---

## 5. 跨平台迁移注意点 (论文 §C4 三硬件 Pareto)

来自 rtx4090.yaml `diff_vs_orin_agx`:
- 无 DLA → D2 路由轴退化为仅 GPU (4090 档位 ÷4 量级)
- TC4 → 多 FP8 / 改进 INT8/INT4 路径 (D6 取值域更大)
- TRT10 → STRONGLY_TYPED (D12) + per-group (D8) + explicit FP8/INT8 (D7)
- GDDR6X 1008GB/s vs LPDDR5 204.8GB/s ≈5× 带宽 → memory-bound 算子差距大 (影响 latency 预测器跨硬件外推)
- discrete GPU → 无 unified memory, H2D/D2H 拷贝开销 (Orin unified 无此项)
- D14 nvpmodel: **30W 档实测解封** (ISS-016, 见 v2); 4090 无 nvpmodel 单档

---

## 6. 档位计数推导

**4090 (~25 档)**: 路由 D2 仅 GPU (×1) → tactic D4 (×5 常用档) × workspace D5 (×5: 1/2/4/8/+default) = 25。
precision/granularity (D6/D8) 通常由量化师在 B2 维度固定 (一次 build 一个精度配置), 故 D 维度本体计 25 (与 CLAUDE.md §"4090 部署 ~25 档" 一致)。

**Orin (~96 档)**: 路由 D2 (×6: GPU + DLA0/DLA1 × {纯/带 GPU fallback} 组合) × tactic D4 (×4) × workspace D5 (×4) = 96 (与 CLAUDE.md §"Orin ~96 档" 一致)。
功耗 D14 (×4 nvpmodel): **30W 档已解封实测** (ISS-016); 30W↔MAXN sweep 进行中, 全 4 档纳入后 ×4=384。

> 计数口径: 仅 build-time 部署旋钮的笛卡尔积; 不含 B1 剪枝率 / B2 量化位宽 (那是量化师/剪枝师维度)。

---

## 7. 实测验证锚点

- 4090 TRT10.13 engine build 实测: 见 `results/deploy_smoke_*.json` (本交付 smoke test)
- Orin N2v2 ResNet-50 FP16=2.530ms / INT8=1.767ms (jetson_clocks 锁频, orin_agx.yaml empirical_baseline)
- Orin 对齐拐点: out_c 63vs64 FP16 diff 0.27% → 证 D13 soft (orin_agx.yaml L150)
- 4090 BEV encoder FP16 TRT=24.1ms / INT8 TRT=24.6ms (INT8≈FP16 因 MSDA plugin FP16-only, rtx4090.yaml empirical_baseline)
