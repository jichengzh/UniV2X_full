# 搜索空间更新验证计划 v2 — Per-stage 量化 + D 空间维度

> 配套文档: `搜索空间一览.md`
> 目的: 验证两个"暂定"维度的真实 Pareto 价值,决定是否纳入完整搜索空间
> 状态: **规划阶段**(尚未启动)
> 创建日期: 2026-05-11

---

## 0. 背景与目标

`搜索空间一览.md` 当前纳入: B (per-stage prune + 2:4) + Q (全局位宽/粒度/目标) + D1 (IP 路由)。

两个"暂定"维度待验证:
- **Q1' per-stage 量化位宽** — 第三节"暂定"标记的 stage 独立 {FP16, INT8} 选择
- **D2+ 其他 D 空间维度** — L4 CUDA Graph / L1 GPU fallback / L3 TRT tactic 等

**核心验证原则**: 每个维度必须满足 **(a) 有可观测 Pareto 改善 + (b) 工程量可控**,两者缺一不纳入完整搜索空间。

---

## Part A: Per-stage 量化有效性验证

### A.1 假设

**H1 (主假设)**: Pyramid Fusion 三个 stage 的量化敏感度不同 — stage 2 (70% 参数) 量化对 AP 影响最大,stage 0/1 (10%/20% 参数) 影响较小。因此**让 stage 2 用 FP16、stage 0/1 用 INT8** 可能在保持 AP 的同时获得 INT8 的速度增益。

**H2 (反假设)**: per-stage 量化引入的 reformat layer (FP16↔INT8 边界) overhead 可能抵消速度增益,反而比全局 INT8 慢。

### A.2 验证设计

#### A.2.1 候选选取(3 个代表 triplet)

从已 bench 的 100 anchor 中挑 3 个跨剪枝率代表点:

| 标签 | (s0, s1, s2) | 剪枝率口径 | 已有数据 |
|------|--------------|----------|----------|
| **T_baseline** | (64, 128, 256) | 0% (baseline) | ✅ FP16/INT8 已测 |
| **T_prune50** | (32, 64, 128) | ~50% | ✅ FP16/INT8 已测 |
| **T_prune75** | (16, 32, 64) | ~75% | ✅ FP16/INT8 已测 |

#### A.2.2 Per-stage 量化组合(8 种 / 含全局 baseline)

每 stage ∈ {FP16, INT8},2³ = 8 种:

| # | s0 | s1 | s2 | 备注 |
|---|----|----|----|------|
| 1 | FP16 | FP16 | FP16 | 全 FP16(已测 baseline) |
| 2 | INT8 | INT8 | INT8 | 全 INT8(已测 baseline) |
| 3 | INT8 | FP16 | FP16 | 仅 stage0 INT8 |
| 4 | FP16 | INT8 | FP16 | 仅 stage1 INT8 |
| 5 | FP16 | FP16 | INT8 | 仅 stage2 INT8(参数大头量化) |
| 6 | INT8 | INT8 | FP16 | 仅 stage2 FP16(保大头精度) **← 主假设 H1** |
| 7 | INT8 | FP16 | INT8 | 仅 stage1 FP16 |
| 8 | FP16 | INT8 | INT8 | 仅 stage0 FP16 |

每 triplet 新增 6 个非 baseline 点(组合 3-8),3 triplet × 6 = **18 个新 bench 点**。

#### A.2.3 实现步骤

1. **新建 `scripts/phase2/p0_perstage_quant_bench.py`** (~150 行)
   - 复用 `m4_8_trt_build_bench.py --precision mixed --mixed-fp16-substr "<stage_name>"`
   - 输入: 3 triplet × 6 per-stage config = 18 个组合
   - 输出: `data/perstage_quant_bench.parquet`(18 行 × ~14 列)

2. **确认 layer 命名能正确匹配 stage**
   - Pyramid backbone 内部 layer 命名形如 `backbone.resnet.layer0.0.conv1` (PyTorch 风格)
   - export ONNX 后 layer 名继承,需要 grep 一次确认 `layer0/layer1/layer2` 子串能干净分组
   - 命令: `python -c "import onnx; m = onnx.load('models/p0_random_cache/onnx_064_128_256.onnx'); [print(n.name) for n in m.graph.node[:50]]"`

3. **bench 执行**
   - 单次 build + bench ~5 min,18 次 × 5 min = **1.5 hr 总 bench 时间**
   - INT8 复用已有 calibration cache(`calibration/pyramid_dair_collab_*.npy`)

### A.3 评估指标 & 决策判据

#### A.3.1 测量字段

每个 bench 点产出:
- `lat_p50_ms` / `lat_p99_ms` (TRT trtexec)
- `engine_size_mb`
- (可选) AP50 — 因 bench 时间限制,可只对 H1 主假设组合 6 跑 AP 验证

#### A.3.2 决策矩阵

| 观察结果 | 决策 |
|---------|------|
| 至少 1 个 per-stage 组合 lat **比同 triplet 全 INT8 慢 >5%**(reformat overhead 显著) | H2 成立 → **不纳入** Q1' |
| 至少 1 个 per-stage 组合 lat **介于全 FP16 和全 INT8 之间,且 AP 比全 INT8 高 >2pp** | H1 成立 → **纳入** Q1' 完整搜索空间 |
| 所有 per-stage 组合 lat / AP 都跟全 INT8 类似(±2% / ±1pp) | per-stage 量化无意义 → **不纳入**,标 reflection |

#### A.3.3 输出 artifact

1. `data/perstage_quant_bench.parquet` (18 行)
2. `results/perstage_quant_pareto.md` — Pareto 图 + 决策结论
3. 若纳入: 更新 `搜索空间一览.md` 第三节 Q1' 从 [暂定] 改为 [已纳入],总数 Q 子空间从 9 → 9 + 8 = 17,4090 协同从 954 → 106 × 17 = 1802

### A.4 风险 & 预案

| 风险 | 预案 |
|------|------|
| TRT mixed precision build 失败(layer name 匹配错或 OBEY 冲突) | 退回到 `--precision int8` 全局,逐 layer 用 `layer.precision = trt.half` 手动 override |
| reformat overhead 把 per-stage 收益完全吃掉 | 即记入 reflection,标"per-stage 量化在 ResNeXt 3-stage 上 Pareto 等价于全局",节省后续工程 |
| 18 个组合 build 失败率高(>30%) | 缩小到 H1 主假设组合 6 一个 + 2 个对照,共 3 × 3 = 9 个 bench |

### A.5 时间预算

| 步骤 | 时间 |
|------|------|
| layer name 排查 | 0.5 hr |
| bench 脚本编写 | 1.5 hr |
| bench 执行(18 点) | 1.5 hr |
| 数据分析 + Pareto 图 | 1.5 hr |
| 报告 + 文档更新 | 1 hr |
| **合计** | **~6 hr / 1 day** |

---

## Part B: D 空间扩展维度验证(按硬件分组)

**重要原则**(2026-05-11 修正): D 维度是**硬件 capability 派生**,不是固定全集。下面按硬件分两组,每组只验证该硬件**实在存在**的维度。

参考 `搜索空间一览.md` 第五·补节的派生 D 空间:
- **4090 派生 D 空间** = 4 维(D1 退化 + D2/D3/D4 暂定)
- **Orin AGX 派生 D 空间** = 6 维(D1 已纳入 + D2/D4/D5 暂定 + D6/D7 未纳入)

---

## Part B-4090: 4090 GPU 通用 D 维度验证

4090 没有 DLA,所以 IP 路由 / GPU fallback / IP 间流水线 等 Orin 特异维度**不存在**。本节只验证 GPU 通用维度。

### B-4090.1 CUDA Graph on/off (派生 D2)

#### B-4090.1.1 假设

**H_CG1**: Pyramid Fusion 含大量小 kernel(3-stage × 多 Conv = ~40 kernel launch),CUDA Graph 能减少 CPU launch 开销,在 baseline 35ms / pruned ~5ms 这种快路径上 latency 改善 +10-20%。

**H_CG2**: CUDA Graph 对 dynamic shape 或 control flow 失效,Pyramid backbone 是静态 shape 应能享 100% 收益。

#### B-4090.1.2 验证设计

**bench 候选**: 同 A.2.1 的 3 个 triplet × 2 precision (FP16/INT8) × 2 (CG on/off) = **12 个 bench 点**(其中 6 个 CG off 已是已有数据,实测只需新增 6 个 CG on)。

**实现**:
- TRT `enqueueV3` + `cuStreamBeginCapture` / `cuStreamEndCapture`
- bench 脚本改 `m4_8_trt_build_bench.py` 加 `--cuda-graph` flag
- 实现入口: 参考 NVIDIA TRT sample `sampleINT8CUDAGraph`

#### B-4090.1.3 决策判据

| 观察结果 | 决策 |
|---------|------|
| CG on 比 CG off **lat_p50 改善 >5%** AND **lat_p99 改善 >10%** | **纳入** 4090 派生 D2 |
| 改善 <5% 或 p99 反而变差 | **不纳入**,标"小 model 已经 CPU-launch-bound 不明显" |
| build 复杂度过高(>2 day debug) | 退到 "always on 作为部署默认",不参与搜索 |

#### B-4090.1.4 时间预算: **~1 day**

---

### B-4090.2 TRT tactic_sources + workspace (派生 D3 + D4)

#### B-4090.2.1 假设

**H_TC1**: TRT 10 默认 tactic_sources = {CUBLAS, CUBLAS_LT, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS}(去掉了 TRT 9 的 CUDNN),改回包含 CUDNN 是否在 Pyramid 上有差异?

**H_TC2**: workspace size 4GB(默认)→ 8GB 是否让 TRT 选到更快 tactic?

#### B-4090.2.2 验证设计

**bench 候选**: 1 个 triplet (T_baseline = 64,128,256) × 4 个 tactic 组合 × FP16/INT8 = 8 点

| # | tactic_sources | workspace |
|---|---------------|-----------|
| 1 | TRT 10 default | 4 GB |
| 2 | TRT 10 default | 8 GB |
| 3 | default + CUDNN | 4 GB |
| 4 | default + CUDNN | 8 GB |

#### B-4090.2.3 决策判据

| 观察结果 | 决策 |
|---------|------|
| 任意 tactic 组合 lat 改善 >5% | **纳入** 4090 派生 D3 + D4 |
| 改善 <5% (符合 d_space_nvidia.md §8 预期) | **不纳入**,记入 D 文档"L3 调优收益 <5%,符合 NVIDIA 调研结论" |

#### B-4090.2.4 时间预算: **~0.5 day**

---

## Part B-Orin: Orin AGX 特异 D 维度验证

Orin 有 GPU + 2 个 DLA,以下维度**仅在 Orin 上有意义**(4090 上不存在)。

### B-Orin.1 CUDA Graph on/off (派生 D2)

**与 B-4090.1 相同假设**(GPU 通用),但需在 Orin 端独立验证因为:
- Orin iGPU 微架构不同(Ampere sm87 vs Ada sm89)
- Orin 内存带宽 ~200GB/s,4090 ~1000GB/s,CPU launch 比重不同

#### B-Orin.1.1 验证设计

**bench 候选**: 3 triplet × FP16/INT8 × {CG on, CG off} = 12 点。CG off 复用 N6 闸门已有数据,新增 ~6 个 CG on。

**实现**: SSH Orin,复用 Part B-4090.1 的 `--cuda-graph` flag(同一脚本两机跑)

#### B-Orin.1.2 决策判据

| 观察结果 | 决策 |
|---------|------|
| Orin 上 CG 改善 >10%(>4090,因 launch 开销占比大) | **纳入** Orin 派生 D2 |
| <5% | 不纳入,记入"Pyramid 单帧推理 CPU launch 不是瓶颈" |

#### B-Orin.1.3 时间预算: **~0.5 day**(共用 B-4090.1 脚本,只需 SSH 跑)

---

### B-Orin.2 GPU fallback policy (派生 D4,Orin 独有)

#### B-Orin.2.1 假设

**H_FB1**: 在 Orin AGX 上,DLA 路由触发的 GPU fallback 有两种策略:
- **strict** (`--useDLACore=N` 无 fallback flag): 任何 fallback 拒绝 build,要求纯 DLA 子图
- **permissive** (`--allowGPUFallback`): 允许 fallback,build 通过但 lat 可能因 DLA↔GPU 切换变慢

不同策略对 Pareto 影响如何?

**注**: 此维度 4090 上**不存在**(4090 无 DLA,fallback 无意义)。

#### B-Orin.2.2 验证设计

**bench 候选**: 3 triplet × {DLA0+strict, DLA0+permissive, GPU-only baseline} × FP16/INT8 = ~18 点

**实现**:
- TRT `--useDLACore=0 --allowGPUFallback`(permissive) vs `--useDLACore=0`(strict)
- 需先 SSH Orin AGX,复用 N6 闸门已 ready 的 trtexec

#### B-Orin.2.3 决策判据

| 观察结果 | 决策 |
|---------|------|
| strict 失败 / permissive 通过 AND lat 差 >20% | **纳入** Orin 派生 D4 |
| 几乎所有 Pyramid layer 都在 DLA 白名单 | **不纳入**,但记入 D 文档 "Pyramid 几乎全 DLA-compatible,fallback 极少触发" |

#### B-Orin.2.4 时间预算: **~1 day**(0.3 day SSH 环境 + 0.5 day bench + 0.2 day 分析)

---

### B-Orin.3 IP 间流水线 (派生 D5,Orin 独有 - 论文核心卖点)

#### B-Orin.3.1 假设

**H_PIPE1**: Orin AGX 有 GPU + 2 个 DLA 三个独立 IP,可以做 IP-level 流水线:
- **方案 A (单 IP)**: 整模型只跑 GPU,DLA idle — 当前主流做法
- **方案 B (双 IP 并发)**: backbone 跑 DLA0,heads 跑 GPU,流水线方式重叠下一帧 backbone 与本帧 heads
- **方案 C (三 IP 并发)**: stage0/1 → DLA0,stage2 → DLA1,heads → GPU,最大化吞吐量

**预期**: 方案 B/C 在 throughput (FPS) 上比方案 A 提升 1.5-3×,但单帧 latency 可能持平或略增(IP 间转发开销)。

**这是 4090 上完全不存在的维度** — 4090 单 IP 没有 IP-level 流水线机会。

#### B-Orin.3.2 验证设计

**bench 候选**: 1 个 triplet (T_prune50 = 32,64,128) × 3 个流水线方案 × INT8 only = 3 点(初探)

| # | 流水线方案 | backbone IP | heads IP | 预期 FPS |
|---|-----------|------------|----------|---------|
| A | 单 IP | GPU | GPU | baseline |
| B | 双 IP | DLA0 | GPU | +1.5× |
| C | 三 IP | DLA0+DLA1 (stage 分配) | GPU | +2× |

**实现**(工程复杂度高):
- 拆 ONNX 为多个子图(backbone / heads)
- 分别 build TRT engine (一个 DLA,一个 GPU)
- 写 runtime 调度脚本,用 cuStream 并发触发 + 等待事件
- 实现入口: NVIDIA `sampleSegmentationDeepLab` 中的 multi-engine runtime + Orin DLA 子图分裂

#### B-Orin.3.3 决策判据

| 观察结果 | 决策 |
|---------|------|
| 方案 B/C 比 A 的 throughput 提升 >50% AND lat 不退化 >20% | **纳入** Orin 派生 D5 — 论文重大卖点 |
| 提升 <30% 或 lat 显著退化 | **不纳入**,记入 reflection "Pyramid 单帧太小不适合流水线" |
| 工程实现 >3 day | 推迟到 Phase 3 整体部署化时再做 |

#### B-Orin.3.4 时间预算: **~2-3 day**(子图分裂 + multi-engine runtime + bench)

#### B-Orin.3.5 论文意义

**这是论文 §C 三硬件对比里 Orin AGX 独有的卖点**: 4090 不能做的事,Orin 能做。如果方案 B/C 有效,paper 可以写:
> "On Orin AGX, our framework discovers a pipelined deployment configuration (B/C in Table X) achieving Y× throughput compared to single-IP execution, leveraging the 2-DLA architecture unavailable on RTX 4090."

---

## Part C: 整合后预期搜索空间数值(按硬件分开)

### C.1 RTX 4090 — 最优纳入情况

| 维度 | 当前 | 全纳入预期 | 来源 |
|------|------|-----------|------|
| B | 106 | 106 | 不变 |
| Q (全局) | 9 | 9 | 不变 |
| Q1' (per-stage) | — | +8 | Part A 验证 |
| D1 (IP 路由,退化) | 1 | 1 | 单 GPU |
| D2 (CUDA Graph) | — | +2 | Part B-4090.1 |
| D3 (tactic_sources) | — | +2 | Part B-4090.2 |
| D4 (workspace) | — | +2 | Part B-4090.2 |
| **协同 4090 总** | **954** | **106 × (9+8) × 1 × 2 × 2 × 2 = 14416** | 15.1× |

### C.2 Orin AGX — 最优纳入情况

| 维度 | 当前 | 全纳入预期 | 来源 |
|------|------|-----------|------|
| B | 323 | 323 | 不变 |
| Q (全局) | 9 | 9 | 不变 |
| Q1' (per-stage) | — | +8 | Part A 验证(若在 4090 验证通过,Orin 复用) |
| D1 (IP 路由) | 3 | 3 | 已纳入 |
| D2 (CUDA Graph) | — | +2 | Part B-Orin.1 |
| D4 (GPU fallback) | — | +2 | Part B-Orin.2 |
| D5 (IP 间流水线) | — | +3 (方案 A/B/C) | Part B-Orin.3 — **论文卖点** |
| **协同 Orin 总** | **8595** | **C11 分支后约 ~155k** | ~18× |

(精确算 Orin 协同总数受 C11 等约束分支约束,此处给量级估算)

### C.3 最坏情况

如果所有 Part A/B 验证都"无价值",维持当前数(954 / 8595),**但工程负债清零** — 我们靠实证证明了"这些维度对 Pyramid 没搜索价值",可以心安理得在 paper 里写"only 1-2 D-dimensions matter for Pyramid"。

---

## Part D: 执行顺序 & 里程碑(按硬件分组重排)

```
Phase 1 (Day 1)           Part A: Per-stage 量化验证 (4090)
                          │  ├─ layer name 排查 + bench 脚本
                          │  ├─ 18 点 bench 跑完
                          │  └─ 决策: Q1' 纳入 / 不纳入
                          │
Phase 2 (Day 2)           Part B-4090.1 + B-4090.2: 4090 GPU 通用维度
                          │  ├─ CUDA Graph 6 点 bench
                          │  ├─ TRT tactic+workspace 8 点 bench
                          │  └─ 决策: 4090 D2/D3/D4 纳入 / 不纳入
                          │
Phase 3 (Day 3)           Part B-Orin.1 + B-Orin.2: Orin SSH 验证
                          │  ├─ SSH Orin: CUDA Graph 6 点
                          │  ├─ GPU fallback 18 点
                          │  └─ 决策: Orin D2/D4 纳入 / 不纳入
                          │
Phase 4 (Day 4-6)         Part B-Orin.3: IP 间流水线验证 (论文卖点)
                          │  ├─ ONNX 子图分裂
                          │  ├─ multi-engine runtime 调度
                          │  ├─ 3 方案 (A/B/C) × 3 triplet bench
                          │  └─ 决策: Orin D5 纳入 / 不纳入
                          │
Phase 5 (Day 6.5)         整合 + 文档更新
                          │  ├─ 更新 `搜索空间一览.md` 两硬件总数
                          │  ├─ 更新 reflection_mistakes.md (若有反思)
                          │  └─ 汇总报告 `results/search_space_v2_validation.md`
```

**总预算**: **~6-7 working day**(比之前 plan 略长,主要 Part B-Orin.3 流水线工程量大)

---

## Part E: 失败兜底

| 失败情景 | 处理 |
|---------|------|
| 任何 Part X bench 失败率 >50% | 暂停该 Part,不纳入,但记录失败原因到 reflection |
| 总时间超 8 day | 砍 Part B-4090.2(TRT tactic,收益最小) |
| Orin SSH 不可用 | Part B-Orin 全部推迟到 Phase 3 真上 Orin 时一起做 |
| Q1' 验证不出 Pareto 价值但用户坚持纳入 | 接受用户决策但在文档标注"无实证收益,仅为完整性纳入" |
| Part B-Orin.3 流水线 ONNX 拆分失败 | 退化到方案 A 单 IP,纳入 D 维度数从 6 → 5,论文卖点减弱 |

---

## Part F: 与论文的对接

按"硬件派生 D 空间"框架,论文 §C 三硬件对比应该写两段不同的故事:

### 4090 章节
- **若 Q1' 纳入**: "per-stage mixed precision provides X% additional speedup over global quantization on RTX 4090"
- **若 4090 D2 (CUDA Graph) 纳入**: "runtime scheduling (CUDA Graph) reduces P99 latency by X% on RTX 4090"
- 强调 4090 = "纯 GPU 优化基线",D 空间退化到 1-4 维

### Orin AGX 章节(预期论文重心)
- **若 Orin D5 (流水线) 纳入**: "On Orin AGX, the framework discovers a 2-DLA pipelined configuration achieving Y× throughput vs single-IP, an option unavailable on RTX 4090" — **核心卖点**
- **若 Orin D4 (fallback) 纳入**: "GPU fallback policy provides Z% lat improvement when DLA whitelist is incomplete"
- 强调 Orin = "异构 IP 协同",D 空间扩展到 6 维

### Supplementary
- 列出所有 27 维 NVIDIA 栈维度,标注"在 Pyramid Fusion + 我们目标硬件上实在派生为 X 维 / Y 维"
- 引用本 plan + survey_raw_2/d_space_nvidia.md 作为 D 空间裁剪的依据

---

*维护: 每完成一个 Part,本 plan 该 Part 标 [✅ DONE / ❌ SKIPPED]并更新 `搜索空间一览.md`*
