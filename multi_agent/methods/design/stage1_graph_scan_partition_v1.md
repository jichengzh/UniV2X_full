# 阶段一：部署场景刻画 + 硬件扫描 + 网络计算图扫描与分区 (Stage 1) v1

> 状态: **设计 + 实现已落地** (2026-06-12, 4 模型 scan_status 全 ok); 含整合层(S2b 搜索旋钮 / S4b 路由段)。**2026-06-14 补子流程 C「延迟刻画与瓶颈定位」(设计, 待实现)**
> 创建: 2026-06-11 · 更新: 2026-06-14
> 实现: `framework/stage1/{hardware_scan,graph_scan,adapters,run_scan}.py` · 产物 `framework/partitions/*_partition.yaml`(+ `README_scan_results_v1.md`)
> 定位: 协同加速框架六步流程的 **step 1 完整展开**。阶段一 = **三条子扫描** → 汇合成搜索空间:
>   - 子流程 A「硬件扫描」: 把目标芯片刻画成 `hw_capability.yaml`
>   - 子流程 B「网络计算图扫描与分区」: 把待加速网络(CoDriving)扫成 `network_partition_manifest.json`
>   - 子流程 C「延迟刻画与瓶颈定位」: 在目标硬件对**基线**逐段/逐单元剖析延迟 → `latency_profile`(Amdahl 上界 + 逐单元延迟权重 + 瓶颈类型)
>   - 汇合: A 的对齐/精度/算子白名单**约束** B 的三视图取值; C 给每个单元贴**延迟权重 + 可加速轴提示** → 输出搜索器可直接消费的合法搜索集合 + 优先级。
> 关联: `协同加速框架_工作流_v1.5.md`(§1 step 1) · `framework/capability_schema.py` + `configs/hardware/rtx4090.yaml`(A 的 schema+实例) · `tools/configurable/depgraph_pyramid.py`(B 的已验证原型) · `dims_pruning_v1.md`/`dims_quantization_v1.md`/`dims_hardware_v3.md`(三维) · `framework/searcher_v0.py`(消费方)。

---

## 0. 动机与范围

`协同加速框架_工作流_v1.5.md` 原本只把 step 1 写成「画硬件 capability 清单」。但优化对象是 **(网络 × 硬件) 的配对** —— 只扫硬件不扫网络, 搜索器拿不到合法的搜索键集合(剪枝率挂在哪些单元?哪些层必须同剪?量化模块边界在哪?哪些节点能上 DLA?)。这些一半是**硬件属性**(对齐/精度/算子白名单), 一半是**网络结构属性**(依赖组/语义模块/op 拓扑)。

而且光知道"哪些单元能调"还不够 —— 还得知道"**时间花在哪**"。项目实测教训(CLAUDE.md): 延迟高度集中(NMS 占 e2e 72.4% / backbone 占稠密核大头 / backbone memory-bound ~30% 带宽)。若不先定位瓶颈, 搜索器对所有单元一视同仁地搜剪枝率/量化档 = 没踩在瓶颈上, 且被 Amdahl 锁死收益。所以还要一条**延迟刻画**子流程(C)。

所以阶段一必须**同时**做三件事, 再**汇合**(A 约束 B 的取值, B 反向裁剪/约束 A, C 给每单元贴延迟权重+可加速轴; 见 §4):
```
        ┌─────────────────────────┐        ┌──────────────────────────────┐
输入 →  │ A. 硬件扫描              │ ◀────▶ │ B. 网络计算图扫描与分区        │
        │  芯片 + 场景 + SLA       │  双向  │  待加速网络 + ckpt             │
        │  → hw_capability.yaml    │  耦合  │  → 三视图(B1剪枝/B2量化/D路由)  │
        └────────────┬────────────┘        └────────────────┬─────────────┘
                     │            ┌───────────────────────┐ │
                     └──────────▶ │ C. 延迟刻画与瓶颈定位   │◀┘  (A 给硬件+基线, B 给单元归因)
                                  │  基线逐段/逐单元剖析    │
                                  │  → latency_profile      │
                                  │    (Amdahl 上界 +       │
                                  │     逐单元延迟权重 +    │
                                  │     memory/compute 类型)│
                                  └───────────┬───────────┘
                     └──────────── 汇合(§4) ──┴──────────┘
                                    ▼
          搜索空间合法集合(键集 + 取值上下界) + 优先级(延迟权重) + Amdahl 天花板 → searcher_v0
```
> 双向: **A→B** 硬件约束软件合法取值(对齐/精度/算子白名单); **B→A** 网络结构反向塑形硬件侧(op-set 裁剪扫描 / footprint×SLA 可行性闸门 / D 路由空间由 B 的 op-tag 定义)。**C 依赖 A(目标硬件 + 基线实测)与 B(单元边界做归因), 产出落在每个搜索旋钮上的延迟权重**。详见 §3.5 / §4。

**范围裁剪 (用户拍板 2026-06-11)**:
- **不模仿 UPAQ 的 DFS root-leaf 通道搜索** —— 分组交给已验证的 DepGraph(torch_pruning), 不自研 DFS。
- **不做半结构化/非结构化剪枝** —— 2:4 / element 拿不到真加速, 从搜索空间剔除 ⇒ 剪枝对象轴 `prune_object` **塌缩为常量 `channel`**。
- 阶段一只「扫描 + 出合法集合」, **不做剪枝/量化/build 本身**(那是 step 3/4)。

---

## 1. 子流程 A — 硬件扫描 (Hardware Capability Scan)

**目的**: 把"一颗芯片 + 一个部署场景"刻画成一份机器可读的 `hw_capability.yaml`, 让后续硬约束与搜索空间**按它自动展开**(不为每种硬件写一份工作流)。

**输入**: 待加速网络(供 baseline 实测) + 部署场景(车端/路侧/数据中心/边缘) + 业务 SLA(目标 FPS / 延迟上限 / 功耗预算 / 显存上限)。

**输出**: `hw_capability.yaml`(schema = `configs/hardware/schema.yaml v1.0`, 校验器 = `framework/capability_schema.py::HardwareCapability`)。

### A.1 探测/查表 (在目标硬件上, 不用开发硬件代替)
| 子项 | 怎么拿 | 填到 YAML 哪 |
|------|--------|-------------|
| 架构 / SM / TC 代际 | `nvidia-smi`, deviceQuery, 厂商手册 | `arch.family/sm/tensor_core_gen` |
| 可用 IP 列表 | 硬件手册(GPU? 几个 DLA? NPU?) | `ips.{gpu,dla,cpu}.enabled` |
| 各 IP 支持精度 | NVIDIA TC 表 / DLA 文档 | `ips.*.precisions` |
| **DLA 算子白名单** | TRT DLA 支持矩阵 | `ips.dla.op_whitelist` (★ B 的 S4 用) |
| 量化约束 | TRT 文档(对称性/per-channel act/位宽/粒度) | `quant_constraints.*` (★ B 的 S3 用) |
| 对齐拐点 | TC tile 要求 + 实证 | `alignment.{int8_dense_channel,...}` (★ B 的 S2 用) |
| 功耗档 / 显存 / 带宽 | nvpmodel / 规格 | `power` / `memory` |
| 工具链版本(锁定) | trtexec --version, jetpack | `toolchain.*` |

### A.2 baseline 实测 (在目标硬件)
原模型在目标芯片上的 e2e / 子模块 latency, 写入 `empirical_baseline`(rtx4090.yaml 已有: `univ2x_coop_pytorch_e2e_ms` / `bev_encoder_fp16_trt_ms` 等)。⚠️ **不用开发硬件 latency 冒充目标硬件**(4090 ≠ Orin, 带宽差 ~5×)。

### A.3 退化规则 (capability 决定搜索空间形状)
- **无 DLA**(rtx4090.yaml `ips.dla.enabled=false`): D.L1 设备路由轴退化为仅 GPU 内部; DLA 相关硬约束(per-tensor/HWC4/算子白名单)整体移除; B2 per-channel 全图可用(搜索空间反而变大)。
- `alignment_enforcement: hard|soft|auto` (capability_schema 已实现): 决定 B 的 `width_floor` 是**硬过滤**(hard)、**惩罚项**(soft, N2v2 在 Orin 实测 implicit padding 抵消对齐效应)、还是**交给数据驱动学习**(auto)。这是阶段一→搜索器的一个实证驱动旋钮。

### A.4 真实例 (已在仓库)
`configs/hardware/rtx4090.yaml` 即一份合格产出: `ips.gpu.precisions=[FP32..INT8,INT4]` / `dla.enabled=false` / `alignment.int8_dense_channel=32` / `quant_constraints.{symmetric_only=true, per_channel_activation_supported=false, bit_widths_w=[4,8,16,32], granularity_a=[per_tensor]}` / `toolchain.framework_version=10.13`。Orin AGX / Orin Nano 各再补一份。

---

## 2. 子流程 B — 目标网络结构 (CoDriving, 已读源码核实)

`opencood/models/center_point_codriving.py::centerpointcodriving`:
```
pillar_vfe (PillarVFE)         # 稀疏, data-dependent scatter — 不可 trace, 跳过
  → scatter (PointPillarScatter) → spatial_features (1, 64, H, W)   # 稠密入口
  → backbone (BaseBEV / ResNetBEVBackbone)   # 可剪 CNN 主干 ★
  → shrink_conv (DownsampleConv, 可选)        # neck ★
  → naive_compressor (可选)
  → fusion_net = CoDriving(codriving_attn)    # 多尺度融合, 通道保持 — trace 走单 agent 路径
  → cls_head (Conv2d 1×1, out=anchor_number)  # 输出头, 通道固定 → 冻结
  → reg_head (Conv2d 1×1, out=8×anchor)       # 输出头, 通道固定 → 冻结
```
与 PyramidFusion 几乎同构, 故 Pyramid 的 `FullTraceNet` 破解可直接迁移。**注意 CoDriving 只有 cls/reg 两头(无 dir_head)**。

---

## 3. 子流程 B — 网络计算图扫描与分区算法 (6 阶段)

### S0. 构造稠密可 trace 核 `FullTraceNet`
只 trace **稠密、固定 shape、通道相关**的部分:
- 入口 = `scatter` 输出 `spatial_features (1,64,H,W)`(跳过稀疏 `pillar_vfe`)。
- `fusion_net` 多尺度融合只做空间重加权、**不改通道数** → trace 走单 agent 路径等价。
- 出口 = `cls_head + reg_head`(+ 若有 aux/occ head 全接出, 逼 DepGraph 不当 dead)。
- ⚠️ **关键教训(Pyramid 根因复现)**: forward 入口必须从真实 `nn.Module`(如 `backbone.forward`)起步, **不能用 method-call 入口**, 否则 DepGraph 看不到首个残差 identity 的来源, 剪后 forward 必崩(`size X must match Y`)。

### S1. 构建依赖图
`DG = tp.DependencyGraph().build_dependency(net, example_inputs=x)` → `groups = DG.get_all_groups()`。DepGraph 自动把"必须同剪"的层归到同一组(残差 add / concat / grouped-conv `groups=g`)。**分组到此完成, 不自研 DFS。**

### S2. 视图① — B1 剪枝单元 (从依赖组; round_to 来自子流程 A)
| 字段 | 含义 | 来源 |
|------|------|------|
| `group_id` / `member_layers` | 同剪的层名集合(暴露跨模块耦合) | DepGraph |
| `prune_dim` / `cur_width` | 代表层输出通道 + 当前宽度 | 图 |
| `round_to` | **来自 A.alignment**: INT8→`int8_dense_channel`(=32) / FP16→`fp16_dense_channel`(=8) | A × 量化归属 |
| `width_floor` | round_to 决定的最小宽度; grouped-conv 还要 `%g==0`, ResNeXt 触发 wpg→16 防 width=0 崩 | 约束 |
| `enforcement` | **来自 A.alignment.alignment_enforcement**: hard 过滤 / soft 惩罚 / auto 学习 | A |
| `max_rate` | `(cur_width - width_floor)/cur_width` ← **D1 剪枝率上界** | 派生 |
| `criterion_pool` | CNN+BN→{L1,FPGM}; attn/Linear→{Taylor}; 默认 L1 ← **D3 准则候选** | 按层类型 |
| `prunable` | 输出头(anchor 固定)/单通道 head → `False`(冻结; 输入侧随依赖图自动跟剪) | 规则 |

> **★ Q1 教训 — 剪枝视图必须以 DepGraph 依赖组为准, 不能用语义 name-prefix 桶 (Pyramid 实测对照)**:
> 把 trace 分组与之前 Pyramid 手工分组对照(`depgraph_pyramid` 报告 §3-§4)发现:
> - **stage 内部一致**: 手工 `transfer_weights` 的 `conv1→conv2→conv3→残差→downsample→下一 stage conv1` 耦合, DepGraph 实测完全复现。
> - **手工分组错 2 处**: deblocks "输出 128 固定→不可剪"、shrink_conv "384 concat→不可剪" —— DepGraph 实测**输入侧通道随上游可剪, 自动联剪**, 旧分析错。
> - **手工分组漏 1 处(最关键)**: `backbone_m1 ↔ pyramid stage0` 因残差 identity **属于同一依赖组必须同剪**; 按语义模块名分桶会把它们当独立模块, 导致"只剪 stage0"组内 ratio 冲突 → 实测 -0.0% 剪不动。
> ⇒ **剪枝单元(视图①)= DepGraph 依赖组**(抓真实数据流耦合); **语义 name-prefix 桶只用于 B2 量化视图(视图②)**, 绝不用来定义剪枝边界。

### S2b. 视图①b — B1 搜索旋钮 (把 N 个耦合组按 stage/角色整合成少数搜索变量)

> **★ 核心区分 — "结构真相" vs "搜索旋钮" (Q: 43 个独立剪枝率组合爆炸)**:
> DepGraph 出的 N 个依赖组是「**结构上能独立剪的最小耦合单元**」(事实, 视图① 保留), **但不等于「该搜的旋钮」**。让每个组独立设率 = `~6^43 ≈ 10^33` 不可行; 且项目实测**剪枝 Pareto 退化**(`dims_pruning §8.4-8.5`: AP 由总容量主导、非逐层分配, 可达范围无精度悬崖) → per-group 精细搜率本就无收益。

**整合逻辑 (参考现有方法)**:

| 方法 | 整合粒度 | 旋钮数 |
|------|---------|--------|
| **UPAQ** | root-leaf group 每组一个统一率 | = 组数 (结构驱动) |
| MetaPruning / AMC | per-stage / per-block | 几个 |
| HALP / NetAdapt | 全局预算 + 重要性重分配 | 1 个预算 |
| APQ | 敏感度分层 (high/med/low) | 3 档 |

**本框架取**: 按 `语义桶 + stage 索引` 把耦合组绑成搜索旋钮 (`backbone.s0/s1/s2 / bev_encoder.s0/s1/s2 / neck`)。tying 永远合法(只放弃自由度); 每旋钮: `max_rate=成员 min`(最紧约束, round_to floor 透出)、`criterion_pool=成员交集`、`grouped`/`round_to`=成员保守值。

**实测整合效果 (4 模型)**: codriving 20→**4**, pyramid_lidar 43→**5**, pyramid_camera 46→**6**, v2xvit 23→**4**。pyramid 搜索空间 `6^43 → 6^5 ≈ 7776` 可枚举/NSGA。搜索器**只看 `view_b1_search_groups`**, `view_b1_prune_groups`(43 组)作结构真相保留不直接搜。
> 诚实校验: camera 的 `other`(冻结 aligner)旋钮 `max_rate=0.0`, 整合视图如实标"不可剪"。

### S3. 视图② — B2 量化单元 (语义桶, 对齐依赖组; 位宽合法集来自 A)
- name-prefix 归语义桶: `backbone / bev_encoder / neck(shrink) / heads / fusion`。
- **硬约束: 一个量化单元 = 该桶内若干"完整依赖组"的并集** → 量化边界永不切碎耦合组(B1/B2 视图天生一致)。
- 每单元产出: `param_count` · `op_composition` · `quantizable` · **`legal_bits`/`legal_granularity`(来自 A.quant_constraints: 如 4090 `symmetric_only`、`per_channel_activation=false`、`bit_widths_w=[4,8,16,32]`)** ← 喂 **B2 模块选择 + 位宽/粒度轴**。

### S4. 视图③ — D 路由候选 (逐节点 op-type tag × A.算子白名单)
- 每节点打 op-type(Conv/ConvT/Linear/BN/Act/Add/Concat …)。
- **若 A 有 DLA**(`ips.dla.enabled=true`): 按 `ips.dla.op_whitelist` 给每节点打 `dla_able`, 并按"路由触发 B2 强制 INT8/per-tensor/HWC4"做 **D→B2 硬约束传播标记**。
- **若 A 无 DLA**(如 4090): 路由视图退化, `dla_able` 全 `false`/`null`, D.L1 仅 GPU。
- 逐节点 tag 是**素材, 不是决策空间**(视图③ 保留)。

### S4b. 视图③b — D 路由段 (逐节点折叠成连续可路由子图 = 真决策粒度)

> **★ Q: 只有 2 DLA + 1 GPU, 逐节点(64-137)路由太细**: 对的。真决策粒度 = **最大连续可路由子图** —— GPU↔DLA 每切换插一次 reformat/拷贝(撞带宽), 所以只在**几个切点**决策, 把图切成 GPU 段 / DLA 段(这些模型 2-4 段), 不是 137 个独立选择。

- **若无 DLA**(4090): 137 节点折叠成 **1 个 GPU 段**, D.L1 退化无路由决策(`view_d_routing_segments.n_segments=1`)。
- **若有 DLA**(Orin): 折叠出多个 DLA-able 连续段, 路由决策 = 选哪些段真放 DLA0/DLA1; DLA-able 段触发 D→B2 传播(强制 INT8/per-tensor/HWC4)。
- caveat: 连续性用 named_modules 顺序近似拓扑序; 精确需 torch.fx 拓扑(路线 C, 延后)。
- **项目实证补充**(CLAUDE.md): Pyramid DLA INT8 build 0/12 失败、单 GPU pipeline 并行已证伪, 只 Orin 异构 GPU∥DLA 多进程有 1.34× → D 路由仅在 Orin 有意义且粗放(整段级)。搜索器**只看 `view_d_routing_segments`**。

### S5. 全局统计 + 一致性校验 (fail-fast)
- 参数分布(如 Pyramid 实测 backbone 68.9% / shrink 26.8%) → 告诉搜索器"剪哪里值" + LGB `model_size` 特征。
- 校验: ① 所有 `prunable` 组 floor 后 `%round_to==0`; ② 每量化单元 = 整组并集; ③ 输出头已冻结; ④ **dry-run 0.5 剪 + forward sanity**(复用 `depgraph_pyramid.run_global_prune`)。

### S5b. 逐层时延 profiling (子流程 C.1+C.3 落地, 描述性旁注非约束)
- 复用 S0 的 `(net, x)` 挂 CUDA-Event forward hook 逐叶子算子计时 → 聚合到 B 的搜索旋钮/量化单元 → `view_latency`。详见 §3.5 C.1/C.3。
- 模式 `--profile-latency {auto|on|off}`: auto=仅 cuda 开, on=总开(cpu 估算), off=关。**不参与合法性判定** —— 保持 §4.1 "逐候选性能反馈推迟到搜索回路"的边界(C 是一次性静态先验 = 刻画输入, 非 step3/4 动态反馈)。

### S6. 落盘 → 接搜索器
`network_partition_manifest.json` 写出, `searcher_v0.py` 读 (A 的 yaml + B 的 manifest + C 的 latency_profile) 得到合法搜索集合 + 优先级(见 §4)。

---

## 3.5 子流程 C — 延迟刻画与瓶颈定位 (Latency Characterization & Bottleneck Localization)

> 状态: **C.1+C.3 已实现 (2026-06-15, `framework/stage1/latency_profile.py`, 作为 `scan()` 内建步骤 S5b); C.2 Amdahl 闸门 + C.4 bound/axis_hint 待实现**。A 扫硬件能力、B 扫图结构, 但都没回答"时间花在哪"。C 对**基线模型一次性**做延迟剖析, 把延迟归到 B 的同一套单元上, 产出三样东西喂搜索器: **(i) Amdahl 上界闸门 · (ii) 逐单元延迟权重 · (iii) 瓶颈类型(memory/compute bound)→ 决定哪条轴(剪枝/量化)真能加速**。

**与 A.2 的关系**: A.2 只测 e2e/子模块**总** latency。C 把它**按 B 的单元(搜索旋钮/量化单元/路由段)逐项拆开**, 让每个搜索旋钮带一个"它占多少延迟"的权重。

**与 B 的互补**: B 的 `FullTraceNet`(S0)**故意跳过**稀疏 VFE / scatter / 通道保持的融合 —— 这些跳过的部分**正是 C 的 Amdahl `f_fixed` 分母**(搜索轴动不了的固定开销)。B 只看可调稠密核, C 必须看完整 e2e, 两者互补不重叠。

**与"性能反馈耦合"边界(§4.1)的区分 — 不越界**: C 剖析的是**未优化基线**, **一次性、静态**, 与 A.2 同性质(刻画输入)。§4.1 排除出阶段一的那条回路是 **step 3/4 逐候选**(measure 每个被搜出的 config)的**动态**反馈。**C ≠ search**: C 给"把注意力放哪 + 天花板多高"的先验, 不给最终配置延迟。

### C.1 测什么 (目标硬件, 空闲)
- 基线模型**逐层/逐段** latency。手段(真测, 非估算): TRT `trtexec --exportProfile=layer.json --profilingVerbosity=detailed`(逐层 ms) / torch profiler + CUDA Event / nsys。
- 至少测 **FP16 基线**; 有 INT8 build 时一并测(作量化收益的分母)。
- ⚠️ 纪律(CLAUDE.md): 必须**空闲 GPU**(util 0% / mem≤50MiB)、**目标硬件**(4090≠Orin, 带宽差 ~5×, 不可冒充)、标注口径(子模块 / e2e / 含不含 NMS)。

> **★ 已实现 (2026-06-15, `latency_profile.py` 步骤 S5b)**: 采 **torch CUDA-Event forward hook** 路径 —— 复用 `scan()` S0 已建的 `(net, x)`(不重建模型), 对所有**叶子模块**挂 pre/post hook 逐算子计时(warmup 30 + measure 100, chain 累加), 与 B 视图**同一套层名**天然对齐。provenance 记录 device / GPU util+mem 快照 / dtype / 口径(random weight, single-agent trace 核, body-only 不含 voxelize+NMS)。
> - `device=cuda` 且空闲 → `status=ok` 真测; 占用高 → `ok_but_contended` 降级警告(不阻断)。
> - `device=cpu` → `status=estimated_cpu`, **明确不代表 GPU 部署延迟**。实证: CPU smoke-test 把 bev_encoder 占比测成 36.8%(GPU 真测应 ~92%, 见 `pyramid_lidar_structure_audit_v1.md §7`)—— grouped conv 在 CPU 上相对耗时与 GPU Tensor Core 截然不同, 故 CPU 路径仅供链路验证, 真测必须空闲 cuda。
> - TRT layer-json / nsys 作**更细/更贴部署**的可选路径, 待实现 (hook 路径已够给搜索器优先级权重)。

### C.2 三段归因 (Amdahl 分解 = 搜索能否加速的总闸门)
| 段 | 例 | 搜索旋钮能否触及 |
|----|----|----------------|
| **pre/post 固定** | voxelize ~10ms / scatter / **NMS(实测 72.4% e2e)** / fusion 通道保持 | ❌ 搜索轴动不了(NMS 已单独 CUDA 化 3.04× = step 0 工程, 不在搜索空间) |
| **稠密可优化核** | backbone / bev_encoder / neck | ✅ B1 剪 + B2 量化 + D 路由都作用于此 |
| **输出头(已冻结)** | cls/reg/dir | ❌ S2 已标 `prunable=false` |

⇒ **Amdahl 上界** `S_max = 1 / (f_fixed + f_core / r_core)`; 若 `f_fixed≈0.70`, 则把核加速到 ∞ 也只 `1/0.70≈1.43×`。这是搜索**开跑前**就该知道的天花板, 写进 manifest 作搜索器的**现实预期 / 早停 / 可行性闸门**(SLA 若要求 2× 而天花板 1.43× → 该模型在该硬件单靠剪枝量化达不到, 需换硬件/改架构/解 NMS)。

### C.3 逐单元归因 → 优先级权重
- 逐层 ms 按**层名**映射到 B1 搜索旋钮(stage 桶) / B2 量化单元(语义桶) / D 路由段 —— 与 B 视图**同一套层名**, 天然对齐。
- 每个 `view_b1_search_groups[k]` 加 `latency_ms` + `latency_share`(占稠密核比例)。搜索器据此**把剪枝率预算优先压在高占比旋钮**(NetAdapt / AMC 的 latency-LUT 思路: 优化高成本层 ROI 最高)。

> **★ 已实现 (S5b)**: `view_latency.by_b1_search_knob[]` 与 `by_b2_quant_unit[]` 各带 `latency_ms` + `pct_of_e2e`(降序), 复用 `graph_scan._search_key` 做旋钮归并键, 与 `view_b1_search_groups` / `view_b2_quant_units` 用同一套层名对齐。另出 `top_layers`(前 15 热点层)+ `attributed_ms`/`unattributed_ms`(叶子和 vs e2e 余项, 诚实暴露未归因部分如 elementwise/fuse)。全量逐层明细落 sidecar `results/stage1_latency_<model>.json`(yaml 只留聚合+top-K, 控体积)。

### C.4 瓶颈类型 → 选对加速轴 (耦合, 接 §4.1)
- 每段标 **memory-bound / compute-bound**(roofline; CLAUDE.md `E_headroom` 实测 backbone ~30% 带宽 = memory-bound)。
- **关键耦合**: memory-bound 段 INT8 的**算力收益打折**(瓶颈在搬数不在算) → 主要靠**通道剪枝**(同时降算力+权重/激活搬运); compute-bound 段 INT8 收益足。⇒ 瓶颈类型决定该单元优先搜哪条轴, 写进 manifest 的 `axis_hint ∈ {prune, quant, both}`。

### C.5 落盘 (并入同一 manifest)
见 §5 `latency_profile` 字段。核心: `amdahl.{f_fixed, f_core, s_max}` + `per_search_group[].{latency_ms, latency_share, bound, axis_hint}` + `caveat`。

> **★ 已实现字段 (S5b, manifest key = `view_latency`, 与 view_b1/b2/d 并列)**: `status` / `provenance`(method+GPU快照+口径note) / `e2e_forward_ms` / `attributed_ms` / `unattributed_ms` / `by_b2_quant_unit[]` / `by_b1_search_knob[]` / `top_layers[]` / `full_detail_sidecar`。`search_space_summary` 另出 `latency_status` + `latency_e2e_forward_ms` 一行摘要。**待补(C.2/C.4)**: `amdahl.{f_fixed,s_max}`(需测含 voxelize+NMS 的完整 e2e, 当前 hook 仅 body 稠密核)与 `bound` / `axis_hint`(需 roofline)。

> ⚠️ **诚实 caveat**: 这是**基线静态先验**, 用于"把搜索注意力放对地方 + 给天花板", **不是**最终配置延迟。剪枝/量化会改变逐层时间, C 给**初始优先级 + Amdahl 天花板**, 最终延迟由 step 3/4 逐候选实测 + LGB 预测器细化(LGB latency 跨数量级崩 → 见 CLAUDE.md §四, 实际可能退化为 rule-based)。

---

## 4. ★ 汇合 — 双产出如何合成搜索空间 (阶段一的核心交付)

阶段一的价值不在两份文件各自, 而在**它们相乘**得到合法搜索集合。对应 `工作流_v1.5 §0.2 / 步骤2` 的硬约束注入:

| 搜索维 | B 提供(网络结构) | A 提供(硬件约束) | 汇合后给搜索器 |
|--------|-----------------|-----------------|----------------|
| **D1 剪枝率** | 视图①b `search_group_id`(整合旋钮, 非 43 组) | `alignment` → `round_to` + `enforcement` | `prune_rate[knob] ∈ [0, max_rate]` |
| **D2 剪枝对象** | — | — | **常量 `channel`**(2:4/element 剔除) |
| **D3 准则** | 视图①b `criterion_pool`(成员交集) | — | `prune_criterion[knob] ∈ pool` |
| **B2 位宽/粒度** | 视图② `unit` + `quantizable` | `quant_constraints.{bit_widths,granularity,symmetric,per_channel_act}` | `q_bits[unit] ∈ legal_bits` |
| **D.L1 路由** | 视图③b 路由**段**(非逐节点) | `ips.dla.{enabled,op_whitelist}` | `route[seg] ∈ {gpu} ∪ ({dla} if dla_able)` |
| **D→B2 传播** | 视图② 单元归属 | DLA 强制 INT8/per-tensor/HWC4 | 路由到 DLA 的单元锁死量化轴 |
| **优先级权重** | 视图①b 旋钮边界(归因锚) | **C 给目标硬件基线逐单元 latency** | 每旋钮 `latency_share` → 搜索器先压高占比单元 |
| **可加速轴提示** | 单元算子构成 | **C 给 memory/compute bound** | `axis_hint[unit] ∈ {prune,quant,both}` |
| **Amdahl 闸门** | B 标出的可调核 vs 固定 pre/post | **C 给 f_fixed / s_max** | 搜索天花板 → 现实预期 / 早停 / SLA 可行性 |

> ★ 搜索器消费的是**整合视图**(`view_b1_search_groups` ~5 旋钮 / `view_d_routing_segments` 2-4 段) + **C 的延迟权重**, **不是**结构真相(43 组 / 137 节点)。后者作事实保留。搜索空间从 `~10^33` 压到 `~10^4`, 再由延迟权重定**搜索顺序/预算分配**。

一句话: **B 决定"图上有哪些可调单元", A 决定"每个单元在这颗芯片上合法取值是什么", C 决定"先动哪个单元、天花板多高、该走剪枝还是量化"**; 三者缺一, 搜索空间(及其优先级)就建不全。这也回答了项目初衷(CLAUDE.md): "搜索空间才能构建起来, 不然所有优化方案速度一样精度一样, 还搜什么" —— 而 C 进一步保证搜索**不空转在非瓶颈上**。

### 4.1 ★ 双向耦合 (Q2) — 不止 A→B, 还有 B→A 与 D↔B2

上表是 **A→B**(硬件约束软件取值)。完整耦合还有两层:

**B→A (网络结构反向塑形硬件侧)**:
| 耦合 | B 提供 | 作用于 A / 搜索空间 |
|------|--------|--------------------|
| op-set 裁剪扫描 | 网络实际用到的算子/精度集合 | A 只需展开相关 IP/精度/算子白名单条目(CoDriving 全 Conv/BN → 无关 attention 的 DLA 约束不展开), 扫描更省 |
| **D 路由空间由 B 定义** | 视图③ 逐节点 op-tag | "哪些子图能上 DLA"是 D 轴(硬件侧)决策, 但合法集合来自 B 的 op-tag → B 喂给 A 侧 |
| **footprint × SLA 闸门** | 参数量 / 激活峰值(S5 stats) | vs A.`memory.capacity`: 超显存预算 → 判该硬件不可行, 或**反推 B1 最小剪枝率下界** |

**D↔B2 双向硬约束传播 (框架 §0.2, manifest 内显式建边)**:
- 模块路由到 DLA → B2 **强制** INT8/per-tensor/HWC4(`view_d_routing[node].propagate_b2` 指向被锁死的量化单元);
- B2 选 per-channel(精度需要)→ 该模块**禁止**走 DLA(反向收窄 D.L1)。
⇒ manifest **不是三个独立列表**, 视图间带传播边。

> **⚠️ 边界(职责不越界)**: 阶段一的耦合是**结构性/约束性**(静态: 谁约束谁的合法取值)+ **基线延迟刻画**(C: 一次性剖析未优化模型, 同 A.2 性质)。**两者都静态**, 都只刻画"输入(网络×硬件)长什么样"。**被排除的是逐候选性能反馈回路**(框架 §0.4: measure 每个被搜出的 config 的 latency/AP → 反馈搜索器)—— 那是 **step 3/4 动态**, 不塞进阶段一。
> 一句话切线: **C(在阶段一)= 对 1 个基线测 1 次, 给优先级先验; step 3/4 = 对 N 个候选各测一次, 给最终延迟。** C 给"该往哪搜", step 3/4 给"搜到的好不好"。把 C 误当后者就会让阶段一退化成搜索, 职责越界。

---

## 5. manifest schema (★已实现, 见 `framework/partitions/<model>_partition.yaml`)

> 落盘格式 = YAML (非 JSON)。下为字段结构 (数值为占位)。实现: `framework/stage1/graph_scan.py`。

```yaml
model: centerpointcodriving
ckpt: <path>  ;  config: <path>  ;  ckpt_status: ok | opv2v_only_no_dair
hw_capability: {name, has_dla, int8_align, legal_bits, ...}   # 子流程 A 汇合摘要
trace: {entry_shape: [1,64,192,704], skipped_modules: [...], note: ...}
prune_object: channel                                          # 锁定 (不搜 2:4/element)
stats: {params_total: 0, param_dist: {backbone: 0.0, neck: 0.0, heads: 0.0}}
search_space_summary:                                          # ★ 结构真相 vs 搜索旋钮
  n_b1_groups_structural: 43      # DepGraph 最小耦合单元 (事实)
  n_b1_search_knobs: 5            # 整合后真正要搜的剪枝率旋钮
  n_b2_quant_units: 4
  n_d_routing_segments: 1
view_b1_search_groups:            # ★ 搜索器用这个 (整合旋钮)
  - {search_group_id: bev_encoder.s2, bucket: bev_encoder, n_b1_groups: 17,
     widths: [256,512], max_rate: 0.875, round_to: 32, grouped_conv: true,
     criterion_pool: [FPGM, L1], member_b1_groups: [g..., ...]}
view_b1_prune_groups:             # 结构真相 (43 组, 保留不直接搜)
  - {group_id: g0, bucket: ..., root_layer: ..., member_layers: [...],
     cur_width: 64, grouped_conv_g: 1, round_to_default: 32, width_floor: 32,
     max_rate: 0.5, criterion_pool: [L1, FPGM], prunable: true, coupled_buckets: [...]}
view_b2_quant_units:
  - {unit: backbone, param_count: 0, op_composition: {Conv2d: 0, BatchNorm2d: 0},
     member_groups: [g0, g1], quantizable: true,
     legal_bits: [INT8, FP16], legal_granularity_w: [per_tensor, per_channel]}
view_d_routing_segments:          # ★ 搜索器用这个 (路由段 = 真决策粒度)
  {n_segments: 1, note: "无 DLA → 整图单 GPU 段", segments: [{device: gpu, n_nodes: 137}]}
view_d_routing:                   # 逐节点 tag (素材, 保留)
  - {node: backbone.layer0.conv0, op_type: Conv2d, dla_able: false, propagate_b2: null}
latency_profile:                  # ★ 子流程 C (基线静态剖析; 待实现)
  hw: rtx4090 ; precision: fp16 ; source: trtexec_exportProfile ; idle_verified: true
  e2e_ms: 26.70 ; core_ms: 0.0    # core = 稠密可优化核
  amdahl:
    f_fixed: 0.0   # voxelize+scatter+NMS+fusion 等搜索动不了的固定开销占比
    f_core:  0.0   # 稠密可优化核占比
    s_max:   1.0   # = 1/(f_fixed)  把核压到 0 的理论天花板 → 搜索可行性闸门
  per_search_group:               # 与 view_b1_search_groups 同 key 对齐
    - {search_group_id: bev_encoder.s2, latency_ms: 0.0, latency_share: 0.0,
       bound: memory | compute, axis_hint: prune | quant | both}
  caveat: "基线静态先验(非逐候选); 剪枝改图后逐层延迟会漂移, 此为优先级+天花板, 非最终延迟"
checks: {all_floor_mod_round_to: true, dryrun_prune05: {status: ok, reduction_pct: 53.9, ...}}
scan_status: ok | partial | fail
latency_status: ok | not_measured | hw_mismatch    # C 未在目标硬件实测时如实标

---

## 6. 阶段一整体伪流程

```
def stage1(model, ckpt, hw_yaml | (chip, scenario, sla)):
    # ── 子流程 A: 硬件扫描 ──
    if hw_yaml is None:
        cap = probe_hardware(chip, scenario, sla)      # A.1 探测/查表
        cap.empirical_baseline = measure_baseline(model, chip)  # A.2 目标硬件实测
        hw_yaml = cap.to_yaml()
    cap = HardwareCapability.from_yaml(hw_yaml)         # 校验 (capability_schema)

    # ── 子流程 B: 网络计算图扫描与分区 ──
    net = FullTraceNet(model, trace_adapter)            # S0
    DG  = build_dependency(net)                         # S1
    b1  = extract_prune_groups(DG, align=cap.alignment) # S2  结构真相(N 耦合组)
    b1k = consolidate_search_groups(b1)                 # S2b 整合 → ~5 搜索旋钮
    b2  = extract_quant_units(DG, qc=cap.quant_constraints)  # S3 (legal_bits 来自 A)
    d   = tag_routing(net, dla=cap.ips.get("dla"))      # S4  逐节点 tag(素材)
    dseg= routing_segments(d, cap)                      # S4b 折叠 → 连续路由段(决策粒度)
    stats, checks = stats_and_validate(net, DG)         # S5 (dry-run 0.5 剪 forward)

    # ── 子流程 C: 延迟刻画与瓶颈定位 (基线一次性, 目标硬件空闲) ──
    view_latency = profile_latency(net, x, adapter, dev) # S5b ★已实现: 复用(net,x) CUDA-Event hook
    #      逐叶子算子 ms → 聚合 by_b1_search_knob / by_b2_quant_unit (C.1+C.3)
    #      cpu→estimated_cpu, 占用高→ok_but_contended; 全量明细落 results/stage1_latency_<m>.json
    #      C.2 Amdahl(f_fixed/s_max) + C.4 bound/axis_hint 待补 (需完整e2e + roofline)

    # ── 汇合 + 落盘 ──
    manifest = assemble(model, cap, b1, b1k, b2, d, dseg, stats, checks, lat)  # §4 汇合
    write_yaml(manifest)                                # S6 → searcher_v0 消费
```

---

## 7. 取舍与增量升级位

| 决策 | v1 选择 | 理由 / 升级位 |
|------|---------|--------------|
| 分组算法 | 复用 DepGraph, 不自研 DFS | 已在 Pyramid 验证(trace+剪+forward 通); UPAQ DFS 与 dependency group 功能等价 |
| 剪枝对象 | 仅 `channel` | 2:4/element 拿不到真加速, 已是项目教训 |
| 硬件扫描 | 复用 `capability_schema.py` + YAML 实例 | rtx4090.yaml 已是合格产出; Orin 各补一份 |
| 剪枝搜索粒度 | 43 组整合为 ~5 旋钮(per-stage) | 项目实测剪枝 Pareto 退化, per-group 无收益; 参考 UPAQ/AMC/HALP(§S2b) |
| D 路由决策粒度 | 折叠为连续路由段(2-4 段) | 逐节点是素材非决策; GPU↔DLA 切换有 reformat 成本(§S4b); 精确连续性留 fx(路线 C) |
| trace 边界 | 每模型薄 adapter | 稀疏 VFE / data-dependent 融合无法通用 trace |
| 量化单元粒度 | = 完整依赖组并集 | 保证 B1/B2 视图一致, 不切碎耦合组 |
| 对齐约束强度 | 读 A.alignment_enforcement(hard/soft/auto) | N2v2 实证: Orin implicit padding 抵消对齐 → soft; 4090 暂 hard |

---

## 8. 实现状态 + 与现有资产的关系

> **★ 已实现并跑通 4 模型 (2026-06-12, scan_status 全 ok)**。产物 `framework/partitions/<model>_partition.yaml` + 索引 `framework/partitions/README_scan_results_v1.md`。

- **代码 `framework/stage1/`**: `hardware_scan.py`(A) + `graph_scan.py`(B 模型无关核 S1-S6 + S2b/S4b/**S5b**) + `adapters.py`(4 trace adapter + 注册表) + `latency_profile.py`(**C.1+C.3, 步骤 S5b**) + `run_scan.py`(CLI, 含 `--profile-latency`)。
- **A 侧复用**: `framework/capability_schema.py` + `configs/hardware/rtx4090.yaml`(真实例)。`hardware_scan.HwCapability` 吸收 schema 字段名差异。待补: Orin AGX/Nano 实例 + `probe_hardware`/`measure_baseline`。
- **B 侧原型**: `tools/configurable/depgraph_pyramid.py`(`PyramidFullTraceNet`/`build_dependency`/`module_bucket`/`run_global_prune`) + `depgraph_v2xvit.py`(`V2XViTBackboneTraceNet`) 上提为模型无关核, 原脚本退化为 adapter。
- **消费方**: `framework/searcher_v0.py` —— 待改造为读 (yaml + manifest 整合视图 + latency_profile), 不再硬编码模块名/上下界。
- **实测结果(A+B)**: codriving 20组→4旋钮 / pyramid_lidar 43→5 / pyramid_camera 46→6 / v2xvit 23→4; D 路由 4090 全 1 段。
- **🟡 子流程 C 部分实现 (2026-06-15)**: ✅ **C.1+C.3 已落地** `framework/stage1/latency_profile.py`(CUDA-Event forward hook 逐叶子算子计时 → 聚合 `view_latency.by_b1_search_knob/by_b2_quant_unit`), 作 `scan()` 步骤 S5b 内建; cpu→`estimated_cpu`(非代表性, 仅链路验证), cuda 空闲→真测。⬜ **C.2 Amdahl(f_fixed/s_max) + C.4 bound/axis_hint 待补**(需测含 voxelize+NMS 完整 e2e + roofline)。⏳ **真测 cuda 数据待补**: 实现已跑通(cpu smoke-test ok), 但 4090 卡 0-7 当前全占用, 需空闲卡(util 0%/mem≤50MiB)重跑 `--device cuda --profile-latency on` 填真值。可复用资产: CLAUDE.md 的 NMS 72.4%/E_headroom roofline/backbone memory-bound 等片段 → C.2/C.4 系统化时归到 B 单元上。

---

## 9. review 决策点 (★多数已实现, 待用户确认/调整)

| # | 决策 | v1 现状 | 待确认 |
|---|------|---------|--------|
| 1 | D 路由粒度 | ✅ 折叠为连续段(S4b), 4090 退化 1 段; fx 精确拓扑延后 | 是否接受段级 + fx 延后 |
| 2 | 剪枝搜索粒度 | ✅ per-stage 整合(S2b), 43→~5 旋钮 | tying 粒度: bev_encoder 是否还要更细/更粗 |
| 3 | trace_adapter 形式 | ✅ 每模型 ~20-40 行 Python adapter | 接受 Python? 还是要声明式 YAML |
| 4 | 硬件扫描自动化 | ⬜ 半人工(直接读已写好的 YAML) | `probe_hardware` 要不要自动填表 |
| 5 | 落盘位置 | ✅ `framework/partitions/<model>_partition.yaml` | 位置是否 OK |
| 6 | 量化单元桶粒度 | ✅ 5 桶(backbone/bev_encoder/neck/heads/other) | backbone 是否随 B1 也按 stage 分 |
| 7 | pyramid_camera | ⚠️ 仅 OPV2V ckpt(无 DAIR) + aligner 冻结 | 是否需补 DAIR camera ckpt / LayerNorm-aware 剪枝 |
| 8 | **子流程 C 延迟刻画** | ⬜ 设计完(§3.5), **待实现+目标硬件实测** | (a) profile 手段定 trtexec layer profile 还是 torch profiler? (b) Amdahl `f_fixed` 是否含 voxelize/NMS(口径)? (c) 先在 4090 测还是直接 Orin? (d) `axis_hint` 的 roofline 阈值取经验值还是实测带宽 |
