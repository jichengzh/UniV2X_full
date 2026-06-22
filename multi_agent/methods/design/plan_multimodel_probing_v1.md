# 计划二:把 stage1 做成全自动扫描器, 探测更多算法以证明其普适有效 (v2, 2026-06-22)

## §0 接手须知 (新窗口冷启动 —— 直接从这里开始, 自包含)

**这份计划是可执行交接, 不需要前序对话上下文。** 先读 §0, 再按 §3 Phase A0 起跑。

**核心目标 (★别跑偏)**: 证明 **stage1 是一个自动扫描器** —— 给 (模型构造器, ckpt,
一个真实输入), stage1 **自动**追踪依赖图、**自动**跳过不可追踪算子(稀疏 VFE / 自定义
fusion)、**自动**定位稠密 conv 核、**自动**产 manifest + 耦合判决。**不读人家源码、不手写
稠密核 forward**。能 push-button 扫通 N 个差异化模型 = stage1 真普适有效。

> ⚠ **v1 的方向错误 (已纠)**: v1 把"每模型读源码手写 `TraceAdapter._Net` 包装"当成主工。
> 那恰恰**不能证明 stage1 自动有效**(只证明能手工适配), 与已删的 `QLookupCoDriving` 子类
> 同病。v2 改为: 先把 stage1 做成自动扫描器, 再 push-button 探测。

**前置状态 (已就绪)**:
- bridge 已 4/4 验收(memory `project-stage1-bridge-construction`): `framework/stage1_bridge.py`
  + `framework/stage1/{run_scan,graph_scan,hardware_scan,adapters}.py` + `framework/partitions/*.yaml`。
- **当前 adapter 是手写的**(`framework/stage1/adapters.py`: 每模型一个 `_Net` 包装类手挑稠密
  核 + 手列 `skipped_modules`/`ignored_layers`)—— 这正是要自动化掉的东西。
- e2e profiler 模板: `scripts/phase2/profile_v2xvit_e2e_breakdown.py`。
- 已探 5 模型(手写 adapter): `results/coupling_dispatch_report.json`。

**本计划第一步 = Phase A0: 建通用 auto-scan + 证明它复现 4 个手写 adapter**(见 §3）。
**判定 stage1 有效的金标准 = auto-scan 在 Pyramid/CoDriving/V2X-ViT 上自动产出的 manifest,
与现有手写 adapter 的 manifest 一致**(耦合判决/旋钮/对齐相同)。复现成功 ⇒ 手写 adapter 里
的知识本就是可自动发现的 ⇒ stage1 是自动的。然后 push-button 探新模型(F-Cooper/AttFuse...)
扩展证明。

**环境 + 纪律 (硬约束)**:
- 真仓库 `/home/jichengzhi/V2X`(**绝不用** `/home/jichengzhi/UniV2X` 断链空壳)。
- conda python `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`; HEAL 模型须 `sys.path`
  加 `/home/jichengzhi/heal_research/HEAL` + `os.chdir(HEAL_ROOT)`(见 profiler 模板)。
- latency **只在空闲 GPU(4/5/6)测**(`nvidia-smi` 先确认); eager vs 编译口径分清。
- **不轻信自报**必复跑; **仅在用户明确要求时 commit**。克隆新模型到 `multi_agent/model/<name>/`。
- **与计划一接口**: 探测产出的瓶颈剖面圈定计划一(transformer)适用集; 见
  `plan_transformer_into_framework_v1.md`。

---

## 1. 为什么 v1 错了 + v2 的正确口径
"通过框架探测"= 把 stage1 当探测器**对准**模型自动扫描; 不是人读源码替每个模型手写
稠密核包装。后者下"探测 N 个模型"的工作量 = N×(读源码+写 adapter), 且证明的是"我们会手工
适配", 不是"stage1 自动有效"。**v2: 自动化掉手写部分, 让每模型的人工输入压到最薄的加载样板
(构造器+ckpt+一个输入样本), stage1 自动完成稠密核隔离→依赖图→manifest→耦合。**

## 2. 通用 auto-scan 设计 (Phase A0 的产物, 一次性建)
新增 `framework/stage1/auto_trace.py`(或扩 adapters.py 加一个 `AutoTraceAdapter`),
取代逐模型手写 `_Net`:

1. **自动定位稠密核 + 自动跳过不可追踪算子**: 给全模型 + 真实样本输入, 前向时挂 hook 记录
   模块执行序与张量 shape; 试用 torch_pruning `DependencyGraph.build_dependency`; 对触发异常的
   模块(稀疏 spconv VFE / 自定义 com_mask fusion / grid_sample 等)**捕获→自动加入 skip→重试**。
   稠密核 = 最大可追踪的 Conv/BN/ConvT 连续子图(从首个 dense conv 到 heads)。
2. **自动识别 heads**: 终端、小 out-channel、产 cls/reg/dir 的 conv → 自动进 `ignored_layers`
   (不剪), 无需手列。
3. **自动语义分桶**: 复用现有 `_generic_bucket` / `_first_conv_in_channels`(已是自动)给组分
   backbone/bev_encoder/neck/heads。
4. **每模型薄胶水 (registry 条目, ~10 行, 不读源码不写 forward)**:
   `{name, build_fn(从 config 造模型), ckpt_path, sample_input_fn(从 dataloader 取或 dummy)}`。

> 诚实边界: 完全无胶水不现实(模型如何 from-config 构造 + ckpt 路径是模型身份, 必须给)。
> v2 的目标是把胶水压到"身份样板", **消灭"读源码手写稠密核 forward / 手列 skip·ignored"**。
> 极少数模型若稠密核入口实在无法自动判, 才回退到"声明入口 submodule 名"(一行), 并如实记录。

## 3. 分阶段计划

### Phase A0 — 建 auto-scan + 复现 4 个手写 adapter (★证明 stage1 自动有效的金标准)
- 实现 §2 的 `auto_trace`。
- **复现验证**: 对 Pyramid_lidar / CoDriving / V2X-ViT / Pyramid_camera 用 auto-scan(只给薄胶水)
  跑出 manifest, 与现有手写 adapter 的 `framework/partitions/*.yaml` **逐项比对**(B1 组数/对齐/
  `int8_buildable_align`/耦合判决一致)。一致 ⇒ **手写知识可自动发现 ⇒ stage1 是自动的**。
- 产物: `framework/stage1/auto_trace.py` + `results/autoscan_reproduce_check.json`(4 模型 diff)。

### Phase A1 — push-button 探 F-Cooper + AttFuse (ckpt 在手, 验证对新模型零源码surgery)
- ckpt(真, 已核): `/home/jichengzhi/heal_research/checkpoints/baselines_hf/`
  `HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10`(config.yaml + net_epoch17.pth) /
  `..._attfuse_2023_08_06_19_58_00`(config.yaml + net_epoch19.pth)。
- 只填薄胶水(build_fn 用 HEAL `train_utils.create_model(hypes)` + ckpt) → auto-scan → manifest
  → `bridge` 耦合判决。**期望 push-button 通, 预期 SEPARABLE**(标准 BaseBEVBackbone)。
- 同时复用 profiler 模板测 e2e 瓶颈剖面(预期 conv-瓶颈: F-Cooper fusion 1.4ms / AttFuse 3.5ms)。

### Phase A2 — push-button 探 V2VNet / Where2comm / DiscoNet (HF zip ckpt, 解压即扫)
- 解压 `baselines_hf/*.zip` 或 HF 拉取; 薄胶水 → auto-scan → 判决 + 瓶颈。
- **V2VNet 重点**: 预期第二个 fusion-瓶颈样本(GNN 28.6ms), 与 V2X-ViT 一起佐证"fusion-瓶颈非
  transformer 独有"+ auto-scan 在 GNN/注意力 fusion 上能否自动跳过不可追踪部分(压力测试)。

### Phase A3 — 跨模型耦合谱 + 证明陈述
- 汇 N 模型: (e2e 瓶颈位置 × 耦合判决 × 空间规模 × **每模型胶水行数**)成谱。
- **证明陈述**: "stage1 对 N 个差异化协同感知模型 push-button 自动扫描(每模型胶水 ≤K 行, 零源码
  surgery), 自动复现 4 个先前手写 adapter, 并自动区分 conv-分组耦合 / conv-可分离 / fusion-瓶颈
  三类。" 产物: `results/cross_model_coupling_spectrum.json`。

## 4. 现实 inventory (实测可行性分级; 探测=auto-scan+薄胶水, 非写 adapter)

| 模型 | OpenCOOD 模块 | ckpt 现状 | 历史 fusion 计时(OPV2V, 待复测) | 预期瓶颈 | 分级 |
|---|---|---|---|---|---|
| **F-Cooper** | `fusion_in_one`(maxpool) | ✅ baselines_hf opv2v(+DAIR zip) | 1.41ms (0 fusion 参数) | conv-瓶颈 | **A1 立即** |
| **AttFuse** | `fuse_modules/self_attn.py` | ✅ baselines_hf opv2v | 3.49ms (SDP, 0 参数) | conv-瓶颈 | **A1 立即** |
| **V2VNet** | `v2v_fuse.py` | ⚠ HF zip(OPV2V) | **28.56ms (6.55M GNN!)** | **fusion-瓶颈** | **A2** |
| **Where2comm** | `where2comm_attn.py` | ⚠ HF zip | 6.69ms (0.40M 稀疏注意) | 中(待测) | **A2** |
| **DiscoNet** | `point_pillar_disconet.py` | ⚠ 核 ckpt | GNN 蒸馏 | 中-重(待测) | **A2** |
| **Who2com/When2com** | `when2com_fuse.py` | ❌ HEAL 无 ckpt | — | 注意-门控 | **A3 需训练/外源** |
| **UniV2X 家族** | V2Xverse(univ2x_full) | ⚠ ckpt 待定位 | — | 含跟踪/规划, 重 | **A3 定位 ckpt** |
| Pyramid / CoDriving / V2X-ViT | — | ✅ 已探(手写 adapter) | — | 复现基准(A0) | ✅ |

> 历史计时来自 `multi_agent/model/model_zoo_survey_v1.md`(OPV2V P0 粗测); 预期瓶颈是**假设**,
> 本轮 auto-scan + profiler **必须实测复核**(尤 V2VNet; DAIR vs OPV2V 口径不混)。

## 5. 诚实边界 / 风险
- **auto-scan 的自动跳过是核心风险**: 不可追踪算子(spconv / com_mask fusion / 自定义 CUDA)能否
  稳健"捕获→排除→重试"未验证; A0 复现 4 模型就是第一道压力测试(它们恰好覆盖 sparse VFE + 协同
  fusion + 分组 conv)。复现不过 ⇒ auto-scan 设计要迭代, 而非掩盖。
- **薄胶水不可消灭到 0**: 模型 from-config 构造 + ckpt 路径是身份, 必须给; 目标是消灭源码 surgery
  (手写 _Net forward / 手列 skip·ignored), 不是消灭加载样板。每模型胶水行数要如实计入 §A3 谱。
- ckpt 可得性: A1 两个在手; A2 三个需解压/配 yaml; A3 两个需训练/外源 → 越后越贵。**先交付
  A0(自动化金标准)+ A1(push-button 首证)**, 已足够支撑"stage1 自动普适"的核心论点。
- 真加速/真 AP 一律实测; eager vs 编译口径分清; 跨数据集 ckpt 不混比; 不信自报必复跑。
