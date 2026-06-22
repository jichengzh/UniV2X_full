# 计划二:用框架对更多协同感知算法做第一步探测 (v1, 2026-06-22)

## §0 接手须知 (新窗口冷启动 —— 直接从这里开始, 自包含)

**这份计划是可执行交接, 不需要前序对话上下文。** 先读 §0, 再按 §2 Phase P0 起跑。

**前置状态 (已就绪, 不用重做)**:
- stage1→stage2 bridge **已构造并 4/4 验收**(memory `project-stage1-bridge-construction`):
  `framework/stage1_bridge.py`(SpaceSpec/KnobSpec, `coupling_summary()` 出 COUPLED/SEPARABLE)
  + `framework/stage1/{run_scan,graph_scan,hardware_scan,adapters}.py`(图扫描产 manifest)
  + `framework/partitions/*.yaml`(已有 4 模型 manifest)。
- e2e 瓶颈 profiler 模板: `scripts/phase2/profile_v2xvit_e2e_breakdown.py`(CUDA-event 逐模块,
  改 CKPT_DIR/CONFIG 即可复用; 输出 `results/<model>_e2e_breakdown.json`)。
- 已探 5 模型(Pyramid/CoDriving/V2X-ViT lidar/camera): 见 `results/coupling_dispatch_report.json`
  + memory `project-v2xvit-attention-bottleneck`。

**本计划第一步 = Phase P0: 探 F-Cooper + AttFuse(ckpt 在手, 无需训练)**:
- ckpt(真, 已核): `/home/jichengzhi/heal_research/checkpoints/baselines_hf/`
  `HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10`(config.yaml + net_epoch17.pth) /
  `HeterBaseline_opv2v_lidar_attfuse_2023_08_06_19_58_00`(config.yaml + net_epoch19.pth)。
- **两步**: ① 复制 `profile_v2xvit_e2e_breakdown.py` → 改 CKPT_DIR/CONFIG/CKPT_FILE 跑 e2e
  瓶颈剖面(F-Cooper/AttFuse 也是 HeterBaseline, 模板几乎直接可用); ② 写
  `FCooperAdapter`/`AttFuseAdapter`(克隆 `framework/stage1/adapters.py:246 V2XViTAdapter` 模板,
  调隔离稠密 conv 核的逻辑) → 加进 `adapters.py` 末尾的 `REGISTRY` →
  `python -m framework.stage1.run_scan --model fcooper --device cpu --profile-latency off`
  → manifest → `python -m framework.stage1_bridge` 看耦合判决。预期 SEPARABLE(标准 BaseBEVBackbone)。

**环境 + 纪律 (硬约束)**:
- 真仓库 `/home/jichengzhi/V2X`(**绝不用** `/home/jichengzhi/UniV2X` 断链空壳)。
- conda python `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`; HEAL 模型须
  `sys.path` 加 `/home/jichengzhi/heal_research/HEAL` + `os.chdir(HEAL_ROOT)`(见 profiler 模板)。
- latency **只在空闲 GPU(4/5/6)测**(`nvidia-smi` 先确认 util 0%); 区分 eager vs 编译口径。
- **不轻信 agent 自报**: 凡"已测/已build"必复跑/读文件核验。**仅在用户明确要求时 commit**。
- 克隆新模型代码到 `/home/jichengzhi/V2X/multi_agent/model/<name>/`(已建)。

**与计划一的接口**: P0 瓶颈剖面圈定计划一(transformer)适用集 —— fusion-瓶颈者
(V2VNet/V2X-ViT)进计划一, conv-瓶颈者(F-Cooper/AttFuse)用现状框架。见
`plan_transformer_into_framework_v1.md`。

---

> 目标: 把 stage1 探测(图扫描→manifest→bridge 耦合判决 + e2e 瓶颈实测)推广到
> F-Cooper / AttFuse / Who2com·When2com / Where2comm / V2VNet / DiscoNet / UniV2X 等
> 通用与最新方法。**探测必须实测**(真 ckpt + 真 forward + 真 GPU 计时), 不估算。
> 克隆目标目录: `/home/jichengzhi/V2X/multi_agent/model/`。

---

## 0. "探测一个模型"产出什么 (单模型交付物)
1. **e2e 瓶颈剖面** (实测, CUDA-event): backbone-conv / fusion / encoder / heads / NMS 各占比
   → 判该模型是 **conv-瓶颈**(框架现状适用)还是 **fusion-瓶颈**(需计划一)。
2. **partition manifest** (stage1 图扫描真产出): B1 剪枝组 / B2 量化单元 / D 路由 +
   `int8_buildable_align`。
3. **耦合判决** (bridge `coupling_summary`): COUPLED / SEPARABLE + 逐旋钮 `κ`。
4. **空间规模**: 旋钮数 / 合法宽度 / 压缩比。
⇒ 汇成**跨模型耦合谱**: 哪些像 Pyramid(conv 分组耦合)/ 哪些像 CoDriving(可分离)/
   哪些像 V2X-ViT(fusion-瓶颈)。**这是框架"可普适"的核心证据。**

## 1. 现实 inventory (实测可行性分级, 已核对盘上资源)

| 模型 | OpenCOOD 模块 | ckpt 现状 | 历史 fusion 计时(OPV2V) | 预期瓶颈 | 分级 |
|---|---|---|---|---|---|
| **F-Cooper** | `fusion_in_one`(maxpool) | ✅ `baselines_hf/...opv2v_lidar_fcooper`(+DAIR zip) | 1.41ms (0 fusion 参数) | **conv-瓶颈** | **T1 立即** |
| **AttFuse** | `fuse_modules/self_attn.py` | ✅ `baselines_hf/...opv2v_lidar_attfuse` | 3.49ms (SDP, 0 参数) | **conv-瓶颈** | **T1 立即** |
| **V2X-ViT** | `transformer_fuse.py` | ✅ (已探, 见 v2xvit_e2e_breakdown) | 216ms(DAIR eager) | **fusion-瓶颈** | ✅ 已探 |
| **Pyramid** | `pyramid_fuse.py` | ✅ (已探) | 14.45ms | conv-瓶颈(分组耦合) | ✅ 已探 |
| **CoDriving** | V2Xverse | ✅ (已探) | — | 可分离 | ✅ 已探 |
| **Where2comm** | `where2comm_attn.py` | ⚠ HF zip(OPV2V), 无 DAIR yaml | 6.69ms (0.40M 稀疏注意) | 中(待测) | **T2 需解压/配置** |
| **V2VNet** | `v2v_fuse.py` | ⚠ HF zip(OPV2V) | **28.56ms (6.55M GNN!)** | **fusion-瓶颈** | **T2 需解压** |
| **DiscoNet** | `point_pillar_disconet.py`(+teacher) | ⚠ 需核 ckpt | GNN 蒸馏 | 中-重(待测) | **T2 核 ckpt** |
| **Who2com/When2com** | `when2com_fuse.py` | ❌ HEAL 无 ckpt | — | 注意-门控 | **T3 需训练/外源** |
| **UniV2X 家族** | V2Xverse(univ2x_full/coop_tiny) | ⚠ baseline_4090 有数据点, ckpt 待定位 | — | 含跟踪/规划, 重 | **T3 定位 ckpt** |

> 注: "历史 fusion 计时"来自 `model_zoo_survey_v1.md` 的 OPV2V 分段计时(P0)。预期瓶颈是
> **假设**, 必须 P0 实测确认 —— 尤其 V2VNet(GNN 6.55M)很可能是第二个 fusion-瓶颈样本。

## 2. 分阶段计划 (按 ckpt 可行性分级, 先摘低垂果实)

### Phase P0 — 立即探测 ckpt-在手的两个 (F-Cooper + AttFuse, ~1-2 GPU session)
- 复用 `scripts/phase2/profile_v2xvit_e2e_breakdown.py` 模板(改 CKPT_DIR/CONFIG): 实测
  e2e 瓶颈剖面。**预期确认 conv-瓶颈**(fusion 便宜) → 框架现状对它们适用, 形成与
  V2X-ViT(fusion-瓶颈)的对照。
- 复用 `framework/stage1/adapters.py` 模板写 F-Cooper/AttFuse adapter(它们 backbone =
  标准 BaseBEVBackbone, 比 Pyramid 简单) → `run_scan` → manifest → bridge 耦合判决。
- **预期**: 标准卷积 backbone → SEPARABLE(像 CoDriving)。产物: 2 个 manifest + 瓶颈 json。

### Phase P1 — 跨模型耦合谱初版 (用已探的 5 个 + P0 的 2 个)
- 汇 Pyramid/CoDriving/V2X-ViT/F-Cooper/AttFuse 的(瓶颈位置 × 耦合判决 × 空间规模)成一表。
- **这一步已能支撑论文"框架可普适 + 耦合是架构相关属性"的核心论点**(7 模型谱:
  conv-分组耦合 / conv-可分离 / fusion-瓶颈 三类)。产物: `results/cross_model_coupling_spectrum.json`。

### Phase P2 — 解压/配置 HF zip ckpt 的三个 (V2VNet / Where2comm / DiscoNet)
- 解压 `baselines_hf/*.zip`(或 HF 拉取); 配 DAIR/OPV2V yaml; 写 adapter; 探测。
- **V2VNet 是重点**: 预期第二个 fusion-瓶颈样本(GNN 28.6ms), 与 V2X-ViT 一起证明
  "fusion-瓶颈不是 transformer 独有, GNN 消息传递同样" → 强化计划一的必要性 + 普适性。

### Phase P3 — 需训练/外源定位的 (Who2com/When2com / UniV2X 家族)
- Who2com: HEAL 无 ckpt → 找原作者 release 或在 DAIR/OPV2V 上训(成本高, 末位)。
- UniV2X: 定位 V2Xverse 的 univ2x_full ckpt(baseline_4090 有数据点 → ckpt 应存在), 探测
  其含跟踪/规划的重型 e2e(注意: 真 AMOTA 只能来自这类有跟踪头的模型, 见 reflection)。

## 3. 单模型探测的标准流程 (可复制 checklist)
1. clone/定位 模型代码 + ckpt 到 `multi_agent/model/<name>/`(或复用 heal/V2Xverse 盘上)。
2. 写 `<Name>Adapter`(继承现有 adapter 模板; 隔离稠密 conv 核, skip sparse encoder/fusion)。
3. `python -m framework.stage1.run_scan --model <name> --device cpu --profile-latency off`
   → manifest。
4. `python -m framework.stage1_bridge` 看耦合判决; 写 e2e 瓶颈 profiler(改 CKPT/CONFIG)。
5. **核验纪律**: latency 在空闲 GPU(4/5/6) 测; 区分 eager/编译口径; 不信自报必复跑。

## 4. 与计划一的接口
P0/P2 的瓶颈剖面**直接圈定计划一的适用集**: 凡 fusion 占 e2e 大头者(V2X-ViT/V2VNet/…)
进计划一; conv-瓶颈者(F-Cooper/AttFuse/…)用现状框架。**建议先做本计划 P0-P1**(便宜、
快、产出谱), 再据瓶颈谱决定计划一投入多少。

## 5. 诚实边界 / 风险
- **adapter 是每模型主工**: 各模型 forward 签名/输入不同(multi-agent/com_mask), trace 隔离
  稠密核需逐个处理(4 个已有 adapter 是模板, 但新模型仍要工)。
- **ckpt 可得性是硬约束**: T1 两个在手; T2 三个需解压/配 yaml; T3 两个需训练/外源 → 越往后
  越贵。**先交付 T1+P1(7 模型谱)**, 它已足够支撑普适性论点。
- 历史 OPV2V 分段计时是 P0 粗测, 预期瓶颈必须本轮重新实测确认(尤其 DAIR vs OPV2V 口径不同)。
- 真加速/真 AP 一律实测; eager vs 编译口径分清; 跨数据集(DAIR/OPV2V) ckpt 不混比。
