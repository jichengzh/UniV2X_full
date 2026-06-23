# CLAUDE.md — 项目 onboarding (M4.8 真加速 phase)

> **当前 session 任务**: M4.8 工程 — 给 PyramidFusion 拿到**真实测的 INT8 + structural pruning 加速**, 替换 M4.6 的 mask-based + fp16-proxy prototype.
>
> **关键文档**:
> - `paper_learning/2. AAAI最终故事/00_故事评估与实验路线_v1.md` — 项目状态总览 (v2.5)
> - `paper_learning/2. AAAI最终故事/reflection_mistakes.md` — **必读**: 22 条反思 (4 方向 + 9 判断 + 6 工程 + 3 流程)
> - `paper_learning/2. AAAI最终故事/协同加速框架_工作流_v1.5.md` — 方法论定义

---

## 〇、最新进展与勘误 (2026-06-02, ★接手先读这条)

> 本项目已从 M4.8 单点加速演进到 **multi-agent 框架数据采集**阶段。**当前事实源 = `multi_agent/`**:
> - `multi_agent/background/00_研究目标与实验档案_v1.md` — 总纲 + 实验数据可信度档案 + 数据现状基线
> - `multi_agent/data/dataset_v2.{csv,parquet}` + `schema_v2.md` — 统一主表 (33 完整点 lat+AP)
> - `multi_agent/methods/design/dims_quantization_v1.md` + `dims_pruning_v1.md` + `dims_hardware_v2.md` — 量化/剪枝/硬件可配置维度(文件名 v1 的内容也已是最新定论; 旧 dims_hardware_v1 在 `multi_agent/archive/`)
> - `multi_agent/figure/` — 数据描述性统计图 (可复跑)

### 本次会话已验证的勘误 (旧假设是错的, 别再被误导)
1. **DLA INT8 在 Pyramid 上不可行**: 实测 DLA0/DLA1 × INT8 = **0/12 build 成功**(kDIRECT_IO + bank 超限); DLA FP16 仅 8/12 部分成功 (比 GPU FP16 快 ~22%)。yaml 里 `int8_dla*_build_success:true` **只对 ResNet50 成立, 不能外推到 Pyramid**。→ Pyramid 的 DLA 路由实际是 FP16-only。
2. **并行调度维度大多"仅声明未测"**: multi-stream / GPU+DLA 并行(dual)/ 跨 agent 流水 都 **0 latency 实测**(GPU+DLA build 通但 lat=NaN)。CUDA Graph 实测仅 **1.05-1.10×**(非数量级)。
3. **per-stage 强制混精量化: AP 有正向点但 Pareto 被 TRT-auto 支配**。根因: **TRT 逐层精度是延迟驱动, 不是精度驱动**(它不测 AP, 按 tactic+reformat 总延迟选层; 保 FP16 是"INT8 不划算"的副作用)。框架应用 TRT-auto, 别枚举手工 per-stage。详见 `dims_quantization_v1.md §二.5`。
4. **DepGraph 全网剪枝已打通 + 收敛实测确认是可用维度**(`tools/configurable/depgraph_pyramid.py`): deblocks/shrink **实测可剪**(之前"只 backbone 可剪"是错的分析)。**收敛 finetune 后 DAIR val 1789 真测**(`results/P1_wholenet_prune_real.csv`, `wholenet_converged_ap.json`): backbone_only(2.58M)/ wn_light(2.20M)/ wn_p50(1.86M)/ **wn_aggr(1.32M, -74.7%)**, lat_fp32_p50 4.40→3.091ms 单调降(★2026-06-05 勘误: 旧写 3.18 与 P1 csv 不符, csv 为准)。
   **★[2026-06-02 勘误 — "AP50 全平 0.75"是假象, 别再被误导]**: `P1_wholenet_prune_real.csv` 的 AP50 列只保留 2 位小数, 把真实信号抹平成 "0.75"; 且 wn 五行**共享同一 backbone=[32,64,128], 行间只动 neck(deblocks/shrink)**, 而 AP 由 backbone 通道主导 → neck 几乎不携带 AP 信号(代价 <0.01 AP50)。**金标准 `stage_a_ap_real.parquet`(4 位真值, 已 finetune)显示 backbone 剪枝 AP 是单调下降的**: AP50 0.791→0.777→0.764→0.757(base→p75, span 0.034), AP70 0.631→0.530(span 0.101), 信号 ≈30× 管线噪声(~0.001)。⇒ **剪枝在已测区间 AP 有微弱但可分辨的单调信号, 不是"无损"**。"wn_aggr 严格支配 backbone_only" 只在 (lat, AP50@2dp) 平面成立, 本质是 **Pareto 退化症状**(0–75% 全在 AP 高原缓降, **尚未踩到精度悬崖**), 不能解读成"剪枝精度无损"。下一步必须补 80/87/93% 找悬崖点, 把退化 Pareto 变真曲线; Pareto 的 AP 轴用 **AP70**(信号最强)。注: 昨夜非单调 AP(0.73/0.75/0.76)是欠拟合噪声, 收敛后消失。图 `multi_agent/figure/data/fig5_wholenet_prune.png`。
5. **[ckpt 陷阱]** `output/doe_dataset_v1/pruned_rtx4090_*/` 的剪枝 ckpt 是 `{"model_state_dict":...}` 包裹格式, HEAL `load_saved_model(strict=False)` 加载会**全 key missing → 从随机权重 finetune**。从剪枝权重 resume finetune **必须先转 flat state_dict**。(`Pyramid_DAIR_m1_pruned{25,50,75}_2026_05_10` 是 flat 正确格式; stage_a AP 用的就是它, 安全。)
6. **流水线调度受 Amdahl 限制**: 必须先解串行 CPU 卡点 (NMS 占 e2e 72.4%, 已 CUDA 化 → 3.04×) 调度才有效; 跨 agent fuse 是同步 barrier = 并行收益硬上限。详见 `dims_hardware_v2.md`。
7. **[2026-06-02 剪枝悬崖 — 两轮真测定论: 可达范围内无悬崖, 剪 pyramid_backbone 几乎"免费"]**: 第一轮 prune85/90/95(wpg=16)+ 第二轮 cliff2_a/b/c(**wpg=4**, 可剪 pb 压到 **0.36M = 比 base 3.76M 砍 ~90%**)收敛真测, **AP50 全稳在 0.74–0.75 / AP70 0.57–0.60**(`results/ap_cliff{,2}_converged.json`), 差异在 finetune 噪声(±0.01)内。**没有精度悬崖**; "AP 由 bottleneck 宽度(wpg)主导"假设也被 wpg=4 推翻。根因 [第三轮研究订正, 我之前归因错]: AP 下限 **不是"被固定 encoder_m1/backbone_m1 撑底"**(那是错的), 而是 **DAIR 任务 + Pyramid head 的固有 AP 上限** —— **base 全模型 AP50 本来就只 0.791**, cliff 仅差 ~0.04 = 剪枝近无损。参数: pyramid_backbone 68.9% / shrink_conv 26.8% / backbone_m1 4.1% / encoder_m1 **0.02%**(DAIR 下仅一层 Linear(10→64), 非重型 VFE)。**encoder_m1/backbone_m1 既可剪(backbone_m1 标准 ResNet, 只是太小不值)也非 AP 主因**; 没悬崖真因 = **模型对 DAIR 严重过参数化**。⇒ **剪 pyramid_backbone Pareto 退化(剪到底即最优)**, 与"per-stage 混精被 TRT-auto 支配"同类。真正可搜 AP-trade-off 在量化 INT8 边界 / 更难任务/模型, 不在剪枝率。详见 `dims_pruning_v1.md §8.4–8.5`。
8. **[2026-06-02 Pareto 维度扩展 — 用户拍板] 吞吐 + 能耗升为一级指标**: 吞吐与延迟**都是硬实时约束**(之前"边缘只关心延迟/吞吐无价值"是错的: RSU 多车/路测是 throughput-binding)。能耗(perf/watt, J/frame)从"括号"升一级轴。**预测器预测全 5 指标 {AP, latency, throughput, energy, model_size}; 选择器 regime 条件切目标**, 默认主前沿 (AP, latency, energy), throughput≥FPS下限+size≤显存预算 作约束, RSU regime 切 throughput 为目标。**加 batch 轴**(空间维=RSU 多车纯吞吐净赚, 时间维=拿延迟换吞吐)是 throughput 脱离"=1/latency 冗余"的唯一途径。**FPGA = 仅 related-work**。详见 `background/00_研究目标与实验档案_v1.md` + `dims_hardware_v2.md`。
9. **[2026-06-02 硬件真测] INT8 真省能耗; 单 GPU pipeline 并行已证伪(只有异构 GPU+DLA 有戏)**: ① E4 真测(4090): INT8 比 FP16 省 **30–52% J/frame**(功率更低+延迟更短复合; `results/E4_energy_4090.csv`)。② E1 multi-stream(data 并行)仅 1.13× 吞吐、单帧延迟劣化。③ E_headroom roofline 显示 backbone memory-bound 欠饱和(~30% 带宽)、stage 互补、余量 0.53–0.63 —— **一度暗示** 单 GPU pipeline 能把 0.13 推向 0.5+。④ **但 E_pipeline 实测定夺推翻了 ③**(`results/E_pipeline_singlegpu_4090.csv`, 数值校验 maxdiff=0): 单 GPU stage 流水峰值仅 **1.08×(3 流), 还低于 E1 的 1.13×**; 最互补的 memory∥compute 对也仅重叠 1.11×。⇒ **roofline 0.53–0.63 是不可达的乐观上界; occupancy 槽位 ≠ 可并发吞吐空隙; 单 GPU 上 pipeline 相对 data 并行无独有价值, 降为已证伪的消融对照**。唯一仍有上行的形态 = **不相交物理资源: 异构 GPU∥DLA(Orin E3)/ GPU∥CPU**。教训: roofline 余量是乐观上界, 必须实跑验证。
   **★[E3 Orin 真测 — dual_A 的 NaN 已跑出真数, 用户的 pipeline 想法在异构上成立]**: Orin DLA0(stage01 14.49ms)∥DLA1(stage2 6.18ms): **进程内双流 = 20.65ms(1.00× 完全串行, TRT 8.5 进程内序列化 DLA 提交)**, 但 **双进程 = 15.4ms(1.34× 真硅片并行)**(`results/E3_orin_dla_pipeline.csv`)。⇒ 异构 GPU∥DLA 重叠真实(1.34×)**但只有多进程编排拿得到**; 这是 stage 流水唯一真生效的形态(vs 单 GPU 1.08×)。★[2026-06-05 勘误: 旧 caveat "跨进程帧交接未建"已过时] E3 csv 实有 `real_xproc_shm_handoff` 真测: 016_032_064 拆分 interframe 16.05ms(真重叠 1.29× vs 串行和)/单帧 e2e 23.71ms/handoff 仅 0.97ms(1.0MB fp16); **但 032_064_136 拆分是负结果**(interframe 32.17ms = 0.91× 比串行还慢, 大 stage2+2MB handoff 撞 LPDDR5 带宽) ⇒ 跨进程流水收益依赖 stage 切分与 handoff 体积, 非普适。
   **[单 GPU pipeline 文献调研补充]**: 1.08× 与文献一致(naive 并发 ~1.13×); 对位更优法 = Opara(CUDA Graph+算子资源感知 overlap, conv-dense 预期仅 1.1–1.3×); MPS/Green Contexts/REEF/HFTA 均多进程/多模型不适用单模型 stage 流水。物理精修: memory-bound stage 撞 **L2/DRAM 带宽墙**(非仅 SM occupancy)。

### 数据现状基线 (截至 2026-06-02)
- **完整点 (lat+AP) = 33**, 全 PyramidFusion/4090, latency_kind=`body_subnet_collab2` (双 agent body, ≠ e2e)。
- **D 维 (硬件) 有 ~140 行真测但未并入主表**: `4090_dspace`(48)/`orin_dspace`(72, 含 DLA 路由)/`cudagraph`(12)/`tactic_workspace`(8), latency-only。
- 多模型: 仅 baseline_4090 含 uniad_tiny(35)/univ2x_full(22); F-Cooper/AttFuse/V2X-ViT 仅 P0 分段计时。

### 行为提醒 (本会话反复确认)
- **不轻信 agent 自我报告** — 凡"实测/已修/已build", 主控必复跑/读文件/git diff 核验 (本会话多次抓到谎报 GPU 占满、dry-run 冒充实测、提前 offload)。
- **latency 必须在完全空闲 GPU (util 0%/mem≤50MiB) 测**; GPU 0-4 常被他人(wuyuegao)训练占用, 跑前必 nvidia-smi 确认。
- 区分 **真测 / 估算 / 仅声明**; 区分 latency 口径 (single-agent body / collab2 body / e2e / raw-PyTorch, 不可混比)。

---

## 一、为什么有这个 task (背景)

之前 M4.6 (commit `744dfb3`-`a848e01`) 跑了 16+15 = 31 个 Pyramid candidates "加速实验", 但全部用 **mask-based pruning** + **`int8_proxy_fp16`** (FP16 autocast 替代 INT8). 实测 lat 全在 25-28ms 范围 ≈ FP16 baseline, 没差异化.

用户原话指出:
> "INT8 之后没有加速你就在结果表格里直接写 fp16 的实验结果? 你要找到原因修改并直到实现预期的加速效果. PyramdFusion 没有 DCN 这种瓶颈, 你应该想方法用剪枝和量化实现预期的加速."
>
> "搜索空间才能构建起来, 不然所有的优化方案, 速度一样, 精度一样, 我们还需要搜索呢."

详细反思见 `paper_learning/2. AAAI最终故事/reflection_mistakes.md`.

---

## 二、立即要做的工作 (Phase A → B → C)

### Phase A — TRT INT8 真实测 (1-2 day, P0)

**目标**: 给 framework 论文级 INT8 加速数据点 (期望 INT8 vs FP16 真加速 1.5-2×).

#### A.1 PyramidFusion → ONNX export

文件参考: `/home/jichengzhi/heal_research/HEAL/opencood/tools/inference.py` (HEAL inference 主流程)
模型 (★DAIR 现役金标准): `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/`
> ⚠️ [2026-06-03 supervisor 勘误 ISS-012] 旧文档此处写 generic `Pyramid_m1_base_..._04_28_12` 是 **OPV2V 版**(盘上也存在但是不同模型); 当前 DAIR AP 金标准 `stage_a_ap_real.parquet` 全部用 DAIR 版 `..._11_42_29`。在 DAIR val 上工作必须用 DAIR 版, 误取 04_28_12 会得跨数据集脏 AP。

挑战:
- 完整模型含 `encoder_m1` (PointPillar VFE, sparse op), 可能 unsupported in ONNX
- 备选: 只 export `pyramid_backbone` 子模块 (3.79M params), 用 dummy `(1, 64, 256, 256)` 输入

建议步骤:
1. 先 export 子模块 ONNX (确保 export 通)
2. 测 ONNX runtime 输出 == PyTorch 输出 (numerical sanity)
3. 再尝试完整模型 export, 如失败再 work around (unsupported op fallback to PyTorch)

输出: `models/pyramid_backbone_fp32.onnx`

#### A.2 ONNX → TRT FP16

```bash
trtexec --onnx=models/pyramid_backbone_fp32.onnx \
        --saveEngine=models/pyramid_backbone_fp16.engine \
        --fp16 --workspace=2048 \
        --avgRuns=200 --warmUp=200 --duration=10
```

记录 mean/p50/p99 latency. **期望**: 比 PyTorch FP32 4.49ms 快 (TRT 编译加速 + Tensor Cores).

#### A.3 ONNX → TRT INT8 with calibration

calibration data 准备:
- 用 OPV2V test 前 100 samples 喂模型, 拿 input tensor `(1, 64, 256, 256)` 缓存到 `calibration/pyramid_calib.bin`
- TRT calibrator: IInt8MinMaxCalibrator (推荐) 或 EntropyCalibrator2

```bash
trtexec --onnx=models/pyramid_backbone_fp32.onnx \
        --saveEngine=models/pyramid_backbone_int8.engine \
        --int8 --calib=calibration/pyramid_calib.bin \
        --workspace=2048 --avgRuns=200
```

#### A.4 4 档 latency 横向对比

| precision | engine size | lat_p50 | vs FP32 PyTorch (4.49ms) | AP50 实测 |
|-----------|------------|---------|--------------------------|-----------|
| PyTorch FP32 (M4.3 已测) | — | 4.49 ms | 1.00× | 0.96 |
| PyTorch FP16 (M4.3 已测) | — | 3.27 ms | 1.37× | 0.96 |
| TRT FP16 (Phase A.2) | ? | ? | 期望 ~2× | ? |
| **TRT INT8 (Phase A.3)** | ? | ? | **期望 ~3×** | 期望 ≥ 0.93 |

注意: A.4 表格是子模块. 完整 e2e 还要加 voxelize ~10ms, 实际 framework 加速倍数会缩水.

### Phase B — Structural pruning + finetune (3-5 day, P0)

**目标**: framework 协同加速里 prune 维度真实拿到加速 (而不是 mask-based 占位).

#### B.1 OPV2V train 下载

HF `gqk/opv2v` 的 `train_parts/train_part_05-13` 共 9 parts × ~5GB = ~45GB.

注意: train_part_00-04 不存在于该仓库 (只 05-13). 需要确认是否 partial train data 也能 finetune (HEAL 训练 stage1 是否需要全 train split).

#### B.2 Structural channel pruning rebuild

PyramidFusion `num_filters [64, 128, 256]` 每 stage 输出 channel:
- conv1: 1×1, output = 2× num_filters[stage] (groups=1)
- conv2: 3×3 grouped (groups=32), output = 2× num_filters[stage], **必须 multiple of 32**
- conv3: 1×1, output = num_filters[stage]
- deblocks: 上采样 to 128 channels (independent of num_filters)
- shrink_conv: input = 384 (=3×128), output = 384

50% reduce: `num_filters [32, 64, 128]` (除以 2, 都 ≥32 满足 group constraint).

实施:
1. Instantiate fresh `PyramidFusion(cfg_smaller)` (修 num_filters)
2. 加载 baseline ckpt 的 weights, 按 L1 norm 选 top-N output channels (per layer)
3. truncate weights to fit smaller dimensions (注意 conv1 output → conv2 input 一致性)
4. shrink_conv 输入维度不变 (deblocks 都输出 128), 不动

#### B.3 Finetune 1-2 epoch

HEAL 训练命令 (参考 README):
```bash
cd /home/jichengzhi/heal_research/HEAL
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid_smaller.yaml \
  --model_dir opencood/logs/Pyramid_m1_base_pruned50
```

预算: 4090 上 ~10-20h training (1-2 epoch).

#### B.4 评估 pruned model AP

跑 HEAL inference.py on OPV2V test 2170 samples, 期望 AP50 ≥ 0.92 (vs baseline 0.9635).

### Phase C — 综合 INT8 + pruning (0.5 day, P0)

#### C.1 Pruned model → ONNX → TRT INT8

同 Phase A 流程, 但用 finetuned smaller model.

#### C.2 最终 Pareto 表

| anchor | 配置 | lat_p50 | AP50 | vs FP32 baseline |
|--------|------|---------|------|------------------|
| 1 | FP32 baseline | 35.31 ms | 0.9635 | 1.00× |
| 2 | FP16 baseline (autocast) | 26.70 ms | 0.9631 | 1.32× |
| 3 | TRT INT8 baseline | ? | ? | 期望 2-3× |
| 4 | 50% L1 pruned + finetune + FP16 | ? | ≥ 0.92 | 期望 1.8-2.5× |
| 5 | **50% L1 pruned + finetune + INT8** | ? | ≥ 0.92 | **期望 3-5×** |

5 个真实测 anchor 才能给 framework 真实 Pareto 前沿.

---

## 三、关键资源

### 3.1 4090 host (当前 cwd)

- repo: `/home/jichengzhi/UniV2X` (git branch: `hw-deploy-d-space`)
- HEAL repo: `/home/jichengzhi/heal_research/HEAL/`
- Pyramid baseline ckpt (★DAIR 现役金标准): `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/` (旧写 `..._04_28_12` 是 OPV2V 版, 勿在 DAIR 上用; 见 ISS-012)
- OPV2V test (已下) 18GB: `/home/jichengzhi/heal_research/dataset/OPV2V_orig/extracted/test/`
- OPV2V train (待下): 同源 HF `gqk/opv2v` `train_parts/train_part_05-13`
- conda env: `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python` (Python 3.9)

### 3.2 Orin AGX (172.16.62.222)

- **SSH 用户名**: `jichengzhi`
- **密码**: 用户口头给 (上 session 是 `shuai123`, 但每个 session 该确认)
- TRT 8.5.2.2 at `/usr/src/tensorrt/bin/trtexec`
- DL4AGX dcnv4-trt 已编译: `/home/jichengzhi/DL4AGX/AV-Solutions/dcnv4-trt/build_native/plugins/libDCNv4_plugin.so` (15.6MB, N6 闸门 1+1.5 已通过)
- venv: `/home/jichengzhi/uniad_venv/`

注意: M4.8 的 TRT INT8 build 在 4090 host 上跑就够 (RTX 4090 有 TRT 10.x). Orin 只用于跨平台 latency 验证 (M2 f 函数已拟合, R²>0.995).

### 3.3 已实测真实数据 (不可丢)

`data/baseline_4090.parquet` 83 行, 含:
- M4.6.0 完整 e2e: FP32 35.31ms, FP16 26.70ms, AP50 0.9635/0.9631
- M4.6.1 单模块 L1/FPGM 7 数据点
- M4.6.2 multi-module 4 configs
- M4.6.3 + M4.6.3 v2 共 20 个 candidates 实测
- 三模型 spectrum: pyramid_fusion 26 行 / uniad_tiny_variant 35 行 / univ2x_full 22 行

---

## 四、模型 (LGB 预测器)

- `models/lgb_v6_amota.txt` — 跨模型 amota 预测器. **★[2026-06-04 勘误 ISS-026] Pyramid 段的 "amota" 实为 AP 错填**(PyramidFusion=HEAL 单帧协同检测, 无跟踪头, 物理上不产真 AMOTA; baseline_4090 里 pyramid "amota" 0.25-0.96 = 已知 AP50 值, row65 amota=0.9635=M4.6.0 AP50). 故该预测器 **pyramid 段是拿 AP 当 AMOTA 拟合**("in-sample MAE 0.010" 衡量的是 AP 拟合, 非 AMOTA); **跨模型 amota 结论勿含 pyramid**(把 pyramid AP 量级 0.25-0.96 与 univ2x/uniad 真 AMOTA 0.02-0.38 当同一指标 → "model_class 效应"是指标定义混淆伪信号). **真 AMOTA 只能来自有跟踪头的 UniV2X 家族**(univ2x_full / univ2x_coop_tiny, plan_b+spd_evaluator 真测).
- `models/lgb_v6_latency.txt` — **崩**, 跨数量级训练给负值, 不用. 改 rule-based 公式 (见 `scripts/phase1/m4_7_v3_pyramid_pareto.py` 中 `rule_based_lat`)
- `models/lgb_v5_1_amota.txt` — 旧版, 不再用

---

## 五、必读 commit 历史 (上下文)

按时间倒序:
1. `9b3a012` v2 反思 — 加速本身才是目标 + INT8 用 FP16 替代是 reporting failure
2. `94d5c0b` v1 反思 18 条 — 第一轮工程错误清单
3. `a848e01` Phase 3+5 — M4.6.3 v2 抽样 15 候选实测 + 三模型整合 + 修复报告
4. `5f68b21` Phase 1+4 — 三模型 Pareto v2/v3 (LGB v6 amota + rule-based lat)
5. `99a5dad` Phase 1 — LGB v6 + M4.7 v2 修复 KNN anchor 错 + AP 预测器接入
6. `fa7aaa9` N6 DCN v4 闸门验证 (Orin 实地)
7. `744dfb3` M4.6.1+2+3 完成 — 16 个真实测试点 + framework 限制暴露
8. `d539468` M4.6.0 完成 — Pyramid 完整 e2e FP32 vs FP16 (1.32× 实测加速)
9. `b6b1760` M4.5 完整版 — Pyramid baseline AP 自测 (AP30/50/70=0.97/0.96/0.93)

---

## 六、行为指南 (来自 reflection)

### MUST do

1. **真测就是真测**: INT8 一定要走 TRT INT8 build, 不能用 fp16 autocast 替代. 拿不到真测就明说 "未实测", 不是用 proxy 替代后当真测报告.
2. **承认工程量**: 5-7 day 工程不是 6h prototype. 不要再用 "高 ROI" 的借口 retreat 到 prototype.
3. **每个加速倍数都标真实测 / 估算 / 预测**: 表格里 lat 数字必须明确来源 (PyTorch CUDA Event / TRT trtexec / KNN / LGB / rule-based).
4. **AP 损失必须有 finetune**: 高剪枝率的 AP 实测必须配 finetune, 否则 AP 暴跌不能用作论文数据.

### MUST NOT do

1. **不要用 mask-based pruning 测 latency** — 浪费时间 (24 小时已浪费在 M4.6.1+2+3).
2. **不要把 fp16 proxy 当 INT8 报告** — 这是 reporting failure.
3. **不要在 inference 之外的 timing 路径里加 post-process** — 已踩过的坑 (M4.6.0 v1 timing 含 NMS, 跟 v2 不公平).
4. **不要跳过 OPV2V train 下载** — finetune 是 P0 必需, 别再 skip.

### 沟通规范

1. 所有 latency 数字标 caveat (口径 / 数据源).
2. 所有"加速倍率"必须 vs 明确 baseline (PyTorch FP32? PyTorch FP16? TRT FP16?).
3. 不确定的地方明说"未测/估算", 不假装知道.
4. 报告时 honest 列出 framework 真实状态 + 限制, 不堆 ✅ 摆样子.

---

## 七、当前 git 状态

- branch: `hw-deploy-d-space`
- 上一次 commit: `9b3a012` (反思 v2)
- 未追踪/未 commit 文件: 见 `git status`
- 主分支: `main` (远程 PR)

下一个 commit 该是 Phase A.1 — `feat(m4_8): PyramidFusion → ONNX export + sanity check`.

---

## 八、问题先确认

接 session 时 **首先确认** 以下事:

1. 用户是否还在做 M4.8 (TRT INT8 + structural pruning + finetune)? 还是切到其他 task?
2. Orin SSH 密码是否还是 `shuai123`?
3. OPV2V train 下载是否要在 4090 host (`/home/jichengzhi/heal_research/dataset/OPV2V_orig_train/`)? 是否有磁盘空间 (df -h)?
4. finetune 是否能用 partial train (train_part_05-13, 缺 00-04)? 不行的话能否找到完整 OPV2V train 来源?

确认后启动 Phase A.1.
