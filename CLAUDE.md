# CLAUDE.md — 项目 onboarding (M4.8 真加速 phase)

> **当前 session 任务**: M4.8 工程 — 给 PyramidFusion 拿到**真实测的 INT8 + structural pruning 加速**, 替换 M4.6 的 mask-based + fp16-proxy prototype.
>
> **关键文档**:
> - `paper_learning/2. AAAI最终故事/00_故事评估与实验路线_v1.md` — 项目状态总览 (v2.5)
> - `paper_learning/2. AAAI最终故事/reflection_mistakes.md` — **必读**: 22 条反思 (4 方向 + 9 判断 + 6 工程 + 3 流程)
> - `paper_learning/2. AAAI最终故事/协同加速框架_工作流_v1.5.md` — 方法论定义

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
模型: `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/`

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
- Pyramid baseline ckpt: `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/`
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

- `models/lgb_v6_amota.txt` — 跨模型 amota 预测器 (Pyramid in-sample MAE 0.010, OK 用)
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
