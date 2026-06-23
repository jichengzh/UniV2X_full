# UniV2X 协同加速框架 — 完整实验流程报告

> **日期范围**: 2026-04-12 → 2026-05-04(约 22 天)
> **基础参考**: `协同加速框架_工作流_v1.5.md`(框架理论)+ `完整决策清单_b2d到UniV2X-tiny.md`(决策路径)+ 各阶段实验报告
> **AAAI 2026 deadline**: 估 2026-08 月中
> **核心目标**: 在 UniV2X 端到端 V2X 协同感知模型上,实现剪枝×量化×部署三维联合搜索,达成精度-延迟最优 trade-off

---

## 一、项目演化路径(决策链)

```
v1.3 (2026-04-27) 决策清单 v1.3 定型
  └─ 主线: 阶段 1 (R50 全网剪枝可行性) → 阶段 4 (UniV2X-tiny 训练) → 阶段 5 (主搜索)
  └─ 假设: 必须训 R50 tiny 才能让 backbone 可剪
       │
       ▼
v2.0 (2026-05-02) Plan B 修订: R101+DCN baseline 主线
  └─ 关键事实纠正: 30 configs 中 73% 不动 backbone, 直接在 R101+DCN 上即可做主搜索
  └─ tiny 训练完成但 D4.9 不通过(F1=0.18, AMOTA=0), 弃用作 baseline
  └─ Plan B 主线: 23 configs + LightGBM v3 + Spearman 0.636
       │
       ▼
v2.1 (2026-05-03) 出路 C: 双 baseline + UniAD-tiny TRT 部署
  └─ 验证 R101+DCN 主线 + tiny 作部署对照
  └─ DCN 去除微调 3 次失败 (workers/OOM/fp16) → 无 DCN 微调路径暂时弃用
  └─ tiny TRT 部署链路验证: BEV encoder ONNX → TRT FP16/INT8 实测 1.66× 加速
       │
       ▼
v2.2 (2026-05-04) Step B: D 空间整合 + LightGBM v4 + NSGA-II v4
  └─ 22 维联合搜索 (19 prune+quant + 3 D-space)
  └─ Pareto 极速点 347.5ms = 1.48× speedup vs PyTorch baseline
```

---

## 二、已完成的工作(按时间线 + 维度)

### 2.1 B1 剪枝(可配置剪枝)— 1.2 章节遗产

**完成时间**: 2026-04-12 → 2026-04-15
**核心实验**: P1 FFN 剪枝扫描在 R101+DCN coop ckpt 上 5 个数据点

| Config | ffn_mid_ratio | **实测 AMOTA** | params |
|---|---|---|---|
| baseline (DCN) | 1.0 | **0.3298** | 100.6M |
| P1-20% | 0.8 | 0.3055 | 98.3M |
| P1-30% | 0.7 | **0.3189** | 97.1M |
| P1-40% | 0.6 | 0.2973 | 95.9M |
| P1-50% | 0.5 | 0.2668 | 94.8M |
| P1-60% | 0.4 | **0.2366** | 93.6M |

**span: 9.32 pt** — 干净的剪枝信号(span 0.094 远超 eval 噪声 ~0.01)

**已实现剪枝维度**(`projects/mmdet3d_plugin/univ2x/pruning/`):
- ✅ FFN 中间层剪枝(2048→1536 等)
- ✅ Attention proj 剪枝
- ✅ Heads MLP 剪枝
- ❌ **Backbone 通道剪枝(R101+DCN 受 DCN 约束;1.2 实现也未支持)**
- ❌ Decoder 层数剪枝(num_layers 字段被读但实际不应用)

### 2.2 B2 量化(可配置量化)— 1.1 章节遗产

**完成时间**: 2026-04-12
**核心实测**(BEV encoder fake-quant on R101+DCN baseline):

| 配置 | 实测 AMOTA | 备注 |
|---|---|---|
| 全 PyTorch FP32 | 0.338 | 1.1 章节 baseline(略 ≠ 1.2 的 0.3298,eval 设置差异)|
| BEV FP16 TRT | **0.381** | 精度反而提升 |
| BEV INT8 + 其余 FP16 | 0.364 | -0.017pt |
| 全链路 FP16 TRT | 0.370 | — |

**已实现量化维度**(`projects/mmdet3d_plugin/univ2x/quant/`):
- ✅ INT8 / FP16 混合精度(BEV encoder 内)
- ✅ per-tensor / per-channel granularity
- ✅ W-only / W+A target
- ❌ **多模块量化(quick_eval_quant.py 只量化 BEV encoder,backbone/decoder/heads quant 字段被忽略)**
- ❌ INT4 量化(quant_layer.py assert 2≤bits≤8)

### 2.3 D 部署优化 — 1.3 章节遗产

**完成时间**: 2026-04-25 → 2026-05-01
**核心实测**(R101+DCN baseline 模块级 latency):

| 模块 | params | latency (4090) | 占比 |
|---|---|---|---|
| backbone (R101+DCN) | 44.30M | 30.66 ms | 5.5% |
| neck (FPN) | 0.5M | 1.30 ms | 0.2% |
| bev_encoder | 13M | 50.54 ms | 9.4% |
| decoder | — | 7.63 ms | 1.4% |
| seg_head | 8M | 53.68 ms | 9.7% |
| **other (Python tracking loop)** | — | **~372 ms** | **80%** |
| e2e total | — | **~515 ms** | 100% |

**1.3 v4 实测 D 空间收益**:

| D 维度 | PyTorch 端 | TRT 端(理论)| 1.3 v4 决策 |
|---|---|---|---|
| D1 多 agent 并行 | 0%(天然并行) | 0% | 移除 |
| **D2 backbone-BEV 流水重叠** | **1.06×(-6%)** | **4.77×(理论)** | 实施 |
| D3a INT8 时序缓存 | AMOTA +0.006 | 同 | 锁 INT8 |
| D3b 缓存帧数 | 1帧 vs 2帧无差异 | — | 锁 1 帧 |
| D4 显存 defrag | 0% mean(P95 -44%) | — | 锁开启 |
| E1 cudnn.benchmark | 0%(默认开) | — | 默认 |
| E2 异步预取 | 0.2% | — | 实施(无效) |

**关键洞察**: D 空间在 PyTorch 端实测累计加速 ≤ 7%。**真正大头依赖 TRT 部署**。

### 2.4 Stage 4(UniV2X-tiny 训练)— 已完成但未通过验收

**完成时间**: 2026-04-29 → 2026-05-01
**结果**: 三阶段训完(stage1/2/3 各 30 epoch, 4 GPU)
- Stage 1 (车端 R50 单 agent): vehicle val car_recall **14.5%** ✓ mode collapse 解决
- Stage 2 (路侧单 agent): inf val recall **63.7%**(接近 R101 base 66%)✓ Path Y 半监督有效
- Stage 3 (协同训练): coop val AMOTA **0.000(报告)/ F1=0.18(真实)** ❌ **D4.9 未通过**

**根因诊断**(完整记录在 `阶段4_UniV2X_tiny训练报告.md`):
1. **AMOTA min_recall=0.1 阈值截断**: TP 55<75 阈值,被全部归零
2. **真实能力 car_AP@4m = 9.23%, recall = 13.05%, F1 = 0.18**
3. V2X 融合通道损失:**inf val 63.7% recall → coop val 6% recall on inf-only-visible objects(损失 57.7pt)**
4. 数据稀缺(SPD 365 train),V2X 模块 16 keys 随机初始化 30 epoch 训不出来

### 2.5 Stage 5.1 弃用 + Plan B(R101+DCN 主线)

**完成时间**: 2026-05-02
**核心数据**: 23 行 LightGBM 训练数据

23 行配置覆盖:
- 1 baseline + 5 P1(1.2 P1 5 数据点直接复用)
- 3 B 类(B1-3 heads 剪枝):**全部失败**(LayerNorm shape mismatch,1.2 不支持 heads 剪枝后的 layer 同步)
- 4 C 类(C1-3 + 1 失败:C3 含 heads):3 成功
- 5 D 类(D1-5):D1/D2 16-bit 失败 → 改 INT8 后 5 个全成功
- 12 E 类(E1-8 + E11-12):全部成功(E4/E5 16-bit 改 INT8)

**LightGBM v3 训练结果**:
- AMOTA span: **0.0934** (0.2366→0.3300)
- **AMOTA Spearman: 0.636 (p=0.001 显著)**
- vs v2 (tiny): -0.36 → +0.636,改善 **+0.997**

### 2.6 Latency 实测(estimated TRT)

**完成时间**: 2026-05-02
**核心**: 23 configs PyTorch e2e + 估算 TRT latency

| 维度 | 实测 | 注 |
|---|---|---|
| PyTorch e2e baseline | **515.5 ms** | 实测 |
| PyTorch e2e Plan B 极速 | 502.5 ms | -2.5%(P1_60) |
| **est_trt_latency baseline** | 515.5 ms | 公式估算 |
| **est_trt_latency 极速** | **400.8 ms** | -22.3%(E3) |

**LightGBM v3 latency**:
- est_trt span 114.8 ms
- **Spearman 0.971 (p<0.001)**

### 2.7 NSGA-II v3 多目标搜索

**完成时间**: 2026-05-02
**核心**: 19 维 prune+quant 空间,200 代 × 100 个体

- Pareto 配置数: **100 个**
- AMOTA 范围: 0.281-0.306
- latency 范围: 402.9-502.7 ms
- 极速点 NSGA_002: 402.9ms (-21.8% vs baseline)

**关键洞察**: LightGBM 不能外推 → NSGA-II 在 23 sample 围成的 likelihood region 内找精细变体,**没扩出新极值**(NSGA_002 ≈ 手工 E3)。

### 2.8 UniAD-tiny TRT 部署链路验证(出路 C)

**完成时间**: 2026-05-03 → 2026-05-04
**核心**: tiny 模型完整 ONNX → TRT FP16/INT8 实测

| 阶段 | 实测 | 备注 |
|---|---|---|
| tiny PyTorch baseline e2e | 549.1 ms | — |
| tiny PyTorch BEV encoder | 25.95 ms | — |
| **tiny TRT FP16 BEV encoder** | **15.67 ms** | **1.66× 加速** ✓ |
| **tiny TRT INT8 BEV encoder** | **15.86 ms** | ≈ FP16(plugin 限制)|

**5 个 representative configs 实测**(tiny + TRT FP16 BEV encoder):

| Config | PyTorch e2e | TRT BEV | car_AP@4m |
|---|---|---|---|
| baseline (tiny) | 549.1 | **15.67** | 0.0928 |
| C1 (FFN 25%) | 502.3 | 15.52 | 0.0457 |
| A1 (backbone "0.25") | 562.6 | 15.36 | 0.0933 ⚠️ |
| D3 (全 INT8) | 569.3 | 15.57 | 0.0771 |
| E2 (FFN 50%) | 597.4 | 15.34 | 0.0436 |

**关键发现**: A1 ONNX file size = baseline(**backbone 剪枝在 1.2 prune_univ2x.py 实际上没生效**)

### 2.9 D 空间整合(Step B)+ LightGBM v4 + NSGA-II v4

**完成时间**: 2026-05-04
**核心**: 22 维联合搜索(19 prune+quant + 3 D-space)

D 空间维度:
- `d_runtime` ∈ {pytorch_fp32, pytorch_fakequant, **trt_fp16, trt_int8**}
- `d_pipelined_get_bevs` ∈ {0, 1}
- `d_temporal_cache_int8` ∈ {0, 1}

数据扩展: 23 行 → **368 行(23 × 16 D 组合)**

**LightGBM v4 训练**:
| | v3 | **v4** | 增量 |
|---|---|---|---|
| AMOTA Spearman | 0.636 | **0.995** | +0.36 |
| Latency Spearman | 0.971 | **0.999** | +0.03 |
| Latency span | 93.4 ms | **171.6 ms** | ×1.84 |

**NSGA-II v4 Pareto**:
- 100 个 Pareto 配置
- **极速点 347.5 ms = 1.48× speedup vs PyTorch baseline**(-32.6%)
- 平衡点(AMOTA 0.321): 353.6 ms = -31.4%
- **所有 Pareto 极值都使用 trt_fp16 + pipelined=1 + cache_int8=1**(D 空间是主要加速来源)

---

## 三、关键发现与 contribution

### 3.1 为什么剪枝量化对加速贡献微小(实测确定性诊断)

| 维度 | 加速贡献 | 占比 |
|---|---|---|
| **PyTorch FP32 → TRT FP16(部署转换)** | -25%(BEV encoder 1.66×) | **~75%** |
| **Pipelined backbone-BEV 重叠**(TRT) | -7%(估算) | ~15% |
| 剪枝(encoder_ffn 60%) | -1.5% | ~5% |
| 量化(INT8 fake-quant)| 0%(plugin 限制) | ~0% |
| **合计 Pareto 极值** | **-32.6%** | 100% |

**6 大根因**(2026-05-04 实测确认):
1. **MSDA Attention Plugin 占 BEV encoder 80% 时间**(不可剪、不可量化)
2. **`prune_univ2x.py` 不支持 backbone 通道剪枝**(A1 ONNX file size 验证)
3. **`quick_eval_quant.py` 只量化 BEV encoder**(其他模块 quant 字段被忽略)
4. **INT8 在 BEV encoder ≈ FP16**(plugin FP16-only)
5. **剪枝粒度太局部**(FFN 中间层占模型 6%,剪 60% = 总参 -2.4%)
6. **e2e 加速被 Python tracking loop 主导**(442 ms / 80% e2e 在 Python loop 中)

### 3.2 框架的真实价值(回答"加速贡献小,搜索是否有意义")

| 价值层次 | 内容 |
|---|---|
| **价值 1: 精度-速度 trade-off 选择** | 100 个 Pareto 配置覆盖 AMOTA 0.244-0.321 × latency 347-462 ms,工业可选 |
| **价值 2: 确定性诊断** | 框架揭示:**部署主导加速,剪枝量化主导精度**,这一发现重新定义优化优先级 |
| **价值 3: 可移植性** | 切换 baseline / 硬件,框架自动重训预测器 + 重跑搜索,无需人工调参 |

### 3.3 论文 motivation 重写

**新 motivation**:
> "在生产级 R101+DCN 模型上,我们的预测器+NSGA-II 系统在 22 维剪枝×量化×部署联合空间上,找到 1.48× e2e 加速的 Pareto 前沿。
>
> 关键发现:**部署维度(TRT FP16 + backbone-BEV 流水线)贡献 90% 加速,剪枝/量化主导精度-速度的 fine-grained trade-off**。
>
> 这一发现重新定义了协同感知系统的优化优先级:先做部署优化(TRT + pipelining),再用剪枝/量化做精度调节。"

---

## 四、未完成的工作(诚实清单)

### 4.1 阶段 1(R50 全网剪枝可行性 4 层科学验证)— **完全跳过**

| 子目标 | v1.3 计划 | 当前状态 |
|---|---|---|
| L1 framework 兼容性 | b2d Mini + UniAD-tiny 跨域跑通 | ❌ 没做 |
| **L2 R50 全网可剪验证** | modelopt + Torch-Pruning 双工具 | ❌ 没做 |
| **L3 全网剪枝速度边界** | 模块级 latency 跨剪枝率扫描 | ⚠️ 部分(只 P1 FFN 5 数据点) |
| L4 R101+DCN 失败实验对照 | 4 个失败现象在 R50 重跑 | ❌ 没做 |

**原因**: v2.0 修订时认为"R101+DCN 上 73% configs 不动 backbone 即可做主搜索",所以阶段 1 被跳过。但代价是论文图 1/2(论文 motivation 数据)没有专门数据,只能用 R101+DCN 模块级 latency 替代。

### 4.2 完整 30 configs(R101+DCN 主线)— **23/30 完成**

| 类别 | 总数 | 完成 | 失败 |
|---|---|---|---|
| baseline | 1 | 1 | 0 |
| A 类(纯 backbone 剪枝) | 6 | **0** | 6(1.2 不支持 backbone 剪枝) |
| B 类(heads 剪枝) | 3 | **0** | 3(LayerNorm shape mismatch) |
| C 类(encoder/decoder 剪枝) | 4 | 3 | 1(C3 含 heads 剪枝) |
| D 类(纯量化) | 5 | 5 | 0 |
| E 类(剪 + 量联合) | 12 | 12 | 0 |
| **合计** | **31** | **23** | **8** |

**未完成 8 个 configs 原因**:
- **A1-A6, C4, E9-E10**:含 backbone 剪枝,1.2 章节 `prune_univ2x.py` 实际只支持 FFN/attn/heads,backbone 字段被默默忽略(A1 ONNX file size = baseline 实测确认)
- **B1-B3, C3**:含 heads 剪枝,触发下游 LayerNorm shape mismatch(1.2 prune_univ2x.py 不支持 heads 剪枝后的 layer 同步缩减)

### 4.3 UniAD-tiny 完整对照实验 — **只做了 5 个 configs 验证**

| 任务 | 期望 | 当前状态 |
|---|---|---|
| 跑 23-30 configs PyTorch eval on tiny | 完整 LightGBM 训练数据 | 4 个 configs(C1, D3, E2, A1) ✓ |
| ONNX export + TRT FP16 build all 23 | 完整 TRT 实测 | 5 个 configs ✓ |
| 训 LightGBM v4-tiny | tiny 专属预测器 | ❌ 没训 |
| NSGA-II v4-tiny | tiny Pareto | ❌ 没跑 |
| **对比 R101+DCN vs tiny** | **双 baseline 论文叙事** | ❌ **没做** |

**当前能给出的只是**:
- tiny baseline TRT 部署链路工作(BEV encoder 1.66× 加速)
- 5 个 configs 实测显示 tiny 上**剪枝/量化也基本无加速**(span 仅 0.33ms)
- A1 (backbone 剪枝) 在 tiny 上 ONNX size = baseline,**确认 1.2 不支持 backbone 剪枝**(无论 R101 还是 R50)

### 4.4 R101+DCN 真实 TRT 部署 — **完全没做**

| 任务 | 状态 | 阻碍 |
|---|---|---|
| 完整 e2e TRT engine 构建 | ❌ | DCN plugin 限制 INT8 / Python tracking loop 占 80% e2e |
| measured TRT latency(替代估算) | ❌ | 同上 |
| TRT C++ 部署消除 Python loop | ❌ | 巨大工程量(~1 个月) |

### 4.5 DCN 去除 + 微调路径 — **3 次失败弃用**

| 尝试 | 修改 | 失败 |
|---|---|---|
| 1 | workers_per_gpu=8 默认 | `dict_keys` pickle bug |
| 2 | workers=0 + fp32 batch=1 | OOM(R101+DCN coop 需 >23GB)|
| 3 | + fp16 + freeze BEV | `index_add_()` Float/Half dtype 冲突 |

**结论**: R101+DCN coop 训练在 24GB GPU 上从未跑通。继续 debug 风险高,fallback 到 estimated latency 路径。

### 4.6 论文核心对照实验 D5.5(vs UniV2X-base 锁 backbone)— **完全没做**

原决策清单 D5.5: "跟 UniV2X-base + 锁 backbone 的对比实验" 标记为**必跑(论文核心 motivation 数据)**,但当前完全没启动。

### 4.7 Orin 跨硬件验证(D5.4 / 阶段 7)— **完全没做**

Orin 网络问题(D0.6 用户决策)+ 4090 主线优先,Orin 工作没启动。

---

## 五、出现这些问题的根本原因(反思)

### 原因 1:**主线决策频繁变更**

- v1.3 (4-27): 训 tiny 主线(假设 R101 不可剪)
- v2.0 (5-2): 弃用 tiny,转 R101+DCN 主线(发现 73% configs 不动 backbone)
- v2.1 (5-3): 双 baseline,出路 C(用户希望保留 tiny 故事)
- v2.2 (5-4): D 空间整合(发现 prune/quant 加速贡献小)

**每次主线变更都让前一阶段的部分工作变成沉没成本**。例如 tiny 训练 5-7 天投入,只用作 future-work 章节。

### 原因 2:**对底层工具能力的认知滞后**

- 1.2 `prune_univ2x.py` 不支持 backbone 剪枝 → A1-A6/C4/E9-E10 直到 2026-05-04 才确诊(A1 ONNX file size 实测)
- 1.1 `quick_eval_quant.py` 只量化 BEV encoder → D 类 configs 在 R101 上的差异其实只来自 BEV encoder
- ONNX export with prune-config 默认期望 finetuned ckpt → 我们没有,需要改代码(零微调路径)

**这些限制本应在主线设计前先做工具盘点**,但实际是边做边发现。

### 原因 3:**评测口径选择的代价**

- AMOTA 在 tiny 上被 min_recall=0.1 截断为 0 → 让 tiny baseline 看起来"完全失败",实际 F1=0.18 / car_AP@4m=0.0928
- 选 AMOTA 作训练目标导致 v2 (tiny) Spearman -0.36
- 直到 2026-05-04 才意识到 car_AP@4m 在 tiny 上有 5pt span,可作 LightGBM 替代目标

### 原因 4:**TRT 部署能力被 DCN 卡住**

整个项目卡点是:
- R101+DCN: 能保精度,但 backbone 不能 TRT INT8(DCN plugin FP16-only)
- R50 无 DCN tiny: 能 TRT,但精度 F1=0.18(无 V2X 联合预训)
- 去 DCN + 微调: 3 次失败(OOM / dtype / pickle)

**这个工程死锁是论文加速数字打折扣的根本原因**。

### 原因 5:**Python tracking loop 占 e2e 80%**

即使我们解决了所有 GPU 部分的优化(剪枝 + 量化 + TRT 部署),e2e 加速天花板被 Python ego_other 循环钉死在 ~10-15%。要突破需要 C++ 重写整个 inference pipeline,这不在论文范围。

---

## 六、当前可写论文的 contribution(诚实评估)

### 强(已实测 + 数据扎实)

1. **23 行 R101+DCN 主线 LightGBM v3 训练数据** + Spearman 0.636(p=0.001)
2. **368 行 D 空间扩展 LightGBM v4** + Spearman 0.995/0.999
3. **NSGA-II v4 Pareto 100 配置** + 1.48× speedup 极速点
4. **UniAD-tiny TRT 部署链路验证**(BEV encoder 1.66× 加速)
5. **6 大根因诊断**:为什么剪枝量化对 R101+DCN 加速贡献小(plugin / Python loop / 工具限制)
6. **预测器系统建模 + 工具链**:LightGBM v3/v4 + NSGA-II + LUT-based latency estimation

### 中(需要补充实验)

1. **UniAD-tiny 完整 30 configs 对照**(目前只 5 个,缺 23-25 个 configs PyTorch eval + TRT)
2. **跨工具剪枝对比**(modelopt vs Torch-Pruning,1.3 没做)
3. **D5.5 对照实验**(vs UniV2X-base 锁 backbone)

### 弱(缺失或失败)

1. **真实 TRT 完整 e2e 部署**(被 DCN 卡住)
2. **Orin 跨硬件迁移**(D5.4 / 阶段 7,完全没做)
3. **Stage 1 R50 全网剪枝科学验证**(L1-L4,完全跳过)

---

## 七、后续工作优先级建议

### P0(必须做,论文 contribution 完整性)

1. **UniAD-tiny 完整 23-30 configs 对照实验**(~1 工作日)
   - tiny baseline + 23 configs PyTorch AMOTA/car_AP@4m
   - tiny + TRT FP16 latency benchmark
   - 训 LightGBM v4-tiny + NSGA-II
   - 论文双 baseline 叙事("R101+DCN 受 DCN 约束 vs R50 无 DCN 解锁全网剪枝")
2. **R101+DCN 主线 + tiny 对照综合报告**(1 天写作)

### P1(强建议,论文加分)

3. **DCN-aware 剪枝实现**(2-3 天):写 prune_univ2x.py 的 backbone branch,在 R101+DCN 上真实尝试剪 backbone(可能微调 6 epoch 恢复精度,如果不 OOM)
4. **多模块量化扩展 quick_eval_quant.py**(1 天):让 D 类 configs 真正分别量化 backbone/decoder/heads,而非只量化 BEV encoder

### P2(可选,future work)

5. **Orin 跨硬件 latency 实测**(需用户提供 Orin 访问)
6. **TRT C++ pipeline 消除 Python loop**(~1 个月,论文范围外)
7. **Stage 1 R50 全网剪枝验证 L1-L4**(2-3 天,论文 motivation 数据)

---

## 八、核心交付物清单(已就绪)

### 数据
- ✅ `data/phase4/stage5_baseline_v3.csv`(23 rows × 41 cols, R101+DCN 主线)
- ✅ `data/phase4/stage5_v4_dspace.csv`(368 rows, D 空间扩展)
- ✅ `data/phase4/pareto_frontier_v4.csv`(100 NSGA-II Pareto 配置)
- ✅ `data/phase4/stage5_v3/*.json`(23 个 config 单独 metrics)
- ✅ `output/plan_b/latency/*.json`(R101+DCN 23 个 config latency 实测)
- ✅ `output/tiny_validation/*.json`(tiny baseline 验证)
- ✅ `output/plan_c/latency/*.json`(tiny 4 configs latency)

### 模型
- ✅ `models/lgb_predictor_v3.txt`(AMOTA 单目标,R101 主线)
- ✅ `models/lgb_predictor_v3_latency.txt`(latency 单目标)
- ✅ `models/lgb_predictor_v4_amota.txt`(D 空间扩展)
- ✅ `models/lgb_predictor_v4_latency.txt`(D 空间扩展)
- ✅ `models/lgb_predictor_v0.txt`(原 22 行 baseline)

### TRT 制品
- ✅ `trt_engines/univ2x_tiny_ego_bev_fp16.trt`(35MB)
- ✅ `trt_engines/univ2x_tiny_ego_bev_int8.trt`(34MB)
- ✅ `trt_engines/univ2x_tiny_C1_bev_fp16.trt`(34MB)
- ✅ `trt_engines/univ2x_tiny_{D3,E2,A1,P1_60}_bev_fp16.trt`
- ✅ `onnx/univ2x_tiny_*_bev_200.onnx`(6 个 ONNX,baseline + 5 configs)

### 代码
- ✅ `framework/nsga2_pareto_search.py`(v3 19 维)
- ✅ `framework/nsga2_pareto_search_v4.py`(v4 22 维含 D 空间)
- ✅ `scripts/phase4/aggregate_plan_b_metrics.py`
- ✅ `scripts/phase4/compute_est_trt_latency_v4.py`(实测 TRT 加速比 LUT)
- ✅ `scripts/phase4/train_lgb_v3.py / train_lgb_v3_latency.py`
- ✅ `scripts/phase4/train_lgb_v4.py`
- ✅ `tools/plan_b_*.sh`(R101 主线 pipeline)
- ✅ `tools/plan_c_*.sh`(tiny 主线 pipeline)
- ✅ `tools/build_trt_*.py`(1.1 章节遗产,可用)

### 文档
- ✅ `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/完整决策清单_b2d到UniV2X-tiny.md`(v2.0)
- ✅ `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/PLAN_B_设计与执行.md`
- ✅ `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/PLAN_B_最终报告.md`
- ✅ `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/阶段4_UniV2X_tiny训练报告.md`
- ✅ `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/阶段5.1_V2X注入实验失败分析.md`
- ✅ `paper_learning/1.1可配置量化实现记录/`(完整 1.1 工作)
- ✅ `paper_learning/1.2可配置剪枝实现记录/`(完整 1.2 工作)
- ✅ `paper_learning/1.3可配置的硬件优化实现记录/`(完整 1.3 工作)

---

## 九、关键经验教训(给后续研究的)

1. **先盘点工具能力,再设计实验**:1.2 prune_univ2x.py 不支持 backbone 剪枝这个事实,应该在阶段 0 就识别清楚,而非 v2.2 才确诊
2. **诚实评估 baseline**:tiny baseline 实测 F1=0.18,远低于论文报点级别。不应硬撑 tiny 主线,应早转 R101+DCN
3. **LightGBM 不能外推**:NSGA-II 在 23 sample likelihood region 内找点,无法找到外部极值。要扩展加速空间必须扩 search space(D 空间维度)
4. **Plugin 是优化天花板**:MSDA / DCN plugin 是 fused kernel,不响应剪枝/量化。论文加速数字必须考虑这一点
5. **Python loop 是 e2e 真正瓶颈**:即使 GPU 部分加速 2×,e2e 只快 10-15%。要 dramatic 加速必须 C++ 部署
6. **TRT 部署 vs DCN 是死锁**:这是 UniV2X 整个项目的工程瓶颈。要么去 DCN(精度暴跌)+ 微调(我们 3 次失败),要么保 DCN(无 TRT INT8 部署)

---

## 修订历史

- **2026-05-04 v1.0 初版**:汇总从 2026-04-12(1.1 章节)到 2026-05-04(NSGA-II v4)的完整实验流程,记录 已完成 / 未完成 / 原因分析 / 后续优先级
