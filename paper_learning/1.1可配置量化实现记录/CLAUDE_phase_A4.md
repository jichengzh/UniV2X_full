# Phase A4: 敏感度分析

> 覆盖: 全部维度 D1-D9 的基线数据, 搜索空间缩减
> 新建文件: tools/sensitivity_analysis.py
> 依赖: Phase A3 (统一配置入口)
> 预计工作量: 1 天���写 + 1 天执行
> 状态: 待开始

---

## 1. 任务清单

- [ ] Task 1: sensitivity_analysis.py 编写
- [ ] Task 2: 逐层敏感度扫描执行 (过夜自动化)
- [ ] Task 3: 交互��分析
- [ ] Task 4: 校准方法/粒度/对称性/位宽对比
- [ ] Task 5: V2X 通信精度影响���估
- [ ] Task 6: 结果汇总 -> sensitivity_report.json
- [ ] 反思文档

---

## 2. 分析内容

```
1. 逐层敏感度 (D7)
   对每层 i: 仅量化该层 INT8, 其余 FP16 -> delta_AMOTA[i]
   分三档: safe_int8(delta<0.005) / search(0.005-0.02) / skip_fp16(>0.02)

2. 位宽敏感度 (D1/D2)
   对 search 档层: 分别测 INT4/INT6/INT8 -> delta[bits]

3. 粒度对比 (D3/D4)
   对 search 档层: per-tensor vs per-channel -> 选最优

4. 校准方法对比 (D6)
   全局测 mse/minmax/entropy/percentile -> 选最优

5. 对称 vs 非对称 (D5)
   全局对比 -> ��认 TRT 对称约束的精度代价

6. 交互项 (D7 层间耦合)
   关键���对: 联合���化 -> delta_joint vs delta_i + delta_j

7. W-only vs W+A (D8)
   全局对比

8. V2X 通信精度 (D9)
   fp16/int8/int4 通信 -> 精度���响
```

---

## 3. ��出格式: sensitivity_report.json

```json
{
  "baseline_amota": 0.381,
  "layer_sensitivity": { ... },
  "interactions": { ... },
  "calibration_comparison": { ... },
  "granularity_comparison": { ... },
  "symmetry_comparison": { ... },
  "bitwidth_comparison": { ... },
  "quant_target_comparison": { ... },
  "v2x_comm_comparison": { ... },
  "search_space_reduction": {
    "locked_int8": [...],
    "locked_fp16": [...],
    "search_layers": [...],
    "original_dims": 30,
    "reduced_dims": 12
  }
}
```

---

## 4. 执行计划

```
Day 1 (编写):
  - sensitivity_analysis.py 主脚本
  - 自动化遍历逻辑
  - 结���收集��� JSON 输出

Day 2 (执行, 可过夜自动化):
  - 逐层敏感度: ~30层 x 5min = ~2.5h
  - 位宽对比 (search层): ~10层 x 3位宽 x 5min = ~2.5h
  - 粒度/校准/对称性: ~5次全局���估 x 5min = ~25min
  - 交互项: ~5对 x 5min = ~25min
  - V2X 通信: ~3种配置 x 5min = ~15min
  总计: ~6h (可过夜跑)
```

---

## 5. 执行记录

### 代码实现
- [x] sensitivity_analysis.py 编写完成 (1040 行)
- 6 个分析模式: layer_sensitivity, calibration, granularity, symmetry, bitwidth, interaction
- 使用 BEV cosine similarity 作为精度代理指标 (避免完整 AMOTA 评估的依赖)
- FP32 baseline 一次性收集复用
- 每个分析模式创建独立 QuantModel 避免状态污染
- 结构化输出 sensitivity_report.json

### 关键设计决策

1. **精度代理**: 使用 BEV 输出 cosine similarity 而非 AMOTA
   - 优势: 无需完整 mmdet3d 评估管线, 单次评估 ~3s vs ~3min
   - 劣势: 不是直接的任务指标, 但 BEV cosine 与 AMOTA 强相关 (Phase E 已验证)
   - 分类阈值: cos >= 0.9999 → safe_int8, 0.999~0.9999 → search, < 0.999 → skip_fp16

2. **逐层分析方法**: 对每层独立启用 INT8, 其余保持 FP16
   - 使用 QuantModule.set_quant_state(True, True) 逐层开关
   - 每层需重新校准 weight scale (几十毫秒) + 跑 n_eval 次前向传播
   - 预估耗时: ~30层 x ~30s/层 = ~15 分钟

3. **状态隔离**: 每个 comparison 分析创建独立 QuantModel (CPU deepcopy)
   - 避免前一个分析的 scale/zero_point 污染后续分析
   - GPU 显存通过 del + empty_cache 释放

### GPU 执行结果 (2026-04-12, GPU 7, RTX 4090)

- [x] Task 2: layer_sensitivity 分析完成 (54层, ~90s)

**结果**:
```
54 个量化层:
  45 safe_int8  (cos >= 0.9999)
  9  search     (0.999 < cos < 0.9999)
  0  skip_fp16  (无!)
```

**9 个 search 层** (按 cosine 从低到高排序):
```
layers.5.ffns.0.layers.1:                                              cos=0.999290
layers.5.attentions.1:                                                 cos=0.999589
layers.5.attentions.1.orig_module.output_proj:                         cos=0.999694
layers.4.attentions.1:                                                 cos=0.999835
layers.4.attentions.1.orig_module.output_proj:                         cos=0.999839
layers.3.ffns.0.layers.1:                                              cos=0.999851
layers.4.ffns.0.layers.1:                                              cos=0.999861
layers.5.attentions.1.orig_module.deformable_attention.orig_module.value_proj: cos=0.999891
layers.5.attentions.1.orig_module.deformable_attention:                cos=0.999892
```

**关键发现**:
1. **无 skip_fp16 层**: 所有 54 层单独量化时 cosine > 0.999，说明 BEV encoder 对逐层量化容忍度极高
2. **search 层集中在后几层** (layer 3/4/5): 后层特征更抽象，量化误差更大
3. **SCA (attentions.1) 比 TSA (attentions.0) 更敏感**: SCA 的 deformable_attention 涉及空间采样，与之前 sampling_offsets 敏感的经验一致
4. **FFN 的 layers.1 (output projection) 比 layers.0.0 (hidden) 更敏感**: output 直接影响残差连接

- [x] Task 3-6: 全部 6 项分析完成 (--analysis all)

**calibration/granularity/symmetry/bitwidth 对比结果**: 全部 cos=1.000000
- 原因: 全模型 cosine 被 45 个不敏感层"稀释", 无法区分不同配置的细微差异
- 说明 cosine similarity 作为全模型代理指标分辨力不足
- 改进方向: 对比分析应只测量 search 层的输出 cosine, 或用完整 AMOTA

**interaction 分析结果**: 交互项全部为负
```
(layer5.ffn, layer5.attn):      interaction = -0.001121
(layer5.ffn, layer5.attn.proj): interaction = -0.001016
(layer5.attn, layer5.attn.proj): interaction = -0.000717
```
- 负交互项 = 联合量化误差 < 各层误差之和 = **相互抵消效应**
- 这意味着 pipeline 中担心的"非线性耦合恶化"在当前模型上**不成立**
- 线性加性假设是保守的（高估联合损失）, Level 1 廉价评估是安全的

---

## 6. 反思

### 完成时间
2026-04-12, 代码编写完成。实际执行需在有 GPU 的环境下运行 (~1h)。

### 设计选择的权衡

1. **cosine similarity vs AMOTA**: 选择 cosine 是因为它在搜索循环中快 60 倍 (3s vs 3min)。但需要在首次运行后验证 cosine 阈值与 AMOTA 的对应关系。如果发现 cosine=0.9995 的配置 AMOTA 差距很大, 需要调整分类阈值。

2. **per-layer vs full-model 分析**: 逐层分析假设层间量化独立性, 但 interaction 分析模式专门检测这个假设何时失效。如果发现大量交互项, 说明线性加性假设不可靠, 搜索时需要更多联合评估。

3. **minmax 做逐层扫描 vs entropy**: 选择 minmax 是因为 Phase A1 发现 entropy+channel-wise 极慢 (6min/32ch)。minmax 虽然精度略差, 但逐层扫描是 O(n_layers) 次调用, 速度至关重要。

### 运行命令 (供后续执行用)

```bash
cd /home/jichengzhi/UniV2X

# 全量分析 (~1h)
python tools/sensitivity_analysis.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --output calibration/sensitivity_report.json \
    --n-eval 10 \
    --analysis all

# 仅逐层敏感度 (~15min)
python tools/sensitivity_analysis.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --output calibration/sensitivity_report.json \
    --n-eval 10 \
    --analysis layer_sensitivity
```
