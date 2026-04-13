# 面向固定平台的软硬件协同优化完整Pipeline

> 项目: UniV2X
> 场景: 在固定GPU/Jetson平台上,通过结构化剪枝+PTQ量化压缩V2X协同感知模型,同时优化精度、延迟和内存
> 核心思路: 两层架构——内层联合搜索(软件压缩+部署配置)、外层固定优化(代价大的增益补充+TRT自动优化)
> 前置文档: [hw_sw_codesign_quant+prune.md](./hw_sw_codesign_quant+prune.md) (搜索空间与协同分析)

---

## 1. 联合搜索空间定义

```
┌─────────────────────────────────────────────────────────────────────┐
│                        联合搜索空间 x = [x_S, x_H]                  │
│                                                                     │
│  ┌─ x_S 软件子空间 ──────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  结构化剪枝:                                                   │  │
│  │    prune_rate[block_i]        逐块通道剪枝率 (5-8个块)         │  │
│  │    head_mid_channels          检测头中间表示宽度               │  │
│  │    prune_criterion            剪枝指标 (L1/Taylor/FPGM)       │  │
│  │                                                               │  │
│  │  PTQ量化:                                                     │  │
│  │    quant_bits[block_i]        逐块位宽 {4, 6, 8, FP16}        │  │
│  │    quant_granularity          粒度 {per_tensor, per_channel}   │  │
│  │    quant_symmetric            对称性 {True, False}             │  │
│  │    calibration_method         校准方法 {minmax, entropy, pct}  │  │
│  │                                                               │  │
│  │  V2X通信压缩:                                                  │  │
│  │    comm_feature_precision     通信特征精度 {fp16, int8, int4}  │  │
│  │    comm_spatial_compress      通信空间压缩 {none, 2x, 4x, roi}│  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ x_H 部署配置子空间 ─────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  精度执行:                                                     │  │
│  │    layer_precision_override[i]  逐层精度覆盖 {follow/fp16/fp32}│  │
│  │                                                               │  │
│  │  算子实现:                                                     │  │
│  │    attention_impl               {trt_native/flash/custom}      │  │
│  │    bev_transform_impl           {grid_sample/custom_plugin}    │  │
│  │                                                               │  │
│  │  内存策略:                                                     │  │
│  │    memory_pool_strategy         {isolated/shared_static/stream}│  │
│  │    temporal_cache_frames        时序缓存深度 {0, 1, 2, 4}      │  │
│  │    temporal_cache_precision     缓存精度 {same/fp16/int8}      │  │
│  │    camera_batching              {all_at_once/per_agent/seq}    │  │
│  │                                                               │  │
│  │  执行调度:                                                     │  │
│  │    multi_stream_strategy        {single/branch_parallel/pipe}  │  │
│  │    agent_batching_strategy      {pad_to_max/dynamic/bucketed}  │  │
│  │    num_agents_opt               optimization profile {1,2,3,4} │  │
│  │                                                               │  │
│  │  图结构:                                                       │  │
│  │    onnx_export_strategy         {monolithic/split_head/split_agent}│
│  │                                                               │  │
│  │  [Orin专用]:                                                   │  │
│  │    dla_offload_layers           DLA offload层集合              │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ 维度间约束 (快速剪枝不可行组合) ────────────────────────────┐  │
│  │  shared_static ⊗ branch_parallel    // 共享静态池排斥多流并行  │  │
│  │  low_prune + all_at_once → OOM检查  // 低剪枝+全相机batch验证  │  │
│  │  temporal_cache≥2 → memory_pool检查 // 多帧缓存加剧显存压力    │  │
│  │  bucketed → 排斥Orin(多engine显存)  // 分桶策略在小显存不可行  │  │
│  │  monolithic → memory_pool=N/A       // 单图无多引擎问题        │  │
│  │  prune后ch%8≠0 → 强制对齐或排除     // INT8通道对齐硬约束      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 完整Pipeline

### Phase 0: 逐层敏感度分析 (一次性, ~3.5小时)

```
  原始FP32模型
       │
       ├─── 量化敏感度扫描 ──────────────────────────────────┐
       │    对每一层/块独立:                                   │
       │      只量化该层→INT4, 其余FP16                       │
       │      测精度降幅 Δacc_quant[i]                        │
       │                                                      │
       ├─── 剪枝敏感度扫描 ──────────────────────────────────┤
       │    对每一层/块独立:                                   │
       │      只剪枝该层50%, 其余不动                          │
       │      测精度降幅 Δacc_prune[i]                        │
       │                                                      │
       └─── 硬件特性profiling ───────────────────────────────┤
            对典型layer配置单独benchmark:                      │
              Conv3x3 ch=64/128/256 × INT8/FP16              │
              Attention head=4/8 × INT8/FP16                  │
              各plugin实现的latency对比                        │
              → latency查表 LUT[layer_type, ch, bits]         │
                                                              │
       ┌──────────────────────────────────────────────────────┘
       ▼
  输出: sensitivity_map + latency_LUT
       │
       ├─ 高敏感层 → 锁定FP16/不剪枝 (移出搜索空间)
       ├─ 低敏感层 → 锁定INT4/高剪枝率 (移出搜索空间)
       └─ 中等敏感层 → 保留为搜索维度
       
  效果: 有效搜索维度从 ~30维 降至 ~12-15维
```

### Phase 1: 两阶段嵌套搜索 (自动化, ~12-15小时)

```
  搜索目标: max accuracy(x)
            s.t. latency(x) < L_target
                 memory(x)  < M_target
       │
       ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 外循环: 结构性决策                                             │
  │ (空间较小 ~20-50种, 可粗搜或穷举)                              │
  │                                                               │
  │ 软件侧决策:                                                    │
  │   · prune_rate[block_i]          各块剪枝率                    │
  │   · head_mid_channels            检测头宽度                    │
  │   · prune_criterion              剪枝指标                      │
  │                                                               │
  │ 硬件侧决策:                                                    │
  │   · onnx_export_strategy         图分割方式                    │
  │   · camera_batching              相机批处理粒度                │
  │   · agent_batching_strategy      agent处理策略                 │
  │   · multi_stream_strategy        多流执行策略                  │
  │   · memory_pool_strategy         内存池策略                    │
  │                                                               │
  │ 为什么这些在外循环:                                             │
  │   软件侧: 剪枝改变网络结构, 量化搜索建立在固定结构上才有意义    │
  │   硬件侧: 图分割/batch/stream决定了部署拓扑, 影响后续所有配置   │
  └──────────────────────┬────────────────────────────────────────┘
                         │
            对每个外循环配置 p_j:
                         │
                         ▼
                  ┌──────────────┐
                  │ 约束预检查    │
                  │              │
                  │ · ch%8对齐？  │
                  │ · 内存策略    │
                  │   与多流兼容？ │
                  │ · 显存粗估    │
                  │   <M_target？ │
                  └──────┬───────┘
                         │
                    不通过 → 跳过
                         │ 通过
                         ▼
                  ┌──────────────┐
                  │ 执行剪枝      │
                  │ → pruned_model│
                  └──────┬───────┘
                         │
                         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 内循环: 非结构性配置搜索                                       │
  │ (贝叶斯优化, 每个剪枝配置跑15-25轮)                            │
  │                                                               │
  │ 软件侧决策:                                                    │
  │   · quant_bits[block_i]          各块位宽                      │
  │   · quant_granularity            量化粒度                      │
  │   · quant_symmetric              对称性                        │
  │   · calibration_method           校准方法                      │
  │   · comm_feature_precision       通信特征精度                  │
  │   · comm_spatial_compress        通信空间压缩                  │
  │                                                               │
  │ 硬件侧决策:                                                    │
  │   · layer_precision_override[i]  逐层精度覆盖                  │
  │   · attention_impl               算子实现选择                  │
  │   · bev_transform_impl           BEV实现选择                   │
  │   · temporal_cache_frames        时序缓存深度                  │
  │   · temporal_cache_precision     缓存精度                      │
  │   · num_agents_opt               optimization profile         │
  │   · [Orin] dla_offload_layers    DLA分配                      │
  │                                                               │
  │ 为什么这些在内循环:                                             │
  │   软件侧: 量化不改变网络结构, 在固定pruned_model上搜索          │
  │   硬件侧: 精度覆盖/plugin/缓存等不改变部署拓扑, 可快速切换     │
  │   代理模型(GP)可以跨外循环迁移先验(warm-start)                  │
  │                                                               │
  │   ┌─────────────────────────────────────────────────────────┐ │
  │   │                   两级评估漏斗                           │ │
  │   │                                                         │ │
  │   │  候选配置 q_k                                            │ │
  │   │       │                                                  │ │
  │   │       ▼                                                  │ │
  │   │  Level 1: 廉价代理评估 (<1秒)                             │ │
  │   │    · FLOPs/Params计算                                    │ │
  │   │    · sensitivity_map加权精度降幅预估                      │ │
  │   │    · latency_LUT查表延迟预估                              │ │
  │   │    · 显存峰值预估:                                        │ │
  │   │        weights_mem = Σ(ch × kernel × bits/8)             │ │
  │   │        act_mem = max_act_size × camera_batch_size        │ │
  │   │        cache_mem = temporal_frames × feature_size        │ │
  │   │        total < M_target ?                                │ │
  │   │    · reformatter边界数量统计                              │ │
  │   │                                                          │ │
  │   │    预估精度 < 阈值 → 丢弃 ─────────┐                     │ │
  │   │    预估延迟 > L_target → 丢弃 ─────┤                     │ │
  │   │    预估显存 > M_target → 丢弃 ─────┤  ~70-80%被过滤      │ │
  │   │       │                             │                     │ │
  │   │       │ 通过                         │                     │ │
  │   │       ▼                             │                     │ │
  │   │  Level 2: 真实评估 (10-30分钟)       │                     │ │
  │   │    · 量化校准 (calibration_data)     │                     │ │
  │   │    · ONNX导出 (按export_strategy)   │                     │ │
  │   │    · TRT engine构建                  │                     │ │
  │   │    · 真实推理profiling:              │                     │ │
  │   │        - 端到端latency               │                     │ │
  │   │        - layer-by-layer breakdown    │                     │ │
  │   │        - GPU memory peak             │                     │ │
  │   │        - 确认无FP32意外回退           │                     │ │
  │   │    · 验证集子集(1/10)精度评估         │                     │ │
  │   │       │                             │                     │ │
  │   │       ▼                             │                     │ │
  │   │  反馈至贝叶斯优化:                    │                     │ │
  │   │    更新GP代理模型                     │                     │ │
  │   │    acquisition function选下一个候选   │                     │ │
  │   │    重复15-25轮                       │                     │ │
  │   └─────────────────────────────────────────────────────────┘ │
  │                                                               │
  │  GP warm-start:                                               │
  │    外循环 p_j → p_{j+1} 时,                                   │
  │    用p_j的GP后验初始化p_{j+1}的GP先验,                         │
  │    因为相邻剪枝率的最优量化配置通常相似                          │
  └───────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  汇总所有外循环结果   │
              │                     │
              │  所有 (p_j, q_best_j)│
              │  按 score 排序       │
              │                     │
              │  score = accuracy    │
              │    - λ₁·max(0,      │
              │      lat-L_target)  │
              │    - λ₂·max(0,      │
              │      mem-M_target)  │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  Top-K 完整验证      │
              │  (K=3~5)            │
              │                     │
              │  · 完整验证集精度    │
              │    mAP / NDS        │
              │  · 完整profiling     │
              │  · 多agent数量场景   │
              │    测试(1/2/3/4)     │
              │  · 显存水位确认      │
              └────────┬────────────┘
                       │
                       ▼
              输出: Top-3 最优配置
                   (pruning_config, quant_config, deploy_config)
```

### Phase 2: 外层固定优化 (半自动, ~10-12小时)

内层搜索找到最优配置组合后，外层施加代价较大的优化手段，以及TRT自动优化的部分。

```
  Top-3 最优配置
       │
       ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  A. 软件侧固定优化                                             │
  │                                                               │
  │  A1. 知识蒸馏 (~2-4小时/配置)                                  │
  │    · Teacher: 原始FP32模型                                     │
  │    · Student: pruned子网络                                     │
  │    · 少量epoch (5-10) 微调                                     │
  │    · 目标: 恢复剪枝导致的精度损失                               │
  │                                                               │
  │  A2. 重新校准量化 (~10分钟)                                     │
  │    · 蒸馏改变了权重分布                                         │
  │    · 用Phase 1搜到的量化配置(bits/粒度/对称性)                  │
  │    · 但重新跑校准数据, 更新scale/zero_point                    │
  │    · 不重新搜索量化策略(经验: 蒸馏不改变最优策略大方向)          │
  │                                                               │
  │  A3. [可选] 针对性QAT微调                                      │
  │    · 如果蒸馏+重新校准后仍有精度gap                             │
  │    · 仅对Phase 0标出的中等敏感层做少量epoch QAT                 │
  │    · 冻结其余层, 降低训练成本                                   │
  │                                                               │
  │  A4. [可选] 校准数据增强                                        │
  │    · 扩大校准集规模                                             │
  │    · 覆盖多场景(白天/夜晚/雨天)                                 │
  │    · 目标: 提升量化鲁棒性                                       │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
  
  ┌───────────────────────────────────────────────────────────────┐
  │  B. 硬件侧固定优化 (TRT自动 + 配置确认)                        │
  │                                                               │
  │  以下维度由TRT builder自动择优, 不纳入搜索空间,                 │
  │  但在外层构建最终engine时由TRT自动执行:                         │
  │                                                               │
  │  B1. 算子融合 (TRT自动)                                        │
  │    · Conv+BN+ReLU融合                                          │
  │    · Element-wise算子链融合                                     │
  │    · 注: 内层搜索的图分割策略决定了TRT可见的融合范围             │
  │                                                               │
  │  B2. Kernel Tactic选择 (TRT自动)                               │
  │    · 对每层自动profiling所有可用kernel实现                      │
  │    · 选择当前shape/dtype下最快的tactic                          │
  │    · 注: 内层搜索的通道数/位宽决定了可选tactic集合              │
  │                                                               │
  │  B3. 内存复用优化 (TRT自动)                                    │
  │    · 激活tensor的生命周期分析                                    │
  │    · 自动复用已释放的内存区域                                    │
  │    · 注: 内层搜索的camera_batching影响峰值内存需求              │
  │                                                               │
  │  B4. Reformatter自动插入 (TRT自动)                             │
  │    · 相邻层精度/格式不匹配时自动插入转换层                       │
  │    · 注: 内层搜索的layer_precision_override决定了               │
  │          reformatter的数量和位置                                │
  │                                                               │
  │  B5. 常量折叠与死代码消除 (TRT自动)                             │
  │    · 编译期计算固定输入的子图                                    │
  │    · 删除不可达节点                                             │
  │                                                               │
  │  B6. Graph Optimization Level配置                              │
  │    · 设为最高级别 (kOPT_LEVEL_5)                               │
  │    · 允许TRT花更多时间搜索最优tactic                            │
  │    · 构建时间更长但运行时更快                                    │
  │                                                               │
  │  B7. Workspace大小配置                                         │
  │    · 设为平台允许的最大值                                       │
  │    · 确保不因workspace不足而错过高效tactic                      │
  │                                                               │
  │  这些自动优化与内层搜索的关系:                                   │
  │  内层搜索的软件+硬件决策 → 决定ONNX图结构和标注 →               │
  │  TRT自动优化在此基础上执行 → 最终engine性能                     │
  │  即: 内层搜索间接控制了TRT自动优化的"输入",                     │
  │       而非直接搜索TRT的"行为"                                   │
  └───────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  C. 最终构建与验证                                              │
  │                                                               │
  │  Step 1: 重新构建TRT engine                                    │
  │    · 使用蒸馏/QAT后的模型                                      │
  │    · 应用Phase 1搜到的deploy_config                            │
  │    · TRT自动执行B1-B7全部优化                                   │
  │                                                               │
  │  Step 2: 最终评估                                               │
  │    · 完整验证集精度 (mAP / NDS)                                │
  │    · 端到端latency                                             │
  │    · 多场景测试 (不同agent数量1/2/3/4)                         │
  │    · 显存峰值确认                                               │
  │    · layer-by-layer profiling确认无异常                        │
  │                                                               │
  │  Step 3: 3个配置横向比较                                        │
  │    · 精度 (蒸馏后)                                              │
  │    · 延迟                                                       │
  │    · 显存                                                       │
  │    · 多场景稳定性                                               │
  │    → 选出最终部署配置                                           │
  └───────────────────────────────────────────────────────────────┘
                         │
                         ▼
              最终输出:
                pruned_model.pth
                quant_config.json
                deploy_config.json
                engine.trt
```

---

## 3. 时间与计算预算估算

```
Phase 0: 敏感度分析
  ├─ 量化敏感度: ~20层 × 5min/层 = ~1.5小时
  ├─ 剪枝敏感度: ~8块 × 10min/块 = ~1.5小时
  └─ 硬件LUT:   ~15种配置 × 2min = ~0.5小时
  合计: ~3.5小时 (一次性)

Phase 1: 嵌套搜索
  ├─ 外循环: ~20个剪枝+部署拓扑配置
  │   ├─ 约束预检查淘汰: ~8个
  │   └─ 进入内循环: ~12个
  ├─ 内循环: 每个外循环配置跑~20轮BO
  │   ├─ Level 1过滤: ~14轮被廉价评估过滤
  │   └─ Level 2真实评估: ~6轮 × 20min = ~2小时/外循环配置
  ├─ 总TRT构建次数: 12 × 6 = ~72次
  └─ 合计: ~12-15小时

Phase 2: 外层优化
  ├─ 软件侧: 蒸馏3配置 × 3小时 = ~9小时
  ├─ 重新校准+TRT构建(含自动优化): 3 × 0.5小时 = ~1.5小时
  └─ 合计: ~10-12小时

总计: ~25-30小时 (可跨2-3天完成)
```

---

## 4. 搜索空间维度汇总

### 4.1 内层搜索空间 (Phase 1参与搜索)

| 循环 | 类别 | 维度 | 变量 | 取值范围 |
|------|------|------|------|---------|
| 外循环 | 软件 | 剪枝率 | prune_rate[block_i] | 5-8个块, 每块{0, 0.1, 0.2, 0.3, 0.5} |
| 外循环 | 软件 | 检测头宽度 | head_mid_channels | {64, 128, 256} |
| 外循环 | 软件 | 剪枝指标 | prune_criterion | {L1, Taylor, FPGM} |
| 外循环 | 硬件 | 图分割 | onnx_export_strategy | {monolithic, split_head, split_agent} |
| 外循环 | 硬件 | 相机批处理 | camera_batching | {all_at_once, per_agent, sequential} |
| 外循环 | 硬件 | agent处理 | agent_batching_strategy | {pad_to_max, dynamic, bucketed} |
| 外循环 | 硬件 | 多流策略 | multi_stream_strategy | {single, branch_parallel, pipeline} |
| 外循环 | 硬件 | 内存池 | memory_pool_strategy | {isolated, shared_static, shared_streaming} |
| 内循环 | 软件 | 量化位宽 | quant_bits[block_i] | 5-8个块, 每块{4, 6, 8, FP16} |
| 内循环 | 软件 | 量化粒度 | quant_granularity | {per_tensor, per_channel} |
| 内循环 | 软件 | 对称性 | quant_symmetric | {True, False} |
| 内循环 | 软件 | 校准方法 | calibration_method | {minmax, entropy, percentile} |
| 内循环 | 软件 | 通信精度 | comm_feature_precision | {fp16, int8, int4} |
| 内循环 | 软件 | 通信压缩 | comm_spatial_compress | {none, 2x, 4x, roi} |
| 内循环 | 硬件 | 精度覆盖 | layer_precision_override[i] | 中等敏感层, {follow, fp16, fp32} |
| 内循环 | 硬件 | attention实现 | attention_impl | {trt_native, flash, custom} |
| 内循环 | 硬件 | BEV实现 | bev_transform_impl | {grid_sample, custom_plugin} |
| 内循环 | 硬件 | 时序缓存深度 | temporal_cache_frames | {0, 1, 2, 4} |
| 内循环 | 硬件 | 缓存精度 | temporal_cache_precision | {same, fp16, int8} |
| 内循环 | 硬件 | opt profile | num_agents_opt | {1, 2, 3, 4} |
| 内循环 | 硬件 | DLA offload | dla_offload_layers | 层索引子集 [Orin专用] |

### 4.2 外层固定优化 (Phase 2不参与搜索, 固定策略执行)

| 类别 | 优化项 | 执行方式 | 与内层搜索的关系 |
|------|--------|---------|----------------|
| 软件 | 知识蒸馏 | 对Top配置执行, 少量epoch | 恢复剪枝精度损失, 蒸馏后需重新校准量化 |
| 软件 | 重新校准量化 | 用搜到的量化配置重新校准 | 蒸馏改变权重分布, 需更新scale/zero_point |
| 软件 | [可选] QAT微调 | 仅中等敏感层, 少量epoch | 修复量化残余精度损失 |
| 软件 | [可选] 校准数据增强 | 扩大校准集, 多场景覆盖 | 提升量化鲁棒性 |
| 硬件 | 算子融合 | TRT builder自动执行 | 内层的图分割决定了可融合范围 |
| 硬件 | kernel tactic选择 | TRT builder自动profiling | 内层的通道数/位宽决定了可选tactic集合 |
| 硬件 | 内存复用 | TRT builder自动优化 | 内层的camera_batching影响峰值内存 |
| 硬件 | reformatter插入 | TRT builder自动插入 | 内层的precision_override决定reformatter数量 |
| 硬件 | 常量折叠/死代码消除 | TRT builder自动执行 | 内层的剪枝可能产生可折叠的子图 |
| 硬件 | graph opt level | 固定为最高级别 | 允许TRT充分搜索最优tactic |
| 硬件 | workspace大小 | 固定为平台最大值 | 确保不因workspace不足错过高效tactic |

---

## 5. 反思与修正建议

> 基于对 49 篇论文的松/紧耦合分类和搜索空间体系化总结（见 搜索空间体系化总结.md），
> 对本 pipeline 的可行性进行反思。
> 结论：**框架整体可行，但存在 6 个需要修正的问题。**

### 问题一：搜索空间分类的术语不够精确（低优先级）

pipeline 中把搜索空间分为 `x_S 软件子空间` 和 `x_H 部署配置子空间`，但按照体系化总结中的分类：

- `x_S` 实际上混合了 **B1（压缩策略）** 和 **B2（数值表示）** 两类性质不同的变量：
  - 剪枝率、剪枝准则 -> B1（改变模型有多少参数）
  - 量化位宽、量化粒度、对称性 -> B2（改变参数的数值格式）
  - 通信空间压缩 -> B1；通信特征精度 -> B2

- `x_H` 实际上全部是 **D（部署配置与执行策略）**，没有涉及 C（硬件架构）和 E（编译）

**影响**：术语层面的问题，不影响可行性，但在论文写作时需要用 B1/B2/D 的体系重新组织叙述，让读者清楚我们在搜什么、为什么这样分。

---

### 问题二：外循环 prune_criterion 导致配置数膨胀（中优先级）

当前外循环包含 `prune_criterion = {L1, Taylor, FPGM}`，每种准则在相同剪枝率下产出不同的 pruned_model，因此放在外循环是合理的。但 criterion 只有 3 个取值，它和 prune_rate 的组合导致外循环配置数膨胀了 3 倍。

**建议**：在 Phase 0 敏感度分析阶段增加一步——对比 3 种准则在固定剪枝率（如 30%）下的精度表现，锁定最优 criterion，将其从外循环搜索变量中移除。外循环配置数缩减至原来的 1/3。

**修改位置**：Phase 0 增加 criterion 预选步骤，Phase 1 外循环移除 prune_criterion 维度。

---

### 问题三：Level 1 精度预估的线性假设存在风险（高优先级）

Level 1 用 `sensitivity_map 加权精度降幅预估` 来过滤候选。这个方法假设**各层量化的精度损失是可加的**。

但从调研中可以看到（特别是 QD-BEV、Q-PETR 的经验），BEV 感知模型中：
- 位置编码层的量化损失与其他层**高度非线性耦合**（Q-PETR 报告单层量化导致 58.2% mAP 下降）
- 多传感器融合点的量化敏感度取决于上游各分支的量化程度

这意味着 sensitivity_map 的线性加权可能**系统性低估某些组合的精度损失**，导致：
- 假阳性：放进本应被过滤的差配置，浪费 Level 2 的真实评估预算
- 假阴性：漏掉好配置（虽然概率较低，因为线性假设通常低估损失）

**建议**：
1. Phase 0 增加**关键交互项的联合敏感度测试**：对 2-3 个已知高度耦合的层对（如位置编码+BEV变换、LiDAR分支+Camera分支融合点），测试它们同时量化时的联合精度损失 delta_ij
2. Level 1 精度预估公式从纯线性加权改为带交互项的版本：

```
pred_acc_drop = sum(w_i * delta_i) + sum(w_ij * delta_ij)
```

其中 delta_ij 是关键交互项的联合损失修正。

**修改位置**：Phase 0 增加交互项测试（额外约 0.5-1 小时），Phase 1 Level 1 评估公式更新。

**额外开销**：假设 3 个交互对 x 3 种位宽组合 = 9 次评估，每次 5 分钟，共约 45 分钟，Phase 0 总时间从 3.5 小时增至约 4.5 小时，一次性开销可接受。

---

### 问题四：Phase 2 蒸馏后可能需要部分重搜索（中优先级）

当前 Phase 2 假设"蒸馏不改变最优量化策略大方向"，因此只重新校准 scale/zero_point，不重新搜索。

但蒸馏改变了权重分布，可能导致：
- 某些层从"量化不敏感"变为"量化敏感"，反之亦然
- 最优的 `layer_precision_override` 可能发生变化
- TRT 选择的 kernel tactic 也可能变化，导致 Phase 1 搜到的部署配置不再最优

**建议**：Phase 2 最终验证（Step 2）中增加一个**条件触发机制**：

```
蒸馏后重新构建 TRT engine
  |
  +-- 测量延迟
  |
  +-- if |latency_after - latency_before| / latency_before > 10%:
  |       触发轻量级内循环重搜索:
  |         固定剪枝配置和部署拓扑（外循环不动）
  |         仅对量化配置和部分 D 空间变量做 5-10 轮快速 BO
  |         预估额外耗时: ~1-2 小时/配置
  |
  +-- else:
          继续使用 Phase 1 的配置
```

**修改位置**：Phase 2 Step 2 之后、Step 3 之前插入条件重搜索步骤。

---

### 问题五：E（编译）空间归属需要厘清（低优先级）

按照搜索空间体系，E（编译与代码生成）是独立子空间。pipeline 把 E 完全交给 TRT 自动处理（Phase 2 的 B1-B7），这在大多数情况下合理。

但 `attention_impl` 和 `bev_transform_impl` 这两个变量放在 D 空间（部署配置），实际上更接近 E 空间（编译/代码生成），因为它们直接决定了底层 kernel 的实现方式。

**影响**：分类问题，不影响可行性。在论文写作中需要明确：我们的 D 空间吸收了一部分 E 空间的变量（自定义 plugin 选择），这是因为在 TRT 框架下，E 空间的大部分决策已被自动化，仅剩的可控维度被归入了 D。

---

### 问题六：时间预算估算偏乐观（中优先级）

Phase 1 估算每次 Level 2 真实评估耗时 10-30 分钟（取 20 分钟），但可能遗漏了：
- ONNX 导出时遇到不支持算子需要手动处理的时间（首次尤其耗时）
- TRT engine 构建在低位宽（INT4）下可能需要更多 tactic profiling 时间
- V2X 模型的动态 shape（不同 agent 数量）需要多个 optimization profile，每个都要单独 build

**建议**：
1. 第一轮执行时预留 1.5-2 倍的时间缓冲，总计预算从 25-30 小时调整为 40-50 小时
2. 自动化脚本中加入超时机制：单次 Level 2 评估超过 45 分钟则标记为失败并跳过，不阻塞整个搜索
3. 首次运行前先手动完成一次完整的 ONNX 导出 + TRT 构建，确认所有算子均已支持或有 plugin，排除阻塞性问题

---

### 修正优先级总结

| 问题 | 严重度 | 是否阻塞可行性 | 建议处理时机 |
|------|--------|---------------|-------------|
| 1. 术语分类 | 低 | 否 | 写论文时 |
| 2. 外循环 criterion 膨胀 | 中 | 否，但浪费搜索预算 | Phase 0 中锁定 |
| **3. Level 1 精度预估线性假设** | **高** | **可能导致搜索效率大幅下降** | **Phase 0 增加交互项测试** |
| 4. 蒸馏后配置漂移 | 中 | 可能导致最终结果非最优 | Phase 2 加入条件重搜索 |
| 5. E 空间归属 | 低 | 否 | 写论文时厘清 |
| 6. 时间估算 | 中 | 否，但影响实验规划 | 加入超时机制 |

**最核心的修改**是问题 3：如果 Level 1 的精度预估不准，70-80% 的过滤率就不可靠。增加关键交互项的联合敏感度测试是成本最低（额外 45 分钟一次性开销）、收益最高的改进。
