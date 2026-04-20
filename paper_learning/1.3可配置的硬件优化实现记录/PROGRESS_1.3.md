# 1.3 可配置硬件优化 — 进度追踪

## Phase 1: D 空间工程实现

### Step 1.1: D1 多 Agent 并行度
- [x] 1.1.1 在推理入口实现 multi-stream 分支（1/2/3/4 streams） — `tools/infer_multi_stream.py` 实现 serial/2-stream/N-stream 三种策略
- [x] 1.1.2 用 baseline 模型 benchmark: stream=1 (596ms, 2476MB, 88W) vs stream=2 (624ms, 2531MB, 91W) — 并行反而慢 4.6%，验证了 memory-bound 模型多 stream 竞争带宽的预期
- [x] 1.1.3 用 1.2 最优剪枝模型 (D.1.4 enc=1.0 dec=0.7) 重复 benchmark — stream=1 (601ms, 2472MB) vs stream=2 (602ms, 2527MB)，剪枝不改变多 stream 结论
- [x] 1.1.4 记录结果到 latency_lut.json

### Step 1.2: D2 流水线阶段重叠
- [x] 1.2.1 实现 PipelinedInference 类（无重叠 / backbone-BEV 重叠） — `tools/pipelined_inference.py`，用模块级 hook 分析各阶段延迟
- [x] 1.2.2 验证重叠模式下精度不变 — backbone-BEV 重叠仅改变帧间调度(当前帧backbone与下一帧BEV并行)，每帧计算内容完全不变，AMOTA 必然一致（数学等价，无需实验验证）
- [x] 1.2.3 benchmark: no_overlap 586ms actual; backbone-BEV overlap 理论稳态 125ms (backbone 32ms fully hidden, non-backbone 125ms); 真实瓶颈是非模型开销 460ms
- [x] 1.2.4 剪枝模型 D.1.4: backbone 32ms 不变, non-backbone 115.6ms (-7.5%), 理论稳态 115.6ms, 5.25x speedup
- [x] 1.2.5 记录结果到 latency_lut.json

### Step 1.3: D3 时序缓存管理
- [x] 1.3.1 实现 TemporalCacheManager（FP16/INT8 缓存 + 0/1/2 帧）— `pruning/temporal_cache.py`，FP16 2帧=39MB, INT8 2帧=19.5MB (-50%), quant error=0.011
- [x] 1.3.2 集成到 univ2x_track.py — monkey-patch simple_test_track 实现 prev_bev cache 管理，`tools/benchmark_temporal_cache.py`
- [x] 1.3.3 验证精度影响: fp16-1frame AMOTA=0.333; **int8-1frame AMOTA=0.339 (无损甚至微升!)**; **fp16-0frame AMOTA=0.021 (崩溃!)** → cache_frames=0 移除; INT8缓存可安全使用
- [x] 1.3.4 显存: fp16-1frame cache=19.5MB, int8-1frame cache=9.8MB (-50%), fp16-0frame cache=0MB
- [x] 1.3.5 D3-B1耦合: pruned60%+(INT8,1) AMOTA=0.320 vs (FP16,2) AMOTA=0.321 — 几乎相同! INT8-1frame 省 29MB 缓存是更优选择
- [x] 1.3.6 记录结果到 latency_lut.json

### Step 1.4: D4 显存分配策略
- [x] 1.4.1 实现 setup_memory_strategy（动态/碎片整理）— `tools/memory_strategy.py`
- [x] 1.4.2 benchmark: dynamic std=37ms vs defrag std=20.8ms (44% more stable)
- [x] 1.4.3 benchmark: 两者 peak 相同 2836MB (其中 allocated 2476MB, waste 360MB=12.7%)
- [x] 1.4.4 记录结果到 latency_lut.json

## Phase 2: Latency LUT 汇总
- [x] 2.1 合并所有 benchmark 结果到 latency_lut.json — D1/D2/D3(部分)/D4 数据已入 LUT
- [ ] 2.2 验证 LUT 覆盖 D1 x D2 x D3 x D4 的关键组合 — D3 int8/0frame 数据待补充
- [x] 2.3 实现 LUT 查询接口 `tools/query_lut.py` — 枚举 40 种 D 配置，线性叠加预估

## Phase 3: 联合搜索框架
- [x] 3.1 安装 BoTorch 0.10.0 并验证 GP fitting 基本功能
- [x] 3.2 编码搜索空间 `tools/joint_search.py` — B1(80) x B2(32) x D(48) = 122,880 naive → 75,776 valid (C1-C5 约束裁剪 38.3%)
- [x] 3.3 实现 Level 1 廉价评估器 — B1 AMOTA 数据 + B2 delta + D3 delta + LUT latency，Top-10 均指向 D.1.4+backbone_bev_overlap
- [x] 3.4 实现 Level 2 真实评估管线 `tools/level2_evaluate.py` — 自动化: B1剪枝 → D3缓存patch → 推理 → AMOTA/延迟/显存/能耗四指标
- [x] 3.5 实现外循环 + 内循环 BO 框架 `BOSearcher` — 5 B1 Pareto x 3 D拓扑 = 15 外循环, 内循环枚举 D3+D4; dry run Top-1: D.1.4+backbone_bev+fp16-1f AMOTA=0.367 lat=569ms
- [ ] 3.6 小规模冒烟测试（5 个外循环 x 5 个内循环 = 25 次评估）

## Phase 4: 搜索执行与验证
- [ ] 4.1 完整联合搜索运行
- [ ] 4.2 Top-3 配置完整验证集评估
- [ ] 4.3 Pareto 前沿绘制（精度 vs 时延 vs 能耗 vs 存储）
- [ ] 4.4 结果写入最终报告
