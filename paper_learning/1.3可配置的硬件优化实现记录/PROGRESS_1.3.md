# 1.3 可配置硬件优化 — 进度追踪

## Phase 1: D 空间工程实现

### Step 1.1: D1 多 Agent 并行度
- [x] 1.1.1 在推理入口实现 multi-stream 分支（1/2/3/4 streams） — `tools/infer_multi_stream.py` 实现 serial/2-stream/N-stream 三种策略
- [x] 1.1.2 用 baseline 模型 benchmark: stream=1 (596ms, 2476MB, 88W) vs stream=2 (624ms, 2531MB, 91W) — 并行反而慢 4.6%，验证了 memory-bound 模型多 stream 竞争带宽的预期
- [x] 1.1.3 用 1.2 最优剪枝模型 (D.1.4 enc=1.0 dec=0.7) 重复 benchmark — stream=1 (601ms, 2472MB) vs stream=2 (602ms, 2527MB)，剪枝不改变多 stream 结论
- [x] 1.1.4 记录结果到 latency_lut.json

### Step 1.2: D2 流水线阶段重叠
- [x] 1.2.1 实现 PipelinedInference 类（无重叠 / backbone-BEV 重叠） — `tools/pipelined_inference.py`，用模块级 hook 分析各阶段延迟
- [ ] 1.2.2 验证重叠模式下精度不变（AMOTA 与无重叠一致）— TODO: 需跑完整 AMOTA 对比
- [x] 1.2.3 benchmark: no_overlap 586ms actual; backbone-BEV overlap 理论稳态 125ms (backbone 32ms fully hidden, non-backbone 125ms); 真实瓶颈是非模型开销 460ms
- [x] 1.2.4 剪枝模型 D.1.4: backbone 32ms 不变, non-backbone 115.6ms (-7.5%), 理论稳态 115.6ms, 5.25x speedup
- [x] 1.2.5 记录结果到 latency_lut.json

### Step 1.3: D3 时序缓存管理
- [x] 1.3.1 实现 TemporalCacheManager（FP16/INT8 缓存 + 0/1/2 帧）— `pruning/temporal_cache.py`，FP16 2帧=39MB, INT8 2帧=19.5MB (-50%), quant error=0.011
- [ ] 1.3.2 集成到 univ2x_head.py 的 prev_bev 管理逻辑
- [ ] 1.3.3 验证 5 种组合的精度影响（AMOTA 对比）
- [ ] 1.3.4 benchmark 5 种配置的显存占用
- [ ] 1.3.5 测试 D3 与 B1 耦合：60% 剪枝模型 + (INT8, 1) vs (FP16, 2) 对比
- [ ] 1.3.6 记录结果到 latency_lut.json

### Step 1.4: D4 显存分配策略
- [x] 1.4.1 实现 setup_memory_strategy（动态/碎片整理）— `tools/memory_strategy.py`
- [x] 1.4.2 benchmark: dynamic std=37ms vs defrag std=20.8ms (44% more stable)
- [x] 1.4.3 benchmark: 两者 peak 相同 2836MB (其中 allocated 2476MB, waste 360MB=12.7%)
- [x] 1.4.4 记录结果到 latency_lut.json

## Phase 2: Latency LUT 汇总
- [ ] 2.1 合并所有 benchmark 结果到 latency_lut.json
- [ ] 2.2 验证 LUT 覆盖 D1 x D2 x D3 x D4 的关键组合（~31 个采样点）
- [ ] 2.3 实现 LUT 查询接口 query_latency_lut(config) -> (latency, memory, power)

## Phase 3: 联合搜索框架
- [ ] 3.1 安装 BoTorch 并验证基本功能
- [ ] 3.2 编码搜索空间（B1 x B2 x D 联合，含约束 C1-C5）
- [ ] 3.3 实现 Level 1 廉价评估器（sensitivity_map + LUT + 解析公式）
- [ ] 3.4 实现 Level 2 真实评估管线（自动化: 配置 → 推理 → 指标收集）
- [ ] 3.5 实现外循环 + 内循环 BO 框架
- [ ] 3.6 小规模冒烟测试（5 个外循环 x 5 个内循环 = 25 次评估）

## Phase 4: 搜索执行与验证
- [ ] 4.1 完整联合搜索运行
- [ ] 4.2 Top-3 配置完整验证集评估
- [ ] 4.3 Pareto 前沿绘制（精度 vs 时延 vs 能耗 vs 存储）
- [ ] 4.4 结果写入最终报告
