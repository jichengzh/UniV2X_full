# 1.3 可配置硬件优化 v4 — 进度追踪

## Phase 0: 搜索维度完整实现

### Step 0.1: D3b 补充实验
- [ ] 0.1.1 baseline 上跑 (fp16, 2帧)：AMOTA + 显存
- [ ] 0.1.2 baseline 上跑 (int8, 2帧)：AMOTA + 显存
- [ ] 0.1.3 D.1.4 剪枝模型上跑 (int8, 2帧)：验证 D3b-B1 耦合
- [ ] 0.1.4 更新 latency_lut.json 中 D3 数据

### Step 0.2: E2 数据异步预取
- [x] 0.2.1 实现 AsyncDataPrefetcher 类 — `tools/async_prefetcher.py`，独立 CUDA stream 异步预取+递归 tensor GPU 搬运
- [ ] 0.2.2 集成到推理主循环
- [ ] 0.2.3 验证：有/无预取的 AMOTA 一致
- [ ] 0.2.4 benchmark：有/无预取的端到端延迟对比

### Step 0.3: D2 流水线重叠完整实现
- [x] 0.3.1 分析完成: get_bevs() 内部 extract_img_feat(backbone+FPN) 和 get_bev_features(BEV encoder) 是拆分点; seg_head 在 UniV2X.forward_test 中调用（与 decoder 串行）
- [ ] 0.3.2 实现 PipelinedEgoForward 类（backbone 在独立 stream，BEV+dec+seg 在默认 stream）
- [ ] 0.3.3 实现三种重叠模式：无重叠 / backbone-BEV 重叠 / 全重叠
- [ ] 0.3.4 验证：backbone-BEV 重叠模式下 AMOTA 与无重叠一致
- [ ] 0.3.5 benchmark：三种模式的实测稳态帧间延迟 / 峰值显存
- [ ] 0.3.6 记录结果到 latency_lut.json

## Phase A: 工程优化底座

### Step A.1: E3 分阶段内存释放
- [x] A.1.1 实现 StagedMemoryManager 类 — `tools/staged_memory.py`，4 个阶段边界 hook + 显存释放 + 统计
- [ ] A.1.2 在 ego agent forward 的 4 个阶段边界插入释放点
- [ ] A.1.3 验证 AMOTA 不变
- [ ] A.1.4 benchmark：实施前后的峰值显存对比

### Step A.2: E1 共享内存优化
- [ ] A.2.1 启用 cudnn.benchmark=True
- [ ] A.2.2 配置 CUDA allocator 优化参数
- [ ] A.2.3 benchmark：延迟变化

### Step A.3: E4 计算图优化
- [ ] A.3.1 用 onnx-simplifier 简化 ONNX 图
- [ ] A.3.2 记录优化前后的算子数量和 engine 大小

## Phase B: 联合搜索

### Step B.1: 搜索执行
- [ ] B.1.1 在优化后基线上跑 6 种 D 配置 × 3 种 B1 配置 = 18 次评估
- [ ] B.1.2 收集每次评估的四指标（AMOTA / 延迟 / 能耗 / 存储）

### Step B.2: 分析与报告
- [ ] B.2.1 绘制 Pareto 前沿（精度 vs 延迟 vs 存储）
- [ ] B.2.2 确定 Top-3 配置
- [ ] B.2.3 Top-3 完整验证集评估
- [ ] B.2.4 写入最终报告
