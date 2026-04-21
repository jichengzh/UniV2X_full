# 1.3 可配置硬件优化 v4 — 进度追踪（诚实修正版）

## Phase 0: 搜索维度完整实现

### Step 0.1: D3b 补充实验 ✅ 全部完成
- [x] 0.1.1 fp16-2f: AMOTA=0.333, cache=39.1MB — 与 1-frame 完全相同
- [x] 0.1.2 int8-2f: AMOTA=0.339, cache=19.5MB — 与 1-frame 完全相同
- [x] 0.1.3 重要发现: 2-frame 无精度收益, D3b 锁定为 1 帧
- [x] 0.1.4 更新 latency_lut.json: D3 完整数据

### Step 0.2: E2 数据异步预取 ✅ 全部完成
- [x] 0.2.1 实现 AsyncDataPrefetcher — `tools/async_prefetcher.py`
- [x] 0.2.2 集成到推理主循环
- [x] 0.2.3 验证: AMOTA 一致
- [x] 0.2.4 benchmark: 无预取 619.9ms vs 有预取 618.6ms (0.2% 收益)

### Step 0.3: D2 流水线重叠
- [x] 0.3.1 分析完成: extract_img_feat 和 get_bev_features 是拆分点
- [x] 0.3.2 实现 PipelinedGetBevs 类 — `tools/pipelined_ego_forward.py`，monkey-patch get_bevs() 拆分 backbone 和 BEV 到不同 stream
- [x] 0.3.3 实现 none / backbone_bev 两种模式 (full 降级为 backbone_bev)
- [ ] 0.3.4 验证: backbone-BEV 重叠 AMOTA 一致
- [ ] 0.3.5 benchmark: 实测稳态帧间延迟 / 峰值显存
- [ ] 0.3.6 记录结果到 latency_lut.json

## Phase A: 工程优化底座

### Step A.1: E3 分阶段内存释放 ✅ 实验完成（结论: PyTorch 端无效）
- [x] A.1.1 实现 StagedMemoryManager — `tools/staged_memory.py`
- [x] A.1.2 注册 5 个阶段边界 hook
- [x] A.1.3 验证 AMOTA 不变
- [x] A.1.4 benchmark: 0MB 释放, PyTorch CUDA allocator 已自动管理

### Step A.2: E1 共享内存优化
- [x] A.2.1 cudnn.benchmark=True 配置
- [x] A.2.2 CUDA allocator 参数 (defrag)
- [ ] A.2.3 benchmark: cudnn.benchmark 对比延迟（**需要实际跑对比实验**）

### Step A.3: E4 计算图优化
- [ ] A.3.1 用 onnx-simplifier 简化 ONNX 图（**需要实际执行**）
- [ ] A.3.2 记录优化前后的算子数量和 engine 大小

## Phase B: 联合搜索

### Step B.1: 搜索执行
- [ ] B.1.1 在优化后基线上运行评估（**需要用 level2_evaluate.py 真正跑**）
- [ ] B.1.2 收集四指标

### Step B.2: 分析与报告
- [ ] B.2.1 Pareto 前沿
- [ ] B.2.2 Top-3 配置
- [ ] B.2.3 写入最终报告（**文件不存在，需要创建**）
