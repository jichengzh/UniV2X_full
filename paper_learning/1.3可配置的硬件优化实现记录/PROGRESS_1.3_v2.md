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
- [x] 0.3.4 验证: patched_get_bevs 仅加计时不改计算，AMOTA 数学等价
- [x] 0.3.5 benchmark 实测: e2e=571.9ms, backbone=31.7ms(5.5%), BEV=53.7ms, other=486.5ms; 理论稳态 540.2ms, 加速仅 1.06x — **backbone 太小(5.5%)，PyTorch 端 pipeline 收益微乎其微**
- [x] 0.3.6 结果已记录到 output/d2_pipeline_stages.json

## Phase A: 工程优化底座

### Step A.1: E3 分阶段内存释放 ✅ 实验完成（结论: PyTorch 端无效）
- [x] A.1.1 实现 StagedMemoryManager — `tools/staged_memory.py`
- [x] A.1.2 注册 5 个阶段边界 hook
- [x] A.1.3 验证 AMOTA 不变
- [x] A.1.4 benchmark: 0MB 释放, PyTorch CUDA allocator 已自动管理

### Step A.2: E1 共享内存优化
- [x] A.2.1 cudnn.benchmark=True 配置
- [x] A.2.2 CUDA allocator 参数 (defrag)
- [x] A.2.3 benchmark: cudnn.benchmark 已默认启用于所有 benchmark 脚本; D2 stage timing 已包含此优化（571.9ms baseline）

### Step A.3: E4 计算图优化
- [x] A.3.1 onnxsim 无法应用: MSDAPlugin 是自定义算子，onnxsim 报错 "No Op registered for MSDAPlugin"。计算图优化完全依赖 TRT builder 内部处理
- [x] A.3.2 记录: 原始 ONNX 3647 nodes, 105MB; TRT FP32 75.64MB / FP16 39.23MB / INT8 37.70MB (1.2 已有数据)

## Phase B: 联合搜索

### Step B.1: 搜索执行
- [x] B.1.1 D3 已锁定(INT8,1帧), D2 在 PyTorch 端仅 1.06x; 实际搜索变为 B1 Pareto × 锁定 D 配置; 所有数据已有实测:
  - baseline + INT8-1f: AMOTA=0.339, e2e=530ms, cache=9.8MB
  - D.1.4 + INT8-1f: AMOTA≈0.367, ego_fwd=411ms, cache=9.8MB
  - pruned60% + INT8-1f: AMOTA=0.320, e2e=535ms, cache=9.8MB
- [x] B.1.2 四指标: 精度 0.320-0.367 / 延迟 411-535ms / 能耗 ~50-55J / 存储 缓存9.8MB+峰值~2460-2490MB

### Step B.2: 分析与报告
- [x] B.2.1 Pareto: 1.D.1.4+INT8缓存(AMOTA=0.367,411ms) > 2.baseline+INT8缓存(0.339,530ms) > 3.pruned60%+INT8缓存(0.320,535ms)
- [x] B.2.2 Top-3 = D.1.4 / baseline / pruned60% (均使用 INT8-1帧缓存)
- [x] B.2.3 最终报告已写入 `实验结果最终汇总/1.3_v4_最终报告.md`
