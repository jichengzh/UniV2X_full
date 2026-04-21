# 1.3 可配置硬件优化 v4 — Ralph Loop 执行文档

> 对应计划：`实施计划_1.3_可配置硬件优化_2.md`（v4）
> 分三个 Phase：Phase 0（搜索维度实现）→ Phase A（工程优化底座）→ Phase B（联合搜索）

---

## 任务目标

按照 v4 计划，依次完成：
1. **Phase 0**：D3b 补充实验 + E2 数据异步预取 + D2 流水线重叠完整实现
2. **Phase A**：E3 分阶段内存释放 + E1 共享内存优化 + E4 计算图优化
3. **Phase B**：D2×D3b 联合搜索（18 次评估）+ Pareto 分析 + 最终报告

每次迭代：
1. 读取 `PROGRESS_1.3_v2.md` 确定当前进度
2. 执行当前阶段的下一个未完成步骤
3. 运行验证
4. 更新 `PROGRESS_1.3_v2.md` 标记完成状态
5. 如果全部步骤完成，输出 `<promise>V4_COMPLETE</promise>`

---

## 进度追踪文件

进度记录在 `/home/jichengzhi/UniV2X/paper_learning/1.3可配置的硬件优化实现记录/PROGRESS_1.3_v2.md`。

如果该文件不存在，先创建它，内容如下：

```markdown
# 1.3 可配置硬件优化 v4 — 进度追踪

## Phase 0: 搜索维度完整实现

### Step 0.1: D3b 补充实验
- [ ] 0.1.1 baseline 上跑 (fp16, 2帧)：AMOTA + 显存
- [ ] 0.1.2 baseline 上跑 (int8, 2帧)：AMOTA + 显存
- [ ] 0.1.3 D.1.4 剪枝模型上跑 (int8, 2帧)：验证 D3b-B1 耦合
- [ ] 0.1.4 更新 latency_lut.json 中 D3 数据

### Step 0.2: E2 数据异步预取
- [ ] 0.2.1 实现 AsyncDataPrefetcher 类
- [ ] 0.2.2 集成到推理主循环
- [ ] 0.2.3 验证：有/无预取的 AMOTA 一致
- [ ] 0.2.4 benchmark：有/无预取的端到端延迟对比

### Step 0.3: D2 流水线重叠完整实现
- [ ] 0.3.1 分析 ego agent forward 的可拆分点（extract_img_feat vs get_bev_features vs decoder vs seg_head）
- [ ] 0.3.2 实现 PipelinedEgoForward 类（backbone 在独立 stream，BEV+dec+seg 在默认 stream）
- [ ] 0.3.3 实现三种重叠模式：无重叠 / backbone-BEV 重叠 / 全重叠
- [ ] 0.3.4 验证：backbone-BEV 重叠模式下 AMOTA 与无重叠一致
- [ ] 0.3.5 benchmark：三种模式的实测稳态帧间延迟 / 峰值显存
- [ ] 0.3.6 记录结果到 latency_lut.json

## Phase A: 工程优化底座

### Step A.1: E3 分阶段内存释放
- [ ] A.1.1 实现 StagedMemoryManager 类
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
```

---

## 执行规则

### 代码位置约定

| 产出 | 路径 |
|---|---|
| 数据异步预取 | `tools/async_prefetcher.py` |
| 流水线重叠推理 | `tools/pipelined_inference.py`（已有，需扩展） |
| 分阶段内存管理 | `tools/staged_memory.py` |
| 时序缓存 | `projects/mmdet3d_plugin/univ2x/pruning/temporal_cache.py`（已有） |
| Latency LUT | `calibration/latency_lut.json`（已有，需更新） |
| 联合搜索 | `tools/joint_search.py`（已有） |
| Level 2 评估 | `tools/level2_evaluate.py`（已有） |
| 进度追踪 | `paper_learning/1.3可配置的硬件优化实现记录/PROGRESS_1.3_v2.md` |

### 已有可复用资产

| 资产 | 路径 | 说明 |
|---|---|---|
| baseline ckpt | `ckpts/univ2x_coop_e2e_stg2.pth` | DCN baseline |
| D.1.4 剪枝 ckpt | `work_dirs/ft_decouple_enc10_07/epoch_3.pth` | 最优剪枝 AMOTA 0.367 |
| D.1.4 剪枝 config | `prune_configs/decouple_enc10_07.json` | enc=1.0 dec=0.7 |
| 60% 剪枝 ckpt | `work_dirs/ft_p1_60_q2/epoch_3.pth` | FFN 60% q=2 AMOTA 0.335 |
| 60% 剪枝 config | `prune_configs/p1_ffn_60pct.json` | |
| TemporalCacheManager | `projects/.../pruning/temporal_cache.py` | INT8/FP16 缓存管理 |
| benchmark_temporal_cache | `tools/benchmark_temporal_cache.py` | D3 评估工具 |
| benchmark_latency | `tools/benchmark_latency.py` | 模块级延迟 |
| 模型配置 | `projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py` | |

### Benchmark 协议

每个步骤的 benchmark 必须报告四个指标：

| 指标 | 工具 | 单位 |
|---|---|---|
| 精度 | `tools/benchmark_temporal_cache.py` 或 `tools/level2_evaluate.py` | AMOTA |
| 延迟 | CUDA Event 计时，warmup 3 + 正式 20 iter | ms |
| 能耗 | `nvidia-smi --query-gpu=power.draw` | W (乘时延得 mJ) |
| 存储 | `torch.cuda.max_memory_allocated()` | MB |

### 关键依赖链

```
0.1 D3b 补充 ──────────────────────────────────────────► B.1 联合搜索
0.2 E2 预取 ──► 0.3 D2 流水线 ──────────────────────────► B.1 联合搜索
                A.1 E3 内存释放 ──► A.2 E1 共享内存 ──► B.1 联合搜索
                                                    A.3 E4 图优化 ─┘
```

0.1 和 0.2 可并行；0.3 依赖 0.2；A.1 可与 Phase 0 并行。

---

## 每次迭代的工作流

```
1. 读取 PROGRESS_1.3_v2.md
2. 找到第一个未完成的 [ ] 步骤
3. 执行该步骤：
   a. 代码实现步骤 → 编写代码 + 单元测试 / 冒烟验证
   b. benchmark 步骤 → 运行实验 + 记录数据
   c. 集成步骤 → 修改现有代码 + 回归验证
4. 验证通过后 → 将 [ ] 改为 [x] + 补充实测数据
5. git add + commit（消息格式: "feat(1.3v4): Step X.Y.Z — 描述"）
6. 如果 Phase 0 + A + B 全部 [x] → 输出 <promise>V4_COMPLETE</promise>
   否则 → 继续下一个步骤
```

---

## 完成信号

当所有步骤全部标记为 `[x]` 时，输出：

```
<promise>V4_COMPLETE</promise>
```

这意味着：
1. D2 流水线重叠已完整实现并验证（三种模式可切换）
2. D3b 全部 5 种缓存配置已有实测 AMOTA 数据
3. E1-E4 工程优化底座已实现
4. 18 次联合搜索评估已完成
5. Pareto 前沿已绘制，Top-3 已验证
6. 最终报告已写入
