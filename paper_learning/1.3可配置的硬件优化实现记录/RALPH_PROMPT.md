# 1.3 可配置硬件优化 — Ralph Loop 执行文档

> 本文档是 `/ralph-loop` 的输入 prompt，用于驱动 D 空间的迭代实现。
> 对应计划：`实施计划_1.3_可配置硬件优化.md`（v2）

---

## 任务目标

按照 `/home/jichengzhi/UniV2X/paper_learning/1.3可配置的硬件优化实现记录/实施计划_1.3_可配置硬件优化.md` 中的计划，**分阶段实现 D 空间的 4 个搜索维度（D1-D4）以及 Latency LUT 构建和联合搜索框架**。

每次迭代：
1. 读取 `PROGRESS_1.3.md` 确定当前进度
2. 执行当前阶段的下一个未完成步骤
3. 运行验证（代码测试 / benchmark / 精度检查）
4. 更新 `PROGRESS_1.3.md` 标记完成状态
5. 如果全部步骤完成，输出 `<promise>D_SPACE_COMPLETE</promise>`

---

## 进度追踪文件

进度记录在 `/home/jichengzhi/UniV2X/paper_learning/1.3可配置的硬件优化实现记录/PROGRESS_1.3.md`。

如果该文件不存在，先创建它，内容如下：

```markdown
# 1.3 可配置硬件优化 — 进度追踪

## Phase 1: D 空间工程实现

### Step 1.1: D1 多 Agent 并行度
- [ ] 1.1.1 在推理入口实现 multi-stream 分支（1/2/3/4 streams）
- [ ] 1.1.2 用 baseline 模型 benchmark: 4 种 stream 配置的时延/显存/功率
- [ ] 1.1.3 用 1.2 最优剪枝模型 (D.1.4 enc=1.0 dec=0.7) 重复 benchmark
- [ ] 1.1.4 记录结果到 latency_lut.json

### Step 1.2: D2 流水线阶段重叠
- [ ] 1.2.1 实现 PipelinedInference 类（无重叠 / backbone-BEV 重叠 / 全重叠）
- [ ] 1.2.2 验证重叠模式下精度不变（AMOTA 与无重叠一致）
- [ ] 1.2.3 benchmark 3 种重叠策略的稳态帧间延迟/峰值显存/功率
- [ ] 1.2.4 用剪枝模型重复 benchmark（验证流水线平衡变化）
- [ ] 1.2.5 记录结果到 latency_lut.json

### Step 1.3: D3 时序缓存管理
- [ ] 1.3.1 实现 TemporalCacheManager（FP16/INT8 缓存 + 0/1/2 帧）
- [ ] 1.3.2 集成到 univ2x_head.py 的 prev_bev 管理逻辑
- [ ] 1.3.3 验证 5 种组合的精度影响（AMOTA 对比）
- [ ] 1.3.4 benchmark 5 种配置的显存占用
- [ ] 1.3.5 测试 D3 与 B1 耦合：60% 剪枝模型 + (INT8, 1) vs (FP16, 2) 对比
- [ ] 1.3.6 记录结果到 latency_lut.json

### Step 1.4: D4 显存分配策略
- [ ] 1.4.1 实现 setup_memory_strategy（动态/静态/碎片整理）
- [ ] 1.4.2 benchmark 3 种策略的时延稳定性（std of latency over 100 frames）
- [ ] 1.4.3 benchmark 3 种策略的峰值显存
- [ ] 1.4.4 记录结果到 latency_lut.json

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
```

---

## 执行规则

### 代码位置约定

| 产出 | 路径 |
|---|---|
| 多 stream 推理 | `tools/infer_multi_stream.py` |
| 流水线重叠 | `tools/pipelined_inference.py` |
| 时序缓存管理 | `projects/mmdet3d_plugin/univ2x/pruning/temporal_cache.py` |
| 显存策略 | `tools/memory_strategy.py` |
| Latency LUT | `calibration/latency_lut.json` |
| LUT 查询 | `tools/query_lut.py` |
| 联合搜索 | `tools/joint_search.py` |
| 进度追踪 | `paper_learning/1.3可配置的硬件优化实现记录/PROGRESS_1.3.md` |
| 最终报告 | `paper_learning/1.3可配置的硬件优化实现记录/实验结果最终汇总/` |

### Benchmark 协议

每个 D 维度的 benchmark 必须报告四个指标：

| 指标 | 工具 | 单位 |
|---|---|---|
| 精度 | `tools/eval_pkl_amota.py` 或 `tools/test_with_pruning.py` | AMOTA |
| 时延 | CUDA Event 计时，warmup 3 + 正式 20 iter | ms |
| 能耗 | `nvidia-smi --query-gpu=power.draw --format=csv -l 1` 在推理期间采样 | W (乘时延得 mJ) |
| 存储 | `torch.cuda.max_memory_allocated()` + `torch.cuda.memory_reserved()` | MB |

### 关键约束（联合搜索中使用）

| 约束 | 条件 | 限制 |
|---|---|---|
| C1 | B1.ffn_ratio < 0.5 | D3.cache_frames >= 2 |
| C2 | D1 >= 3 | 峰值显存 > 阈值时需 B2=INT8 |
| C3 | D2 = 全重叠 | 需额外 ~200MB 显存 |
| C4 | D3.precision=INT8 且 B1.enc_ratio < 0.8 | 需验证缓存精度损失 |
| C5 | D4 = 静态预分配 | 显存硬上限确定 |

### 已有可复用的模型和工具

| 资产 | 路径 | 说明 |
|---|---|---|
| baseline checkpoint | `ckpts/univ2x_coop_e2e_stg2.pth` | DCN baseline |
| 剪枝微调 ckpt (D.1.4) | `work_dirs/ft_decouple_enc10_dec07/epoch_3.pth` | 最优剪枝 AMOTA 0.367 |
| 剪枝配置 (D.1.4) | `prune_configs/decouple_enc10_dec07.json` | enc=1.0 dec=0.7 |
| 模型配置 | `projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py` | 主配置 |
| 评估工具 | `tools/test_with_pruning.py` | 剪枝模型评估 |
| Latency 工具 | `tools/benchmark_latency.py` | 模块级 CUDA Event 计时 |
| 量化工具 | `tools/quick_eval_quant.py` | fake-quant 评估 |
| 敏感度数据 | `calibration/sensitivity_report.json` | 1.1 逐层敏感度 |

---

## 每次迭代的工作流

```
1. 读取 PROGRESS_1.3.md
2. 找到第一个未完成的 [ ] 步骤
3. 执行该步骤：
   a. 如果是代码实现步骤 → 编写代码 + 单元测试
   b. 如果是 benchmark 步骤 → 运行实验 + 记录数据
   c. 如果是集成步骤 → 修改现有代码 + 回归验证
4. 验证通过后 → 将 [ ] 改为 [x] + 补充实测数据
5. git add + commit（消息格式: "feat(1.3): Step X.Y.Z — 描述"）
6. 如果 Phase 1-4 全部 [x] → 输出 <promise>D_SPACE_COMPLETE</promise>
   否则 → 继续下一个步骤
```

---

## 完成信号

当所有步骤全部标记为 `[x]` 时，输出：

```
<promise>D_SPACE_COMPLETE</promise>
```

这意味着：
1. D1-D4 四个搜索维度全部实现并独立验证
2. Latency LUT 构建完成
3. 联合搜索框架可运行
4. 至少一次完整搜索执行完成
5. Top-3 Pareto 配置已验证
