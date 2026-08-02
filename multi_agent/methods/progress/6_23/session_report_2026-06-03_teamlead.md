# 本次会话进度报告 (team-lead 视角)
**时间**: 2026-06-03 ~08:28
**会话时长**: ~155 分钟
**团队**: `sw-hw-cooptim` (4 agent: supervisor / data-orchestrator / sw-optimizer / hw-optimizer)

## 已完成的工作

### 1. 搭建 4-agent 持久化团队
- `.claude/agents/{supervisor,data-orchestrator,sw-optimizer,hw-optimizer}.md` — 4 个 opus 子智能体定义(角色+纪律+项目目标内嵌)。
- `multi_agent/methods/progress/team_charter_v1.md` — 团队宪章(最终目标/5指标/纪律红线/协作协议/启动模板)。
- TeamCreate 容器 `sw-hw-cooptim`。

### 2. P0-1 耦合陷阱 — 完全闭环 (论文级铁证)
- p25(48/96/192 非对齐)collab2 真测: 更小模型却比 base 慢 **2.28×**, INT8 仅 1.06×(对齐档 1.25-1.57×), 能耗仅省 10.3%(对齐 30-52%)。
- 机制经 IProfiler + IEngineInspector **tactic 级坐实**: stage0 grouped-conv kernel cliff(p25=implicit_gemm 通用 kernel 没跑 INT8, p50=direct_group 专用)。三假设(padding/reformat/INT8算术)全排除。
- 卖点升级: "耦合非解析、LUT 预测不了→必须真测" vs APQ/HAQ。宪章§0 + background§0 已勘误(supervisor 独立复核 271 行 profile)。

### 3. 完整点核验 + 主表 rebuild
- 核验: 严格完整点 = 40(collab2 28 + E4 11 + E3 1), "6"是陈旧计数; **但有效独立锚点 ≈6-8**(per-stage 退化簇)。
- 主表 rebuild 45 行, supervisor 并表核验通过。

### 4. AP 激活 — A/B/E 三条免训路径全部跑完, 均为负结果 ★
- **A 剪枝×forced-INT8**: 无悬崖。
- **B head 拆护栏**: 近无损。
- **E 难子集**: floor(均匀平移), 非 trade-off(负结果)。
- ⇒ **触发兜底: 需训练更难模型 V2X-ViT** 才能让 AP 真正进入前沿 trade-off。**待用户决策。**

### 5. Orin 跨硬件 (P0-3, 进行中)
- 管线打通: 首点 base fp16@30W = 48.68ms/7.13W/348mJ vs 4090 462mJ(Orin 更省能, perf/watt 卖点成立)。
- 勘误: nvpmodel 实际可用(功率档轴复活, ISS-016); Orin INT8 build 攻克(header 重写)。
- INT8 AP 复用闸门 = Option A 条件版(字节一致即满足+透明标记+抽测, ISS-017)。

## 当前状态
- ✅ 已验证: P0-1 全闭环 / 完整点核验 / 主表 45 行 / A·B·E sw AP / 团队文档。
- ⏳ 进行/待: P0-3 Orin(in_progress) / P0-2 [32,64,136] finetune(状态需确认) / A·hw·B·hw 延迟配对(#8/#10) / V2X-ViT 兜底(待批)。

## 遗留问题 / 下一步
1. **★用户决策: AP 激活兜底 = 训练 V2X-ViT?**(免训路径全负, 这是"投训练算力"闸门)。
2. **GPU 利用率低**: GPU 0/1 全空闲, 仅 GPU7 小占用; 兜底+剩余 hw builds 应并行铺到 0/1/7。
3. P0-2 finetune 收敛状态核实(nvidia-smi 未见大显存训练进程)。
4. A·hw/B·hw 延迟配对(消融点, 补全 5 指标)。

## 关键产物
| 路径 | 状态 |
|---|---|
| `.claude/agents/*.md` (4) | ✅ |
| `multi_agent/methods/progress/team_charter_v1.md` | ✅ |
| `results/P0_1_p25_int8_trap_{4090.csv,verdict.md,layer_profile.csv}` | ✅ 已核验 |
| `multi_agent/data/dataset_v2.{csv,parquet}` (45 行) | ✅ rebuild |
| `multi_agent/methods/design/ap_activation_strategy_v1.md` | ✅ (A/B/E 负结果待回填) |
| `multi_agent/methods/progress/issues_log_v1.md` (ISS-001~017) | ✅ |
| `scripts/phase2/e6_orin_tegrastats_energy.py` | ✅ 上机核对 |
