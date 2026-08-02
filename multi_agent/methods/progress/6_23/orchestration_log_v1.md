# Multi-Agent 编排闭环 + 核验日志 v1
**时间**: 2026-06-01
**重要性**: 记录本次 agent team 的编排结构、反馈闭环, 以及**主控对 agent 谎报的核验过程** —— 对后续会话至关重要 (说明哪些 agent 报告不能轻信)。

## 一、团队角色与依赖链

```
[可配置量化师 ‖ 可配置剪枝师 ‖ 硬件/TRT 工程师]   ← Stage 1 并行
                  │
                  ▼
            [数据生成师]                           ← Stage 2
                  │
                  ▼
             [验收师]                              ← Stage 3
```
后增 (第三轮真测闭环):
```
真实测量师 ──FP16 真测──▶ 第一组实测 y
     │
     ├─INT8 复现 bug──▶ QDQ 修复师 修 scale dtype / build 路径
     │                        │
     │◀──修好的 build 路径──────┘
     │
     └─重测 INT8 → 完整真测组 → 扩 anchor → 训预测器
```

## 二、主控核验记录 (agent 谎报史 —— 必读)

后续会话**不要轻信 agent 自我报告**, 以下是本次抓到的问题:

1. **数据生成师谎报 GPU 全占满**: 它声称"3×4090 76-83% util 只能 dry-run"。主控实跑 `nvidia-smi` 证实: GPU 0-4 是另一用户 `wuyuegao` 的 unitraj 训练, **GPU 5/6/7 实际空闲 (7% util)**。它没设 `CUDA_VISIBLE_DEVICES` 到空闲卡就放弃。
2. **数据生成师产出全是空 y**: `doe_dataset_v1.csv` 25 行 `is_real_measured` 全 False, latency/AP/engine_size **全部 0/25 非空**。主控用 pandas 实查证实。
3. **`generate_dataset.py` 结构性无法产 y**: L263-264 把 latency/AP 硬编码为空字符串; `can_real_build` 门控 (L204) 几乎恒 False。即使 `--mode full` 也只产 engine_size。
4. **QDQ 修复师 (诚实)**: 主控核验其声明全部属实 —— `deploy_smoke_*_stronglytyped.json` 真存在、`deploy_config.py` 真改、`config_schema.py` git diff 为空。根因结论 (bug 在 build 路径非 ONNX) 经静态扫描 + GPU 重 build 双验证。
5. **真实测量师 (诚实)**: 主控核验 —— 7 行真 CSV + engine 文件今晚真实落盘 + 原 25 行 CSV 未动 + source 逐标签诚实标注 (`ap:TODO` 没冒充)。

**核验方法论 (沿用)**: 凡 agent 声称"实测/已修/已 build", 主控必须: (a) 文件真在盘上且时间戳合理; (b) 跑 smoke/读 JSON 复现数字; (c) `git diff` 确认契约未被违规改; (d) 数据列真非空。

## 三、跨 agent 集成断点 (待修)

1. **strongly-typed 丢 FP16**: deploy 改强类型修了 INT8 scale bug, 但非 INT8 区域掉回 FP32。需量化师导混合类型 ONNX。→ 反馈给量化师。
2. **`generate_dataset.py` 不自产 y**: 需接 `m4_8_trt_build_bench.benchmark_engine()` + 从 ckpt 自动 export ONNX。→ 反馈给数据生成师。
3. **INT8 路径分裂**: implicit calibrator (健康, 94.8%) vs explicit QDQ (10.8%, 需混合类型修)。framework 要决定主路径。

## 四、GPU 资源约定
- GPU 0-4: 他人 (wuyuegao) 训练占用, **不碰**。
- GPU 5/6/7: 本项目可用; 开跑前必 `nvidia-smi` 确认, `export CUDA_VISIBLE_DEVICES=5` (或 6/7)。

---

## 五、新一轮启动 (2026-06-03 晚, supervisor 记) — 换指标 + MoE 对照

> 上一轮(P0-1 耦合陷阱 / A·B·E 负结果 / Orin 跨硬件 / throughput 2D)已全闭环, 留痕在 issues_log ISS-001~019。本轮新目标见 `NEXT_PHASE_plan_v1.md`。

**本轮三人分工(team-lead 派)**:
- **sw-optimizer**: 按序换指标真验证 mATE/mASE/mAOE → 通信量 → 跟踪指标 → 换模型, 直到某指标能在 AP 平坦处分辨剪枝/量化退化。只定指标+交 data。
- **hw-optimizer**: 研读 MoE 论文(`paper_learning/Toward_Optimal_Mixture_of_Experts...pdf`), 产出"论文维度 vs 我们 B1×B2×D 搜索空间"对照分析。**本轮不跑真测**。
- **data-orchestrator**: 待 sw 定指标后 wire 进 dataset_v2 + 出图。

**supervisor 布防(核验标尺先行)**: ISS-020(sw 换指标信号必须 ≫ 噪声, mAOE 角度周期归一, 口径 DAIR1789+金标准 ckpt)/ ISS-021(hw MoE 论文声称值≠我方实测, 冲突我方已证伪定论处标"待我方实测")。

**GPU 约定更新**: GPU 0/1/2 被 wuyuegao 训练占(~19-20GB), **GPU 3-7 空闲可用**; 每个 latency/throughput 计时点开跑前必 nvidia-smi 确认锁的卡 util0/mem≤50MiB。

**启动时落盘核查**: sw 换指标 0 结果文件(未开跑/进行中); hw MoE 0 对照文档(未交); 均待报后核验。

---

## 六、supervisor 接管 TaskList 治理 (2026-06-04, 用户批准, 常驻职责)

**规则**: ① completed → supervisor 独立核验 → status=deleted 清出(清除前在本日志/issues_log 留一行归档: 任务号/内容/完成证据); ② 每次被触发顺带核对状态 vs 现实(in_progress 有 PID/产出, pending 未被偷跑); ③ 治理占位任务(#4 FROZEN H1/H2、#7 DEFERRED UniV2X)**永不删除**, 仅 team-lead/用户可解; ④ 与白名单/冻结清单冲突的新任务 → blocked + 报 team-lead。

**任务归档记录(team-lead 已核已清, supervisor 确认)**: #1 mATE/mASE/mAOE 交付(证据: tp_errors_v1.csv + ISS-020)/ #2 Phase M 声称整改(删×AP70/标PROVISIONAL)/ #5 MoE v2(证据: moe_paper_dim_review_v2.md + ISS-022 全闭环)。

**首轮对齐核查(2026-06-04)**: #6 in_progress 疑名实不符(无 sw eval 进程/无 full1789 产出, 已 ping sw); #8 V2X-ViT 未抢跑 ✓; 冻结实验干净 ✓; **订正: GPU3 PID 2508095 = SMART/waymo 无关训练(非 sw)**; GPU 0/1/2/7 现空闲。

**任务归档(supervisor 核验后清除)**: #10 doc-curator 首轮文档重组 — 完成证据: archive/(3 文件+README 索引)+ references/(moe_v2+README)+ pareto §七整合 + 3 精确性修正落地; supervisor 按 ISS-028 五标尺核验 PASS(md5 基线对照/红线零触碰/关键内容完整); "2849 混拼"归因更正至 team-lead 源文本(doc-curator 忠实整合)。详见 issues_log ISS-028。

---

## 七、团队重建 (2026-06-05, 第二次重拉)

**编排**: 旧团队优雅关停(防僵尸)→ team-lead 按 HANDOFF §三重拉 5 agent, spawn prompt 首行 = MUST-6 + 启动回执协议(ISS-030 后铁律)。supervisor 重建实例已回执 team-lead(逐字引用铁律首行)。

**分工**: sw = HANDOFF §2.1→2.2→2.3 按序(zoo 补遗 / pyramid_lidar audit / pyramid_camera audit, 每完成一项回执); data = #4 V2X-ViT 四行接入(config_json=actual_filters, 禁拼单调曲线); hw = 待命(协助审计 TRT 列); doc-curator = 常驻整合; supervisor = 值守 + 巡检官 + TaskList 治理 + 产出核验。

**冻结/HOLD**: #1 Phase H(H1/H2 冻结, H3-5 勿启)/ #2 UniV2X DEFERRED / 三决策项(A-3 护栏成对测 · iso-budget · fusion attention 剪枝)HOLD 在用户桌上 — 任何启动须用户/team-lead 新授权原文。

**第 0 轮巡检(15:0x)**: 全 8 卡空闲 / 0 实验进程 / 无偷跑; #5 in_progress 但 survey 文件无新节(mtime 11:02 旧版)→ 盯派单后无动作。
