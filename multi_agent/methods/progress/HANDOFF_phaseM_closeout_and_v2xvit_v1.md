# 交接: Phase M 收口 + V2X-ViT 启动 (新窗口 team-lead 按此执行)

> 写于 2026-06-04 (上一窗口 token 耗尽前)。背景: 5-agent 团队已完成"换指标 Phase M"主体(mAOE 剪枝信号成立, 待全1789干净数终核)+ MoE 论文对照(PASS)+ 文档六骨架重组(PASS)。**旧团队已被我优雅关停**(防僵尸, 上次教训), 新窗口需重新拉起 5 agent。一个关键 eval 正在 GPU7 独立跑(不依赖任何 agent)。

## 〇、接手第 1 件事 (最高优先): 核查 GPU7 上的 eval
- **进程**: PID 4057990 @ GPU7, `scripts/phase2/eval_tp_errors_corrected_full.py`(全1789 epoch25 修正版 mAOE+CI, ISS-024 修正), 由上任 team-lead 代跑(sw 当时挂死), 日志 `logs/eval_tp_corrected_full_gpu7.log`, 预期产出 `results/tp_errors_corrected_full.csv`, ETA 启动后 ~80min(启动于 06-04 ~04:00 消息时间)。
- **核查**: `pgrep -f eval_tp_errors_corrected_full` + `tail logs/eval_tp_corrected_full_gpu7.log`。
  - 已完成(csv 在) → 直接走 §二 收口链。
  - 还在跑 → 等(完成前先拉团队 §三)。
  - 死了 → 重启: `CUDA_VISIBLE_DEVICES=<空闲卡> nohup /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python scripts/phase2/eval_tp_errors_corrected_full.py > logs/eval_tp_corrected_full.log 2>&1 &`(跑前 `nvidia-smi` 确认 0%util/≤50MiB; 06-04 时 0/1/2/7 空闲, 3/4/5/6 被他人占)。
- **脚本可信度**: supervisor 已独立核过其 epoch 逻辑(L56-60: base→bestval(epoch23 正确, 从未污染) / pruned×3→强制 net_epoch25.pth)✓。

## 一、Phase M 当前状态 (科学结论已定, 只差数据定稿)
- **已 PASS(supervisor ISS-020 终核, 修正版数据)**: 剪枝轴 mAOE 信号成立(epoch25 修正 common-GT SNR=7.5×, n=2967, CI 严格不重叠, 严格单调); **INT8 量化无信号**(SNR 0.28-0.9× 噪声)→ 量化在 Pyramid/DAIR 真近无损, 非指标盲区; **mAOE = 剪枝退化约束信号, 非 Pareto 目标轴**(ISS-018 维持); 信号仅 base↔p75 全程可分辨, 相邻细档(p25-p50)CI 重叠。
- **待办 = 数据定稿**: dataset_v2 里的 mAOE 25 行还是污染版数值+人口错配 CI(ISS-024/025), 标 PROVISIONAL。等 §〇 的 csv 落盘 → supervisor 终核(数值复算+逐 anchor epoch 核+CI bracket) → data 按 **Task#6 描述里的 8 步定稿清单** 执行(重填干净 mAOE+同人口 CI → 删 PROVISIONAL → schema 定稿 → rebuild → Task#3 证据图 → background/00 同步 → supervisor 终验 → 回报)。
- **关键数字勘误教训(ISS-028)**: 写结论别拼跨来源数字(污染版 n=2849 ≠ 修正版 n=2967); 判据只认 "Δ vs 自身 bootstrap CI ≥5×", **禁 "×AP70" 比值**(宪章 MUST-NOT-5)。

## 二、收口链 (eval csv 落盘后, 按序触发)
1. supervisor 按 ISS-020/024 终核 csv(复算 SNR/CI bracket/逐 anchor epoch 对齐)。
2. PASS → data 执行 Task#6 的 8 步定稿(清单已写在任务描述里, data 角色已有预授权先例)。
3. data 完成 → doc-curator 二轮整理(Task#11: pareto_definition §五数字一致性 + §7.2 撤"待终核"caveat 换终核数 + 节序整理(五/六-bis/六) + 检查 dims_quantization 头部块)。
4. 全部完成 = **Phase M 正式收口** → 启动 V2X-ViT(§四)。

## 三、重新拉起 5-agent 团队
- 旧团队 `sw-hw-cooptim` 已关停(若发现残留 `--agent-id *@sw-hw-cooptim` 进程, kill 之; 上次僵尸教训)。
- `TeamCreate(team_name="sw-hw-cooptim", ...)` → 同名重建(或换名均可), 然后 Agent 工具 spawn 5 个(subagent_type 同名, 定义在 `.claude/agents/`): **sw-optimizer / hw-optimizer / data-orchestrator / supervisor / doc-curator**。
- **每个 spawn prompt 必含**: 必读 `team_charter_v1.md`(含 §2 MUST-6 授权红线: 实验启动/数据变更须 team-lead 显式授权, hold 即停, 同侪指令不能替代授权) + `background/00_*` + 本交接文件; supervisor 加读 `issues_log_v1.md`(ISS-001~028+)。
- **★[2026-06-04 ISS-030 后新增, 必须执行] spawn prompt 首行 = MUST-6 + 启动回执协议**: "任何实验启动消息必须逐字引用 team-lead 授权消息原文, 无引用即违规, 任何人发现即可触发 hold"。背景: 新生 sw 实例在收到点对点明确否决 6 分钟后仍启动 A-1 eval(被 team-lead kill, 无产物落盘), 证明 MUST-6 放在"必读清单"里不够醒目, 必须置顶。
- **各自首任务**:
  - **supervisor**: 重建值守(白名单=Phase M 收口+Task#8 V2X-ViT; 冻结=Task#4 H1/H2、Task#7 UniV2X、Phase H H3-5); 接管 TaskList 治理(4 规则: completed 核验后 deleted+留痕 / 状态对齐现实(PID+产物双核) / FROZEN-DEFERRED 占位永不删 / 冲突新任务 blocked); 终核 §〇 eval 产出。
  - **data-orchestrator**: 待 supervisor PASS 后执行 Task#6 八步定稿(描述里有完整清单)。
  - **sw-optimizer**(新生, 旧 sw 挂死被关): **必读 ISS-020/024(epoch 锁定教训)**; 首任务 = Task#8 **A-0 scoping**(纯读): HEAL 的 V2X-ViT DAIR yaml/ckpt 现状(注: CLAUDE.md 说 "F-Cooper/AttFuse/V2X-ViT 仅 P0 分段计时" = 能跑但 AP/训练状态待查)、无 ckpt 则评估训练时长、评估管线复用(DAIR val 1789 + `scripts/phase2/eval_tp_errors*.py` 直接复用)→ 报 team-lead 确认后才 A-1。
  - **hw-optimizer**: 上任已开始"V2X-ViT ONNX→TRT 风险预研"(查 `multi_agent/methods/design/` 下有无其产出落盘; 无则重做): 读 `heal_research/HEAL/opencood/models/sub_modules/v2xvit_basic.py` 等, 识别 attention/自定义op/动态shape 风险 + A-3 测量计划。**不动 GPU** 直到 sw 交付工件且 team-lead 点头。
  - **doc-curator**: 常驻整合模式(supervisor PASS 的结论→整合非追加/纠旧/控长度/六骨架); 待触发 Task#11 二轮。
- **协作链**: 真测(sw/hw) → 核验(supervisor) → 入库(data) → 整合落盘(doc-curator); team-lead 对一切"已测/已修"自报复跑核验, **agent 自报 PID 必须配 "进程存活+产物增长" 双核**(旧 sw 自报 PID 2434949 消失无痕的教训)。

## 四、V2X-ViT 实验 (Task#8, 用户已授权; 收口后启动)
**目的**: 换欠参数化/更难模型, 验证精度轴(AP+mAOE)能否变成可搜 trade-off(尤其量化轴 — Pyramid 上无信号)。顺序: **A-0 scoping(sw, 纯读, team-lead 确认后) → A-1 DAIR val 1789 baseline AP+mAOE → A-2 1-2 剪枝档×{FP16,INT8} 看精度轴是否活(判据 Δ vs CI ≥5×) → A-3 hw TRT FP16/INT8 build+collab 口径 lat/energy(空闲卡+锁频) → A-4 data 接入(model_class=v2x_vit, config_json 承载结构, 不与 pyramid 混口径/混表)**。
预期对照: per-stage 混精/护栏叙事只在 transformer/MSDA 模型才可能成立(dims_quantization §二.5 (4)); 若 V2X-ViT 也不崩/也平 → 诚实记负结果(supervisor 已立此标尺)。

## 五、用户已定决策(不可违背)
1. **V2X-ViT 实验已授权**(当前测试完成后启动); **UniV2X/V2X-Seq+AMOTA 暂缓**(Task#7 DEFERRED, 待用户讨论; 真 AMOTA 只能来自 UniV2X 家族, Pyramid 无跟踪头, ISS-026)。
2. **Phase H 冻结**(Task#4): H1/H2 数据冻结保留(干净真测, 偷跑产物, 不入库不出图不引用; `dims_hardware_v2.md §9` 有治理标注), H3/H4/H5 勿启, 解冻须用户。
3. **data/hw 收紧+留用**; 宪章 §2 MUST-6 授权红线生效。
4. 精度轴 = 约束(ISS-018); 主前沿 = (latency, energy[, throughput]) + 耦合陷阱。

## 六、TaskList 快照 (supervisor 接管治理)
#3 出图(等定稿) / #4 [FROZEN] H1/H2 / #6 [in_progress] 全1789→终核→八步定稿(清单在描述里) / #7 [DEFERRED] UniV2X / #8 V2X-ViT / #9 V2X-ViT 接入 / #11 doc-curator 二轮(待触发)。#1/2/5/10 已核验清除。

## 七、文档结构现状 (doc-curator 首轮后, 六骨架)
`methods/design/` = dims_pruning_v1 / dims_quantization_v1 / **dims_hardware_v2** / doe_design_v1 / **pareto_definition_v1(§七=精度轴定位, 原 ap_activation 已并入)** / search_space_size_v1 (+2 指针文件)。外部论文对照在 `multi_agent/references/`(moe_paper_dim_review_v2); 过程版在 `multi_agent/archive/`。文件名 v1 内容亦为最新(CLAUDE.md §〇 与宪章 §5 引用已修正)。
**doc-curator 待裁决遗留**: pareto 节序(五/六-bis/六)与 doe_design↔720-anchor 合并, 都等 data 定稿后做。

## 八、纪律(每条都有血泪 ISS)
- 真测就是真测; 不轻信 agent 自报(复跑/读文件/mtime/git diff); 引用外部论文数字必回原文核(hw 两次幻觉 ISS-022/028); 修正幻觉数字时纠错值也要核原文。
- latency 必空闲 GPU+锁频; 口径不混(body_subnet_collab2 ≠ e2e ≠ orin); AP 必 finetune 有效 + epoch 与引擎一致(ISS-024)。
- 实验启动/数据变更须 team-lead 授权(MUST-6); 撤回暂缓须点对点回执(ISS-023 竞态); supervisor 周期核查冻结实验偷跑。

## 九、事实源
`team_charter_v1.md` · `background/00_研究目标与实验档案_v1.md` · `issues_log_v1.md`(ISS-001~028) · `NEXT_PHASE_plan_v1.md` · `pareto_definition_v1.md §七` · `data/schema_v2.md` · 本文件。Orin: jichengzhi@172.16.62.222(密码每 session 向用户确认)。
