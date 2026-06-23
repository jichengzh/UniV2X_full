# 本次会话进度报告 (Multi-Agent Team v1)
**时间**: 2026-06-01 00:30
**会话时长**: ~95 分钟
**分支**: hw-deploy-d-space

## 本次会话做了什么

用户要求"启动 agent team", 经三轮纠偏:

- **第一轮 (设计)**: 3 个 agent 出量化/剪枝/预测器设计文档 (纯文本, 未落代码)。
- **第二轮 (实现)**: 用户要 5 个角色交付**可用工程产物** + 验收 + 反馈闭环。落地 3 个配置驱动工具 + 3 份维度清单 + 数据生成管线 + 验收报告。
- **第三轮 (真测+闭环)**: 用户指出数据全是 dry-run 非实测、GPU 其实空闲、缺反馈闭环。重构编排, 跑出**第一批 7 行真实测 latency**, 并修复 QDQ INT8 bug。

## 当前状态

### ✅ 已验证 (主控亲自核验, 非轻信 agent 报告)
- **7 行真实测 y**: `output/doe_dataset_v1/doe_dataset_v1_real.csv`, engine 文件 6/1 00:07–00:23 真实落盘。
  - base FP32 2.674ms / FP16 0.825ms (3.24×) / INT8 0.530ms (5.05×)
  - 剪枝 p25/p50/p75 FP16 + p50 INT8
- **QDQ INT8 bug 已修**: 根因在 `tools/configurable/deploy_config.py` 建 weakly-typed network; 改 STRONGLY_TYPED 后 Myelin 报错消失, build 180s→58s (GPU7 实测)。
- **契约零违规**: `framework/config_schema.py` 全程未改 (git 验证)。
- **关键科学发现**: p25 FP16 (2.43ms) 比 baseline (0.825ms) **还慢** —— 通道非 32 对齐 → TRT 选了非 tensor-core kernel。round_to=8 不够, 这正是"为什么需要搜索空间"的实证。

### ⏳ 待验证 / 未做
- **AP 全部待测**: 7 行 latency 真, AP 只有 baseline 旧值 (0.97/0.96/0.93), 剪枝行 `ap:TODO` (未 finetune, 需 HEAL inference ~20-40min/config)。
- **INT8 两条路径不一致**: `m4_8_trt_build_bench.py` 用 implicit calibrator (Pyramid 94.8% INT8 覆盖, 健康); explicit QDQ-ONNX 路径才有 10.8% bug (BEV transformer 结构所限)。
- **strongly-typed 副作用**: 非 INT8 区域掉回 FP32 (丢 FP16), 需量化师导"FP16 主体 + INT8 孤岛"混合类型 ONNX。

## 遗留问题 / 下一步 (按优先级)
1. **补 AP**: 对 7 行 (剪枝行先 finetune) 跑 HEAL inference, 补齐 y 的 AP 维度。
2. **修 `generate_dataset.py` 结构缺陷**: L263-264 latency/AP 硬编码为空、`can_real_build` 门控几乎恒 False。改为成功 build 后调 `m4_8` 的 `benchmark_engine()` + 自动从 ckpt export ONNX, 让管线自产 y。
3. **量化师导混合类型 ONNX** (修 strongly-typed 丢 FP16)。
4. **扩 anchor** (用户要求, 25 不够): 补量化校准器/混精/剪枝×INT8 交互维度; Pyramid 缺的 INT8 档 (p25/p75)。
5. y 补齐后才训预测器。

---

## 【2026-06-03 第二轮 · 4-agent team 开工 + supervisor 核验】(supervisor 追加)

> 本轮转向"激活 AP 故事"战略 + P0 补完整点。supervisor 全程核验(读原始文件/pandas/nvidia-smi/git diff),问题落 `issues_log_v1.md`(ISS-001~014)。

### 已核验落定 (supervisor 实查, 非轻信)
- **完整点计数订正 (ISS-010/011)**: 旧"6"是陈旧保守计数。实查 dataset_v2: 严格完整点 **40 行**(collab2 28 + E4 11 + E3 1)。collab2 28 行 AP 全真测(21 独立逐行 + 7 金标准 EXACT 复用), 0 finetune 违规。但**有效多样性薄 ≈4 架构 × per-stage 退化变体, 独立锚点 ≈6-8**。宪章 §1 已更新。
- **CLAUDE.md ckpt 笔误修 (ISS-012)**: base ckpt 改 DAIR `..._11_42_29`(旧 `..._04_28_12` 是 OPV2V 版, 不同模型)。
- **AP 激活策略把关 (ISS-013)**: Path A(剪枝×forced-int8)信号真(ΔAP70 至 34×噪声), 但 **forced-int8 全档被 auto-int8 双轴支配** → 只作预测器训练信号 + 护栏消融, **非前沿 trade-off**。data 全盘采纳, 文档定稿措辞已修。前沿级 trade-off 需靠 C(V2X-ViT)/兜底。
- **P0-1·hw 核验通过 (ISS-014)**: GPU7 干净卡、口径 collab2、数字与 dataset_v2 一致、未用脏比值。现象真(p25 非对齐更小却慢 2.28×, INT8 仅 1.06×)。**但耦合机制(IMMA padding)被 hw 自证伪**(p75 pad 更多却更快), 替代解释(tactic cliff)未证实 → 责成 hw 补 `trtexec --dumpProfile` 坐实, 机制不得写成已证结论。
- **P0-1·sw forced AP 预检**: ckpt=DAIR finetuned pruned25, ap50 0.7743 健康。新 p25 点使序列变"低剪枝平缓/高剪枝放大"(非严格单调), 已提醒 data 定稿改措辞。

### 进行中 / 门控
- Task#1 P0-1·sw(automix 分支)· Task#3/4 P0-2 [32,64,136](门控 finetune 收敛)· Task#5 P0-3 Orin(门控 SSH 密码)· Task#6 data 修 5 行 ap:TODO 标注 · hw trtexec profile 回补。

## 关键文件清单
| 文件路径 | 改动类型 | 状态 |
|---------|---------|------|
| tools/configurable/quant_config.py | 新增 (量化师) | smoke 通过, 真 TRT 待测 |
| tools/configurable/prune_config.py | 新增 (剪枝师) | 真 rebuild 已验证 |
| tools/configurable/deploy_config.py | 新增+修复 | 真 build engine, strongly-typed 已验证 |
| tools/configurable/generate_dataset.py | 新增 (数据师) | **有结构缺陷: 不产 latency/AP** |
| output/doe_dataset_v1/doe_dataset_v1_real.csv | 新增 (测量师) | **7 行真测, 已核验** |
| output/doe_dataset_v1/real_engines/*.engine | 新增 | 真实落盘 |
| multi_agent/methods/design/dims_*.md | 迁移 | 三份维度清单 |
| multi_agent/methods/acceptance/acceptance_report_v1.md | 迁移 | 验收报告 |
| multi_agent/methods/design/{search_space_size,doe_design}_v1.md | 迁移 | 规模+DoE |

---

## 【2026-06-04 第三轮 · Phase M 收口 (团队重建, supervisor 重建实例值守)】(supervisor 追加)

### 终核: 全1789 修正版 mAOE (Task#3, ★PASS)
- **产物**: `results/tp_errors_corrected_full.csv/.json`(team-lead 代跑 @GPU7, 00:35→01:10, 脚本自创建未改动)。supervisor 五步独立核验全过, 细节见 **issues_log ISS-020 终核条目**:
  ① 剪枝轴 mAOE 单调 + **SNR 13.1–17.1×**(Δ vs 自身 bootstrap CI, 禁 ×AP70)② 8/8 行 CI bracket + **同人口**(ISS-025 结构修复; 仅 p25↔p50 相邻档 CI 重叠, 须诚实标注)③ 逐 anchor epoch 对齐(base=bestval23 / pruned=强制 net_epoch25; 脚本 L56-60 + 日志加载行 + csv epoch_used 三处一致)④ gpu7 二跑 2-anchor 交叉一致(base fp16 mAOE 4dp 完全相同)⑤ INT8 轴: p25/p50/p75 SNR 0.58/0.05/0.82× 噪声级 + 跨档非单调 → 无量化信号(base 档 +0.0030 单点可分辨但非轴信号, 措辞勿写"全部噪声级")。
- **附加**: p75 污染版 0.022 AP 偏差被 epoch 修正完全消除(ISS-024 金丝雀闭环 ✓); 新发现 **p50 AP70 偏金标准 +0.009(两精度同向)→ ISS-029 备查**, 定稿须显式选源标 caveat。
- **下一步**: data 执行 Task#3 八步定稿(supervisor 已放行), 定稿 mAOE+CI 唯一来源 = corrected_full.csv → supervisor 复核 → 删 PROVISIONAL → doc-curator 二轮(#7, blockedBy #3)。

### ★Phase M 正式全收口 (2026-06-04 午后, 闸一闭合)
- **收口链全 PASS**: 终核(ISS-020)→ data 八步定稿(step⑦ 数值零差错 + 审计)→ doc-curator 二轮(3 必修逐词复审 PASS)。Task#1/#3/#7 核验后 deleted+留痕; ISS-020/024/025/028/030/031 全闭环, ISS-029 备查。
- **本日新立 ISS**: ISS-029(p50 AP 两源差 +0.009, 备查)/ ISS-030(sw 否决后启动 A-1, 终定性=程序性违规+双重流程缺失, 回执协议入宪章)/ ISS-031(QuantV2X "75.1→29.9" 幻觉拼接, 真值 57.4→40.0/49.5→11.0, 4 文档修正)。
- **团队教训固化×2**: ① 启动回执协议(逐字引用授权原文, 验收≠授权, 宪章 §2 MUST-6); ② 数字下发必须「人口+口径」双标注。
- **V2X-ViT A-1 就绪**: 用户闸二已开; supervisor 脚本核验 6/6 PASS(epoch17 锁定/同方法学跨模型可比); team-lead 授权待发(GPU1)。A-1 核验锚: base AP70≈0.5215。

### 值守基线 (2026-06-04 ~13:00)
- **GPU**: 0/1/2/7 空闲; 3/4/5/6 他人占用(与 HANDOFF 基线一致)。**无 H1/H2/H3-5/UniV2X 冻结实验进程**; 01:11 后新文件仅 gpu7 二跑产物(合法)→ **无偷跑**。
- **TaskList 治理**: 白名单 = #3(Phase M 收口)+ #5(V2X-ViT, 启动须 team-lead 点头); 冻结占位 = #2/#4(永不删)。#7 已有 blockedBy #3(但状态 in_progress 与"等待触发"不完全对齐, 已报 team-lead 留意)。

### 团队重建值守基线 (2026-06-05 ~15:0x, supervisor 重建实例第 0 轮巡检)
- **背景**: 旧 5-agent 团队已关停, 按 HANDOFF_model_reselection_and_pyramid_audits_v1.md 重拉。supervisor 必读 4 件完成 + 已向 team-lead 回执(含铁律首行逐字引用)。
- **GPU 基线**: **全 8 卡空闲**(util 0% / mem ≤5MiB, wuyuegao 训练已结束)— 难得的全空闲窗口, 但白名单内只有 #7 camera profiling 微跑需 GPU(逐个宣告制)。
- **进程基线**: 0 个团队实验进程(pgrep v2xvit/finetune/trtexec/profil 空); results/scripts 14:00 后 0 新文件 → **无冻结实验偷跑**(H3-5/UniV2X/三决策项全干净)。
- **TaskList 现状**: #1/#2 冻结占位 ✓; #3 V2X-ViT(INT8/A-3 HOLD); #4 data 接入 in_progress; #5 sw zoo 补遗 in_progress **但产物未动**(model_zoo_survey_v1.md mtime 11:02 = 旧窗口阶段1版, 无"模态维度"节)— 下轮巡检盯"派单后无动作"(idle 唤醒漏消息 bug 三次复现史); #6/#7 pending。
- **值守白名单**: #4(data)/#5/#6/#7(sw); 冻结: #1/#2 + 三决策项(A-3 护栏成对测/iso-budget/fusion attention 剪枝)HOLD 在用户桌上。

### ★里程碑: 选型审阅包四件齐备 (2026-06-05 ~15:1x, 全链核验收口)
- **HANDOFF §二 sw 三任务全完成且核验通过**: 2.1 zoo survey §八模态全景(ISS-034 核验) / 2.2 pyramid_lidar 审计(ISS-036: 4硬错+4错引+5标注打回→17+1 项修订→终验 PASS) / 2.3 pyramid_camera 审计(ISS-037: 5 修→PASS)。Task#4/5/6/7 全部核验后清除留痕。
- **审阅包**: ① v2xvit_structure_audit_v1.md(v1.3) ② model_zoo_survey_v1.md(v1.1, §1.1 参数勘误 14.45M→5.465M + §八三轴) ③ pyramid_lidar_structure_audit_v1.md(15:03) ④ pyramid_camera_structure_audit_v1.md(15:02)。doc-curator 跨引用一致性两项检查 PASS(supervisor 抽验确认), 无需再改。
- **核验质量痕**: 全数字回源(stage_a 4dp/dataset_v2/P1 csv/m4_8 json/config.yaml); 拦截值(0.7912/0.6233/0.956/0.788/0.639/72M/~0.3ms/14.45M/26.8×)两审计 grep 全 0; 高危修除 = 跨配置 AP 错栽("剪枝提升"假象源)+ padding 证伪机制复活 + encoder-scope 混比。
- **附带产出**: ISS-035(mean/p50 同 run 裁决+使用规约) / dataset_v2 → 65×70(V2X-ViT 4 行) / dims_pruning §8.6 跨模型节(修订后) / CLAUDE §〇.4 3.091 勘误 / sw 微跑漏宣告台账×2(此后无豁免)。
- **挂账**: 三决策项(A-3 护栏成对测/iso-budget/fusion attention 剪枝)HOLD 在用户桌; #1/#2 冻结占位; GPU0 现被 wuyuegao 占用(后续计时避开)。
