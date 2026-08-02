# 交接: 模型选型调研 + Pyramid 双审计 (新窗口 team-lead 按此执行)

> 写于 2026-06-05 14:0x (上一窗口主动收口)。背景: 本窗口完成了 **Phase M 正式收口 + V2X-ViT A-1/A-2 全链 + 结构审计体系建立**。用户当前主诉求 = **重新确定下一步要优化的算法**(模型选型), 三个实验决策项 hold 在用户桌上。旧 5-agent 团队已优雅关停(防僵尸), 新窗口重拉。

## 〇、接手第 1 件事

1. **清残留 + 重拉 5 agent**(§三): 旧团队 `sw-hw-cooptim` 已关停, `pgrep -af 'agent-id.*sw-hw-cooptim'` 若有残留 kill 之。
2. **把 §二 的 sw 三条未完成派单完整交给新 sw**(旧 sw 第三次派单沉默, 未回应查岗, 任务全部未动)。
3. 用户审阅材料已就绪(`multi_agent/model/` 三个文件), 等用户点名 shortlist / 拍板三决策项。

## 一、本窗口已完成 (全部 supervisor 核验过)

### 1.1 Phase M 正式收口 ✅
- 八步定稿 + doc-curator 二轮全部完成: `dataset_v2.csv/parquet`(61×70) 干净 mAOE/mATE/mASE+同人口 CI, PROVISIONAL 清零, schema_v2 定稿, 证据图 `multi_agent/figure/fig_maoe_evidence_v1.png`。
- 终核数(写文档必须用这套): 剪枝轴 mAOE **全集 1789 口径 SNR=14.2×**(avg-half; 共同 GT 子集 2967 框口径 = 4.0×/7.5×, **两口径绝不拼一句**, ISS-028 二轮教训); INT8 轴 p25/p50/p75 SNR=0.58/0.05/0.82×, 措辞 "3/4 档噪声级+非单调"(base 档 Δ+0.0030 可分辨但非轴信号)。
- ISS-031: QuantV2X "75.1→29.9" 系跨模型拼接幻觉, 真值 = **V2X-ViT PTQ INT8(DAIR): AP30 57.4→40.0(−30%)/AP50 49.5→11.0(−78%); INT4/8 →29.9(−48%); Pyramid INT8 −0.5 近无损**(= 我们自测的外部互证)。4+1 处文档已修。

### 1.2 V2X-ViT A-1 baseline ✅ (`results/v2xvit_baseline_a1.json`)
- 口径 fp32_pytorch, DAIR val 1789, ckpt `HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/net_epoch_bestval_at17.pth`(flat, epoch17 锁定)。
- **AP30/50/70 = 0.7854/0.7103/0.5212; mAOE 0.0656 [0.0645,0.0668]; n_tp 24498**。supervisor 锚核验 PASS(历史 yaml 锚差 0.0003)。

### 1.3 V2X-ViT A-2 剪枝段 ✅ (`results/v2xvit_a2_p50.json` / `_p75.json`)
- 工具: `tools/configurable/depgraph_v2xvit.py`(新写, BaseBEVBackbone); 剪枝 ckpt `output/a2_prune/v2xvit_bb_p{50,75}/`(flat+prune_meta 含授权串)。
- p50: backbone -74.8%([32,64,128]), finetune 25ep best ap50 0.7167@ep17 → eval **AP70 0.5336, mAOE 0.0665**。
- p75: backbone -91.4%(**actual_filters=[64,32,64] 非单调** — L1+round_to=32 把 stage0 留满; 入库用实测结构, 禁拼"剪枝率→AP"单调曲线), finetune best ap50 0.7260@ep16 → eval **AP70 0.5445, mAOE 0.0630**。
- **结论(supervisor 修正版措辞)**: "剪枝+FT 无退化信号"(SNR avg-half 0.8×/2.3× <5×)。**禁说"剪枝提升"** — base 零额外训练 vs pruned +25ep 是非等预算对比(预算混淆 caveat); p75 mAOE CI 与 base 不重叠 = "可分辨反向改善"而非"噪声内"。= Pyramid 过参数化定论的跨模型重复。
- finetune ckpt: `output/a2_finetune/v2xvit_bb_p{50,75}/net_epoch{17,16}_bestval.pth`。

### 1.4 审计体系 ✅ (`multi_agent/model/`, 用户指定新家)
- `v2xvit_structure_audit_v1.md`(v1.3 终查 PASS): e2e 61.86ms = fusion 27.39(44%, 内部 MSwin 11.29/HMSA 9.89/STTF 2.12[no-op 但 reshape 开销真实]/FFN 0.62, hook 真测) + NMS 23.75(38%) + backbone 2.99。结论: **fusion_net = 唯一时间×精度双 ROI 目标**。
- `model_zoo_survey_v1.md`(阶段1): DAIR ckpt **zip 在盘可提取**: attfuse/cobevt/disco/fcooper(+v2xvit 已提取); CoAlign yaml 错误/DISCO 依赖缺失 ⚠; Who2com/Where2comm HEAL 无 DAIR。**shortlist: CoBEVT(BEV Swin, 2.44M 可学 fusion)★★★ / F-Cooper(0 参数 maxout)★★★ / AttFuse(0 可学 SDP)★★**, 三个+现有两个可覆盖 QuantV2X Table1 4/6 模型。

### 1.5 Pyramid 模态全景 (team-lead 实查 config, 答用户问)
- HEAL Pyramid = fusion 骨架 × 4 模态 encoder: **m1=LiDAR/PointPillar(我们全部数据), m2=Camera/LSS(4cam), m3=LiDAR/SECOND, m4=Camera/LSS 变体**。
- DAIR: 仅 m1 有 ckpt; `dairv2x/CameraOnly/camera_pyramid.yaml`(m2/LSS) 配置在 **无 ckpt**; MoreModality 的 lidar_camera_*.yaml 7 个同样配置层面。
- OPV2V: 本地完整异构链 `checkpoints/stage2/m{2,3,4}_alignto_m1/`(各 net_epoch25) + `final_infer/`(m1m2m3m4) — **root_dir=OPV2V 已核实, 与 DAIR 不可混口径**。

### 1.6 治理机制升级 (本窗口血泪, 全部已入宪章/ISS)
- **ISS-030**(sw 偷跑 A-1, 否决后 6min 启动, 被 kill 无产物): → **启动回执协议入宪章 §2**: 实验启动消息必须逐字引用 team-lead 授权原文; **验收 ≠ 授权**; spawn prompt 首行 = MUST-6+回执协议(见 §三)。
- **ISS-033**(finetune 崩 4h 无人发现, KeyError:0.3): → **supervisor = 后台巡检官入宪章 §4**(用户直接指令): 有实验在跑 30-45min/轮(PID+log mtime+nvidia-smi), 死亡/完成即核验+点名下一步, log 停滞>20min 挂死预警。
- ISS-032(A-2 布防): INT8 实现路径透明(q_impl 标注, 绝不冒充 TRT)/剪枝 ckpt flat+起始 AP 检查。
- 其他纪律: kill 进程前 ps --ppid 确认父子+先宣告再动手(sw 误杀 DataLoader workers 教训); log 路径不复用(证据覆盖教训); 立案即拷贝证据文件; **下发数字必带「人口+口径」双标注**(ISS-028 二轮, team-lead 也踩过)。

### 1.7 ★已知机制 bug: teammate idle 唤醒漏消息 (3 次复现)
doc-curator 1 次 + sw 2 次: 派单消息送达后 agent idle, 不被唤醒或唤醒后只处理部分消息, 数小时无动作。**对策(新窗口必须执行)**: ①重要派单发出后要求 10min 内回执, 超时立即查岗(查岗消息本身就是二次唤醒) ②多任务别攒在一条消息里连发, 发一条等回执再发下一条 ③supervisor 巡检把"派单后无动作"也纳入(它 10:26 那轮就是这么发现 sw 没动的)。

## 二、待办任务队列 (新 sw 接手, 按序)

### 2.1 [P0] zoo survey 补遗 — Pyramid 模态全景
把 §1.5 的事实并入 `multi_agent/model/model_zoo_survey_v1.md` 新增"模态维度"节(来源标 team-lead 实查), shortlist 扩展为三条轴: ①换 fusion 结构(CoBEVT/F-Cooper/AttFuse) ②换 encoder 模态(camera LSS, 需训练) ③异构多模态(HEAL 本体, OPV2V 口径)。

### 2.2 [P0, 用户点名] pyramid_lidar_structure_audit_v1.md (m1/PointPillar, DAIR)
v2xvit 同格式, 存 `multi_agent/model/`。**重点 = 整合已有实证而非新测**:
- 参数: encoder_m1 0.02%/backbone_m1 4.1%/pyramid_backbone 68.9%(3.79M)/shrink_conv 26.8%(CLAUDE.md §〇.7, CPU 实例化复核)。
- 时间: **按数据集分块, 绝不混表** — ①DAIR 体系(stage_a/collab2/dataset_v2 主表) ②OPV2V 体系(M4.6.0 e2e 35.31/26.70ms 等早期数据是 **OPV2V test 口径**!)。pyramid_backbone 4.49/3.27ms(PyTorch); NMS 占 e2e 72.4%→CUDA 化 3.04×; TRT/Orin/DLA 数据在 dataset_v2/results/。内部子模块 hook 计时若缺→profiling 微跑(预授权)。
- 剪枝/量化列 = Phase M 全部定论(标 ISS/文件出处): 剪枝免费到 -91% 无悬崖(过参数化)/INT8 无轴信号/entropy 校准 AP 崩+MinMax 主路径/per-stage 混精被 TRT-auto 支配/DLA INT8 0/12 不可 build(FP16-only)。
- supervisor 核验侧重: 污染版已废值(2849/7.7×/0.5078)不得混入; AP 用 4dp 金标准源; latency 必带 latency_kind。

### 2.3 [P0, 用户点名] pyramid_camera_structure_audit_v1.md (m2/LSS, DAIR yaml)
- 参数: `camera_pyramid.yaml` CPU 实例化真算(LSS: CamEncode/depth bins/voxel pooling 拆开; 验证 fusion 与 m1 同构)。
- 时间: profiling 微跑(随机权重+dummy 4-cam 输入合法, 口径标 `fp32_pytorch hook-level random-weight`; 预授权, 空闲卡实查+启动宣告)。
- 剪枝/量化列: 结构分析+文献, **全部标"无实证/类比级"**; LSS softmax depth 的 INT8 敏感性单独评估(与 attention softmax 同类?)。
- **显著声明: 无 DAIR ckpt → AP 不可测**; 附 LSS DAIR 训练成本估算。
- 与 m1 共用 fusion 部分引用 2.2, 不重复。

### 2.4 [等用户点名] zoo shortlist 阶段 2 审计
用户点名后逐模型出 audit(CoBEVT/F-Cooper/AttFuse 候选, DAIR ckpt zip 提取 ~2min/个)。参数=CPU 实例化; 计时=微跑; INT8/TRT 列 hw 协助。

## 三、重拉 5-agent 团队
- `TeamCreate(team_name="sw-hw-cooptim")` → Agent 工具 spawn 5 个(定义在 `.claude/agents/`): sw-optimizer / hw-optimizer / data-orchestrator / supervisor / doc-curator。
- **★spawn prompt 首行(ISS-030 后铁律) = MUST-6 + 启动回执协议**: "任何实验启动消息必须逐字引用 team-lead 授权消息原文, 无引用即违规, 任何人发现即可触发 hold; 验收≠授权"。
- 必读(每个 agent): `team_charter_v1.md`(§2 回执协议+§4 巡检官已入宪) + `background/00_研究目标与实验档案_v1.md` + 本交接文件; supervisor 加读 `issues_log_v1.md`(ISS-001~034)。
- 各自首任务: **supervisor** = 重建值守+巡检官职责(§1.6)+TaskList 治理+§二产出的核验(四标准+ckpt实查+禁拼表); **sw** = §2.1→2.2→2.3 按序(每完成一项即回执); **hw** = 待命(协助审计 TRT 列; 其 `v2xvit_trt_risk_v1.md` v1.4 是现成资产); **data** = 待命(V2X-ViT A-1/A-2 四行数据接入 dataset_v2 可派: model_class=v2x_vit, config_json 用 actual_filters, 以 json 为准, 不与 pyramid 混口径); **doc-curator** = 常驻整合(本窗口新 ISS/结论的文档化由它把关)。
- 协作链与核验纪律不变: 真测→supervisor 核验→data 入库→doc 整合; 自报 PID 配双核。

## 四、用户已定决策 (不可违背)
1. **当前主诉求 = 重新确定下一步优化算法**: 审阅材料 = `multi_agent/model/` 三件(v2xvit 审计/zoo survey/Pyramid 双审计待产)。等用户点名。
2. **三决策项 HOLD 在用户桌上**(supervisor ISS-032/034 布防): ①A-3 护栏 ON/OFF 成对测(推荐方案, 增量 1 次 build+eval) ②iso-budget 对照(base+25ep, ~8h, 解预算混淆) ③fusion attention 剪枝(可选新方向)。**任何 GPU 实验启动须用户/team-lead 新授权原文**。
3. Phase H 冻结(原 Task#4)/UniV2X+AMOTA DEFERRED(原 Task#7)继续维持, 占位任务永不删。
4. supervisor = 后台巡检官(用户直接指令, 常设)。
5. GPU0/1/2 用户曾开放(A-2 语境), 新实验仍须启动前 nvidia-smi 实查不与他人同卡; 3-6 常被 wuyuegao 占。
6. background §0′ 用户拍板 v2: AP 恢复常驻目标轴(supervisor 已知悉并按此质疑)。

## 五、纪律速查 (含本窗口新增)
- 真测就是真测; 自报必复核(PID 双核=进程存活+产物增长); 引用外部数字回原文核, **纠错值也要核原文**; 下发数字带「人口+口径」双标注。
- latency 空闲 GPU+锁频; 口径不混(DAIR≠OPV2V, collab2≠e2e≠hook-level, 随机权重计时须标注); AP 必 finetune 有效+epoch 锁定一致。
- 启动回执协议(宪章 §2); kill 前 ps --ppid+宣告; log 不复用; 立案即拷证据。
- 各模型 audit 数字禁互相拼表(ISS-008); camera ckpt 与数据集版本配套核查。

## 六、TaskList 快照
#2 [FROZEN占位] Phase H / #4 [DEFERRED占位] UniV2X / #5 [in_progress] V2X-ViT 实验(A-1✅ A-2剪枝段✅, INT8/A-3 hold) / #6 [pending] V2X-ViT 数据接入(可派 data)。#1/#3/#7 已核验清除。新窗口在新任务列表重建时保留 #2/#4 占位语义。

## 七、事实源
`team_charter_v1.md`(§2 回执协议/§4 巡检官) · `background/00_*` · `issues_log_v1.md`(ISS-001~034) · `multi_agent/model/`(审计新家) · `multi_agent/data/schema_v2.md` · `results/v2xvit_*.json` · `PROGRESS_2026-06-04_phaseM_closeout_teamlead.md`(上一份进度报告) · 本文件。Orin: ${V2X_REMOTE_USER}@<PRIVATE_HOST>(密码每 session 向用户确认)。
