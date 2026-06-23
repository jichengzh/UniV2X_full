# HANDOFF — 协同加速框架: 架构已设计 + 下一步实验计划 (v1, 2026-06-19)

> **接手先读本页**。本页 = 框架架构(已设计落盘)+ 下一步要跑的实验(W_g 探针→ablation)+ 已确立素材 + 资源指针。上下文已清理, 据此续接。
> 关联: 设计文档见 §1; 大局叙事 [HANDOFF_codesign_unified_v1.md](HANDOFF_codesign_unified_v1.md); 文献 [study_joint_search_methods_v1.md](../../references/study_joint_search_methods_v1.md)(ALT/CHaNAS/AutoTVM)。

---

## §0 一句话状态 + 接手第一步
框架搜索架构(ALT 三部分 + TVM 集成 + 有效性证明)**已全部设计落盘**(§1)。**W_g 存在性探针 Phase 1 已跑完 = 正**(2026-06-20, 见 §3A + `design_wg_probe_int8_plan_v1.md §9`): **W_g=trap25 / P_g=pad64 完整配对**(同 AP70=0.590 仅 stage0 对齐差), 机理定位 stage0。
**★接手第一步 = 按 [`HANDOFF_three_arm_ablation_exec_v1.md`](HANDOFF_three_arm_ablation_exec_v1.md) 执行三臂真搜索 ablation**(B1 延迟LUT / B2 AP finetune / B3 搜索内核 三线**可并行起 3 agent**, 见该文档 §4 分工 DAG; 三臂必须真搜索非枚举)。**量化 Q 轴 = prune×schedule 完成后的持久下一步, 别忘**(该文档 §5)。

---

## §1 框架架构(已设计, 4+1 文档)
`multi_agent/methods/design/auto-tuning/`(ALT 三部分 + 证明):
- **1_design_space_building_v1.md** — 空间构建/收缩: ALT 两阶段(joint/schedule-only)× CHaNAS 分块 schedule-LUT(R·B·S+R^B, 防爆炸)× 对齐谓词(in_per_g∈2^k)预筛。把指数空间收缩成线性建表 + 外环查表。
- **2_design_cost_model_v1.md** — 代价模型: 分层异构 = 内环 schedule 复用 TVM `xgb_model`; 外环 prune 用 **rank-loss LightGBM**(修崩掉的 LGB latency, lambdarank+group by model/hw); 跨硬件 global+local(用 M2 R²>0.995); AP 走"真测漏斗"(不进纯预测); CHaNAS 加性 LUT 仅串行段成立(fusion/异构破加性, 用混合组合算子)。
- **3_design_exploring_tvm_integration_v1.md** — 探索+TVM 集成: 外环 NSGA-II(自建)× 内环 MetaSchedule(复用), ALT cross-exploration 粘合; P=导入前 DepGraph 重建 / schedule=内环 MetaSchedule; "先搭 pipeline 后补 cost model 数据"= 正确顺序, MVP 清单给出; 升级 `framework/nsga2_pareto_search_v4.py`(现 D 维只是粗枚举)。
- **4_design_ablation_proof_v1.md** — 有效性证明: 用**搜索过程(非枚举)**证单侧贪心→局部最优; 三臂 A-joint/A-noS/A-serial; 机制 = TVM prune×schedule 的 (AP,lat) Pareto 重排。
- `multi_agent/paper/design_wg_probe_int8_plan_v1.md` — W_g 探针(下一步实验, §3)。

---

## §2 两点更正(已贯穿全部 5 文档, 必须遵守)
1. **W_g 定义**: W_g = **单维(贪心所搜那一维)最优、但多维(全局)非最优**的点 = 单侧贪心落入卡住的**局部最优**(贪心选它因单维最优; 非全局因到达真全局需接受贪心已丢的单维次优)。**P_g** = 被错过的全局点(单维次优/多维最优)。[旧"单维被支配但联合解锁"是 P_g 不是 W_g。]
2. **平台: 已迁 TVM, 不以 TRT/INT8 为重心**。主轴 = **TVM prune × schedule**(MetaSchedule)。量化(INT8)因 relax 无 INT8 pass 降为 **future/次要**, 不作主依赖。

---

## §3 ★下一步实验计划: W_g 探针 → ablation
**目标**: 证 W_g 存在(贪心吸引子: 单维最优/多维非最优)→ ablation 有载体。

**信号(锐化版)= (AP, latency) Pareto 重排**(不是纯 latency flip):
- **default-schedule 的 (AP,lat) Pareto ≠ tuned-schedule 的**。W_g = default-Pareto 上、tuned 掉出的宽度; P_g = default 外、tuned 上的宽度。
- ★**trap25 已是现成 W_g 候选**(实测): default 下 trap25=(0.590,42355µs) 在 Pareto(0.59 AP niche, 比 base 快)→ 串行选它; tuned 后 base=(0.631,6214µs) 支配 trap25=(0.590,21615µs)→ 掉出。根因 = schedule 调优余量按宽度差异巨大(base 9.06× vs trap25 1.96×), **未调度估计误排宽度**。⇒ 探针大概率为正。

**Phase 1**(H800 TVM, 全空, latency-first): 宽度 grid(base/trap25/p50/p75 + ~3 个 stage 级对齐混合)× {default dlight, MetaSchedule-tuned} latency; 拼已有 AP 画两个 Pareto, 标 W_g/P_g 类点。**证普遍误排(多点), 非 trap25 单点。**
**Phase 2**: 候选宽度 finetune 补 AP, 在 (AP70,tuned-lat) 确认 W_g 被贪心锁、P_g 被错过。
**Ablation(doc4)**: 跑三臂真搜索 —— 串行(按 default/未调度估计选宽度→落 W_g)vs 联合(co-tune→到 P_g), 用收敛解+点云+跨 seed 证结构性局部最优。**载体**: Pyramid 先(重排明显)+ CoDriving 对照(标准 conv 预期可分离 = model-dependent 判据)。
**退路**: 无重排 → 换载体/升级机制/接受"可分离"判据(故事 B), 诚实非失败。

---

## §3A ★W_g 探针 Phase 1 实测结果 (2026-06-20, 已跑完 = 正)
`results/gap1_grid_corrected.json` + `results/diag_retune.csv`(H800)。可信脚本 `scripts/phase2/s2_2e_e2e.py` **逐宽度独立进程**重测(fresh work dir, idle GPU)。
- **★W_g/P_g 完整配对(同 AP70=0.590, 仅 stage0 对齐不同)**: default-Pareto {base,p50,p75,**trap25**} → tuned-Pareto {base,p50,p75,**pad64**}。
  - **W_g = trap25**[48,96,192]: 0.59-niche 内 default 42355<pad64 47407 故贪心选它, tuned 仅 1.96×→21615µs 被 pad64 支配, 掉出。= 单维(default)最优/多维(tuned)非最优的贪心吸引子。
  - **P_g = pad64**[64,96,192](trap25 权重 s0 零填充48→64, AP 不变): default 被 trap25 支配(贪心丢), tuned 7.71×→6152µs 成 niche 全局最优。= 单维次优/多维最优的被错过点。串行锁 W_g 永远到不了 P_g。
- **机理定位 stage0**(iso 消融): 仅 s0 失配(iso_s0 in_per_g=3) 2.36×≈trap25 1.96×; 仅 s1/s2 失配(iso_s1/s2) 7.2–7.5×; pad64(s0修复) 回到 7.71× ⇒ 余量塌缩由 stage0(最高分辨率 grouped conv, in_per_g=3)主导, 非全网均摊。
- **推翻旧结论(已实测)**: 旧 `gap1_schedule_lut.json` "pad64 ratio=1.0/pad救援负结果/对齐全网属性" = **buggy harness 假数**。`pad64_retest` 实测 **7.71×** ⇒ pad救援有效, pad64 即 P_g。
- **★方法学教训(必守)**: TVM tune+apply 必须 **fresh work dir + 逐宽度独立进程**(顺带隔离 CUDA illegal-access 崩溃); 切勿单进程循环复用同名 work dir。

---

## §4 已确立素材(corrected, 可作论文证据)
- **AP 轴(公平, 不塌缩)**: Pyramid AP70 单调 0.631→0.590→0.564→0.530(stage_a, base=官方收敛 ckpt); CoDriving iso-budget(公平 base AP50 0.626 > 所有剪枝)。**用 AP70 不用 AP50**(span 太小)。剪枝真崩 AP、finetune 恢复(过参数化), 非"剪枝免费"; "超越 base"=baseline 欠训 confound 非 harness bug。详见 unified §8-C + memory `project-dair-ap-axis-collapse`。
- **Gap1 schedule 首轮(H800)**: 对齐感知联合(选 p50)比串行(剪到 25%→trap25 kernel-cliff)快 **7.18×**@−0.026 AP70; pad 局部救援负结果证对齐全网属性。根因精炼: grouped conv in_per_g=2·nf/32∈2^k 才命中快核(48→3 非 2^k)。`results/gap1_schedule_lut.json`。
- **INT8 真实代价未测定**(simulated 不可信); CoDriving 真 TRT INT8 掉 AP 且随剪枝放大(旁证)。
- **文献**: ALT(layout↔loop, cross-exploration, ALT-OL/WP 消融)/ CHaNAS(arch×schedule, W/WO 1.68× iso-acc, 分解 R·B·S+R^B, divisible-split)/ AutoTVM(rank loss, XGBoost, 进化退火)。

---

## §5 资源 / 平台 / 脚本
- **H800 TVM(主, 当前全空)**: `ssh -p 30001 jichengzhi@222.95.84.215`(密码每会话确认; 代理 7897); env `/exdata/jichengzhi/tvm310/bin/python`(Unity 改版: tvm.s_tir.meta_schedule/dlight, get_sblock); 跑前 export PATH(cuda-12.2)+LD_LIBRARY_PATH(cat /exdata/jichengzhi/tvm_nvlibs.path)+CUDA_VISIBLE_DEVICES。资产 `/exdata/jichengzhi/s2_tvm/models/{base,p50,trap25}_backbone.onnx`(+codriving_cache)。
- **4090(本地, 边缘绝对)**: env `UniV2X_2.0`; TRT 10.13。**真仓库 `/home/jichengzhi/V2X`**(UniV2X 是断裂符号链接)。
- **脚本**: `scripts/phase2/{gap1_run_v2, gap1_pad_rescue_v3, s2_2e_e2e}.py`(schedule); `framework/{searcher_v0, nsga2_pareto_search_v4}.py`(待升级带 schedule 轴/评估器/Pareto)。
- **结果**: `results/{gap1_schedule_lut, codriving_isobudget_verdict, v2xvit_ap_corrected}.json` + `data/stage_a_ap_real.parquet`。

---

## §6 纪律 / 诚实边界
- latency 必空闲卡(H800 全空 / 4090 看 nvidia-smi)min-of-N; agent 自报必复核(复跑/读文件/git diff)。
- backbone-only ≠ e2e(Amdahl: backbone Pyramid ~13.6%); 跨平台不拼单一 e2e(H800 TVM / 4090 TRT 分报)。
- DEFAULT=dlight 是"旋钮关"基线, default/tuned 比 = 开搜索价值, 非 vs TRT。
- 负结果是资产(可分离=判据数据点); 不为要 W_g/flip 硬凑参数。
