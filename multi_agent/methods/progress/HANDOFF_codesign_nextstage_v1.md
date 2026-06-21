# HANDOFF — 协同加速下一阶段执行计划 + agent 分工 (nextstage v1, 2026-06-21)

> **前置**: 三臂 prune×schedule ablation(B1–B5)已完成并 commit(`243753c`)。现状入口 = `HANDOFF_three_arm_ablation_results_v1.md`;命题与四层证据 + 真数据结果 = `auto-tuning/4_design_ablation_proof_v1.md §9`。
> **本页**: 定义 4 个下一阶段任务(T1–T4)+ agent 分工 + DS 曲面评估。接手按本页起 agent。

---

## §0 一句话状态
ablation 主结论已坐实(A-joint 99.6% / A-serial 86.0% / A-noS 48.0% HV,p=4.9e-4;1 个干净 shipped 协同赢 3.29×@iso-AP0.6362)。下一阶段 = **扩维度 + 补对照 + 修闭环**,共 4 条线,T1/T2/T3 可即刻并行,T4 是长杆(gated on Pyramid→V2Xverse+CARLA 移植)。

---

## §1 ★DS 计算勘误(已落 doc4 §9.5,新工作必须遵守)
当前 `DS = DS_lat(latency)`(β=0)把 Pyramid 延迟投影到 **CoDriving** 实测 τ→DS 曲线,**隐含假设 Pyramid 感知精度/特征提取 = CoDriving** —— 不成立。后果:β=0 下同延迟不同 AP 得同 DS,但低 AP=漏检多=更不安全。**DS 必须由 (AP, latency) 共同决定。**
- **仍有效**:对内 W_g/P_g(AP 恒等)的 DS 差纯延迟驱动,与 β 无关 → +7.95/+10 DS 可留。
- **失效**:跨 AP 的 DS 比较(尤其含 A-noS)不可信,仅占位;主结论以 HV/Wilcoxon + 对内 DS 差为准。
- **修法**:见 §6(T4 的 Pyramid DS(AP,τ) 曲面)。在此之前,DS 只用于"同 AP 对内"对比。

---

## §2 T1 — 量化 Q 轴(prune × quant × schedule 三轴)
**目标**:把搜索空间从 P×S 扩到 **P×Q×S**,Q = 量化档(FP16 / INT8)。证 Q 是否引入新的 rank-flip(对齐宽度只在 INT8+tensorize 下显优),以及 Q 对 W_g/P_g 倍率的放大。
**纪律(铁律)**:
- INT8 必须**真 TRT INT8 build**(relax 无 INT8 pass;gap1 已记录 simulated INT8 不可信,见 [[project-dair-ap-axis-collapse]])。
- 跨口径不混:TVM(prune×schedule)延迟与 TRT(量化)延迟分开标,不并入同一 Pareto 数轴混算。
- INT8 AP 必须真测(DAIR val,TRT INT8 引擎跑),不能用 FP16 AP 代。
**落地**:扩 `framework/search_three_arm.py` 的 decode 加 Q 维(或先建 Q-augmented LUT+AP 表,复用现内核);产出 `results/latency_lut_pyramid_q.json` + INT8 AP 表。
**分工**:
- **data-orchestrator(lead)**:下采样指令,定 P×Q 采样点(哪些宽度 × {fp16,int8}),核验回报口径,并入 dataset_v2。
- **hw-optimizer**:真 TRT INT8 build + latency + energy(4090;Orin 选做),校准器 IInt8MinMaxCalibrator,记录 engine size/p50/p99 + J/frame。
- **sw-optimizer**:DepGraph 剪枝 × QuantV2X 量化耦合,INT8 calibration 数据准备,真 INT8 AP(DAIR 1789),per-stage 混精 vs TRT-auto 对比(已知 TRT-auto 支配手工 per-stage,验证是否随剪枝率变)。
**预期**:INT8 放大 W_g/P_g 倍率但不改本质;若出现"仅 INT8 下才 rank-flip 的对齐宽度"→ 新机理点(doc4 §2.2 future 项坐实)。

---

## §3 T2 — pair3 s2_128 崩排查(补第 3 AP 档的对)
**现状**:pair3 = mix_d [48,128,128](W_g,延迟已测)/ s2_128 [64,128,128](P_g)。s2_128 TVM MetaSchedule 调优**持续 CUDA illegal-access 崩(2× exit134)**→ P_g 延迟拿不到 → pair3 降级。mix_d 的 AP(0.6369)已 finetune,只差 s2_128 的 tuned 延迟。
**任务**:排查 s2_128 在 H800 上的 TVM 调优崩因(fresh workdir、builder timeout=300、逐宽度进程隔离、continue-on-error;参考 [[feedback-tvm-tune-apply-fresh-workdir]])。若修通 → 第 3 个 AP 档(0.6369)的完整对,强化"结构性、非 cherry-pick"。
**分工**:**hw-optimizer**(H800/TVM 调优专长)。环境:`ssh -p 30001 jichengzhi@222.95.84.215`(pw 12345678);tuner `/exdata/jichengzhi/s2_tvm/s2_2e_bumped.py`;python `/exdata/jichengzhi/tvm310/bin/python`。
**退路**:若 s2_128 本身是该形状的 TVM codegen bug 修不动 → 诚实记为 TVM 限制,2 对(两 AP 档)已足够支撑论点,不阻塞。

---

## §4 T3 — CoDriving 可分离对照臂(双模型判据)
**目标**:在 **CoDriving(标准 conv)** 上跑同一三臂搜索,**预期 A-joint ≈ A-serial**(可分离),与 Pyramid 显耦合(A-joint≫A-serial)形成对照 → 坐实 **"协同价值 = 耦合强度的函数"** 的 model-dependent 判据(doc4 §4.1 / §5)。
**为什么预期可分离**:CoDriving 标准 conv 对宽度不挑,TVM schedule 旋钮价值低(实测 ~2× vs Pyramid grouped 8–10×)→ 三轴近独立 → 锁一侧不致命。见 [[project-codriving-optimization-pilot]] / `HANDOFF_codriving_tvm_migration_v1.md`。
**落地**:**复用 `framework/search_three_arm.py` 内核**,只换数据 —— 建 CoDriving 的 LUT(各宽度 TVM default/tuned 延迟)+ AP 模型(已有 CoDriving DAIR 开环 AP,见 [[project-codriving-dair-openloop-ap]]);跑 `run_b4_ablation` 变体。产出 CoDriving 版 b4 结果(预期 HV gap 小、Wilcoxon 不显著)。
**分工**:
- **sw-optimizer(lead)**:CoDriving 剪枝档 × TVM schedule LUT(已有 TVM 迁移基础 P2–P4),CoDriving AP 表。
- **data-orchestrator**:核验数据可学习性,并入对照。
**判读铁律**:A-joint≈A-serial **本身就是结论**(不是 bug),切勿硬凑成"联合赢"。

---

## §5 T4 — 真 CARLA 闭环 + Pyramid DS(AP, τ) 曲面(长杆)
**目标**:① 用真 Pyramid 闭环替换 DS 估算;② 修 §1 勘误 —— 建 **Pyramid 专属 DS(AP, τ) 二维曲面**。
**前置(大工程)**:Pyramid→V2Xverse 移植完成 + CARLA 0.9.10.1 装好(本机/H800 未装)。见 [[heal_v2xverse_migration_design_v1]] / `HANDOFF_pyramid_v2xverse_migration_v1.md`。
**步骤**:
1. 完成 Pyramid 集成进 V2Xverse 感知链路(时延感知回灌 + 算力隔离,sim-integrator 已有基础设施)。
2. 真 Orin e2e τ_perc sweep(trap25/pad64/base 等)→ 替换 `closedloop_objective_query.py` 的线性缩放估算。
3. **★建 DS(AP, τ) 曲面**:在闭环里**同时扫**感知精度(不同剪枝/量化档给不同 AP)× 延迟(τ),测 driving score/碰撞率 → 得 DS 对 (AP, latency) 的真实联合响应,替换借来的 CoDriving 纯延迟曲线。并入 dataset_v2。
**分工**:**sim-integrator(lead, 全程)**;hw-optimizer 配合提供 trap25/pad64 的真 Orin TRT body 延迟。
**注**:T4 不阻塞 T1–T3;在 T4 完成前,DS 维持"同 AP 对内"用法,论文 DS 论断全标 model-estimated。

---

## §6 ★评估:是否需要建 Pyramid 的耗时-DS 曲线(类似 CoDriving)?
**结论:需要,但要建的是 2D 曲面 DS(AP, τ),不是 1D 的 DS(τ) 曲线;且 gated on T4 移植。**
- **1D DS(τ) 曲线(照搬 CoDriving 形式)不够**:它只解决"借用 CoDriving 的 τ-敏感曲线形状"问题(换成 Pyramid 自己在 DAIR 上的 τ-敏感度),但**仍固定在该模型自身的 AP 上** —— 不解决用户刚指出的"DS 须随 AP 变"。两个不同 AP 的 Pyramid 配置在 1D DS(τ) 上仍会因同 τ 得同 DS。
- **正解 = 2D 曲面 DS(AP, τ)**:闭环里同时扫 AP(剪枝/量化档)× 延迟 → DS 对 (AP, latency) 的联合响应。这才同时满足"Pyramid 自己的敏感度"+"AP 进入 DS"。
- **代价/前置**:两者都要 CARLA 闭环(Pyramid→V2Xverse 移植),是 T4 的一部分;1D 曲线是 2D 的退化,既然都要装 CARLA,直接做 2D。
- **过渡期**:移植未完成前,DS 仅用于"同 AP 对内"对比(对内 AP 恒等,1D/2D 区别消失,现有估算够用且结论可信)。**不必为了 1D 曲线单独提前投入** —— 等 T4 一次性建 2D 曲面。

---

## §7 优先级 / 并行建议
| 任务 | 可即刻起? | 依赖 | 建议 agent | 阻塞 ablation 主线? |
|---|---|---|---|---|
| T1 量化 Q 轴 | ✅ | 无(4090/Orin TRT 已通) | data-orchestrator + hw + sw | 否(扩展) |
| T2 pair3 排查 | ✅ | H800 可用 | hw-optimizer | 否(强化) |
| T3 CoDriving 对照臂 | ✅ | CoDriving TVM 基础(已有) | sw-optimizer + data | 否(双模型判据) |
| T4 真闭环 + DS 曲面 | ⏳ 长杆 | Pyramid→V2Xverse+CARLA 移植 | sim-integrator(+hw) | 否(修 DS 估算) |

**建议**:T1/T2/T3 三线并行起 3 组 agent(data-orchestrator 协调 T1、hw-optimizer 主 T2、sw-optimizer 主 T3);T4 由 sim-integrator 并行推进移植(慢,不等)。**supervisor** 全程核验"已测/已修/已build"自报(复跑/读文件/git diff);**doc-curator** 把已核验结论整合回 doc4 + dataset_v2(非追加)。

---

## §8 关键文件 / 纪律速查
- 内核 `framework/search_three_arm.py`(含 `detect_wg_pg_pairs`)、驱动 `framework/run_b4_ablation.py`、B5 审计 `scripts/phase2/b5_verify_convergence.py`、闭环 `scripts/phase2/closedloop_objective_query.py`。
- 数据 `results/{latency_lut_pyramid.json, ap70_model_pyramid.json, b4_ablation_results.json, b5_convergence_verification.json}`。
- 设计 `auto-tuning/4_design_ablation_proof_v1.md`(§9 真数据 + §9.5 DS 勘误)、`closedloop_b4_plugin_v1.md`(顶部 DS 勘误)。
- 纪律:真测=真测(INT8 走真 TRT,不用 proxy);区分 真测/估算/仅声明;latency 在空闲 GPU(util 0%/mem≤50MiB);跨硬件/跨口径不混 Pareto;TVM tune 用 fresh workdir 逐宽度进程隔离;不轻信 agent 自报。
