# HANDOFF — 协同加速下一阶段 v3 (2026-06-21 晚, 取代 v2)

> ## ★★★ [2026-06-22] 全部 v3 任务链已完成(本次重启执行完毕)
> | 任务 | 结果 |
> |---|---|
> | **1. 三轴 P×Q×S ablation(P0)** | ✅ A-joint 100%/A-serial 85.3%/A-noS 56.7%, **Wilcoxon p=4.88e-4**, serial 锁定前沿全 int8 不可 build(命中 0 vs joint 4)。`results/pqs_ablation_results.json` + `framework/run_pqs_ablation.py` + 图 `pqs_*.png` + doc4 §9.8d |
> | **2. CoDriving int8 复验 + L3 verdict** | ✅ SEPARABLE(Cin=48/64 均 build int8 1.42×/1.32×, max_rel_err=0.0, 无 rank-flip)。`codriving_int8_verify.json` + `codriving_coupling_verdict.md` + doc4 §9.8a。INT8 放大假设证否 |
> | **3. L4 latency 网格扩充** | ✅ 仅原 3 对 rank-flip, 新 5 对非翻转 → **rank-flip 是 (s1,s2) 条件性**(边界刻画)。pair4 干净复测确认 non-flip。`l4_new_widths.csv` + doc4 §9.8e |
> | **4. L4 AP 崖口(DELIVERABLE A)** | ✅ wholenet 84/89/93% finetune: AP70 0.585/0.561/0.537, **soft_knee**(range 0.094=94×噪声, 拐点 84-89%)。`ap_cliff_l4.json` + 图 `ap_cliff_wholenet.png` + doc4 §9.8f |
> | 5. [可选] 全 backbone int8 | ⏸ 未做(定性框架已足够; int8 延迟用 1.449× proxy 已标口径) |
>
> **核验**: B4 P×S(99.9/85.5/47.7%, p=4.88e-4, ratios [3.83,3.51,3.29]) 与 PQS 三轴均复跑核验, 共享 kernel 改动后向兼容。**全部未 commit(等用户授权)。** 下文为本次重启前的原始计划, 留作背景。


> **取代 `HANDOFF_codesign_nextstage_v2.md`**(v2 的 "Option B/TVM int8" 判断已被本轮真实证否+重新解决;见 §1)。
> **新窗口阅读顺序**: §0 一句话状态 → §1 本轮已验证结论(L1/L2) → §2 L3/L4 现状与未决 → §3 重启任务链(具体命令+GPU 分配) → §4 环境/铁律/教训。
> 深挖: `auto-tuning/4_design_ablation_proof_v1.md §9` + `results/q_tvm_int8_verdict.md`。

---

## §0 一句话状态(2026-06-21 晚)
**L2(P×S 主结果)+ L1(量化轴 int8)都已真测+主控核验完成**;L3(CoDriving 对照)fp16 verdict 稳、int8 主张被驳回待 L1 复验;L4(网格扩充)在跑但未落地。**多层 agent 卡在 orchestration 不落 GPU**, 故改为本文档重启整条链。**全部未 commit。**

### ★诚实"已证 vs 未证"对照(2026-06-21 晚,勿混淆)
| 论点 | 状态 | 缺什么 |
|---|---|---|
| **P×S 强耦合** | ✅ **已证**(ablation HV+Wilcoxon p=4.88e-4) | — |
| **真 int8 可实现** | ✅ **已证**(stage0 微基准:dp4a/WMMA,数值 max_rel_err=0.0,1.45×) | 全 backbone int8(stage1/2)未做 |
| **量化→定性耦合(机理)** | ✅ **机理已证**(s0=48 结构性 build 不出 int8) | — |
| **三轴(P×Q×S)强耦合** | ✅ **[2026-06-22] 已证** | `run_pqs_ablation.py` 跑出 A-joint 100%/A-serial 85.3%/A-noS 56.7%, **Wilcoxon p=4.88e-4** rank-biserial=1.0; categorical_pass=True(serial 锁定前沿全 int8 不可 build, int8 命中 0 宽度 vs joint 4 宽度)。`results/pqs_ablation_results.json` + 图 `pqs_*.png` + doc4 §9.8d。延迟用 1.449× uniform proxy(全 backbone int8 是可选后续) |
| **CoDriving 可分离(普适性)** | ✅ **[2026-06-22] 已证** | fp16 s0 探针无 rank-flip + int8 复验(G1): Cin=48/64 均 build int8(1.42×/1.32×, max_rel_err=0.0), 无 rank-flip → 标准 conv 软惩罚非定性陷阱。caveat: int8 WMMA 走 MS-DB trace+数值确认(relax VM 不暴露 CUDA source) |

核心叙事(注意第 4 块是机理强证据但 ablation 未跑):
1. **P×S 强耦合**(主):3 对 ablation A-joint 99.9% / A-serial 85.5% / A-noS 47.7%, Wilcoxon **p=4.88e-4**, STRUCTURAL PASS; shipped 头条 **pair3 = 3.83× @ iso-AP70=0.6369**。✅ 已证。
2. **真 int8 实现**:topi NCHWc int8 = 真 dp4a/WMMA,数值核验,stage0 比 fp16 快 **1.45×**。✅(微基准)。
3. **量化 = 定性耦合机理**:Pyramid s0=48 失配宽度**结构性无法 build int8**(in_per_g=3÷4),只 s0=64 可 → 对齐耦合从 fp16 定量(2×vs7.9×)升为 int8 定性(能/不能 build)。✅ 机理。**但三轴 ablation 量化证明仍缺**(见上表)。
4. **普适性 = 架构相关**:Pyramid grouped 强/定性 vs CoDriving 标准 conv 可分离(K=Cin×9 大,WMMA 只 pad,软惩罚无定性排除)。🟡 fp16 已证可分离,int8 待复验。

---

## §1 本轮已验证结论(L1/L2,主控读文件/复跑核验过)

### L2 — P×S 三对 ablation:✅ 完成+核验
- 网格 9 宽度(含 s2_128),`detect_wg_pg_pairs` 自动识别 **3 对** rank-flip:
  - pair3 mix_d[48,128,128]/s2_128[64,128,128] @AP0.6369 = **3.83×**(shipped,P_g 在全局 Pareto)
  - pair1 trap25/pad64 @0.5905 = 3.51×(仅机理,被 s2_128 支配)
  - pair2 mix_b/s1_64 @0.6362 = 3.29×(仅机理,被 s2_128 支配)
- HV: A-joint 99.9% / A-serial 85.5% / A-noS 47.7%; Wilcoxon **p=4.88e-4** rank-biserial 1.0; STRUCTURAL PASS(96 起点/对全过); B5 审计所有收敛解真测。
- 闭环 DS 轴(model-est,口径已标):P_g 比 W_g +7.95~+10 DS。
- 文件:`results/{b4_ablation_results.json, b5_convergence_verification.json, latency_lut_pyramid.json(9宽度), ap70_model_pyramid.json}`;图 `multi_agent/figure/b4_*.png`;回写 `auto-tuning/4_design_ablation_proof_v1.md §9.1–9.8`。
- 复跑命令:`PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m framework.run_b4_ablation --seeds 12` + `scripts/phase2/b5_verify_convergence.py`。

### L1 — Q 量化轴:✅ int8 核心完成+核验
**(a) 关键负结果(已定论)**:**TVM relax QDQ-ONNX 路线对 grouped conv 出的是假 int8**(FP32-GEMM+QDQ overhead,TIR conv buffer=float32 + int8-default 普遍慢 11–44% + screenA"证明"实为 groups=1 稠密 conv)。⇒ QDQ-ONNX→relax 路线作废。

**(b) 解决(本轮最大工程成果)**:**直接用 topi `conv2d_NCHWc_int8` 建 int8 grouped conv = 真 int8**。主控独立核验:
- 生成 CUDA 有 **5 个 `__dp4a` + PTX `dp4a.u32.s32`**;MetaSchedule 在 sm90 选了更快的 **WMMA INT8**(`wmma::mma_sync` + INT8 fragment + INT32 累加 + 动态 shared mem)。
- **数值精确**:max_rel_error = **0.0**(int8×int8→int32 精确整数),spot 521.0==521.0。
- 文件:`results/{q_int8_ms_stage0_result.json, int8_correctness_verify.json, q_int8_dp4a_pairs.csv, q_tvm_int8_verdict.md}`;H800 工作脚本在 `/exdata/jichengzhi/s2_tvm/`(复用它改 groups/Cin,别重写)。

**(c) 定性耦合(头条机理)**:
| | s0=48(W_g 失配,in_per_g=3) | s0=64(P_g 对齐,in_per_g=4) |
|---|---|---|
| fp16 | 可用(tuned 7.71×) | 可用(tuned 3.83×) |
| int8 能否 build | ❌ **结构性不可能**(NCHWc IC_BN=4÷4 除零) | ✅ 可用 |
| int8 延迟(stage0) | N/A | **150.1µs = 比 fp16 217.5µs 快 1.45×** |
→ 串行按 fp16-default 锁失配 W_g = 永久失去 int8 路径。失配宽度要用 int8 唯一办法 = 补 in_per_g 3→4 = s0 48→64 = 变成 P_g。

**(d) ★口径限制(务必带)**:int8 数字 = **stage0 单 conv 微基准(s0=64),非全 backbone**;3 个 P_g 对 stage0 维度相同 → 同 150µs,**不区分对、填不进全 backbone P×Q×S HV**。全 backbone int8(stage1/stage2 也做)= 未做的后续工作。

**(e) int8 AP**:`results/q_int8_ap.json` DAIR val 1789,Δap70 ≈ **-0.008**(TRT MinMax 真测,非 simulated)。

---

## §2 L3 / L4 现状与未决

### L3 — CoDriving 对照:fp16 稳,int8 主张被驳回待复验
**点数对照(回答"几个点"):** CoDriving = **4 fp16 backbone 真测点**(p75/p50/p25/base)+ 3 个 s0 探针(s0_48 tuned 崩)+ 6 个 int8 stage0 配置(**未验真 int8**);Pyramid = 9 宽度。**能否找到 Pyramid 那种陷阱:fp16 无(标准 conv 不挑对齐);int8 大概率也只是软惩罚非定性陷阱 —— 找不到干净陷阱正是"可分离/普适性"的论点,不是缺陷。**
- **fp16 多点真测(4 backbone 宽度,H800 TVM,杀估算)** `codriving_tvm_p25_p75.csv`:
  | 宽度 | channels | batch | default µs | tuned µs | ratio |
  |---|---|---|---|---|---|
  | p75 | [16,32,64] | 1★ | 2832 | 2451 | 1.155× |
  | p50 | [32,64,128] | 2 | 3643 | 1609 | 2.264× |
  | p25 | [48,96,192] | 2 | 11885 | 10467 | 1.135× |
  | base | [64,128,256] | 2 | 16680 | 8058 | 2.070× |
  (★p75 ONNX batch=1 定维,绝对值不可直接跨比;p25/p75 ratio 低=MetaSchedule 对 non-pow2 欠调优,非耦合)
- **fp16 s0 失配探针(固定 s1=128,s2=256,仅变 s0)** `codriving_s0probe_fp16.csv` —— **无 rank-flip**:
  | config | channels | default µs | tuned µs | ratio |
  |---|---|---|---|---|
  | s0_32 | [32,128,256] | 15096 | 11218 | 1.346× |
  | s0_48 | [48,128,256] | 15601 | **CRASH** | N/A |
  | s0_64=base | [64,128,256] | 16680 | 8058 | 2.071× |
  default 单调 15096<15601<16680;tuned"翻转"(s0_32 11218 > s0_64 8058)= MS 对 hybrid [32,128,256] 欠调优 artifact(两端都 K÷16 对齐);s0_48 tuned 反复 CUDA illegal-access 崩。**结论:CoDriving fp16 无对齐耦合/无陷阱。**
- **int8 stage0(6 配置,`codriving_s0probe_int8.csv`,★未验真 int8)**:batch=2(relax)Cin32 76.1µs / Cin48 80.9µs / Cin64 96.7µs → **batch=2 无 flip**(80.9<96.7);batch~1(screen,batch 推断)Cin32 50.4 / Cin48 98.9 / Cin64 84.2 → 仅此处有 flip。**agent 报"COUPLED",主控已驳回**(证据不足):① int8 kernel **未验证是真 int8**(dp4a/wmma grep 全失败);② 干净 batch=2 **无** flip,只有来路不明 inferred-batch=1 翻转;③ 标准 conv K=Cin×9 大 → WMMA 只 pad(软惩罚 ~1.5×)≠ Pyramid 定性 build-or-not。
- **诚实 verdict(待写 `results/codriving_coupling_verdict.md`)**:CoDriving = **可分离/弱耦合**;int8 标 OPEN 待 L1 复验;**Pyramid 强/定性 vs CoDriving 弱/可分离的非对称 = 普适性贡献**。
- **未决**:由 **L1 用 work 的 CUDA-dump 工具**复验 CoDriving 标准 conv int8 是否真 int8 + Cin=48 vs Cin=64 同一干净 batch 是否真有 rank-flip。
- 文件:`results/{codriving_tvm_p25_p75.csv, codriving_s0probe_fp16.csv, codriving_s0probe_int8.csv}`。

### L4 — Pyramid 网格扩充:在跑未落地
- **DELIVERABLE A(AP 崖口)**:激进剪枝 80/87/93% structural prune + stage_a 协议 finetune(DAIR val 1789)→ 找 AP70 真悬崖(把 AP 轴从高原变真曲线)。4090 GPU0/1。
- **DELIVERABLE B(rank-flip 对扩到 ≥5,跨 ≥4 个 AP70 档 0.55–0.63,≥3 对 shipped)**:新 s0 失配/对齐宽度对 → H800 TVM fp16 default+tuned latency(fresh workdir 逐宽度)+ finetune AP → 并入 LUT/AP 表 → 重跑 `run_b4_ablation`。
- 停止判据(满足即停,见 v2 §3-L4):AP 找到悬崖 / ≥5 对跨 ≥4 档 ≥3 shipped / 加一批后 HV gap 变化 <2% 且无新档新对 / config ≥3×budget。

---

## §3 重启任务链(新窗口照这个起)

> **建议:不要再起多层 orchestrator agent**(本轮它们卡在 dispatch 不落 GPU)。直接起**单层 hw/sw agent 或主控直驱**,每个 agent 明确"立刻 ssh H800 跑 + 回报 PID + GPU util 非零",并复用现成脚本不重写。

**GPU 分配(2026-06-21 晚, 可用 = H800 GPU0/1/2/6)**:⚠️ H800 GPU3/4/5 = 闭环仿真(CARLA,他人),勿用。
- **GPU0** = CoDriving int8 复验(hw,复用 q_int8 脚本)
- **GPU6** = L4 新宽度 H800 TVM latency
- **GPU1/2** = 备用(全 backbone int8 若做 / 额外 L4 latency)
- **4090 GPU0/1** = L4 AP 崖口 finetune(跑前 nvidia-smi 确认 util 0%)

> **★dp4a"测 int8"核心已完成,不必再为测 int8 启动 hw agent。** 剩下三轴 ablation 是**纯 Python 无 GPU**(主控直驱)。

**并行线(按优先级)**:
1. **★三轴 ablation(P0,纯 Python,无 GPU,最该补)**:把"定性耦合"接进 `framework/search_three_arm.py` 的 `CostModelPQS`/`run_*_pqs`/`detect_q_rank_flip_pairs` → 跑 P×Q×S 三臂(A-joint-PQS vs A-serial-lock-quant)→ 出 HV+Wilcoxon。**判据**:int8 路径仅对齐宽度可达(s0=48 不可 build int8)作硬约束 → A-serial 锁了 fp16-default 的失配 W_g 就拿不到 int8 → 量化证明三轴强耦合。**这是"证明三轴强耦合"的缺失一步**,做完才算证。(注:int8 latency 用 stage0 1.45× 作宽度内 int8 收益的代表,或建合成全 backbone int8 估算并标口径。)
2. **CoDriving int8 复验(GPU0,hw)**:复用 q_int8_ms_stage0 脚本改 groups=1/Cin=48,64;dump CUDA grep dp4a/wmma + 数值 max_rel_err + 同 batch=2 测 Cin48 vs 64 有无 rank-flip。**预期**:标准 conv int8 可 build(无定性陷阱),至多软效率惩罚 → 坐实"CoDriving 可分离" 的非对称。
3. **L3 verdict(无 GPU)**:写 `results/codriving_coupling_verdict.md`(§2 诚实 verdict,引用 L1 复验)。
4. **L4(4090 0/1 + H800 GPU6)**:AP 崖口(80/87/93% finetune)+ ≥5 rank-flip 对 + 重跑 ablation(§2 DELIVERABLE A/B)。
5. **[可选,重] 全 backbone int8**(GPU1/2):stage1/stage2 也建 int8 → 全 backbone int8-tuned 延迟,把三轴 ablation 从"定性约束"升级为"真延迟 HV"。不做则用定性框架(已足够立论)。

**收尾**:① 三线落地后把已验证结论整合进 `doc4 §9`(已大部分回写,补 Q 轴 §9.8b + L4 新对 + L3 非对称)。② **commit**(用户授权)已验证的 L2+L1。③ L5 真闭环 = 其他 agent,本团队不安排。

---

## §4 环境 / 铁律 / 本轮教训

**环境**:
- 真仓库 `/home/jichengzhi/V2X`(⚠️ UniV2X 是断链空壳)。
- framework 脚本用 conda:`/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`(+ `PYTHONPATH=/home/jichengzhi/V2X`)。
- H800 ssh:`sshpass -p 12345678 ssh -p 30001 -o ConnectTimeout=45 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`(连太频会限流,批量执行)。
- H800 TVM env:`export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)`;python `/exdata/jichengzhi/tvm310/bin/python`;工作目录 `/exdata/jichengzhi/s2_tvm/`。
- 内核 `framework/search_three_arm.py`(`detect_wg_pg_pairs` + `CostModelPQS`/`run_*_pqs`/`detect_q_rank_flip_pairs`);驱动 `framework/run_b4_ablation.py`;B5 `scripts/phase2/b5_verify_convergence.py`。

**铁律(必守)**:
- **真 int8 = 必须 dump CUDA grep `dp4a`/`wmma::mma_sync` + 数值 vs fp32 max_rel_err**。QDQ-ONNX→relax 路线出假 int8,不可用。绝不 simulated/fp16 代。
- 延迟只在 **idle pinned GPU**(util 0%/mem≤50MiB)测;**fresh workdir 逐宽度 + subprocess 隔离**(复用 workdir 出假 ratio≈1.0 / CUDA illegal-access)。
- **不跨硬件/口径混**:`latency_lut_pyramid.json`=H800 TVM;`latency_lut_pyramid_q.json`=**4090 TRT(别当 H800)**;stage0 微基准 ≠ 全 backbone。
- **不轻信 agent 自报**:凡"已测/已修/已 build"主控复跑/读文件/grep 核验(本轮抓到:假 int8、跨硬件混 LUT、未验数值就报完成、CoDriving 过度声称 coupled)。

**本轮教训(写 [[feedback-no-premature-impossible]] 同类)**:
- "TVM 无 int8"是 QDQ-ONNX 路线的局部真相;直接 topi NCHWc int8 可行 → 穷尽路线再下结论。
- grouped conv int8:per-group K=3/4 << WMMA K,dp4a(K÷4)才是路径;但 MetaSchedule 可能用全 conv reduction(Cin×kh×kw)拼出 WMMA。失配宽度真正卡在 **NCHWc 通道打包 ÷4**(in_per_g=3 建不出)= 定性排除。
- 多层 orchestrator agent 易卡 dispatch 不落 GPU;重活宁可单层 agent/主控直驱 + 强制回报 PID。

---

## §5 关键文件清单
| 文件 | 内容 | 状态 |
|---|---|---|
| `results/b4_ablation_results.json` | 3 对 P×S ablation | ✅ 验证 |
| `results/b5_convergence_verification.json` | B5 审计 | ✅ |
| `results/latency_lut_pyramid.json` | H800 TVM 9 宽度 fp16 | ✅ |
| `results/ap70_model_pyramid.json` | 真 AP70 锚点 table | ✅ |
| `results/q_tvm_int8_verdict.md` | Q 轴定论(假 int8→真 dp4a/WMMA+定性耦合) | ✅ 核验 |
| `results/q_int8_ms_stage0_result.json` | int8 150µs vs fp16 217µs(stage0) | ✅ |
| `results/int8_correctness_verify.json` | max_rel_err=0.0 | ✅ |
| `results/q_int8_dp4a_pairs.csv` | 6 宽度 dp4a 可行性(s0=48 NOT_APPLICABLE) | ✅ |
| `results/q_int8_ap.json` | int8 AP Δ-0.008 DAIR | ✅ 真测 |
| `results/codriving_{tvm_p25_p75,s0probe_fp16,s0probe_int8}.csv` | CoDriving 多点 | fp16 稳/int8 待复验 |
| `results/codriving_coupling_verdict.md` | CoDriving 诚实 verdict | ⏳ 待写 |
| `auto-tuning/4_design_ablation_proof_v1.md §9` | 命题+真数据 | ✅ 大部分回写,待补 Q/L4/L3 |
