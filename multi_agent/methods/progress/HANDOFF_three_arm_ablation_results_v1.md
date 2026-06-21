# HANDOFF — 三臂 ablation 结果与收尾 (results v1, 2026-06-21)

> **接手先读本页 + `HANDOFF_three_arm_ablation_exec_v1.md`(原计划 B1–B5)+ `results/B4_integration_checklist.md`(整合细节/发现)。**
> 本页 = 过夜执行的**实测结果** + **唯一两个收尾待办**(B4 多对判据 + 闭环轴整合)+ B5。

---

## §0 一句话状态 + 接手第一步
> **★[2026-06-21 收尾完成 update]** §3 两个收尾 + B5 **已全部做完并验证**,doc4 §9 已回写。本节以下旧描述保留作历史;最新状态见本 update 块 + §9。

**最终状态(2026-06-21)**:B1/B2/B3/B6/B7 + **B4 收尾 + B5 全部完成**。
- B4 结构性判据已**泛化多对**(`detect_wg_pg_pairs` 自动识别,不再硬编码 pad64);**STRUCTURAL RESULT: PASS**(2 对 × 12 seed × 6 起点 = 72 run 全过)。
- 闭环 DS 轴**已接入**(plan B,`CostModel` rec 附 `ds_model`/`e2e_orin_ms_est`;AP 轴保留)。
- B5 审计完成(`scripts/phase2/b5_verify_convergence.py`):所有臂收敛解均真测点。
- **核心结果**:A-joint 99.6% / A-serial 86.0% / A-noS 48.0% HV,Wilcoxon **p=4.9e-4**。**1 个干净 shipped 协同赢 = pair2(s1_64 vs mix_b)3.29× @iso-AP70=0.6362**(两端点都在各自臂 Pareto 上、都真测);pair1(trap25/pad64)3.51× 是**仅机理示范**(pad64 被 s1_64 全局支配,非出货倍率)。闭环:同 AP 下 P_g 比 W_g 高 +7.95/+10 DS。
- 产物:`results/{b4_ablation_results.json, b5_convergence_verification.json}` + `multi_agent/figure/b4_*.png` + doc4 `4_design_ablation_proof_v1.md §9`。

**下一步(全新工作)**:§9.7 列出 —— ① 量化 Q 轴(prune×quant×schedule,真 TRT INT8);② pair3 s2_128 CUDA 崩排查补第 3 AP 档;③ CoDriving 可分离对照臂;④ 真 CARLA 闭环替换 DS 估算。

---
### (以下为收尾前的历史记录)
~~B1/B2/B3/B6(AP扩展)/B7(闭环建模)**全部完成并验证**。**B4 已在真数据上跑通**,但**判据/报告未干净收尾**:① 结构性判据硬编码在单对 pad64;② 闭环 DS 轴还没接进 B4。**B5 未做**。~~ → 均已完成,见上。

---

## §1 各任务状态
| 任务 | 状态 | 关键产出 |
|---|---|---|
| B1 延迟网格(H800 TVM) | ✅ 完成 | `results/latency_lut_pyramid.json`(直接网格,18-19 宽度真测) |
| B2 AP 模型 | ✅ 完成 | `results/ap70_model_pyramid.json`(已加 `table` 键=9 真AP宽度) |
| B3 三臂搜索内核 | ✅ 完成 | `framework/search_three_arm.py` |
| B6 AP 扩展(DepGraph finetune) | ✅ 完成 | `results/ap70_depgraph_expansion.json`(mix_b 0.6362/mix_d 0.6369) |
| B7 闭环目标建模 | ✅ 完成 | `results/closedloop_objective_model.json` + `scripts/phase2/closedloop_objective_query.py` + `multi_agent/methods/design/closedloop_b4_plugin_v1.md` |
| **B4 跑三臂 + 收尾** | ✅ **完成(多对判据 PASS + 闭环 DS 接入)** | `results/b4_ablation_results.json` + `multi_agent/figure/b4_{hv_boxplot,convergence,pointcloud}.png` |
| **B5 收敛解真测审计** | ✅ **完成** | `results/b5_convergence_verification.json` + `scripts/phase2/b5_verify_convergence.py` |
| **doc4 回写** | ✅ **完成** | `4_design_ablation_proof_v1.md §9`(B1–B5 真数据 + shipped vs 机理诚实区分) |

---

## §2 实测结果(真数据,已核验)

### 2.1 W_g/P_g 对(机制复现 = 核心科学结果)
失配 s0(in_per_g=3)= W_g 调优只 ~2× 卡住;补齐 s0→64(in_per_g=4,零填充权重不变=同 AP)= P_g 调优 ~7.9× → 同精度下 P_g 远快于 W_g。**两对独立复现,两个 AP 档:**

| 对 | W_g(tuned µs, ratio) | P_g(tuned µs, ratio) | 同 AP70 | P_g 比 W_g 快 |
|---|---|---|---|---|
| 1 trap25[48,96,192]/pad64[64,96,192] | 21615 (1.96×) | 6152 (7.71×) | 0.5905 | 3.51× |
| 2 mix_b[48,64,256]/s1_64[64,64,256] | 19405 (2.14×) | 5895 (7.88×) | 0.6362 | 3.29× |
| 3 mix_d[48,128,128]/s2_128[64,128,128] | 21071 (2.01×) | **崩(无延迟)** | 0.6369 | — |
- pair3 的 P_g=s2_128 [64,128,128] TVM 调优**持续 CUDA illegal-access 崩**(2 次 exit134),延迟拿不到 → **降级为 2 完整对**。mix_d(W_g)延迟已测。

### 2.2 B4 三臂(8 宽度真网格,12 seed,LUT=b1_direct 真延迟,AP=b2 真AP)
- HV:**A-joint 99.6% / A-serial 86.0% / A-noS 48.0%**(ref_hv=6.95e3)。真实显著差距(占位数据时退化为 99.9% vs 100%)。
- **Wilcoxon A-joint vs A-serial p=4.9e-4**,rank-biserial=1.0(12 seed 全偏向联合)。
- pair-2 干净倍率:A-serial 收敛 mix_b/tuned 19405 vs A-joint s1_64/tuned 5895 = **3.29× @ iso-AP70=0.636**。

---

## §3 ★B4 收尾两个待办(唯一阻塞)

### 3.1 泛化多对结构性判据(`framework/run_b4_ablation.py`)
**问题**:当前判据/headline 硬编码在**单对 pad64**:报 "STRUCTURAL RESULT: CHECK"(非 PASS),"serial locks W_g drops P_g across all 36 = False",且 EXACT-level headline 选了错配标签(报 "8.91× @ AP0.6309" = base default-vs-tuned,标签错位)。**底层数据/结论是对的**,只是报告逻辑没泛化。
**修法**:从 LUT 自动识别所有 W_g/P_g 对(同 AP70、s0 失配 vs 补齐:W_g ratio≈2×、P_g ratio≈7.9×),对**每一对**报:① A-serial 是否结构性漏掉该 P_g 的 tuned 分支;② 该对的 iso-AP 延迟倍率(W_g_tuned / P_g_tuned)。headline = 各对倍率(pair1 3.51× / pair2 3.29×)而非单点。多起点锁定统计也要按"每对是否被锁 W_g"分别报。

### 3.2 接入闭环 DS 轴(用户明确要求,尚未做)
Agent-CL 已建好模型+plugin 方案(`multi_agent/methods/design/closedloop_b4_plugin_v1.md`,方案 B:最小侵入)。**在 `CostModel.evaluate()` 的 rec 里追加** `ds_model` 和 `e2e_orin_ms_est`:
```python
from scripts.phase2.closedloop_objective_query import backbone_to_e2e_latency, driving_score
rec["ds_model"] = round(driving_score(ap, backbone_to_e2e_latency(lat), beta=0.0), 2)
rec["e2e_orin_ms_est"] = round(backbone_to_e2e_latency(lat), 1)
```
**口径必标**:model-estimated(CoDriving τ 曲线外推 Pyramid)、e2e ±30%、β=0(无 AP→DS 实测)、非真 sim。闭环对论点的贡献:同 AP 下 trap25 DS 85.6 vs pad64 DS 95.6(差 10 分)= 驾驶安全角度独立支撑。**用户强调 AP 轴必须保留**(本就是目标轴,别删)。

### 3.3 重跑 + 验证
`python -m framework.run_b4_ablation --seeds 12` → 确认 PASS(多对)+ DS 轴出现在 rec + 3 张图重绘。复跑核验(读 JSON/图)。

---

## §4 B5 — 收敛解真测(未做)
好消息:**两对的 W_g/P_g 延迟都已是真 MetaSchedule tuned + 真 finetune AP**(不是估算),所以 B5 大部分=**确认各臂收敛解 == 已测的对端点**:A-serial 收敛到 W_g(trap25/mix_b),A-joint 收敛到 P_g(pad64/s1_64)。headline = iso-AP70 延迟倍率(3.51×/3.29×,已真测)。若要更严:对收敛解再独立复跑一次 tuned 延迟核验(H800,builder timeout=300,`s2_2e_bumped.py`,fresh workdir)。

---

## §5 文件 / 命令 / 数据位置
- **整合脚本**(从原始数据重建 LUT+AP 表):`scripts/phase2/b4_integrate.py`(读 gap1_grid_corrected.json + lut_results_grid.csv + ap70_depgraph_expansion.json → 写 latency_lut_pyramid.json + ap70_model_pyramid.json 的 table 键)。
- **B4 驱动器**:`framework/run_b4_ablation.py`;**内核**:`framework/search_three_arm.py`(含 `candidate_widths`=取 apm.exact ∩ 可定价延迟;`LatencyLUT`/`APModel` 接口,b1_direct/b2 模式)。
- **数据**:`results/{latency_lut_pyramid.json, ap70_model_pyramid.json, ap70_depgraph_expansion.json, lut_results_grid.csv, gap1_grid_corrected.json, b4_ablation_results.json}`;`data/stage_a_ap_real.parquet`。
- **图**:`multi_agent/figure/b4_{hv_boxplot,convergence,pointcloud}.png`。
- **设计/纪律**:`multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md`(命题+四层证据,收尾后回写结果);`results/B4_integration_checklist.md`(整合细节+所有发现)。
- **H800**(只 B5 补测才需):`ssh -p 30001 jichengzhi@222.95.84.215`(pw 12345678);env `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=0`;tuner `/exdata/jichengzhi/s2_tvm/s2_2e_bumped.py`(timeout=300 已修);python `/exdata/jichengzhi/tvm310/bin/python`。**纪律**:fresh workdir 每点、idle GPU 测延迟、continue-on-error。

---

## §6 关键发现 / 必带 caveat(写论文/报告时)
1. **加性延迟模型失败** → 用离散真测网格(调优余量是全局对齐属性、非逐 stage 可加)。
2. **AP 轴弱(过参数化)**:单 stage/混合剪枝 finetune 后 AP 几乎不掉(~0.63);只有激进均匀剪枝才掉(p50 .564/p75 .530)。**AP 轴仍保留为目标轴**(用户要求),即使浅。
3. **协议一致性**:B6 扩展用 stage_a 一致协议(structural_prune L1 + epoches=31,gate pruned25=0.5931≈0.5905 验证);B2 的 iso/mixed 是另一协议面(L1-transfer),**不要混进同一 AP 网格**。注:mix_b/mix_d AP(.636/.637)略 > base(.6309),因多 8 epoch finetune——对内 AP 相等由零填充权重恒等保证,跨对绝对值差是这个原因。
4. **s2_128 持续崩** → pair3 P_g 缺失,降级 2 对。2 对(两 AP 档)已足够支撑"结构性、非 cherry-pick"。
5. **闭环 = 估算非真测**:Pyramid 闭环需装 CARLA(本机没装),留作后续真验证。
6. **跨硬件不混**:延迟全程 H800;不与 4090/Orin 数混 Pareto。

---

## §7 之后(prune×schedule 完成后)
**量化 Q 轴**(`HANDOFF_three_arm_ablation_exec_v1.md §5`,用户强调别忘):prune×**quant**×schedule 三轴。Q 需真 TRT INT8(relax 无 INT8 pass),跨口径不混。INT8 会放大 W_g/P_g 不改本质。
