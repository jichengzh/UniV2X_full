# W_g 存在性探针方案 (v1, 2026-06-19 更正版 — TVM prune×schedule)

> ★**2026-06-19 两点更正(用户强调)**:
> 1. **W_g 定义更正**: W_g = **单维(贪心所搜的那一维)最优、但多维(全局)非最优**的配置点 —— 即单侧贪心搜索会**落入并卡住的局部最优**。贪心选它, 因为它在所搜单维上最优; 它非全局, 因为到达真全局需接受一个单维**次优**的选择(贪心已把它丢弃)。[旧稿把 W_g 写成"单维被支配但联合解锁全局"——那描述的是**被错过的点**, 不是吸引子。以本更正为准: **W_g = 贪心吸引子(单维最优/多维非最优)**; 另设 P_g = 被错过的全局点(单维次优/多维最优)。]
> 2. **平台更正: 已迁移到 TVM, 不再以 TRT/INT8 结果为重心**。本探针主轴 = **TVM 的 prune × schedule**(MetaSchedule tuned vs default dlight)。量化(INT8)因 TVM relax 无 INT8 pass, **降为 future/次要**, 不作为 W_g 主实验依赖(原稿围绕 TRT INT8 latency 的设计已废, 见 §7)。

> **定位**: ablation 有效性证明([4_design_ablation_proof_v1.md](../methods/design/auto-tuning/4_design_ablation_proof_v1.md))的 **gating 前置实验**。W_g 存在 → ablation 有载体; 不存在 → 换载体/机制 or 退 story B。

---

## §0 为什么 gating
ablation 要证"单侧贪心搜索→局部最优(非全局)"。这需要一个**贪心会落入的吸引子 W_g(单维最优/多维非最优)**真实存在, 且存在贪心够不到的更优全局点 P_g。**没有这对点, 贪心天然走到全局最优, 证明无从谈起。** 先低成本探明, 再决定 ablation 投不投。

---

## §1 W_g 的精确定义 + 判定准则(更正版)
设宽度集合 {W}, schedule ∈ {default dlight, MetaSchedule-tuned}。串行贪心按单维(P, 在 default schedule 下)选最优宽度并锁定, 再 tune schedule。
- **W_g(贪心吸引子)**: 宽度 P\*, 它在**单维评估(default schedule)下最优**(贪心据此选它), 但 **P\*+其最佳 schedule 不是全局最优**。
- **P_g(被错过的全局)**: 宽度 P′, 它在 default schedule 下**次优**(贪心丢弃), 但 **P′+tuned-schedule 是全局最优**, 支配 P\*+tuned。
- ★**判定信号(2026-06-19 锐化, 用这个不用纯 latency)**: **(AP, latency) Pareto 重排** —— **default-schedule 的 (AP,lat) Pareto ≠ tuned-schedule 的 (AP,lat) Pareto**。W_g = 在 default-Pareto 上、但 tuned 后掉出(被支配)的宽度; P_g = default 下被支配(贪心丢)、tuned 后进 Pareto 的宽度。根因 = **各宽度 schedule 调优余量不均匀**(对齐/算子形状相关), 使**未调度的估计误排宽度**。
- (纯 latency 排序大概率不 flip——p50 在 default/tuned 都领先; **必须带 AP 轴**才看到重排, 因 AP 给某些宽度 default 下的 niche, tuning 后被高余量宽度抹掉。)
- 成立则: 串行(按 default/未调度估计选宽度)锁 W_g, 永远到不了 P_g → **局部最优实证**。

---

## §2 TVM 主轴 = prune × schedule(机制 + 现有数据)
H800 TVM schedule-LUT(default dlight vs MetaSchedule-tuned, 真测, `results/gap1_schedule_lut.json`):
| width | default µs | tuned µs | **调优余量** |
|---|---|---|---|
| base 64/128/256 | 56317 | 6214 | **9.06×** |
| p50 32/64/128 | 26242 | 3011 | 8.72× |
| trap25 48/96/192 | 42355 | 21615 | 1.96× |

**调优余量按宽度差异巨大(1.96×–9.06×)= 重排的来源**。纯 latency 序无 flip(p50 两者皆最优)。**但加上 AP 轴就出现 Pareto 重排, 且 trap25 已是现成 W_g 候选**:

| width | AP70 | default µs | tuned µs | default-Pareto? | tuned-Pareto? |
|---|---|---|---|---|---|
| base 64/128/256 | 0.631 | 56317 | 6214 | ✓ | ✓ |
| **trap25 48/96/192** | **0.590** | 42355 | 21615 | **✓(0.59 niche, 比 base 快)** | **✗ 被 base 支配** |
| p50 32/64/128 | 0.564 | 26242 | 3011 | ✓ | ✓ |

- **trap25 = 现成 W_g**: default schedule 下它在 Pareto 上(0.590 AP niche, 比 base 快)→ 只看未调度估计的串行会选它; tuned 后 base=(0.631,6214) 支配 trap25=(0.590,21615)(AP 更高+快 3.5×)→ trap25 掉出 → 它是"单维最优/多维非最优"的贪心吸引子。**这不是"trap25 慢"的老故事, 是"trap25 在 default 下看着 Pareto 竞争、贪心才上当"。**
- ⇒ 探针**大概率为正**(不像之前 INT8 那个可能为空)。但**不能只摆 trap25 一点**(那是枚举)——须扩 grid 证这是普遍误排现象 + 跑真搜索。
- **机制须实测**: schedule 调优余量不可靠预测(simulated 不可信), 只能 MetaSchedule 实跑。

---

## §3 探针设计(纯 latency 先行, TVM, H800 idle)
**Phase 1 — 宽度 grid × {default, tuned} 的 (AP, latency) Pareto 重排扫描**
- **grid(~6-8 宽度)**: base[64,128,256] / trap25[48,96,192] / p50[32,64,128] / p75[16,32,64] + 补 ~3 个 stage 级对齐类混合宽度(部分 ÷32 部分非÷32, 制造 schedule 余量分散)。用已打通的 DepGraph backbone 子图流程生成 ONNX。
- 每宽度测 **default dlight + MetaSchedule-tuned latency**(H800 全空, min-of-N)。**latency-first**: 先用已有 AP(base/p25/p50/p75 = stage_a)拼 (AP,lat); 新宽度先只摆 latency, 进入候选再 finetune 补 AP。
- 画 **default-(AP,lat)-Pareto vs tuned-(AP,lat)-Pareto**, 标出所有 **W_g 类点**(default-Pareto 上、tuned 掉出)与 **P_g 类点**(default 外、tuned 上)。目标: 证明这是**普遍误排**(多个点), 非 trap25 单点。

**Phase 2 — 确认 W_g 是局部最优(加 AP 轴构 Pareto)**
- 对 flip 候选 (P\*, P′) 补 AP(finetune 后真测 AP70), 在 **(AP70, tuned-lat)** 平面确认:
  - 串行(按 default 选 P\*, 再 tune)收敛到 P\*+tuned;
  - 联合(P×schedule 同搜)收敛到 P′+tuned, 且支配 P\*+tuned。
- ⇒ W_g(=P\*)被贪心锁死、P_g(=P′)被错过 = 结构性局部最优实证。

---

## §4 判读 + kill criterion + 退路
- **找到 (W_g, P_g) flip 对** → ablation 三臂(串行锁 W_g / 联合到 P_g)有载体, 开跑。
- **扫遍无 flip**(诚实负结果, 有可能): 在该模型 prune×schedule 上不存在结构性局部最优 → ①换载体(CoDriving/V2X-ViT 的 schedule 调优余量分布可能不同)②诚实判据"该空间可分离"(故事 B)③升级机制(加 schedule 内多旋钮 / per-layer)。
- **铁律**: 不为要 flip 硬挑参数; 无则报无 = "何时可分离"判据。

---

## §5 平台口径 / 资源复用
- **平台**: **H800 TVM(全空, idle)** 主 —— relative/机制坐实(Gap1 是相对比较, 平台无关)。**不拼 TRT/跨平台。**
- **复用**: schedule-LUT 雏形(`gap1_schedule_lut.json`)+ backbone ONNX(H800 `/exdata/.../s2_tvm/models/{base,p50,trap25}`)+ MetaSchedule 流程(`s2_2e_e2e.py` / `gap1_run_v2.py`)+ stage_a FP16 AP(候选宽度 AP 须新 finetune)。
- **新建**: 新宽度的剪枝 ckpt + backbone ONNX(DepGraph; latency 不需 finetune, AP 需)。
- latency 必空闲卡 min-of-N。

---

## §7 量化(INT8)轴 — 降为 future, 不作主依赖
- TVM relax **无 INT8 量化 pass**; 之前围绕 TRT INT8 latency 的 W_g 设计(找"FP16 平庸但 INT8 解锁"宽度)**已废**(本稿不再用 TRT 结果)。
- 量化轴若日后纳入须 BYOC-TRT, 跨口径, 与 TVM schedule 臂不混; 列为 future, 非本探针前提。
- **本探针只在 TVM prune×schedule 上找 W_g/P_g**。

---

## §8 已定默认(2026-06-19, 可改)
1. **grid**: §3 的 base/trap25/p50/p75 + ~3 个对齐混合宽度(latency 几乎免费, 先扫)。
2. **latency-first**: 是 —— 先 latency + 已有 AP 拼 Pareto, 候选才 finetune 补 AP。
3. **载体**: Pyramid 先行(grouped, 余量分散大 → 重排明显, trap25 已是候选); CoDriving 同跑作 **model-dependent 对照**(标准 conv, 余量小 → 预期重排弱/可分离)。
4. **平台**: H800 TVM(全空), relative 坐实, 不拼 TRT/跨平台。
5. **退路**: 若某模型 prune×schedule 无重排 → 换载体 / 升级机制(schedule 内多旋钮)/ 接受"可分离"判据(故事 B)。

---

## §9 ★Phase 1 实测结果 (2026-06-20, Pyramid, H800 TVM) — W_g 探针 = 正

> 数据 `results/gap1_grid_corrected.json` + `results/diag_retune.csv`(H800)。全部用**可信脚本 `s2_2e_e2e.py` 逐宽度独立进程**重测(fresh work dir, idle GPU)。
> ⚠️ **方法学坑(已记取)**: 首版 `gap1_grid_v2.py` 把所有宽度塞进**单进程循环 + work-dir 存在即复用**, 撞上早先孤儿化运行残留的同名 `ms_work_grid_*` 残缺 db → apply 假数 ratio≈1.0(p75 假报 1.0, 真值 26×)。**教训: tune+apply 必须 fresh work dir + 逐宽度独立进程(顺带隔离 CUDA illegal-access 崩溃)。**

**调优余量(ratio=default/tuned)× 对齐:**
| 宽度 | num_filters | in_per_g | default µs | tuned µs | ratio |
|---|---|---|---|---|---|
| base | [64,128,256] | [4,8,16] 全2^k | 56321 | 6320 | **8.91×** |
| p50 | [32,64,128] | [2,4,8] 全2^k | 26242 | 3011 | **8.72×** |
| p75 | [16,32,64] | [1,2,4] 全2^k | 12689 | 484 | **26.24×** |
| **iso_s0** | [48,128,256] | **[3**,8,16] 仅s0失配 | 51249 | 21752 | **2.36×** |
| iso_s1 | [64,96,256] | [4,**6**,16] 仅s1失配 | 51622 | 6911 | 7.47× |
| iso_s2 | [64,128,192] | [4,8,**12**] 仅s2失配 | 51992 | 7244 | 7.18× |
| trap25 | [48,96,192] | [3,6,12] 全失配 | 42355 | 21615 | **1.96×** |
| **pad64** | [64,96,192] | [4,6,12] s0修复对齐 | 47407 | **6152** | **7.71×** |

**(AP70, latency) Pareto 重排(AP 已知: base/p50/p75/trap25/pad64; trap25 与 pad64 同 AP70=0.590, 仅 stage0 对齐不同——pad64 = trap25 权重把 s0 零填充48→64, 输出不变故 AP 不变):**
- default-Pareto = {base, p50, p75, **trap25**}; tuned-Pareto = {base, p50, p75, **pad64**}。
- **★W_g = trap25 / P_g = pad64, 完整配对(仅差 stage0 对齐 = 干净单变量):**
  - **W_g = trap25**: 0.59-AP niche 内 default 42355 < pad64 47407(s0 零填充多算 FLOPs), 贪心据 default 估计选它; tuned 仅 1.96× → 21615µs, 被 pad64 严格支配 → 掉出 tuned-Pareto。= **单维(default)最优 / 多维(tuned)非最优**的贪心吸引子。
  - **P_g = pad64**: default 下被 trap25 支配(贪心丢弃), tuned 7.71× → 6152µs 成为 0.59-AP niche 全局最优, 进 tuned-Pareto。= **单维(default)次优 / 多维(tuned)最优**的被错过点。
  - ⇒ 串行(按 default 估计选宽度→锁 trap25=W_g, 再 tune)结构性到不了 pad64=P_g; 联合(宽度×schedule 同搜 / 宽度感知 schedulability)才选 pad64。**ablation 所需的教科书式 W_g/P_g 对**。

**机理 = 对齐余量塌缩"定位到 stage0"(系统性, 非单点 trap25):**
- 仅 stage0 失配(iso_s0, in_per_g=3) 就把余量从 ~9× 砸到 **2.36× ≈ trap25 1.96×**; 仅 s1/s2 失配(iso_s1/s2) 仍 7.2–7.5×(接近对齐); **pad64(修复 s0、s1/s2 仍失配) 回到 7.71×**。⇒ **罚分由 stage0(最高分辨率 128×256 的 grouped conv, in_per_g=3 最奇异)主导**, 非全网均摊。doc4 §0/§2.2 要求的"非枚举、可消融"机理证据齐全。

**推翻旧结论(已实测确认):**
- 旧 `gap1_schedule_lut.json` 的 **"pad64 ratio=1.0 / pad 局部救援负结果 / 对齐是全网属性"是 buggy harness 假数**(单进程循环复用撞名残缺 db)。`pad64_retest` 实测 **7.71×** ⇒ **pad 救援有效**, 且 pad64 正是 P_g。

**结论 → ablation 有载体**: W_g 存在性闸门(doc4 §6.1/§8.1) **通过, 且 W_g/P_g 对齐配对完整**。Phase 2(可选加固) = 给 iso_s0 finetune 补 AP70 确认它也是 trap; 然后跑 doc4 三臂真搜索 ablation。
