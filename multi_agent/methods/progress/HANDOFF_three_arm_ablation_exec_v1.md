# HANDOFF — 三臂搜索 ablation 执行计划 (B1–B5) + Agent 分工 + 量化持久待办 (v1, 2026-06-20)

> **接手先读本页**。本页 = 三臂真搜索 ablation 的可执行计划(B1–B5)、可并行的 Agent 分工 DAG、量化(Q)轴的持久待办。
> **上游已坐实**: W_g 探针 Phase 1 = 正(`design_wg_probe_int8_plan_v1.md §9` + `results/gap1_grid_corrected.json` + `results/gap1_grid_retune_h800.csv`)。本页据此推进。
> 关联: 架构设计 [auto-tuning/1-4](../design/auto-tuning/)(Space Building/Cost Model/Exploring/Ablation) · 命题/论证纪律 [4_design_ablation_proof_v1.md](../design/auto-tuning/4_design_ablation_proof_v1.md) · 大局 [HANDOFF_codesign_nextstep_v1.md](HANDOFF_codesign_nextstep_v1.md)。

---

## §0 一句话状态 + 接手第一步
框架**已设计未建**(searcher_v0 只有采样+约束, 无 schedule 轴/评估器/Pareto)。W_g/P_g 载体**已坐实**(真测)。**接手第一步 = 并行起 3 个 agent 跑 B1/B2/B3**(§4 分工), 三条线无相互依赖, 完成后主控合 B4(跑三臂)→B5(真测验证)。**三臂必须是真搜索(搜索器在 cost model 上自己走), 不是枚举**(doc4 §0 红线; 已做完的 W_g 探针是枚举, 那只是存在性闸门, 不是 ablation)。

---

## §1 要证的命题 + 已坐实载体
**命题**: 单侧/串行贪心搜索因**结构性**原因陷局部最优、到不了全局; 联合搜索能到。**靠真跑搜索过程比收敛解, 不靠枚举**(否则"串行又不傻"一句打穿)。

**已坐实载体(全真测, H800 TVM, backbone-only, idle GPU)**:
- **W_g = trap25[48,96,192]** (in_per_g=[3,6,12]): default 42355µs/tuned 21615µs, **ratio 1.96×**, AP70 0.590。default-Pareto 上(贪心据未调度估计会选), tuned 后被 pad64 支配 → 掉出。= 单维(default)最优/多维(tuned)非最优的贪心吸引子。
- **P_g = pad64[64,96,192]** (in_per_g=[4,6,12], = trap25 权重 s0 零填充48→64, AP 不变): default 47407µs(被 trap25 支配)/tuned **6152µs(ratio 7.71×)**, AP70 0.590。tuned 后成 0.59-AP niche 全局最优 → 进 tuned-Pareto。= 单维次优/多维最优的被错过点。
- **机理 = 对齐余量塌缩定位到 stage0**(iso 消融): 仅 s0 失配 iso_s0 2.36×≈trap25; 仅 s1/s2 失配 iso_s1/s2 7.2–7.5×; 全对齐 base/p50/p75 8.7–26×。⇒ stage0(最高分辨率 grouped conv, in_per_g=3 最奇异)主导, 非全网均摊。
- **数据**: `results/gap1_grid_corrected.json`(分析) + `results/gap1_grid_retune_h800.csv`(7宽度真值) + `scripts/phase2/gap1_grid_analyze.py`(Pareto算法)。

---

## §2 三臂定义(doc4 §1; 变量隔离)
| 臂 | 搜索空间 | 过程 | 隔离变量 |
|---|---|---|---|
| **A-joint** | P×S 同搜 | 多目标搜索器在 剪枝宽度×schedule 联合空间搜 | 基准 |
| **A-noS** | 只 P (S 固定 dlight default) | 只搜宽度, schedule 永远 default | vs joint = schedule 进搜索空间的独立收益 |
| **A-serial** | P→锁→S | 先按 **default-schedule** 估计贪心选宽度并**单点锁定**, 最后才 tune schedule | vs joint = 先锁一侧 vs 同搜(**核心**) |

预期 A-joint⪰A-noS⪰A-serial; 但**论点 = A-serial 的可达解集结构性排除 P_g**(锁 W_g 后永不回头改宽度 → P_g 整条分支被切掉), 非"快多少"。A-serial 主报**单点锁定**(最强局部最优), 辅报 Top-k(放松仍不及联合)。

---

## §3 执行计划 B1–B5

### B1 — 延迟 LUT (CHaNAS 加性分块, 纯 H800 TVM, 无 AP 依赖)
**目标**: 让搜索能 O(1) 查任意宽度的 (default-lat, tuned-lat), 不每点真 tune。
**做法**: `Lat(s0,s1,s2, sched) ≈ Σ_k f_k(width_k, sched)`。单 stage 扫点标定 f_k: 固定其它 stage=base, 单 stage 取 {16,32,48,64}(s0)/{32,64,96,128}(s1)/{64,128,192,256}(s2)。已有 base/iso_s0/iso_s1/iso_s2 = 每 stage 1 个 off-base 点; **补 ~9 个单 stage 变体**(ONNX 用 `tools/export_backbone_onnx_new_widths.py` 改 num_filters 即出, 随机权重即可——latency 不需训练)。
**验证加性(必做)**: 另测 ~5 个全组合宽度真值, 比 Σf_k, 报误差; 加性不成立(fusion/跨stage耦合)就退到"搜索访问到的组合直接真测"(慢但真)。
**产出**: `results/latency_lut_pyramid.json`(per-stage f_k default+tuned + 加性误差) + 查询函数。
**成本**: ~9+5 = 14 个 tuned 点 × ~16min ≈ 3.7h H800。**用 proven 逐宽度独立进程 harness `s2_2e_e2e.py`(见 §7 纪律), 切勿单进程循环。**

### B2 — AP 处理 (GPU finetune, 长项; 策略待用户定, 见末)
**目标**: 给搜索的每个宽度一个 AP70(多目标的 AP 轴)。
**已有真 AP70 锚点**(stage_a, DAIR val, finetuned): base[64,128,256]=0.631 / p25=trap25[48,96,192]=0.590 / p50[32,64,128]=0.564 / p75[16,32,64]=0.530。pad64 AP70=0.590(=trap25, 零填充不改权重)。
**默认策略(用户暂未拍板, 推荐"模型引导+只真测赢家")**: 用锚点 + finetune ~3–4 个 iso 宽度拟合**单调 AP 模型**(stage_a 已证 AP 由 backbone 通道主导、单调)引导搜索; **各臂最终收敛解才真 finetune 确认 headline**(B5)。⇒ 搜索是真过程, 结论在收敛点真测。
**产出**: AP 模型 + 校准/验证集真 AP70。
**成本**: ~6–8 finetune(每个数小时, DAIR 1789 val)。**注**: finetune recipe 见 `scripts/phase2/stage_b_finetune_ap.py` / `ap_cliff*_finetune.py`; 剪枝 ckpt 走 DepGraph `tools/configurable/depgraph_pyramid.py`; ⚠️ ckpt flat state_dict 陷阱见 CLAUDE.md 〇.5。

### B3 — 搜索内核 (纯 Python 工程, 无 GPU 依赖)
**目标**: 在 `framework/searcher_v0.py` 上建真三臂搜索器。
**加**: ① schedule 轴(每宽度 default/tuned, 查 B1 LUT); ② 多目标评估器 (AP70 查 B2 模型, latency 查 B1 LUT, +energy 可选); ③ NSGA-II(可改 `nsga2_pareto_search_v4.py` 或新写); ④ **三种调度模式** A-joint/A-noS/A-serial 共用同一 explorer 内核+同 cost model+同总评估预算, 只切"消哪轴/是否分阶段锁定"; ⑤ 过程产物记录(每代评估点、当前最优 HV)。
**纪律(doc4 §3)**: 全局对搜索器隐藏(只给 cost model + 有限真测预算, 不喂全局 Pareto); reference Pareto 仅用于算 HV、不进任何一臂。
**产出**: `framework/search_three_arm.py` + 单元 smoke(小空间跑通三臂)。
**成本**: ~半天工程。

### B4 — 跑三臂 (依赖 B1+B3, AP 用 B2 模型)
跑 A-joint/A-noS/A-serial × N seed × 多起点 × 多贪心顺序(P→S / S→P)。同预算同 cost model。产 §6 四层证据产物。

### B5 — 验证 (依赖 B4)
各臂收敛解**真测**(真 MetaSchedule tune + 真 finetune AP70)确认: A-serial 收敛含 W_g/被 P_g 支配, A-joint 收敛含 P_g。headline = iso-AP70 latency 倍率。

---

## §4 ★Agent 分工 + 并行 DAG
**B1 / B2 / B3 三条线无相互依赖, 可并行起 3 个 agent。** 主控负责核验(复跑/读文件/git diff, 不轻信自报)+ 串 B4/B5。

```
   ┌─ Agent-LUT (hw-optimizer): B1 延迟 LUT (H800 TVM) ──┐
   ├─ Agent-AP  (sw-optimizer): B2 AP finetune (GPU)  ──┤→ 主控 B4 跑三臂 → B5 真测验证
   └─ Agent-Kernel (general):   B3 搜索内核 (纯Python) ─┘
```

| Agent | 类型 | 任务 | 输入 | 产出 | 核验点 |
|---|---|---|---|---|---|
| **Agent-LUT** | hw-optimizer | B1: 补 ~9 单stage扫点+~5全组合, 建加性延迟LUT | `tools/export_backbone_onnx_new_widths.py`, `s2_2e_e2e.py`, H800 env | `results/latency_lut_pyramid.json`+查询函数 | 复跑1点对csv; 加性误差必须报 |
| **Agent-AP** | sw-optimizer | B2: finetune ~6-8宽度→真AP70, 拟合单调AP模型 | DepGraph剪枝, `stage_b_finetune_ap.py`, DAIR | AP模型+校准集真AP70 | 抽1宽度复eval; flat-ckpt陷阱 |
| **Agent-Kernel** | general-purpose | B3: searcher_v0加schedule轴+多目标评估器+NSGA-II+三模式 | `framework/{searcher_v0,nsga2_pareto_search_v4}.py`, doc4 | `framework/search_three_arm.py`+smoke | 读码; 小空间三臂smoke跑通 |

**注**: B2(finetune)是长项且最可能拖, 优先起; B1/B3 较快。H800 ssh **今晚多次持续断连**(各17–33min), 所有 H800 任务必须 **detached(setsid) + 增量写盘 + 断连重试**(见 §7)。

---

## §5 ★量化 (Q) 轴 — 持久待办 (用户明确要求"时刻把量化作为下一步, 不要忘")
**地位**: prune×schedule 跑完后的**下一条轴**, 目标 = prune×**quant**×schedule 三轴耦合的更强故事。**不是做不到, 是贵且需跨口径。**
**TVM INT8 现状(半截)**: ✅ TIR 层有 `MMA_i8i8i32_INTRIN`+LDMATRIX_i8(int8 图能 tensorize/调优); ❌ relax **无自动 INT8 量化 pass**(不会插QDQ/标定scale/产int8图)。
**纳入 Q 的三前置(必须先解决)**:
1. **拿到可调优 int8 图**: (a) 外部量化成 QDQ-ONNX(onnxruntime/TRT-ModelOpt)再 `from_onnx` —— 须验证 relax onnx 前端吃 QuantizeLinear/DequantizeLinear(neck 已踩过 shape bug, 有风险); 或 (b) BYOC-TRT(等于把INT8交回TRT, 跨口径); 或 (c) 手写量化TIR。
2. **真 INT8 AP**: simulated fake-quant 不可信(memory `project-dair-ap-axis-collapse`: FP32累加/注意力scale粗糙/12层未覆盖)→ 必须真 TRT INT8 标定+eval(4090 独立管线)。
3. **跨口径不混**: prune×schedule 全程 H800 TVM 相对延迟; INT8 延迟走 TRT/BYOC 口径, 不能直接进同一 Pareto。
**科学洞察(记下别丢)**: **INT8 会放大 W_g/P_g, 不改其本质** —— INT8 MMA 对通道对齐要求更硬(k%16), 失配宽度对 INT8 比 fp16 grouped conv 更不友好。所以 prune×schedule 是干净核心, Q 是放大器。
**落点**: 见 `design_wg_probe_int8_plan_v1.md §7`、doc4 §4.2/§8.2。Q 就绪后 = 三臂升级为 P×Q×S(A-joint=P×Q×S同搜 / A-serial=P→Q→S逐一锁定)。

---

## §6 证据四层 (doc4 §2) + 判读退路
1. **跨 seed/起点/贪心顺序**(N≥10) → HV 分布箱线图 + Wilcoxon 配对(证结构性非运气)。
2. **收敛曲线**(评估数 vs 最优HV) → A-serial **早plateau+封顶低** = "到不了"非"没搜够"(排除预算不够反驳)。
3. **评估点云**((W,S)投影到(AP70,lat)) → A-serial 点云**整体缺 P_g 区域**(直接可视化贪心切掉分支)。
4. **headline** = iso-AP70 latency 倍率 + 三目标 HV。
**判读铁律(防 p-hacking)**: A-joint≈A-serial **本身是结论**(协同价值条件化/可分离, 故事B), 不硬凑大赢; 区分"没搜够(曲线还涨)"vs"到不了(plateau)"。**载体对照**: Pyramid(grouped, 预期显耦合 A-joint≫A-serial) + CoDriving(标准conv, 预期可分离 A-joint≈A-serial)= model-dependent 判据数据点, 两个结果都有用。

---

## §7 资源 / 脚本 / 纪律
- **H800 TVM(主)**: `ssh -p 30001 jichengzhi@222.95.84.215`(密码 `12345678`, 代理7897; 每会话确认); env `/exdata/jichengzhi/tvm310/bin/python`; 跑前 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=0`。资产 `/exdata/jichengzhi/s2_tvm/models/{base,p50,p75,trap25,trap25_pad64,iso_s0,iso_s1,iso_s2}_backbone.onnx`。
- **★H800 纪律(今晚踩过)**: ssh **持续断连(17–33min)** + `nvidia-smi`/`pkill` 偶尔 hang → 所有长任务 **`setsid` detached + 写 logfile + 结果增量写盘 + 断连后台重试循环**; latency 必空闲卡(util≤2%/foreign mem≤50MiB, min/mean-of-N)。
- **★TVM 调优铁律(memory `feedback-tvm-tune-apply-fresh-workdir`)**: tune+apply **必须 fresh work dir + 逐宽度独立进程**; 单进程循环复用撞名残缺db会出**假 ratio≈1.0**(p75假报1.0真值26×); 坏kernel `CUDA illegal memory access` 会 abort 整进程→须进程隔离。可信单点脚本 = `scripts/phase2/s2_2e_e2e.py <onnx> <label> <trials> <csv> [reps] [seed]`。
- **4090(边缘/finetune)**: env `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`; HEAL `/home/jichengzhi/heal_research/HEAL`; **真仓库 `/home/jichengzhi/V2X`**(UniV2X 断裂符号链接)。ONNX 导出器 `tools/export_backbone_onnx_new_widths.py`(改 num_filters 出任意宽度 backbone 子图, 已验证)。
- **脚本**: `scripts/phase2/{s2_2e_e2e, gap1_grid_analyze, diag_iso, diag_retune}.py/sh`; `framework/{searcher_v0, nsga2_pareto_search_v4, constraints, propagation}.py`。
- **数据/结果**: `results/{gap1_grid_corrected.json, gap1_grid_retune_h800.csv, gap1_schedule_lut.json}` + `data/stage_a_ap_real.parquet`(AP70 锚点)。
- **纪律**: 不轻信 agent 自报(复跑/读文件/git diff 核验); backbone-only≠e2e(Amdahl); 不拼跨平台单一e2e; 负结果是资产(可分离=判据)。commit/push 只在用户要求时。

---

## §8 接手第一步(再强调)
1. 确认 H800 可达(`echo > /dev/tcp/222.95.84.215/30001`)+ env。
2. **并行起 3 agent**(§4): Agent-AP 优先(长项)、Agent-LUT、Agent-Kernel。每个给 §3 对应任务 + §7 纪律 + 核验要求。
3. 主控核验三线产出 → 跑 B4(三臂, doc4 §2 四层证据) → B5(收敛解真测)。
4. 跑完按 doc4 §5 判读规则定 framing; 回写 `4_design_ablation_proof_v1.md` 结果 + 更新本页。
5. **prune×schedule 完成后 → 启动 Q 轴(§5), 别忘。**

---

## §9 B1 执行进度 (2026-06-20 14:05 CST, Agent-LUT) ← 更新: 白天 blocker 修复 + 全网格运行中

### ★ 关键更新 (14:05 CST): 加性 LUT 改为离散真测网格

**team-lead 决策**: 加性模型不成立（tuned arm p50 err=672%, p75 err=4711%；default arm p50 err=38%, p75 err=140%）→ **放弃加性 LUT，改为离散真测网格**，B4 直接查表 O(1)，不做插值。

### 白天 Blocker 修复 ✅
- 根因确认: `LocalBuilder(timeout_sec=30)` 在 load=42-52 下 → 30s 内 NVCC 编译未完成 → 全超时 → ratio=1.000
- 修复: `LocalBuilder(timeout_sec=300)` ← 10× 扩展
- **实测验证**: 白天 load=42, 64 builds 在 87s 完成（13:48:36→14:02:22）→ "Sending 64 samples to runner" 出现 ✅
- 新脚本: `/exdata/jichengzhi/s2_tvm/s2_2e_bumped.py` (只改 builder timeout)

### 已完成
- ✅ **12 个新 ONNX 导出** (9 已有 + 3 新: mix_d=[48,128,128], mix_e=[64,96,128], mix_f=[32,64,64])
  所有在 `models/stage_a_cache/` 且已传 H800 `/exdata/jichengzhi/s2_tvm/models/`
- ✅ **全网格批次运行中**: H800 **PID 2076337**, 脚本 `run_lut_grid.sh`, 输出 `lut_results_grid.csv`
  - 首个模型 s0_16 已进入 Task #1 builder (14:03:11)，验证 bumped timeout 正常工作
  - 预计: 每模型 ~30min (load=42-52 下) → 12 模型 × 30min = **~6h → 完成于 ~20:00 CST**

### 完整离散网格 (20 点)
| # | label | num_filters | 状态 |
|---|---|---|---|
| 1-8 | base/p50/p75/trap25/pad64/iso_s0/iso_s1/iso_s2 | 见 gap1_grid_corrected.json | ✅ 已测 |
| 9  | s0_16 | [16,128,256] | 🔄 运行中 |
| 10 | s0_32 | [32,128,256] | 🔄 排队 |
| 11 | s1_32 | [64,32,256]  | 🔄 排队 |
| 12 | s1_64 | [64,64,256]  | 🔄 排队 |
| 13 | s2_64 | [64,128,64]  | 🔄 排队 |
| 14 | s2_128| [64,128,128] | 🔄 排队 |
| 15 | mix_a | [32,96,192]  | 🔄 排队 (s0 aligned, s1+s2 misaligned) |
| 16 | mix_b | [48,64,256]  | 🔄 排队 (s0 misaligned, s1+s2 aligned) |
| 17 | mix_c | [16,128,128] | 🔄 排队 |
| 18 | mix_d | [48,128,128] | 🔄 排队 (s0 misaligned only, s2 small) |
| 19 | mix_e | [64,96,128]  | 🔄 排队 (s1 misaligned only) |
| 20 | mix_f | [32,64,64]   | 🔄 排队 (all aligned, all small) |

### 接手步骤 (预计 ~20:00 CST 完成)
```bash
# 1. 检查结果
sshpass -p '12345678' ssh -p 30001 jichengzhi@222.95.84.215 \
  'tail -10 /exdata/jichengzhi/s2_tvm/lut_grid.log && cat /exdata/jichengzhi/s2_tvm/lut_results_grid.csv'

# 2. scp
sshpass -p '12345678' scp -P 30001 \
  jichengzhi@222.95.84.215:/exdata/jichengzhi/s2_tvm/lut_results_grid.csv \
  /home/jichengzhi/V2X/results/lut_results_grid.csv

# 3. 核验: 每行 e2e_ratio 必须 > 1.05 (非 1.0); 含对齐/非对齐各组
python3 -c "
import csv
rows = list(csv.DictReader(open('results/lut_results_grid.csv')))
print(f'Rows: {len(rows)}')
for r in rows:
    flag = '✅' if float(r['e2e_ratio']) > 1.05 else '❌'
    print(f'{flag} {r[\"label\"]:10s} def={float(r[\"default_us\"]):.0f}µs tun={float(r[\"tuned_us\"]):.0f}µs ratio={r[\"e2e_ratio\"]}')
"

# 4. 合并 gap1 + grid → 完整 20 点表 (B4 可用)
python scripts/phase2/merge_discrete_grid.py \
  --gap1 results/gap1_grid_corrected.json \
  --grid_csv results/lut_results_grid.csv \
  --out results/discrete_lut_v1.json

# 5. 通知 team-lead: B1 完成，B4 可跑
```

### 如网格部分模型 ratio=1.0 (builds failed despite bumped timeout)
说明 load 进一步飙高 (>80)。对策:
- 剩余失败模型: 在 load < 30 时 (深夜/凌晨) 补跑
- 用低并行 s2_2e_lowpar.py (max_workers=8, timeout=300) 单跑失败项
