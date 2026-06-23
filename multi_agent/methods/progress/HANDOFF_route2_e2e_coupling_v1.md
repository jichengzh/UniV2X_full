# HANDOFF — route2 / 端到端提速 + HW×SW 耦合证明 (v1, 2026-06-18)

> ★ 跨模型整合视角(Pyramid+CoDriving 对照 + 统一结论)见 [HANDOFF_tvm_codesign_crossmodel_v1.md](HANDOFF_tvm_codesign_crossmodel_v1.md); 本页是 Pyramid/HEAL 单模型执行细节。
> **接手先读**: 本页(自足) + [dims_hardware_v4.md](../design/dims_hardware_v4.md)(维度定义 §2 / 集成判定 §5) + memory `project-hwsw-codesign-route2`。
> 历史细节(别重做): [HANDOFF_tvm_s2_2b_coupling_v1.md](HANDOFF_tvm_s2_2b_coupling_v1.md)(S2.0/2.1/2.2a/2.2b/2.2c 全过程 + 环境配方 + 已解工具坑) + [HANDOFF_tvm_migration_route2_v1.md](HANDOFF_tvm_migration_route2_v1.md)(TVM-on-H800 env 配方原始版)。

---

## §0 一句话现状 + 下一步

route2 = 用 TVM 把编译调度变成"可搜硬件轴"挣 co-design 标签。**S2.0/2.1/2.2a/2.2b/2.2c 已完成**(工具链通 + 数值对齐 + 旋钮逐个延迟消融)。

**用户 2026-06-18 指出两个核心缺口**:
1. **per-operator ≠ end-to-end**: 旋钮加速(D1 2.3× / L1 1.46× / L5 67×)全是**单算子微基准**, 未反映到端到端(Amdahl)。
2. **耦合未证明**: 只证"旋钮加速算子", 未证"**最优旋钮配置依赖软件决策(剪枝/量化)**"——这才是 co-design 耦合的本质。

**★2026-06-18 进展 (E-couple-A/B 已完成, 见 §6)**: 用户腾出 H800 GPU0(空闲实测), 跑完 tile×剪枝宽度 32点+12点复测(`results/s2_2d_couple_tile.csv` / `s2_2d_rep.csv`)。**结论: 耦合真实但中等** —— argmin(tile|宽度) 确实随剪枝宽度漂移(16×16×16 在 W=48/64 最优、在 W=128 变最差, +21% 干净 rank-reversal, 3 次复测稳定无重叠); 但存在宽度无关折中 tile(16×16×8)全宽度 ≤7.3% 损失。⇒ co-design 在 tiling 旋钮上有**可量化但有界**的价值: 比朴素默认省 21%, 比聪明固定 tile 仅省 ~7%。**这驳倒了"TVM 把所有耦合都消解→完全可分离→NO-GO"的最坏情形**(至少 tiling/reduction-depth 旋钮上耦合存在)。
**E-e2e 已完成(答缺口#1)**: backbone 子图 tuned/default = **base 10.34× / p50 8.74×** —— per-op 旋钮**确实**折算到端到端(~10×, 非被 Amdahl 抹掉), 因 backbone=50conv 旋钮全覆盖。**关键 caveat**: 这是 conv backbone 子图 e2e, 非全 pipeline(VFE/scatter/decode/NMS 非 TVM 调度, 会按 Amdahl 稀释); 且 DEFAULT=dlight"旋钮关"基线非 vs TRT。详见 §6。

---

## §1 已确立的事实 (别重做)

**TRT 侧(S0b, `results/S0b_coupling_clean_4090.csv`, 4090)**: 固定硅片 D 维几乎全塌缩成 TRT-auto; 唯一稳的耦合 = 通道对齐×INT8(trap25 48_96_192 非÷32 → INT8 仅 1.08× vs p50 1.32×)。

**TVM 工具链(S2.0/2.1, H800)**: relax 导入 Pyramid backbone + 数值对齐(maxdiff 2.4e-6) + MetaSchedule 全链路通 + INT8 tensorize 可表达。

**S2.2b 耦合发现(H800, 61 tune 点, 详见 s2_2b handoff §2.5/2.5b)**:
- D1 全宽度无悬崖(TVM-WMMA-K16 把 K 补到 ×16, 48 不掉 TC)——**TVM 消解了 TRT 的 ÷32 对齐悬崖**。
- 真实 trap conv 16/16 全留 WMMA(mitigation CONFIRMED, 定性)。
- Q/DQ 被 fuse 进 conv epilogue——**TVM 也消解了 TRT 的 P4 融合断裂耦合**。
- grouped g=32 全掉 SCALAR(候选耦合#1, 但 TRT 同样不行 = 非 TVM 专属)。
- **★诚实综合**: 两个 TRT 耦合在 TVM 都被"消解"非"复现" ⇒ route2 价值偏向"去耦合器"(§5判据2)而非"揭示新耦合"(§5判据1弱)。

**S2.2c 旋钮逐个延迟消融(H800, 详见 s2_2b handoff §2.5c)**:
- 方法学: rule-drop 消融被搜索预算混淆(弃); **确定性手写 schedule 消融**是正解。
- T1 旋钮效应(4/6 干净): D1 tensorize **2.3×** / L1 shared 暂存 **1.46×** / L2 线程绑定=GPU 必需 / L5 跨线程 reduction **67×**(reduction-heavy GEMV)。L3 storage_align / L4 software_pipeline = 参数依赖, 手设未显效。
- 🔑 **复用工具**: 手写 schedule 编译前必打 `func.with_attr("tirx.is_scheduled", True)` + `mod.update_func` 否则 `tvm.compile` 的 dlight 重排报错; shared 暂存必配 cooperative fetch(shared-load 循环 fuse+split+bind threadIdx)否则 24× 假慢。

**⚠️ 关键缺口(本阶段要补)**: 以上全是"HW 旋钮在固定软件下的价值", **从未测端到端, 也从未变软件轴看最优 HW 配置是否随之改变**。

---

## §2 环境 / 路径 / 命令速查

- **H800**: `ssh -p 30001 jichengzhi@222.95.84.215`(sshpass 密码私存; 非交互每命令显式 `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897`)。GPU6/7 常空闲; 跑前 `nvidia-smi` 确认 idle(mem≤50MiB)。
- **TVM env**: `/exdata/jichengzhi/tvm310/bin/python`(TVM 0.20.dev1070 改版 Unity: `tvm.tir`→`tvm.s_tir`/`tvm.tirx`; MetaSchedule 在 `tvm.s_tir.meta_schedule`; `get_block`→`get_sblock`)。跑前必 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=<idle>`。
- **素材(H800 `/exdata/jichengzhi/s2_tvm/`)**: `models/stage_a_cache/{p50,trap25,base}_backbone.onnx`(S2.1 PASS); 脚本 `s2_2{b_screen,b_mm_k32,b_d4_fusion,c_knob_ablation,c_manual_gemm}.py`。
- **本地 repo**: 脚本 `scripts/phase2/s2_2*.py`; 数据 `results/s2_2*.csv` + `results/H800_*.log`。
- **关键 API**: target=`tvm.target.Target.from_device(tvm.cuda(0))`; tune 前必 `Sequential([LegalizeOps,AnnotateTIROpPattern,FuseOps,FuseTIR])`; tune=`from tvm.s_tir.meta_schedule import relax_integration as ri; ri.tune_relax(...,space=,seed=)`; 调度后查=`relax.transform.MetaScheduleApplyDatabase(work_dir=)` + `mod.script()`; 手写 schedule 编译=打 `tirx.is_scheduled` + `tvm.compile(mod,target)` + `relax.VirtualMachine` + `time_evaluator`; 延迟首选 DB `run_secs`(tune 时)或 time_evaluator(手写时)。

---

## §3 本阶段实验设计

### E-couple — HW×SW 耦合(★route2 GO/NO-GO 决定性, 先做)

> 命题: **最优 HW 旋钮配置是否依赖软件决策(剪枝宽度/量化档)?** 是 ⇒ 必须联合搜(co-design 有价值, GO); 否(可分离: 先定 SW 再独立调 HW) ⇒ route2 无价值(NO-GO, 退 route1)。

- **E-couple-A(逐旋钮交互扫描, 主推, 复用 `s2_2c_manual_gemm.py`)**: 对每个旋钮扫 **旋钮设置 × 软件配置** 网格, 看 **argmin(旋钮|SW) 是否随 SW 漂移**。
  - D3 tiling: tile T∈{16,32,64} × 剪枝宽度(=GEMM/conv 维)∈{32,48,64,128} → best-T 随宽度变?
  - D1 intrinsic: K16/K32/scalar × 精度(int8/fp16) × 宽度。
  - L1 staging 深度 × 宽度。
  - **判据**: argmin 随 SW 漂移 ⇒ 该旋钮与 SW 耦合。
- **E-couple-B(transfer-penalty 矩阵, 整体加固)**: 每个 SW 配置 tune 出最优 schedule, **交叉应用**(A 的最优套到 B 的 workload)实测 → N×N 矩阵。对角 ≪ 非对角 ⇒ HW 最优依赖 SW ⇒ 耦合。工程注意: 不同宽度维度不同, schedule trace 维度相关不能直接套 ⇒ 用参数化结构(tile 倍数/staging 深度)实例化到两 shape, 或等价用 E-couple-A 的 argmin 漂移。
- **最锐框架**: **剪枝宽度的延迟排序是否随 HW tuning/backend 翻转?**(S0b 就是一例: trap25 在 TRT-INT8 灾难、在 TVM-WMMA 正常 → SW 排序依赖 HW backend)。排序 HW-不变 = 可分离。

### E-e2e — 端到端旋钮价值 + Amdahl 归因

- 目标函数 = **PyramidFusion backbone 子图全图**(`models/stage_a_cache/{p50,trap25}_backbone.onnx`, fp16; 全图 INT8 受 relax 无量化 pass 限制)。
- 三臂 idle-GPU 实测 e2e: ① TVM dlight-default(旋钮最小) ② TVM MetaSchedule-tuned(全旋钮, budget≥1000) ③ 参考 TRT fp16(S0b 有 backbone 数)。
- **Amdahl 归因**: profile tuned backbone 各 op 延迟占比, 指明主导 op + 旋钮在主导 op 上的实际加速 → 折算 e2e 净加速。
- **判据**: e2e tuned/default 比 = 旋钮栈真 e2e 价值(诚实预期 ≪ per-op 极值; 可能追不上 TRT 绝对值——价值在机理/耦合非 SOTA)。

### 执行顺序
E-couple-A 先(最便宜、最直接、复用现成脚本)→ 见耦合则 E-couple-B + E-e2e 加固 → 喂 dims_hardware_v4 §5 GO/NO-GO 终判。

---

## §4 诚实风险 + 纪律

- **★最大风险**: S2.2b 已显示 TVM 倾向**消解**耦合(WMMA-K16 让对齐无关、Q/DQ fuse 进 epilogue)。⇒ E-couple 很可能测出 **transfer penalty 小 / argmin 不随 SW 漂移 = 可分离 = route2 NO-GO**。这是用户质疑要求的诚实测试: 若可分离, 按纪律退 route1(重锚 Orin 异构 GPU∥DLA 真并行 1.34×), 把"调度可分离于剪枝/量化"作为负证据写进论文。**无论哪个结果都正面回答用户两个问题。**
- 纪律: ① 延迟必空闲 GPU(util≤2%/foreign≤50MiB)实测; 调度产物静态分析对噪声免疫。② TVM 绝对延迟常追不上 TRT, 价值在发现/量化/缓解耦合非刷 SOTA。③ 任何"耦合/可分离"结论必跨配置(seed/trial)验稳(防单配置假象, 历史 S0/S2.2b 已犯)。④ supervisor 对每个"已测/已 build/加速/耦合"复跑+读文件+数值 diff 核验。⑤ 区分 真测/估算/仅声明。⑥ H800 是 Hopper 数据中心卡 ≠ 边缘标的; 边缘绝对延迟最终须 4090/Orin 实测, H800 上只坐实相对/机理结论。

---

## §5 接手第一步

1. 读本页 + dims_hardware_v4 §2/§5 + memory + (需深细节时) s2_2b handoff §2.5c。
2. 确认 H800 env 未腐(nvlibs 路径 + nvcc PATH); GPU6/7 idle。
3. **起 E-couple-A**: 改/调 `s2_2c_manual_gemm.py` 暴露 tile 因子(TM/TN/BK)为参数, 对 **tile × GEMM 维度(=剪枝宽度代理)** 网格实测延迟, 出 best-tile-per-width, 看 argmin 是否随宽度漂移。**先回答: 最优 tile/staging/intrinsic 是否随剪枝宽度/精度改变。**
4. 见耦合 → E-couple-B(transfer 矩阵)+ E-e2e(全图三臂)加固; 不见 → 诚实记"可分离", 触发 §5 NO-GO 讨论。
5. 全程记负结果(可分离的旋钮也写表, 喂 §5)。

---

## §6 结果 (2026-06-18, H800 GPU0 空闲实测)

### E-couple-A — tile × 剪枝宽度 argmin 漂移 (DONE)
脚本 `scripts/phase2/s2_2d_couple_tile.py` + 驱动 `run_s2_2d_couple.sh`。fp32 方阵 GEMM 代理(M=256 固定, K=N=W=剪枝宽度), 手写 level-1 schedule(bind+shared+cooperative-fetch+`tirx.is_scheduled`), tile=(TM,TN,BK)。网格 4 宽度{32,48,64,128}×8 tile=32 点(`results/s2_2d_couple_tile.csv`)+ top4 tile×3 复测(`results/s2_2d_rep.csv`)。

**复测 argmin(tile|W)**: W32→8×8×8(2.567µs) / W48→16×16×16(2.617) / W64→16×16×16(2.750) / W128→16×16×8(3.745)。
- **干净 rank-reversal**: 16×16×16 在 W=48/64 **最优**, 在 W=128 **最差**(4.532 vs 3.745=+21%, 3 次复测 4.435/4.559/4.603 全高无重叠 → 非噪声)。
- **结构耦合**: W=48(非2幂宽度) **禁用整个 32 粒度 tile 族**(BK=32 / TN=32 / 32×32 全 indivisible) —— trap25 错位的静态形态。
- 32×32 大 thread tile 惩罚随宽度增长: W32 1.34× → W128 1.95×。

### E-couple 完整逐旋钮扫描 (7 旋钮 × 软件轴, DONE — 用户要求"每个旋钮都扫")
脚本 `scripts/phase2/s2_2f_gemm_knob.py`(L1/L2/L3/L4, `results/s2_2f_rep.csv` min-of-3) + `s2_2g_reduce.py`(L5, `results/s2_2g_reduce.csv`) + s2_2d(D3) + S2.2b screen(D1)。每旋钮只变该旋钮、扫软件轴(剪枝宽度 W 或输出 Cout), 看 **argmin(旋钮|软件) 是否漂移**。

| 旋钮(维度) | 旋钮轴 | 软件轴 | argmin 随软件漂移? | 效应量 | 耦合判定 | 证据 |
|---|---|---|---|---|---|---|
| **D1 tensorize(intrinsic)** | WMMA/scalar | 宽度×精度; 结构 | 宽度: **否**(WMMA 全 24–128, int8+fp16 都上); 结构: **是**(grouped g32→SCALAR) | 类别型(TC 开/关) | **宽度可分离; 与层结构耦合**(grouped 无 TC, 且 TRT 同样→非 TVM 专属) | s2_2b_screenA_{int8,fp16}, grouped.csv |
| **D3 tiling(BK 归约 tile)** | TM/TN/BK | 宽度 | **是**(16×16×16 在 W48/64 最优→W128 最差) | **+21%** 最差错配 (折中 tile 7.3%) | **中等干净耦合** | s2_2d_rep(3×空闲) |
| **L1 内存暂存(staging)** | global/shared | 宽度 | 是(弱)(小 W global 略优→W128 shared 优) | ~+13% @W128 | **弱耦合**(shared 仅大宽度才回本) | s2_2f_rep |
| **L2 线程 tile** | 8/16/32 | 宽度 | 部分(16 稳健; 32 在 W48 不可行/W128 +48%) | 最大 +48%(32@128) | **弱耦合**(可行性+大 tile 惩罚随宽度) | s2_2f_rep, s2_2d |
| **L3 bank-align** | off/4/8 | 宽度 | **否**(无一致模式) | ≤15%, 噪声地板 | **可分离/不可判**(shared tile 维由 BK/TM 定非 W, 无随宽度 bank 冲突; 需 bank-conflict-heavy workload) | s2_2f_rep |
| **L4 软件流水** | off/2-stage | 宽度 | **否**(W32/128 帮、W48 不帮, 无单调) | 9–25%, 无规律 | **不可判**(浅 K scaffold; 需 deep-K) | s2_2f_rep |
| **L5 跨线程 reduction** | serial/cross | Cout(输出宽度) | **是(强)**(cross@Cout≤32 ↔ serial@Cout≥64) | 4.6×@8→翻转→11.6×@512 | **强干净耦合** | s2_2g(空闲 GPU7) |

**★逐旋钮综合(诚实, 真实验依据)**: 旋钮**不等价耦合**。**强/干净耦合 = L5 reduction(剪 Cout 翻转最优归约策略)+ D3 tiling(21% rank-reversal)**; **结构型 = D1**(与 grouped/精度可行性耦合, 非宽度, 且 TRT 共享); **弱耦合 = L1/L2**(~10–48% 但有界); **不可判/可分离 = L3/L4**(效应 ≤15–25% 在小 W GEMM 噪声地板, 需专门 workload 才显)。⇒ co-design 价值**集中在 归约策略 + tiling + intrinsic↔结构**, 非均匀分布于所有旋钮。**caveat**: L1–L4 是亚 5µs fp32 GEMM, H800 共享集群偶发竞争把效应推到噪声地板(min-of-3 兜); L5/D3/D1 效应大(4–11×/类别型)噪声免疫。

### E-couple-B — transfer-penalty 矩阵 (DONE, 由 A 数据直算)
T[src→dst]=lat(best_tile(src)@dst)/lat(best@dst):
```
          dst32  dst48  dst64  dst128
src32  8x8x8   1.000  1.054  1.086  1.194
src48  16x16x16 1.000 1.000  1.000  1.210
src64  16x16x16 1.000 1.000  1.000  1.210
src128 16x16x8  1.073  1.046  1.048  1.000
```
**worst off-diag = 1.210 (21%)**(窄宽度最优 tile 套到 W=128)。但**宽度无关折中 tile 16×16×8 全宽度 worst-case 仅 7.3%**(8×8×8=19.4% / 16×16×16=21.0%)。

⇒ **判定: 中等耦合, GO-leaning 但有界**。HW tile 最优**确实**依赖剪枝宽度(argmin 漂移真实), 但聪明固定 tile 能吃掉大部分收益。co-design 价值 = 比朴素默认 21%, 比聪明固定 ~7%。**诚实 caveat**: fp32 合成方阵代理, 非真 int8/fp16 Pyramid conv shape; 端到端折算看 E-e2e。

### E-e2e — 端到端旋钮价值 (DONE)
脚本 `scripts/phase2/s2_2e_e2e.py` + `run_2e.sh`, `results/s2_2e_e2e.csv`。base/p50 backbone 各跑 DEFAULT(dlight 默认 GPU lowering=旋钮关) vs TUNED(MetaSchedule budget1000=全旋钮搜) e2e 实测(GPU0 空闲, min≈mean 干净)。

| 配置 | DEFAULT(dlight) | TUNED(MS-1000) | **e2e default/tuned** | tune_s |
| base(64_128_256) | 64.17ms | 6.21ms | **10.34×** | 1202 |
| p50(32_64_128) | 26.23ms | 3.00ms | **8.74×** | 1145 |

- **★回答缺口#1: per-op 旋钮 DOES 折算到端到端**(8.7–10.3×, 非被 Amdahl 抹掉)。机理: backbone=50 conv 主导, tiling/tensorize/staging 旋钮**作用于全部 conv**而非单个可忽略 op ⇒ 广泛适用的旋钮复合成 ~10×(per-op 67× 那个是 GEMV 离群 op, 但 D1 tensorize 2.3×+tiling 等广覆盖旋钮才是 e2e 主力)。
- **★★关键 scope caveat(诚实, 即用户 Amdahl 担心的真正落点)**: 此 e2e = **conv backbone 子图**, **非全 pipeline**。全 pipeline 含 VFE/scatter/decode/NMS 等**非 conv、TVM 不调度**的阶段(上一会话已证这些是瓶颈), 它们会按 Amdahl 稀释这 10×。⇒ **"backbone 内旋钮折 10×" ≠ "全 pipeline 提速 10×"**; backbone 占 pipeline 比例决定真实端到端净收益。下一步真 e2e 须接全 pipeline 或用 backbone 占比折算。
- **caveat-2**: DEFAULT=dlight 是"旋钮关"基线(generic 未调度), 10× = 开启搜索的价值, **非 vs 强手写/TRT 基线**。绝对竞争力(vs TRT fp16)还缺 H800 同卡参考(S0b TRT 数在 4090, 跨卡不可直比)。
- **caveat-3**: fp32(TVM relax 无量化 pass); INT8 e2e 待量化 pass 或 BYOC-TRT。
- **未做(可选精修)**: op-level Amdahl 归因(profile tuned backbone 各 fused-conv 延迟占比, 指明主导 op + 其旋钮贡献)。DB 已存 `/exdata/jichengzhi/s2_tvm/ms_work_2e_{base,p50}`, 可从 `db.get_all_tuning_records()` per-task run_secs 直读。

### (a) 全 pipeline Amdahl 折算 (DONE, 真实测数据直算, 无需新跑)
TVM 只能导 conv backbone 子图(neck attention 导入 blocked, VFE sparse), 全 pipeline 不能在 TVM 跑 ⇒ 真端到端 = **Amdahl 折算**: 用真测 per-stage 时延(`results/E8_orin_e2e_fullchain.csv`, D1_B2_FP32_v2, Orin CUDA-Event PyTorch FP32, 含 NMS)拿 backbone 占比, 折 TVM 10.34×。

| stage (B=2 collab) | ms | % |
| voxelize | 10.0 | 3.9 |
| encoder(VFE) | 33.5 | 12.9 |
| **backbone(TVM调)** | **35.5** | **13.6** |
| **fusion neck** | **142.7** | **54.9** |
| head | 29.7 | 11.4 |
| NMS | 8.5 | 3.3 |

- **★(a)结论: backbone 10.34× 折成全 pipeline 净 = 1.14×**(B=1 为 1.12×)。backbone 仅占 14%, **attention fusion neck(55%)是真瓶颈且 TVM 未调(neck 导入 blocked)**。⇒ **用户 Amdahl 担心在全 pipeline 层面被证实**: 子图 10× 真实但端到端被 scope 卡到 ~1.14×。
- **scope 对比**: body-only(conv 链 backbone+shrink+heads, 排除 VFE/fusion/NMS)backbone 占 87% → 折 4.73×。⇒ **答案完全取决于 scope**: "backbone 内 10×" / "conv body 内 4.7×" / "全 collab pipeline 1.14×"。
- caveat: E8=Orin fp32, TVM 10×=H800 fp32, 跨卡; 但"fusion neck 主导/backbone 少数"的 Amdahl 天花板跨 HW 稳健。**真正 headroom = fusion neck 55%**(若 torch 前端导入 neck 后可调, 故事变)。

### RSU/车端两段拆分 + RSU 段加速可行性 (DONE, 用户 2026-06-18 拍板按 RSU/车端 两段汇报; 模型=Pyramid 协同检测无规划)
**物理分工**: RSU(路端)只做单体感知=voxelize+encoder(VFE)+backbone→输出特征传车端; **fusion neck 是车端专属**(把 RSU 特征 warp+attention 融合)。
- **RSU 段**(B=1 Orin 真测): voxelize 5.09 + encoder 17.49(VFE+scatter)+ backbone 18.46 = **41.04ms**。backbone 占 45%, encoder 占 42.6%。
- **车端段**(B=1): 自身感知 41.04 + **fusion 72.88(48%, 最大瓶颈)** + head 29.81 + nms 8.55 = 152.28ms。backbone 仅占 12%。Pyramid 无 planning(规划是 CoDriving/UniV2X)。

**★VFE 能否 TVM 加速 = 实测否定(`scripts/phase2/s2_2h_vfe.py`, `results/s2_2h_vfe.csv`, H800 GPU0)**: encoder_m1 结构=单层 Linear(10→64)+BN+ReLU+max-pool(点维) + point_pillar_scatter(数据依赖索引). VFE **稠密计算编译后 dlight 12.7–18µs / tuned 8.4–11.6µs(1.5×)** —— 比 E8 encoder 17.49ms **小 ~1000×**(跨卡即使 Orin 慢 10× 也仍 ≪)。⇒ **encoder 17.49ms 几乎全是 scatter(数据依赖, 导不进 TVM/ONNX, 同 grid_sample 难点)+ eager 启动开销**, VFE 矩阵计算可忽略。
- **⇒ RSU 段 TVM 能真加速的只有 backbone**(已 10× 子图); VFE 计算太小不值; scatter 需手写 CUDA(独立工程, 类似 NMS CUDA 化); eager 开销靠整段编译运行时(TRT/CUDA-Graph)消除。
- **RSU 段真加速路径 = backbone 编译(TVM/TRT 已验证)+ scatter 向量化 + 编译运行时替 eager**(后两者非 TVM 调度范畴)。

### scatter 优化实测 (DONE, 用户拍板 scatter CUDA 化; `scripts/phase2/scatter_microbench.py`, 4090)
point_pillar_scatter 的 eager 陷阱: `coords[:,0].max().item()`(强制 GPU→CPU 同步)+ python for-loop over batch + 每 batch `torch.zeros(64,140800)`(36MB)。**纯 torch 向量化重写**(单次 flat index_put `flat[b*HW+idx]=feats`, 无 loop 无 per-iter sync)**数值等价 maxdiff=0.000(无碰撞坐标, byte-identical)**。
| 4090 卡 | baseline | 向量化 | 加速 |
| GPU0 最干净 | 0.69ms | 0.38ms | 1.79× |
| GPU1 轻竞争 | 3.21ms | 0.22ms | 14.5× |
| GPU3 重竞争 | 10.3ms | 0.51ms | 20× |
- **★4090 结论**: 向量化稳定 0.2–0.5ms, baseline 随竞争暴涨; 4090 干净卡 ~2×。**无需手写 .cu**, 纯 torch 向量化即可。

### ★Orin 边缘卡实测 + encoder 完整拆解 (DONE, Orin 172.16.62.222, GR3D 0% 空闲, `scatter_microbench.py`+`vfe_forward_microbench.py`)
**两个反直觉实测发现**:
1. **向量化必须避免布局转置(边缘卡专属陷阱)**: 错误向量化(写 [B*HW,C] 再 permute+contiguous)在 Orin **慢 0.51×**(大转置在低带宽卡贵); **正确版直接 scatter 到 NCHW**(`spatial[b,:,col]=feats` 无转置)= **4.58×**(M=2000)。4090 高带宽看不到此陷阱, Orin 暴露。数值等价 maxdiff=0。
2. **cudnn-OFF 不是陷阱**: HEAL 的 `cudnn.enabled=False` 包 BN 在 Orin **反而快 3×**(cudnn-on=0.33×), 小 BN 上 cudnn kernel 开销不划算 —— 原作者是对的, 我先前"陷阱"假设被推翻(数值等价)。

**Orin scatter(NCHW 向量化)/ VFE-forward(eager) 随 M scaling**:
| M | scatter base | scatter nchw | 加速 | VFE-fwd eager |
| 2000 | 4.26ms | 0.93ms | 4.58× | 2.48ms |
| 15000 | 5.85ms | 2.42ms | 2.42× | **18.1ms** |
| 30000 | 7.74ms | 4.16ms | 1.86× | 37.4ms |

**★encoder=17.49ms(B=1)完整拆解(真实 M≈1万+)**: **VFE-forward eager ~12–18ms(大头!)+ scatter ~5ms**。
- **VFE-forward 的慢全是 eager**(6+ kernel launch + 2 次大 permute + BN, 随 M 暴增); **计算本身编译后仅 ~10µs**(s2_2h 已测)⇒ 编译化(TVM/TRT)可把 VFE-forward 从 ms 压到 µs(**>100×**)。
- scatter 向量化 Orin 1.86–4.58×(随 M 降, baseline 随 M 慢增 4→7.7ms)。
- **⇒ RSU 段 encoder 真加速路径(实测支撑)= ① VFE-forward 编译化(TVM/TRT, 收益最大 >100×, 因 eager 主导)+ ② scatter NCHW 向量化(2–4×, 无需 .cu)。两者都是消 eager, 非优化算法。** backbone(18.46ms)已 TVM 10×。voxelize(5ms)预处理不动。
- **下一步: 把 VFE 编译 + scatter 向量化接回 RSU 链, Orin 实测 encoder 17.49ms→压到多少 + RSU 段端到端 41ms→多少。**

### ★全流程 TVM 化各阶段 + 剪枝量化实际加速 (DONE, 用户要求; ⚠️跨平台/口径不可拼单一 e2e, 分阶段诚实给)
**RSU 段各阶段 TVM/编译化(代表性 M≈12000)**:
| 阶段 | baseline | 优化后 | 加速 | 工具/平台 |
| voxelize | 5.09ms(Orin) | 不动(预处理) | — | spconv |
| VFE 计算 | eager 18ms(Orin M15k)/ dlight 111µs(H800) | tuned **85µs**(H800) | vs Orin-eager **>200×**(全是 eager 开销; 编译消除) | TVM H800 |
| scatter | eager ~5ms(Orin) | 向量化 ~2ms(Orin) | **2–4×**(导不进 TVM, torch NCHW 向量化) | torch Orin |
| backbone(base) | dlight 64.2ms(H800) | tuned **6.21ms** | **10.34×** | TVM H800 |

**剪枝实际加速(TVM-tuned backbone, H800 同平台同工具, `results/s2_2e_e2e.csv`)**:
| 剪枝档 | planes | tuned 延迟 | vs base |
| base 未剪 | 64_128_256 | 6.21ms | 1.0× |
| p50 剪50% | 32_64_128 | 3.00ms | **2.07×** |
| trap25 失配 | 48_96_192 | **21.63ms** | **0.29×(更慢!)** |
⇒ 对齐剪枝(p50)真加速 2.07×; **失配 trap25 反而慢 3.5×**(欠剪+非÷32, dlight→tuned 仅 1.96× 救不回)= 对齐纪律活证据。

**量化实际加速(⚠️relax 无 INT8 pass → TRT 参考, 4090 collab2 口径, `results/S0b_coupling_clean_4090.csv`)**:
| 档 | fp16 | int8 | 量化加速 |
| dense | 1.26ms | 0.79ms | **1.60×** |
| p50 | 1.00ms | 0.76ms | **1.32×** |
| trap25 | 2.93ms | 2.71ms | **1.08×**(失配, INT8 救不回) |
⇒ 对齐量化 1.32–1.60×; 失配 trap25 仅 1.08×。

**★三者耦合(route2 核心论点再印证)**: trap25(非÷32 失配)在**剪枝(0.29×)+ 量化(1.08×)双输** ⇒ co-design 必须守 ÷32 对齐, 失配则剪枝量化都救不回(TVM/TRT 皆然)。这与 S0b 一致, TVM 侧再次确认。
**诚实边界**: ① VFE/backbone=H800 TVM, scatter=Orin torch, voxelize=Orin spconv —— 跨平台/口径**不拼单一 RSU e2e**; ② 量化=TRT(relax 无 INT8 pass)4090 collab2 口径, 非 TVM 非 Orin; ③ M 为代表性估算。
