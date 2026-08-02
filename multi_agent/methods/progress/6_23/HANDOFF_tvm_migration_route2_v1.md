# HANDOFF — Route-2 软硬件协同 / TVM 迁移与 smoke (v1, 2026-06-17)

> 接手先读这页 + memory `project-hwsw-codesign-route2.md`。本文件自足：含背景决策、已核验事实、S0/S0b 实测、两份耦合陷阱清单、S2(TVM) 计划与闸门、所有路径/命令/纪律、以及多 agent 分工建议。

---

## 0. 一句话现状

论文号称"软硬件协同优化"，但固定硅片(4090/Orin)上硬件维度几乎全塌缩成 TRT-auto。已**实测证据闭环确认**：TRT 上真正立得住、auto 救不回的耦合陷阱**只有"通道对齐×INT8"一个**，其余 auto-dominated 或负结果 ⇒ **TRT 被证明太薄**（= 纪律要求的"转 TVM 的负证据"）。**下一阶段 = S2：TVM 迁移可行性 smoke**，目标是把 schedule(tiling/layout/fusion/cache/tensorize) 变成可搜变量，发现 TRT 看不见的多个 prune/quant×schedule 耦合。

---

## ★ 进展更新 2026-06-17 PM（S2.1 PASS + S2.2 素材就绪；接手从这里续）

**平台改 H800**（用户拍板；4090×8 长期被占，H800 GPU6/7 空闲）。**S2.1 闸门 PASS**，**S2.2 backbone 素材已备**。MetaSchedule 调度搜索+耦合 demo = 下一步（可用 hw-optimizer agent）。

**① TVM-on-H800 env 配方（已固化，全部坑已解，勿重走）**：env=`${V2X_DATA_ROOT}/tvm310`（系统 py3.10 `venv --without-pip`+get-pip；H800 无可用 conda/缺 ensurepip）。装 **mlc-ai-nightly-cu124**（=TVM **0.20.dev1070 Unity，relay 已删只有 relax**；cu122 已下线；驱动 535=CUDA12.2）。**4 坑**：①cu124 runtime 缺符号 `cudaGraphAddDependencies_v2`→装 pip `nvidia-cuda-{runtime,nvrtc}-cu12==12.4.127`+cublas12.4.5.8+cudnn9.1.0.70，site-packages/nvidia/*/lib 加 LD_LIBRARY_PATH（路径存 `${V2X_DATA_ROOT}/tvm_nvlibs.path`）；②`apache-tvm-ffi` 默认 0.1.12 与 dev1070 不兼容(TypeAttr 重复注册)→**pin 0.1.11**(+装 pytest)；③CUDA codegen 用 nvcc→PATH 加 `/usr/local/cuda-12.2/bin`(nvcc12.2 匹配驱动535)；④API 变：device 用 `dev.exist`(非exists)、NDArray 用 `tvm.runtime.tensor(np,device=dev)`、TE 旧 `te.create_schedule` 已删、vm 输出是 tvm Array 容器(`list(out)` 拆)。SSH 非交互每命令显式 `export https_proxy=http://<PRIVATE_HOST>:7897`(用户端口转发, .bashrc 不读)。env 搭建脚本 `scripts/phase2/h800_tvm_{setup,fix,finalize}.sh`+`h800_ffi_sweep.sh`。

**② S2.1 闸门 PASS（backbone, 实测 `results/H800_s2_1_backbone.log`）**：relax 导入+build(nvcc编译cuda kernel,驱动535 minor-compat **实测确认**)+run+数值对齐 vs ORT，**maxdiff 2.4e-6**(<1e-3)。**三个耦合相关 backbone 全 PASS**：base/dense(64_128_256÷32) 2.4e-6 / **p50(32_64_128÷32) 5.2e-6** / **trap25(48_96_192✗÷32) 2.9e-6**。子图=纯 conv backbone(50conv/48relu/16add, `spatial_features(2,64,128,256)`→3层级relu输出)，**co-design 耦合(对齐×INT8/tiling/layout/tensorize)全在此**。通用对齐脚本 `scripts/phase2/s2_1_backbone_align.py <onnx>`(读onnx输入自适应)。子图 onnx：`models/stage_a_cache/{base,p50,trap25}_backbone.onnx`(已 scp 到 H800 `${V2X_DATA_ROOT}/s2_tvm/models/`)。

**③ relax onnx 前端吃下硬算子但 neck shape 推断有≥3 bug → neck 转 torch 前端(用户拍板作并行次要)**：grid_sample×6/einsum/softmax/where/isnan **全能导入**，但 attention neck 静态 shape 推断打地鼠：(a)einsum `nij,hwj->nhwi` 推错(agent维N泄漏进spatial)→已在 `tools/export_onnx_pyramid_collab_v2.py` 用 **bmm 替代** 修(`models/stage_a_cache/base_v2.onnx`,ORT rel0.35%); (b)又冒 grid_sample/where dim 256vs128; (c)onnxsim(`base_v2_sim.onnx`)未解。⇒ **onnx 前端修 neck 放弃**；robust=**relax torch 前端**(4090 装 TVM cu128+`torch.export(PyramidCollabSubnetN2)`+target=llvm CPU 数值对齐,不需GPU；4090 torch2.0.1 可能要升)。neck 非耦合所在，不阻塞 S2.2。

**④ 下一步 S2.2（backbone 子图, H800 GPU6/7）—— 接手从这做起**：MetaSchedule/Ansor 搜 layout+tile，复现+缓解 B1/B2 对齐机理。关键 demo=**对齐 p50 vs 失配 trap25 backbone**：证 TVM co-search layout/tile/precision 能看见 TRT 看不见的「剪枝宽度×对齐↔可量化性/tensor-core 占用」耦合，并能把 trap25 的对齐悬崖缓解(repack/pad) vs TRT-auto fallback。素材已 PASS 就绪。

**★④a S2.2 工具链 go/no-go = PASS（2026-06-17 PM 实测 `results/H800_s2_2_probe_p50.log`，脚本 `scripts/phase2/s2_2_probe.py`）。** 这个 mlc-ai-nightly **0.20.dev1070 是大改版 Unity，模块全被改名/挪位**（不是阉割版，别被 ImportError 误导）：
  - **`tvm.tir` 没了 → 拆成 `tvm.s_tir`（可调度 TIR + Schedule）+ `tvm.tirx`（IR 节点）**。手工调度 = `tvm.s_tir.Schedule`，含 split/reorder/**tensorize**/blockize/cache_read/cache_write/compute_at/**transform_layout**/**pad_einsum**（Roller 构造式 + ALT/FAST pad/layout 全齐）。
  - **MetaSchedule 挪到 `tvm.s_tir.meta_schedule`**（`from tvm import meta_schedule` 会 ImportError）。relax 自动调优入口 = `from tvm.s_tir.meta_schedule import relax_integration as ri`；`ri.tune_relax(mod,params,target,work_dir,max_trials_global,...)`（evolutionary+xgb，签名已验）+ `ri.compile_relax(db,mod,target,params)` + `ri.extract_tasks`。dlight 在 `tvm.s_tir.dlight`（`ApplyDefaultSchedule`/`gpu.Matmul`，免调优快速默认调度）。`relax.transform` 有 `MetaScheduleTuneIRMod/TuneTIR/ApplyDatabase`、`ConvertLayout`、`pad_einsum`。
  - **INT8 tensor-core tensorize 可表达（关键正面结论）**：`tvm.s_tir.tensor_intrin.cuda` 有 `LDMATRIX_i8_A/B_INTRIN` + **`MMA_i8i8i32_INTRIN`**（还有 fp16 `MMA_f16f16f32` / fp8 e4m3/e5m2）⇒ **对齐×INT8 demo（S0b 强绑定）在 TVM 可做，不必退回 fp16-only**。
  - **4 个 API 坑（全已解，脚本已固化）**：①target 字符串形式 `"cuda -arch=sm_90"` 0.20 已禁 → 用 `tvm.target.Target.from_device(tvm.cuda(0))`（自动填 arch+max_threads_per_block+shared mem，MetaSchedule auto-bind 必需；裸 dict 缺 max_threads_per_block 会崩）；②`tune_relax` 前必须先 legalize 把 relax-op 降成 TIR PrimFunc 否则 "No tasks to tune"：`Sequential([LegalizeOps, AnnotateTIROpPattern, FuseOps, FuseTIR])`（legalize 后 p50 backbone 抽出 **21 tasks**）；③cost model 需 `pip install xgboost`（已装 3.2.0 进 tvm310）；④fp32 子图 tune 出 `tensor_core_traces=0` 是**正常**（TC 只在 fp16/int8 触发），不是失败。
  - **go/no-go 实测链路全通**：21 tasks 抽取 → 32 trials 调优 → tune_relax/compile_relax COMPLETED → VM run n_out=3 正确。⇒ **S2.2 可全速推进，不再有工具链阻塞。**

**④b S2.2 真正第一步（接手做这个）**：① 把 backbone 转 fp16（`relax.transform.ToMixedPrecision`，待验该 transform 在本 build 名字）或直接走 INT8，tune 后确认 **aligned(p50) vs misaligned(trap25) 的 tensor_core_traces / MMA 命中差**（首个真耦合信号，要跨配置稳定，防 §3.1 单配置假象）；② Roller 构造式对照（手工 `s_tir.Schedule` 指定对齐 vs 失配 tile/layout + `MMA_i8i8i32` tensorize，确定性量 TC 占用/延迟差，不靠漫长 autotune）；③ ALT/FAST 缓解 demo（`ConvertLayout`+`pad_einsum` 把 48→64 救回 TC vs TRT-auto fallback）。纪律：延迟空闲 GPU 实测（H800 八卡当前全有外部负载，build/正确性/TC占用可带噪做，最终延迟数等干净窗口）；TVM 绝对延迟常追不上 TRT，价值在**发现/量化/缓解耦合**非刷 SOTA；supervisor 核验每个"已 build/对齐/加速/发现耦合"跨配置验稳定。MetaSchedule relax 集成 API 见上 ④a。

**⑤ S2.2 做法优化（2026-06-17 文献调研启发，详见 [references/survey_compiler_schedule_search_v1.md](../references/survey_compiler_schedule_search_v1.md)）**：调研 Ansor/CHaNAS/FAST/TLP/Roller/Hidet/ALT(均把 tiling/loop/layout/fusion 枚举成搜索点, 区别于我们委托 TRT-auto)后，S2.2 做法精化（不推翻，降本+强化论点）：
  - **[Roller OSDI'22 → 降本] 先做便宜的"构造式对照"，不必只靠漫长 Ansor autotune**：Roller 的 rTile=与硬件 MMA 单元对齐的 tile，秒级构造+微性能模型。我们 trap25(48_96_192✗÷32) 正是 rTile 失配实例。⇒ S2.2 第一步 = 在 TVM 里**显式指定对齐 vs 不对齐的 layout/tile + INT8 tensorize**，对 p50(对齐) vs trap25(失配) 直接量 tensor-core 占用/延迟差，确定性复现"对齐×可量化性"耦合（比 autotune 更便宜/更可控）；Ansor autotune 作补充上界。
  - **[ALT EuroSys'23 + FAST ASPLOS'22 → 缓解 demo 落地] "协同 vs 串行"demo 的具体动作 = layout 联合搜(ALT) + tensor padding 48→64(FAST)**：串行(prune-then-quant, trap25 失配→INT8 撞 MMA 墙→fallback) vs **协同(TVM co-search: pad/repack 48→对齐 + INT8 tensorize 救回 tensor-core)**，给协同后延迟 vs TRT-auto fallback 对比。**这是"硬件维度被协同地配置"的真 demo**——正面回应用户此前关键质疑"S0b 只证耦合存在、未证硬件被协同配置"。ALT 已发表 layout+loop 联合搜 1.5×>Ansor，背书"TRT 固定 layout=看不见对齐耦合"且指明 **layout 必须进搜索空间**(只用 Ansor 固定 layout 不够)。
  - **[CHaNAS TECS'22 → 论文定位] related-work 主线 = CHaNAS(NN×schedule co-search,1.6–1.9×)/ALT/Roller/Ansor**；我们**空白占位** = 剪枝×量化×schedule 联合 + 量化对齐耦合机理 + 边缘 V2X 多指标 Pareto（CHaNAS/FAST 不碰量化对齐, ALT/Roller 不碰剪枝/量化软件维, 我们接起两侧）。
  - **[TLP/Hidet → 次要]** TLP(调度原语即特征的代价模型)留 S3 预测器; Hidet(仅GPU无DLA)作GPU侧机理对照不进边缘异构主线。

---

## 1. 背景与已定路线（用户拍板）

- **危机**：方法是 software-hardware co-design，但固定 GPU 上 D 维塌缩成 TRT-auto，看不出协同。文献证实这是结构性事实（真协同搜索 Auto-NBA/NAAS/DiGamma/HASCO/FAST 全靠可重构 FPGA/ASIC）。
- **路线**：route1(重锚 Orin+谦逊措辞) 与 route2(补真硬件耦合搜索轴挣 co-design 标签) 都要，**先走 route2**。**纪律：转 route1 前必须用实验证明 route2 不可行，结论不能源于猜测/推断。**
- **定位护盾**：QuantV2X(arXiv:2509.03704) 全量化 V2X 协同感知自称 efficiency/deployment 不碰 co-design → 此角度不用 co-design 标签也能发顶会。taxonomy 引文：Sekanina IEEE Access 2021(三分法)、Sze et al. Proc.IEEE 2017。空白可占：引擎路由×量化×剪枝 5 指标 Pareto 固定车载 SoC 实测=无人做。

---

## 2. 已核验事实（primary-source，非推断）

1. **repo 后端 100% TensorRT**（65 文件，0 真 TVM；grep 命中的 `tvm` 全是 `import torchvision.models as tvm` 别名）。TRT 版本 **<PRIVATE_HOST>**。env=`${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`(py3.9)。
2. **TRT 是封闭 auto-tuner**（核验自 NVIDIA 头文件/文档）：只能在 TRT 自带 tactic 里**选择**(IAlgorithmSelector 白名单 / setTacticSources / builderOptimizationLevel 0-5 / 逐层精度 setPrecision+OBEY/PREFER / editable timing cache / SPARSE_WEIGHTS)，**无 API 定义 tile/loop-order/fusion-point/layout/cache/tensorize**；plugin 是黑盒。
3. **"最新 HW 配置研究都基于 TVM"是错的**：仅 auto-schedule 一脉是 TVM 中心(Ansor/Bolt/ALT/AMOS/MetaSchedule；独立例外 Hidet=无 TVM 但仅 GPU 无 DLA，Roller=nnfusion)。引擎映射 GPU↔DLA(JEDI/AxoNN/HaX-CoNN)=TRT-only(DLA 只能经 TRT)；量化部署多在 TRT(LiDAR-PTQ/QuantV2X)；2:4 稀疏=纯 NVIDIA 栈。
4. **route2 四轴里只有"生成式调度/tiling 搜索"才需离开 TRT→TVM**(或 Hidet 但无 DLA)；逐层 INT8 搜索/2:4/Orin 引擎映射今天就能在 TRT 做。⇒ "必须先迁 TVM"前提只对调度轴成立；且 TRT-auto 换 Ansor-auto 本身≠协同。

---

## 3. S0/S0b 实测结论（GPU 干净实测，已落 results/）

### 3.1 教训（务必记住，防重犯）
我曾 overclaim "INT8 层数随剪枝单调塌缩 55→28→5" = **错**，那是只比 opt5_ws1024 一个配置的假象。全网格显示**同一 p75 模型 INT8 层数随 opt/ws 从 5 到 54 剧变** → 逐层精度被 TRT-auto build 旋钮主导，**非干净剪枝律**(再次印证 CLAUDE.md 发现#3：TRT 逐层精度延迟驱动)。**任何"耦合/单调"结论必须跨 build 配置验稳定性，且延迟必须空闲 GPU 实测，否则是噪声。**

### 3.2 S0b 干净延迟定论（GPU7 空闲，fixed opt3/ws4096，body_subnet_collab2，`results/S0b_coupling_clean_4090.csv`）
| prune | planes | fp16ms | int8ms | INT8加速 | int8层 |
|---|---|---|---|---|---|
| dense | 64_128_256 ÷32 | 1.264 | 0.790 | **1.60×** | 60 |
| p50 | 32_64_128 ÷32 | 1.003 | 0.762 | **1.32×** | 54 |
| **trap25** | **48_96_192 ✗÷32** | **2.932** | **2.715** | **1.08×** | 12 |
| p75 | 16_32_64 | 0.772 | 0.649 | 1.19× | 5 |

- **唯一经得起推敲、build-knob-STABLE 的耦合 = 通道对齐**：失配的 trap25(非÷32) 在 opt0/3/5×ws **所有**配置下 INT8 可用层恒=12(硬上限，TRT-auto 抹不掉)；且 trap25(比 p50 多~2× 参数) fp16 比 p50 慢 **2.9×** → **欠剪但失配 远差于 激进但对齐**，延迟+可量化性双输。对齐模型(dense/p50)的 INT8 层数随 build 旋钮波动=TRT-auto 延迟选择，**非干净轴**。
- ⚠️ 用户正确指出：S0b 是**固定 HW 只变软件**，只证明了"耦合存在"，**未展示"硬件维度被协同地配置"**(co-design 闭环)。且对齐陷阱用户早先时延实验已证，S0b 只是复现。

### 3.3 资产（可复用，勿重建）
- 子模块 ONNX+calib：`models/stage_a_cache/{base,pruned25,pruned50,pruned75}.onnx` + `*_int8_calib.cache`(dense/p25/p50/p75)。
- S0/S0b engine：`models/s0_coupling_cache/`、`models/s0b_cache/`(8 engine @ opt3_ws4096)。
- 脚本：`scripts/phase2/s0_hwsw_coupling_probe.py`、`s0b_coupling_clean.py`(build/bench 两段)、`s0b_bench_watch.sh`(等干净 GPU retry)。复用了 `scripts/phase2/h2_workspace_scan.py` 的 build/bench/idle-gate/layer-count。
- **TRT 10.13 层精度解析器修复**：key=`Format/Datatype`∈{Float,Half,Int8}(旧解析器找 `DataType`/`float16` 全落 "other")；compute 层=CaskConvolution/GemmConvolution/DeconvolutionV2，按 input[0] datatype 分类。

---

## 4. 两份耦合陷阱清单（本阶段核心交付，已与用户对齐）

### List A — TRT 上能证明的协同维度（裁决：基本只剩 1 个）
| # | 耦合 SW×HW | TRT 能配？ | 状态/裁决 |
|---|---|---|---|
| **A1** | **剪枝宽度 × INT8 IMMA tile(÷32, +groups=32)** | 只能观察，不能修 | ✅ **唯一真陷阱**(已证)；是对联合搜索空间的硬约束 |
| A2 | 量化粒度 × DLA 路由(Orin) | 可配 | ⚠️ Pyramid 上 DLA INT8 0/12 失败；是约束非可调收益 |
| A3 | INT8 × workspace 预算 | 可配 | ⚠️ H2 测，仅~5%，本质 build auto-tune |
| A4 | 逐 stage 混精 × reformat 边界 | 可配 | ❌ perstage_quant 测：被 TRT-auto 严格支配 |
| A5 | 2:4 稀疏 × sparse-TC × 算术强度 | 可配 | ❌ H1 测：memory-bound→仅 1.02× 负结果 |
> 裁决：A1 是唯一 auto 救不回的真耦合，其余 auto-dominated/负结果 ⇒ **TRT 太薄**。

### List B — TVM 上能发现的耦合陷阱（schedule 可设定→与软件耦合；均为有机理依据的候选，未实测）
| # | TVM 旋钮 | 与剪枝/量化的耦合陷阱 | 为何 TRT 看不见 | 类型 |
|---|---|---|---|---|
| **B1** | layout(NCHW/NHWC/NC32HW32) | A1 对齐陷阱本质=layout 打包；TVM 可 co-search layout+repack/pad 把硬悬崖变可调 | TRT auto 选 layout，撞了只能 fallback | **发现+缓解** |
| **B2** | tensorization(MMA 原语映射) | 层能否上 tensor-core 取决于维度 fit MMA 形状(fp16 16³/int8 16²×32)；剪枝宽度×精度×原语三方耦合，TVM 可 pad-to-fit | TRT 不 fit 即 fallback，无 pad 选项 | **发现+缓解**(A1 机理级泛化) |
| **B3** | tiling/split | 最优 tile 随通道数(剪枝)和 MMA 形状(精度)变；小通道留 tensor-core 碎片 | TRT 内部定 tile | 剪枝宽度×tile 占用率 |
| **B4** | compute_at/cache_read/cache_write | backbone memory-bound(~30%BW, E_headroom)；剪枝更 memory-bound→缓存/下沉策略决定撞不撞带宽墙 | TRT 不暴露缓存放置 | **潜在最大**：剪枝移 roofline→最优 cache 调度变 |
| **B5** | fusion point × 量化 Q/DQ 放置 | Q/DQ 打断融合(问题4 残差融合被破)；最优融合分组取决于量化边界×剪枝后残差结构 | TRT auto-fuse 不可覆盖 | 量化边界×融合×访存流量 |
| B6 | reorder/unroll/vectorize | 剪枝小维度填不满 SIMD lane→浪费 | TRT auto | 剪枝宽度×向量化效率 |
| B7 | bind/parallel | 跨 SM 负载均衡取决于 per-stage 计算量(per-stage 剪枝) | TRT auto | per-stage 剪枝分布×并行映射 |
> 裁决：TVM 把调度变量化 → 至少 B1–B5 五个可研究耦合，B1/B2 还能把唯一的对齐陷阱从"硬墙"升级成"可缓解旋钮"。

---

## 5. S2 计划（TVM 迁移与 smoke）—— 闸门式，最低成本先验

> **不全量迁移。** 按 go/no-go 闸门推进；每过一关 supervisor 核验后再投入下一关。

- **S2.1 [闸门0·必须串行单点做] TVM build + 数值对齐**
  - 装 TVM(建议带 CUDA + cuDNN/CUTLASS BYOC)，把 `models/stage_a_cache/base.onnx`(Pyramid backbone 子模块) 经 Relay/relax 导入并 build。
  - **数值对齐**：TVM 输出 vs PyTorch/ONNXRuntime maxdiff < 1e-3(fp32)。
  - **风险**：Pyramid 含 deformable/grid_sample fusion/sparse VFE 自定义算子，Relay 可能不支持。先验**纯 backbone 子模块**(base.onnx 只有 conv/bottleneck，无 VFE/grid_sample)，大概率能过；完整 e2e 不必在 S2.1 强求。
  - **过不了这关 → TVM 路线对完整模型否决**，回报用户考虑 route1 或只在 backbone 子模块上做 TVM 研究。
- **S2.2 [闸门1] Ansor 搜调度 + 复现 B1/B2 对齐机理**
  - 取一个剪枝点(对齐 p50=32_64_128 vs 失配 trap25=48_96_192)，用 Ansor/MetaSchedule 搜 layout+tile。
  - **目标**：证明 TVM 能"看见 TRT 看不见的东西"——量化对齐耦合的 layout/tensorize 机理，且能 co-search layout 把 trap25 的悬崖**缓解**(repack/pad)，给出 TVM 缓解后 vs TRT fallback 的延迟对比。
  - 这是"协同配置 vs 串行"第一个真 demo：串行(prune-then-quant, 失配 48+INT8 撞墙) vs 协同(TVM co-search layout/tile/precision 把 48 救回或证明必须改宽度)。
- **S2.3+ [并行] B3–B5 耦合实验** + 搜索器接入(联合 剪枝×量化×schedule)。

### 关键纪律（违反=数据作废）
1. **延迟必须空闲 GPU 实测**(util≤2% / foreign mem≤50MiB)。4090×8 常被外部(wuyuegao)训练占满，干净窗口仅秒级 → 用 watcher+per-cell wait_until_idle(见 s0b_bench_watch.sh)。
2. **TVM 绝对延迟常追不上 TRT**(conv-dense)；TVM 价值=研究层面"发现并联合优化多耦合"，**不是刷 SOTA 延迟**。别把"TVM 更慢"当失败——重点是能否发现/量化/缓解 List B 的耦合。
3. **真测=真测**：区分 真测/估算/仅声明；标 latency 口径(sub-module/e2e/含 NMS)。
4. **不轻信 agent 自我报告**：凡"已 build/已对齐/已加速"，主控/supervisor 必复跑/读文件/git diff 核验(本项目历史多次抓到谎报)。

---

## 6. 多 agent 协作 / supervisor / 分工建议

**是否需要多 agent？分阶段：**
- **S2.1(build+数值对齐) = 不要 fan-out，单 agent 串行。** 这是高不确定性 go/no-go 闸门(自定义算子能否 build)，过不了全停，并行是浪费。主控或单个 hw 向 agent 专注做。
- **S2.2 起可适度并行**，S2.3 起明显受益于多 agent。

**建议分工（沿用项目已有 agent，新增 TVM 专长）：**
| 角色 | 职责 | 备注 |
|---|---|---|
| **hw-optimizer(扩 TVM 专长)** | TVM 环境/build/Ansor 调度搜索/数值对齐/延迟真测 | 现 hw-optimizer 是 TRT 向，需补 TVM；这是 S2 主力 |
| **sw-optimizer** | 提供剪枝/量化配置(对齐 vs 失配点、per-stage)，分析 schedule×prune/quant 耦合机理 | 拥有 DepGraph 剪枝×量化耦合知识 |
| **supervisor(必须接入)** | 核验每个"已 build/已对齐/已加速/发现耦合"：复跑+读文件+数值 diff+确认空闲 GPU+确认非噪声(防 §3.1 那种 overclaim) | **本工程必配**，鉴于谎报史 + 我自己犯过 overclaim |
| **doc-curator** | 把每关核验结论整合进 method/references(非追加)，控长度 | List A/B 应落进 `references/survey_hwsw_codesign_v1.md` 或新 method 节 |
| data-orchestrator | 若 TVM 数据要并入 dataset_v2 主表，定列/口径/溯源 | S2.3+ 才需要 |

**supervisor 检查清单(本工程专用)：**
1. TVM build "成功" → 实际跑一次 + 数值 maxdiff 核验，不信日志。
2. 任何延迟"加速" → 确认空闲 GPU(nvidia-smi)、确认 std 小、确认口径一致、确认 vs 正确 baseline(TRT? PyTorch? TVM-default?)。
3. 任何"发现耦合" → 跨 build/schedule 配置验稳定性(防 §3.1 单配置假象)；区分真耦合 vs auto/调度噪声。
4. TVM vs TRT 比较 → 同模型/同口径/同 GPU；不可拿 TVM 慢当"TVM 没用"，要看耦合发现价值。

---

## 7. 关键路径速查
- repo: `${V2X_ROOT}`(真仓库；UniV2X 是断裂符号链接空壳，勿用)。branch `hw-deploy-d-space`。
- env: `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`(TRT 10.13)。TVM 建议新建独立 env。
- Pyramid DAIR 金标准 ckpt: `${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/`。
- 子模块 ONNX/calib: `models/stage_a_cache/`。S0b 结果: `results/S0b_coupling_clean_4090.csv` + `results/S0b_layercount_4090.csv`(若 build 阶段写出)。
- 设计依据: `multi_agent/methods/design/{dims_hardware_v2,dims_quantization_v1,dims_pruning_v1}.md`、`background/00_研究目标与实验档案_v1.md`、`references/survey_hwsw_codesign_v1.md`。
- memory: `project-hwsw-codesign-route2.md`(决策+事实+S0b 定论)。

---

## 8. 接手第一步
1. 读本文件 + memory `project-hwsw-codesign-route2.md` + §4 两份清单。
2. 确认用户是否仍走 route2/S2(TVM)。确认 GPU 可用性(nvidia-smi，找 util≤2/mem≤50 的卡)。
3. 起 S2.1：单 agent 装 TVM + build `models/stage_a_cache/base.onnx` + 数值对齐 PyTorch/ORT。**这是 go/no-go 闸门，过了再谈并行与搜索。**
4. supervisor 核验 S2.1 结果后再投 S2.2。
