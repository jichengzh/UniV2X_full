,# HANDOFF: CoDriving 软硬件优化 — 瓶颈归因纠错 + 全网口径 (v2, 2026-06-18)

> 全新交接(替代 `HANDOFF_codriving_optimization_grid_v1.md`, 旧文档仅留作历史与原始数据索引)。
> 读完此文即可接续。配套 memory: `project-codriving-optimization-pilot`。
> ★本版核心 = **用户两轮驳回后, 重测纠正了"瓶颈"与"剪枝是否加速"的错误结论**。第一轮 forward-hook 计时已证不可信、作废。

---

## 0. 一句话现状
模仿 Pyramid 路线给 CoDriving(DAIR-V2X 开环口径)做了剪枝×量化×硬件真测。**两个已坐实的真结论**:
1. **"剪枝↑AP" = 训练协议 confound, 不是剪枝/过参数化**(iso-budget 对照: 不剪枝 base 跑同样第二轮退火 AP50 0.561→0.626, 涨幅大于所有剪枝档)。
2. **剪枝确实加速 backbone**(隔离实测 eager 1.56× / TRT 1.7×); 但在**当前 eager 全网管线**里被 CPU/launch/copy 开销稀释, 全网 wall-clock 提速不明显 —— **根因是 eager 开销(5460 kernel launch + 880 次 copy/帧), 非剪枝无效**。要兑现到全网须 **修 decode + 整管线 TRT**。

**下一步主线**(用户将清理上下文后开始): 修掉后处理 CPU-meshgrid → 整管线 TRT/降 eager 开销 → 以**全网口径**重测剪枝/量化是否兑现。

---

## 1. ★三个被反复确认/纠正的关键结论

### 1.1 瓶颈归因(用户驳"backbone 是瓶颈 vs 你说 VFE/decode 矛盾")
- **两种口径要分清**: backbone 占 ~85% 参数 + 最大 FLOPs(当初选它做优化目标**正确**); 但 **eager wall-clock 不由单一阶段主导**。
- **torch.profiler 真值(base, `codriving_profile.py`)**: `wall 61.5ms ≫ GPU 自时间 22.6ms`(CPU 自时间 103.7ms)⇒ **eager 下被 CPU/同步/Python 开销主导, 不是 GPU-bound**。
  - 头部开销: `aten::copy_` **54.6ms CPU**(880 次/帧)、`cudaLaunchKernel` 6.7ms(**5460 次启动/帧**)、`cudaMalloc` 18ms。
  - 真 GPU 计算: `cudnn_convolution` 4.3ms + cutlass/xmma gemm ~3ms(卷积总计 ~7ms)。
- ⇒ **真瓶颈 = eager 海量小算子 + 主机-设备拷贝**, 不是某单阶段。

### 1.2 剪枝确实加速 backbone(用户驳"剪 91% 零加速不可能")
隔离基准(固定输入 (2,64,256,512), 200 runs, `codriving_verify_bottleneck.py`):
| 档 | backbone 参数 | eager latency | TRT body(旧测) |
|---|---|---|---|
| base [64,128,256] | 7.58M | 3.63ms | 0.47ms |
| p50 [32,64,128] | 1.91M | 3.10ms | 0.27ms |
| p75 [64,32,64] | 0.71M | **2.32ms (eager 1.56×)** | 0.28ms (TRT 1.7×) |
- 用户对: 剪枝**真的加速 backbone**。我第一轮 hook 测的"backbone 平 5.3ms 不随剪枝降"是**测量错误, 作废**。
- 参数降 10.7× 而 eager 只 1.56× = 小尺度部分 launch/memory-bound; TRT 因 kernel 融合更接近线性。

### 1.3 后处理: CoDriving vs Pyramid 差异(用户问"NMS 不该压到 2ms 吗")
- CoDriving 的 `generate_predicted_boxes`(`center_point_codriving.py`)每次调用都 `meshgrid(arange(H),arange(W))` **在 CPU 建坐标网格 + `.to(device)` 拷 GPU**(×4/帧)→ 主机-设备同步。
- 实测: 原版 **2.05ms/call**; 缓存网格(GPU 预建一次)版 **0.086ms/call = 24×**(`logs/verify_bottleneck.json`)。
- ⇒ **这就是 CoDriving 后处理比 Pyramid 慢的原因**: Pyramid 锚框在 init 预生成一次(NMS ~2ms 高效), CoDriving 每帧重建。**一行级可修, 非本质成本**。
- 注: 我第一轮 hook 报的 decode "62ms" 也是测量错误, 真值 2ms/call。

### 1.4 剪枝↑AP = 训练 confound(已 final, 见 §2 数据)
iso-budget 对照: 对**不剪枝的 base** 跑与剪枝档**完全相同**的第二轮 30ep 退火(warm-start bestval@11, lr0.002 multistep[10,20])。结果 base 不剪枝 AP50 0.561→**0.626(+0.065), 涨幅大于所有剪枝档**(p25+0.026/p50+0.054/p75+0.057)。
- ⇒ **"越剪越准" 100% 是第二轮 LR 退火(SGDR warm-restart)的训练产物, 不是剪枝/过参数化**。iso-budget 下剪枝反而轻微掉 AP。
- 文献对齐(用户要引用源): **Liu et al. ICLR2019 "Rethinking the Value of Network Pruning"**(同预算下表观剪枝增益消失, 最直接) / **Renda et al. ICLR2020 "Comparing Rewinding and Fine-tuning"**(LR rewinding 解释第二轮退火) / **Li et al. ICLR2017 "Pruning Filters for Efficient ConvNets"**(中等剪枝近无损, 无单调"越剪越准") / **Fang et al. CVPR2023 "DepGraph"**(所用方法)。
- caveat: ImageNet ResNet 剪枝**有** wall-clock 加速(compute-bound); 本结论是此小 BEV + eager 管线特性, 非反驳文献。

---

## 2. 已采真测数据(口径见 §3)

### 2.1 协同 AP — iso-budget 对照(final, `results/codriving_isobudget_verdict.csv`)
| 模型 | backbone剪率 | 训练 | AP30/50/70 |
|---|---|---|---|
| base 原始 | 0 | scratch 30ep bestval@11 | 0.674/**0.561**/0.369 |
| **base-isobudget** | **0** | +第二轮30ep退火 | 0.718/**0.626**/0.406 |
| p25 | 44.7% | 剪+退火 | 0.693/0.586/0.366 |
| p50 (×3 seed) | 74.8% | 剪+退火 | 0.615/0.606/0.610 (σ≈±0.005) |
| p75 | 90.7% | 剪+退火 | 0.715/0.618/0.405 |
> 注: 这些 AP 含训练 confound(剪枝档比 base 多一轮退火), 解读见 §1.4。

### 2.2 硬件轴(`results/codriving_dair_grid_8point_clean.csv`)
| 档 | body lat FP16/INT8 (TRT, ms) | 能耗 J/frame FP16/INT8 | peak 吞吐 fps(FP16 batch-sweep) | engine MB FP16/INT8 |
|---|---|---|---|---|
| base | 0.47/0.29 | 0.179/0.081 | 2423 | 16.0/9.1 |
| p25 | 0.33/0.25 | 0.114/0.073 | 3507 | 9.6/5.9 |
| p50 | 0.27/0.21 | 0.081/0.055 | 5053 | 5.4/3.7 |
| p75 | 0.28/0.21 | 0.079/0.052 | 4806 | 2.1/2.1 |
- INT8 省 35-54% J/frame; 吞吐 batch-sweep peak@~b8(空间批 ~2.8× + 剪枝再叠 ~2×)。
- ⚠️ **以上 body lat 是 TRT 单核子网口径**, 不是全网。全网口径见 §1.1/§2.3。

### 2.3 全网时延(eager / hybrid 口径)
- **all-PyTorch 全网**(profiler): base wall ~61.5ms(GPU 计算仅 22.6ms, 其余 eager 开销)。
- **hybrid(TRT backbone+融合核 + PyTorch VFE/scatter/decode)**: base FP16 28.75 / INT8 29.15 / p50 26.5 / p75 24.1ms; 纯 PyTorch 98.4ms。hybrid 全网平(~24-29ms)是因 backbone 已被 TRT 压到极小、剩 eager 前后处理主导 —— **同 §1.1 根因, 不是剪枝无效**。

---

## 3. 指标定义(钉死)
- **剪枝率 = backbone L1 channel 比率 0.25/0.5/0.75**(DepGraph, round_to=32, 只作用 backbone, 占全模型 85% 参数)。总参数降 44.7/74.8/90.7%。p75 实得不规则 [64,32,64]。
- **body latency**: TRT 单 agent 稠密核(backbone→heads), 不含 VFE/scatter/fusion/decode/NMS。
- **全网 latency**: 完整 forward, eager 或 hybrid; 是部署真口径(用户令: 今后以全网为准)。
- **协同 AP**: DAIR val 1789, 范围 102.4×51.2, 单类车, opencood eval_utils。
- **能耗**: 整卡功率×单帧时间, 空闲 GPU(idle 25W)。**吞吐**: FP16 动态 batch 引擎 peak。

---

## 4. ★下一步(用户主线 — 让剪枝/量化兑现到全网)
1. **修 decode CPU-meshgrid**(已定位, 一行级): `generate_predicted_boxes` 把坐标网格按 (H,W) 预建在 GPU 缓存, 别每次 `.to(device)`。2.05→0.086ms/call(24×)。**在隔离副本改, 勿动主框架**。
2. **消 eager 全网开销**(根因 = 5460 launch + 880 copy/帧): ①整管线 TRT(VFE 稀疏算子是难点; scatter/fusion/heads/decode 可 TRT)②CUDA Graph 消 launch(项目既有: 单 GPU 仅 1.05-1.10×, 有限)③batch 摊薄。
3. **全网口径重测剪枝/量化**: 修好 ①②后, 重测全网 base/p50/p75 —— GPU 计算成主导时, 剪枝的 1.56-1.7× 才会显现到全网。**这是验证"搜索空间能否动部署指标"的关键实验**。
4. **若全网仍不显著**: 说明此小模型 + DAIR 任务下 prune/quant 搜索轴对部署时延价值有限, 需转向更大/compute-bound 模型或换优化轴(同 Pyramid 既有结论)。

---

## 5. 环境/路径/脚本
- **4090**(TRT + 全网/profiler 实测): env `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`, TRT10.13。DAIR 数据 `/data/jichengzhi_dair/dair_eval/`(48G, split json 在 `dair_eval/split_json/`)。V2Xverse 副本 `${V2X_ROOT}verse`(opencood, 已编 box_overlaps py39)。
- **H800**(训练/剪枝/导出): `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认)。隔离副本 `${V2X_DATA_ROOT}/V2Xverse_pyramid`, `PYTHONPATH=${V2X_DATA_ROOT}/tp_lib:/data/jichengzhi_v2x/t2lib:.`。
- **脚本(4090, `scripts/phase1/`)**:
  - `codriving_verify_bottleneck.py` — 隔离 backbone latency + decode 缓存对照(真值来源)。
  - `codriving_profile.py` — torch.profiler 全网 op 级 breakdown(可信, 替代 hook)。
  - `codriving_perstage_timing.py` — ⚠️ forward-hook 版, **已证不准, 勿用**(留作反面教材)。
  - `codriving_hybrid_ap_eval.py` / `codriving_{p25,p50,p75}_hybrid_ap_eval.py` — 协同 AP(含 lat_acc 全网计时)。
  - `bench_pareto_axes.py`(能耗+latency) / `bench_throughput.py`(batch-sweep)。
- **模型 forward**: `${V2X_ROOT}verse/opencood/models/center_point_codriving.py`(`generate_predicted_boxes` = decode 待修处; backbone 多尺度在 `fusion_net` 里复用一次)。
- **结果**: `results/codriving_isobudget_verdict.csv`(AP final) + `codriving_dair_grid_8point_clean.csv`(硬件轴) + `output/codriving_pilot/logs/{verify_bottleneck,perstage_*,e2e_*,pareto_axes_results,throughput_sweep_results}.json`。

## 6. 方法学教训(避雷)
- ★**forward-hook + CUDA event 在复杂 forward(模块多次复用 / async overlap)里测不准** —— 第一轮据此得出"backbone 平 / decode 62ms"全错。**逐阶段占比用 torch.profiler; 单模块加速用隔离固定输入基准**。
- **区分 FLOPs/参数口径 vs wall-clock 口径**: backbone 是前者的最大项(选目标依据), 不一定是后者瓶颈。
- **eager wall ≫ GPU 自时间 ⇒ CPU/launch/copy-bound**: 此时优化算子(剪枝/量化)收益被开销稀释, 须先消 eager 开销(TRT/CUDA Graph)。
- **真测才下结论**: "剪枝免费/无加速"在隔离重测前都是未证; 凡 agent/自报"已测"必复核(读 log/隔离重跑)。
- 不动仿真主框架代码, 改在隔离副本, 改前确认。

---

## 7. ★反思: 实测 GPU 条件与结果可信度(用户 2026-06-18 追问)
**诚实结论: 没有一个时延/能耗实验是在"完全空闲卡(util 0% 且显存 ≤50MiB)"上跑的。** 4090 是多人共享机(本 session 常被 collision_warning/drivevla/unitraj 等他人进程占用), 基本没等到过完全干净的卡。各实验真实条件:

| 实验 | 脚本 | 卡/条件(日志实记) | 可信度 |
|---|---|---|---|
| 能耗 J/frame + body latency | `bench_pareto_axes.py` | **GPU7**, idle power **25.4W**(= 4090 真空闲基线), mem 494MiB(=本进程 torch context); 选卡时 util==0 | **较可信**: 25.4W 空闲基线证当时无他人计算; 但**仅在选卡+基线采样时验证空闲, 未在 ~64s 运行全程连续监控**, 不能 100% 排除中途他人进程启动 |
| 吞吐 batch-sweep | `bench_throughput.py` | **GPU0**, util==0 但**他人进程 ~18GB 显存常驻**(我显式放宽到"仅 util==0"); 未记录功率 | **绝对值不可信**(动态引擎 opt=8 致 b1 偏低 + 显存非净); **只可信 scaling 形状**(peak@~b8, 剪枝抬升峰值) |
| 隔离 backbone + decode | `codriving_verify_bottleneck.py` | 低 util 但**共享卡**(按 util≤5+最大空闲显存选), 未功率监控 | **定性/相对可信**(剪枝加速 backbone、decode 24× 都是大效应+固定输入 200 runs); 绝对 ms 带 ±10-20% 竞争不确定性 |
| 全网 profiler | `codriving_profile.py` | **GPU5**(20GB 空闲, 即他人占 ~3GB), 低 util | **结构结论可信**(wall≫GPU 自时间 ⇒ CPU/launch/copy-bound 是大比例事实); 绝对 wall(61.5 vs 早先 88ms)本身就显run-to-run/卡间方差 |
| 原始 8 点 body latency | (上一 session) | memory 记 util=0% 但**非 mem≤50MiB**, 以 std 极小佐证干净 | 同上, util 净但显存非净 |

### 7.1 哪些结论稳、哪些数要打折
- **稳(不受竞争影响的定性/相对结论)**: ①剪枝加速 backbone(1.56×/1.7×, 固定输入隔离测+TRT 交叉印证); ②decode CPU-meshgrid 24× artifact(同卡原版 vs 缓存版对照, 竞争同等抵消); ③全网 eager 是 CPU/launch/copy-bound(wall≫GPU 是数量级差异); ④INT8 省能耗(单调+量级与 Pyramid 既有一致); ⑤剪枝↑AP=训练 confound(AP 评测不涉 GPU 计时, 不受此影响)。
- **要打折(绝对数值)**: 所有绝对 ms / J/frame / fps 带 **±10-20% 竞争不确定性**; **能耗最脆弱**(NVML 读的是整卡功率, 若运行中他人计算则虚高); **吞吐绝对值最不可信**(显存非净 + 动态引擎)。
- **跨 session 不可直接比绝对值**: 同一 base body latency 在不同卡/时刻 0.466↔0.471ms、全网 88↔61.5ms, 卡间/竞争方差真实存在。

### 7.2 要做到"金标准可信"还差什么(接手补)
1. **独占卡**: 申请/协调一张无他人进程的卡(或 MPS 关闭 + 预约), 确保 util 0% 且 mem 仅本进程。
2. **运行全程连续监控**: 跑测时并行采 util/power.draw, 记录运行期间 max util; 若中途 >0(他人计算)则**丢弃该次重跑**。当前脚本只在开头采一次基线, 不够。
3. **多次重复 + 报方差**: 每点 ≥3 次, 报 p50±std; 当前多为单次。
4. **能耗减空载基线**: 报 (运行功率 − idle 25W) × 时间, 或至少同时记录运行期 idle 漂移。
5. ⇒ **当前数据足以支撑定性/相对结论与数量级判断, 不足以支撑论文级绝对数值**; 写论文前关键点(尤其能耗/全网时延)须在独占卡 + 连续监控 + 多次重复下复测。

### 7.3 教训
- ★**"选卡时 util==0" ≠ "运行全程空闲"**: 必须运行期连续监控并对竞争敏感的指标(能耗/时延)做 contention-gate, 否则只能下相对结论。
- ★**共享集群上别承诺绝对数值可信**: 老老实实标"util净/显存净/是否全程监控", 区分定性结论(稳)与绝对数值(打折)。
