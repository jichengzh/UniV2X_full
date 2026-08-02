# HANDOFF: CoDriving DAIR 软硬件优化实测网格 + 三大问题反思 (v1, 2026-06-18)

> 自足交接。读完能接续。配套 memory: `project-codriving-optimization-pilot`(基础设施+全数据)。
> 上游: `project-codriving-dair-openloop-ap`(开环 AP 基线) · `HANDOFF_codriving_dair_openloop_v1`(迁移)。
> ★本文档核心 = **诚实反思用户提出的三个问题**, 不是邀功。数据已采但结论有方法学缺陷, 接手者必读 §3。

---

## 0. 一句话现状
模仿 Pyramid 路线给 CoDriving(DAIR-V2X 开环口径)采了 **base/p25/p50/p75 × {FP16,INT8} = 8 点**真测(协同 AP + body latency + engine size), 流水线全打通且经主控核验。用户审阅后指出三个根本问题(§3), 我认同: ①不构成 Pareto 前沿(退化簇); ②"剪枝越多 AP 越高"用"过参数化"解释是错的; ③指标定义不清 + Pareto 轴不全。

> **★[2026-06-18 §4 决定性实验已完成 — 结论见 §8]** iso-budget 对照(Exp1)定性: **"剪枝↑AP" 100% 是训练协议 confound(第二轮退火), 不是剪枝/过参数化** —— 不剪枝的 base 跑同样的第二轮 30ep 退火, AP50 从 0.561→**0.626(+0.065)**, 比任何剪枝档涨得都多。用户的怀疑完全正确。同时补齐能耗(Exp2)+ 吞吐(Exp2)+ p25 latency 重测(Exp4=0.329ms)+ 种子方差(Exp3=±0.005)。Exp5 Pareto 可行性分析见 §8.4。**§4 五个实验全部完成**。**★[2026-06-18 第二轮纠错, 见 §9 — 接手必读, 覆盖 §8 关于"瓶颈/全网"的措辞]**: 用户驳回后重测证实 ①剪枝**确实加速** backbone(隔离 eager 1.56×/TRT 1.7×, 我之前 hook 测的"平"是错的); ②后处理 decode 真值 2ms/call(可缓存到 0.09ms, 24×), 非 hook 误报的 62ms; ③全网 eager 是 **CPU/launch/copy-bound**(wall 61.5ms 中 GPU 仅 22.6ms, 5460 launch+880 copy/帧)。⇒ 剪枝全网提速**被 eager 开销稀释非无效**, 需 TRT 整管线 + 修 decode 才兑现。§8 的"e2e 与剪枝无关/Pareto 塌平"是 eager 开销假象, 以 §9 为准。

---

## 1. 采到的数据(主控核验值, 口径见 §2)
`results/codriving_dair_grid_8point_clean.csv`(已统一; 原始 `E_codriving_collab_ap_4090.csv` + `E_codriving_trt_base_4090.csv` schema 混乱, 勿直接 DictReader)。

| 档 | backbone剪率 | 总参数↓ | stage_filters | FP16 协同AP30/50/70 | INT8 协同AP30/50/70 | body lat FP16/INT8(ms) | body engine FP16/INT8 |
|---|---|---|---|---|---|---|---|
| base | 0 | 0% (7.59M) | [64,128,256] | 0.674/0.561/0.368 | 0.672/0.558/0.364 | 0.466/0.290 | 16.0/9.1MB |
| p25 | 0.25 | 44.7% (4.19M) | [32,96,192] | 0.686/0.578/0.360 | 0.654/0.553/0.347 | 0.357/0.252 | 9.6/5.9MB |
| p50 | 0.50 | 74.8% (1.91M) | [32,64,128] | 0.704/0.615/0.385 | 0.686/0.595/0.375 | 0.262/0.210 | 5.4/3.7MB |
| p75 | 0.75 | 90.7% (0.71M) | **[64,32,64]** | 0.708/0.609/0.400 | 0.659/0.547/0.351 | 0.274/0.211 | 3.1/2.1MB |

- **FP16 sanity**(TRT-FP16 vs PyTorch-FP32): base/p50 ~0.001(完美); p25 gap 0.008(PyTorch 0.586 vs TRT 0.578); p75 gap 0.009(PyTorch 0.618 vs TRT 0.609)= **大平移 warp 的 FP16 精度残差**(已用 Einsum/GridSample 强制 FP32 把 p75 从崩溃 0.278 救回 0.609, 残差未到 FP32)。
- **可分辨的真信号只有一处**: INT8 ΔAP50 随剪枝放大 base −0.003 → p25 −0.025 → p50 −0.020 → **p75 −0.062**(prune×quant 耦合); 及 latency 非单调 p50 0.262 < **p75 0.274**(非32对齐通道 [64,32,64] kernel-cliff)。

---

## 2. ★指标定义(用户问题③: 之前没说清, 这里钉死)
- **剪枝率 p25/p50/p75 = L1 channel 剪枝比率 0.25/0.5/0.75, 只作用于 backbone**(`prune_target=backbone`, ResNetBEV, DepGraph 依赖组, round_to=32)。**不是全模型比率**。backbone 占全模型 85% 参数。
- **总参数降幅 ≠ 剪枝率**: 剪 channel 同时缩 in/out 维, 参数近平方降 → ratio0.25→总降44.7%, 0.5→74.8%, 0.75→90.7%。报告里务必区分"backbone剪率"与"总参数%"。
- ⚠️**剪枝分配不规则**: 全局 L1 + round_to=32 使各层非均匀, p75 得 **[64,32,64]**(layer0 反比 p50 多)→ **p25/50/75 不是干净嵌套架构序列**, 这本身污染"更多剪枝"的单调叙事。
- **协同 AP**: DAIR-V2X val **1789** 全集, hybrid 推理(PyTorch VFE+scatter → **TRT 协同核(N=2 双 agent+pairwise_t_matrix, 含 where2comm fusion)** → PyTorch CenterPoint 解码+NMS), IoU 同 opencood eval_utils。FP16/INT8 来自各自引擎, PyTorch-FP32 作 sanity。范围 102.4×51.2, 单类车。
- **body latency**: ⚠️ **单 agent 稠密核(backbone→heads)TRT, 不含 voxelize/scatter/fusion/NMS** —— 是**部分时延**, 非 e2e。CUDA-event warmup500/run500, 4090 空闲卡。对齐 Pyramid 的 body_subnet 口径(可跨模型比 body, 但≠端到端)。
- **engine size**: TRT body 引擎 MB。
- ⚠️**Pareto 轴不全(用户问题③)**: 当前只有 {AP, body-latency, size}。**能耗(J/frame)= 未测; 吞吐 = 未测; e2e 时延(含 VFE/scatter/NMS)= 未测**。`pareto_definition_v1.md` 定义的 5 指标(+闭环 DS/RC)远未集齐。

---

## 3. ★三大问题的诚实反思(用户提出, 我认同)

### 问题① 没有 Pareto 前沿
**事实**: 8 点不构成精度-成本 trade-off 前沿。剪枝+INT8 在**速度更快**的同时 AP **不降反升或持平**(p50/p75 FP16 AP > base, 且更快更小)→ 没有"用精度换速度"的下行前沿, 而是**所有点支配 base 的退化簇**。这与项目对 Pyramid 的既有定论 **"剪枝 Pareto 退化"**(`dims_pruning_v1.md §8.4-8.5`)同病。**唯一出现真 trade-off 的角落** = prune×INT8 cliff(p75 INT8 用 AP −0.062 换速度)。⇒ 当前形态下 CoDriving/DAIR 上**剪枝轴无法产生 Pareto 信号**, 与 Pyramid 一致。

### 问题② "剪枝越多 AP 越高"—— "过参数化"是错的偷懒解释
**用户质疑正确**: 标准 ResNet 剪枝不会"越剪越准"。真因不是过参数化, 而是**训练协议 confound**(我之前只标 caveat 没查实, 还在小结里写"过参数化确认"= 过早结论, 违反项目纪律):
1. **第二轮 LR 退火周期(主因, SGDR/warm-restart 效应)**: 剪枝档 = 从 base 权重 warm-start + **全新 30ep finetune**(Adam lr=**0.002**, multistep[10,20] gamma0.1), 全部 bestval@**5**。而 base = 原训练 bestval@**11**, **没吃这第二轮退火**。给已训模型再来一轮完整 LR 退火本身就常提精度 —— 剪枝档白赚了这轮, base 没有。
2. **剪枝作正则(次因)**: DAIR 单类 + val 仅 1789 + 模型重度偏大 → 去容量减过拟合, val AP 升。但与①纠缠, 无法单独归因。
3. **窄带可能含 finetune 方差**: 全部 AP50 在 0.56–0.62 窄带, 单 seed, 升幅可能部分是退火/seed 噪声。
4. **不规则剪枝分配**(§2): p25/50/75 非嵌套架构, "更多剪枝"轴本身不干净。

**⇒ 结论**: 当前协议下**无法分离"剪枝效应"与"多一轮训练"**; AP 升高最可能是**训练协议产物, 不是剪枝/过参数化的模型规律**。

### 问题③ 指标定义不清 + Pareto 轴不全
已在 §2 钉死定义。轴不全是硬伤: 只有 AP+body-latency+size, **缺能耗/吞吐/e2e 时延**, 不足以画 `pareto_definition_v1.md` 要求的多目标前沿。

---

## 4. ★决定性的下一步实验(接手必做, 按优先级)
1. **[最高] iso-budget 对照**: 对 **base(不剪枝)**做**完全相同的 finetune**(从 bestval@11 续训 30ep, lr0.002 multistep[10,20])。
   - 若 base-refinetune 也到 ~0.61 → **AP 升高纯是第二轮训练, 剪枝零贡献(甚至负)**, "剪枝近免费"说法作废, 改写为"训练协议 artifact"。
   - 若 base 仍 ~0.56 而剪枝档 0.61 → 剪枝确有正则增益(但仍非"过参数化免费")。
   - **这一个实验就能定性问题②**。命令见 §6。
2. **补全 Pareto 轴**: 能耗(E4 式 NVML power.draw 采样, body+e2e)、吞吐(batch sweep)、**e2e 时延**(含 VFE/scatter/NMS, 才是部署真延迟; body-only 误导)。
3. **多 seed finetune**(≥3)定 AP 窄带方差, 确认 0.56→0.62 是信号还是噪声。
4. **重测 p25 body FP16 latency**(0.357 来自 agent 报告, CSV 行曾错位, 需 1 次干净重测)。
5. **想清楚 CoDriving/DAIR 能否产生真 Pareto**: 若剪枝轴注定退化(同 Pyramid), 则精度-成本 trade-off 只能来自 ①INT8 cliff(已见 p75)②更难数据/模型(V2X-ViT 路线)③量化深度轴。**不要硬画一个退化簇当 Pareto**。

---

## 5. 环境/路径/产物(接手用)
- **H800**(数据+训练+导出): `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认)。隔离副本 `${V2X_DATA_ROOT}/V2Xverse_pyramid`, env `PYTHONPATH=${V2X_DATA_ROOT}/tp_lib:/data/jichengzhi_v2x/t2lib:. python3`(torch2.1.2, **torch_pruning 1.6.0 隔离装在 tp_lib, 勿改 t2lib**)。
- **4090**(TRT+协同AP harness): env `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`, TRT10.13。**DAIR 数据已传 4090** `/data/jichengzhi_dair/dair_eval/`(48G)。4090 V2Xverse opencood 已编 box_overlaps py39。
- **关键脚本**:
  - 剪枝: H800 `${V2X_DATA_ROOT}/V2Xverse_pyramid/depgraph_codriving.py`(`--ratio R --out-dir ...`)
  - body ONNX: depgraph_codriving 的 CoDrivingTraceNet @ (1,64,256,512) 单类
  - 协同 ONNX(★含 pairwise_t_matrix 输入分辨率统一归一化 fix): `tools/export_onnx_codriving_collab.py`
  - 协同 hybrid AP harness: `scripts/phase1/codriving_{p25,p50,p75}_hybrid_ap_eval.py`(--engine 换即可; 注意各档 arch 不同)
  - mAOE/AP eval: H800 `eval_tp_errors_codriving.py <model_dir>`
- **ckpt**: base `opencood/logs/dair_centerpoint_codriving_2026_06_15_21_19_14/net_epoch_bestval_at11.pth`; 剪枝档 `output/codriving_pilot/{p25,p50,p75}_warmstart/net_epoch_bestval_at5.pth`。
- **数据(H800)**: data_dir `/data/jichengzhi_v2x/dair/cooperative-vehicle-infrastructure`; split `/data/lxf_data/DAIR-V2X-C/.../{train,val}.json`(4811/1789)。
- **引擎**: `output/codriving_pilot/collab_engines/*.engine`(协同) + `{base,p25,p50,p75}` body 引擎。
- **结果**: `results/codriving_dair_grid_8point_clean.csv`(干净) + 原始两 CSV。

## 6. iso-budget 对照命令(问题②决定性实验)
```bash
# H800: base 不剪枝, 同 finetune 协议续训(用 base config + base bestval@11 作 model_dir resume)
cd ${V2X_DATA_ROOT}/V2Xverse_pyramid
# 把 base bestval@11 拷成新 model_dir 的 resume 起点, 用与剪枝档同样的 config(30ep lr0.002 multistep[10,20])
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=${V2X_DATA_ROOT}/tp_lib:/data/jichengzhi_v2x/t2lib:. \
  python3 opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/dairv2x/lidar_only/dair_centerpoint_codriving.yaml \
  --model_dir output/codriving_pilot/base_refinetune_isobudget   # 需先把 bestval@11 放入作 resume
# 完成后 eval, 比 0.561 vs 剪枝档 0.61
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=... python3 eval_tp_errors_codriving.py output/codriving_pilot/base_refinetune_isobudget
```
> 注: train.py 的 resume 机制需确认(它从 model_dir 找最新 epoch resume); 要让它从 base bestval@11 起、再训 30ep。细节接手时核 train.py。

## 7. 坑与教训(供避雷)
- ★**别再用"过参数化"一句带过**: 没跑 iso-budget 对照前, "剪枝免费/有益"都是未证结论(项目纪律: 真测才下结论)。
- **agent 反复早返回 + 重复 spawn eval + 过度宣称"已验证"**: 本轮 hw-optimizer agent 多次"设后台监视器就返回", 还把 base ONNX 导成多类(cls=3)、漏跑 p25 PyTorch sanity、p75 FP16 崩 0.278 未自检。**凡 agent 报"已测/已验证"必复核**(读 log/查形状/核 sanity)。本轮主控抓到全部。
- **CSV schema 混乱**: 不同脚本写不同列序, DictReader 会错位。统一表 `codriving_dair_grid_8point_clean.csv` 为准。
- **协同 AP 必须用协同核引擎**(含 fusion + pairwise_t_matrix), body 引擎(跳 fusion)只能给 latency。
- **FP16 大平移 warp 精度**: 协同核的 GridSample/Einsum 在 DAIR 大平移(80-120m)下 FP16 会崩, 必须对这些层强制 FP32(precise 引擎)。
- **H800 无 onnx 包/无 tensorrt**: ONNX 导出用 torch-only(跳 checker)或 tvm310 的 onnx; TRT 全在 4090。

---

## 8. ★§4 决定性实验结果 (2026-06-18 overnight, 主控自跑+核验)
> 数据: `results/codriving_isobudget_verdict.csv`(AP 对照)+ `results/codriving_dair_grid_8point_clean.csv`(已补能耗/吞吐列)+ `output/codriving_pilot/logs/{pareto_axes,throughput_sweep}_results.csv`。
> ✅ AP 为 **final**(2026-06-18 04:35 锁定): bestval 自 epoch9 起稳定, 穿过 epoch-10 与 epoch-20 两次 LR 降档不变(seed2/seed3 @epoch21 已过第二降档确认; base_isobudget @epoch19 bestval@5 稳定 14 epochs 含整个 lr=0.0002 段)。eval 对最终 bestval ckpt 确定性 → 数值锁定。三训练留跑至 30ep(无害)。

### 8.1 Exp1 iso-budget 对照 — 问题② 定性结论(★最重要)
**做法**: 对 **不剪枝的 base** 用与剪枝档**完全相同**的 finetune 协议(从 base bestval@11 warm-start + 全新 30ep, Adam lr0.002 multistep[10,20], batch6), 唯一变量 = 不剪枝。

| 模型 | backbone剪率 | 训练 | AP30/50/70 | vs base ΔAP50 |
|---|---|---|---|---|
| base 原始 | 0 | scratch 30ep, bestval@11 | 0.674/**0.561**/0.369 | — |
| **base-isobudget** | **0** | **+第二轮30ep退火** | 0.718/**0.626**/0.406 | **+0.065** |
| p25 | 25 | 剪+退火 | 0.693/0.586/0.366 | +0.025 |
| p50 | 50 | 剪+退火 | 0.703/0.615/0.385 | +0.054 |
| p75 | 75 | 剪+退火 | 0.715/0.618/0.405 | +0.057 |

**结论(无歧义)**: 不剪枝的 base 仅靠第二轮退火就涨 **+0.065**, **大于所有剪枝档的涨幅**。⇒ **"剪枝越多 AP 越高" 完全是训练协议产物(第二轮 LR 退火 / SGDR warm-restart 效应), 与剪枝无关**。"过参数化"解释作废, 用户怀疑正确。**iso-budget 下剪枝反而轻微掉 AP**(0.626 不剪 → 0.586-0.618 剪)。

### 8.2 Exp3 种子方差 — finetune 噪声底
p50 三次独立 finetune(原/seed2/seed3): AP50 = 0.615/0.606/0.610 → 均值 0.610, 极差 0.009, **σ≈±0.005**。⇒ 8.1 的 base-iso(0.626) vs 剪枝(0.586-0.618)差距 >噪声, 可信; 剪枝档之间(p50↔p75)差异在噪声内。

### 8.3 Exp2/Exp4 补齐 Pareto 轴(真测, 4090 GPU 空闲 25W)
- **能耗 J/frame**(整卡功率×单帧时间): base fp16 0.179 / int8 0.081; p25 0.114/0.073; p50 0.081/0.055; p75 0.079/0.052。**INT8 省 35-54% J/frame**(与项目 Pyramid 既有结论一致), 剪枝单调降功率(398→247W)。
- **吞吐(FP16 batch-sweep, 空间批=RSU 多车)**: peak@~b8 — base 2423 / p25 3507 / p50 5053 / p75 4806 fps; 空间批比 b1 提 ~2.8×, 剪枝再叠 ~2×。caveat: 动态引擎 opt=8 致 b1 偏低, 看 scaling 形状别看绝对 b1; int8 未 sweep。
- **Exp4 p25 body FP16 latency 重测**: 0.357(旧, 偏高)→ **0.329ms**(干净)。其余 latency 与旧表一致。
- **★e2e 时延(最后一根缺轴, 现已补 — 重磅)**: hybrid harness 自带 lat_acc(PyTorch VFE+scatter→TRT 协同核 N=2+fusion→PyTorch decode, 不含末端 NMS), DAIR val p50_lat: **base FP16 28.75ms / base INT8 29.15ms**(全 1789 帧)、300 帧复测 base 24.0 / **p50 26.5 / p75 24.1ms**、**纯 PyTorch(无 TRT)98.4ms**。⇒ ①**body 核仅占 e2e ~2%**(body 0.47ms vs e2e ~25ms); ②**剪枝/量化在 e2e 层面几乎不可见** —— base INT8 e2e 29.15≈FP16 28.75, p50/p75/base e2e 全 ~24-26ms 无剪枝趋势(body 省的 0.2ms 淹没在 ~24ms PyTorch 前后处理里); ③**唯一大 e2e 杠杆 = TRT 化整核**(98.4→28.75 = 3.4×)。这是部署 Amdahl 现实, 呼应项目 Pyramid "NMS 占 e2e 72%" 既有定论。**⇒ §1/§8 的 body-latency Pareto(剪枝1.7×/INT8 1.6×)只在 body 子网口径成立; 到真 e2e 几乎塌平。**

### 8.4 ★Exp5 — CoDriving/DAIR 能否产生真 Pareto?(去 confound 后重答问题①)
用 **iso-budget AP**(去除训练 confound)配 latency/energy 重画:

| 点 | AP50 | body_lat(ms) | J/frame | 支配关系 |
|---|---|---|---|---|
| base-iso(0%剪) | 0.626 | 0.471 | 0.179 | Pareto(AP 顶) |
| p75(90.7%) | 0.618 | 0.280 | 0.079 | Pareto |
| p50(74.8%) | 0.610 | 0.272 | 0.081 | Pareto(最快) |
| p25(44.7%) | 0.586 | 0.329 | 0.114 | **被支配**(AP更低且更慢) |

**结论**:
1. **去 confound 后确实存在一条真 Pareto 前沿**(base-iso ≻方向 p75 ≻ p50, p25 被支配)—— 之前"退化簇/剪枝支配 base"是 confound 制造的**假象/倒置前沿**; 修正后 base 是 AP 领头, 剪枝拿少量 AP 换大量速度/能耗 = 方向正确的前沿。
2. **但前沿极浅**: AP 仅 0.626→0.610(跨度 0.016 ≈3×噪声), 而 latency 跨 1.73×、energy 跨 2.2×。**剪 90.7% 参数只掉 ~0.01 AP50, 可达区间内无精度悬崖** —— 与 Pyramid 的过参数化结论一致(`dims_pruning_v1.md §8.4-8.5`)。根因 = DAIR 单类任务 AP 天花板低(base 仅 0.626)+ 模型过参数化。
3. **唯一陡峭的精度-成本拐点 = INT8×高剪枝 corner**: p75 INT8 用 ΔAP50 **−0.062** 换额外速度/能耗(prune×quant 耦合, 见 8.3/§1)。这是当前空间里唯一真 trade-off knee。
4. **要拿到有意义弯曲的 Pareto**, 必须: ①更难任务/模型(多类 V2Xverse / V2X-ViT / 更密场景, 容量真正紧)②把剪枝推过 90% 找悬崖 ③量化深度轴。**别把这条浅前沿当成强协同信号**。
5. **CoDriving/DAIR 实用操作点(body 子网口径)**: p50/p75 FP16(≈等 AP, 1.7× 快, 2.2× 省电)与 p75 INT8(极致速度/能耗, 若 −0.06 AP 可接受); base FP32/FP16 在成本轴被支配且 AP 优势可忽略。
6. **★但用 e2e 时延轴(部署真口径)重判, 上面这条浅 Pareto 进一步塌平**: e2e ~24-29ms 与剪枝/量化无关(body 仅占 ~2%), 故 (AP, e2e-latency) 平面上 **base-iso(AP 顶, 同 e2e 时延)反而支配剪枝档** —— 剪枝在 e2e 时延轴零收益。剪枝/量化的真收益只剩 **能耗(部分, 因 body 能耗也只是 e2e 一部分)、模型体积、核吞吐(batch)**, 以及"若把 PyTorch VFE/scatter/decode 也 TRT 化/批处理"后才释放的潜力。**⇒ 论文里谈 CoDriving 协同加速 Pareto, 必须区分 body-subnet 口径(有信号)与 e2e 部署口径(信号塌平); 否则会高估剪枝/量化的部署价值。** 这与 Pyramid Amdahl 定论一致, 是跨模型稳健结论。

### 8.5 复跑/产物
- iso-budget 训练: H800 `output/codriving_pilot/base_refinetune_isobudget/`(config=base 原 config 不剪枝, seed `net_epoch_bestval_at0.pth`=base bestval@11; launch 见 §6 同理但用 base config + 不剪枝)。
- 种子重跑: `output/codriving_pilot/p50_seed{2,3}/`。
- prelim eval: `output/codriving_pilot/{D}_evalsnap/tp_errors_result.json`; 脚本 `eval_tp_errors_codriving.py <snapdir>`。
- 4090: 能耗/latency `output/codriving_pilot/bench_pareto_axes.py --gpu N`(N=空闲); 吞吐 `bench_throughput.py`(动态引擎 `dynbatch_engines/`, ONNX 经 `export_onnx_body_dyn.py` 加 dynamic batch 轴重导)。
- **e2e 时延(已做)**: hybrid harness `scripts/phase1/codriving_hybrid_ap_eval.py` 自带 lat_acc(p50_lat/mean_lat 进 `results/E_codriving_collab_ap_*.json`); 跑 base/p50/p75 = `logs/e2e_*.log`。**仅末端 NMS(post_process)在计时区外**, 若要含 NMS 把 line373 post_process 移进 t_a..sync 区间。pruned 档可借基线 harness 换 --engine 测时延(scatter 输出恒 64ch, 接口一致; AP 无意义只取 lat)。

### 8.6 复核状态 (已锁定, 2026-06-18 04:35)
AP 已 final: bestval 穿过 epoch-10/epoch-20 两次 LR 降档保持不变(iso@5/seed2@5/seed3@9, 自 epoch9 稳定)。eval 确定性 ⇒ 无需再跑。三训练继续跑至 30ep 不影响结论(bestval 不会再降)。verdict CSV status 已 prelim→final。

---

## 9. ★[2026-06-18 用户驳回 + 全网逐阶段实测纠错] —— 必读, 推翻 §8.3/§8.4 的"瓶颈"措辞

用户驳两点: ①"backbone 是瓶颈" vs 我说"VFE/scatter/decode 是瓶颈"矛盾; ②剪 91% 参数零加速不可能, 要么数据假要么深挖。主控两轮重测, **第一轮 forward-hook 计时被证实污染(作废), 第二轮隔离基准+torch.profiler 为准**。

> ⚠️**作废**: 第一轮 `codriving_perstage_timing.py`(forward-hook+CUDA event)给出 backbone 平 5.3ms / decode 62ms / 全网 88ms —— **hook 在复杂 forward(backbone 多尺度复用+async overlap)里测不准, 数值不可信, 勿引用**。下面隔离基准才是真值。

### 9.1 真值①: 隔离 backbone 基准(固定输入 (2,64,256,512), 200 runs, `codriving_verify_bottleneck.py`)
| 档 | backbone 参数 | eager latency | TRT body(旧测) |
|---|---|---|---|
| base [64,128,256] | 7.58M | 3.63ms | 0.47ms |
| p50 [32,64,128] | 1.91M | 3.10ms | 0.27ms |
| p75 [64,32,64] | 0.71M | **2.32ms (eager 1.56×)** | 0.28ms (TRT 1.7×) |
⇒ **剪枝确实加速 backbone**(eager 1.56×, TRT 1.7×)。用户对, 我第一轮"backbone 平"是 hook 测量错误。注: 参数降 10.7× 但 eager 只 1.56×(小尺度部分 launch/memory-bound), TRT 因 kernel 融合更接近线性。

### 9.2 真值②: decode 不是 62ms, 是 2ms/call, 且可缓存到 0.09ms
- `generate_predicted_boxes` 原版 **2.05ms/call**(非 hook 误报的 15ms); 内含 `ys,xs=meshgrid(arange(H),arange(W))` 在 **CPU 建网格 + `.to(device)` 每次调用拷 GPU**(×4/帧)。
- 缓存网格(GPU 预建一次)版 = **0.086ms/call → 24×**。⇒ **这就是 CoDriving vs Pyramid 后处理差异**: Pyramid 锚框 init 时预生成一次(NMS~2ms 高效); CoDriving 每次重建坐标网格 + H2D。**可修(缓存网格), 非本质**。

### 9.3 真值③: 全网是 CPU/launch/copy-bound, 不是某单阶段(torch.profiler, base, `codriving_profile.py`)
- **wall 61.5ms ≫ GPU 自时间 22.6ms ≫**(CPU 自时间 103.7ms)⇒ **模型 eager 下被 CPU/同步/Python 开销主导, 不是 GPU-bound**。
- 头部: `aten::copy_` **54.6ms CPU**(880 次/帧, = meshgrid `.to()`×4 + VFE/fusion 拷贝)、`cudaMalloc` 18ms、`cudaLaunchKernel` 6.7ms(**5460 次 kernel 启动/帧!**)。GPU 头部: `cudnn_convolution` 4.3ms(真卷积) + cutlass/xmma gemm ~3ms。
- ⇒ **真瓶颈 = eager 海量小算子(5460 launch/帧)+ 数百次主机-设备拷贝**, 不是单一 backbone 或 decode。剪枝**确实降 GPU 卷积时间**(22.6ms 里那 ~7ms 卷积会随剪枝降), 但被 ~40ms 的 CPU/launch/copy 开销稀释 → 全网 eager 提速被淹没(**不是零, 是被稀释**)。

### 9.2bis ★重新回答"剪枝对全网加速": 是被 eager 开销稀释, 非无效
- 剪枝在 **GPU 计算层真实加速**(backbone 1.56-1.7×); 在 **eager 全网**被 CPU/launch/copy 开销(占 wall 大头, 与剪枝无关)稀释。
- **部署口径(TRT 整管线 / 修 decode 缓存 / 融合算子)下剪枝收益会显著回升** —— 因为消掉了 5460 launch + 880 copy 的 eager 开销, GPU 计算(剪枝直接受益)才成主导。
- **用户指令遵守**: 今后**全网口径**评估, 不单报 subnet; 但结论从"剪枝全网零加速"**修正为"eager 管线下被开销稀释, 需 TRT 整管线+修 decode 才能兑现"**。

### 9.3 与 ResNet 剪枝文献对齐(用户要引用源)
"iso-budget 下剪枝精度中性偏负, 表观涨幅是训练 confound" **与文献一致**:
- **Liu et al., "Rethinking the Value of Network Pruning", ICLR 2019** — 最直接: 同预算下 prune+finetune ≈ from-scratch, 表观剪枝增益在控训练预算后消失。= 本 iso-budget 结论。
- **Renda et al., "Comparing Rewinding and Fine-tuning in NN Pruning", ICLR 2020** — LR rewinding/再退火回收甚至超基线 = 支持"第二轮退火"机理。
- **Li et al., "Pruning Filters for Efficient ConvNets", ICLR 2017** — ResNet-56/110@CIFAR 中等剪枝近无损, ResNet-110 剪后**微超基线(正则)**; 但**从无"越剪越准单调"**, 是 flat-then-cliff。我的单调上升 = confound 非真效应。
- **Fang et al., "DepGraph: Towards Any Structural Pruning", CVPR 2023** — 所用方法。
- ⚠️**不可过度对齐处**: 标准 ImageNet/ResNet 剪枝**确实**有 wall-clock 加速(224×224 compute-bound); 我"剪枝不加速 wall-clock"是本小 BEV 检测器 + decode-bound eager 管线的特性, **非反驳 ResNet 文献**。

### 9.4 接手下一步(基于此纠错, 已验证根因)
1. **修 decode 的 CPU-meshgrid(已定位, 一行级修复)**: `center_point_codriving.py` 的 `generate_predicted_boxes` 把 `ys,xs=meshgrid(arange(H),arange(W))` 预建在 GPU 上缓存(按 H,W 缓存), 别每次 `.to(device)`。实测 2.05→0.086ms/call(24×, 见 `logs/verify_bottleneck.json`)。**在隔离副本改, 勿动主框架**。
2. **消 eager 全网开销才是关键(根因 = 5460 launch + 880 copy/帧)**: 全网 wall 61.5ms 里 GPU 计算仅 22.6ms, 其余是 launch/copy/同步开销。要让剪枝兑现到全网, 须 ①整管线 TRT(VFE 稀疏算子是难点, scatter/fusion/heads/decode 可 TRT)②或 CUDA Graph 消 launch 开销(项目既有: 单 GPU 仅 1.05-1.10×, 有限)③或 batch 摊薄。
3. **剪枝全网价值的正确判法**: 在 ①修好 decode + ②整管线 TRT 后, 重测全网 base vs p50 vs p75 —— 那时 GPU 计算成主导, 剪枝的 1.56-1.7× 才会显现。当前 eager 全网平 ≠ 剪枝无效, 而是开销稀释。
4. **量化/剪枝以全网口径评估**(用户令), body 子网仅作机理诊断; 但须在 **TRT 整管线**口径下比, 否则 eager 开销掩盖信号。
5. **重跑被作废的 hook 数据**: 若需逐阶段全网占比, 用 torch.profiler(`codriving_profile.py`)不要用 forward-hook+event(已证不准)。
