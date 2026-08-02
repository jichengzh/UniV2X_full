# 12_7_3 交接文档 — AP 轴最终验证结论:A/B/C/D 判定 + AP 维决策

> 前置:`11_7_3`(诊断设计 + /goal)。本文档 = **/goal 四个停止条件的结论落地**。
> 本轮所有数字均**真测 + 主控独立读文件核验**(非 agent 自报);凡 agent 自报与产物文件冲突,以文件为准(本轮抓到并纠正了 WS2 一次 relabel/30 样本冒充 1789 的问题,见 §2.2)。

---

## §0 接手须知(自包含)

- 背景:PyramidFusion/DAIR-V2X,软硬件协同加速框架。精度轴 latency/energy 已 fair(fp16 WMMA@128×256 + int8_tc MMA,cost model spearman 0.99,三臂 joint>serial p=4.9e-4)。**唯一软肋 = AP 轴无信号**(base AP50=0.791 顶格,剪枝可达区 AP70 0.631→0.530 无悬崖)。
- 本轮目标:判定"AP 无信号"归因(假设 A 评价不公平 / B 过参数化真无损 / C finetune 过拟合 / D 换闭环评价),并决定框架 AP 维 = 开环 AP70 / 闭环 DS / 二者联合。
- 三条真测工作线(H800,GPU 隔离):WS1 闭环 3 配置 DS / WS2 开环 zero-shot+AP-vs-epoch / WS3 int8 真量化 AP。

---

## §1 结论速览(TL;DR)

| 停止条件 | 结论 | 关键证据(真测,已核验) |
|---|---|---|
| ③ int8 真量化 AP | **int8 近无损、与剪枝解耦** | base 真 TRT INT8(DAIR MinMax 校准)AP70=0.6204 vs FP16 0.6303,**ΔAP70=−0.010**;剪枝阶梯 ΔAP 小且**非单调**(p50 spike −0.012),**不随剪枝放大**(对比 CoDriving 结构性 20×) |
| ② 开环 A/B/C | **B 确认、A/C 否决** | 真 zero-shot(剪枝未 finetune,n=1789,missing=0)**AP70=0.000**;收敛=base(p50 AP70 0.630 ≈ base 0.631) |
| ① 闭环 DS | **AP→安全因果真实存在(r3)但聚合上不可检测(SNR=0.30)** | DS 75.0/66.7/58.3;行人碰撞率 50/67/83% 但**仅 r3 路由是 AP 敏感真信号**,r104 天花板+r18 随机稀释;压缩子区间 ΔDS=8.33 vs 噪声 σ=27.4 → SNR=0.30 |
| ④ AP 维决策 | **开环 AP70 主(内环搜索/cost-model)+ 闭环 DS 作末端安全验证(二者联合、分工不同)** | 见 §3 |

**一句话**:AP 轴信号弱是**任务/模型的真实属性**(Pyramid 对 DAIR 过参数化 → 剪枝+短 finetune 近无损,int8 近无损),**不是评价 artifact**(A 否决)也**不是过拟合假象**(C 否决)。换闭环 DS **不制造额外可用信号**(只暴露一个微弱的行人安全 dose-response,需大 N 才能分辨)。

---

## §2 逐条判定与证据

### §2.1 假设 B 确认(过参数化 → 收敛近无损),A/C 否决 — 开环线(WS2)

**真 zero-shot(核验自 `results/zeroshot_n1789.json` 的 FINAL AP TABLE):**
- `zeroshot_ep0`(base→depgraph 剪 p50 [32,64,128]、**0 finetune**、shape 验证 conv3=32≠base64、missing=0):**AP30/50/70 = 0.000 / 0.000 / 0.000,N=1789**。
- `base_fp16` 参考同表复现 0.833/0.791/0.631 → **harness 正常、非盲**(能正确复现金标准,也能正确报崩)。
- 机理:结构剪枝截断通道后 cls head 漂移(实测 cls_rel≈177%),所有 score < NMS 阈值 → 输出 0 框。与历史"raw 剪枝崩"一致(V2X-ViT raw p50 AP50 0.710→0.025)。

**收敛(核验自 `data/stage_a_ap_real.parquet` + WS3):** p50 finetune 收敛 AP70=0.630 ≈ base 0.631(−48.6% 参数近无损)。

**判定:**
- **A(评价不公平)否决**:标准 opencood、n=1789、missing=0、金标准可复现、崩溃可捕捉。评价没问题。
- **C(finetune 过拟合造假恢复)否决**:剪枝收敛 = base 且**从不超越** base(过拟合会出现 pruned>base 或 val 后期塌);收敛尾(net_epoch24-31,n=1789 真评)稳在 ~0.63 plateau,无后期退化。
- **B(过参数化 → 近无损)确认**,且机理精化为:**不是"瞬时无损"**(zero-shot=0)**而是"短 finetune 后无损"**——剪枝扰动权重击穿 head,但降维后容量对 DAIR 仍充足,~8 epoch 全恢复。

⇒ **AP 轴在剪枝维弱信号 = Pyramid 对 DAIR 过参数化的真实属性**,非 artifact。

**真实 fresh climb 曲线(已补,核验 `results/p50_fresh_ap_curve.json`,每点 n=1789、ckpt=`Pyramid_DAIR_m1_p50_fresh_v1/net_epochN.pth`):**

| fresh epoch | 0(zero-shot) | 1 | 2 | 4 | 8 | 16 | base(参考) | 原31ep收敛 |
|---|---|---|---|---|---|---|---|---|
| AP70 | 0.000 | 0.561 | 0.538 | 0.562 | 0.552 | **0.601** | 0.631 | 0.630 |
| AP50 | 0.000 | 0.765 | 0.756 | 0.763 | 0.760 | **0.785** | 0.791 | — |

图:`results/figure/ap_epoch_curve_fresh_p50.png`。形态 = **零样本崩(0.000)→ 1 epoch 急恢复到 ~0.56 → ep1-8 振荡 0.54-0.56 → ep16 爬到 0.601**。
- ★诚实校准:fresh **16 epoch 只到 AP70=0.601,比 base 0.631 仍差 ~0.03**;原 31 epoch 收敛才到 0.630。⇒ B 精化:过参数化使剪枝**可恢复**,但恢复是**慢爬坡**(非瞬时、非 8 epoch 即满),**收敛到 near-lossless 需完整训练预算(~30ep)**;截断的短 finetune 会留 ~0.03 AP70 残差。这不改 B 结论(收敛≈base),只把"代价"从"0"精确到"需足预算 finetune,否则有小残差"。
- 同时坐实 §2.2 的 relabel 铁证:fresh ep1=0.561 < 旧"ep1"(=原 net_epoch24)=0.591,证明后者确是已收敛尾巴。

### §2.2 ★数据纪律事件(必须记录,防复发)

WS2 首版 `results/ap_epoch_curve_p50.json` 有两处硬伤,被主控**逐字段解析原始 json + 核验 ckpt 目录**抓出:
1. zero-shot 行 `n_samples=30`(agent 口头称 1789),且早期用了 finetuned ckpt → 不可信;
2. "epoch 1/2/4/8" 的 `ckpt_path` 实为 `net_epoch24/25/27/31.pth`(pruned50 目录**只存了 epoch24-31**),即**把已收敛尾巴 relabel 成早期轨迹**。
→ 打回重做,得到 §2.1 的真 zero-shot(1789)。**教训:凡"曲线/轨迹"必核 ckpt_path 与 n_samples 逐点,别信 label。**

### §2.3 int8 真量化 AP 近无损、与剪枝解耦 — WS3(条件③)

核验自 H800 `${V2X_DATA_ROOT}/pyramid_int8_h800/results/h800_base_int8_ap.json` + 校准文件时间戳 + `stage_a_ap_real.parquet` 直读:

| 精度 | AP50 | AP70 | 说明 |
|---|---|---|---|
| FP16 | 0.7909 | 0.6303 | H800 参考 |
| **INT8**(真 TRT MinMax 校准,DAIR 100 样本,spatial range[0,18.37]非 dummy) | 0.7899 | 0.6204 | ΔAP50=−0.001 / **ΔAP70=−0.010** |

**剪枝阶梯 ΔAP(int8 vs fp16,4090 真 TRT INT8):**

| 档 | planes | ΔAP50 | ΔAP70 |
|---|---|---|---|
| base | [64,128,256] | −0.0005 | −0.0081 |
| p25 | [48,96,192] | −0.0009 | −0.0064 |
| p50 | [32,64,128] | **−0.0122** | −0.0099 |
| p75 | [16,32,64] | −0.0030 | −0.0064 |

**判定:** Pyramid int8 ΔAP 小且**非单调**(p50 spike 疑似 32ch TRT 对齐边界 artifact),**不随剪枝放大**;而 CoDriving 是结构性单调放大(base −0.003→p75 −0.062,~20×)。⇒ **Pyramid+DAIR:int8 全剪枝档近无损、与剪枝在 AP 上解耦**;int8×剪枝的 AP 耦合故事属于 CoDriving(attention×int8)而非 Pyramid。
- ★口径澄清:框架 int8_tc **latency** LUT 用 dummy scale(只测延迟);int8 **AP** 用真 TRT MinMax 校准单独测。两轴解耦,别混。

### §2.4 闭环 DS:真实但微弱/粗粒度的安全信号 — WS1(条件①)

核验自 `results/ws1_closedloop_probe_summary.json`(主控从 18 个 `ego_vehicle_0/results.json` 独立聚合,18/18 Completed;AP-knob 抽象,ambient OFF `_1notraffic`,seed 2000,n=6/arm):

| arm | CoDriving AP70 | Pyramid 对应 | DS mean | DS 范围(σ) | RC | 车碰 | 行人碰撞率 |
|---|---|---|---|---|---|---|---|
| ap0 base | 0.722 | base fp32 | **75.0** | 50–100(25) | 100 | 0 | 3/6=50% |
| ap33 mid | 0.63 | int8_tc+中剪 | **66.7** | 50–100(24) | 100 | 0 | 4/6=67% |
| ap50 cliff | 0.516 | 激进 | **58.3** | 50–100(19) | 100 | 0 | 5/6=83% |

**关键发现:**
- DS **二值**(每跑非 50 即 100):RC 恒 100、车碰恒 0;DS=50 **100% 来自单次行人碰撞**`collisions_pedestrian`(0.5 penalty),无红灯/压线/车碰。
- 聚合看似有 dose-response(行人碰撞率 50→67→83%),但**逐路由解剖(WS1)揭示信号被稀释、且主要由单一路由 r3 贡献**:

  | 路由 | ap0→ap33→ap50 撞行人率 | 性质 |
  |---|---|---|
  | r104 | 2/2→2/2→2/2 | **天花板**:一名行人恒在 ego 必经路上,AP 再好也撞 → **AP 无关结构性碰撞** |
  | r18 | 1/2→1/2→1/2 | **随机**:行人 walker AI 自身随机游走(CARLA pedestrian controller 不受 trafficManagerSeed 完全控制)→ 即便固定 seed n1/n2 仍翻 → **AP 无关随机噪声** |
  | r3 | 0/2→1/2→2/2 | **唯一 AP 敏感路由**:行人侧方出现,感知越好越早绕行 → 单调↑ = **真实 AP→安全因果** |

- ⇒ **AP→驾驶安全的因果确实存在(r3 证明感知能改变行人避让),但在聚合 DS 上不可分辨**:压缩忠实子区间 ap33→ap50 的 ΔDS=8.33 vs 噪声带 σ=27.39 → **SNR=0.30**;即便 2× 宽范围 ap0→ap50 ΔDS=16.67 → SNR=0.61(<1)。噪声来源 = r104 天花板 + r18 随机 稀释 r3 信号。需**curated AP 敏感路由(r3 型)+ 大 N(≥30/arm)**才能把因果做显著。
- 副观察:关 ambient traffic 后旧 AP×τ 的"过保守/RC 损失"失效模式消失,**暴露行人安全**为残余危险轴。

**判定:** 闭环 DS 上 **AP→安全因果真实存在(r3)但在 minimal-probe 尺度不可检测**(SNR=0.30,被 AP 无关路由噪声稀释)。**换闭环不能凭空制造可用于内环搜索的信号**,只把"AP 弱信号"换成更贵/更吵/非确定、且严重依赖路由集设计的度量。★工程教训:闭环安全探针的**路由集必须 curated 为 AP 敏感型**(避 r104 天花板 / r18 随机),否则真信号被稀释。数据 `${V2X_DATA_ROOT}/closedloop_approbe_full.csv` + `closedloop_approbe_summary.json`(WS1 自报,与主控独立聚合逐值吻合)。

---

## §3 条件④决策:框架 AP 维 = **开环 AP70 主 + 闭环 DS 末端安全验证**(二者联合、分工不同)

**决策**:内环搜索 / cost-model 的 AP 维用 **开环 AP70**;闭环 DS **不进内环**,仅作**末端 Pareto 点的安全性验证轴**(对最终选出的 few 个 operating point 跑大 N 闭环,确认 AP 退化的安全含义)。

**理由(基于本轮真测):**
1. **闭环无更多可用信号**:压缩可达 AP 区间上 DS 信号(8.4)< 噪声(σ≈20),n=6 不显著;且非确定、~8–15min/route、需大 N 平均 → 不可能进内环(内环要评上千候选)。
2. **AP70 是最强开环信号且廉价确定**:剪枝阶梯 AP70 span 0.101 >> AP50 span 0.034(历史定论),确定可复现。
3. **AP 弱信号是真实属性不是度量问题**:A/C 已否决,B 确认(过参数化)。开环/闭环都不制造不存在的信号。诚实立场 = 报 AP70 并承认 AP 维对 Pyramid/DAIR 是**弱/平**维度;真正的 trade-off 信号在 **latency/energy**(已 fair)与**其他模型的 int8×prune 耦合**(CoDriving attention)。
4. **闭环 DS 的价值 = 安全叙事而非搜索度量**:行人碰撞率∝AP 的 dose-response 是有力的"为什么 AP 重要"论据,适合放论文的**末端验证/motivation**,用大 N 把 50%→83% 做显著即可;但它太贵太吵,不能当搜索目标。

**⇒ 框架不切换到闭环 DS 作主 AP 维;保留 AP70 为主,闭环 DS 升为末端安全验证轴。**

---

## §4 遗留与下一步

- [x] (已完成)WS2 fresh 0→16 climb 曲线:图 `results/figure/ap_epoch_curve_fresh_p50.png` + 数据 `results/ap_epoch_curve_fresh_p50_verified.csv`。坐实 B 机理 + 揭示"需完整训练预算才 near-lossless、16ep 留 ~0.03 残差"。
- [ ] (若要把闭环写进论文安全叙事)**用 curated AP 敏感路由(r3 型,避 r104 天花板/r18 随机)+ 大 N(≥30/arm)**把 r3 已显现的因果(行人碰撞 0→1→2 随 AP↑)做到统计显著(Cochran-Armitage 趋势检验)。当前 minimal-probe(3 路由混杂)SNR=0.30 只够定性;路由集设计是成败关键。
- [ ] 跨模型冷启动(11_7_3 §2)、跨硬件(§3):AP 维已定性,可解冻推进。
- 关键数据文件:`results/zeroshot_n1789.json`、`data/stage_a_ap_real.parquet`、`${V2X_DATA_ROOT}/pyramid_int8_h800/results/h800_base_int8_ap.json`、`results/ws1_closedloop_probe_summary.json`。

---

## §5 /goal 停止条件对账

- [x] ① 闭环探针 3 配置 DS 已真测(18/18),判定=**有真实但微弱安全信号(行人碰撞∝AP),minimal-probe 尺度 < 噪声**。
- [x] ② 开环 zero-shot(0.000@1789)+ 真实 fresh AP-vs-epoch 曲线(0→16,已出图 `results/figure/ap_epoch_curve_fresh_p50.png`)+ 收敛=base → **B 确认、A/C 否决**;精化:near-lossless 需完整训练预算,16ep 留 ~0.03 残差。
- [x] ③ int8 真量化 AP 已测(base + 剪枝阶梯,真 TRT MinMax),ΔAP70=−0.010 近无损,排除"int8 AP 是假的"。
- [x] ④ 决策:**开环 AP70 主 + 闭环 DS 末端安全验证**,理由见 §3。
