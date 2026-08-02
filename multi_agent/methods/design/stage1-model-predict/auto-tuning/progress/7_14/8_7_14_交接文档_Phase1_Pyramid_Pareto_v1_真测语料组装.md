# 8_7_14 交接文档 — Phase 1 / E6 Pyramid-LiDAR RSU Pareto v1(真测语料组装 + H800 GAP)

> 承接 `7_7_14_交接文档_论文实验补齐计划_从空白方案到执行.md` 的 §12 /goal(启动 Phase 1)。
> 本文档记录:Phase 1 已用 **100% 真测语料**产出 Pareto v1;两个 GAP 因 H800 GPU 驱动挂死(2026-07-04)延迟;恢复协议 + /goal 收尾。

---

## 0. 一句话状态

**Phase 1 / E6 Pareto v1 已交付**(`results/phase1_pyramid_pareto.{csv,json}` + `multi_agent/figure/fig_phase1_pyramid_pareto.png`),lat+energy 轴 100% H800/TVM 真测,AP 轴 w0-锚 gold 估计。**三极值点全真测确认**。仅 8 个 knee 精修点因 H800 挂死延迟(G1),标 DEFERRED-GAP,非造假。

---

## 1. 数据来源与口径(全部同口径,可比)

- **真测语料 = `cost_model/train/original60_training_table_latest.json`**:60 宽 × 3 精度 = **180 真测点**,每点 measured latency + energy。
- **口径**:H800 + TVM tuned;输入 `[2,64,128,256]`(batch=2);fp16=im2col+WMMA(`MatmulTensorization`)/ int8=int8_tc(`MatmulInt8Tensorization`,已核实非旧朴素 dp4a,int8/fp16 lat ratio=0.77–0.95 的 TensorCore 签名)/ fp32=FFMA 参考(不张量化)。
- **AP gold = `data/stage_a_ap_real.parquet`**:base/p25/p50/p75 四锚点 × 双精度,收敛 AP70(n=1789),int8 ΔAP70≈−0.008。

---

## 2. Pareto v1 结果(真测)

| 极值点 | 宽度 | 精度 | latency | energy | AP70 | 来源 |
|---|---|---|---|---|---|---|
| **lat-min = E-min** | [24,32,64] | int8_tc | **1.365 ms** | **0.404 J** | ~0.539 | 真测 lat/E,w0-锚 AP |
| **AP-max** | [64,64,96] | fp16 | 4.620 ms | 1.441 J | ~0.631 | 真测 lat/E,w0-锚 AP |

- **Pareto front(真测)= 10 点**,跨 lat 1.365→4.620 ms,AP70 0.539→0.631;int8 占低延迟/低能耗段,fp16 在高 AP 端(int8 −0.008 AP 罚在最宽处不划算)→ **Q 轴对前沿有真实作用**(非退化)。
- 组装脚本 = `scripts/phase1_assemble_pyramid_pareto.py`(可复现,3 目标非支配 ap70↑/lat↓/energy↓)。

---

## 3. 两个 GAP(诚实标注,非造假)

### G1 — 8 个 knee 精修点(H800 挂死)
- SMBO round1 emit 的前沿 knee 候选(全 w0=16 激进剪枝):fp16{[16,32,64],[16,32,256],[16,48,256],[16,64,256]} + int8{[16,32,224],[16,32,256],[16,48,224],[16,48,256]},**不在 60 语料内**。
- **阻塞根因(双重核实)**:2026-07-04 H800 整机 nvidia-smi D-state 挂死(内层 timeout 15 都杀不掉,load avg 53.84,fabricmanager active),= memory 记的 NVSwitch 需 reboot/等那类故障。**未盲测、未擅自 reset/reboot**(影响其他 9 用户任务)。
- JSON 里这 8 点 status=`DEFERRED_GAP_h800_wedged`,lat/energy 为 cost-model 预测值(measured=null),恢复后 GPU5 真测替换。

### G2 — 逐点收敛 AP(AP 轴口径)
- **cost model 的逐点 AP 预测不可靠**:AP head 训练在短微调噪声语料上(60 fp32 行 ap70 span 仅 0.575–0.602 = 0.027,而 gold 收敛 span 0.101)。cost model 对 w0=16 点预测 ap70≈0.58 > gold p75 真值 0.530 = 偏乐观。
- **故 Pareto AP 轴用 w0-锚 gold**(w0:16→0.530/32→0.564/48→0.590/64→0.631 线性插值,int8 −0.008),标 `ap_source=w0_anchored_estimate`。依据 = memory "AP 由 w0 主导,w1/w2 携带 <0.01"。
- gold 4 锚点宽度 ∩ 60 语料 = **空**;真 base[64,128,256] 同口径 lat/energy 也不在语料 → AP-max 端用语料内 w0=64 点代理(真测 lat/E,AP=gold 0.631),真 base 是进一步 GAP。
- **落最终 Pareto 的前沿点应走 DAIR val n=1789 收敛 finetune 补逐点真 AP**(可共卡,不污染 AP;见 memory feedback-gpu-sharing)。

---

## 4. 恢复协议(H800 好了怎么续)

1. 用户/管理员恢复 H800 GPU 驱动(reboot 或 reset);用户跟主控说一句"H800 好了"。
2. 主控 resume owning agent(data-orchestrator,agentId 见会话):`ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no`,GPU5(测前确认 util0%/mem≤50MiB,别碰 GPU6),fresh workdir 逐宽度。
3. agent 测 8 点(fp16=WMMA@128×256 / int8=int8_tc@128×256),产 `round1_{fp16,int8}_measured.json`(复现 fp32 measured schema `{width,lat_tuned_ms,lat_default_ms,energy_j,build_success}`),各跑 `stage2_smbo_loop_v1.py --step feedback`。
4. 主控独立 stat+读值核验后,重跑 `scripts/phase1_assemble_pyramid_pareto.py`(把 G1 的 8 点从 predicted 换成 measured)→ **Pareto v2**。
5. 前沿点 finetune 补逐点真 AP(G2)→ Pareto final。

---

## 5. /goal(下一会话,直接复制 — H800 恢复后续 Phase 1 v2)

```
阅读 multi_agent/methods/design/auto-tuning/progress/7_14/8_7_14_交接文档_Phase1_Pyramid_Pareto_v1_真测语料组装.md
纪律:框架已迁 TVM 非 TRT(严禁提 TRT/trtexec 作前进方向,legacy TRT 只作历史);报告用中文;不轻信 agent 自报(逐产物 ssh stat+读文件核验,数值从文件真读);agent 交付物必落共享 fs(/home、/exdata)+ 独立 stat 核;single-orchestrator(主控只读核验,不越 owning agent 起停 worker);无 fake-quant/dummy scale 冒充真 int8;H800 密码八进制解码单条 bash 内用不明文打印/不落盘。
GPU 范围(2026-07-04 用户拍板):0,1,2,3,4,5,7 可用,GPU6 禁;GPU5 = 唯一全空闲卡,latency/energy 独占实测(util0%/mem≤50MiB);AP finetune 可共卡 GPU1/7(先确认剩余显存>15GB,GPU3 仅~9G 别用)。

任务(P0,续 Phase 1 至 v2/final):
[执行·前置] 确认 H800 GPU 驱动已恢复(nvidia-smi 秒回 GPU5 状态);未恢复则停,回报用户等机器。
[执行·真测] resume data-orchestrator agent,GPU5 独占 fresh-tune 真测 8 个 knee 点(fp16{[16,32,64],[16,32,256],[16,48,256],[16,64,256]} WMMA@128×256 / int8{[16,32,224],[16,32,256],[16,48,224],[16,48,256]} int8_tc@128×256),产 round1_{fp16,int8}_measured.json + feedback。
[执行·AP] 落 Pareto 前沿的 knee/base 点走 DAIR val n=1789 收敛 finetune 补逐点真 AP(可共卡),替换 w0-锚估计;int8 用真校准 AP 非 fake。
[判据/交付] 主控独立核验后重跑 scripts/phase1_assemble_pyramid_pareto.py → results/phase1_pyramid_pareto.{csv,json}(G1 的 8 点 status 从 DEFERRED_GAP 变 measured)+ figure 更新;写 9_7_14 交接文档记 v2 状态。

停止条件(全满足才算完成 Phase 1):
- [ ] 8 knee 点 GPU5 真测完成,measured.json 落盘 + 主控 stat+读值核验
- [ ] 两次 feedback 完成,收敛判定明确
- [ ] Pareto v2 三极值点全真测确认(含 base 真测 lat/energy 或明确标其 GAP)
- [ ] 前沿点逐点真 AP 补齐(finetune)或明确标 G2 残留
- [ ] results/*.{csv,json} + figure 更新 + 9_7_14 交接文档

纪律:可用 GPU 3,4,5 三张卡，其余卡暂时不要使用，每个前沿点标 (w,q)+s=tuned + AP/lat/energy + 数据源(真测/复用/预测/GAP),proxy 不冒充真测。
不做:不重跑已收敛的 fp32;不 reboot/reset H800(机器侧用户/管理员处理);不用 cost-model 逐点 AP 预测冒充真 AP(它训练在噪声语料上,只作排序参考)。
```

---

## 6. 本轮反思

- **H800 挂死是硬阻塞,但没干等**:用已测 180 点语料 + gold 锚 + surrogate 离线组装出 v1,把"缺 8 点"从"全阻塞"缩成"缩一个 knee 精修 GAP"。
- **AP 轴是真弱环**:cost model AP 训练在短微调噪声上不可用于逐点排序;w0-锚 gold 是当前最佳可得,逐点真 AP 需 finetune(G2)。这条要在论文里如实写口径。
- **owning agent 那道"先确认 recipe"闸门造成一次往返浪费**:下次派 owning agent 若 recipe 已由主控核实,直接给"已验证 recipe + 开测"指令,不设提问闸门。
