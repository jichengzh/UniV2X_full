# 3_7_14 交接文档 — 下一步实验计划(三线)

> 前置: `2_7_14`(本轮结论: A严IoU收敛不放大/B欠训AP80@e2可排序smoke/目标二冷启动AP70有价值) + `12_7_3`(AP维定论)
> 日期: 2026-07-04。用户看过2_7_14后给了三点方向 + 一个口径纠错, 据此制定下一步。

## §0 ★口径纪律(用户强调"一直记住")

**框架已全面迁移到 TVM。所有下一步/int8/张量化/latency 口径一律用 TVM(MetaSchedule/dlight + topi/tir 张量化), 不再提 TRT。**
- 历史上的 TRT 数据(如 base int8 AP70=0.6204 来自 TRT engine)是**迁 TVM 前的 legacy, 仅历史参考, 不作框架现役口径**。
- int8 现役路 = TVM int8_tc(dp4a / WMMA-int8 真张量化); fp16 现役路 = im2col + WMMA。
- 见 memory [[feedback-framework-on-tvm-not-trt]]。2_7_14 §1/§6 的 TRT 提法已更正。

---

## §1 计划一 — AP80@e2 满60点实验(判 AP 信号是否真存在, 非仅smoke)

**动机**: 2_7_14 方向B 只在 4 点阶梯(base/p25/p50/p75)smoke 出"(AP80, e2)可排序, spread 0.049≈10×噪声"。**这只是 smoke**。要判框架 AP 轴是否真有可用信号, 须在**完整 original60 配置空间**上用同协议测。

**协议**:
- **评价指标 = AP80**(2_7_14 证明它比 AP70 在欠训区间更可分)。
- **微调预算 = e2**(2 epoch fresh finetune from pruned-init; e4起阶梯塌缩错序, e2是信号窗口)。
- **样本 = DAIR val n=1789**(每点必真测, 逐点核 ckpt_path + n_samples, 不冒充)。
- **配置集 = original60**(Pyramid cost-model 的 60 个 width/prune 配置; 见 `cost_model/train/original60_training_table_latest.*` 的配置定义)。

**执行**:
1. 对 60 个配置各自: 从 base 按该配置 DepGraph 剪枝→pruned-init(flat state_dict)→fresh finetune 2ep(train_ddp_patched.py 单卡, lr0.002, missing_keys=0)→eval AP80@e2(用 `tools/eval_ap_epoch_curve_iou.py`, n=1789)。
2. 附 e0(pruned-init zero-shot, 崩点=0 确认机理)作对照; 有余力加 e4 看塌缩。
3. 复用已有: p25/p50/p75 的 fresh_v1 已到 8ep(可直接取 e2 点); base 收敛参考已有。只需补 original60 里其余 ~57 个配置。

**判据(要回答的)**:
- **AP80@e2 在完整60点上是否给出 spread >> 噪声(~0.005)的可排序信号?** 若是, 框架 AP 轴 = AP80@e2-budget-capped(recoverability 语义)。
- **AP80@e2 的 config 排序是否与某个有意义量单调相关?**(与剪枝率? 与收敛AP? 与参数量?)—— 明确它测的是 recoverability 而非收敛AP(2_7_14 nuance), 但要看这个 recoverability 信号在全空间是否稳定单调、可作内环 cost-model 的 AP label。
- 若60点上信号塌成噪声(不排除, 因 original60 的 width 变化不一定像 prune 阶梯那样单调伤 recoverability)→ 则 AP 轴放大失败, 回落 AP70@收敛 + 闭环DS末端(12_7_3 定论)。

**成本**: ~57 配置 × 2ep 训练(单卡~8-10min/2ep 小模型)+ eval, GPU0-6 并行 ~2-3h。★真测纪律: 空闲GPU、逐点核 ckpt+n_samples、e0 崩点标注(p75式 n可小但标)。

---

## §2 计划二 — 跨模型冷启动: 值预测 + **序预测**(用户纠正)

**用户点(认可+补强)**: latency 值预测(value MAPE)误差大我认可(模型结构确实不同); **但我们的预测是"值预测+序预测", 冷启动数据在序预测(rank)上应该有帮助**。2_7_14 只报了 value MAPE(latency zero-anchor 564%→253%), 漏了 rank 维度——这正是搜索/优化框架真正关心的(cost-model 只要把配置**排对序**就能引导搜索, 绝对值偏差可容忍)。

**下一步**:
1. **对 latency 和 AP 都补 rank-prediction 指标**: Spearman ρ + Kendall τ(预测序 vs 真序), zero-anchor vs +anchor(留一)。
2. **关键假设待验**: 即便 Pyramid-only 预测器的 CoDriving latency **绝对值**偏(grouped vs 标准conv ~5-7×系统性偏移), 它的**排序**可能仍对(剪枝率↑→延迟↓的单调性跨模型共享)→ 若 zero-anchor rank ρ 已高, 说明冷启动在 latency **序**上本就迁移, value 偏移可用少量 anchor 校准(仿射校正)。反之若 rank 也乱, 才需 anchor 补序。
3. **报**: value(MAPE)与 rank(ρ/τ)分开, zero-anchor / +anchor 各一组, latency 和 AP70 各一份。给"冷启动对值/序各自的价值"分离结论。
4. 数据: 复用本轮已核验真数据(cod_e2e.csv latency 4点 + codriving_pilot AP70 4点 + Pyramid original60 表); n 小(4 CoDriving 点)→ rank 统计弱, 需配合计划三补更多 CoDriving 真点(fp16-TC/int8-TC latency)增大 n 后 rank 才稳。

**产物**: `coldstart_rank_report.json`(value+rank 双维, 分 target)。

---

## §3 计划三 — 为 CoDriving 量身建 TVM Tensor-Core lowering 路(fp16 **且** int8)

**用户点(推翻我此前"scope B缓做")**: 标准 conv 用不上 Pyramid 的 grouped-conv TC 路, 确实如此 → **必须为 CoDriving 量身建能 lower 到 Tensor Core 的 TVM 路; 这是证据回流(feedback)前必做的, 不止 fp16, 还有 int8**。

**为什么现在必做**:
- 本轮 CoDriving latency 的 fp16≈fp32(标准conv无自动TC)= **CoDriving 根本没吃到 Tensor Core**, 现有 latency 是"无TC基线", 不公平也不是框架该展示的加速。
- int8-TC whole-model 本轮标 GAP(仅 stage0 micro)→ CoDriving 的 int8 加速轴 + int8×剪枝真耦合(vs Pyramid 解耦)**未在 TVM 上真测**, 证据回流前这条腿缺失。

**技术路径(全 TVM, 非 TRT)**:
1. **fp16-TC**: 给 CoDriving 标准 NCHW conv 建 im2col→matmul→**WMMA 张量化**路。复用 dlight `MatmulTensorization`(fp16, `tvm.s_tir.dlight.gpu.matmul`), 但适配标准 conv(非 group)。参考 Pyramid 的 full-engine-group-conv-rewrite probe, 出标准conv变体。
2. **int8-TC**: 同结构换 `MatmulInt8Tensorization`(int8→int32, get_wmma_intrin_group in_dtype=int8; 见 [[project-precision-axis-double-unfairness]] 已验证该 fork 有此 pass, 闸门脚本 `scripts/stage2_int8_tensorcore_gate_v1.py`)。需 im2col 折进 matmul shared-mem load(全引擎融合)才拿网络级加速, 非单块微基准(微基准 memory-bound 反慢)。
3. **验收**: CoDriving whole-model fp16-TC / int8-TC 真 build + tensorize(报 mma_sync/wmma 计数)+ 网络级 latency vs 无TC基线(fp16 8.06ms / naive)。int8-TC 再测真 AP(int8×剪枝耦合 base→p75, 对照 Pyramid 解耦 + legacy TRT delta base-0.004→p75-0.049 复现)。

**产出解锁**: ①CoDriving 公平 latency(吃到TC)②int8-TC 真加速轴 + 真AP ③补全12-anchor 的 fp16-TC/int8-TC latency 列(本轮 GAP)④int8×剪枝真耦合(TVM口径)⑤计划二 rank 统计的更多真点。

**成本**: 数天 TVM schedule 工程(标准conv→TC模板), 但用户已定为证据回流前置必做。

---

## §4 优先级 / 执行顺序 / 资源

| 优先级 | 计划 | 依赖 | GPU |
|---|---|---|---|
| P0 并行 | 计划一 AP80@e2 满60点 | 无(工具/协议已就绪) | GPU0-3 并行训eval |
| P0 并行 | 计划三 CoDriving TVM-TC路 | dlight pass 已有, 需schedule工程 | GPU4-6 tune |
| P1 依赖 | 计划二 值+序预测 | 计划三补 CoDriving 真点后 rank 才稳 | 轻(CPU预测器) |

- 计划一/三可立即并行起(不同GPU、不同栈)。计划二先用现有4点出 rank 初值, 待计划三补点后定稿。
- ★纪律: 空闲GPU实测; 逐产物读文件核验(agent交付物必落**共享fs**+独立stat, 见 [[feedback-agent-bash-sandbox-overlay]]); 区分真测/proxy; **全程TVM口径不提TRT**([[feedback-framework-on-tvm-not-trt]])。

## §5 /goal(下一会话, 完整版 — 直接复制)

```
/goal 阅读 multi_agent/methods/design/auto-tuning/progress/7_14/3_7_14_交接文档_下一步计划_AP80e2满60点_冷启动值序双预测_CoDriving_TVM_TC路.md 了解背景(§0是TVM口径纪律必读)。本轮三线并行推进,所有实验真测(空闲GPU实测、区分真测/proxy、逐点核ckpt_path+n_samples、agent产物必落共享fs+独立stat核、不信agent自报)。★全程TVM口径,严禁提TRT/tensorrt/trtexec作前进方向(框架已迁TVM;legacy TRT数据只作历史参考)。

计划一(P0,AP信号真伪判定):判"AP80指标 + e2微调预算"能否在完整original60配置空间给出可排序AP信号(2_7_14只在4点阶梯base/p25/p50/p75 smoke出AP80@e2 spread 0.049≈10×噪声,仅smoke)。
- [执行] original60的60个width/prune配置,各从base DepGraph剪枝→pruned-init(flat state_dict)→fresh finetune 2ep(train_ddp_patched.py单卡,lr0.002,resume pruned-init,missing_keys=0)→eval AP80@e2 n=1789(tools/eval_ap_epoch_curve_iou.py,一次推理覆盖IoU{50,70,80,90}+报n_tp/n_gt)。复用已有:p25/p50/p75 fresh_v1的e2点+base收敛参考,只需补其余~57配置。附e0(pruned-init zero-shot崩点=0对照,n可小但如实标注不冒充1789)。
- [判据] AP80@e2在完整60点上spread是否>>噪声(~0.005)可排序? 排序与何量单调(剪枝率/收敛AP/参数量)? 明确它测的是recoverability-under-budget非收敛AP(2_7_14 nuance)。信号成立=框架AP轴=AP80@e2-budget-capped(须标recoverability语义);塌成噪声=AP轴放大失败,回落AP70@收敛+闭环DS末端(承12_7_3定论)。

计划二(P1,冷启动值+序双预测):补rank预测维度(用户纠正:预测=值预测+序预测,冷启动数据应在序预测上有帮助)。
- [执行] 对latency和AP70都算Spearman ρ + Kendall τ(预测序vs真序),zero-anchor(Pyramid-only) vs +anchor(留一)。value维仍报MAPE。
- [判据] 即便latency值MAPE大(grouped vs标准conv结构不同已认可),排序是否跨模型迁移(剪枝率↑→延迟↓单调性跨模型共享)? zero-anchor rank ρ已高=序本就迁移,value偏移用少量anchor仿射校正;rank也乱=需anchor补序。分latency/AP70、zero/+anchor报 值(MAPE)+序(ρ/τ) 双维,给"冷启动对值/序各自价值"分离结论。n小(4 CoDriving点)rank统计弱→待计划三补fp16-TC/int8-TC真点增大n后定稿。

计划三(P0,CoDriving TVM Tensor-Core路):为CoDriving标准conv量身建能lower到Tensor Core的TVM路,fp16且int8(用户定为证据回流前置必做;标准conv用不上Pyramid的grouped-conv TC路)。
- [执行] ①fp16-TC:标准NCHW conv im2col→matmul→WMMA张量化,复用dlight MatmulTensorization(tvm.s_tir.dlight.gpu.matmul)适配标准conv(非group),参考Pyramid full-engine-group-conv-rewrite probe出标准conv变体。②int8-TC:同结构换MatmulInt8Tensorization(int8→int32,get_wmma_intrin_group in_dtype=int8;闸门脚本scripts/stage2_int8_tensorcore_gate_v1.py),im2col折进matmul shared-mem load做全引擎融合(非单块微基准,微基准memory-bound反慢)。
- [验收] CoDriving whole-model fp16-TC/int8-TC真build+tensorize(报mma_sync/wmma计数)+网络级latency vs 无TC基线(fp16 8.06ms/naive);int8-TC再测真AP(int8×剪枝耦合base→p75,对照Pyramid解耦 + 复现legacy参考delta base-0.004→p75-0.049)。
- [解锁] CoDriving公平latency(真吃到TC)+ int8真加速轴+真AP + 补全12-anchor的fp16-TC/int8-TC latency列(本轮GAP)+ int8×剪枝真耦合(TVM口径)+ 计划二rank统计更多真点。

停止条件(全满足即STOP,写4_交接文档):
- [ ] 计划一:original60全60点AP80@e2已测(逐点n=1789+ckpt_path核,e0崩点如实标),给"(AP80,e2)可排序信号是否在完整配置空间成立"定量判定(spread vs噪声0.005 + 与何量单调 + recoverability语义);据此定框架AP轴口径。
- [ ] 计划三:CoDriving whole-model fp16-TC + int8-TC 真build+tensorize(报mma_sync)+ 网络级latency vs无TC基线加速比 + int8-TC真AP(base→p75耦合)。
- [ ] 计划二:冷启动 值(MAPE)+序(Spearman/Kendall)双指标报告,分latency/AP70、zero/+anchor;给"冷启动对值/序各自价值"结论。
- [ ] 决策:框架AP轴最终口径(AP80@e2-budget vs AP70@收敛)+ 跨模型冷启动价值(值/序分离)+ CoDriving是否真吃到TC加速,附证据。

纪律:空闲GPU实测(GPU7=wuyuegao勿用,跑前nvidia-smi确认util 0%/mem≤50MiB);各线并行不同GPU(计划一GPU0-3训eval / 计划三GPU4-6 TVM tune / 计划二轻量CPU预测器);不信自报,逐产物读文件核验;agent交付物必落共享fs(H800 /home、/exdata)+独立stat核(防sandbox overlay坑,见[[feedback-agent-bash-sandbox-overlay]]);修好的launcher别再踩train_ddp rank坑;★全程TVM口径,不提TRT/tensorrt(见[[feedback-framework-on-tvm-not-trt]])。
不做:跨硬件(§3 in 1_7_14)留到最后;V2X-ViT等attention模型冷启动留到transformer入框架计划(conv-TC对其无效);不用fake-quant/dummy scale冒充真int8 AP。
```
