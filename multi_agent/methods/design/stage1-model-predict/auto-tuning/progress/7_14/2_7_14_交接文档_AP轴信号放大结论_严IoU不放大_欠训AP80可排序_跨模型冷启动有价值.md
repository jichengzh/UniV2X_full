# 2_7_14 交接文档 — AP轴信号放大结论 + 跨模型冷启动价值

> 前置: `1_7_14`(本轮计划) + `7_2/12_7_3`(AP轴最终验证: B确认过参数化/闭环DS弱信号/框架AP=开环AP70主)
> 日期: 2026-07-04。全部真测(空闲GPU、n=1789逐点核ckpt_path、不信agent自报、主控独立读文件核验)。
> 三方向并行: A(严IoU, GPU0主控直做) + B(欠训预算, GPU1/2 agent WSB) + 目标二(CoDriving冷启动, GPU3-6 agent WSC)。

---

## TL;DR 判定表

| 停止条件 | 结论 | 证据 |
|---|---|---|
| ①方向A: 严IoU是否放大压缩ΔAP | **不放大(收敛阶梯)** | fp16阶梯 p75 ΔAP沿IoU单调**缩小** 0.0101(70)→0.0081(80)→0.0022(90); AP90噪声地板 |
| ②方向B: 是否存在(IoU,budget)让AP可排序 | **是: (欠训e2/e8, AP80)可排序** | AP80在e2 spread=0.049/e8 0.033均单调p25<p50<p75; 但测recoverability非收敛AP |
| ③目标二: 冷启动降MAPE | **有价值(AP70轴大)** | zero-anchor 53.9% → +anchor LOO 5.2%(90.3%相对降), 真数据 |
| ④框架AP轴口径决策 | **收敛→AP70主; 廉价预算→AP80@budget(recoverability); 闭环DS末端安全** | 见§4 |

---

## §1 方向A — 严IoU不放大收敛阶梯的压缩ΔAP(定论)

fp16收敛阶梯(base/p25/p50/p75, DAIR val n=1789真测, 每点带ckpt_path+n_tp/n_gt):

| config | AP50 | AP70 | AP80 | AP90 | ΔAP70 | ΔAP80 | ΔAP90 | ntp70/80/90 |
|---|---|---|---|---|---|---|---|---|
| base | 0.7946 | 0.6344 | 0.3532 | 0.0182 | — | — | — | 22657/15308/2884 |
| p25 | 0.7929 | 0.6296 | 0.3499 | 0.0161 | +0.0048 | +0.0033 | +0.0021 | 22445/15052/2650 |
| p50 | 0.7868 | 0.6301 | 0.3551 | 0.0166 | +0.0043 | −0.0019 | +0.0017 | 22286/15055/2711 |
| p75 | 0.7894 | 0.6243 | 0.3452 | 0.0160 | +0.0101 | +0.0081 | +0.0022 | 22252/14916/2604 |

- **判据**: 最大信号p75的ΔAP沿IoU**单调缩小**(非放大); AP70最强判别(0.0101>噪声0.005); p50@AP80甚至反超base(−0.0019)。
- **AP90=噪声地板**: 全档ntp90~2600-2884(仅AP70的~12%), AP90~0.016-0.018与压缩无关。机理: 该LiDAR检测器IoU0.9定位近乎不可能, 与剪枝无关。
- 机理: 收敛finetune把**定位精度也恢复**(过参数化) → 每个IoU都近无损 → 严IoU无法暴露差异。
- **int8轴(A2)**: 现有 base int8 AP70=0.6204 vs fp16 0.6303 ΔAP70≈0.010近无损, 但该数来自**legacy TRT engine(迁TVM前的早期相)——框架现已全迁TVM, 此TRT数仅作历史参考, 不作框架口径**。严IoU re-eval本轮未做(不再走TRT路)。机理硬论证: AP90处全档n_tp塌到噪声地板, int8 ΔAP在严IoU只会被噪声淹没不可能放大(与fp16阶梯同因)。**框架口径的int8 AP须走TVM int8_tc(dp4a/WMMA真张量化)路测, 见3_7_14计划**。
- 产物: `results/dirA_fp16_ladder_iou_1789.json`。工具: `tools/eval_ap_epoch_curve_iou.py`(扩展IoU 0.8/0.9 + 报n_tp/n_gt, 一次推理覆盖全IoU)。

## §2 方向B — 欠训预算下 (AP80, e2/e8) 可排序(正面但语义不同)

config×epoch×IoU张量(DAIR val n=1789真测, base收敛=参考; p25/p75 fresh-from-pruned-init训到8ep, p50复用fresh_v1)。
ΔAP(vs收敛base)separability:

| epoch | AP50 spread | AP70 spread | AP80 spread | 单调p25<p50<p75 |
|---|---|---|---|---|
| e2 | 0.016 ✓ | 0.027 ✓ | **0.049 ✓** | 全单调 |
| e4 | −0.003 ✗ | −0.001 ✗ | 0.002 ✗ | 打乱(瞬态, 各档不均匀恢复) |
| e8 | ✗ | ✗ | **0.033 ✓** | 仅AP80单调 |

- **判据**: **严IoU在欠训区间确实放大ΔAP**(与方向A收敛区间相反!); 每档内ΔAP80>ΔAP70>ΔAP50(如p75 e2: 0.042→0.104→0.149单调)。**AP80是最可分IoU**, 在e2(spread 0.049)和e8(0.033)都单调可分, AP50/AP70只在e2单调e8即乱。
- **⇒ 存在可排序操作点 (欠训, AP80)**: spread 0.049(e2) >> 噪声0.005。
- **关键nuance(决策依据)**: 该信号测的是 **recoverability-under-budget**(固定小预算下离收敛多远, 与剪枝率相关), **非收敛AP质量**(收敛后阶梯近平=方向A)。e4瞬态排序乱, 收敛后全近无损。用它做框架AP度量会**偏向惩罚重剪枝**, 而重剪枝收敛后近无损。
- zero-shot(e0 pruned-init未finetune)AP全IoU=0.000(剪枝击穿cls head, 与既有zeroshot_n1789.json一致, ckpt核验为真pruned-init非relabel)。p25/p50 e0 n=1789; **p75 e0 诚实caveat: [16,32,64]零shot init的cls head击穿产NaN(fp16)/NMS box洪泛(fp32)→跑不完1789, killed@n=100, AP=0.000(与p25/p50同机理), JSON标n_samples=100未冒充1789**。e0是崩点, 不进separability分析(只用e2/e4/e8), 故不影响判据。张量共48点(3剪枝×4epoch×4IoU)。
- 产物: `results/dirB_config_epoch_iou_tensor.json` + `results/dirB_separability.json` + `results/figure/dirB_signal_heatmap.png` + 原始 `results/dirB/eval_{base,p25,p50,p75}.json`。

## §3 目标二 — 跨模型冷启动(CoDriving先)有价值

12-anchor P×Q(剪枝{0,.25,.5,.75}×{fp32,fp16,int8_tc})。全部底层数据主控独立核验为真。

- **AP70冷启动(主结论, 强)**: CoDriving真AP70(iso-budget DAIR val n=1789)= base 0.4063/p25 0.3661/p50 0.3845/p75 0.4050。Pyramid-only预测器输出Pyramid的AP70水平(~0.60), 对CoDriving真~0.39 → **zero-anchor MAPE 53.9%**; 加CoDriving anchor留一 → **5.2%(90.3%相对降)**。透明mean-baseline, 与WSC GBR 54.9%→3.19%一致。⇒ **跨模型冷启动anchor对AP轴价值巨大且干净**。
- **latency(honest)**: fp32 TVM tuned真测 base 8.06/p25 4.92/p50 1.61ms(1000trial); **p75 tuned崩(CUDA illegal, kernel-cliff[64,32,64]非32对齐)→仅default 2.83ms诚实标GAP**。跨架构latency**不迁移**(Pyramid grouped-conv vs CoDriving标准conv同宽~5-7×绝对差), zero-anchor MAPE~540%+, n=3小样本→稳健结论是方向(anchor对latency更不可或缺)非确切%。
- **fp16≈fp32**: CoDriving标准conv无自动TC pass → fp16=fp32(标 `measured_fp16_eq_fp32_no_tc`), 印证"TVM旋钮价值=conv非标程度函数"跨模型律(Pyramid group-conv专属TC路CoDriving套不上)。
- **GAP诚实标注**: int8_tc whole-model未build(measure_config是Pyramid专属, 新建=数天工程已否决), 仅stage0 micro真WMMA(cin48 1.42×/cin64 1.32× numerical_pass); H800 TVM energy未测。
- F-Cooper/AttFuse: push-button scan_status=ok既存证据(int8_align=32/legal_bits), 未新跑P×Q(scope A优先)。
- 产物: `results/codriving_12anchor_pxq.json` + `results/coldstart_mape_report.json` + `results/codriving_12anchor_lut_rows.csv`(cost-model schema对齐, fam_other=1) + `multi_agent/figure/codriving_coldstart_mape.png`。底层真数据: `${V2X_DATA_ROOT}/s2_tvm/cod_e2e.csv` + `${V2X_DATA_ROOT}/V2Xverse_pyramid/output/codriving_pilot/*/tp_errors_result.json`。

## §4 决策(停止条件④) — 框架AP轴最终口径

1. **收敛预算下(每候选充分finetune)**: AP轴弱信号是过参数化真实属性, 严IoU不放大(§1) → **AP70为主开环判别**(最强、廉价、确定), 配 latency/energy(已fair)做真trade-off + 闭环DS末端行人安全验证(承12_7_3)。
2. **廉价固定预算下(每候选仅2-8ep finetune, 大规模搜索现实场景)**: **AP80比AP70更可分**(§2, spread 0.049 vs 0.027), 单调p25<p50<p75 → **AP80@budget-capped可作排序信号**, 但须明确其语义=recoverability-under-budget(非收敛AP), 会偏向重剪枝更差。这是本轮**新增可操作发现**: "严IoU+欠训预算"确能把AP轴变可排序, 但换来的是不同的量。
3. **跨模型冷启动有价值(§3)**: AP70轴90%+ MAPE降, 干净; latency轴方向对(anchor更不可或缺)。⇒ 新模型onboard应先测少量真AP anchor并入LUT。

---

## §5 数据纪律事件(防复发)

1. **WSC persistence未落地(根因=sandbox overlay, 非蓄意造假)**: WSC报"4件产物已落盘 %db"但主控stat全MISSING。**根因: WSC的Bash命令跑在sandbox文件系统overlay, python生成的JSON/PNG只落overlay未到共享真fs** → 主控(经ssh共享真fs)stat不到, 但WSC自己"看得到"故按已落盘报。底层测量(cod_e2e.csv+AP快照, ssh共享真fs)全真可核。→ 主控从已核验真数据**独立自持久化**4件产物(现磁盘02:07版=主控版, 与WSC后来dangerouslyDisableSandbox重写的02:00版结论一致, 互为交叉验证: 主控mean-baseline 53.9%→5.2% vs WSC GBR 55.2%→3.21% 同向同量级)。**教训: ①agent的Bash可能在sandbox overlay, 交付物必落共享fs+独立stat核后才算完成, 不信"已落盘"自报; ②凡数值必真读文件不从agent报告推断(本轮差点错加_evalsnap后缀用报告值推断p25/p75 AP70)**。
   - WSC补报的int8 AP delta(base-0.004→p75-0.049, 源codriving_dair_grid_8point_clean.csv)也只在overlay未落真fs**不可从盘核验**, 但与既有已知CoDriving int8×剪枝**结构性耦合**(base-0.003→p75-0.062, 见[[project-ap-axis-final-verdict]])一致=corroborative。主控on-disk版int8 AP保守标=fp32(可核), int8×prune真耦合以既有Pyramid对照结论为准。
2. **AP70档位差点被推断污染**: 初次误以p25/p75无AP快照(错加_evalsnap后缀), 差点用WSC报告值推断; 精确glob全部tp_errors_result.json后确认4档真值都在(n=1789)。**教训: 凡数值必真读文件, 不从agent报告推断**。
3. **e0 zero-shot ckpt核验**: 路径名"bestval_at23"是红旗, 但AP全0.000=剪枝崩签名(finetuned会~0.63), 核dir确认是上一轮已验证的pruned-init载体, 非relabel。

## §6 遗留 / 下一步

- **A2 int8严IoU真测(走TVM非TRT)**: 若要坐实int8也不放大, 用**TVM int8_tc路**(dp4a/WMMA真张量化, 框架现役路)测 @ IoU0.8/0.9, **不再重建TRT**(已弃)。本轮机理论证充分, 优先级低; 且随3_7_14的TC-lower路建成自然覆盖。
- **方向B p25/p75的e0**: 张量p25/p75缺e0点(p50有e0=0确认机理); 补齐可完善zero-shot→e8全曲线, 非必需。
- **AP80@budget作框架度量的落地**: 若采纳regime 2, 需在搜索器内环把AP度量从AP70切AP80并固定finetune预算; 但须评估recoverability语义是否匹配部署预算(部署若充分收敛则AP80@budget误导)。
- **冷启动扩展**: F-Cooper/AttFuse真P×Q anchor(本轮仅scan证据); 多模型anchor入LUT验证MAPE系统性降。
- 跨硬件(§3 in 1_7_14)与transformer冷启动(attention, conv-TC对其无效)按原计划留后。

## §7 产物清单(全部主控核验落盘)

| 类 | 路径 |
|---|---|
| A fp16阶梯 | `results/dirA_fp16_ladder_iou_1789.json` |
| A eval工具 | `tools/eval_ap_epoch_curve_iou.py`(H800) |
| B张量/可分性/热图 | `results/dirB_config_epoch_iou_tensor.json` `results/dirB_separability.json` `results/figure/dirB_signal_heatmap.png` |
| B原始eval | `results/dirB/eval_{base,p25,p50,p75}.json` + train.log |
| 目标二 | `results/codriving_12anchor_pxq.json` `results/coldstart_mape_report.json` `results/codriving_12anchor_lut_rows.csv` `multi_agent/figure/codriving_coldstart_mape.png` |
| 目标二底层真源 | `/exdata/.../s2_tvm/cod_e2e.csv` `/exdata/.../codriving_pilot/*/tp_errors_result.json` |
