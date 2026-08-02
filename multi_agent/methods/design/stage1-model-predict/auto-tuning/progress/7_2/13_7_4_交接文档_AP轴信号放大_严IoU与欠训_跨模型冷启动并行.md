# 13_7_4 交接文档 — AP 轴信号放大(严 IoU + 欠训)+ 跨模型冷启动(并行)

> 前置:`12_7_3`(AP 轴归因定论)+ `11_7_3 §2`(跨模型冷启动设计)。
> 本轮 = 在 12_ 定论(**AP 弱信号 = Pyramid/DAIR 过参数化真实属性**)基础上,**主动尝试把 AP 轴的信号"放大"到可用**,并**并行启动跨模型冷启动**(后台 GPU 富余)。

---

## §0 接手须知(自包含)

- 项目:PyramidFusion/DAIR-V2X 软硬件协同加速框架,已全迁 TVM,H800。精度轴 latency/energy 已 fair;唯一软肋 = **AP 轴信号弱**。
- **12_ 已定论**:真 zero-shot AP70=0.000@1789 → 收敛≈base(31ep 0.630 vs base 0.631,16ep 只到 0.601 差 ~0.03);int8 真量化 ΔAP70=−0.010 不随剪枝放大;闭环 DS 因果存在(r3)但 SNR=0.30 不可检测。**A/C 否决、B 确认**。⇒ AP 弱信号是过参数化 + 足预算 finetune 的真实属性,不是 artifact。
- **12_ 已给的两条"放大信号"线索**(本轮据此设计):
  1. **AP70 span(0.101)>> AP50 span(0.034)** → 更严 IoU 阈值携带更多信号。自然外推:**AP80/AP90 是否进一步放大?**
  2. **16ep(0.601)vs 31ep(0.630)差 ~0.03** → **欠训(finetune≤8)会留更大残差**,且不同压缩配置残差不同 → 放大 config 间 AP spread。
- SSH H800:`printf -v PW '<REDACTED_LEGACY_SECRET>'; ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码 printf 编码,勿明文)。GPU 0-6 常空闲(跑前 nvidia-smi 确认 util 0%/mem≤50MiB),**GPU 7 长期被他人(wuyuegao)占,勿用**。多方向可并行分配不同 GPU。
- 纪律(硬):latency/AP 必空闲 GPU 实测;区分真测/估算/proxy;**凡"曲线/轨迹/已测"必逐点核 ckpt_path + n_samples**(上轮抓到 relabel 造假);每步复跑核验,不信 agent 自报。

---

## §1 方向 A(★核心假设):更严 IoU(AP80/AP90)是否放大压缩对 AP 的影响?

**动机**:AP50/AP70 容忍较松的定位;剪枝/量化引入的小权重扰动**更可能伤定位精度**(box 偏移),这在 AP50/70 被容忍、但在 AP80/AP90(要求紧贴 GT)会暴露。若压缩配置间的 ΔAP 随 IoU 阈值升高而**单调放大**,AP 轴就在高 IoU 下拿到了可排序信号。

**实验(廉价,先做——re-eval 现有收敛 ckpt/engine,无需重训):**
- 载体 = 现有**已收敛**阶梯:base + prune{p25,p50,p75} × {fp16, 真 int8}(ckpt/engine 已在:`data/stage_a_ap_real.parquet` 对应的 pruned ckpt + `${V2X_DATA_ROOT}/pyramid_int8_h800/` 的 int8 engine)。
- 把 opencood AP eval 的 IoU 阈值从 {0.3,0.5,0.7} 扩到 **{0.5,0.7,0.8,0.9}**,对每个 ckpt/engine 在 DAIR val 1789 重测。
- **判据**:对每档压缩,画 ΔAP(config vs base) 随 IoU∈{0.5,0.7,0.8,0.9} 的曲线。若 ΔAP 随 IoU **单调放大**(如 p50 int8 在 AP90 掉 5-10% 而 AP50 只掉 1%)→ **严 IoU 放大信号成立**,记录信号最强的 IoU。
- 坑:AP90 可能因 TP 太少而噪声大(n_tp 骤降);须报每档 n_tp,信号必须 >> eval 噪声(~0.001-0.005)。

## §2 方向 B:欠训(finetune ≤8)是否放大压缩配置间的 AP spread?

**动机**:12_ 已证 over-param 在**足预算**下抹平压缩代价(31ep→base)。反过来**卡预算(≤8ep)** 时恢复不完全,且**重压缩配置恢复更慢** → config 间 AP 拉开 → AP 轴携带可排序信号。这本质是把"finetune 预算"当成一个让 AP 轴显影的旋钮。

**实验(需重训,可并行——各 config 一个 GPU):**
- 对 prune{p25,p50,p75} 各做 **fresh finetune 到 8 epoch**(save_freq=1,从剪枝 init;p50 已有 `Pyramid_DAIR_m1_p50_fresh_v1` 可直接用,补 p25/p75)。★用**修好的 launcher**(`tools/train_ddp_patched.py` 或 runner `stage2_h800_ap_finetune_smoke_runner.py`),别再踩 `train_ddp.py` rank 崩。
- 在 epoch∈{0,2,4,8} × IoU∈{0.5,0.7,0.8,0.9} 测 AP(与方向 A 合并成一张 **config × epoch × IoU** 的 AP 张量)。
- **判据**:找 (epoch, IoU) 操作点使 **base/p25/p50/p75 的 AP 单调可分、spread >> 噪声**。预期信号最强 = **低 epoch(2-4)+ 高 IoU(0.8-0.9)**。若找到 → 框架 AP 度量切到该口径(如 AP80@ep4);若各处都抹平 → AP 轴确无可用信号,坐实过参数化定论,框架 AP 轴降权、主打 latency/energy。

> §1+§2 合并交付一张 config×epoch×IoU AP 张量 + 一张"信号强度 vs (IoU,budget)"热力图。

## §3 方向 C(并行启动):跨模型冷启动探针(11_7_3 §2.2)

**目标**:cost model 只在 Pyramid original60 上训,迁到新模型无锚点。onboard 新模型时先做**轻量全模型 P×Q 探针**作冷启动锚点。

**实验(与 §1/§2 并行,占另一批 GPU):**
- 目标模型:先用**已 auto_scan 通**的 **F-Cooper / AttFuse**(Phase A1 已探通,见 [[project-autoscan-phase-a0-a1]])。
- **探针网格**:P = 全模型统一剪枝率{0,0.25,0.5,0.75}(不分 stage)× Q = {fp32,fp16,int8_tc} → **12 anchor/模型**。
- 测量:复用 `framework/measure_config.py` 三精度接口(已 ready)+ §1 真 AP 口径(含真 int8)。
- 并入:训练表加 `model_class` one-hot + `is_coldstart_anchor` 标记;cost model 迁移/分层(Pyramid 全量 base + 目标 12 anchor 微调)。
- **判据**:留出目标模型若干分区配置作 test,比"零 anchor 直接迁"降 MAPE 多少 = 冷启动价值。
- 顺带:whole-model 探针 = 三臂里 S0/serial 基线,可为跨模型提供 argmin-drift 载体(不同模型 fp16-最优 vs int8-最优 全局压缩率漂移)。
- 坑:AttFuse/F-Cooper 的 fusion neck TVM 导入可能 blocked(见 unified handoff);attention 模型(V2X-ViT)瓶颈非 conv,现有 conv-TC rewrite 无效 → 冷启动之外还有算子覆盖问题,**本轮先只做 conv-backbone 类(F-Cooper/AttFuse),V2X-ViT 留到 transformer 入框架计划**。

---

## §4 并行 GPU 分配建议(0-6 空闲,7 勿用)

| 方向 | 内容 | 建议 GPU | 依赖 |
|---|---|---|---|
| A | AP80/90 re-eval 现有阶梯(廉价,先出) | GPU 0 | 无(用现有 ckpt/engine) |
| B | p25/p75 fresh finetune≤8(p50 已有)+ 合并测 AP 张量 | GPU 1,2 | 修好 launcher |
| C | F-Cooper 12-anchor 冷启动探针 | GPU 3,4 | auto_scan manifest |
| C | AttFuse 12-anchor 冷启动探针 | GPU 5,6 | auto_scan manifest |

---

## §5 /goal — 下一阶段停止目标(compact 后直接用)

```
/goal 阅读 multi_agent/methods/design/auto-tuning/progress/7_2/13_7_4_交接文档_AP轴信号放大_严IoU与欠训_跨模型冷启动并行.md 了解背景。本轮两大目标并行推进,所有实验真测(空闲GPU实测、区分真测/proxy、逐点核ckpt_path+n_samples、不信agent自报)。

目标一(AP轴信号放大):判定能否通过"更严IoU + 欠训预算"把AP轴从无信号变成可排序信号。
- [P0,先做] 方向A:re-eval现有收敛阶梯(base+prune{p25,p50,p75}×{fp16,真int8})在IoU{0.5,0.7,0.8,0.9}的DAIR val 1789 AP;判定ΔAP(config vs base)是否随IoU单调放大(须报n_tp,信号>>噪声~0.005)。
- [P0,并行] 方向B:prune{p25,p50,p75}各fresh finetune到8ep(用修好的launcher,p50已有fresh_v1);在epoch{0,2,4,8}×IoU{0.5,0.7,0.8,0.9}测AP,合成config×epoch×IoU张量;找(epoch,IoU)操作点使base/p25/p50/p75 AP单调可分spread>>噪声。

目标二(跨模型冷启动,并行占另一批GPU):在已auto_scan通的F-Cooper/AttFuse上做12-anchor whole-model P×Q探针(P剪枝率{0,.25,.5,.75}×Q{fp32,fp16,int8_tc}),测lat/energy/真AP并入LUT,验证冷启动降MAPE价值。

停止条件(全满足即STOP,写14_交接文档):
- [ ] 方向A:现有阶梯AP80/AP90已测,给出"严IoU是否放大压缩ΔAP"的定量结论(含n_tp与噪声对比)。
- [ ] 方向B:config×epoch×IoU AP张量+信号热力图已出;给出"是否存在(IoU,budget)操作点让AP轴可排序"的判定;若有,指明框架AP度量新口径。
- [ ] 目标二:F-Cooper或AttFuse(至少1个)12-anchor探针已测并入LUT,report冷启动vs零anchor的MAPE降幅。
- [ ] 决策:框架AP轴最终口径(AP70/AP80@budget-capped/降权)+ 跨模型冷启动是否有价值,附证据。

纪律:空闲GPU实测;各方向并行分配不同GPU(A:GPU0/B:GPU1,2/C:GPU3-6);不信自报,逐产物读文件核验;修好的launcher别再踩train_ddp rank坑。
不做:跨硬件(§3)留到最后;V2X-ViT等attention模型冷启动留到transformer入框架计划(现有conv-TC rewrite对其无效)。
```

---

## §6 关联

- 前置定论 [[project-ap-axis-final-verdict]](12_)、[[project-dair-ap-axis-collapse]]。
- 跨模型资产 [[project-autoscan-phase-a0-a1]](auto_trace 已探通 F-Cooper/AttFuse)、[[project-two-plans-transformer-probing]]。
- 精度轴 fair 前置 [[project-precision-axis-double-unfairness]]。
- 遗留小修(承 11_7_3 §附):`run_pqs_ablation` 去墙诊断打印文案 stale(实为 argmin drift)。
