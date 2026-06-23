# sw-optimizer 会话进度报告
**时间**: 2026-06-03 16:25 · **会话时长**: ~91 分钟 · **分支**: hw-deploy-d-space

## 角色与本轮范围
软件优化专家。本会话从团队初始化(只读检查)→ 接 data-orchestrator 采样指令真跑实验。全程**零训练**(仅 build+eval 现成 finetuned ckpt),严守纪律红线(真测即真测/口径标注/AP=DAIR val 1789 finetuned/latency≠AP)。

## 已完成的实验(均真测,带文件佐证)

### Task#1 P0-1 p25 INT8 AP
- ① p25 auto-int8 AP **确认早已在金标准** `stage_a_ap_real.parquet`(ap50=0.7760/ap70=0.5841),不重跑。
- ② p25 forced-all-int8 实测:ap50=0.7743/ap70=0.5804(0 FP16/155 INT8)→ `results/p0_1_p25_forced_int8_ap.json`。
- nuance:p25 forced ΔAP70=-0.0101 ≈ base(-0.0106),**未落预期 base/p50 之间** → forced"越窄越掉"从 ~50% 才激活。

### Task#7 Path A 剪枝×forced-INT8 三档(AP 激活)
- cliff2_c(0.364M)ap70=0.6217 / prune90(0.270M)0.6105 / prune95(0.088M)0.5956 → `results/pathA_forced_int8_ap.json`。
- **发现:全量无悬崖**,forced-int8≈各自 FP16(过参数化),引擎存 perstage cache 供 hw 同引擎测 lat。

### Task#9 Path B head-INT8 护栏消融
- base ap70=0.6263(Δ-0.0046)/ pruned50 0.5685(Δ+0.0044)→ `results/pathB_head_int8_ablation.json`。
- **发现:Pyramid head INT8 近无损,"score崩"假设不成立** → 护栏对 Pyramid 不必要(对 transformer 才有意义)。encoder 不在子网引擎无法做。

### Task#11 Path E 距离分箱重算 AP(★负结果,全量1789,7箱)
- 剪枝 ΔAP70(base_fp16→pruned75_int8):near **-0.147** / mid -0.147 / r50_60 -0.128 / r60_80 -0.098 / **r80plus -0.074**。
- **与"难样本退化更多"假设相反** —— 剪枝在 near 掉最多、远尾掉最少(floor 效应:far AP 本就 ~0.48)。**前沿 trade-off 未激活**。
- INT8 全箱近免费(far 尾 -0.0008)。smoke(60样本)信号反向、不代表 → `results/pathE_distance_binned_ap.json`。

## 跨 A/B/E 综合结论(已报 data,触发兜底)
**Pyramid/DAIR 上 INT8 处处近免费**(全层含 head、全距离箱、全剪枝档);**剪枝是唯一 AP 杠杆但不悬崖、且在难样本上 spread 压缩而非放大**。⇒ **A/B/E 三条均未在 Pyramid/DAIR 激活前沿级 AP trade-off** → 需更难 task/模型(V2X-ViT,QuantV2X 对位)才有真 trade-off。这是诚实负结果,触发 team-lead 兜底讨论。

## 当前状态
- 已验证:上述 4 个 results/*.json 全为真测(DAIR val 1789,空闲 GPU,minmax)。
- 进行中:[32,64,136] finetune(Task#3 门控)epoch46,bestval 仍在推进,未收敛。

## 遗留 / 下一步
1. **[32,64,136] finetune 收敛后** → 通知 data 开 Task#3(FP16+INT8 AP)。
2. **兜底决策待 team-lead**:是否投 V2X-ViT INT8(~2-4天,TRT transformer 无 INT8 → 可能仅 AP-only)激活真 AP trade-off。
3. hw 侧:A 三档/B/p25 forced 引擎已存 cache 供同引擎测 latency(Task#8/#10)。

## 关键文件清单
| 文件 | 类型 | 状态 |
|---|---|---|
| scripts/phase2/p0_1_p25_forced_int8_ap.py | 新增 | 已跑 |
| scripts/phase2/pathA_forced_int8_ap.py | 新增 | 已跑 |
| scripts/phase2/pathB_head_int8_ablation.py | 新增 | 已跑 |
| scripts/phase2/pathE_distance_binned_ap.py | 新增 | 已跑(含距离切分定义) |
| results/p0_1_p25_forced_int8_ap.json | 新增 | 真测 |
| results/pathA_forced_int8_ap.json | 新增 | 真测 |
| results/pathB_head_int8_ablation.json | 新增 | 真测 |
| results/pathE_distance_binned_ap.json | 新增 | 真测(负结果) |
