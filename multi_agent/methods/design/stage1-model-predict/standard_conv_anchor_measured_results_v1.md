# Standard Conv Anchor Measured Results v1

日期: 2026-06-23

这份记录只做一件事: 把目前可以引用的 H800/TVM 实测证据绑定到 Stage1 低成本预测器的判据里, 避免再次把 `groups=1` 或 static census 误升级成“模型可分离”结论。

## 结论

当前最稳妥的结论是:

- CoDriving 的标准卷积 ResNet/backbone 范围可以作为“阴性锚点”: C0c' 显示 P/Q/S 低维三臂 `SERIAL`, C1cod 没有稳定 QxS width-specific trap, C6 高维陷阱复核后为 `NO_ROBUST_HIGHDIM_TRAP`。
- 这个结论只能用于 CoDriving 已测 scope, 不能迁移到 F-Cooper / AttFuse / V2X-ViT, 也不能覆盖 Pyramid 的 mixed grouped+standard 场景。
- 标准卷积不是默认可分离。Stage1 预测器的安全输出应是 `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` 或 scoped 的 `ANCHOR_PROBED_LOW_RISK`, 只有同 scope 的真实 joint-vs-serial 证据才允许 `MEASURED_SEPARABLE`。

## 实验编号对照

这些编号来自 `results/coupling_map_matrix.json` 的 coupling-map cell, 不是模型名:

| 编号 | 全称 | 目标 | 结果文件 | 当前判决 |
|---|---|---|---|---|
| C0c' | CoDriving P×Q×S 三臂低维消融 | 比较 joint 搜索和 serial 搜索在 CoDriving 标准卷积上的 HV 差距 | `results/coupling_map/C0c_codriving_pqs.json` | `SERIAL`: A-joint=A-serial=100% HV, 0 rank-flip |
| C1cod | CoDriving Q×S 固定-P 复核 | 检查量化 Q 与调度 S 在 CoDriving 上是否有 width-specific trap | `results/coupling_map/C1_QxS_codriving.json` | `NO_WIDTH_SPECIFIC_TRAP`: p25 soft trap 被判为 cast-chain artifact |
| C6 | CoDriving high-dimensional trap hunt | 把 CoDriving 从低维 P/Q/S 扩展到 batch×width、stage0 K-alignment 等高维陷阱搜索 | `results/coupling_map/C6_codriving_highdim.json` | `NO_ROBUST_HIGHDIM_TRAP`: 原 high-dim trap 结论已撤回 |

## CoDriving FP16 Dense Core 实测

来源:

- `results/cod_e2e.csv`
- `results/codriving_tvm_p25_p75.csv`
- `remote:/exdata/jichengzhi/s2_tvm/l3_fresh_p25_gpu2.log`
- `remote:/exdata/jichengzhi/s2_tvm/l3_fresh_p75_gpu3.log`

| label | ONNX | trials | effective batch | default us | tuned us | default/tuned | source |
|---|---|---:|---:|---:|---:|---:|---|
| base | base_backbone.onnx | 1000 | 2 | 16680.47 | 8057.79 | 2.070 | H800_TVM_fp16_real_existing |
| p50 | p50_backbone.onnx | 1000 | 2 | 3643.15 | 1609.10 | 2.264 | H800_TVM_fp16_real_existing |
| p25 | p25_backbone.onnx | 500 | 2 | 11885.00 | 10467.30 | 1.135 | H800_TVM_fp16_real_seed42 |
| p75 | p75_backbone.onnx | 500 | 1 | 2831.70 | 2450.80 | 1.155 | H800_TVM_fp16_real_seed42 |

注意: p75 ONNX 的输入维度实际固定为 batch=1, 所以不能把 p75 与 p25/base/p50 当成同 batch 直接比较吞吐。

这些 p25/p75 结果修补了此前“estimated latency”的缺口, 但也显示 tuning gain 随形状变化明显, 因此它们支持“CoDriving 已测 envelope 为阴性锚点”, 不支持“标准卷积普遍可分离”。

## CoDriving P/Q/S 与高维复核

来源:

- `results/coupling_map/C0c_codriving_pqs.json`
- `results/coupling_map/C1_QxS_codriving.json`
- `results/coupling_map/C6_codriving_highdim.json`

实测/复核要点:

- C0c': `SERIAL`, A-joint=A-serial=100% HV, 0 rank-flip pairs。
- C1cod: p25 soft trap 被复核为 cast-chain artifact, 没有稳定 width-specific QxS trap。
- C6: 原 `HIGHDIM_TRAP_FOUND` 撤回, 改为 `NO_ROBUST_HIGHDIM_TRAP`; batch-width rank flip 只来自 1.2% 噪声级近平局, 且 p25 更快主要因为模型更小, 不是 K-alignment 陷阱。

所以 CoDriving 是一个有用的负例, 但它只是“groups=1 standard conv 在 CoDriving scope 下未发现稳定耦合陷阱”, 不是所有 standard conv 的通用定理。

## INT8 Stage0 锚点

来源:

- `results/codriving_s0probe_int8.csv`
- `remote:/exdata/jichengzhi/s2_tvm/results/cod_int8_screen.csv`
- `remote:/exdata/jichengzhi/s2_tvm/results/C1_cod_int8_fullbb.json`

| label | cin | K | K mod 32 | tuned us b2 | screen us b1 |
|---|---:|---:|---:|---:|---:|
| p75_s0 | 16 | 144 | 16 | 19.76 | 13.0 |
| p50_s0 | 32 | 288 | 0 | 76.08 | 50.4 |
| p25_s0 | 48 | 432 | 16 | 80.91 | 98.9 |
| base_s0 | 64 | 576 | 0 | 96.71 | 84.2 |

解释: K alignment 确实影响 stage0 的 tuning gain/绝对延迟, 但 C6 复核说明它没有构成 CoDriving 的 robust high-dimensional trap。预测器可以把这类信号作为“需要锚点”的特征, 不能把它当作普遍耦合或普遍无耦合规则。

## 非 CoDriving Gate

V2X-ViT:

- 必须挂 `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND`。
- 原因: 已有 C4 说明 Q-granularity x P 有 AP 敏感性, C5 说明 routing/fusion 可行域会约束 Q/S; dense backbone static scan 不能覆盖这些机制。

F-Cooper:

- 仍是 `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE`。
- 必须补 BaseBEVBackbone schedule anchor 和 MaxFusion/routing coverage, 才能给 dense-subgraph scoped low-risk。

AttFuse:

- 仍是 `FUSION_UNCOVERED_UNKNOWN`。
- attention/fusion 未覆盖时不能给 full-model verdict。

Pyramid lidar/camera:

- 局部 standard Conv2d 只能是 local dense-subgraph 信息。
- 因为 grouped-conv IC_BN P-hub 仍支配模型级判据, 所以局部 `groups=1` 不能解除 `P_HUB_COUPLED`。

## 对预测器的约束

- Static-only 不得输出 `PREDICTED_SEPARABLE_LOW_RISK`。
- 低风险输出必须绑定真实 measured/anchor 文件, 且默认叫 `ANCHOR_PROBED_LOW_RISK`, scope 要写清楚。
- full-model `MEASURED_SEPARABLE` 只允许来自同 scope 的真实 joint-vs-serial 证据。
- 有未覆盖 fusion/routing/custom 子图时, full-model separability wording 必须阻断。
- 含 `Conv2d + ConvTranspose2d + head/neck` 的混合 search group 必须先拆 subgroup, 不能混称 standard Conv2d。
