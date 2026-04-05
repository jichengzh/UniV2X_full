# AdaRound 架构决策记录

> 每条决策一旦写入此文件即视为**不可变更**，除非明确标注 [SUPERSEDED]。
> 每次 session 开始必读。

---

## ADR-002：AdaRound 仅应用于 BEV encoder（不扩展到检测头/下游头）

**状态**：ACCEPTED  
**日期**：2026-04-02  
**决策者**：PM

**背景**：  
当前 vanilla PTQ（AMOTA 0.364）与 FP16 baseline（0.370）差距仅 0.006。BEV encoder 是精度瓶颈（50个TSA激活层，temporal依赖使entropy scale低估）。检测头已有 zero-padding 损失（-0.009），下游头未量化。

**决策**：  
第一阶段 AdaRound 仅优化 BEV encoder。检测头/下游头 AdaRound 不在当前 scope。

**原因**：  
- BEV encoder 是最大收益/风险比
- 检测头 AdaRound 与 zero-padding 损失叠加分析困难
- 最小化本次改动范围

---

## ADR-003：新建 calibrate_univ2x_adaround.py 而非修改现有脚本

**状态**：ACCEPTED  
**日期**：2026-04-02  
**决策者**：PM

**决策**：  
创建新脚本 `tools/calibrate_univ2x_adaround.py`，不修改 `calibrate_univ2x.py`。

**原因**：  
- 保留 vanilla PTQ 脚本作为 fallback
- AdaRound 需要额外参数（iters, warm-up, λ），接口不兼容
- 降低风险

---

## ADR-004：AdaRound 重建顺序 + ONNX 导出策略

**状态**：ACCEPTED  
**日期**：2026-04-02

**重建顺序**：QuantModule 级 Sequential（前层到后层逐一重建），不支持 Block 级。`calibrate_univ2x.py` 已按此正确实现。

**temporal prev_bev**：`save_inp_oup_data()` 采集时已包含完整 forward（含 TSA），重建时不需额外处理。

**ONNX 导出策略**：使用方案 B（重新 export）而非 ONNX patch：
1. `save_quantized_weight(qmodel)` → `module.weight.data = W_fq`
2. `module.org_weight.copy_(module.weight.data)` → forward 使用 W_fq
3. `set_quant_state(False, False)` → 禁用量化器
4. 用该模型重新 export ONNX（复用 BEVEncoderWrapper）

**quant_params 一致性**：`calibrate_univ2x.py` 已将 `weight_quant_params`/`act_quant_params` 存入 .pth，export 脚本从 ckpt 读取，保证一致。

---

## 继承自 ADR-001（quant_result.md）

ADR-001 决策保持不变：  
`sampling_offsets` 和 `attention_weights` 保持 FP16，不参与 AdaRound 重建。
