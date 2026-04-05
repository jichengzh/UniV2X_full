# Phase D 验证指南

> 状态：待执行（依赖 Phase C 产出物）

---

## D-1：模块级 Cosine 验证

```bash
conda run -n UniV2X_2.0 python tools/validate_quant_bev.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --quant-weights calibration/quant_encoder_adaround.pth \
    --n-samples 10
```

**验收标准**：
- cosine(AdaRound W8A8, FP32) > **0.997**（高于 vanilla PTQ 的 0.9947）
- 若低于 0.997，检查：
  1. `--adaround-iters` 是否过少（尝试 10000）
  2. `scale_method` 是否与 calibration 一致

结果写入：`docs/quant/adaround/outputs/phase_D/cosine_result.md`

---

## D-2：端到端 AMOTA 验证

```bash
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \
    --plugin plugins/build/libuniv2x_plugins.so \
    --use-lane-trt --use-agent-trt \
    --eval bbox
```

**验收标准**：

| 指标 | 目标 | 对照 |
|------|------|------|
| AMOTA | ≥ 0.370 | vanilla PTQ INT8: 0.364 |
| AMOTP | ≤ 1.446 | FP16 baseline: 1.446 |
| mAP | ≥ 0.074 | vanilla PTQ: 0.0744 |

---

## D-3：若 AMOTA < 0.370 的降级策略

```
尝试 1: 增加 iters 至 10000（重新运行 C-1）
尝试 2: 改 scale_method 为 entropy（重新运行 C-1 + C-2）
尝试 3: 保留 vanilla PTQ 结果（AMOTA 0.364），记录 AdaRound 未带来额外收益
```

---

## D-4：更新文档（D-2 完成后）

1. 在 `docs/quant/quant_result.md` 精度表新增 AdaRound 行
2. 在 `result.log` 新增 Phase AdaRound 节
3. 将 `docs/quant/adaround/PROGRESS.md` 状态改为 `COMPLETE`
