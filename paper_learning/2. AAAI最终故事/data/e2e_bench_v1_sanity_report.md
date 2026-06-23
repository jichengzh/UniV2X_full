# `e2e_bench_v1.csv` AP sanity rerun 报告 (2026-05-26)

> 目的: 验证 csv 4704 行 AP 列是否可信 — 因 memory `feedback_verify_ap_via_eval_yaml.md` caveat:
> "A.0 AP 列来自 buggy subnet 管线, 用前必须抽样校验". 虽然 schema §八 写
> v1.2 (2026-05-15) 已统一切到 e2e_eval_ap.py n=1789, 但 caveat 未失效, 出图前必须抽样确认.

## 协议

- **采样**: 5 anchor 跨 4 prec × 4 triplet × prune 跨度 0/14/75/89% × D1_default_4gb
- **重测脚本**: `scripts/phase2/e2e_eval_ap.py --n-samples 1789` (DAIR-V2X val 全量)
- **判据**: max |Δ AP50| ≤ 0.02 → 出图; > 0.02 → 暂停 + 列偏差最大 anchor
- **GPU**: RTX 4090 (GPU 6+7 并行), 单 anchor ~3.5 min, 总 wall ~8 min

## 结果

| anchor | csv AP50 | rerun AP50 | Δ AP50 | csv AP30 | rerun AP30 | csv AP70 | rerun AP70 | csv 写入时间 |
|---|---|---|---|---|---|---|---|---|
| T1_base + Q_fp32 | 0.5492 | 0.5491 | **−0.0002** | 0.6422 | 0.6422 | 0.3773 | 0.3774 | 2026-05-14T18:19 |
| T1_base + Q_int8_mm | 0.5458 | 0.5459 | **+0.0001** | 0.6414 | 0.6409 | 0.3701 | 0.3705 | 2026-05-14T18:19 |
| T6_p75 + Q_fp16 | 0.5417 | 0.5413 | **−0.0004** | 0.6411 | 0.6410 | 0.3390 | 0.3389 | 2026-05-14T19:00 |
| T11_p14 + Q_int8_pc_wo | 0.5229 | 0.5231 | **+0.0002** | 0.6298 | 0.6302 | 0.3406 | 0.3431 | 2026-05-17T02:50 |
| T22_p89 + Q_fp32 | 0.5412 | 0.5414 | **+0.0002** | 0.6480 | 0.6479 | 0.3417 | 0.3420 | 2026-05-17T05:04 |

**统计**:
- mean Δ AP50 = +0.0000
- max |Δ AP50| = **0.0004**
- std Δ AP50 = 0.0002
- 阈值: 0.02 → **PASS** (max |Δ| 比阈值小 50×)

## 结论

✅ **csv AP 列可信, 可作图**.

误差量级 ≤ 0.0004 与 DAIR-V2X val 1789 sample 单次 eval 噪声 (~±0.001) 同级, 排除"subnet 管线遗留偏置"; schema §八 v1.2 的 e2e n=1789 切换在 a0_refresh_ap.py + 后续 A.1/A.2 阶段已完整回填.

旧 caveat `feedback_verify_ap_via_eval_yaml.md` 应当**收敛为 "已验证, 2026-05-26"**, 不再 block 使用 e2e_bench_v1 AP.

## 仍存的限制 (不影响出图, 影响 narrative)

1. **FT epoch 不统一**: 21 triplet 中 T2/T4/T6=25 epoch, T1=23, T3/T5/T7/T8/T10-T22 Track A v2=33 (但 T22_bestval=47, T11=33 是 P4 Convergence Gate 选不同 epoch). csv 没有 `finetune_epochs` 列记录, 属 schema §3.4 "可选元数据" 未填.
2. **所有 anchor 都在 FT-equilibrium plateau**: 详见 [问题.md §7.1-7.4](./问题.md). csv "AP 不敏感" 跨 B/Q 现象是 FT≥25 epoch 后的稳态, 不是 fusion robustness.
3. **整 dataset 仍可看作 FT-equilibrium 视角**, 跟 stats_v3 FT=8 / stats_v3_ft6 FT=6 同质 (都在 plateau 内), 不是平行视角.

## 数据落点

- 5 个 rerun JSON: `/tmp/sanity_e2e_v1/{anchor}.json`
- 整合表: `paper_learning/2. AAAI最终故事/data/sanity_rerun_e2e_v1.json`
- 进度日志: `/tmp/sanity_e2e_v1/_progress.log`

## 下一步建议

1. **可以出 14 张图到新目录** `stats_v3_bench/` (沿用 stats_v3 模板, 21T × 7Q × 32D, 不画 FT 轴, 脚注标 "FT-equilibrium 视角, FT epoch 各 triplet 23-47 不等").
2. **若坚持 FT 严格统一**: 需要重做 — 选定一个 FT epoch (如 FT=33, 跟 Track A v2 一致), 把 T1/T2/T4/T6 老 ckpt **重新 finetune 到 epoch 33** 后再生成. 预算: 4 triplet × 4 prec × 1 D ≈ 16 anchor 即可, 用于做 "FT-统一子集" 对照故事; 完整 4704 重做不必要 (老 anchor 数据可作 "FT-equilibrium 但 epoch 各异" 的扩展视角保留).
