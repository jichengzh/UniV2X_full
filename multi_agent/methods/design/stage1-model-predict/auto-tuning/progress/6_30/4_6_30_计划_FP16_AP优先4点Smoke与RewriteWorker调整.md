# 6_30 阶段三计划: FP16 AP 优先 4 点 Smoke 与 Rewritten Worker 调整

## 1. 结论

可以先做 AP 测试。AP 的核心目标是生成/验证完整资源链路和精度趋势，不要求像 latency/energy 一样在绝对空闲 GPU 上取得可比数值。因此在后台 GPU 被占用时，优先推进 AP 是合理的。

但旧的 `stage2_original60_fp16_*` 大规模 AP worker 不能原封不动复用。旧 worker 的后半段是:

```bash
stage2_h800_true_fp16_ap_eval.py --precision-mode amp_fp16
```

它不会导出 AP-shape ONNX，不会 build AP-shape rewritten TensorCore engine，也不会调用 rewritten activation bridge。因此旧 worker 只能复用调度思想和 checkpoint 生成/等待逻辑，不能直接作为本阶段 worker。

## 2. 当前已调整内容

新增 worker:

```text
scripts/stage2_original60_fp16_rewritten_ap_worker.py
```

它按 label/GPU 隔离资源目录，并串联以下步骤:

1. 检查或生成 `Pyramid_DAIR_m1_stage2_ap_<label>_2026_06_28` checkpoint。
2. 调用 `scripts/stage2_h800_export_checkpoint_multiscale_onnx.py` 导出 AP-shape multiscale ONNX。
3. 调用 `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py --mode full-engine-group-conv-rewrite --cast-fp16-source` build rewritten full engine。
4. 对 rewrite report 执行 gate: `status=success`、`tensorcore_gate=true`、`.so export_library` 存在。
5. 调用 `scripts/stage2_h800_fp16_rewritten_activation_bridge.py` 进行 AP smoke 或 full-val。
6. 写入 `rows/fp16_rewritten_original60_ap_rows_v1.jsonl`，失败时写 blocker 而不是静默跳过。

同步修正:

```text
scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

新增 `--label`，避免多配置并发时所有 rewrite report 都写成 `lhc_07`。输出文件现在按 label 隔离:

```text
fp16_<label>_full_engine_group_conv_rewrite_latest.json
fp16_<label>_full_engine_group_conv_rewrite_latest.md
```

固定规则:

```text
AP-shape ONNX 必须使用 input_shape=2,64,256,256。
```

原因是目前已修复的 group-conv matcher/rewrite 是在 AP runtime activation shape `[2,64,256,256]` 上验证通过的。`1,64,256,256` 会导致 Relax/TIR shape 改变，当前 matcher 不命中，表现为 `replace_records=0`、`tensorcore_gate=false`。

## 3. 4 点 Smoke 计划

优先使用已经在旧 AP 计划中出现过的 4 个点，覆盖不同 width，并把 lhc_07 作为已验证路线的控制点:

| GPU | label | width | 目的 |
|---:|---|---|---|
| 2 | lhc_07 | 见 queue | golden/control，验证新 worker 与已通过路线一致 |
| 3 | frontier_16 | 48,80,192 | 旧 pilot 点，测试非 lhc 图是否能稳定 rewrite/build |
| 4 | frontier_18 | 48,96,160 | 旧 pilot 点，测试 AP-shape ONNX/export/rewrite 兼容性 |
| 5 | frontier_20 | 56,112,224 | 旧 pilot 点，测试不同 channel 组合 |

GPU6 暂不放入 4 点 smoke，作为备用 lane。如果其中一个 GPU 被其它任务抢占或出现 CUDA OOM，可把失败点迁移到 GPU6 后继续，不应直接终止阶段。

## 4. H800 后台启动命令

在 H800 `${V2X_ROOT}` 下执行:

```bash
cd ${V2X_ROOT}
mkdir -p ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs

nohup ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_original60_fp16_rewritten_ap_worker.py \
  --gpu-id 2 --labels lhc_07 --mode smoke --num-samples 1 --input-shape 2,64,256,256 \
  > ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu2_lhc_07_smoke.out \
  2> ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu2_lhc_07_smoke.err &

nohup ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_original60_fp16_rewritten_ap_worker.py \
  --gpu-id 3 --labels frontier_16 --mode smoke --num-samples 1 --input-shape 2,64,256,256 \
  > ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu3_frontier_16_smoke.out \
  2> ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu3_frontier_16_smoke.err &

nohup ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_original60_fp16_rewritten_ap_worker.py \
  --gpu-id 4 --labels frontier_18 --mode smoke --num-samples 1 --input-shape 2,64,256,256 \
  > ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu4_frontier_18_smoke.out \
  2> ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu4_frontier_18_smoke.err &

nohup ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_original60_fp16_rewritten_ap_worker.py \
  --gpu-id 5 --labels frontier_20 --mode smoke --num-samples 1 --input-shape 2,64,256,256 \
  > ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu5_frontier_20_smoke.out \
  2> ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/logs/gpu5_frontier_20_smoke.err &
```

## 5. 产物位置

总行输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_rewritten_original60_ap_rows_v1.jsonl
```

每点 raw artifact:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/<label>_gpu<gpu>/
```

每点关键文件:

```text
apshape_onnx/<label>_apshape_multiscale.onnx
apshape_onnx/<label>_apshape_multiscale_export_report.json
rewrite_exports/fp16_<label>_full_engine_group_conv_rewrite_latest.json
bridge_exports/fp16_<label>_rewritten_ap_smoke_latest.json
fp16_rewritten_ap_worker_blocker.json
```

## 6. Smoke 通过标准

每个 label 必须满足:

1. checkpoint 存在且 epoch >= 25；若不存在，worker 自动调用旧 train launcher 生成。
2. ONNX export 成功，`stage2_checkpoint_multiscale_onnx_export_report_v1.status=success`。
3. rewrite 成功，`rewritten_full_engine.tensorcore_gate=true`，`.so` 真实导出。
4. bridge 成功，`processed_samples > 0`，`smoke_gate_passed=true`。
5. row 中 AP30/AP50/AP70 非空，prediction 非空。

## 7. 大规模 Worker 调整

4 点 smoke 通过后，后台大规模 worker 不再使用旧 `stage2_original60_fp16_lane_runner.py` 作为 AP eval 入口，而是使用:

```text
scripts/stage2_original60_fp16_rewritten_ap_worker.py
```

可按 GPU lane 分配多个 labels，例如:

```bash
python scripts/stage2_original60_fp16_rewritten_ap_worker.py --gpu-id 2 --labels lhc_01,lhc_05,lhc_09 --mode full-val
python scripts/stage2_original60_fp16_rewritten_ap_worker.py --gpu-id 3 --labels lhc_02,lhc_06,lhc_10 --mode full-val
python scripts/stage2_original60_fp16_rewritten_ap_worker.py --gpu-id 4 --labels lhc_03,lhc_07,lhc_11 --mode full-val
python scripts/stage2_original60_fp16_rewritten_ap_worker.py --gpu-id 5 --labels lhc_04,lhc_08,lhc_12 --mode full-val
python scripts/stage2_original60_fp16_rewritten_ap_worker.py --gpu-id 6 --labels frontier_16,frontier_18,frontier_20 --mode full-val
```

这里 deliberately 不做 busy-GPU 排除，因为本阶段 AP 的优先目标是资源链路和精度有效性，不是 latency/energy 的严格可比测量。

## 8. 下一步 /goal

## 9. 已执行 Smoke 结果

H800 远端:

```text
host: <PRIVATE_HOST>
raw_root: ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630
row_file: ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_rewritten_original60_ap_rows_v1.jsonl
```

4 点 AP smoke 最终全部通过。注意: 本轮 AP smoke 的正式结果只包括 AP30/AP50/AP70、processed、prediction 非空等精度/链路指标；`rewrite gate ratio` 来自 rewrite 阶段为了确认 TensorCore 路线而顺带执行的 default/rewrite engine timing，不是本轮正式 latency/speedup 测量，不能直接写入最终 latency 总表。

AP-only 表:

| label | GPU | width | AP30 | AP50 | AP70 | processed | pred_nonempty | smoke |
|---|---:|---|---:|---:|---:|---:|---:|---|
| lhc_07 | 2 | 24,48,128 | 0.737500 | 0.737500 | 0.626339 | 1 | 1 | pass |
| frontier_16 | 3 | 48,80,192 | 0.836349 | 0.836349 | 0.721805 | 1 | 1 | pass |
| frontier_18 | 4 | 48,96,160 | 0.835913 | 0.835913 | 0.774458 | 1 | 1 | pass |
| frontier_20 | 5 | 56,112,224 | 0.892222 | 0.842500 | 0.842500 | 1 | 1 | pass |

Rewrite gate timing 证据表，仅用于判断 rewritten TensorCore engine 是否真实落地:

| label | GPU | default engine ms | rewritten engine ms | rewrite gate ratio | replace_records | wmma | tvm_mma_sync |
|---|---:|---:|---:|---:|---:|---:|---:|
| lhc_07 | 2 | 36.596502 | 24.463300 | 1.495976 | 2 | 72 | 2 |
| frontier_16 | 3 | 77.210407 | 9.824513 | 7.858955 | 5 | 180 | 5 |
| frontier_18 | 4 | 86.298411 | 11.076047 | 7.791445 | 5 | 180 | 5 |
| frontier_20 | 5 | 102.085760 | 11.974100 | 8.525548 | 5 | 180 | 5 |

关键 artifact:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/lhc_07_gpu2/
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/frontier_16_gpu3/
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/frontier_18_gpu4/
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630/frontier_20_gpu5/
```

本轮修复记录:

1. `CUDA_VISIBLE_DEVICES` 不能在 worker 父进程里 mask 后再把物理 GPU id 传给子脚本，否则 PyTorch export 会报 `invalid device ordinal`。已改为子脚本自行管理物理 GPU。
2. checkpoint 目录里存在多个 `net_epoch_bestval_at*.pth` 时，HEAL 原生 `load_saved_model` 会断言失败。ONNX export 已改为显式调用 `best_checkpoint()` 选择最新 post-baseline checkpoint。
3. AP-shape ONNX 必须用 `2,64,256,256`。batch=1 会导致 matcher 不命中。
4. full-engine group-conv matcher 原先硬编码 `C=256, weight=(256,8,3,3)`。已泛化为 group=32 的动态 channel 规则，覆盖 `C=320/384/448`。

相关验证:

```bash
python -m unittest \
  framework.tests.test_stage2_original60_fp16_rewritten_ap_worker \
  framework.tests.test_stage2_h800_export_checkpoint_multiscale_onnx \
  framework.tests.test_stage2_fp16_tensorcore_group_conv_classifier -v

python -m py_compile \
  scripts/stage2_original60_fp16_rewritten_ap_worker.py \
  scripts/stage2_h800_export_checkpoint_multiscale_onnx.py \
  scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

## 10. 下一步 /goal

```text
/goal 基于 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_30/4_6_30_计划_FP16_AP优先4点Smoke与RewriteWorker调整.md 继续推进 FP16 AP 优先阶段。允许使用 H800 GPU2,3,4,5,6；AP 测试不要求绝对空闲 GPU。先启动并监控 4 点 smoke: lhc_07@gpu2, frontier_16@gpu3, frontier_18@gpu4, frontier_20@gpu5。必须使用 scripts/stage2_original60_fp16_rewritten_ap_worker.py，而不是旧 true-FP16 AP eval worker；AP-shape ONNX 固定使用 input_shape=2,64,256,256，不允许退回 batch=1。每点必须完整生成 checkpoint/ONNX/rewrite engine/activation bridge raw artifact，并保存 rewrite gate、AP30/AP50/AP70、prediction distribution、blocker。4 点 smoke 通过后，把 worker 扩展为 original60 FP16 rewritten AP full-val 大规模补点；遇到 checkpoint 缺失先启动微调生成，遇到 rewrite/bridge 失败先写 blocker 并做根因分析，不允许静默跳过或把旧 true-FP16 AP 行当 rewritten AP 行。
```
