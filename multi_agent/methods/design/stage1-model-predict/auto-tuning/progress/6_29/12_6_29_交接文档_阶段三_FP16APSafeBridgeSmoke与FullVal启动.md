# 66_6_29 交接文档: 阶段三 FP16 APSafe Bridge Smoke 与 FullVal 启动

日期: 2026-06-29

## 0. 当前状态一句话

`lhc_07` 的 TVM FP16 backbone/subnet output -> PyTorch head/postprocess/AP eval bridge 已经完成 1-sample smoke:

```text
processed_samples=1
pred_nonempty_count=1
pred_total_count=18
AP30=0.7406249999999999
AP50=0.7406249999999999
AP70=0.6293154761904762
smoke_gate_passed=true
```

但 full-val 尚未完成。full-val 已在 H800 后台启动。注意本文件 2026-06-29 追加了 full-val v1/v2 状态修正:

```text
v1 raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v1
v1 结果: 由于脚本 --num-samples 默认值仍为 1, 实际只跑了 1-sample, 不能算 full-val。

v2 raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
v2 python_pid=3698217
v2 状态: 真 full-val 正在 H800 后台运行, 命令中没有 --num-samples。
```

## 1. 必须保留的固定口径

1. 65 号文档中 H800/sm90 speed gate 已闭合, 不需要重复证明能否加速。
2. 65 号 speed gate 的真实 TensorCore engine 输入 shape 是 `[2,64,128,256]`, rewritten latency 为 `12.79650124 ms`, TensorCore gate 为 true。
3. 本轮 AP bridge smoke 使用 checkpoint-consistent AP-shape engine, 输入 shape 是 `[2,64,256,256]`。
4. AP-shape engine 当前没有命中 TensorCore gate:

```text
default_latency_ms=36.48966236
rewritten_latency_ms=36.4855846
tensorcore_gate=false
scheduled_counts={"wmma":0,"tvm_mma_sync":0}
```

因此本轮 smoke 只能证明:

```text
TVM FP16 AP-shape backbone/subnet output 可以接入 HEAL head/postprocess/AP eval, 且 1-sample AP 非空合理。
```

不能把本轮 AP-shape smoke 写成:

```text
AP-shape TensorCore full-engine speed gate 已闭合。
```

## 2. 本轮新增脚本

```text
scripts/stage2_fp16_tvm_worker.py
scripts/stage2_h800_fp16_rewritten_activation_bridge.py
framework/tests/test_stage2_fp16_tvm_worker.py
framework/tests/test_stage2_h800_fp16_rewritten_activation_bridge.py
```

本地和 H800 均已验证:

```bash
python -m unittest framework.tests.test_stage2_h800_fp16_rewritten_activation_bridge framework.tests.test_stage2_fp16_tvm_worker -v
python -m py_compile scripts/stage2_h800_fp16_rewritten_activation_bridge.py scripts/stage2_fp16_tvm_worker.py
```

H800 使用:

```bash
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m unittest framework.tests.test_stage2_h800_fp16_rewritten_activation_bridge framework.tests.test_stage2_fp16_tvm_worker -v
${V2X_DATA_ROOT}/tvm310/bin/python -m py_compile scripts/stage2_h800_fp16_rewritten_activation_bridge.py scripts/stage2_fp16_tvm_worker.py
```

当前测试数: 10 tests, OK。

2026-06-29 追加修复:

```text
framework/tests/test_stage2_h800_fp16_rewritten_activation_bridge.py 新增
test_parse_args_defaults_to_full_validation:
  parse_args 不传 --num-samples 时必须返回 None。

scripts/stage2_h800_fp16_rewritten_activation_bridge.py:
  --num-samples default 从 1 改为 None。

当前本地/H800 测试数: 11 tests, OK。
```

## 3. 关键 root cause 与修复记录

### 3.1 旧 speed-gate engine 与 AP runtime shape 不一致

最初用 65 号 speed-gate `.so` 跑 AP bridge 失败:

```text
engine expects R.Tensor((2,64,128,256), float16)
AP runtime activation is [2,64,256,256]
```

这不是 checkpoint 缺失, 也不是 `.so` 缺失, 而是导出 shape/callsite 与 AP checkpoint config 不一致。

修复:

1. 使用 `scripts/stage2_h800_export_checkpoint_multiscale_onnx.py` 从 AP checkpoint 导出 checkpoint-consistent multiscale ONNX。
2. input shape 固定为 `[2,64,256,256]`。
3. 输出 shape 为:

```text
pyramid_level0=[2,24,256,256]
pyramid_level1=[2,48,128,128]
pyramid_level2=[2,128,64,64]
```

AP-shape ONNX:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260629/lhc_07_apshape_multiscale.onnx
```

### 3.2 AP eval 中 batch=1, TVM engine 固定 batch=2

第一次 bridge smoke 失败:

```text
shape[0] mismatch: 1 vs 2
```

修复:

```text
prepare_worker_activation(): batch=1 pad 到 engine batch=2
slice_worker_output_to_original_batch(): worker output 再切回 original batch=1
```

### 3.3 chdir(HEAL) 后相对路径失效

bridge 在 `run_bridge()` 里会 `os.chdir(heal_root)`, 因此相对 `--rewrite-report` 会失效。

修复:

```text
_resolve_paths() 在 main 启动初期把 rewrite_report/artifact_path/worker_script/ckpt_dir/raw_dir/heal_root/export_report_json 全部 resolve 成绝对路径。
```

### 3.4 numpy bool 无法 JSON 序列化

1-sample AP 实际已完成, 但首次写 report 失败:

```text
TypeError: Object of type bool_ is not JSON serializable
```

修复:

```text
smoke_gate_status() 强制返回 Python bool。
```

## 4. 已完成 AP smoke 证据

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_smoke_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_smoke_latest.json
```

本地 raw summary copy:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_rewritten_ap_bridge/lhc07_apshape_smoke_v3/
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_smoke_v3
```

AP-shape `.so`:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260629/rewrite/lhc07_full_engine_group_conv_rewrite/rewritten_full_engine.so
```

## 5. Smoke output error

TVM FP16 AP-shape output 与 PyTorch reference 对比:

| output | shape | max_abs_err | mean_abs_err | mean_abs_err/ref_abs_mean |
|---|---:|---:|---:|---:|
| output0 | `[1,24,256,256]` | 0.029115676879882812 | 0.00121464638505131 | 0.001472104568857695 |
| output1 | `[1,48,128,128]` | 0.035881638526916504 | 0.0008529233746230602 | 0.002676371086409351 |
| output2 | `[1,128,64,64]` | 0.02682185173034668 | 0.00022862741025164723 | 0.0023954663096400694 |

这与 65 号 speed-gate engine 的 output2 drift 不同: 本轮 AP-shape engine 没有 TensorCore replacement 命中, 所以 output error 很小, 但也不提供 TensorCore AP-safe 证明。

## 6. Prediction / score / box distribution

| tensor | shape | min | max | mean | std |
|---|---:|---:|---:|---:|---:|
| pred_score | `[18]` | 0.2636182904243469 | 0.987476646900177 | 0.6933290362358093 | 0.28908663988113403 |
| pred_box_tensor | `[18,8,3]` | -30.928897857666016 | 99.93978881835938 | 13.124533653259277 | 26.384536743164062 |
| gt_box_tensor | `[20,8,3]` | -48.09828567504883 | 101.95936584472656 | 14.554669380187988 | 30.269208908081055 |

## 7. Full-val 后台任务

### 7.1 v1 不是 full-val

v1:

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v1
```

v1 生成了 `fp16_rewritten_activation_bridge_report.json`, 但检查发现:

```text
processed_samples=1
requested_num_samples=1
```

根因:

```text
bridge CLI 的 --num-samples 默认值仍是 1。
启动 full-val 时未显式覆盖该参数, 因此 v1 其实是 smoke。
```

已修复:

```text
--num-samples default=None
```

不要把 v1 当作 full-val 证据。

### 7.2 v2 真 full-val 正在运行

v2 已启动命令等价于:

```bash
cd ${V2X_ROOT}
CUDA_VISIBLE_DEVICES=4 ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_h800_fp16_rewritten_activation_bridge.py \
  --label lhc_07 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_lhc_07_2026_06_28 \
  --raw-dir ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2 \
  --gpu-id 4 \
  --keep-detailed-samples 1 \
  --persistent-worker \
  --rewrite-report multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260629/fp16_lhc07_full_engine_group_conv_rewrite_latest.json \
  --export-report-json multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json
```

注意该命令没有 `--num-samples`, 因此按修复后的默认值执行完整验证。

监控:

```bash
RAW=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
cat $RAW/pid.txt
ps -p $(cat $RAW/pid.txt) -o pid,ppid,stat,etime,cmd
tail -80 $RAW/stdout.txt
tail -80 $RAW/stderr.txt
find $RAW -maxdepth 1 -type d -name 'bridge_call_*' | wc -l
```

当前已知:

```text
python_pid=3698217
stderr 目前只有 timm/pkg_resources/numpy warning
stdout 尚未输出完整 AP report
已进入真实 bridge 调用, keep_detailed_samples=1 后只有 bridge_call_000 保留, 后续成功样本会在 worker_tmp 中创建并清理。
截至最近检查: worker_tmp/bridge_call_000018 出现, 表示至少推进到第 18 次 TVM worker request。
```

吞吐提示:

```text
v2 约 15 分钟推进到 bridge_call_000018。
full-val 可能需要较长后台时间。
如果后续长时间没有新的 worker_tmp request 或 report/blocker, 先检查 ps wchan、stderr、persistent_worker_stderr.txt, 不要直接判失败。
```

2026-06-30 追加监控记录:

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
pid=3698217
worker_pid=3754707
elapsed≈24min38s 时:
  main process: Sl, pipe_read, CPU≈39.7%, RSS≈1.9GB
  TVM worker: Dl, rwsem_down_write_slowpath, CPU≈3.3%, RSS≈415MB

60 秒 /proc/io 对比显示仍在推进:
  main rchar: 1182048582 -> 1242150291
  main wchar: 1224826264 -> 1291939900
  worker rchar: 1272705213 -> 1341920197
  worker wchar: 830628728 -> 876774296

判断: v2 不属于无进展卡死, 仍在处理 full-val。不要 kill, 继续后台等待 report/blocker。
```

2026-06-30 00:22 左右追加监控记录:

```text
pid=3698217 仍在运行
worker_pid=3754707 仍在运行
active tmp request: worker_tmp/bridge_call_000088/fp16_tvm_worker_request.json
latest activation: worker_tmp/bridge_call_000088/activation_float16.npy
report/blocker: 尚未产出
stdout: 仍为空
stderr: 只有 timm/pkg_resources/numpy warning

判断: v2 已从 bridge_call_000018 推进到 bridge_call_000088, 有明确进展。
不要重启/kill。继续等待 full-val report 或 blocker。
```

## 8. 下一步计划

P0: 继续监控 full-val。若完成:

1. 检查 `processed_samples>=1789`。
2. 检查 `pred_nonempty_count>0`。
3. 检查 AP30/AP50/AP70 finite 且趋势合理。
4. 同步 full-val report、output_error_summary、postprocess_summary、head_output_summary、worker_response_summary 到本地。
5. 生成 `fp16_lhc07_rewritten_full_engine_ap_fullval_latest.md`。

P1: 如果 full-val 失败:

1. 不停止在失败描述。
2. 保存 blocker、stdout/stderr、traceback、request_json、worker_response。
3. 按失败类型处理:
   - TVM worker failure: 查 shape/dtype/LD_LIBRARY_PATH/.so。
   - postprocess failure: 查 score/box/head 分布。
   - empty prediction: 查 output0/1/2 error 与 head 分布。
   - CUDA/IO stall: 查进程状态、stderr、bridge_call 目录。

P2: TensorCore AP-safe 真正闭合:

1. 当前 AP-shape rewrite 没有命中 TensorCore gate, 要继续定位 AP-shape TIR 函数命名/结构为何没有被 `_replace_group_conv_primfuncs_for_full_engine()` 命中。
2. 目标是生成 `[2,64,256,256]` AP-shape TensorCore rewritten engine, 同时满足:
   - `wmma/tvm_mma_sync > 0`
   - rewritten latency < default latency
   - output0/1/2 error 可解释
   - bridge smoke/full-val AP 合理

## 9. /goal 建议

```text
/goal 基于 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/66_6_29_交接文档_阶段三_FP16APSafeBridgeSmoke与FullVal启动.md 继续推进 FP16 APSafe 收口。固定口径: 65 号 H800/sm90 [2,64,128,256] TensorCore speed gate 已闭合, 不再重复证明能否加速；本轮新增 [2,64,256,256] AP-shape TVM FP16 bridge smoke 已通过但 AP-shape TensorCore gate=false, 不能混作 speed gate。下一步先监控 ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v1 的 full-val 后台任务(pid=3614076), 若完成则同步 full-val report/summaries/stdout/stderr, 检查 processed_samples>=1789、pred_nonempty_count>0、AP30/AP50/AP70 finite 且趋势合理, 生成 fp16_lhc07_rewritten_full_engine_ap_fullval_latest.md。若失败, 保存 blocker 和全部 raw artifact 后按 TVM worker/shape/postprocess/empty prediction/CUDA IO 分类定位并修复。随后推进 AP-shape TensorCore rewrite: 定位为什么 [2,64,256,256] full-engine group-conv replacement 没有命中 TensorCore, 生成 AP-shape TensorCore rewritten engine, 再重复 smoke/full-val, 最终闭合 AP-safe TensorCore route。
```
