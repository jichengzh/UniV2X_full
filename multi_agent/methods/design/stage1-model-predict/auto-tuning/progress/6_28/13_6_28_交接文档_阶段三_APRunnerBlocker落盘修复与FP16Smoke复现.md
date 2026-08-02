# 19_6_28_交接文档_阶段三_APRunnerBlocker落盘修复与FP16Smoke复现

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/18_6_28_交接文档_阶段三_FP16INT8_EnergyAP补点收口执行计划.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/17_6_28_交接文档_阶段三_APTrueEval队列与单点Adapter计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`

## 0. 当前权威状态

以当前 worktree 和 `original60_quant_20260627` rows 为准, 不再采用早期“只有两个点 measured”的旧状态。

FP16 + INT8 当前覆盖:

| axis | FP16 | INT8 | 合计 |
|---|---:|---:|---:|
| latency | 60/60 measured | 60/60 measured | 120/120 measured |
| energy | 60/60 measured | 60/60 measured | 120/120 measured |
| AP70 | 0/60 measured | 0/60 measured | 0/120 measured |

目标仍是:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 60 measured rows
rows/native_int8_original60_ap_rows_v1.jsonl = 60 measured rows
```

latency 口径保持:

```text
H800 TVM backbone/subnet module latency in ms
full_network_claim = false
```

不能扩大为完整感知网络端到端 latency 或真实 RSU 物理设备实测。

## 1. 本轮新增代码修复

本轮补强:

```text
scripts/stage2_h800_true_fp16_ap_eval.py
framework/tests/test_stage2_original60_quant_completion.py
```

新增 runner 能力:

1. `validate_execute_inputs(args)`: 在 GPU preflight / HEAL eval 前检查 checkpoint dir、`config.yaml`、checkpoint 文件和 HEAL root。
2. `classify_failure(exc)`: 将失败分为 `configuration_error`、`partial_eval_not_importable`、`environment_error`、`runtime_error`。
3. `write_blocker(...)`: 失败时稳定写:

```text
ap_eval_blocker.json
stdout.txt
stderr.txt
```

4. `resolve_cli_paths(args, launch_cwd)`: 启动时把 `raw_dir`、`rows_out`、`report_json`、`ckpt_dir`、`heal_root` 解析成绝对路径。

第 4 点修复了一个真实问题: `run_true_fp16_eval()` 会 `chdir` 到 HEAL root。旧实现如果传入相对 `--raw-dir`, 失败 blocker 会写到 HEAL 目录下或在异常路径写失败, 导致 V2X 指定 raw 目录里找不到 `ap_eval_blocker.json`。

## 2. 本轮新增测试

新增测试:

```text
test_true_fp16_ap_eval_execute_failure_writes_blocker_logs
test_true_fp16_ap_eval_keeps_relative_raw_dir_under_launch_cwd_after_chdir_failure
```

验证命令:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion
python -m py_compile scripts/stage2_h800_true_fp16_ap_eval.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py framework/tests/test_stage2_original60_quant_completion.py
```

验证结果:

```text
Ran 11 tests in 4.190s
OK
py_compile OK
```

敏感信息扫描已执行, 结果无凭据命中。

## 3. H800 FP16 AP smoke 现状

H800 简短连通性曾确认成功:

```text
hostname = <PRIVATE_HOST>
GPU = 8 x NVIDIA H800
GPU util = 0%
```

随后同步 runner 并运行 `s0_024` 1-sample `amp_fp16` smoke。执行到 HEAL 模型前向后失败:

```text
RuntimeError:Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.
```

这说明 FP16 AP eval 的当前 blocker 是模型前向中的 dtype mismatch, 不是 AP row ingestion 问题。

但该次运行使用的是修复前 runner, 失败后 `ap_eval_blocker.json` 没有出现在 V2X 预期 raw 路径。根因是相对 `raw_dir` 在 `chdir(heal_root)` 后漂移。本轮已用测试固定并修复该问题。

修复后尝试再次同步/复跑时, H800 SSH 在握手阶段多次返回:

```text
kex_exchange_identification: Connection closed by remote host
Connection closed by <PRIVATE_HOST> port 30001
```

这应记录为远端连接频率/会话限制类环境现象, 不是 AP eval 自身的新模型结论。下一轮连接恢复后应直接用修复后的 runner 复跑同一 smoke, 让 blocker 稳定落在 V2X raw 目录。

## 4. 下一步执行顺序

### 4.1 先复跑 FP16 s0_024 smoke 并拿到稳定 blocker

在 H800 连接恢复后执行:

```text
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile scripts/stage2_h800_true_fp16_ap_eval.py
CUDA_VISIBLE_DEVICES=0 ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python scripts/stage2_h800_true_fp16_ap_eval.py \
  --label s0_024 \
  --width 24,128,256 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26 \
  --raw-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_024_smoke_amp_fp16_retry3 \
  --rows-out multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/_tmp_fp16_true_ap_smoke_rows_should_not_use.jsonl \
  --num-samples 1 \
  --execute \
  --precision-mode amp_fp16 \
  --created-at 2026-06-28T00:00:00Z
```

预期不导入 measured AP row。若仍失败, 必须检查:

```text
raw/ap_eval_original60/fp16_true_s0_024_smoke_amp_fp16_retry3/ap_eval_blocker.json
raw/ap_eval_original60/fp16_true_s0_024_smoke_amp_fp16_retry3/stderr.txt
raw/ap_eval_original60/fp16_true_s0_024_smoke_amp_fp16_retry3/gpu_preflight.json
```

### 4.2 对 dtype mismatch 做最小复现

若 blocker 仍为:

```text
Index put requires the source and destination dtypes match
```

下一步必须从 traceback 定位到 HEAL 具体文件/行, 判断是哪一个 tensor assignment 在 autocast/model half 下把 Float source 写入 Half destination。

处理原则:

1. 先定位 exact op/file/line, 不猜。
2. 写最小复现或最小 smoke。
3. 如果修复是安全 dtype cast, 先单点复测。
4. 如果必须局部禁用 autocast, row 里不能宣称 full true-FP16, 必须记录 mixed precision scope。
5. 单点 AP full eval 通过后才扩展 original60。

### 4.3 INT8 AP route

INT8 AP 仍未开始 measured 导入。FP16 单点 runner/blocker 稳定后, 再做 native INT8 AP adapter:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
backend = model_eval
measurement_source = true_eval
full_network_claim = false
```

如果 native INT8 artifact 无法接入 eval, blocker 必须写清楚 input/output tensor、shape、dtype、downstream 期望和 traceback。

## 5. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 AP true-eval 收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP70=0/120 measured; latency 是 H800 TVM backbone/subnet module latency in ms, full_network_claim=false。上一轮已修复 scripts/stage2_h800_true_fp16_ap_eval.py 的失败 blocker 落盘问题: 失败会写 ap_eval_blocker.json、stdout.txt、stderr.txt, 并修复相对 raw_dir 在 chdir(HEAL root) 后漂移的问题; 相关 11 个测试和 py_compile 已通过。下一步先在 H800 用修复后的 runner 复跑 s0_024 FP16 amp_fp16 1-sample smoke, 目标是在 V2X 指定 raw/ap_eval_original60/... 目录拿到稳定 ap_eval_blocker.json 或 ap_eval_report.json。若仍出现 Index put Half/Float dtype mismatch, 先从 traceback 定位 HEAL 具体 op/file/line, 做最小复现和修复, 单点通过后再跑 s0_024 full AP, 产出 rows/fp16_true_original60_ap_rows_v1.jsonl 的首个合规 measured row。FP16 单点闭环后扩展 original60 60 行, 并启动 native INT8 AP adapter, 最终产出 rows/fp16_true_original60_ap_rows_v1.jsonl=60 measured rows 和 rows/native_int8_original60_ap_rows_v1.jsonl=60 measured rows, 刷新 canonical summary/review/gap/queue, 验收 FP16/INT8 latency=120/120 measured、energy=120/120 measured、AP70=120/120 measured、jobs_requiring_action=0。遇到任何 build/eval/import/SSH 问题, 先保存 blocker artifact 和 stdout/stderr, 再反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境确实不可解决。
```
