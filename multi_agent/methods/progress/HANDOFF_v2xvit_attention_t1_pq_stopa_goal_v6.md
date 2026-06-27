# 中文交接 - V2X-ViT Attention T1-P/T1-Q 继续到 Stop-A 通过 (v6, 2026-06-23)

> 这是清理上下文后继续工作的入口文档。目标仍然只有一个：完成 T1-P/T1-Q 的真实 TVM mixed-INT8 端到端验收，让 `results/attention_e2e_pq_v1.json` 通过 Stop-A validator。不要进入 T2，不要把 subnet、逐算子、pilot、fake-quant、TensorRT 或 direct-matmul 写成最终 e2e 证据。

---

## 0. 下一次 /goal 建议目标

建议新窗口直接用：

```text
阅读 multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v6.md，继续完成 V2X-ViT attention T1-P/T1-Q 的最终 Stop-A：在 H800 上核对同协议 baseline/shortft/TVM AP 与 latency 口径，修正或重跑 TVM mixed-INT8 e2e measurement，生成真实 attention-p50-int8/mixed 端到端 row/report，并让 results/attention_e2e_pq_v1.json 通过 Stop-A validator。期间不得进入 T2，不得用 subnet/逐算子/direct-matmul/fake-quant/TensorRT 冒充 e2e。
```

当前 gate：

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

Stop-A 未通过前禁止进入：

```text
Phase T2 - stage1 扩到 attention
```

---

## 1. 计划位置和当前真实进度

对照 `multi_agent/methods/design/plan_transformer_into_framework_v1.md`：

- T0 已完成：V2X-ViT fusion/attention breakdown 已定位。
- T1-S 已完成：MSwin `relative_indices` CPU buffer 问题已修复，S 轴有明确收益。
- T1-P 基本完成：HMSA/MSwin 结构化 head pruning scanner、50% attention pruning、短微调 checkpoint 已完成。
- T1-Q 工程实现已推进：已有 full-model/fusion-subgraph TVM mixed-INT8 runner，支持 MSwin/HMSA TVM patch、W8A8/W8A16、latency/AP e2e measurement、row builder 和 Stop-A 报告链路。
- Stop-A 仍未通过：不是因为 runner 完全缺失，而是 H800 当前 AP 口径与旧 baseline report 不一致，最终 row 还不能可信验收。

当前结论：

```text
T1-P/T1-Q 进入 Stop-A 收口阶段；下一步不是写更多设计，而是解决 H800 同协议 baseline/AP 口径 mismatch，并用同一协议重建 final report。
```

---

## 2. 当前必须记住的核心发现

### 2.1 旧 full-val report 仍是本地参考，但不能盲用作 H800 当前 AP 对照

本地已有：

- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_e2e_checkpoint_eval_full_v1.csv`
- `results/attention_e2e_checkpoint_eval_full_v1.md`

旧 report 数值：

| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | delta AP50 | delta AP70 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | official ckpt | 38.6740 | 1.0000x | 0.71013 | 0.52161 | 0 | 0 |
| attention-p50-fp16 no-ft | 50% | none | 38.9024 | 0.9941x | 0.66884 | 0.46274 | -0.0413 | -0.0589 |
| attention-p50-shortft-fp16 | 50% | 100 steps lr=1e-4 | 35.8636 | 1.0784x | 0.69729 | 0.51322 | -0.0128 | -0.0084 |

解释：

- 50% attention pruning 必须接短微调，否则 AP 下降过大。
- 最终 TVM mixed-INT8 应基于 `attention-p50-shortft-fp16` checkpoint。
- 但 H800 当前重跑得到的 shortft PyTorch AP 约为 0.414/0.253，和旧 report 的 0.697/0.513 不一致；这说明现在不能直接把旧 report 当作 H800 当前实验的 baseline。

### 2.2 TVM subnet/static-HMSA 证据只能作支持，不能作 Stop-A

已有：

- `results/attention_full_tvm_bench_v3_static.json`
- `results/attention_subnet_accel_v2_static.{json,csv,md}`

subnet 数值：

| config | prune | quant | subnet p50 ms | speedup vs base FP16 | speedup vs same-prune FP16 |
|---|---:|---|---:|---:|---:|
| attention-subnet-base-fp16 | 0% | fp16 | 0.928630 | 1.0000x | 1.0000x |
| attention-subnet-base-mixed-int8 | 0% | mixed_int8 | 0.913952 | 1.0161x | 1.0161x |
| attention-subnet-p50-fp16 | 50% | fp16 | 0.564448 | 1.6452x | 1.0000x |
| attention-subnet-p50-mixed-int8 | 50% | mixed_int8 | 0.542453 | 1.7119x | 1.0405x |

边界：

- 这是 subnet，不是 full-model e2e。
- 这是 latency 支持证据，不是 AP 证据。
- Stop-A 最终必须使用 `latency_scope == e2e` 的 full-model/fusion-subgraph runtime measurement。

### 2.3 H800 DAIR val HMSA type coverage 已全量完成

远端日志：

```text
/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json
```

内容：

```text
status: OK
dataset_split: DAIR val
sample_count: 1789
hmsa_call_count: 5367
observed_type_orders: [[0, 0]]
covers_dynamic_type_dispatch: false
```

解释：

- 当前 DAIR val 实际只观察到 static `[0,0]`。
- HMSA TVM static `[0,0]` path 可以作为当前 DAIR val 的实证实现。
- 最终 report 必须诚实写 `covers_dynamic_type_dispatch=false`，不能声称泛化覆盖了所有 type dispatch。

---

## 3. 本轮已经完成的代码工作

### 3.1 `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`

本轮已把它从 row builder/pilot 扩成实际 e2e measurement runner，主要能力：

- `--run-tvm-e2e-measurement`
- `--from-measurements`
- `--tvm-scope {all,mswin,hmsa}`
- `--quant-policy {w8a8,w8a16}`
- `--latency-samples`
- `--latency-warmup`
- `--baseline-report`
- `--out-measurement`

新增/强化的合同：

- 要求 `dataset_split == "DAIR val"`。
- 要求 `n_samples == 1789`。
- 要求 `latency_scope == "e2e"`。
- 要求 TVM artifact 路径存在。
- 要求 `tvm.runtime_stats.total_tvm_call_count > 0`。
- 要求 coverage log 存在且 `sample_count == 1789`。
- 拒绝 fake/prior/simulated AP。
- 拒绝 TensorRT/TRT/QDQ/fake backend。
- 拒绝 subnet/direct-matmul 冒充 e2e。

TVM path 已实现：

- MSwin `BaseWindowAttention`：
  - W8A8：activation INT8 + q/k/v weight INT8，TVM 内 dequant，softmax/output FP16。
  - W8A16：weight-only INT8，输入 FP16，q/k/v INT8 dequant，softmax/output FP16。
  - 已加入 attention scale 和 output bias。
  - TVM 输出 cast 回输入 dtype，避免下游 FP32/FP16 dtype mismatch。
- HMSA `HGTCavAttention`：
  - static 2-agent `[0,0]` lowering。
  - q/k/v INT8 projection + relation/softmax/output FP16。
  - 已加入 mask lowering，不再要求 mask 全 1。

### 3.2 `tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py`

已覆盖：

- row builder 正常生成 Stop-A row。
- 拒绝 `n_samples != 1789`。
- 拒绝 TVM runtime call count 为 0。
- 拒绝 coverage sample_count 不等于 1789。
- 拒绝 fake/subnet/direct/non-TVM。
- CLI 能分发 `--run-tvm-e2e-measurement`、`--tvm-scope`、`--quant-policy`。

本地验证：

```text
python -m py_compile scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py
pytest tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py -q
14 passed
```

---

## 4. H800 真实实验结果

远端：

```text
host: jichengzhi@222.95.84.215 -p 30001
hostname: zs-nj-tap-gpu18
workdir: /exdata/jichengzhi/v2x_t1_attention
HEAL root: /data/jichengzhi_v2x/HEAL
torch env: /data/jichengzhi_v2x/t2venv/bin/python
TVM site: /exdata/jichengzhi/tvm310/lib/python3.10/site-packages
TVM version: 0.20.dev1070+gb628d91fa
torch version: 2.1.2+cu121
numpy version in torch path: 1.26.4
```

### 4.1 `all` scope W8A8 full-val

文件：

```text
/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_all_fullval_v1.json
```

结果：

```text
latency p50: 15.5688 ms
speedup: 2.4841x
AP50: 0.413849
AP70: 0.252565
delta vs old baseline AP50: -0.2963
delta vs old baseline AP70: -0.2690
runtime: MSwin + HMSA TVM called, fallback 0
```

状态：

```text
不能通过 AP guardrail；但后续 no-TVM shortft 复核显示 AP 下降不是 TVM 单独造成。
```

### 4.2 `mswin` scope W8A8 full-val

文件：

```text
/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_mswin_fullval_v1.json
```

结果：

```text
latency p50: 23.0398 ms
speedup: 1.6786x
AP50: 0.4143175
AP70: 0.2527861
delta vs old baseline AP50: -0.2958
delta vs old baseline AP70: -0.2688
runtime_stats.total_tvm_call_count: 16416
mswin_call_count: 16416
hmsa_call_count: 0
fallback_count: 0
```

### 4.3 `mswin` scope W8A16 full-val

文件：

```text
/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_fullval_v1.json
```

结果：

```text
latency p50: 16.9646 ms
speedup: 2.2797x
AP50: 0.4145151
AP70: 0.2527416
delta vs old baseline AP50: -0.2956
delta vs old baseline AP70: -0.2689
runtime_stats.total_tvm_call_count: 16416
mswin_call_count: 16416
hmsa_call_count: 0
fallback_count: 0
```

说明：

- 这是目前最有希望的候选：速度达标，TVM runtime 确实参与 full-model forward，fallback 为 0。
- 但 AP delta 是按旧 high-AP baseline 计算的，当前不能验收。
- 下一步必须用同一 H800 协议重建 baseline/shortft/TVM 三行，或查明为何 H800 shortft AP 与旧 report 不一致。

### 4.4 no-TVM shortft PyTorch H800 full-val 复核

日志：

```text
/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/attention-p50-shortft-fp16_h800_check_latency_v1.json
/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/attention-p50-shortft-fp16_h800_check_ap_v1.json
```

结果：

```text
latency p50: 118.5025 ms
AP50: 0.41462898967971146
AP70: 0.2529802974735779
sample_count: 1789
```

关键解释：

- no-TVM shortft PyTorch AP 约等于 TVM W8A8/W8A16 AP。
- 因此当前 AP collapse 不能直接归因于 TVM/INT8。
- 更可能是 H800 当前 data/eval/checkpoint 协议与旧 `attention_e2e_checkpoint_eval_full_v1.json` 不一致。

### 4.5 MSwin 单模块数值对齐

随机输入单模块比较：

```text
W8A16 vs PyTorch:
max_abs: 0.001026
mean_abs: 5.97e-05
rel_mean: 0.00108
cosine: 0.9999991

W8A8 vs PyTorch:
max_abs: 0.001714
mean_abs: 8.41e-05
rel_mean: 0.00152
cosine: 0.9999983
```

解释：

- MSwin TVM subgraph 数值非常接近 PyTorch。
- 这进一步支持“当前 AP mismatch 是协议/数据/ckpt 问题，不是 MSwin TVM 数值崩溃”的判断。

---

## 5. 当前真正阻塞点

阻塞不是：

- 不是没有 TVM runner。
- 不是 TVM 完全跑不起来。
- 不是 MSwin TVM 数值明显错误。
- 不是 H800 无法跑 full-val。

真正阻塞是：

```text
H800 当前 same-runner no-TVM shortft AP ≈ 0.414/0.253，而旧 report shortft AP ≈ 0.697/0.513。
在查明或重建同协议 baseline 前，任何 delta_ap50/delta_ap70 都不可信。
```

下一轮必须先解决以下二选一：

1. 如果 H800 当前 official baseline AP 也约为 0.414/0.253：说明旧 report 与 H800 当前协议不同，应以 H800 同协议 baseline/shortft/TVM 重建 `attention_e2e_checkpoint_eval_*` 和 final report。
2. 如果 H800 official baseline AP 仍约为 0.710/0.522：说明 shortft checkpoint/manifest/load path 或 pruning artifact 在 H800 上有问题，必须先修 checkpoint/加载流程，再重跑 TVM。

---

## 6. 下一轮第一步：H800 official baseline no-TVM 同协议复核

先跑 baseline，不要先生成 final report。

命令模板：

```bash
cd /exdata/jichengzhi/v2x_t1_attention
export CUDA_VISIBLE_DEVICES=0
export V2X_REPO_ROOT=/exdata/jichengzhi/v2x_t1_attention
export V2X_HEAL_ROOT=/data/jichengzhi_v2x/HEAL
export V2XVIT_CKPT_DIR=/exdata/jichengzhi/v2x_t1_attention/checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26
export V2X_PYTHON=/data/jichengzhi_v2x/t2venv/bin/python
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2:/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/HEAL \
/data/jichengzhi_v2x/t2venv/bin/python - <<'PY'
import json
from pathlib import Path
from attention_e2e_checkpoint_eval import default_eval_configs, load_eval_model, build_dataset_from_hypes
from t1_attention_e2e_pq import evaluate_ap, measure_model_forward_latency, LOG_DIR

out = Path("/exdata/jichengzhi/v2x_t1_attention/results/attention_baseline_pytorch_h800_check_v1.json")
cfg = next(c for c in default_eval_configs() if c["config"] == "baseline")
model, hypes, load_info = load_eval_model(cfg, "cuda:0")
dataset = build_dataset_from_hypes(hypes)
lat = measure_model_forward_latency(
    model, dataset, "cuda:0", "fp16", 5, 30,
    LOG_DIR / "baseline_h800_check_latency_v1.json", 2,
)
ap = evaluate_ap(
    model, dataset, "cuda:0", "fp16", 1789,
    LOG_DIR / "baseline_h800_check_ap_v1.json", 2,
)
res = {"config": cfg, "load_info": load_info, "latency": lat, "ap": ap}
out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"latency": lat, "ap": ap}, indent=2, ensure_ascii=False))
PY
```

注意：

- 使用绝对输出路径。`load_model_and_hypes` 会切到 HEAL cwd，相对路径容易写错位置。
- baseline/shortft/TVM 必须同一 dataset split、同一 eval script、同一 thresholds、同一样本数。

---

## 7. 根据 baseline 复核结果分支

### 分支 A：H800 baseline 也低，约 0.414/0.253

结论：

```text
旧 report 与 H800 当前协议不同。可以不否定 TVM，但必须基于 H800 当前协议重建三行。
```

执行：

1. 用 H800 baseline log 和 no-TVM shortft log 生成新的 H800 e2e checkpoint report，例如：

```text
results/attention_e2e_checkpoint_eval_h800_current_v1.json
results/attention_e2e_checkpoint_eval_h800_current_v1.csv
results/attention_e2e_checkpoint_eval_h800_current_v1.md
```

2. 重新跑或重写 TVM W8A16 measurement 的 baseline reference。

推荐重跑，避免手工改 delta：

```bash
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2 \
/data/jichengzhi_v2x/t2venv/bin/python scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py \
  --run-tvm-e2e-measurement \
  --device cuda:0 \
  --precision fp16 \
  --tvm-scope mswin \
  --quant-policy w8a16 \
  --latency-warmup 5 \
  --latency-samples 30 \
  --eval-samples 1789 \
  --num-workers 2 \
  --baseline-report results/attention_e2e_checkpoint_eval_h800_current_v1.json \
  --out-measurement results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_h800_current_v1.json
```

3. 从 measurement 生成 row：

```bash
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2 \
/data/jichengzhi_v2x/t2venv/bin/python scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py \
  --from-measurements results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_h800_current_v1.json \
  --out-row results/attention_p50_tvm_mixed_int8_row_v1.json
```

4. 生成 final report：

```bash
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2 \
/data/jichengzhi_v2x/t2venv/bin/python scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_h800_current_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md
```

5. 跑 Stop-A validator：

```bash
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2 \
/data/jichengzhi_v2x/t2venv/bin/python scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

### 分支 B：H800 baseline 高，约 0.710/0.522

结论：

```text
H800 shortft checkpoint 或 pruning/加载路径异常，不能用当前 TVM AP。
```

优先排查：

```bash
cd /exdata/jichengzhi/v2x_t1_attention
sha256sum \
  models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth \
  models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json \
  checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/net_epoch_bestval_at17.pth \
  checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/config.yaml
```

还要检查：

- `default_eval_configs()` 里 shortft checkpoint path 是否指向正确文件。
- H800 远端 shortft checkpoint 是否与本地一致。
- manifest 的 `hmsa_keep_heads`、`mswin_keep_heads`、`dim_256_preserved` 是否正确。
- checkpoint 是否真的包含短微调后的 pruned 权重，而不是 no-ft 或坏权重。
- `load_eval_model()` 是否按 manifest 正确应用结构化 pruning 再加载 state_dict。

修正后必须重跑：

- no-TVM shortft full-val AP/latency。
- TVM W8A16/W8A8 full-val AP/latency。
- final row/report/validator。

---

## 8. 下一轮必须拉起两个 agent

继续使用双 agent，避免“执行者自证成功”。

| agent | 建议名 | 权限 | 职责 |
|---|---|---|---|
| Agent-A | `attention-tvm-executor` | 可改代码、可跑 H800 实验 | 跑 H800 baseline 复核；按分支修复协议或 checkpoint；重跑 TVM W8A16/W8A8 e2e；生成 row/report |
| Agent-B | `attention-pq-reviewer` | 只读审查 | 批判性审查 logs/measurement/row/report；确认真 TVM、真 e2e、同协议、AP/latency 不自相矛盾 |

Agent-B 必须阻断以下情况：

- `latency_scope != e2e`。
- TVM runtime call count 为 0。
- TVM artifact/build log 缺失。
- AP 用 fake/prior/simulated 或不同 split。
- baseline、shortft、TVM 不同 eval protocol。
- speedup/AP 刚好达标但没有 raw log、样本数、设备和命令。
- 只测 MSwin pilot、HMSA subnet 或 direct matmul。
- 使用 TensorRT/TRT/QDQ 替代 TVM。

---

## 9. Stop-A 验收标准

机器 gate：

```text
python scripts/phase2/attention_e2e_pq_validator.py --mode stop-a --report-json results/attention_e2e_pq_v1.json
```

必须返回：

```text
verdict == ACCEPTABLE_E2E_SCHEMA
```

最终 report 必须含三行：

```text
baseline
attention-p50-fp16
attention-p50-int8/mixed
```

`attention-p50-int8/mixed` 最低要求：

```text
latency_scope == e2e
dataset_split == DAIR val
n_samples == 1789
quant_backend contains TVM
quant_backend does not contain TensorRT/TRT/QDQ/fake
speedup >= 1.10
delta_ap50 >= -0.02
delta_ap70 >= -0.02
tvm.runtime_stats.total_tvm_call_count > 0
coverage.sample_count == 1789
```

人工 gate：

- TVM runtime 确实参与 `model(batch["ego"])` 的 full-model forward。
- AP 与 latency 不是 subnet、逐算子或 direct-matmul。
- baseline/shortft/TVM 使用同一个 DAIR val protocol。
- pruned row 有 manifest summary，且 `dim=256` preserved。
- dynamic HMSA coverage 边界写清楚：当前只证明 DAIR val static `[0,0]`。

---

## 10. 关键文件清单

设计与交接：

- `multi_agent/methods/design/plan_transformer_into_framework_v1.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v5.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v6.md`

runner/report/validator：

- `scripts/phase2/t1_attention_e2e_pq.py`
- `scripts/phase2/attention_e2e_checkpoint_eval.py`
- `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`
- `scripts/phase2/attention_final_acceptance_report.py`
- `scripts/phase2/attention_e2e_pq_validator.py`
- `scripts/phase2/attention_subnet_accel_report.py`

tests：

- `tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py`
- `tests/phase2/test_t1_attention_e2e_pq.py`
- `tests/phase2/test_attention_final_acceptance_report.py`
- `tests/phase2/test_attention_e2e_pq_validator.py`

本地已有结果：

- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_full_tvm_bench_v3_static.json`
- `results/attention_subnet_accel_v2_static.json`
- `results/attention_final_acceptance_v2_static.json`

H800 关键结果：

- `/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_all_fullval_v1.json`
- `/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_mswin_fullval_v1.json`
- `/exdata/jichengzhi/v2x_t1_attention/results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_fullval_v1.json`
- `/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/attention-p50-shortft-fp16_h800_check_latency_v1.json`
- `/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/attention-p50-shortft-fp16_h800_check_ap_v1.json`
- `/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json`

目标结果：

- `results/attention_e2e_checkpoint_eval_h800_current_v1.json`，如果走分支 A。
- `results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_h800_current_v1.json`，如果走分支 A。
- `results/attention_p50_tvm_mixed_int8_row_v1.json`
- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`

checkpoint：

- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`

---

## 11. H800 环境和连接纪律

远端环境变量模板：

```bash
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm_ffi:$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export V2X_REPO_ROOT=/exdata/jichengzhi/v2x_t1_attention
export V2X_HEAL_ROOT=/data/jichengzhi_v2x/HEAL
export V2XVIT_CKPT_DIR=/exdata/jichengzhi/v2x_t1_attention/checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26
export PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2:/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/HEAL
```

连接纪律：

- 不要把远程密码写进文档、命令或最终汇报。
- 使用 `SSHPASS` 环境变量，并从已有 design 文档解析。
- 不要设置 `sandbox_permissions`。
- 不要清理或回滚用户已有的 dirty worktree。

---

## 12. 硬边界

- 不进入 T2。
- 不把 subnet/static-HMSA bench 写成 full-model e2e。
- 不把 MSwin pilot 写成最终 Stop-A row。
- 不用 TensorRT/TRT/QDQ/ONNX fake path 替代 TVM。
- 不用 fake/prior/simulated AP。
- 不用 direct matmul speedup 推断 e2e speedup。
- 不在协议不一致时硬凑 delta AP。
- 不声称 dynamic HMSA dispatch 已完成；当前只证明 DAIR val static `[0,0]`。
- 如果同协议重跑后 speedup 或 AP guardrail 不达标，应写 Stop-C 负证据，而不是包装成成功。

