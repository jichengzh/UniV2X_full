# 中文交接 — V2X-ViT Attention T1-P/T1-Q 继续到 Stop-A 通过 (v5, 2026-06-23)

> 这是下一次清理上下文后用 `/goal` 启动工作的入口文档。核心目标只有一个：把 `attention-p50-int8/mixed` 做成真实 TVM mixed-INT8 端到端实测行，并通过 Stop-A。不要进入 T2，不要用 subnet、逐算子、fake-quant 或 TensorRT 替代最终证据。

---

## 0. /goal 建议目标

建议新窗口直接用下面目标启动：

```text
阅读 multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v5.md，继续完成 V2X-ViT attention T1-P/T1-Q 的最终 Stop-A：实现 full-model 或 attention fusion-subgraph TVM mixed-INT8 runner，生成真实 attention-p50-int8/mixed 端到端 latency/AP 行，并让 results/attention_e2e_pq_v1.json 通过 Stop-A validator。期间不得进入 T2，不得用 subnet/逐算子/direct-matmul/fake-quant/TensorRT 冒充 e2e。
```

当前 gate：

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

Stop-A 通过前禁止进入：

```text
Phase T2 — stage1 扩到 attention
```

---

## 1. 当前真实进度

和 `multi_agent/methods/design/plan_transformer_into_framework_v1.md` 对齐后，当前推进位置是：

- T0：attention/fusion breakdown 已完成。
- T1-S：MSwin `relative_indices` CPU buffer 问题已定位并修复，S 轴已有实现修复收益。
- T1-P：HMSA/MSwin 结构化 head pruning scanner、50% attention pruning、短微调 checkpoint 已完成。
- T1-Q：subnet/static-HMSA TVM bench 已完成，但最终 full-model/fusion-subgraph TVM mixed-INT8 e2e 仍未完成。
- Stop-A：仍未通过，当前状态应视为 `BLOCKED_TVM_MIXED_INT8_E2E_MISSING`。

已经完成的可用基础：

- `attention-p50-shortft-fp16` 是下一步 TVM mixed-INT8 应使用的 checkpoint。
- row builder/validator/report 框架已具备，不允许从假数据生成最终 row。
- H800 环境、TVM 和 HEAL/PyTorch 的组合导入问题已经定位，有可用 bootstrap 方案。
- DAIR val 全量 HMSA type coverage 已完成：实际观察到固定 type order `[[0, 0]]`。

仍未完成的关键项：

- `attention-p50-int8/mixed` 的真实 full-model e2e latency/AP 行。
- TVM mixed-INT8 在完整检测任务上的 AP 保真。
- `results/attention_e2e_pq_v1.json` 通过 Stop-A validator。

---

## 2. 当前关键证据

### 2.1 Full-val FP16 checkpoint

来源：

- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_e2e_checkpoint_eval_full_v1.csv`
- `results/attention_e2e_checkpoint_eval_full_v1.md`

| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | delta AP50 | delta AP70 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | official ckpt | 38.6740 | 1.0000x | 0.71013 | 0.52161 | 0 | 0 |
| attention-p50-fp16 no-ft | 50% | none | 38.9024 | 0.9941x | 0.66884 | 0.46274 | -0.0413 | -0.0589 |
| attention-p50-shortft-fp16 | 50% | 100 steps lr=1e-4 | 35.8636 | 1.0784x | 0.69729 | 0.51322 | -0.0128 | -0.0084 |

解释：

- 直接剪掉 50% attention heads 后，AP 明显下降。
- 短微调后 AP 恢复到 Stop-A 容忍范围内，但并不是超过 baseline；它只是从 no-ft 恢复。
- 下一步 TVM mixed-INT8 必须基于 shortft checkpoint，不能基于 no-ft checkpoint。

### 2.2 TVM subnet/static-HMSA 证据

来源：

- `results/attention_full_tvm_bench_v3_static.json`
- `logs/attention_e2e_pq_v1/attention_full_tvm_bench_v3_static.log`
- `results/attention_subnet_accel_v2_static.{json,csv,md}`

HMSA v3 scope：

```text
full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8
```

TVM subnet 数值：

| config | prune | quant | subnet p50 ms | speedup vs base FP16 | speedup vs same-prune FP16 |
|---|---:|---|---:|---:|---:|
| attention-subnet-base-fp16 | 0% | fp16 | 0.928630 | 1.0000x | 1.0000x |
| attention-subnet-base-mixed-int8 | 0% | mixed_int8 | 0.913952 | 1.0161x | 1.0161x |
| attention-subnet-p50-fp16 | 50% | fp16 | 0.564448 | 1.6452x | 1.0000x |
| attention-subnet-p50-mixed-int8 | 50% | mixed_int8 | 0.542453 | 1.7119x | 1.0405x |

解释：

- subnet 层面剪枝带来明显收益，mixed-INT8 相比 same-prune FP16 只有小幅收益。
- 这说明继续做 full-model/fusion-subgraph TVM 是合理的，但不能说明 e2e 已经加速。
- 当前 latency 不明显变化的根因还没有最终判定；需要 full-model profiling 区分 attention 不是瓶颈、TVM launch/QDQ overhead、HMSA/MSwin dispatch 开销、还是未覆盖足够子图。

### 2.3 H800 HMSA type coverage

H800 上全量 DAIR val coverage 已完成：

```text
host: zs-nj-tap-gpu18
workdir: /exdata/jichengzhi/v2x_t1_attention
log: /exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json
status: OK
sample_count: 1789
hmsa_call_count: 5367
observed_type_orders: [[0, 0]]
covers_dynamic_type_dispatch: false
elapsed_secs: 218.801
```

解释：

- 当前 HEAL/DAIR val 实际路径固定为 `[[0, 0]]`，不是 v3 static bench 里假定的 `[0, 1]`。
- 下一阶段可以优先实现/验证 static `[0,0]` HMSA TVM path，这是面向当前 DAIR val 的可证路径。
- 但最终 row 里必须保留 `covers_dynamic_type_dispatch=false`，不能声称已覆盖动态 HMSA dispatch。

---

## 3. 本轮新增/修改的本地实现

### 3.1 `scripts/phase2/t1_attention_e2e_pq.py`

已做：

- 增加环境变量覆盖，方便在 H800 上复用同一脚本：
  - `V2X_REPO_ROOT`
  - `V2X_HEAL_ROOT`
  - `V2X_PYTHON`
  - `V2XVIT_CKPT_DIR`
  - `V2XVIT_CONFIG_YAML`
  - `V2XVIT_CKPT_FILE`

### 3.2 `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`

已做：

- `--from-measurements`：从真实 measurement JSON 生成 Stop-A row。
- row builder 会拒绝：
  - `latency_scope != e2e`
  - fake/prior/simulated AP
  - direct-matmul/subnet 冒充 e2e
  - 非 TVM backend
- `bootstrap_h800_tvm_then_torch()`：
  - 先 import TVM，再插入 HEAL/t2lib。
  - 删除已加载的 NumPy 2.x，再从 t2lib 载入 NumPy 1.x。
  - 解决 H800 上 TVM 与 torch 2.1.2/CUDA runtime 的导入冲突。
- `--record-type-coverage`：
  - 跑真实模型 forward，记录 HMSA `prior_encoding` type order。
- `--run-mswin-tvm-pilot`：
  - 将 MSwin `BaseWindowAttention` 的 Q/K/V projection 接到 TVM Relax int8 matmul。
  - softmax/output projection 仍为 FP16。
  - 该模式只是 pilot，不是最终 Stop-A row。

重要边界：

- 当前 runner 还没有生成最终 `attention-p50-int8/mixed` AP/latency measurement。
- 当前 MSwin pilot 不是最终证据，因为 HMSA TVM path 和 AP eval 还未接入。
- 本地 bootstrap 已修复 NumPy 问题，但需要在 H800 上同步后重新跑 pilot。

### 3.3 `tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py`

已新增测试：

- row builder 保留 Stop-A 合同字段。
- row builder 拒绝 subnet/direct/fake/non-TVM。
- CLI `--from-measurements` 写 row。
- CLI `--record-type-coverage` 只记录 coverage，不生成最终 row。
- CLI `--run-mswin-tvm-pilot` 只跑 pilot，不生成最终 row。
- H800 bootstrap 导入顺序：`tvm -> numpy -> torch`。

最新本地验证：

```text
py_compile: passed
pytest tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py -q: 10 passed
```

---

## 4. H800 环境事实

远端：

```text
ssh host: jichengzhi@222.95.84.215 -p 30001
hostname: zs-nj-tap-gpu18
GPU: H800
workdir: /exdata/jichengzhi/v2x_t1_attention
HEAL root: /data/jichengzhi_v2x/HEAL
```

Python/TVM：

```text
TVM env: /exdata/jichengzhi/tvm310/bin/python
TVM version: 0.20.dev1070+gb628d91fa
Torch/HEAL env: /data/jichengzhi_v2x/t2venv/bin/python
Torch version: 2.1.2+cu121
Torch site: /data/jichengzhi_v2x/t2lib
```

导入规则：

- 不要直接在 torch 环境里先 import torch 再 import TVM。
- 正确顺序是：先用 TVM site import TVM，再插入 HEAL/t2lib，清理 NumPy 2.x，导入 NumPy 1.x 和 torch。
- 这个逻辑已经封装在 `bootstrap_h800_tvm_then_torch()`。

远端已同步的关键资产：

```text
/exdata/jichengzhi/v2x_t1_attention/checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/config.yaml
/exdata/jichengzhi/v2x_t1_attention/checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/net_epoch_bestval_at17.pth
/exdata/jichengzhi/v2x_t1_attention/models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth
/exdata/jichengzhi/v2x_t1_attention/models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json
```

安全连接方式：

```bash
# 不要把密码写进命令、文档或最终汇报。用 SSHPASS 环境变量。
export SSHPASS="$(python - <<'PY'
from pathlib import Path
import re
text = Path('multi_agent/methods/design/plan_transformer_into_framework_v1.md').read_text(encoding='utf-8')
match = re.search(r"sshpass\s+-p\s+'?([0-9A-Za-z_@.-]+)'?\s+ssh\s+-p\s+30001", text)
if not match:
    raise SystemExit('missing H800 password pattern')
print(match.group(1))
PY
)"
sshpass -e ssh -p 30001 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215 'hostname'
```

同步本地 runner 到 H800：

```bash
sshpass -e rsync -av -e 'ssh -p 30001 -o StrictHostKeyChecking=no' \
  scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py \
  jichengzhi@222.95.84.215:/exdata/jichengzhi/v2x_t1_attention/scripts/phase2/
```

---

## 5. 下一窗口必须拉起两个 agent

继续保持双 agent 机制：

| agent | 建议名 | 权限 | 职责 |
|---|---|---|---|
| Agent-A | `attention-tvm-executor` | 可改代码、可跑 H800 实验 | 实现/调试 TVM mixed-INT8 full-model 或 fusion-subgraph runner，生成真实 measurement、row、report |
| Agent-B | `attention-pq-reviewer` | 只读审查 | 批判性检查结果是否真 TVM、真 e2e、同协议、AP/latency 是否可信；发现不合理必须阻断 |

Agent-A 的最小任务：

```text
把 shortft p50 checkpoint 的 attention path 接入 TVM mixed-INT8，跑完整 DAIR val latency/AP，产出 attention_tvm_mixed_int8_e2e_measurement_v1 JSON。
```

Agent-B 的最小任务：

```text
审查 measurement/row/report/log：确认不是 subnet、不是逐算子、不是 fake AP、不是 TensorRT；确认 TVM artifact/runtime 实际参与 full-model forward；确认 speedup/AP guardrail 真实满足。
```

Agent-B 如果发现以下任一情况，应要求返工：

- `latency_scope` 不是 `e2e`。
- AP 不是同一 runner、同一 DAIR val split、同一 checkpoint。
- TVM 只出现在字段名里，没有 build/runtime/log 证据。
- 只测 MSwin pilot、HMSA subnet 或 direct matmul。
- speedup/AP 刚好达标但没有 raw log、命令、样本数、设备信息。

---

## 6. 下一阶段执行路线

### Step 1: 先同步并重跑 MSwin TVM pilot

目的：

- 验证本地 NumPy bootstrap 修复在 H800 上有效。
- 验证 TVM Relax qkv-int8 path 可以嵌入 full-model forward。
- 只作为工程打通，不作为 Stop-A 最终证据。

命令模板：

```bash
cd /exdata/jichengzhi/v2x_t1_attention
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm_ffi:$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export V2X_REPO_ROOT=/exdata/jichengzhi/v2x_t1_attention
export V2X_HEAL_ROOT=/data/jichengzhi_v2x/HEAL
export V2XVIT_CKPT_DIR=/exdata/jichengzhi/v2x_t1_attention/checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26
export V2X_PYTHON=/data/jichengzhi_v2x/t2venv/bin/python
PYTHONPATH=/exdata/jichengzhi/v2x_t1_attention/scripts/phase2 \
/data/jichengzhi_v2x/t2venv/bin/python scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py \
  --run-mswin-tvm-pilot --device cuda:0 --eval-samples 1 --num-workers 0 \
  --out-log logs/attention_e2e_pq_v1/mswin_tvm_qkv_int8_pilot1_v1.json
```

如果失败：

- 先看 traceback，不要改 validator。
- 常见可能点：Relax scalar shape、DLPack 0-d tensor、TVM compile target、CUDA visible device 映射、NumPy/torch import 顺序。

### Step 2: 补 HMSA TVM static `[0,0]` path

依据：

- 全量 DAIR val 实测 HMSA order 固定为 `[[0,0]]`。
- v3 static bench 的 `[0,1]` 不是当前 DAIR val 实际路径。

实现目标：

- 在 full model forward 内，HMSA 的 Q/K/V 或更完整 HMSA fusion-subgraph 进入 TVM mixed-INT8。
- 如果先只接 Q/K/V projection，需要在 measurement 里诚实写 scope；若收益不够，继续扩大到 relation/core/output projection。
- 最终 row 必须说明 `dynamic_type_dispatch_coverage=false` 和 `observed_type_orders=[[0,0]]`。

不接受：

- 只在 standalone HMSA subnet 上跑 TVM。
- 继续沿用 `[0,1]` static bench 作为当前 DAIR val 的证据。

### Step 3: 生成真实 full-model/fusion-subgraph measurement

建议 measurement 文件：

```text
results/attention_p50_tvm_mixed_int8_measurement_v1.json
```

必须包含：

```json
{
  "schema_version": "attention_tvm_mixed_int8_e2e_measurement_v1",
  "config": "attention-p50-int8/mixed",
  "attention_prune_pct": 50,
  "quant": "int8/mixed",
  "quant_backend": "TVM Relax int8/mixed",
  "checkpoint_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth",
  "manifest_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json",
  "finetune": "100_steps_lr1e-4_seed20260623_all_params",
  "dataset_split": "DAIR val",
  "n_samples": 1789,
  "latency": {
    "latency_scope": "e2e",
    "e2e_latency_ms": "<real>",
    "speedup": "<real>",
    "latency_command": "<real command>",
    "latency_log": "logs/attention_e2e_pq_v1/<real latency log>"
  },
  "ap": {
    "ap50": "<real>",
    "ap70": "<real>",
    "delta_ap50": "<real>",
    "delta_ap70": "<real>",
    "ap_command": "<real command>",
    "ap_log": "logs/attention_e2e_pq_v1/<real ap log>",
    "ap_source": "real_dair_val_eval"
  },
  "tvm": {
    "artifact_path": "<real TVM artifact or build cache>",
    "runtime": "TVM Relax VM",
    "mixed_precision_policy": "<actual covered ops>"
  },
  "dynamic_type_dispatch_coverage": {
    "status": "static_type_order_observed_on_full_dair_val",
    "covers_dynamic_type_dispatch": false,
    "observed_type_orders": [[0, 0]],
    "sample_count": 1789,
    "log_path": "logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json"
  }
}
```

### Step 4: 从 measurement 生成 final row

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py \
  --from-measurements results/attention_p50_tvm_mixed_int8_measurement_v1.json \
  --out-row results/attention_p50_tvm_mixed_int8_row_v1.json
```

### Step 5: 生成 final report 并跑 Stop-A validator

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md

/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

---

## 7. Stop-A 通过标准

必须同时满足：

```text
validator verdict == ACCEPTABLE_E2E_SCHEMA
attention-p50-int8/mixed speedup >= 1.10
delta_ap50 >= -0.02
delta_ap70 >= -0.02
```

人工验收必须确认：

- `latency_scope == e2e`
- `quant_backend == TVM Relax int8/mixed`
- AP 是真实 DAIR val eval，不是 simulated/fake/prior。
- baseline、attention-p50-shortft-fp16、attention-p50-int8/mixed 使用同一 DAIR val split。
- checkpoint 和 manifest 对应同一 50% structured attention head pruning。
- latency/AP command、log path、样本数、设备、runtime/artifact 可复核。
- dynamic HMSA coverage 诚实记录为当前 DAIR val 的 static `[0,0]`，而不是泛化到所有类型组合。

---

## 8. 如果下一阶段仍未达标

不要包装成成功。按下面顺序处理：

1. 如果 TVM attention path 没有真实参与 full-model forward，继续修 runner。
2. 如果只覆盖 MSwin 或只覆盖 HMSA QKV，先扩大到更多 attention fusion 子图。
3. 如果 speedup < 1.10，做 profiling，判定瓶颈是：
   - attention 本身不是 e2e 主瓶颈；
   - TVM launch/QDQ/DLPack overhead 抵消收益；
   - HMSA/MSwin dispatch 太碎；
   - mixed-INT8 覆盖面不足；
   - 非 attention 模块占主导。
4. 如果 AP 下降超过 -0.02，尝试 calibrated mixed-INT8、W8A16/weight-only INT8、或 QAT/短微调，但必须重新跑真实 AP。
5. 如果多轮真实 runner 后仍无法满足 Stop-A，写 Stop-C 负证据，不能进入 T2。

可尝试替代方案：

- QKV + output Linear 用 TVM mixed-INT8，softmax/LN 保 FP16。
- W8A16 或 weight-only INT8，降低 activation quantization 误差和 Q/DQ overhead。
- TVM fused QKV + Q/DQ + output projection，减少 kernel launch。
- TVM MetaSchedule/TensorIR 自定义 attention schedule。
- 针对 DAIR val static `[0,0]` 的 HMSA specialized lowering，再单独记录泛化边界。

---

## 9. 关键文件清单

设计和交接：

- `multi_agent/methods/design/plan_transformer_into_framework_v1.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v4.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v5.md`

runner/report/validator：

- `scripts/phase2/t1_attention_e2e_pq.py`
- `scripts/phase2/t1_attention_tvm_bench.py`
- `scripts/phase2/attention_subnet_accel_report.py`
- `scripts/phase2/attention_e2e_checkpoint_eval.py`
- `scripts/phase2/attention_final_acceptance_report.py`
- `scripts/phase2/attention_e2e_pq_validator.py`
- `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`

tests：

- `tests/phase2/test_attention_tvm_mixed_int8_e2e_runner.py`
- `tests/phase2/test_t1_attention_e2e_pq.py`
- `tests/phase2/test_attention_final_acceptance_report.py`
- `tests/phase2/test_attention_e2e_pq_validator.py`

已有结果：

- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_full_tvm_bench_v3_static.json`
- `results/attention_subnet_accel_v2_static.json`
- `results/attention_final_acceptance_v2_static.json`

目标结果：

- `results/attention_p50_tvm_mixed_int8_measurement_v1.json`
- `results/attention_p50_tvm_mixed_int8_row_v1.json`
- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`

checkpoint：

- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`

H800 coverage：

- `/exdata/jichengzhi/v2x_t1_attention/logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json`

---

## 10. 硬边界

- 不要进入 T2。
- 不要把 subnet v2/v3 static 写成 full-model e2e。
- 不要把 MSwin pilot 写成最终 Stop-A row。
- 不要用 fake-quant/prior/simulated AP。
- 不要用 direct matmul speedup 推算 e2e。
- 不要用 TensorRT/TRT/QDQ-ONNX 替代 TVM。
- 不要声称 dynamic HMSA dispatch 已完成；当前实测只是 DAIR val static `[0,0]`。
- 不要覆盖旧结果文件；新结果使用 `attention_*_v1` 或后续递增版本。
- 不要在文档、命令历史或汇报中写远程密码明文。

