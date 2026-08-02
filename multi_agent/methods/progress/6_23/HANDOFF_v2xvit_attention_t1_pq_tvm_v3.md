# HANDOFF — V2X-ViT Attention T1-P/T1-Q TVM Final Gate (v3, 2026-06-23)

> 2026-06-23 update: 下一阶段如果用 `/goal` 持续推进到 Stop-A 通过，请从中文入口
> [`HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v4.md`](HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v4.md)
> 开始。

> 新窗口接手先读本页。v2 是上一阶段入口; v3 记录 static-HMSA v3 subnet 已完成后的最新状态。

Background design:

- `multi_agent/methods/design/plan_transformer_into_framework_v1.md`

Current gate:

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

Reason:

- T1-P pruning/scanner is usable.
- T1-Q subnet-level TVM mixed-INT8 now has positive static-HMSA evidence.
- Full-model TVM mixed-INT8 e2e latency/AP row is still missing.
- Therefore Stop-A is still blocked and T2 must not start.

## Completed In This Stage

### H800 static-HMSA v3 TVM bench

Generated:

- `results/attention_full_tvm_bench_v3_static.json`
- `logs/attention_e2e_pq_v1/attention_full_tvm_bench_v3_static.log`

Remote command:

```bash
cd ${V2X_DATA_ROOT}/v2x_t1_attention
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path)
CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/t1_attention_tvm_bench.py \
  --out-json results/attention_full_tvm_bench_v3_static.json \
  --full-attention --full-only --number 3 --repeat 3
```

Results:

| target | FP16 p50 ms | mixed INT8 p50 ms | mixed/FP16 |
|---|---:|---:|---:|
| `mswin_bwa_full_attention` | 0.455243 | 0.464523 | 0.9800x |
| `mswin_bwa_p50_full_attention` | 0.243040 | 0.254997 | 0.9531x |
| `hmsa_full_attention` static | 0.473387 | 0.449429 | 1.0533x |
| `hmsa_p50_full_attention` static | 0.321408 | 0.287456 | 1.1181x |

HMSA scope:

```text
full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8
```

Coverage:

- q/k/v projection: covered
- relation_att/msg core: covered
- output projection: covered
- dynamic type dispatch: not covered

### Attention subnet v2 static

Generated:

- `results/attention_subnet_accel_v2_static.json`
- `results/attention_subnet_accel_v2_static.csv`
- `results/attention_subnet_accel_v2_static.md`

| config | prune | quant | subnet p50 ms | speedup vs base FP16 | speedup vs same-prune FP16 |
|---|---:|---|---:|---:|---:|
| `attention-subnet-base-fp16` | 0% | fp16 | 0.928630 | 1.0000x | 1.0000x |
| `attention-subnet-base-mixed-int8` | 0% | mixed_int8 | 0.913952 | 1.0161x | 1.0161x |
| `attention-subnet-p50-fp16` | 50% | fp16 | 0.564448 | 1.6452x | 1.0000x |
| `attention-subnet-p50-mixed-int8` | 50% | mixed_int8 | 0.542453 | 1.7119x | 1.0405x |

Decision:

```text
Do not Stop-C at subnet level.
Proceed to full-model/fusion-subgraph TVM mixed-INT8 runner.
```

### Final staging table v2 static

Generated:

- `results/attention_final_acceptance_v2_static.json`
- `results/attention_final_acceptance_v2_static.csv`
- `results/attention_final_acceptance_v2_static.md`

Status:

```text
BLOCKED_TVM_MIXED_INT8_E2E_MISSING
```

Stop-A validator result:

```text
REJECT
```

Expected reason:

- `attention-p50-int8/mixed` lacks e2e latency/AP/log/command.
- `latency_scope` is still `missing_tvm_e2e`, not `e2e`.

## Updated Docs

- `results/attention_e2e_pq_blocker_v2_static.md`
- `results/attention_e2e_pq_review_v2_static.md`

## Next Work

Start a new T1-Q implementation stage:

```text
Implement full-model or fusion-subgraph TVM mixed-INT8 runner for attention-p50-shortft.
```

Required final row:

```text
config: attention-p50-int8/mixed
checkpoint_path: models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth
manifest_path: models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json
quant_backend: TVM Relax int8/mixed
latency_scope: e2e
dataset_split: DAIR val
n_samples: 1789
```

Then generate:

```bash
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

Stop-A success criteria:

```text
validator verdict == ACCEPTABLE_E2E_SCHEMA
attention-p50-int8/mixed speedup >= 1.10
delta_ap50 >= -0.02
delta_ap70 >= -0.02
```

## Hard Boundaries

- Do not enter T2.
- Do not treat subnet v2 static as final e2e.
- Do not claim TVM mixed-INT8 AP preservation until the final e2e row exists.
- Do not claim dynamic HMSA dispatch is covered by v3 static-HMSA.
- Do not use fake-quant prior AP or direct-matmul results in final acceptance.
