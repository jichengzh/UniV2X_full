# 5_6_26_交接文档_阶段二_PhaseC_AP稳定产出规范

日期: 2026-06-26

序号: 5

继承文档:

- `4_6_26_交接文档_阶段二_AP稳定微调规则补充.md`
- `PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md`
- `RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 1. 本文目的

本文专门补齐 Phase C stable smoke 中“怎样稳定产出首批配置 AP”的可执行规则。后续 AP-agent 清上下文后, 必须优先读取本文和主计划的 `5.0 Phase C AP Stable Smoke 硬规则`。

核心结论:

1. Phase C 的 AP 不是预测值、拟合值或插值, 必须来自真实 full-model eval。
2. 首批新配置固定 5 个, 使用 H800 GPU0-4 五卡并行微调。
3. 每个新配置默认 1 次 finetune, `epoches=31`, 不做机械 repeat。
4. AP finetune/eval 不要求 GPU 完全空闲; 只要求记录并发状态并避免明显 OOM。
5. 数据集固定为 DAIR-V2X-C train split 微调, DAIR-V2X-C `val_1789` full eval。

## 2. 首批配置和 GPU 绑定

首批 original60 新配置如下, 不允许临时改成只跑 p50/p75:

| job_id | label | width | GPU | master_port | finetune_runs | epoches | train split | eval split |
|---|---|---|---:|---:|---:|---:|---|---|
| ap_ft_01 | s0_024 | [24,128,256] | 0 | 29700 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_02 | s0_040 | [40,128,256] | 1 | 29701 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_03 | s0_056 | [56,128,256] | 2 | 29702 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_04 | s1_048 | [64,48,256] | 3 | 29703 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_05 | s2_160 | [64,128,160] | 4 | 29704 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |

GPU 规则:

1. GPU0-4 五卡并行, 每张卡一个单卡 DDP 训练进程。
2. `CUDA_VISIBLE_DEVICES` 固定到表中 GPU。
3. `master_port` 固定为 `29700 + gpu_id`。
4. GPU5 默认留给 AP replay、digest 扫描、eval 补救或换卡。
5. AP finetune/eval 不套用 latency/energy 的 idle gate; 但必须记录 `nvidia-smi` 和 `nvidia-smi pmon -c 1`。
6. 如果某张卡显存明显不足或出现 OOM, 优先将该配置换到 GPU5; 若仍失败, 只 quarantine 当前配置, 其他配置继续。

## 3. 微调过程

每个新配置必须走完整链路:

```text
DAIR base ckpt
  -> structural_prune_pyramid.py 按 width 生成 full-model config + init ckpt
  -> 检查 config.yaml 中 epoches=31
  -> 检查 init ckpt 是 flat state_dict, 必要时 unwrap model_state_dict
  -> train_ddp.py --half 单卡 DDP 微调
  -> 锁定 best checkpoint
  -> 导出 full-model ONNX
  -> 构建 TRT FP16 engine
  -> DAIR-V2X-C val_1789 full AP eval
  -> 写 canonical AP row 或 quarantine row
```

默认微调预算:

| 项 | 默认值 |
|---|---|
| finetune_runs | 1 |
| epoches | 31 |
| init_epoch | 23 |
| effective_finetune_epochs | 约 8 |
| seed | 首批不强制多 seed; 若 runner 支持则记录 `20260626` |
| eval_runs | 1 次 full val_1789 |

这意味着首批 5 个新配置默认一共跑 5 次 finetune, 不是每个配置跑多轮。repeat 只用于异常排查和 paper-grade 子集。

追加训练或 repeat 的规则:

| 触发条件 | 动作 |
|---|---|
| AP70 明显高于 base 或不符合相邻宽度趋势 | 同配置补第 2 seed |
| best checkpoint 出现在最后 epoch | 延长到 `epoches=48` |
| 延长后仍不稳定或进入 paper claim 子集 | 延长到 `epoches=70`, 并做 2 seed |
| 训练、ONNX、TRT 或 eval 任一步失败 | quarantine 当前配置, 不写 measured AP |

## 4. 数据集和评测口径

H800 上固定使用 HEAL/OpenCOOD DAIR-V2X-C 口径:

```text
logical dataset: DAIR-V2X
HEAL dataset path: ${V2X_HOME}/heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure
preferred physical path: ${V2X_DATA_ROOT}/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure
train split: train.json, historical count ~= 4811
eval split: val.json, canonical name val_1789, num_samples=1789
```

AP row 字段必须包含:

```text
dataset=DAIR-V2X
dataset_train=DAIR-V2X-C train.json
eval_split=val_1789
num_samples=1789
metric=AP70
secondary_metrics.AP30
secondary_metrics.AP50
finetune_protocol=structural_prune_pyramid_train_ddp_half_epoches31_TRT_FP16_DAIR_val_1789
ap_source_kind=true_eval
claim_status=claimable_true_eval 或 provisional_missing_digest
```

不允许把以下结果写成 measured AP:

```text
predicted AP
model-fit AP
interpolated AP
mini-val AP
只有 backbone artifact、没有 full-model ckpt/config 的 AP
随机初始化或 ckpt 格式错误产生的 AP
```

## 5. 命令入口

队列文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/jobs/ap_finetune_smoke_queue_v1.jsonl
```

启动前 dry-run:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_h800_ap_finetune_smoke_runner.py \
  --base-dir "$BASE"
```

真实启动:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626
mkdir -p "$BASE/logs"
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_h800_ap_finetune_smoke_runner.py \
  --base-dir "$BASE" \
  --max-workers 5 \
  --execute \
  > "$BASE/logs/ap_finetune_smoke_runner.out" 2>&1 &
echo $! > "$BASE/jobs/ap_finetune_smoke_runner.pid"
```

## 6. 产物和验收

成功产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/rows/ap_anchor_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/raw/<label>/
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/exports/ap_stability_summary.md
```

失败产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/jobs/ap_finetune_smoke_job_state_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/quarantine/ap_unstable_or_unclaimable_v1.jsonl
```

Phase C AP stable smoke 完成条件:

1. 5 个新配置全部完成 full chain 并写入 true_eval AP row, 或全部有明确 quarantine reason。
2. 每个 row 能看到 label、width、GPU、finetune_runs、epoches、dataset/split、AP30/AP50/AP70、ckpt/config/source。
3. measured AP row 中 0 条来自 predicted/model-fit/interpolation。
4. `ap_stability_summary.md` 可以一眼审查配置、结果和失败原因。

## 7. 下一次启动提示

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 Phase C AP stable smoke。先阅读 multi_agent/methods/design/auto-tuning/progress/5_6_26_交接文档_阶段二_PhaseC_AP稳定产出规范.md、PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md 的 5.0 节和 RUNBOOK_stage2_h800_server_access_v1_zh.md。目标是稳定产出首批配置 AP: s0_024/s0_040/s0_056/s1_048/s2_160 使用 H800 GPU0-4 五卡并行微调, 每个配置默认 1 次 finetune, epoches=31, DAIR-V2X-C train.json 微调, DAIR-V2X-C val_1789 full eval AP30/AP50/AP70。AP finetune/eval 不要求 GPU 完全空闲, 但必须记录 nvidia-smi/pmon 和显存状态; OOM 或失败只 quarantine 当前配置。stop condition: 5 个新配置全部完成 full chain 或全部有明确 quarantine reason, 并生成 ap_stability_summary.md。
```
