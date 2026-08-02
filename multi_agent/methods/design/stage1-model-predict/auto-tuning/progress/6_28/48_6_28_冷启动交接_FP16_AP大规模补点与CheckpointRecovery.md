# 54_6_28_冷启动交接_FP16_AP大规模补点_CheckpointRecovery与四卡微调生成

## 0. 任务边界

本窗口只推进 FP16 AP 大规模补点, 不处理 INT8 精度崩溃。目标是把:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
FP16 AP measured: 5/60 -> 60/60
```

执行 GPU 固定为 H800 GPU3/GPU4/GPU5/GPU6。启动前重新审计这四张卡, 不依赖旧快照。

## 0.0 执行决议更新

基于 2026-06-28 最新口头决议, 本文后续执行口径统一为:

```text
1. 55 个 no_claim label 没有优先级顺序, 全量补齐。
2. checkpoint recovery 只做极短搜索; 目标是快速确认“基本无现成 AP checkpoint”而不是继续大规模盘点。
3. 默认训练预算先跑到 epoch 31; 只有没有 post-baseline bestval、AP 明显异常或训练证据不足时才延长到 epoch 48。
4. rows/fp16_true_original60_ap_rows_v1.jsonl 不做去重清理; 维持 append-only。
5. 启动时先做小批次闭环验证; 闭环通过后立即进入全量大规模生成+微调+AP eval。
6. 若小批闭环不通过, 必须定位根因并修正后继续, 不能因为首批失败而直接停止整个方向。
7. checkpoint recovery 的预期结果不是“找出很多现成 ckpt”, 而是用极短搜索快速确认大多数 label 仍需走生成/微调主线。
```

## 0.1 启动初期必读材料

新窗口启动后先读以下材料, 再执行任何远端任务:

```text
本文件:
multi_agent/methods/design/auto-tuning/progress/6_27/54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md

共享背景:
multi_agent/methods/design/auto-tuning/progress/6_27/53_6_28_交接文档_阶段三_GPU45_FP16Checkpoint阻断与INT8ScaleAwareBuilder就绪.md

当前 completion 状态:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json

FP16 checkpoint 审计:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_ap_checkpoint_inventory_20260628_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_ap_checkpoint_inventory_20260628_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_checkpoint_inventory_20260628_v2/fp16_missing_checkpoint_audit.json

现有 FP16 AP rows 与 blockers:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/

必须读源码:
scripts/stage2_h800_true_fp16_ap_eval.py
scripts/stage2_generate_original60_quant_state_coverage.py
scripts/stage2_generate_fp16_int8_original60_completion_queue.py
framework/stage2/original60_quant_completion.py
scripts/stage2_build_ap_stability_artifacts.py
scripts/phase2/dataset_a_prepare_ckpts.py
tools/structural_prune_pyramid.py
scripts/stage2_h800_ap_env_restore.sh
```

快速阅读命令:

```bash
cd ${V2X_ROOT}
sed -n '1,220p' multi_agent/methods/design/auto-tuning/progress/6_27/54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md
sed -n '1,180p' multi_agent/methods/design/auto-tuning/progress/6_27/53_6_28_交接文档_阶段三_GPU45_FP16Checkpoint阻断与INT8ScaleAwareBuilder就绪.md
sed -n '1,220p' scripts/stage2_h800_true_fp16_ap_eval.py
sed -n '800,890p' scripts/stage2_build_ap_stability_artifacts.py
sed -n '54,170p' scripts/phase2/dataset_a_prepare_ckpts.py
```

## 0.2 固定规则与红线

1. 单 agent 执行, 不启动 agent team。
2. 只使用 H800 GPU3/GPU4/GPU5/GPU6。启动前必须保存 `nvidia-smi` 与 compute-apps 审计结果。
3. 不明文写 H800 密码。只使用 `export '<REDACTED_LEGACY_SECRET>')"`。
4. latency 指标统一使用 ms; 本窗口不改 latency/energy rows。
5. FP16 latency/energy 的 `*_backbone_true_fp16.onnx` 只能证明 backbone-only TVM 测速, 不能当作 AP checkpoint。
6. FP16 AP measured row 只能来自 exact-label full model checkpoint + 1789-frame true eval。禁止用测速 ONNX、旧 smoke、partial eval、不匹配 checkpoint 或手填 AP 写 measured。
7. AP eval 必须保留 `full_network_claim=false`; 当前 AP 口径是 full model eval row, 不是 RSU 物理端到端部署 latency。
8. 每个 label 必须保存 raw artifact、runner pid、stdout/stderr、GPU id、eval command、AP30/AP50/AP70、ckpt/config digest。
9. 不能“蒙头找 checkpoint”。checkpoint recovery 必须 time-box; 没找到 exact-label checkpoint 后, 立即进入 checkpoint 生成/微调分支。
10. 找不到 checkpoint 且生成/微调也失败时, 才写 per-label blocker 并继续其他 label; 不能静默跳过, 也不能用相近宽度 checkpoint 代替。
11. structural prune / finetune 不得覆盖已有 checkpoint 目录中的 `.pth`; 发现已有 `.pth` 先审计再 resume/eval。
12. 不使用破坏性 git/文件操作, 不回滚用户已有改动。
13. 本窗口默认主线是“批量生成 + 微调 + full AP eval”, 不是 checkpoint inventory 扩展项目。
14. 四卡训练/评测允许非均匀调度; 不要求像 latency/energy 一样强制每张卡始终各挂一个独立进程。核心约束只有: 单卡同一时刻最多一个 train_ddp 或一个 full AP eval。
15. 微调确认启动后, 巡检不要做分钟级高频轮询。除非出现明确异常信号, 远端 train/eval/row 巡检统一控制在每 15-30 分钟一次。

## 1. 当前权威状态

路径根目录:

```text
${V2X_ROOT}
```

当前 completion review:

```text
FP16 latency = 60/60 measured
FP16 energy  = 60/60 measured
FP16 AP      = 5/60 measured
INT8 latency = 60/60 measured
INT8 energy  = 60/60 measured
INT8 AP      = 0/60 measured
total latency = 120/120
total energy  = 120/120
total AP      = 5/120
jobs_requiring_action = 115
```

权威文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_ap_checkpoint_inventory_20260628_latest.json
```

已完成 FP16 AP 的 5 个 label:

```text
s0_024, s0_040, s0_056, s1_048, s2_160
```

仍需补点的 55 个 label:

```text
frontier_01, frontier_02, frontier_03, frontier_04, frontier_05,
frontier_06, frontier_07, frontier_08, frontier_10, frontier_11,
frontier_12, frontier_14, frontier_15, frontier_16, frontier_17,
frontier_18, frontier_19, frontier_20, frontier_22, frontier_25,
frontier_26, frontier_27, frontier_31, frontier_32, frontier_33,
lhc_01, lhc_02, lhc_03, lhc_04, lhc_05, lhc_06, lhc_07,
lhc_08, lhc_09, lhc_10, lhc_11, lhc_12, lhc_14, lhc_15,
lhc_16, lhc_17, lhc_18, lhc_19, lhc_20, lhc_21, lhc_22,
lhc_23, lhc_24, lhc_25, lhc_26, s1_064, s1_080, s1_112,
s2_096, s2_224
```

## 2. 为什么现在不能直接大规模启动

FP16 latency/energy 60/60 使用的是 backbone-only ONNX/TVM artifact, 不是 AP eval 所需训练 checkpoint。

测速 row 的典型 evidence:

```text
source_files:
  ${V2X_DATA_ROOT}/s2_tvm/fp16_true_smoke_20260627/.../<label>_backbone_true_fp16.onnx
  ${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>
  ${V2X_DATA_ROOT}/s2_tvm/fp16_true_smoke_20260627/.../layer_precision_summary.json

optimized_scope = backbone_only
quant_scope = backbone_only
full_network_claim = false
input = spatial_features
```

AP eval row 必须来自 full model eval, 需要:

```text
ckpt_dir/config.yaml
ckpt_dir/net_epoch_bestval_at*.pth 或 ckpt_dir/net_epoch*.pth
1789-frame AP eval report
AP30/AP50/AP70
layer_precision_summary.json
ckpt_digest/config_digest
```

已完成 AP 的 5 个 label 使用的 checkpoint 形如:

```text
${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26/
  config.yaml
  net_epoch_bestval_at29.pth
```

当前 checkpoint inventory 只找到这 5 个 label 的 AP checkpoint。55 个 no_claim label 没有发现 `Pyramid_DAIR_m1_stage2_ap_<label>_*` 目录。因此, 不能用测速 ONNX 直接写 AP measured row。

## 2.1 已验证的真实 FP16 AP 生成闭环

本窗口已反查当前 5 个 measured label 的 row / raw artifact / 历史训练证据, 结论是:

```text
s0_024, s0_040, s0_056, s1_048, s2_160
```

都不是“从 latency/energy row 推导出来”的, 而是来自真实的 checkpoint + AP eval 闭环。

现有 row 证据:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
  quality_gate_status = true_fp16_original60_ap_eval
  provenance = stage2_original60_true_fp16_h800_model_eval
  raw_artifact = .../raw/ap_eval_original60/fp16_true_<label>_full_amp_fp16_v1
  ckpt_path = .../Pyramid_DAIR_m1_stage2_ap_<label>_2026_06_26/net_epoch_bestval_at*.pth
```

其上游训练来源可以在:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/jobs/ap_finetune_smoke_queue_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/raw/<label>/
```

中看到。该历史闭环的最小公共链路是:

```text
1. tools/structural_prune_pyramid.py
   从 Pyramid_DAIR_m1_base_2023_08_14_11_42_29 生成目标 width 的 ckpt_dir + config.yaml + init checkpoint

2. HEAL/opencood/tools/train_ddp.py --half
   单卡微调, 默认把 config 中 epoches patch 到 31
   典型产物:
     net_epoch25.pth
     net_epoch27.pth

## 2.2 2026-06-28 现地核查: “FP16 没启动”是误判

2026-06-28 下午再次核查远端 H800 GPU3/GPU4/GPU5/GPU6 后, 可以明确排除“FP16 AP 大规模补点没有启动”的判断。当前事实是: 4 条训练链路已经启动, 只是还没有完成到自动触发 true AP eval 并写入 row 的阶段。

远端主训练进程:

```text
frontier_06 -> PID 906865
frontier_07 -> PID 906863
frontier_08 -> PID 906950
frontier_15 -> PID 906948
```

对应 watcher 也已挂起等待训练完成后自动触发:

```text
scripts/stage2_original60_fp16_eval_when_ready.py
```

核查时 GPU compute-apps 可见上述 4 个训练主进程均占用 GPU3/GPU4/GPU5/GPU6 显存, 因此不是“没起进程”。

进一步的关键信号:

```text
1. train_stdout.txt 最新 tail 仍停在 epoch 26, 所以从纯日志看容易误判为“卡住/没启动”。
2. 但 ckpt_dir 中已实际产出 net_epoch27.pth:
   - frontier_06: net_epoch27.pth at 15:45
   - frontier_07: net_epoch27.pth at 15:46
   - frontier_15: net_epoch27.pth at 15:44
3. 这说明真实训练进度至少已经跨过 epoch 27, stdout 刷新与 checkpoint 生成不同步。
4. frontier_08 核查时仍只有 epoch25/bestval25, 说明其训练进度相对更慢。
```

因此当前更准确的判断应写成:

```text
不是“FP16 AP 没有启动”;
而是“FP16 AP 已经进入四卡训练阶段, 但尚未训练完成, watcher 尚未触发 true FP16 AP eval, row 也因此还没有继续增长”。
```

这次核查也再次确认原始 FP16 AP 不是由 latency/energy 60/60 backbone-only ONNX/TVM 直接推导得到, 而必须走:

```text
structural_prune_pyramid.py
-> train_ddp.py --half
-> stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16
-> rows/fp16_true_original60_ap_rows_v1.jsonl
```

截至本次同步, 本地权威状态为:

```text
fp16_ap_measured = 13
fp16_ap_no_claim = 47
```

已 measured 的 13 个 label:

```text
frontier_01
frontier_02
frontier_03
frontier_04
frontier_05
frontier_10
frontier_12
s0_024
s0_040
s0_056
s1_048
s1_064
s2_160
```

## 2.3 2026-06-28 后续推进快照: 已进入 epoch27

在后续复查中, 四个当前在跑的补点 label 已不再停留在 epoch26, 训练日志已经进入 epoch27, watcher 记录到的 best checkpoint 也随之上升:

```text
frontier_15:
  train stdout 已进入 epoch27
  eval_watcher_status.best_epoch_seen = 27
  ckpt_dir 已有 net_epoch27.pth + net_epoch_bestval_at27.pth

frontier_06:
  train stdout 已进入 epoch27
  eval_watcher_status.best_epoch_seen = 27
  ckpt_dir 已有 net_epoch27.pth + net_epoch_bestval_at27.pth

frontier_07:
  train stdout 已进入 epoch27
  eval_watcher_status.best_epoch_seen = 27
  ckpt_dir 已有 net_epoch27.pth + net_epoch_bestval_at27.pth

frontier_08:
  train stdout 尚未看到 epoch27 tail 回传时的 post-bestval 提升
  ckpt_dir 已有 net_epoch27.pth, 但 watcher 仍记录 bestval_at25
```

这一快照说明:

```text
1. 训练不是“卡在 epoch25/26”。
2. checkpoint 存档已经继续向前推进到 epoch27。
3. save_freq=2 / eval_freq=2 / epoches=31 的配置与当前产物一致。
4. 之所以 measured row 仍未继续增长, 是因为当前安全策略仍要求:
   训练完成标记出现后, watcher 才正式触发 true_fp16_ap_eval 写 row。
```

因此, 当前的等待时间主要由两部分构成:

```text
A. 单卡 2405 iter/epoch 的真实训练时间
B. 为避免 train/eval 同卡冲突而保留的“训练完成后再 eval”安全门
```

## 2.4 `true_fp16_ap_eval/` 目录的判读

后续复查时会在以下 raw 目录下看到 `true_fp16_ap_eval/`:

```text
fp16_ckpt_gen_frontier_15_gpu3_v1/true_fp16_ap_eval
fp16_ckpt_gen_frontier_06_gpu4_v1/true_fp16_ap_eval
fp16_ckpt_gen_frontier_07_gpu5_v1/true_fp16_ap_eval
```

这些目录容易被误读为“watcher 已经正式触发过新的 AP eval”。实际核查结论不是这样:

```text
1. 这些文件时间戳集中在 2026-06-28 15:27 左右。
2. runner_stdout.txt 只停在:
     resuming selected checkpoint at epoch 25
     ------ Loading Checkpoint ------
3. 没有 ap_eval_report.json / layer_precision_summary.json / 追加 row 的证据。
4. 当前 eval_watcher_status.json 仍然是:
     status = waiting_for_train_exit
     training_finished_marker = false
```

因此这里应当统一解释为:

```text
它们是早先误触发 / 被清理的旧 eval 尝试残留,
不是本轮“训练完成后正式派发”的 true FP16 AP 成果。
```

当前是否真的进入“正式 eval 阶段”, 必须以以下信号为准, 不能只看目录是否存在:

```text
A. eval_watcher_status.json.status 从 waiting_for_train_exit 变为 completed / eval_failed
B. 出现新的 ap_eval_report.json
C. rows/fp16_true_original60_ap_rows_v1.jsonl 实际新增对应 label
```

## 2.4.1 巡检频率约束

根据最新执行约束, 在四卡微调已经确认启动、且没有异常报警时:

```text
1. 不再做 1-2 分钟级别的人工巡检。
2. 远端状态检查、row 同步确认、文档化快照, 统一改为每 15-30 分钟一次。
3. 只有在出现如下异常信号时, 才允许打破该节奏提前巡检:
   - 训练主进程消失
   - eval_watcher_status 进入 blocked / eval_failed
   - row_count 异常跳变
   - 明确的 stderr / OOM / filesystem / import failure
```

本地自动观察器也应服从这一约束, 不应继续使用 120s 轮询。

## 2.5 2026-06-28 15:42 CST 再核查: 无停滞迹象

再次核查 `train_stdout.txt` 的 mtime 与 tail 后, 可以确认 4 个作业仍在继续前进, 当前“没有新 row”不能解释为训练挂起。

15:42 CST 左右的最新进度:

```text
frontier_15: epoch27 [2339/2405]
frontier_06: epoch27 [2151/2405]
frontier_07: epoch27 [1532/2405]
frontier_08: epoch27 [1150/2405]
```

对应 `train_stdout.txt` 的修改时间均在:

```text
2026-06-28 15:42:46.xxx +0800
```

因此当前结论应固定为:

```text
1. 训练进程仍然活跃, 并未停滞在 epoch27 早期。
2. 目前主要耗时仍是单卡继续跑完 epoch27->29->31 的训练本身。
3. 在 watcher 仍坚持“训练完成后再触发 true eval”的安全门前提下,
   row_count=13 持续不变属于预期现象, 不能单独视为异常。
```
     net_epoch29.pth
     net_epoch31.pth
     net_epoch_bestval_at*.pth

3. full AP eval
   历史 5 点最早是 ap_stability_20260626 的 true-eval 闭环产物
   当前 original60 权威口径统一收敛到:
     scripts/stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16

4. append row
   gate 通过后追加:
     rows/fp16_true_original60_ap_rows_v1.jsonl
   再刷新 completion / coverage / review
```

因此, 对剩余 55 个 label, 当前最短主线就是:

```text
checkpoint recovery 极短搜索
-> 无 exact-label ckpt 则 structural prune
-> train_ddp --half 跑到 31
-> stage2_h800_true_fp16_ap_eval.py
-> append row + refresh review
```

不需要再把 latency/energy backbone-only artifact、额外的 TRT/INT8 route 或非 AP 闭环混进这个主线。

## 3. H800 连接与环境

不要明文写密码。使用:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

远端常用路径:

```text
V2X_ROOT=${V2X_ROOT}
HEAL_ROOT=${V2X_HOME}/heal_research/HEAL
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
ROWS=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
RAW_ROOT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60
```

启动前审计 GPU3/GPU4/GPU5/GPU6:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
```

### 3.1 2026-06-28 当前远端运行状态

本窗口已重新审计 H800, 结果显示 GPU3/GPU4/GPU5/GPU6 并非空闲, 而是已经有本方向的 4 个 FP16 train_ddp 进程在运行:

```text
GPU3: frontier_10
  ${V2X_DATA_ROOT}/heal_research/HEAL/opencood/tools/train_ddp.py
  --model_dir ${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_10_2026_06_28

GPU4: frontier_12
  ${V2X_DATA_ROOT}/heal_research/HEAL/opencood/tools/train_ddp.py
  --model_dir ${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_12_2026_06_28

GPU5: frontier_01
  ${V2X_DATA_ROOT}/heal_research/HEAL/opencood/tools/train_ddp.py
  --model_dir ${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_01_2026_06_28

GPU6: s1_064
  ${V2X_DATA_ROOT}/heal_research/HEAL/opencood/tools/train_ddp.py
  --model_dir ${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s1_064_2026_06_28
```

对应 launcher pid:

```text
frontier_10 -> 3390165
frontier_12 -> 3390194
frontier_01 -> 3390209
s1_064     -> 3390176
```

对应 watcher 也已经启动:

```text
scripts/stage2_original60_fp16_eval_when_ready.py
  frontier_10 -> pid 3550163
  frontier_12 -> pid 3550160
  frontier_01 -> pid 3550282
  s1_064      -> pid 3550283
```

训练证据表明这些作业已经推进到 epoch 30 附近, 并非“FP16 还没有启动”。例如:

```text
frontier_10:
  ckpt_dir 已有 net_epoch25 / 27 / 29 / 31 / bestval_at29

frontier_12:
  ckpt_dir 已有 net_epoch25 / 27 / 29 / bestval_at29

frontier_01:
  ckpt_dir 已有 net_epoch25 / 27 / 29 / bestval_at23 / bestval_at29

s1_064:
  ckpt_dir 已有 net_epoch25 / 27 / 29 / bestval_at25
```

因此当前需要重点检查的是:

```text
1. train_ddp 是否正常收尾退出
2. eval watcher 是否在 train 退出后正确转入 true FP16 AP eval
3. eval/import 完成后是否成功 append 到 rows/fp16_true_original60_ap_rows_v1.jsonl
```

### 3.2 本窗口新增根因判断与修复

本窗口继续向下追查后确认, 训练 stderr 中出现的:

```text
sh: 1: python: not found
```

来自 HEAL 原生 `train_ddp.py` 收尾阶段的内部调用:

```python
cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
os.system(cmd)
```

该调用不是 Stage2 true FP16 AP eval 主线, 但它要求 launcher 环境中的 `PATH` 可解析 `python`。原始
`scripts/stage2_original60_fp16_train_launcher.py` 只设置了:

```text
CUDA_VISIBLE_DEVICES
PYTHONPATH
```

没有保证 `env_python` 所在目录进入 `PATH`, 因此 HEAL 内部 shell 调 `python ...` 会报错。

已经完成的修复:

```text
文件:
  scripts/stage2_original60_fp16_train_launcher.py

新增:
  build_env(gpu_id, heal_root, env_python)

行为:
  1. 继续设置 CUDA_VISIBLE_DEVICES
  2. 继续设置 PYTHONPATH
  3. 将 Path(env_python).parent prepend 到 PATH
```

对应测试:

```text
framework/tests/test_stage2_original60_fp16_train_launcher.py
```

验证命令:

```bash
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_original60_fp16_train_launcher \
  framework.tests.test_stage2_original60_quant_completion \
  framework.tests.test_stage2_original60_fp16_ap_launch_plan \
  framework.tests.test_stage2_original60_fp16_ap_batch_launcher
```

说明:

```text
1. 该修复已同步到 H800 的 ${V2X_ROOT}/scripts/stage2_original60_fp16_train_launcher.py
2. 该问题影响的是“后续批次新启动的 train launcher 环境”
3. 当前已进入 eval 的几个 label 不需要依赖 train_ddp 内部 inference 成功, 因为 Stage2 权威 AP 行仍由
   scripts/stage2_h800_true_fp16_ap_eval.py 追加
```

### 3.3 2026-06-28 当前 true FP16 AP eval 进展

继续审计后确认, 上述 4 个 train job 中:

```text
frontier_10
frontier_12
s1_064
```

已经各自生成 `true_fp16_ap_eval/` 目录, watcher 已经拉起:

```text
scripts/stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16
```

其 raw 目录下可见:

```text
eval_dispatch.json
gpu_preflight.json
runner_command.json
runner_stdout.txt
runner_stderr.txt
```

当前 stderr 只见通用 warning, 尚未看到早期 fatal traceback:

```text
timm FutureWarning
torch.utils.cpp_extension pkg_resources warning
```

`frontier_01` 当前尚未看到 `true_fp16_ap_eval/` 目录, 仍需继续盯 watcher 切换。

注意:

```text
eval_watcher_status.json 当前仍可能停留在 waiting_for_train_exit。
这不是新的 blocker 证据, 因为 watcher 在 subprocess.run(eval) 返回前不会刷新完成态。
是否真正推进, 以 true_fp16_ap_eval/ 目录、runner stdout/stderr、以及 rows/fp16_true_original60_ap_rows_v1.jsonl
是否追加 measured row 为准。
```

### 3.4 AP 异常点处理口径

截至本窗口当前状态, `frontier_01` 已经成功:

```text
1. 使用 net_epoch31.pth 而非 bestval_at23
2. 跑通 true FP16 AP eval
3. append 进 rows/fp16_true_original60_ap_rows_v1.jsonl
4. 在本地 completion queue 中变为 ap_status=measured
```

但是它的 AP 值明显异常:

```text
AP30 = 3.288157873745785e-07
AP50 = 0.0
AP70 = 0.0
```

因此它不能被视为“最终高质量收口样本”。按本窗口既定执行口径:

```text
若 AP 明显异常, 需要从默认 epoch31 转入 epoch48 延长修复。
```

建议处理方式:

```text
1. 保留当前 append-only row 作为已执行证据, 不删除。
2. 将 frontier_01 记为“measured but abnormal”, 排入下一轮 31->48 修复队列。
3. 修复时优先复用同一 ckpt_dir, 将 config epoches patch 到 48 后继续训练, 再重新执行
   scripts/stage2_h800_true_fp16_ap_eval.py。
4. 后续若修复成功, 继续 append 新 row, 由最新合格 row 覆盖 completion 口径。
```

## 4. 第一阶段: 重新生成待补队列

在本地或 H800:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python - <<'PY'
import json, pathlib
queue = pathlib.Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl")
for line in queue.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("precision") == "fp16" and row.get("ap_status") == "no_claim":
        print(row["label"], ",".join(map(str, row["width"])))
PY
```

这应输出 55 个 label。若不是 55, 先刷新:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

## 5. 第二阶段: time-box checkpoint recovery

对每个 no_claim label 搜索 exact-label checkpoint。候选必须同时满足:

```text
目录名或 manifest 能对应当前 label
存在 config.yaml
存在 net_epoch_bestval_at*.pth 或 net_epoch*.pth
宽度/配置与 completion queue 中 label 的 width 一致或有明确证据可追溯
```

搜索 roots:

```text
${V2X_HOME}/heal_research/checkpoints
${V2X_DATA_ROOT}/heal_research/checkpoints
${V2X_ROOT}/results
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60
```

远端示例:

```bash
label=frontier_01
for root in \
  ${V2X_HOME}/heal_research/checkpoints \
  ${V2X_DATA_ROOT}/heal_research/checkpoints \
  ${V2X_ROOT}/results \
  ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60
do
  test -d "$root" || continue
  find "$root" -type d \( -iname "*${label}*" -o -iname "*stage2_ap_${label}*" \) 2>/dev/null | head -n 50
done
```

checkpoint recovery 只能作为极短前置步骤, 不能长时间“蒙头找”。建议策略:

```text
每个 label 先做自动 inventory + 明确 roots 顺扫, 不做复杂搜索排序。
单 label 搜索只保留 2-5 分钟窗口; 重点是确认“是否存在可直接进入 full AP eval 的 exact-label checkpoint”, 不是继续做历史资产考古。
若短窗口内没有可用 exact-label checkpoint 证据, 立即进入 checkpoint 生成/微调分支。
只有生成/微调也失败, 才写 per-label blocker。
```

checkpoint 合格判据进一步固定为:

```text
1. label 必须 exact match, 不能用相邻宽度或相似命名目录替代。
2. 必须同时存在 config.yaml 与 net_epoch_bestval_at*.pth / net_epoch*.pth。
3. width/config 必须与 completion queue 中当前 label 一致, 且可追溯到同一目标配置。
4. checkpoint 必须是可用于 full model true AP eval 的训练产物; 仅有测速 artifact、smoke artifact、structural-prune 初始点、partial eval 残留目录都不算可直接 claim 的 AP checkpoint。
5. 若只能证明“目录存在”但不能证明其是 exact-label full AP 可用训练 checkpoint, 一律按不可用处理, 直接进入生成/微调主线。
```

已有 blocker 根:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

## 6. 第三阶段: checkpoint 生成与四卡微调

如果没有可用 exact-label checkpoint, 直接生成。已有可复用路线是:

```text
tools/structural_prune_pyramid.py
  从 ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29
  生成目标 width 的初始 checkpoint/config。

opencood/tools/train_ddp.py --half
  单卡 nproc_per_node=1, AMP 微调, 从生成目录 resume。
```

本阶段默认执行理念:

```text
1. 主线不是“找现成 ckpt”, 而是“快速确认没有后立刻生成”。
2. 训练任务允许不均匀发车; 不强制四张卡同时起 4 个任务。
3. 以吞吐与稳定性为主: 某张卡训练耗时更长时, 其他卡可继续推进别的 label, 无需等待齐步走。
```

启动前必须做 preflight:

```bash
cd ${V2X_ROOT}
test -x ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
test -d ${V2X_HOME}/heal_research/HEAL
test -d ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29
test -f tools/structural_prune_pyramid.py
test -f scripts/stage2_h800_ap_env_restore.sh
```

若 H800 环境缺失, 先运行:

```bash
cd ${V2X_ROOT}
bash scripts/stage2_h800_ap_env_restore.sh
```

单 label 生成/微调模板:

```bash
cd ${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
HEAL_ROOT=${V2X_HOME}/heal_research/HEAL
BASE=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29

label=frontier_01
width=24,64,128
gpu=3
port=$((29700 + gpu))
ckpt_dir=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_${label}_2026_06_28
raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_ckpt_gen_${label}_gpu${gpu}_v1
mkdir -p "$raw_dir"

CUDA_VISIBLE_DEVICES="$gpu" "$PY" tools/structural_prune_pyramid.py \
  --orig-dir "$BASE" \
  --out-dir "$ckpt_dir" \
  --num-filters-new "$width" \
  --width-per-group 4 \
  --groups 32 \
  > "$raw_dir/structural_prune_stdout.txt" \
  2> "$raw_dir/structural_prune_stderr.txt"

# 快速生成 checkpoint: 先确认/调整 epoches。
# 第一批可跑到 epoch 31 附近; 若没有 post-baseline bestval 或 AP 明显异常, 再延长到 epoch 48。
export CKPT_DIR="$ckpt_dir"
export TARGET_EPOCHES=31
python - <<'PY'
from pathlib import Path
import os, re
cfg = Path(os.environ["CKPT_DIR"]) / "config.yaml"
text = cfg.read_text()
target = int(os.environ.get("TARGET_EPOCHES", "31"))
patched = re.sub(r"epoches:\s*\d+", f"epoches: {target}", text)
if patched != text:
    cfg.write_text(patched)
print(f"config={cfg} target_epoches={target}")
PY

PYTHONPATH="$HEAL_ROOT" CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env \
  --master_port="$port" \
  "$HEAL_ROOT/opencood/tools/train_ddp.py" \
  --hypes_yaml "$ckpt_dir/config.yaml" \
  --model_dir "$ckpt_dir" \
  --half \
  > "$raw_dir/train_stdout.txt" 2> "$raw_dir/train_stderr.txt" &
echo $! > "$raw_dir/train_runner.pid"
```

四卡调度规则:

```text
每张卡同时最多一个 train_ddp 或 AP eval。
全局以 GPU3/GPU4/GPU5/GPU6 为资源池, 不要求每张卡必须同时都有进程。
每个 label 先 structural prune, 再 train_ddp --half, 再 AP eval。
训练阶段优先保证“有空卡就发下一个 label”; eval 阶段优先插入已完成训练的 label, 不要求 train/eval 完全对称铺满。
如果 checkpoint 目录已有 .pth, 不重新 structural prune, 先 resume/eval。
所有 train stdout/stderr、runner.pid、gpu id、ckpt_dir、width 必须落盘。
```

checkpoint 生成验收:

```bash
find "$ckpt_dir" -maxdepth 1 -name 'net_epoch_bestval_at*.pth' -o -name 'net_epoch*.pth'
test -f "$ckpt_dir/config.yaml"
```

若只生成 `net_epoch_bestval_at23.pth` 这类 structural-prune 初始 checkpoint, 不能直接当最终 AP checkpoint; 必须至少有 post-baseline training 产物或明确记录为低置信 trial, 并优先继续微调。

### 6.1 小批闭环验证

大规模铺开前先做小批闭环验证, 目的不是补点数量, 而是确认生成/微调/AP import 链路可稳定复用。

建议闭环:

```text
1. 任选 2-4 个 no_claim label, 覆盖至少 2 张 GPU。
2. 对每个 label 走完整链路: queue width -> structural prune -> train_ddp --half -> stage2_h800_true_fp16_ap_eval.py --execute -> rows append -> coverage refresh。
3. 闭环通过标准:
   - train 至少跑到 epoch 31 方案并产出 post-baseline .pth
   - ap_eval_report.json num_samples = 1789
   - rows 追加成功
   - 无路径错误、环境错误、dtype 错误、ckpt import 错误
4. 若闭环失败, 必须先记录 failure reason 与 root cause, 修复同类问题后重试闭环, 不能直接停止总任务。
5. 闭环通过后, 立即切换到全量批量生成; 不再额外拉长 checkpoint 搜索阶段。
```

### 6.2 已知阻塞与对应优化

1. checkpoint 基本缺失:
   这是主线前提, 不是新的崩溃点。优化方式不是继续扩大 inventory, 而是把搜索窗口压缩到 2-5 分钟后直接转入 structural prune + 微调生成。

2. 四卡训练调度:
   不采用 latency/energy 阶段那种“每卡固定一个独立进程且始终铺满”的刚性编排。训练阶段使用 GPU3/4/5/6 资源池调度: 哪张卡空闲就发哪个下一个 label; 单卡同一时刻最多一个 train_ddp 或一个 AP eval 即可。

3. checkpoint 可用性误判:
   最容易拖慢节奏的不是“没有目录”, 而是把 structural-prune 初始点、旧 smoke 目录、partial eval 目录误当成可直接 AP 导入的 checkpoint。优化方式是按第 5 节合格判据做硬门控, 不满足就直接继续微调, 不在灰区反复试错。

4. 小批闭环首批失败:
   首批失败不能当作整条链路终止信号。优化方式是把失败样本保留为诊断用例, 先定位 root cause, 修复后用同类 label 回放验证, 验证通过再扩到大规模批量。

5. partial eval 或数据不完整:
   `stage2_h800_true_fp16_ap_eval.py` 若 `num_samples < 1789`, 结果只能作为 blocker 或诊断 artifact, 不可导入 measured row。优化方式是优先修正数据/环境/路径后重跑同 label, 而不是保留半成品结果。

6. 远端环境漂移:
   启动前强制 preflight + 必要时执行 `scripts/stage2_h800_ap_env_restore.sh`; 闭环阶段优先暴露环境问题, 不把问题留到 55 label 全量展开后再处理。

## 7. 第四阶段: GPU3/GPU4/GPU5/GPU6 分片启动 full AP

单 label 命令模板:

```bash
cd ${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
label=frontier_01
width=24,64,128
gpu=3
ckpt_dir=/path/to/exact/Pyramid_DAIR_m1_stage2_ap_frontier_01_...
raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_${label}_full_amp_fp16_v1
rows=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl

nohup "$PY" scripts/stage2_h800_true_fp16_ap_eval.py \
  --label "$label" \
  --width "$width" \
  --ckpt-dir "$ckpt_dir" \
  --raw-dir "$raw_dir" \
  --rows-out "$rows" \
  --execute \
  --precision-mode amp_fp16 \
  --gpu-id "$gpu" \
  --run-id "20260628_fp16_true_ap_${label}_full_amp_fp16_gpu${gpu}_v1" \
  --created-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$raw_dir/stdout.txt" 2> "$raw_dir/stderr.txt" &
echo $! > "$raw_dir/runner.pid"
```

并行策略:

```text
每个 GPU 同时只跑 1 个 full AP eval
不要求四张卡同时都在跑 eval; eval 可以插空到已经完成训练的卡上。
每个 label 完成后检查 ap_eval_report.json 和 row 是否追加成功
```

不要把 smoke、partial eval 或旧 row 写成 measured。AP row 必须是 1789-frame true eval。

## 7.1 2026-06-28 14:43 CST 当前实测状态

这轮不是“FP16 还没启动”, 而是已经进入四卡并行微调阶段, 尚未全部跑到 `epoch 31 -> full AP eval` 的切换点。

本地/远端当前一致事实:

```text
FP16 AP measured: 9/60
FP16 AP no_claim: 51/60
远端 rows/fp16_true_original60_ap_rows_v1.jsonl 当前 9 个 label:
s0_024, s0_040, s0_056, s1_048, s1_064, s2_160, frontier_01, frontier_10, frontier_12
```

2026-06-28 14:43 CST 远端 H800 GPU3-6 审计:

```text
GPU3: 10963 MiB / 81559 MiB, util 28%, compute-app pid 68737
GPU4: 11065 MiB / 81559 MiB, util 23%, compute-app pid 69690
GPU5: 10043 MiB / 81559 MiB, util 28%, compute-app pid 161964
GPU6: 11101 MiB / 81559 MiB, util 30%, compute-app pid 71009
```

对应 lane:

```text
GPU3 -> frontier_05, watcher=waiting_for_train_exit, best_epoch_seen=25
GPU4 -> frontier_02, watcher=waiting_for_train_exit, best_epoch_seen=25
GPU5 -> frontier_03, watcher=waiting_for_train_exit, best_epoch_seen=25
GPU6 -> frontier_04, watcher=waiting_for_train_exit, best_epoch_seen=25
```

当前 checkpoint 进度:

```text
frontier_05: net_epoch25.pth + net_epoch_bestval_at25.pth
frontier_02: net_epoch25.pth + net_epoch_bestval_at25.pth
frontier_03: net_epoch25.pth, 目前仍只有 baseline bestval_at23 + plain epoch25
frontier_04: net_epoch25.pth + net_epoch_bestval_at25.pth
```

结论:

```text
1. 问题不在“本来的 FP16 AP 生成链路不清楚”, 该链路已核实并已在 9 个 measured label 上复现。
2. 当前主耗时来自 train_ddp 单卡跑到 epoch31 的正常训练时长, 不是启动失败。
3. 现阶段不应再回到大范围 inventory/文档学习, 而应持续监控四个 lane 到 train exit, 随后自动进入 true FP16 AP eval, 再立即接续下一批 label。
```

### 7.2 已启动的四条长期 lane runner

为避免当前 4 个 label 完成后还依赖人工补发, 已在远端启动四条顺序消费 lane:

```text
GPU3 lane runner pid: 353050
GPU4 lane runner pid: 352973
GPU5 lane runner pid: 352970
GPU6 lane runner pid: 353133
pid 文件目录:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_lane_gpu{3,4,5,6}.pid
状态文件目录:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_lane_gpu{3,4,5,6}_status.json
```

当前每条 lane 的顺序分配:

```text
GPU3: frontier_01, frontier_05, frontier_10, frontier_15, frontier_19, frontier_26, frontier_33, lhc_04, lhc_08, lhc_12, lhc_17, lhc_21, lhc_25, s1_112
GPU4: frontier_02, frontier_06, frontier_11, frontier_16, frontier_20, frontier_27, lhc_01, lhc_05, lhc_09, lhc_14, lhc_18, lhc_22, lhc_26, s2_096
GPU5: frontier_03, frontier_07, frontier_12, frontier_17, frontier_22, frontier_31, lhc_02, lhc_06, lhc_10, lhc_15, lhc_19, lhc_23, s1_064, s2_224
GPU6: frontier_04, frontier_08, frontier_14, frontier_18, frontier_25, frontier_32, lhc_03, lhc_07, lhc_11, lhc_16, lhc_20, lhc_24, s1_080
```

说明:

```text
1. lane runner 会自动跳过已 measured 的 label, 因此 frontier_01 / frontier_10 / frontier_12 / s1_064 不会重复执行。
2. 当前 frontier_05 / frontier_02 / frontier_03 / frontier_04 完成训练并写入 measured row 后, lane runner 会自动启动下一项, 不需要再手工发 train launcher。
3. 若某条 lane 出现 eval_failed 或 blocked, 先保留 raw artifact 与 status.json, 再对单条 lane 修复, 不影响其他三条 lane 继续推进。
```

### 7.3 本地同步与刷新

已新增本地脚本:

```text
scripts/stage2_sync_original60_fp16_ap_progress.py
```

用途:

```text
1. 从远端 H800 拉取最新 rows/fp16_true_original60_ap_rows_v1.jsonl
2. 本地重跑:
   - stage2_generate_original60_quant_state_coverage.py
   - stage2_generate_fp16_int8_original60_completion_queue.py
   - stage2_generate_original60_quant_ap_true_eval_queue.py
3. 打印最新 fp16_ap_measured / fp16_ap_no_claim
```

使用方式:

```bash
# Use SSH-key authentication configured in ~/.ssh/config; do not set a password variable.
```

仅验证本地刷新链路而不拉远端:

```bash
python scripts/stage2_sync_original60_fp16_ap_progress.py --skip-sync
```

2026-06-28 当前 `--skip-sync` 验证结果:

```text
fp16_ap_measured = 9
fp16_ap_no_claim = 51
measured_labels = frontier_01, frontier_10, frontier_12, s0_024, s0_040, s0_056, s1_048, s1_064, s2_160
```

真实远端拉取验证:

```bash
# Use SSH-key authentication configured in ~/.ssh/config; do not set a password variable.
```

2026-06-28 14:50 CST 结果:

```text
远端 rows 仍为 9 条, 本地 refresh 后仍得到:
fp16_ap_measured = 9
fp16_ap_no_claim = 51
```

### 7.4 2026-06-28 14:50 CST 短轮询确认

短轮询后确认首批四条 lane 不是 silent stall, 而是继续向 `epoch31` 正常推进:

```text
frontier_05:
  watcher best_epoch_seen = 27, best_checkpoint = net_epoch_bestval_at27.pth
  train_stdout 已推进到 [epoch 27][1611/2405]

frontier_02:
  watcher best_epoch_seen = 27, best_checkpoint = net_epoch_bestval_at27.pth
  train_stdout 已推进到 [epoch 27][2329/2405]

frontier_03:
  watcher best_epoch_seen = 25, best_checkpoint = net_epoch_bestval_at25.pth
  train_stdout 已推进到 [epoch 26][1564/2405]

frontier_04:
  watcher best_epoch_seen = 27, best_checkpoint = net_epoch_bestval_at27.pth
  train_stdout 已推进到 [epoch 27][1869/2405]
```

结论:

```text
1. watcher 当前没有父进程选错问题; train_pid 指向 torch.distributed.launch 父进程, 仍然存活。
2. 当前没有 evidence 表明四条 lane 进入 deadlock 或卡死; 只是还没跑到 train exit。
3. 下一观察点应是:
   - 是否写出 net_epoch29 / net_epoch31 / post-baseline bestval_at31
   - train 退出后 watcher 是否创建 true_fp16_ap_eval/
   - rows 是否从 9 条增长到 10+ 条
```

### 7.5 本地自动监控 watcher

已新增本地监控脚本:

```text
scripts/stage2_watch_original60_fp16_progress.py
```

职责:

```text
1. 轮询远端 rows/fp16_true_original60_ap_rows_v1.jsonl 的 row_count 与 label 集合
2. 轮询远端 GPU3-6 的 lane status.json
3. 一旦 row_count 增长, 自动触发:
   scripts/stage2_sync_original60_fp16_ap_progress.py
   以刷新本地 coverage/review
```

本地后台 watcher 已启动:

```text
pid: 1970458
pid 文件:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_original60_remote_watch.pid
stdout:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_original60_remote_watch.stdout
stderr:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_original60_remote_watch.stderr
state:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_original60_remote_watch_state_latest.json
log:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_original60_remote_watch_log_latest.jsonl
poll_seconds: 120
```

当前 watcher 首次记录:

```text
row_count = 9
lane_signature:
  GPU3 -> frontier_05 running
  GPU4 -> frontier_02 running
  GPU5 -> frontier_03 running
  GPU6 -> frontier_04 running
```

### 7.6 2026-06-28 15:18 CST 最新闭环结果

已确认第一批新增 4 个 FP16 true AP 点落地, 远端 rows 从 9 增长到 13:

```text
新增 measured label:
frontier_02
frontier_03
frontier_04
frontier_05
```

对应 AP:

```text
frontier_05: AP30=0.7904, AP50=0.7499, AP70=0.5966
frontier_02: AP30=0.7874, AP50=0.7462, AP70=0.5926
frontier_03: AP30=0.7851, AP50=0.7440, AP70=0.5897
frontier_04: AP30=0.7883, AP50=0.7490, AP70=0.5918
```

本地同步后状态:

```text
fp16_ap_measured = 13
fp16_ap_no_claim = 47
```

这里还定位并修复了一个关键阻塞:

```text
HEAL train_ddp.py 在 Training Finished 后会自动执行:
python opencood/tools/inference.py --model_dir ...
```

这一步对当前 Stage2 true FP16 AP row 没有价值, 但会拖住 watcher 等待 train_pid 退出。修复方式:

```text
1. 更新 scripts/stage2_original60_fp16_eval_when_ready.py:
   允许在 train_stdout 出现 "Training Finished" 后直接放行 true FP16 AP eval
2. 清掉 frontier_05 / frontier_02 / frontier_04 的尾部 inference 进程
3. 重启 watcher, 使其立即 dispatch true FP16 AP eval
```

当前第二批已经自动续跑, 不需要人工补发:

```text
GPU3 -> frontier_15
GPU4 -> frontier_06
GPU5 -> frontier_07
GPU6 -> frontier_08
```

截至 15:18 CST 第二批进度:

```text
frontier_06 / frontier_07 / frontier_08 / frontier_15
均仍在 epoch 23 中段
当前 ckpt 仍只有 net_epoch_bestval_at23.pth
watcher 状态仍是 best_epoch_seen = -1
```

截至 16:23 CST 的再审计快照:

```text
本地 rows 同步后仍为:
  fp16_ap_measured = 13
  fp16_ap_no_claim = 47

远端四卡 lane runner 全部存活:
  GPU3 -> frontier_15
  GPU4 -> frontier_06
  GPU5 -> frontier_07
  GPU6 -> frontier_08

远端 H800 资源:
  GPU3 memory.used = 8709 MiB, util = 33%
  GPU4 memory.used = 8355 MiB, util = 35%
  GPU5 memory.used = 8071 MiB, util = 31%
  GPU6 memory.used = 8451 MiB, util = 33%
  compute-apps 均为 UniV2X_2.0 环境下的 train_ddp 主进程

第二批 4 个 label 当前都已推进到 epoch 24:
  frontier_06 -> [epoch 24][626/2405]
  frontier_07 -> [epoch 24][444/2405]
  frontier_08 -> [epoch 24][379/2405]
  frontier_15 -> [epoch 24][656/2405]

四个 watcher 当前状态一致:
  status = waiting_for_train_exit
  training_finished_marker = false
  best_checkpoint_seen = null
  best_epoch_seen = -1

结论:
  当前不是“未启动”, 而是第二批 4 个 label 正处于正常训练阶段;
  由于尚未到 post-baseline bestval/checkpoint 产生节点, rows 还没有继续增长。
```

截至 2026-06-28 15:26 CST 的进一步审计:

```text
远端自动 watch 仍在正常轮询:
  exports/fp16_original60_remote_watch_state_latest.json
  loop = 16
  row_count = 13
  changed = false
  lane_signature:
    GPU3 -> frontier_15 (index=3, running)
    GPU4 -> frontier_06 (index=1, running)
    GPU5 -> frontier_07 (index=1, running)
    GPU6 -> frontier_08 (index=1, running)

第二批 train stdout 持续前进, 说明不是假活跃:
  frontier_06 -> [epoch 24][1210/2405]
  frontier_07 -> [epoch 24][1007/2405]
  frontier_08 -> [epoch 24][924/2405]
  frontier_15 -> [epoch 24][1274/2405]

四个 ckpt 目录当前仍只有:
  config.yaml
  net_epoch_bestval_at23.pth

四个 watcher 仍未看到 post-baseline bestval:
  status = waiting_for_train_exit
  training_finished_marker = false
  best_checkpoint_seen = null
  best_epoch_seen = -1

结论:
  当前无需人工干预。
  只要训练继续推进到新的 bestval / Training Finished, watcher 就会接手 true FP16 AP eval。
```

截至 2026-06-28 15:28 CST 的一次提速调整:

```text
为了缩短“训练结束 -> watcher 发现 -> true AP eval 派发”之间的空窗,
已将以下两个远端脚本的轮询节奏从 60s 收紧到 15s:

1. scripts/stage2_original60_fp16_eval_when_ready.py
2. scripts/stage2_original60_fp16_lane_runner.py

并已完成远端热更新:
  watcher pid:
    GPU3 -> 1085695
    GPU4 -> 1086372
    GPU5 -> 1086998
    GPU6 -> 1087436
  lane runner pid:
    GPU3 -> 1099059
    GPU4 -> 1100169
    GPU5 -> 1100539
    GPU6 -> 1100707

注意这里没有采用“checkpoint 一出现就并发 eval”的策略。
原因是当前约束仍要求单卡同一时刻只跑 train 或 full AP eval 之一, 不能在训练未结束时抢占同卡做 true eval。

但四个第二批 label 现在都已经出现了 ready checkpoint:
  frontier_06 -> net_epoch25.pth
  frontier_07 -> net_epoch25.pth
  frontier_08 -> net_epoch25.pth
  frontier_15 -> net_epoch25.pth

四个 watcher 当前状态统一为:
  status = waiting_for_train_exit
  ready_checkpoint_seen = true
  poll_seconds = 15

结论:
  这 4 个 label 已经跨过“是否生成出可用 checkpoint”的门槛;
  现在只差训练完成, watcher 就会在最多 15 秒内接管 true FP16 AP eval。
```

截至 2026-06-28 15:30 CST 的阻塞修复:

```text
在 15:28 之后再次审计时, 发现上一版 watcher 曾错误地在训练未结束时提前发起 true FP16 AP eval:
  frontier_15 -> eval parent 1074316
  frontier_06 -> eval parent 1074897
  frontier_07 -> eval parent 1075404

这与当前资源约束冲突:
  单卡同一时刻只能跑 train 或 full AP eval 之一, 不能同卡训练/评测重叠。

现场表现:
  GPU3/4/5 显存从单 train 的约 8-9 GiB 抬升到 10-11 GiB
  `ps -ef` 可见 train_ddp 与 stage2_h800_true_fp16_ap_eval.py 同时存在
  rows 仍未增长到 14/60, 说明这些 eval 不是有效收口, 只是在抢同卡资源

处理:
  1. 终止误发的 eval parent 及其 worker:
     1074316, 1074897, 1075404
  2. 保留新的 watcher 逻辑:
     status = waiting_for_train_exit
     ready_checkpoint_seen = true
     poll_seconds = 15
  3. 不动 train 主进程, 保持四卡继续向 epoch 31 / Training Finished 推进

修复后状态:
  远端仅保留 train_ddp + watcher, 不再有运行中的 stage2_h800_true_fp16_ap_eval.py
  nvidia-smi:
    GPU3 = 11429 MiB
    GPU4 = 11075 MiB
    GPU5 = 10789 MiB
    GPU6 = 11171 MiB
  watcher 统一为:
    best_checkpoint_seen = net_epoch_bestval_at25.pth
    best_epoch_seen = 25
    training_finished_marker = false
    status = waiting_for_train_exit

结论:
  当前主线重新回到正确状态:
    train 持续推进
    watcher 15s 轮询
    训练结束后再发 true FP16 AP eval
```

截至 2026-06-28 15:34 CST 的再审计:

```text
本地同步后仍为:
  fp16_ap_measured = 13
  fp16_ap_no_claim = 47

第二批四个 label 当前都还未进入 Training Finished, 因此还没有新的 row append:
  frontier_06 -> epoch 25 [1109/2405]
  frontier_07 -> epoch 25 [833/2405]
  frontier_08 -> epoch 25 [675/2405]
  frontier_15 -> epoch 25 [1225/2405]

四个 watcher 当前统一状态:
  status = waiting_for_train_exit
  ready_checkpoint_seen = true
  training_finished_marker = false
  poll_seconds = 15

四个 ckpt 目录当前已经都有:
  net_epoch25.pth
  net_epoch_bestval_at25.pth

说明:
  当前已经不存在“有没有可用 checkpoint”的问题,
  也不存在新的 eval/import 阻塞。
  主路径剩余耗时主要来自 train_ddp 自身推进到训练收尾。
```

截至 2026-06-28 15:39 CST 的补充说明:

```text
第二批 4 个 label 继续在 epoch 25 中推进:
  frontier_06 -> [1521/2405]
  frontier_07 -> [1211/2405]
  frontier_08 -> [1046/2405]
  frontier_15 -> [1649/2405]

当前仍未出现:
  Training Finished, checkpoints saved to ...

因此 watcher 仍未切到 true AP eval, rows 仍停留在 13/60。

另外发现一个“历史残留而非当前阻塞”的现象:
  frontier_06 / frontier_07 / frontier_15 的 raw_dir/true_fp16_ap_eval/ 下
  还保留了上一次误发 eval 生成的文件:
    eval_dispatch.json
    gpu_preflight.json
    runner_command.json
    runner_stdout.txt
    runner_stderr.txt

但当前远端进程中已经没有运行中的 stage2_h800_true_fp16_ap_eval.py,
说明这些只是残留 artifact, 不是新的并发评测占卡。

结论:
  当前没有新的 watcher / inference / row append 阻塞。
  主线仍然只是等待 train_ddp 跑完, 然后由 watcher 在 15 秒内接管 true AP eval。
```

截至 2026-06-28 15:44 CST 的进程树核查:

```text
第二批 4 个 label 继续推进到 epoch 25 后段:
  frontier_06 -> [1969/2405]
  frontier_07 -> [1631/2405]
  frontier_08 -> [1458/2405]
  frontier_15 -> [2118/2405]

watcher 仍然一致:
  status = waiting_for_train_exit
  training_finished_marker = false
  best_checkpoint_seen = net_epoch_bestval_at25.pth

train 进程树核查结果:
  launcher pid 生存时间约 887-888s
  主 train pid:
    frontier_06 -> 906865
    frontier_07 -> 906863
    frontier_08 -> 906950
    frontier_15 -> 906948
  dataloader/worker 子进程生存时间约 178-261s, CPU 持续高占用

解释:
  这些子进程仍在持续滚动更新, 不是僵尸或挂死。
  远端 GPU3/4/5/6 显存仍保持在单 train 水位:
    11429 / 11075 / 10789 / 11171 MiB

结论:
  当前没有新的 train 收尾异常, 也没有 inference 尾巴或 eval 并发占卡。
  这四个 label 仍处于正常训练中, 离 watcher 接管 true AP eval 还差训练本身结束。
```

截至 2026-06-28 15:34 CST 的收尾时间判断:

```text
四个 label 当前推进位置:
  frontier_06 -> epoch 26 [18/2405]
  frontier_07 -> epoch 25 [2072/2405]
  frontier_08 -> epoch 25 [1894/2405]
  frontier_15 -> epoch 26 [181/2405]

这说明:
  frontier_06 / frontier_15 已经跨入 epoch 26
  frontier_07 / frontier_08 还在 epoch 25 尾段

按最近几轮 stdout 观察到的迭代速度粗估:
  约 230-280 iter/min

若继续以当前速度推进到默认 epoch 31, 则:
  frontier_07 / frontier_08 剩余约 4.1-4.2 个 epoch
  frontier_06 / frontier_15 剩余约 5.0 个 epoch
  第二批整体进入 Training Finished 的粗略时间窗大约还需 45-55 分钟

这不是严格 ETA, 只是用来判断“当前是正常慢推进”而不是卡死。
只要 train_ddp 继续滚动, watcher 侧无需额外干预。
```

截至 2026-06-28 15:35 CST 的最新闭环位置:

```text
四个 label 当前推进为:
  frontier_06 -> epoch 26 [469/2405]
  frontier_07 -> epoch 26 [88/2405]
  frontier_08 -> epoch 25 [2319/2405]
  frontier_15 -> epoch 26 [647/2405]

变化点:
  frontier_07 已从 epoch 25 尾段跨入 epoch 26
  frontier_08 仍是第二批中最靠后的一个, 但也只剩下 epoch 25 的最后约 86 iter

因此当前第二批的相对顺序大致是:
  最快: frontier_15
  其次: frontier_06
  再次: frontier_07
  最后: frontier_08

watcher 仍然统一为:
  status = waiting_for_train_exit
  training_finished_marker = false

说明:
  当前没有新的收尾阻塞, 第二批整体仍在朝 epoch 31 正常推进。
```

截至 2026-06-28 15:36 CST 的状态收敛:

```text
第二批四个 label 现已全部进入 epoch 26:
  frontier_06 -> [774/2405]
  frontier_07 -> [378/2405]
  frontier_08 -> [197/2405]
  frontier_15 -> [963/2405]

这意味着此前最慢的 frontier_08 也已从 epoch 25 跨入 epoch 26,
第二批不再存在“有人还卡在上一 epoch 尾段”的分化。

当前仍未出现:
  Training Finished, checkpoints saved to ...
也没有运行中的:
  stage2_h800_true_fp16_ap_eval.py
  inference.py

结论:
  当前 Stage2 original60 FP16 AP 主线是稳定的:
    四卡 train 正常推进
    watcher 正常等待
    第二批全体已进入 epoch 26
  下一阶段的关键事件只剩:
    Training Finished -> watcher 接管 -> true FP16 AP eval -> rows 增长
```

## 8. 完成后刷新与验收

每批完成后:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

验收:

```text
fp16_true_original60_ap_rows_v1.jsonl 有 60 个 label
completion review 中 FP16 AP measured = 60/60
每个 label 有 raw artifact、runner command 或 eval_command、stdout/stderr、GPU id、AP30/AP50/AP70
没有现成 checkpoint 的 label 必须有 checkpoint generation artifact 或 finetune blocker, 不能静默跳过
```

## 9. 冷启动 /goal

```text
/goal 只推进 Stage2 original60 FP16 AP 大规模补点, 不处理 INT8。目标是把 rows/fp16_true_original60_ap_rows_v1.jsonl 从当前 5/60 measured 补到 60/60 measured。使用 H800 GPU3/GPU4/GPU5/GPU6 四张卡, 启动前重新审计 nvidia-smi 和 compute-apps。先从 completion review/queue 中筛出 fp16 + ap_status=no_claim 的 55 个 label, 不设优先级, 全量补齐。checkpoint recovery 只保留 2-5 分钟极短搜索窗口, 采用 roots 顺扫的简单秩序, 目标只是确认是否存在可直接进入 full AP eval 的 exact-label checkpoint; 理论上当前大多数 label 都应转入生成/微调主线。注意: 已完成 FP16 latency/energy 60/60 使用的是 backbone-only ONNX/TVM artifact, 不是 AP eval checkpoint, 不能直接用于 AP row。对无 checkpoint 的 label, 使用 tools/structural_prune_pyramid.py 从 Pyramid_DAIR_m1_base_2023_08_14_11_42_29 生成目标 width 的 ckpt_dir/config, 再用 HEAL opencood/tools/train_ddp.py 单卡 nproc_per_node=1 --half 在 GPU3/4/5/6 资源池上推进微调生成 net_epoch_bestval_at*.pth。默认先跑到 epoch 31, 仅当没有 post-baseline bestval、AP 明显异常或训练证据不足时才延长到 epoch 48。大规模铺开前先做 2-4 个 label 的小批闭环验证; 闭环通过后立即进入全量批量生成+微调+AP eval; 若闭环失败, 必须先定位根因并修复后继续, 不能直接停止。生成可用 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16, 每个 label 必须保存 raw artifact、train/eval runner pid、stdout/stderr、GPU id、ckpt_dir、eval_command、AP30/AP50/AP70, gate 通过后追加 rows/fp16_true_original60_ap_rows_v1.jsonl 并刷新 coverage/review。禁止用不匹配 checkpoint、测速 ONNX、structural-prune 初始未微调 checkpoint、smoke AP、partial eval 或旧 row 写 measured。遇到 checkpoint 生成、finetune、eval/import 问题, 保存 blocker 和 failure reason, 做根因判断, 修复后继续该方向并并行推进其他 label。
```
