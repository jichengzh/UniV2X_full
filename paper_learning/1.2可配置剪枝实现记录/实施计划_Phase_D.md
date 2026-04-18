# Phase D 实施计划：剪枝解耦 + 联合量化 PyTorch 端实验

> 日期：2026-04-18
> 前置：Phase B（剪枝 Pareto 完成）+ Phase C.0/C.1（TRT 工具链 + latency 基线）
> 目标：在 PyTorch 端完成剪枝搜索空间的最后扩展（P1a/P1b 解耦）+ 剪枝×量化联合精度评估
> 硬件：RTX 4090 × 2（并行）
> 预计总耗时：~5.5h（2 GPU 并行）

---

## 一、当前状态与本阶段目标

### 1.1 已有数据

| 维度 | 已覆盖 | 数据点 | 最优点 |
|---|---|:---:|---|
| P1 FFN 绑定（P1a=P1b） | 20-80% × q=1/q=2 | 13 | FFN 60% q=2 AMOTA 0.3354 |
| P9 decoder 层数 | 3/4/5/6 × q=2 | 4 | 5 层 AMOTA 0.3095 |
| P11 seg MLP | 零微调 30% | 1 | AMOTA 零影响 |
| 量化（独立） | 1.1 已完成 | — | BEV INT8 AMOTA 0.364 |
| **剪枝 × 量化联合** | **0** | **0** | **空白** |

### 1.2 本阶段要回答的 4 个问题

| # | 问题 | 由哪个实验回答 |
|:---:|---|---|
| 1 | encoder/decoder FFN 解耦是否在多任务 Pareto 上优于绑定？ | D.1.1-D.1.4 |
| 2 | 剪枝后模型对 INT8 量化是否更敏感？ | D.2.2 vs D.2.1 |
| 3 | 解耦 + 量化的三维联合最优点在哪？ | D.2.7 |
| 4 | 极致压缩（INT4）在剪枝模型上可行吗？ | D.2.5 |

---

## 二、依赖与执行顺序

```
D.0 工程改造 (~2h 编码)
 │
 ├──► D.1 剪枝解耦实验 (~3h GPU)
 │         │
 │         ├─ D.1.1 enc=1.0, dec=0.3 (encoder 不剪)
 │         ├─ D.1.2 enc=0.8, dec=0.3 (encoder 保守)
 │         ├─ D.1.3 enc=0.7, dec=0.4 (对标绑定 60%)
 │         └─ D.1.4 enc=1.0, dec=0.7 (最小改动)
 │
 └──► D.2 联合量化实验 (~2h GPU)  ← 可与 D.1 并行
           │
           ├─ D.2.1 baseline + INT8 W+A (纯量化对照)
           ├─ D.2.2 S1 剪枝 + INT8 W+A (核心联合点 ⭐)
           ├─ D.2.3 S1 + INT8 W-only (保守)
           ├─ D.2.4 S1 + INT8 W+A per-channel
           ├─ D.2.5 S1 + INT4 W-only (激进)
           ├─ D.2.6 FFN 70% + INT8 W+A
           └─ D.2.7 D.1 最优 + INT8 W+A (解耦+量化 ⭐)
                      ↑
                      依赖 D.1 结果选最优配置

──► D.3 指标汇总 + Pareto + commit (~1h)
```

---

## 三、D.0 工程改造

### D.0.1 剪枝解耦支持

**文件**：`projects/mmdet3d_plugin/univ2x/pruning/prune_univ2x.py`

**当前代码**（`apply_prune_config` 内）：

```python
encoder_cfg = prune_cfg.get("encoder", {})
decoder_cfg = prune_cfg.get("decoder", {})
# 但 _apply_ffn_pruning 内部统一读 encoder/decoder 的 ffn_mid_ratio
```

**改动**：让 `_apply_ffn_pruning` 区分 encoder 和 decoder FFN 的 ratio：

```python
# 现在: 遍历所有 FFN pair, 统一用同一个 ratio
# 改为: 根据 pair_name 判断属于 encoder 还是 decoder, 使用各自的 ratio
for pair_name, first, second in ffn_pairs:
    if "encoder" in pair_name:
        ratio = encoder_cfg.get("ffn_mid_ratio", 1.0)
    elif "decoder" in pair_name:
        ratio = decoder_cfg.get("ffn_mid_ratio", 1.0)
    else:
        ratio = encoder_cfg.get("ffn_mid_ratio", 1.0)  # 默认跟 encoder
    # ... 按 ratio 剪枝
```

**验证**：用 `enc=0.8, dec=0.3` dry-run，确认 encoder FFN 512→408，decoder FFN 512→152。

**预计耗时**：30 min

### D.0.2 新建解耦 prune_configs

4 个新 JSON 配置：

```
prune_configs/decouple_enc10_dec03.json   # D.1.1: encoder 不剪, decoder 激进
prune_configs/decouple_enc08_dec03.json   # D.1.2: encoder 保守, decoder 激进
prune_configs/decouple_enc07_dec04.json   # D.1.3: 对标绑定 60%
prune_configs/decouple_enc10_dec07.json   # D.1.4: 最小改动
```

**格式**（以 D.1.1 为例）：

```json
{
  "version": "1.2",
  "_description": "解耦 D.1.1: encoder FFN 不剪, decoder FFN 剪 70% (保留 30%)",
  "locked": {
    "importance_criterion": "l1_norm",
    "pruning_granularity": "local",
    "iterative_steps": 5,
    "round_to": 8
  },
  "encoder": { "ffn_mid_ratio": 1.0, "attn_proj_ratio": 0.0, "head_pruning_ratio": 0.0 },
  "decoder": { "ffn_mid_ratio": 0.3, "attn_proj_ratio": 0.0, "head_pruning_ratio": 0.0, "num_layers": 6 },
  "heads": { "head_mid_ratio": 1.0 },
  "finetune": { "epochs": 0 },
  "constraints": {
    "skip_layers": ["sampling_offsets", "attention_weights"],
    "min_channels": 64,
    "channel_alignment": 8
  }
}
```

**预计耗时**：10 min

### D.0.3 联合量化工具改造

**文件**：`tools/quick_eval_quant.py`

**当前流程**：
```
load baseline model → apply_quant_config → 校准 → 评估 AMOTA
```

**改造后流程**：
```
load baseline model → [可选: apply_prune_config + load finetuned ckpt] → apply_quant_config → 校准 → 评估 AMOTA
```

**新增参数**：
```python
parser.add_argument('--prune-config', default=None,
                    help='剪枝配置 JSON (先剪枝再量化)')
parser.add_argument('--finetuned-ckpt', default=None,
                    help='剪枝微调后的 checkpoint (在剪枝后 load 覆盖)')
```

**改动要点**：
1. 在 `apply_quant_config` 之前插入剪枝逻辑
2. 剪枝顺序和 `finetune_pruned.py` 一致：先 load baseline → 剪枝 → load finetuned ckpt
3. 量化的校准数据（`calibration/bev_encoder_calib_inputs.pkl`）是 baseline 采集的；剪枝后激活分布变了，可能需要观察校准质量
4. 如果校准数据不匹配，fallback 用 `scale_method='mse'` 做在线校准

**预计耗时**：1h

### D.0.4 新建联合量化 quant_configs

```
quant_configs/int8_wa_pt.json         # INT8 W+A per-tensor (默认)
quant_configs/int8_wonly_pt.json      # INT8 W-only per-tensor
quant_configs/int8_wa_pc.json         # INT8 W+A per-channel
quant_configs/int4_wonly_pt.json      # INT4 W-only per-tensor
```

**预计耗时**：10 min

---

## 四、D.1 剪枝解耦实验

### 4.1 实验矩阵

| ID | P1a (encoder) | P1b (decoder) | P9 | P11 | 微调 | 对照 |
|:---:|:---:|:---:|:---:|:---:|---|---|
| D.1.1 | **1.0** | **0.3** | 6 | 1.0 | q=2 3ep | vs S1 (0.4/0.4) |
| D.1.2 | **0.8** | **0.3** | 6 | 1.0 | q=2 3ep | vs D.1.1 |
| D.1.3 | **0.7** | **0.4** | 6 | 1.0 | q=2 3ep | vs S1 (0.4/0.4) |
| D.1.4 | **1.0** | **0.7** | 6 | 1.0 | q=2 3ep | 最小改动 baseline |

### 4.2 执行命令模板

```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/jichengzhi/UniV2X python tools/finetune_pruned.py \
  projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
  ckpts/univ2x_coop_e2e_stg2.pth \
  --prune-config prune_configs/decouple_enc10_dec03.json \
  --work-dir work_dirs/ft_decouple_enc10_dec03 \
  --epochs 3 --lr-scale 0.1 \
  --train-modules pts_bbox_head \
  --skip-aux-heads \
  --queue-length 2 \
  --no-validate
```

评估：
```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/jichengzhi/UniV2X python tools/test_with_pruning.py \
  projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
  ckpts/univ2x_coop_e2e_stg2.pth \
  --prune-config prune_configs/decouple_enc10_dec03.json \
  --finetuned-ckpt work_dirs/ft_decouple_enc10_dec03/epoch_3.pth \
  --out output/decouple_enc10_dec03_results.pkl \
  --eval bbox --launcher none
```

### 4.3 每个实验报告指标

| 类别 | 指标 |
|---|---|
| 精度（跟踪）| AMOTA, AMOTP, MT, ML, IDS |
| 精度（检测）| mAP, NDS |
| 精度（分割）| drivable / lanes / crossing / contour IoU |
| 参数 | ego_agent (M), 相对 baseline (%) |

### 4.4 D.1 决策规则

完成 D.1.1-D.1.4 后，选出 D.1 最优配置用于 D.2.7：

| 条件 | 选择 |
|---|---|
| D.1.X 的 AMOTA ≥ 0.33 **且** lanes IoU ≥ 0.18 | 该配置为"多任务最优" |
| 多个满足 → 选参数省最大的 | — |
| 无一满足 → 选 AMOTA 最高的 | 多任务 Pareto 无解，记录为负面结论 |

### 4.5 时间估算

| 步骤 | 耗时 | GPU |
|---|---:|---|
| 4 个 q=2 微调（各 ~40 min） | 160 min | 2 GPU 并行 → 80 min |
| 4 个 AMOTA eval（各 ~12 min）| 48 min | 2 GPU 并行 → 24 min |
| **D.1 总计** | — | **~1.7h 挂钟** |

---

## 五、D.2 联合量化实验

### 5.1 实验矩阵

| ID | 剪枝配置 | 微调 ckpt | 量化 | 目的 |
|:---:|---|---|---|---|
| D.2.1 | 无（baseline） | baseline stg2 | INT8 W+A per-tensor | 纯量化对照 |
| D.2.2 | S1 (FFN 60%) | ft_p1_60_q2/ep3 | **INT8 W+A per-tensor** | **核心联合点 ⭐** |
| D.2.3 | S1 | ft_p1_60_q2/ep3 | INT8 **W-only** per-tensor | 保守（只量化权重） |
| D.2.4 | S1 | ft_p1_60_q2/ep3 | INT8 W+A **per-channel** | 更高精度粒度 |
| D.2.5 | S1 | ft_p1_60_q2/ep3 | **INT4 W-only** per-tensor | 激进量化 |
| D.2.6 | FFN 70% | ft_p1_70_q2/ep3 | INT8 W+A per-tensor | 更激进剪枝 + 量化 |
| D.2.7 | D.1 最优 | D.1 最优 ckpt | INT8 W+A per-tensor | **解耦 + 量化 ⭐** |

### 5.2 执行命令模板

```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/jichengzhi/UniV2X python tools/quick_eval_quant.py \
  --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
  --checkpoint ckpts/univ2x_coop_e2e_stg2.pth \
  --prune-config prune_configs/p1_ffn_60pct.json \
  --finetuned-ckpt work_dirs/ft_p1_60_q2/epoch_3.pth \
  --quant-config quant_configs/int8_wa_pt.json \
  --cali-data calibration/bev_encoder_calib_inputs.pkl \
  --eval-samples 168
```

### 5.3 每个实验报告指标

| 类别 | 指标 |
|---|---|
| 精度（跟踪）| AMOTA |
| 精度（检测）| mAP, NDS |
| 量化设置 | 位宽, 粒度, 目标 (W/A/W+A) |
| 权重体积 | 相对 FP32 baseline 的压缩比 |

### 5.4 预期结果范围

| 配置 | 预期 AMOTA | 依据 |
|---|---|---|
| D.2.1 baseline + INT8 | 0.32-0.36 | 1.1 报告 BEV INT8 = 0.364 |
| D.2.2 S1 + INT8 | 0.30-0.34 | 如果剪枝不加剧量化敏感度 |
| D.2.3 S1 + W-only | 0.33-0.34 | W-only 比 W+A 更安全 |
| D.2.5 S1 + INT4 | 0.20-0.30 | INT4 通常掉点显著 |
| D.2.6 FFN 70% + INT8 | 0.28-0.32 | 双重压缩 |

### 5.5 时间估算

| 步骤 | 耗时 | GPU |
|---|---:|---|
| 7 个量化评估（各 ~15 min）| 105 min | 2 GPU 并行 → 53 min |
| **D.2 总计** | — | **~1h 挂钟** |

**注**：D.2.1-D.2.6 不需要微调训练（直接在已有 ckpt 上做 fake quant + eval）。D.2.7 依赖 D.1 结果。

---

## 六、D.3 指标汇总

### 6.1 Pareto 表格式（最终产出）

| 配置 | P1a | P1b | Quant | 参数(M) | 体积 | AMOTA | mAP | lanes IoU |
|---|:---:|:---:|---|---:|---:|---:|---:|---:|
| Baseline FP32 | 1.0 | 1.0 | FP32 | 100.63 | 100% | 0.330 | 0.072 | 0.21 |
| Baseline INT8 | 1.0 | 1.0 | INT8 | 100.63 | 25% | D.2.1 | | |
| S1 绑定 FP32 | 0.4 | 0.4 | FP32 | 93.63 | 93% | 0.335 | 0.069 | 0.10 |
| **S1 绑定 INT8** | 0.4 | 0.4 | **INT8** | 93.63 | **23%** | D.2.2 | | |
| 解耦 A FP32 | 1.0 | 0.3 | FP32 | ~96 | ~95% | D.1.1 | | D.1.1 |
| **解耦 A INT8** | 1.0 | 0.3 | **INT8** | ~96 | **~24%** | D.2.7 | | D.2.7 |
| FFN 70% INT8 | 0.3 | 0.3 | INT8 | 92.50 | 22% | D.2.6 | | |
| S1 INT4 W | 0.4 | 0.4 | INT4-W | 93.63 | 12% | D.2.5 | | |

### 6.2 更新文档

| 文档 | 更新内容 |
|---|---|
| `实验结果汇总_2026-04-16.md` | 新增 § D.1 解耦结果 + § D.2 联合量化结果 |
| `搜索空间_联合优化_2026-04-16.md` | 用 D.1 数据更新 P1a/P1b 取值范围建议 |
| `PROGRESS.md` | Phase D 状态更新 |

---

## 七、风险与回滚

| 风险 | 触发条件 | 回滚 |
|---|---|---|
| 解耦代码改动影响已有实验 | D.0.1 后已有 prune_configs 行为变化 | 用 `enc=0.4, dec=0.4` 回归测试，AMOTA 应 = S1 |
| 量化校准数据不适配剪枝模型 | D.2.2 AMOTA < 0.20（远低于预期）| 用剪枝后模型重新采集校准数据 |
| fake quant 和 TRT INT8 精度不一致 | D.2 结果与未来 TRT 结果偏差大 | 记录为 limitation，后续对比 |
| D.1 解耦无 Pareto 优势 | 所有 D.1.X 的 AMOTA 和 seg IoU 都不优于 S1 | 记录为负面结论，解耦搜索空间无意义 |

---

## 八、立即行动项

1. **D.0.1**：修改 `prune_univ2x.py` 的 `_apply_ffn_pruning` 支持解耦
2. **D.0.2**：生成 4 个解耦 prune_configs
3. **D.0.3**：修改 `quick_eval_quant.py` 加 `--prune-config` 和 `--finetuned-ckpt`
4. **D.0.4**：生成 4 个量化 quant_configs
5. 启动 D.1.1-D.1.4 训练（2 GPU 并行，~80 min）
6. 同时启动 D.2.1-D.2.6（不依赖 D.1，2 GPU 并行，~53 min）
7. D.1 结果出来后启动 D.2.7
8. D.3 汇总 + commit + push
