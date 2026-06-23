# Plan B: R101+DCN baseline + 多维度联合优化(取代 tiny 路径)

> **日期**: 2026-05-02
> **状态**: 设计完成,baseline 验证中
> **取代**: 阶段 4 tiny 训练路径(D4.9 未通过)+ 阶段 5.1 (SNR 不足)

---

## 一、为什么切换到 Plan B

### 旧路径(tiny)的实测问题
- Stage 3 coop_tiny ep30: F1=0.18, AMOTA=0(被 min_recall 截断)
- 7 configs F1 跨度仅 0.74pt(< 0.3pt eval 噪声底)
- LightGBM v2 Spearman = -0.36(无信号)
- V2X 注入修复实测失败(注入后 SNR 进一步恶化)

### 旧 Stage 2.5 决策树前提错误
> "为了让 backbone 可剪 → 必须训 R50 tiny"

实际事实:**30 configs 中 73% 不动 backbone**;backbone 只占 GPU compute 16%(参数 44%);剩余 84% 的加速空间在 transformer 模块(BEV encoder + decoder + heads),可全网剪枝且与 DCN 无关。

### Plan B 选择
- **Baseline**: `univ2x_coop_e2e_stg2.pth`(R101+DCN, 实测 AMOTA = 0.3298)
- **30 configs 中 22 个直接可跑**(无 backbone 剪枝)
- **9 个 backbone 相关 configs 改造为 backbone 量化变体**(INT8/INT4 PTQ)
- **复用 1.2 章节 P1-FFN 5 个数据点**(span 9.3pt 干净信号)
- **论文 motivation**:"在生产级 R101+DCN(DCN 阻止 backbone 剪枝)的工业约束下,剪枝 × 量化 × 部署三维联合搜索找到 Pareto 最优"

---

## 二、修订版 30 configs 分配

| 类 | configs | 内容 | Plan B 状态 |
|---|---|---|---|
| baseline | 1 | FP32 无剪枝 | ✅ 直接 |
| **A 类(改造)** | A1-A6 | 旧:backbone 剪 0.25/0.375/0.5;新:backbone INT8/INT4 量化变体 | 🔄 改造 |
| B 类 | B1-B3 | heads 剪枝 / heads+encoder | ✅ 直接 |
| C 类 | C1-C3 | encoder/decoder 剪枝(不含 backbone) | ✅ 直接 |
| **C4(改造)** | C4 | 旧:含 backbone 剪;新:backbone INT8 + 其他剪 | 🔄 改造 |
| D 类 | D1-D5 | 量化(含 backbone INT8) | ✅ 直接(需扩展量化覆盖) |
| E 类 | E1-E8, E11-E12 | encoder/decoder 剪 + 量化 | ✅ 直接(联合 pipeline) |
| **E9-E10(改造)** | E9, E10 | 旧:backbone 剪 + 量化;新:backbone INT8 + 其他剪 | 🔄 改造 |

**改造数**: A1-6 + C4 + E9-10 = 9 个
**直接跑数**: 1 baseline + 3 B + 3 C + 5 D + 10 E = 22 个
**总数**: 31 configs

---

## 三、A 类改造细节

| Config | 旧设计 | 新设计 |
|---|---|---|
| A1 | backbone 剪 0.25 | **backbone INT8 W+A per-tensor** |
| A2 | backbone 剪 0.375 | **backbone INT8 W+A per-channel** |
| A3 | backbone 剪 0.5 | **backbone INT4 W+A per-tensor**(极激进) |
| A4 | backbone 0.25 + encoder 0.25 | **backbone INT8 + encoder 剪 0.25** |
| A5 | backbone 0.375 + encoder 0.375 | **backbone INT8 + encoder 剪 0.5** |
| A6 | backbone 0.5 + 多模块剪 | **backbone INT4 + encoder 0.5 + decoder 0.5** |

---

## 四、Pipeline 设计

### 三种执行路径(由 config 类型决定)

```
plan_b_run_one_config.sh <CONFIG_ID> <GPU_ID>
   │
   ├── 纯剪枝(B/C1-3): test_with_pruning.py 直接跑(1.2 已验证)
   │     → output/plan_b/<id>.pkl + log
   │
   ├── 纯量化(D/A1-3): quick_eval_quant.py 直接跑(1.1 已验证)
   │     → log 含 AMOTA(无 .pkl)
   │
   └── 联合(E/A4-6/C4): quick_eval_quant.py --prune-config(1.1 已支持)
         → log 含 AMOTA
```

### 输出统一:
- 每个 config: `data/phase4/stage5_v3/<id>.json`(AMOTA + 其他指标)
- 汇总: `data/phase4/stage5_baseline_v3.csv`(LightGBM-ready)

---

## 五、关键工程缺口(后补,不阻塞主流程)

### `quick_eval_quant.py` 当前只量化 BEV encoder

实测代码 (line 460): `qmodel = apply_quant_config(encoder, ...)` — 没扩展到 backbone/decoder/heads。

意味着:
- D 类 configs(如 D1: backbone INT8)**实际跑出来只是"BEV encoder INT8 + 其他不变"**,backbone 量化设置是装饰
- A 类改造 configs 也受同样问题

### 解决路径(后补,~1-2 天)
1. 扩展 `quick_eval_quant.py` 支持多模块量化:对 backbone, decoder, heads, V2X 通信各调用一次 `apply_quant_config`
2. 注册 `register_backbone_specials`(R101 + DCN 的量化包装)
3. 测试 backbone INT8 fake-quant 可跑通且精度合理

### 现阶段策略
- **先跑 22 个直接可跑的 configs**(B/C/D/E 不含改造)
- 这 22 个有真实 prune+quant 信号(D 类 BEV encoder 量化是真实的)
- 已经能训 LightGBM v3 看 Spearman 是否突破 0.7
- 后补 backbone 量化扩展,跑 9 个改造 configs,得到完整 31 configs 数据

---

## 六、时间估算

| 阶段 | 工期 | 累计 |
|---|---|---|
| baseline 验证(确认 AMOTA ≈ 0.3298) | 17 min | 17 min |
| 跑 22 个直接 configs(6 卡并行) | ~3-4 hr | ~4 hr |
| LightGBM v3 训练 + Spearman 验证 | 30 min | ~4.5 hr |
| 扩展 quick_eval_quant.py 支持多模块量化 | 1 天 | 1.5 天 |
| 跑 9 个改造 configs | ~1.5 hr | 1.5 天 + 2 hr |
| 最终 LightGBM v3(31 configs)+ 报告 | 30 min | **~2 天** |

---

## 七、复用资产

- **1.2 章节 P1-FFN 5 个数据点**(20%/30%/40%/50%/60% AMOTA span 0.93pt → 实际是 baseline 0.3298 - P1-60% 0.2366 = 9.3pt)
- **1.1 章节量化代码**: `projects/mmdet3d_plugin/univ2x/quant/`(已就绪)
- **1.2 章节剪枝代码**: `projects/mmdet3d_plugin/univ2x/pruning/`(已就绪)
- **stage5 pipeline 框架**(merge 逻辑去掉,改 single ckpt 输入)

---

## 八、deliverables 清单

- ✅ `tools/plan_b_run_one_config.sh`(三路 routing)
- ✅ `tools/plan_b_batch.sh`(多 GPU 并行)
- ✅ `tools/plan_b_extract_amota.py`(metrics 提取)
- ⏳ baseline AMOTA 验证(进行中)
- ⏳ 22 configs 数据
- ⏳ `tools/quick_eval_quant.py` 扩展(后补)
- ⏳ 9 个改造 configs 数据
- ⏳ `data/phase4/stage5_baseline_v3.csv`
- ⏳ `models/lgb_predictor_v3.txt`
- ⏳ `results/phase4_stage5_lgb_v3_metrics.json`
- ⏳ `paper_learning/.../PLAN_B_最终报告.md`

---

## 九、修订历史

- **2026-05-02 v1.0 初版**: 基于 R101+DCN baseline 切换决策,设计 31 configs 修订版
