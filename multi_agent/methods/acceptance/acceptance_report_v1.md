# 验收报告 v1 — 可配置剪枝/量化/硬件 → 数据生成 → 预测器流水线

> 验收师 (QA) 独立验收, 2026-05-31。所有结论附实跑证据。
> conda python: `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`
> 验收方法: 实跑 3 个 smoke/单测 + 解析 checkpoint/CSV/engine JSON + git 只读核对 schema + 独立复现 searcher pass_rate。

---

## 0. 总体结论 (TL;DR)

- **4 个 agent 的工具全部真实存在、全部消费同一个未被修改的 `framework/config_schema.py` Config 契约、3 个 smoke/单测全 PASS。** 契约统一性这一最关键项 **通过**。
- **真做出来的硬核**: ① 量化分模块混精 + 硬约束编码 (真) ② 剪枝真结构化 rebuild + 真 finetune 后 2.8M ckpt 落盘 (真, 已逐张量验证通道数) ③ TRT engine 真 build 出 2 个 (FP16 + QDQ INT8, layer-info 对得上) ④ 数据生成管线真接通三工具, 12 个剪枝点是真 rebuild。
- **核心缺口 (阻塞预测器训练)**: DOE 数据集 **25 行全部 `is_real_measured=False`** —— 即**没有任何一行带真实测的 latency/AP/throughput**。剪枝是真的 (rebuild + ckpt), 但量化/部署在数据生成里走 **dry_run** (deploy 不真 build), 所以响应面 (y 标签) 是空的。**当前数据不足以训练 latency/AP 预测器。**
- 评级: 量化师 **PASS**, 剪枝师 **PASS**, 硬件师 **PASS-with-caveats**, 数据生成师 **PASS-with-caveats**。

---

## 1. 跨 agent 接口一致性 (最关键, 先讲)

### 1.1 schema 未被违规修改 ✅
```
$ git -C /home/jichengzhi/UniV2X status --porcelain framework/
 M framework/capability_schema.py
 M framework/searcher_v0.py
```
- `framework/config_schema.py` **无任何 working-tree 改动** (git diff 干净, 仍是 commit 3de2167 版本)。4 个 agent 均未碰它。✅
- 唯二改动的 `capability_schema.py` / `searcher_v0.py` 是 commit 之前已有的脏文件 (M4.8 历史遗留), 与本批 agent 无关。

### 1.2 全部工具消费真 Config ✅
逐文件 grep 确认 `from framework.config_schema import Config`:
- `quant_config.py:43`、`prune_config.py:53`、`deploy_config.py:492`、`generate_dataset.py:38`、`doe_anchors.py:43` —— 全部 import 真契约, 无影子定义。
- `generate_dataset.py` 真正 wire 起三工具: `from tools.configurable.prune_config import ...` / `quant_config import resolve_quant_plan` / `deploy_config import build_plan` (line 40-44)。管线是真接通的, 不是各写各的。

### 1.3 字段访问全部落在 schema 内 (无幻觉字段) ✅
grep `config.<field>` 全集:
- quant: `q_bits / q_granularity / q_object / q_calibrator / modules() / config_id`
- prune: `prune_rate / prune_object / prune_criterion / q_bits / modules() / config_id`
- deploy: `d_routing` (其余 `config.set_flag` 等是 TRT `IBuilderConfig`, 非本 schema, 命名空间不冲突)
- 全部是 schema 真字段。**无任何 agent 引用 schema 不存在的字段。**

### 1.4 建议补充是否冲突 / schema 小瑕疵
- `q_calibrator` 是 schema 真字段 (line 61) 且被 quant 工具消费, 但 **未列入 `__all__`** (line 212-221 缺它)。不影响 import (直接 `from ... import Config` 拿到的是类), 属文档级小瑕疵, 非违规。
- 各 agent 的"建议 schema 补充"散落在各自 dims 文档 (如 quant 的 per-module W-only 需新增 inject 路径、hardware 的 per-group/AWQ 维度未纳入)。**未发现互相冲突**, 都是"未来扩展"性质, 均明确写"本批不改 schema"。

**接口一致性结论: 契约统一, 无违规, 无幻觉字段, 管线真接通。这一关 PASS。**

---

## 2. 逐 agent 评级

### Agent 1 — 可配置量化师 → **PASS**

证据 (`test_quant_config_smoke.py` 实跑, EXIT=0, 14/14 PASS):
- (b) 分模块混精**真可配**: 配置 A 全 INT8 → `trt_precision=int8`; 配置 B → `trt_precision=mixed`, `int8_modules=['backbone','encoder']` + `fp16_modules=['decoder','heads','v2x_comm']` 共存。两 Config 产出**不同 manifest** (PASS)。
- (c) 硬约束**真编码**且在 smoke 里触发: `[encoder] entropy 禁用 → 回退 minmax`、`activation 强制 per-tensor`、敏感层 (decoder/attention/MSDA/cls/reg) 强制 FP16 —— 全部断言 PASS。
- (a) 消费 Config 未改 schema (见 §1)。
- (d) 维度清单 `dims_quantization_v1.md` (10169 B) 存在, 字段与 schema 对齐。
- gap: per-module W-only 目前是**全局 flag** (smoke 自己 WARN 承认: "m4_8 --w-only 是全局 flag, per-module 需 inject_qdq_from_config.py 路径"), 即 W-only 还做不到真正按模块隔离。已诚实标注。

### Agent 2 — 可配置剪枝师 → **PASS**

证据 (`smoke_prune_config.py` 实跑 EXIT=0 + 直接解析 ckpt 张量):
- (b) **真结构化 rebuild, 非 mask** —— 逐张量验证 `pruned_rtx4090_sens_prune_50/net_epoch_bestval_at23.pth`:
  - `model_state_dict` **总参数 2,825,013** (用 torch 实算)。
  - conv3 输出通道: layer0=32 / layer1=64 / layer2=128 = 精确 `num_filters [64,128,256]→[32,64,128]` (除 2), **真缩小**。
  - conv2 grouped: layer2 = `(256, 8, 3, 3)` → 8×32=256, **%32 对齐成立**。
  - smoke 自报 `forward 通 (小模型): outputs=[(1,2,128,256),(1,14,128,256),(1,4,128,256)]` —— rebuild 后能前向。
- 声称 **5.46M→2.81M 属实**: CSV `params_total_new` p50=2,808,311, 落盘 ckpt 实测 2,825,013 (差异因 CSV 数 backbone 段 / ckpt 是 finetune 后全模型, 同量级吻合)。
- (c)+(d) 跨层依赖 + grouped-conv 约束**正确处理**, 且**诚实做了 DepGraph 可用性实测**: smoke 显示 DepGraph `build_dependency trace 通` 但 `local 剪枝 forward FAIL: size 32 must match 64` (ResNeXt 残差不一致), 结论"Pyramid 默认走手写 rebuild"。这是教科书级的诚实负结论。
- gap: 真 rebuild ✅ 但 **`output/prune_manifest.json` 标 `dry_run:true`** —— 该 manifest 是工具默认 dry-run 产物 (非数据生成产物), 真 rebuild 在 DOE 管线里才发生。命名易误导, 但不算造假。

### Agent 3 — 硬件/TRT 工程师 → **PASS-with-caveats**

证据 (`test_deploy_config.py` 实跑 7/7 PASS + 2 个 engine 文件 + 2 个 layer-info JSON):
- (b) **真 build 出 engine**: `trt_engines/smoke_bev_encoder_fp16.engine` (42.8MB) + `smoke_bev_encoder_qdq_int8.engine` (40.8MB) 文件真在。`results/deploy_smoke_bev_encoder_qdq.json` 含真 layer-info: `num_layers=6055`, `QUANTIZE/DEQUANTIZE 各 103`, `build_sec=180.1`, TRT 10.13.0.35, 7 个 plugin (MSDA/DCN/Rotate/Inverse) 全 visible。
- (c) plugin 路径**真注册** (plugins_visible 列出 v1/v2 creator)。
- (d) 平台门控**正确**: 单测 `test_dla_routing_degraded_to_gpu_on_4090` PASS + `test_dla_routing_kept_on_orin` PASS —— DLA 在 4090 自动降级 GPU、在 Orin 保留。`dims_hardware_v1.md` (9928 B) 完整。
- **caveat (跨 agent bug)**: QDQ INT8 engine **真实 INT8 层占比仅 10.8%** (`layer_precision_count: Int8=38 / 总 353`, `int8_layer_pct=10.8`)。即"真 build 出 INT8 engine"成立, 但**绝大多数层退回 Half**, 加速被严重稀释。硬件师自报根因是 QDQ ONNX scale dtype 不一致。这是真实存在、有据可查的集成缺陷 (见 §3)。

### Agent 4 — 数据生成师 → **PASS-with-caveats**

证据 (独立复现 searcher + 解析 CSV):
- (a) 搜索空间规模**有据且我已独立复现**: 我用 `random_search(hw=HardwareCapability.from_yaml("configs/hardware/rtx4090.yaml"), n_candidates=30, max_attempts=50000, seed=1, modules=("model",))` 实跑得 `pass_rate=0.0005, n_legal=25` —— **精确复现** `search_space_size_v1.md` 的 Pyramid 0.05% 崩塌结论。诚实地把这作为"给 framework owner 的优化建议"提出 (单模块该从对齐网格采样)。
- (b) DoE **小而高信息、非随机**: 25 anchor = baseline(3) + sens OAT(6) + pareto_edge(3), 4090/Orin 镜像。`doe_design_v1.md` 给了信息论理由 (主效应 OAT + 四角覆盖 + 避免 selection bias)。设计合理。
- (c) 管线**真接通三工具** (见 §1.2)。
- (d) **12 个真剪枝点属实**: CSV 25 行里 6 个配置 (p25/p50/p75 × {sens,pareto}) × 2 硬件 = 12 行标 `exec_path=handwritten_pyramid` + `prune:real_rebuild` + 非空 `params_total_new`。其中 rtx4090 的剪枝点有真 ckpt 落盘 (已验)。
- **caveat (最重)**: CSV **全部 25 行 `is_real_measured=False`** —— latency/throughput/ap30/ap50/ap70/engine_size/build_secs **列全空**。`pipeline_note` 诚实写"quant/deploy 用 baseline ONNX 占位 (标 dry_run)" / "deploy 不真 build"。即:**剪枝 (x 侧) 是真的, 但响应面 (y 侧标签) 一个真测都没有。**

---

## 3. 集成断点清单 (按优先级)

| # | 优先级 | 断点 | 影响 | 证据 |
|---|--------|------|------|------|
| 1 | **P0 阻塞预测器** | DOE 25 行 `is_real_measured` 全 False, 无任何真测 latency/AP | **预测器无 y 标签可训** —— 这套流水线目前只产出了"配置 + 真剪枝小模型", 没有响应面。LGB latency/AP 预测器**现在训不了** | `doe_dataset_v1.csv` 全列空 + Counter({'False':25}) |
| 2 | **P0 阻塞 INT8 数据点** | QDQ ONNX scale dtype 不一致 → TRT INT8 实际只覆盖 10.8% 层 | 即便后续真 build, INT8 anchor 测出的 latency 也是"伪 INT8"(88% 层是 FP16), 加速倍率不可信, 污染预测器交互项 | `deploy_smoke_bev_encoder_qdq.json: Int8=38/353, int8_layer_pct=10.8` |
| 3 | P1 | 数据生成里 deploy 全程 dry_run, 真 build 只在 smoke 单独跑过 2 个 engine | engine_size/build_secs/真实 latency 无法回填到 DOE | `generate_dataset.py:205` 真 build 门控只在 4090 且当前未对 DOE anchor 触发 |
| 4 | P1 | 剪枝点只有 rtx4090 落了真 ckpt, Orin 镜像 anchor 无真 ckpt (跨平台靠 f 函数外推) | Orin 侧 12 行里 6 行是镜像/估算, 跨硬件 latency 需 M2 f 函数补 | CSV: orin_agx_* 行 params 有值但无独立 ckpt 目录 |
| 5 | P2 | per-module W-only 是全局 flag (量化师自述) | 混合 W-only/W+A 的搜索空间维度暂不可真区分 | quant smoke WARN 行 |
| 6 | P2 | searcher 单模块 pass_rate 0.05% (近对齐率几乎全否决) | 若用 searcher 自动产 Pyramid anchor 效率极低 (DoE 已绕过, 改手工锚) | 我复现 pass_rate=0.0005 |

---

## 4. 整体结论 — 流水线到了哪一步, 下一步补哪个洞

**已到达**: "可配置三维度工具 (剪枝真 rebuild / 量化真混精 / 部署真 build) + 统一 Config 契约 + DoE 锚矩阵 + 数据生成骨架" 全部就位且互通。这是扎实的**工程地基**, 接口一致性零违规, 真做出来的部分 (2.8M 剪枝 ckpt、QDQ INT8 engine、searcher 0.05% 复现) 都经得起核验。

**尚未到达**: **响应面 (y 标签) 为空** —— 数据集有 x (配置 + 真小模型) 没有 y (真测 latency/AP)。所以"→ 预测器"这最后一棒**还没接上**。

**下一步该补的洞 (按依赖序)**:
1. **先修断点 #2** (QDQ scale dtype), 否则后面所有 INT8 latency 都是伪的。
2. **再补断点 #1/#3**: 让 `generate_dataset.py` 对 12 个真剪枝 ckpt 真正跑 deploy build + latency bench + AP eval, 把 y 列填满 (至少 4090 侧 12 行)。
3. Orin 侧 (断点 #4) 用已拟合的 M2 f 函数 (R²>0.995) 跨平台外推, 明确标"估算"。
4. y 填满后才谈训 LGB 预测器。

**一句话**: 工具和契约是真的、剪枝是真的、engine 能 build, 但**数据集目前是"半成品"(只有 x 没有 y), 预测器训练阶段尚不能启动**。优先级最高的是修 INT8 scale dtype bug + 让数据生成真测出 latency/AP。
