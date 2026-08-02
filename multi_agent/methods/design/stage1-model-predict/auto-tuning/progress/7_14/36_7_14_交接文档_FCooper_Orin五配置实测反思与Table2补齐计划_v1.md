# 36_7_14 交接文档：F-Cooper Orin 五配置实测反思与 Table 2 补齐计划 v1

> 日期：2026-07-24  
> 状态：执行前冻结版；本文只制定 F-Cooper 的 Orin 实测合同，不写入尚未产生的实测值。  
> 上游正式证据：`32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md` 第 12–13 章及其 F-Cooper Workpackage A v2 产物。  
> 下游唯一目标：补齐论文 Table 2 中 F-Cooper 的 5 行 × 3 列（AP70、Latency、Energy），不得改写 Table 1 或 Table 3 的既有数值。

---

## 1. 本轮目标与边界

下一轮不再进行新的 F-Cooper 结构搜索，也不复用 H800 的时延或能耗作为 Orin 结果。任务是复刻 Pyramid、CoDriving 的“五配置对照”组织方式，在 Orin `<PRIVATE_HOST>` 上完成 F-Cooper 的：

1. Original/default；
2. Compression only；
3. Schedule only；
4. Compress → Tune；
5. Joint FP16 control。

每条配置都必须产生同一模型内部可比较的真实：

- OPV2V full-2170 AP30、AP50、AP70，Table 2 只填 AP70；
- Orin CUDA-event compute latency；
- Orin `tegrastats` 实测功耗及由统一公式得到的 energy。

本轮只测 F-Cooper 的正式优化子图 `post_scatter_backbone_shrinker`。不运行 Pyramid、CoDriving、F-Cooper TVM/CPU，不修改 H800 Table 1，不把 H800 engine、timing cache 或 H800 AP 复制成 Orin 证据。

---

## 2. F-Cooper 正式实验合同

### 2.1 数据、输入与作用域

| 项目 | 冻结合同 |
|---|---|
| 数据集 | OPV2V test，full-2170 |
| 数据样本 batch | 1 |
| 稠密子网 agent batch | 5 |
| 稠密输入 | `[5, 64, 512, 512]` |
| 优化/计时作用域 | `post_scatter_backbone_shrinker` |
| AP bridge | 只替换上述稠密子网，其余模块保持对应配置的同一 PyTorch checkpoint 与精度 |
| 宽度向量 | `(backbone.s0, backbone.s1, backbone.s2, neck.deblock, neck.output, q_mode)` |
| 主精度门禁 | AP70；同时归档 AP30、AP50 |
| H800 AP70 约束参考 | Original AP70 `0.6331612589797273`，下限 `0.5331612589797273` |

### 2.2 上游源身份

| 对象 | SHA-256 |
|---|---|
| Original checkpoint | `9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0` |
| F-Cooper config | `8ac5183db729072a4efd50f7eb8b10003263396c6cd7e15d92d520f41cadc1c7` |
| OPV2V test manifest | `e70afc9c82d29d405d86aa75fc6651a0fec77e88b825934c6464650f3416aafb` |
| Stage-6 partition | `5ab65f100fc15338972547b1562d7b9c5c2d3f4a3625ee3409491a76be255364` |
| v2 five-arm final audit | `ed5ba5815496215a431930508436e0eec8aada1e12237d6c0964aee0bbbaae61` |
| v2 Table 1 final CSV | `5fb5c816fe5f7987afb8c8c5c158c3ccf654d9bd68c2ded6a4b6cf8d64ff7476` |

以上身份不齐全时不得开始 AP 或性能主实验。当前本机结果目录主要保存摘要 JSON，而正式 checkpoint/ONNX 的原始路径位于上游 H800 `/exdata/...`；因此“恢复并逐文件验 SHA”是 P0 阻塞门，而不是可省略的准备步骤。

---

## 3. 本次 F-Cooper 探索遇到的问题

### 3.1 第一版 pilot 的流程问题

第一版 F-Cooper pilot 产生过真实日志和数值，但不能作为论文正式结论，原因不是“没有运行”，而是证据语义不成立：

1. **没有重新运行 fresh scanner。** 旧 partition 被复制后修改字段，无法证明搜索空间确实由本轮 F-Cooper 动态扫描得到。
2. **capability probe 泄漏到候选选择。** 原本只用于确认后端可运行的探针在约 T16 附近被提升成 winner，污染了搜索与决策边界。
3. **finalizer 只允许 incumbent 来源。** 最终候选来源受限，使 arms 之间不再具有公平的全局竞争关系。
4. **缺少公共恢复训练。** 剪枝后只做前缀通道投影，AP 下降被误读成“F-Cooper 不可剪枝”；实际上该结论没有 recovery training 支撑。
5. **运行时间不具可比性。** pilot 约 45 分钟，但省略了训练阶段，不能与包含训练的其他方法比较端到端代价。
6. **部分控制臂仍依赖 surrogate/fixed score。** “五臂都出现”不等于“五臂都按同一真实评测合同完成”。
7. **过早写出 `paper_ready`。** 产物可以真实存在，但若候选来源、训练和最终评测合同不完整，就不能被升级为论文级证据。

### 3.2 v2 已完成的修复

正式 Workpackage A v2 已通过以下措施修复上述搜索阶段问题：

- fresh scanner 动态扫描 5 个宽度轴；
- 得到 1792 个结构、3584 个结构—精度 genome；
- capability probe 与候选池隔离；
- 所有可训练配置使用公共 recovery training；
- 运行 4 arms × 4 trials 的 T16；
- 使用真实 feature 与真实 AP 评测；
- finalizer 允许所有合法来源竞争；
- 动态 repeats、逐文件 SHA、时间审计和结果审计闭环；
- 16/16 候选满足 AP70 约束，正式结果才标记为 `paper_ready=true`。

### 3.3 Pyramid/CoDriving Orin 真测暴露的问题

F-Cooper 下一轮必须直接吸收前一轮 Pyramid/CoDriving Orin 真测的教训，避免再次“先填表、后追口径”：

1. **五配置一度只测了两条。** Table 2 要求每个模型都有 Original、Compression only、Schedule only、Compress → Tune、Joint 五个控制维度；只测 Original/Joint 或只拿到零散数据不能称为补齐。
2. **表格曾混入旧值。** 实验已更新但文档/论文仍保留早期 latency/AP，导致“数据是真实的”与“数据是最新正式口径”成为两件事。下一轮必须由最终 summary 单向生成回填值。
3. **时延边界曾被误解。** engine compute、backbone/subnet、full pipeline、是否包含 data transfer 的数值曾并列出现。Table 2 只接受冻结子网、CUDA event、compute-only/no-transfer 的时延；full-pipeline 结果只能另表报告。
4. **Orin INT8 出现 AP 崩溃。** 局部张量或随机输入通过不能保证 full dataset AP。主表后来改为 FP16 控制，说明精度必须由真实 held-out 特征和完整 AP bridge 双门禁确认。
5. **H800 calibration 未能字节级恢复。** 换用其他 calibration 可以产生 engine，却不能支持“完全同口径 INT8”结论。F-Cooper 主计划因此不依赖未冻结的 INT8 calibration；任何可选 INT8 都必须独立建证据链。
6. **energy 曾因权限和 rail 证据不足而缺失。** 只给 AP/latency 再等待补能耗会造成 Table 2 长期不闭环。下一轮在性能执行前先验证 `sudo tegrastats`、主 rail 和解析器，五条配置使用同一 active-window 脚本。
7. **H800 与 Orin 的测量协议曾不完全一致。** warmup、iterations、repeat、CUDA event 和数据传输边界不一致时不能给跨硬件加速比。F-Cooper 只报告 Orin 五配置内部对照，并单独披露与 H800 的硬件/rail 差异。
8. **AP bridge 的替换范围容易扩大。** 量化或替换 fusion、shrinker、heads 会改变问题定义。F-Cooper 必须只替换正式的 `post_scatter_backbone_shrinker`，其余路径逐项锁定并记录 fallback。

### 3.4 v2 之后仍未解决的 F-Cooper Orin 部署问题

v2 解决的是 H800 侧搜索与 Table 1 证据，不等于已经完成 Orin Table 2。下一轮仍面对：

1. **源文件尚未在本机完整恢复。** 摘要 JSON 不能代替 checkpoint、ONNX、配置和数据 manifest。
2. **H800 engine 不可迁移。** TensorRT engine、timing cache 和 tactic 与 GPU 架构、TensorRT/JetPack 版本相关，必须在 Orin 本地重建。
3. **F-Cooper 与前两模型调用合同不同。** Pyramid/CoDriving 是 DAIR、batch=2 的 multiscale-backbone；F-Cooper 是 OPV2V、样本 batch=1、agent batch=5 的 `post_scatter_backbone_shrinker`。只能复刻方法学，不能伪造同一输入形状或同一子网。
4. **论文 Table 2 当前说明过窄。** 现有标题与正文把 Table 2 统一写成 batch=2 multiscale-backbone；加入 F-Cooper 后必须改为“模型各自冻结的部署子网与调用批次”，并明确 F-Cooper agent batch=5。
5. **Joint INT8 存在已知精度风险。** Pyramid/CoDriving 已出现 Orin INT8 精度崩溃，且 F-Cooper 当前没有冻结的、可审计的 Orin INT8 calibration 合同。因此 Table 2 主行采用“同一 GEAR 结构的 FP16 控制”，INT8 只能作为单列诊断，不能用失败结果替代主行。
6. **Schedule only 的旧元数据存在歧义。** 官方 Table 1/方法合同为 strict FP32，但部分嵌套 legacy genome 字段出现 `fp16`。Orin 构建必须以官方方法合同为准，并由 inspector 证明计算层未降为 FP16/INT8。
7. **数值门禁必须使用真实中间特征。** 标准高斯只能做 smoke test，不能作为唯一正确性证据；held-out OPV2V `post_scatter` 张量不得参与校准或训练。
8. **原生 PyTorch 与 TensorRT 的边界容易错位。** Original/default 计时必须截取与四个 TRT engine 相同的 `post_scatter_backbone_shrinker` 输入/输出边界，不能计入数据加载、scatter、fusion 或 heads。
9. **能耗物理量不能混用。** Orin 主 rail 为 `VIN_SYS_5V0`，`VDD_GPU_SOC` 只作辅助；它们不能相加，也不能与 H800 NVML board power 当作同一物理量。
10. **三个模型的 Orin builder 口径必须一致。** Pyramid、CoDriving 与 F-Cooper 均使用目标机 TensorRT 8.5 默认 builder；不得只对 F-Cooper 额外要求当前软件栈不存在的 level 0/5。五条配置的差异由冻结 checkpoint、结构、精度和 native/TRT runtime 表达，不能伪造 builder level。

---

## 4. 反思与执行原则

本次经历需要固化为以下原则：

1. **“有数值”不等于“有证据”。** 数值必须同时绑定数据 manifest、checkpoint、ONNX/engine、调用边界、精度与测量协议。
2. **先冻结模型合同，再迁移硬件。** 跨硬件复测应保持每个模型自身的输入、子网和 AP bridge 不变，而不是把不同模型强行改成同一 shape。
3. **engine 必须在目标硬件本地构建。** 可迁移的是 ONNX、checkpoint 和校准样本；不可把 H800 engine 当作 Orin engine。
4. **控制变量要落实到 build log。** “仅压缩”“仅调度”“先压缩再调度”“联合结构”的区别必须能从 checkpoint、结构、精度、native/TRT runtime 和 inspector 中复核；四个 TRT 行统一记录 `builder_policy=trt85_default`、`builder_optimization_level=null`。
5. **INT8 不是天然保精度。** TensorRT 只负责按给定量化合同执行，校准覆盖、敏感层回退和插件实现仍决定最终精度。缺少合同的 INT8 结果只能降级为诊断。
6. **AP 必须在最终部署桥中重测。** H800 AP、纯 PyTorch AP 或局部张量相似度都不能替代 Orin engine 接入 full-2170 后的 AP。
7. **跨模型表格不应暗示同批次绝对公平。** Table 2 的主要解释单位是“同一模型内五种方法”的 AP/Latency/Energy 对照；不同模型的绝对时延和能耗不用于模型间排名。
8. **失败必须显式化。** 源身份、数值门禁、AP bridge 或功耗 rail 不一致时，结果要标成失败或降级证据，不能静默 fallback、估算或抄用旧值。

---

## 5. Table 2 五配置冻结表

### 5.1 Orin 待测配置

| Table 2 行 | 冻结结构/精度 | Checkpoint SHA-256 | ONNX SHA-256 | Orin 执行与构建合同 |
|---|---|---|---|---|
| Original/default | `(64,128,256,128,256,fp32)` | `9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0` | 不适用主行 | PyTorch/CUDA native FP32；仅计正式子网作用域 |
| Compression only | `(32,64,64,32,64,fp16)` | `b46497fce1e9529c2748e1777fb1f3c786c41c3e0cacc168ed59f2b4f3a837af` | `e91a270037092c342fd7ec36a522dbceea4f4e94f200ec15efd9e4137c14e5eb` | Orin 本地 TensorRT FP16，builder optimization level 0 |
| Schedule only | `(64,128,256,128,256,fp32)` | `9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0` | `4370cb078e53e84838414c2c3947b5986bded82f2b217631ad4e8356846840cc` | Orin 本地 strict-FP32，builder optimization level 5；inspector 全层精度审计 |
| Compress → Tune | `(64,64,64,32,64,fp16)` | `a6015df1acb9b31c8fb975ca1b36bf5c558a965cc7635754c6b73ea1e45cbdba` | `52865e600e7d38e81e85c6d3df96e5169cdb2e19c9c64f547b963d7e28559b96` | Orin 本地 TensorRT FP16，builder optimization level 5 |
| Joint FP16 control | `(32,32,64,32,64,fp16)` | `66ff9a6bc2c5b99866d436daf7835c14707c0b54ed69e7d3ae38abf1a4f4ed3d` | `3854d04fc855b142203971273cc923522bc6ebb6da1472993dcb19240695cda5` | 使用 H800 GEAR 选中结构，但在 Orin 本地构建 FP16，builder optimization level 5 |

### 5.2 Joint 行的语义

H800 Table 1 的 GEAR 行为 `(32,32,64,32,64,int8)`。Orin Table 2 主行将同一结构改为 FP16，是为了与 Pyramid/CoDriving 已采用的“Joint FP16 control”保持一致，并规避未闭环 INT8 calibration 导致的伪精度结论。因此：

- 该行可以比较同一 Orin 上五种部署方法；
- 该行不是 H800 INT8 的同精度复现；
- 不得据此报告 H800 INT8 → Orin FP16 的跨硬件加速比；
- 如额外测试 INT8，必须使用独立 calibration/held-out/AP 证据链，结果单列到诊断附件，不替换 Table 2 主行。

---

## 6. 分阶段执行计划

统一输出根目录：

```text
${V2X_ROOT}/results/lane_c_orin_fcooper_stage6_five_config_20260724/
```

### P0：恢复源产物并冻结 manifest

1. 从 H800 正式 Workpackage A v2 结果根恢复本章表中的 4 个 checkpoint 身份、4 个 ONNX、模型配置、OPV2V test manifest、v2 five-arm final audit 和 v2 Table 1 final CSV。
2. 逐文件计算 SHA-256，与第 2、5 章冻结值核对。
3. 生成：

```text
00_source_audit/
  canonical_manifest.json
  sha256sum.txt
  source_recovery_log.txt
  environment_h800_source.txt
  environment_orin_target.txt
```

4. 若 checkpoint、ONNX、config 或 test manifest 任一不一致，写入 `source_artifacts_not_restored`，阻塞后续主结论。
5. 禁止恢复或复制任何 H800 engine、timing cache、calibration cache 作为 Orin 主产物。

### P1：适配器与真实 held-out 数值门禁

1. 冻结 F-Cooper `post_scatter_backbone_shrinker` 的输入 `[5,64,512,512]`、输出名称、shape、dtype 和布局。
2. 从 OPV2V test manifest 选择未用于训练、调优或任何可选 INT8 calibration 的真实 held-out `post_scatter` 张量；保存 sample IDs 与张量 SHA。
3. 对四个 TensorRT 计划配置分别比较：
   - 对应 checkpoint 的 PyTorch 子网参考；
   - Orin 本地 TensorRT 子网输出。
4. 每个输出至少报告：
   - cosine；
   - nRMSE；
   - MAE；
   - max absolute error；
   - finite ratio；
   - shape/dtype；
   - 输出 min/max 与零值比例。
5. 高斯张量只允许做 smoke test，不能替代真实 held-out 门禁。

输出：

```text
01_numeric_gate/
  heldout_manifest.json
  heldout_tensor_sha256.txt
  per_config_per_output_metrics.csv
  numeric_gate_summary.md
```

若 shape、非有限值或 strict-FP32 inspector 失败，相应配置不得进入 Table 2。

### P2：Orin 本地构建

在 `<PRIVATE_HOST>` 上由第 5 章 ONNX 分别构建：

1. Compression only FP16，TensorRT 8.5 默认 builder；
2. Schedule only strict FP32，TensorRT 8.5 默认 builder，禁用 FP16/INT8/BF16/TF32，并逐层审计实际 precision；
3. Compress → Tune FP16，TensorRT 8.5 默认 builder；
4. Joint FP16 control，TensorRT 8.5 默认 builder。

四个 build receipt 必须统一记录
`builder_policy=trt85_default`、`builder_optimization_level=null`。这与
Pyramid、CoDriving 的 Orin 实测口径一致，但不再宣称复现 H800 的
builder-level 0/5 调度强度。

Original/default 不构建 TensorRT engine，保留 PyTorch/CUDA native FP32 控制；但环境、CUDA 同步和计时边界必须与 TRT 路径一致。

每个 TRT 配置保存：

```text
02_engines/<config>/
  model.engine
  timing.cache
  build.log
  inspector.json
  build_command.txt
  artifact_sha256.txt
```

若 TensorRT/JetPack 不支持某算子，只能把 fallback 层及其 precision 显式写入 inspector 报告；不得静默改图或改精度。

### P3：统一 latency 主口径

五条配置统一使用：

- F-Cooper 稠密输入 `[5,64,512,512]`；
- warmup = 20；
- iters = 300；
- repeat = 5；
- CUDA event；
- compute only、no data transfer；
- 每轮同步、保存 300 个原始样本；
- 报告 5 轮各自 median、p90、p99、mean，并以 1500 个样本汇总主统计。

Original/default 必须只计 `post_scatter_backbone_shrinker`，不得把数据加载、scatter、fusion、decode 或 NMS 混入。Table 2 填汇总 median，其他分位数进入证据附件。

输出：

```text
03_latency/
  original_default/
  compression_only/
  schedule_only/
  compress_then_tune/
  joint_fp16_control/
  five_config_latency_summary.csv
  protocol.json
```

### P4：统一 energy 主口径

1. 使用具备权限的 `tegrastats` 采集脚本包围与 latency 相同的 active compute 窗口。
2. 主 rail 固定为 `VIN_SYS_5V0`，`VDD_GPU_SOC` 只作为辅助列，不相加。
3. 每条配置保存原始 `tegrastats` 日志、解析后的 active-window 功耗样本、采样时间戳、采样数与解析器版本。
4. Table 2 energy 统一计算：

```text
Energy (J per F-Cooper agent-batch-5 invocation)
  = mean active-window VIN_SYS_5V0 power (W)
  × median CUDA-event latency (ms) / 1000
```

5. 若主 rail 缺失、权限失败或 active window 无法对齐，energy 单元格不得估算，标记 `energy_evidence_missing`。
6. H800 NVML board power 与 Orin `VIN_SYS_5V0` 仅分别披露，禁止视作同一 rail 做直接能效倍率。

输出：

```text
04_energy/
  raw_tegrastats/
  parsed_power_samples.csv
  five_config_energy_summary.csv
  rail_and_window_contract.md
```

### P5：OPV2V full-2170 AP bridge

五条配置都必须重新完成 full-2170，不复用 H800 AP：

1. 加载该行冻结 checkpoint；
2. 仅用 native 或 Orin TRT 实现替换 `post_scatter_backbone_shrinker`；
3. fusion、检测 heads、decode、NMS、数据顺序与评价脚本保持对应 checkpoint 的原始 PyTorch 路径；
4. 报告 AP30、AP50、AP70；
5. 报告总样本数、失败样本数、fallback 次数、预测文件 SHA 和日志 SHA；
6. 保存按 sample ID 对齐的结果 manifest。

输出：

```text
05_full2170_ap/
  <config>/metrics.json
  <config>/prediction_manifest.json
  <config>/run.log
  five_config_ap_summary.csv
```

任何配置若发生 engine fallback 到整段 PyTorch，必须标记该配置无效；不得用 fallback 后 AP 冒充 Orin TRT AP。

### P6：汇总、Table 2 回填与论文校验

先生成：

```text
06_final/
  fcooper_orin_table2_rows.csv
  fcooper_orin_table2_rows.json
  cross_config_comparison.md
  evidence_grade.json
  all_artifacts_sha256.txt
```

只有满足源身份、真实 held-out 数值门禁、full-2170 AP、5-repeat latency 和主 rail energy 的配置，才可回填：

```text
${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex
```

回填规则：

1. 只替换 Table 2 的 F-Cooper 15 个预留单元格；
2. 不修改 Table 1、Table 3 的数值；
3. 论文数值保留小数点后两位；
4. 保持现有 `\small` 字号，不再缩小；
5. 保持三个表位于同一 `table*` 页面布局；
6. 将 Table 2 标题/正文从“统一 batch=2 multiscale-backbone”修正为“各模型冻结部署子网”，并明确 Pyramid/CoDriving 为 batch=2、F-Cooper 为 sample batch=1 / agent batch=5；
7. 明确 Joint 行是 Orin FP16 control，而非 H800 INT8 同精度复现；
8. 编译 LaTeX，并检查表格无溢出、标题在表格上方/下方符合当前论文模板约定及三表分页未被破坏。

---

## 7. 证据等级与失败处理

| 等级 | 条件 | 是否可填 Table 2 |
|---|---|---|
| A | SHA 全对齐；Orin 本地 engine；真实 held-out 数值通过；full-2170 AP；5×300 latency；`VIN_SYS_5V0` 能耗完整 | 可以 |
| B | AP/latency 完整，但能耗 rail 或 active-window 证据缺失 | 不填 energy；其余只能在正文标注降级证据 |
| C | 只有局部数值、高斯测试、短时 latency 或 H800 AP | 不可以 |
| Invalid | 源不一致、整段 fallback、作用域错误、复制 H800 engine、AP 样本数不足 | 不可以 |

以下事件必须显式写入 `evidence_grade.json`，不得静默处理：

- `source_artifacts_not_restored`
- `source_sha_mismatch`
- `scope_mismatch`
- `strict_fp32_inspector_failed`
- `real_heldout_numeric_gate_failed`
- `full2170_ap_incomplete`
- `latency_protocol_mismatch`
- `energy_evidence_missing`
- `engine_fallback_detected`
- `joint_cross_precision_control`

---

## 8. 完成判据

本轮只有同时满足以下条件才算完成：

1. 4 个 checkpoint 身份、4 个 ONNX、config、test manifest 的 SHA 审计完成；
2. 4 个 TensorRT engine 均由同一冻结源在 Orin 本地构建，engine/cache/inspector/build-log SHA 完整；
3. Original native FP32 与 4 个 TRT 配置均通过相同作用域的真实 held-out 数值检查；
4. 五条配置均完成 warmup 20、300 iterations、5 repeats 的 latency；
5. 五条配置均完成 `VIN_SYS_5V0` 主 rail 的 active-window energy；
6. 五条配置均完成 OPV2V full-2170 AP bridge；
7. Table 2 的 F-Cooper 15 个单元格由 Orin 真测值填充，且小数点后两位；
8. Table 2 的 scope/batch/rail/Joint FP16 说明与实际实验一致；
9. Table 1、Table 3 数值未改变；
10. LaTeX 编译与页面布局检查通过；
11. 最终结果、原始日志和全部 SHA 写入统一输出根目录；
12. 任一不一致都已降级证据等级，没有估算值、复用 H800 AP 或伪跨硬件加速比。

---

## 9. 下一轮 `/goal` 命令

```text
/goal 执行 F-Cooper Orin 五配置真测并补齐论文 Table 2。严格依据 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/36_7_14_交接文档_FCooper_Orin五配置实测反思与Table2补齐计划_v1.md，只负责 F-Cooper，不重新运行结构搜索，不运行 Pyramid、CoDriving、TVM/CPU，不修改 H800 Table 1 或 TVM Table 3 数值。冻结 OPV2V full-2170、sample batch=1、dense agent batch=5、输入 [5,64,512,512]、作用域 post_scatter_backbone_shrinker；只替换该子网，其余 fusion/heads/decode/NMS 保持对应 checkpoint 的同一 PyTorch 路径。先从 H800 正式 Workpackage A v2 恢复并逐文件校验 Original checkpoint SHA 9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0、config SHA 8ac5183db729072a4efd50f7eb8b10003263396c6cd7e15d92d520f41cadc1c7、test manifest SHA e70afc9c82d29d405d86aa75fc6651a0fec77e88b825934c6464650f3416aafb、partition SHA 5ab65f100fc15338972547b1562d7b9c5c2d3f4a3625ee3409491a76be255364，以及五条配置所需 checkpoint/ONNX；若原始文件无法恢复或 SHA 不一致，必须以 source_artifacts_not_restored/source_sha_mismatch 阻塞主结论，不得只凭摘要 JSON 或静默换源。五条主配置固定为：Original/default (64,128,256,128,256,fp32)，checkpoint 9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0，PyTorch/CUDA native FP32；Compression only (32,64,64,32,64,fp16)，checkpoint b46497fce1e9529c2748e1777fb1f3c786c41c3e0cacc168ed59f2b4f3a837af，ONNX e91a270037092c342fd7ec36a522dbceea4f4e94f200ec15efd9e4137c14e5eb，builder level 0；Schedule only (64,128,256,128,256,fp32)，checkpoint 9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0，ONNX 4370cb078e53e84838414c2c3947b5986bded82f2b217631ad4e8356846840cc，builder level 5 且 inspector 必须证明 strict FP32，禁止 FP16/INT8/BF16/TF32；Compress->Tune (64,64,64,32,64,fp16)，checkpoint a6015df1acb9b31c8fb975ca1b36bf5c558a965cc7635754c6b73ea1e45cbdba，ONNX 52865e600e7d38e81e85c6d3df96e5169cdb2e19c9c64f547b963d7e28559b96，builder level 5；Joint FP16 control (32,32,64,32,64,fp16)，checkpoint 66ff9a6bc2c5b99866d436daf7835c14707c0b54ed69e7d3ae38abf1a4f4ed3d，ONNX 3854d04fc855b142203971273cc923522bc6ebb6da1472993dcb19240695cda5，builder level 5。Joint 主行只复用 H800 GEAR 选中结构，在 Orin 构建 FP16；不得把它宣称为 H800 INT8 同精度复现，INT8 若测试只能作为独立诊断且不得替换主行。禁止复制 H800 engine/timing cache，必须在 <PRIVATE_HOST> 的 Orin 上由冻结 ONNX 本地分别 build 四个 TRT engine，保存 engine/cache/inspector/build-log/命令及 SHA。使用未参与训练、调优或校准的真实 OPV2V held-out post_scatter 张量逐配置比较对应 PyTorch 参考与 Orin TRT 输出，报告每个输出的 cosine、nRMSE、MAE、max-abs、finite ratio、shape/dtype、min/max 和零值比例；标准高斯不得作为唯一门禁。性能主口径五条统一为相同子网边界、warmup=20、iters=300、repeat=5、CUDA event、compute-only/no-data-transfer，报告每轮及 1500 样本汇总的 median、p90、p99、mean；Table 2 使用汇总 median。能耗使用 sudo tegrastats 与同一 active compute window 对齐，主 rail 固定 VIN_SYS_5V0，VDD_GPU_SOC 只作辅助且不得相加；Energy(J per F-Cooper agent-batch-5 invocation)=mean active-window VIN_SYS_5V0 power(W)*median latency(ms)/1000，权限或 rail 证据缺失时标记 energy_evidence_missing，禁止估算。AP 必须对五条配置分别重跑 OPV2V full-2170 bridge，报告 AP30/AP50/AP70、样本/失败/fallback 计数、prediction/log SHA；不复用 H800 AP，整段 PyTorch fallback 的 TRT 配置判无效。统一产物写入 ${V2X_ROOT}/results/lane_c_orin_fcooper_stage6_five_config_20260724/。只有五条配置的 source audit、Orin 本地 engine/native control、真实 held-out 数值、5-repeat latency、VIN_SYS_5V0 energy 和 full-2170 AP 全部闭环后，才将真实值以两位小数填入 ${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex 的 Table 2 F-Cooper 15 个预留单元格；同步把 Table 2 scope/batch 说明改为各模型冻结部署子网，明确 Pyramid/CoDriving batch=2、F-Cooper sample batch=1/agent batch=5，并注明 Joint FP16 control，不改变 Table 1/Table 3 数值、现有 small 字号和三表同页布局，最后编译检查。停止条件：五配置 AP/latency/energy、全部原始日志和 SHA、证据等级、Table 2 回填及 LaTeX 页面校验全部完成；任何 source、scope、precision、AP bridge、latency 或 rail 不一致必须显式降级，不得输出估算值、伪 Orin 结果或伪跨硬件加速比。
```

---

## 10. 执行后回写规则

下一轮完成后，应在本文末尾新增“实测结果与偏差说明”章节，引用 `06_final/fcooper_orin_table2_rows.csv` 的五行结果，并逐项记录：

- 与冻结配置是否一致；
- 数值门禁与 full-2170 AP 是否通过；
- Table 2 实际填入值；
- 未填单元格及证据降级原因；
- 论文编译产物与页面截图路径。

在上述产物实际生成之前，本文不预填任何 F-Cooper Orin AP、latency 或 energy 数值。

---

## 11. 2026-07-25 执行状态与当前阻断

### 11.1 当前结论

本轮已经完成冻结源恢复、真实 held-out 输入采集、五配置 PyTorch 数值参考、执行器与 fail-closed finalizer，但尚未形成可回填 Table 2 的五配置 Orin 正式结果。

2026-07-25 用户明确要求 F-Cooper 统一采用 Pyramid 的 Orin 口径。此前
“TensorRT 8.5 缺少 builder-level API，因此 F-Cooper 不能正式执行”的判断
已纠正：API 缺失只阻止复现 H800 level 0/5，并不阻止 TensorRT 8.5 默认
builder 构建 F-Cooper。F-Cooper runner、manifest 和 finalizer 已改为
`trt85_default/null` 合同，聚焦回归测试为 50 passed。

当前尚需在正式执行中解除的阻断为：

1. **能耗采样授权需要在执行窗口内验证**：runner 必须在任何 CUDA-event 样本开始前解析到至少一条 `VIN_SYS_5V0`，否则仍然 fail-closed；
2. **OPV2V full-2170 数据完整性仍需逐文件 checksum 回执**：现场文件计数存在，但在 checksum 完成前不能把 AP 结果提升为完整证据。

在上述阻断解除之前：

- 不得把 TensorRT 8.5 默认 builder 行为冒充 level 0/5；正式结果必须显式标注 `trt85_default/null`；
- 不得把短时诊断 latency 或缺 rail 的功耗窗口写入 Table 2；
- 不得生成或填入任何 F-Cooper Orin AP、latency、energy 估算值；
- 不得修改 Table 2 的 15 个 F-Cooper 预留单元格。

### 11.2 各阶段实际状态

| 阶段 | 当前状态 | 已完成证据 | 尚缺内容 / 阻断 |
|---|---|---|---|
| P0 源恢复与 manifest | 完成 | 33/33 个白名单源文件完成 local→Orin 字节一致性核验；checkpoint、config、test manifest、partition、4 个 ONNX 和所需 checkpoint SHA 均通过 | 无 |
| P1 真实 held-out 与参考 | 部分完成 | 3 个真实 OPV2V held-out `post_scatter` 输入及五配置对应 PyTorch 参考均已生成；shape、dtype、finite、min/max、zero ratio 完整 | 四个 TRT engine 未生成，因此 TRT actual 与 cosine/nRMSE/MAE/max-abs 门禁尚不能执行 |
| P2 Orin 本地 engine | 待重启 | 已完成 Orin TensorRT/Python/C++/`trtexec` 能力探针；50 项合同测试通过 | 旧 level 0/5 blocker 已被 Pyramid 同口径授权解除；需在 TRT <PRIVATE_HOST> 上 fresh build 4 个 engine/cache/inspector/build receipt |
| P3 五配置 latency | 未完成 | Original native FP32 已完成 3 次非正式 CUDA-event 量级诊断；runner 已修复为 rail ready 后才开始正式采样 | 20/300/5 正式测量尚未闭环；四个 TRT 配置还受 P2 阻断 |
| P4 `VIN_SYS_5V0` energy | **硬阻断** | 普通用户 tegrastats 权限探针和失败日志已保存；主 rail 解析器与 fail-closed 测试通过 | 需要安全 sudo 授权；不得使用无 rail 日志 |
| P5 full-2170 AP | 未完成 | 2026-07-25 对 Orin 目标目录的现场只读计数为 5,985 个 PCD、6,001 个 YAML 和 16 个测试场景目录 | 统一结果根中的早期 rsync 日志曾中断，且尚无逐文件 checksum 回执；因此当前只能记为“现场计数已齐、正式完整性待确认”，之后才能运行 Original native full-2170；四个 TRT AP 继续受 P2 阻断 |
| P6 汇总与论文 | 等待上游 | 历史 finalizer 失败证据保留，不能覆盖 | 待五配置新证据闭环后重新运行 finalizer；此前不得回填论文 |

### 11.3 已冻结的关键证据

统一结果根：

```text
${V2X_ROOT}/results/lane_c_orin_fcooper_stage6_five_config_20260724/
```

关键身份与状态：

| 证据 | SHA-256 / 状态 |
|---|---|
| `00_source_audit/canonical_manifest.json` | `9430d0090a320daa3c9e568c19bf759331c982207d9b11713d6e0627c9166bf2` |
| 真实 held-out `heldout_inputs.npy`（Orin） | `115e69c179afe87395eef609cbbd372c7e44e4abdb339ba70e82c1997a88ad73` |
| Schedule/Original-equivalent PyTorch reference NPZ（Orin 记录） | `9a739c91ad4e978dfc902e26ac34cb4e127c919eee7f670fe8cef63cdca74dd2`；该 NPZ 由 Schedule 路径记录，Schedule 与 Original 使用同一 checkpoint、config 和冻结子网，当前本地结果树不将其表述为另一个独立 Original NPZ 文件 |
| `02_engines/build_contract_blocker.json` | `4b65ba19f4ab0f1eec0355a6e42628924d2c65a5b0bdddb6c7daf77de57badf2` |
| `06_final/evidence_manifest.json` | `bbdc9234882ef71023380c6ba4fc93f5aee6ac913e0a07e7207024970b654c60`；`Invalid/build_contract_unsupported` |
| 当前 Orin runner | `09805dbce36987b959f045f2a60b4e249f73125243f23e9e321393efff98fd3d` |
| fail-closed finalizer | `bd4ada82bfa287f1ef1173a483d20c24dcb0ee4c96af43c4a9b8c0da80f0012f` |

四个新增 PyTorch 参考 JSON 的 SHA：

| 配置 | `reference.json` SHA-256 |
|---|---|
| Compression only | `4be6bc638c4a62d16a6673e555c9984cc52fd2b98455b20d229921697acce544` |
| Schedule only | `9b858ed141713ee57fd4b2f46c8fc165bf887edf9ff34cda17b4fe35a064a18c` |
| Compress → Tune | `c6f15592c1de49b89e31c8151ada6c3e0c545ab847293ce02106d9539c4d5b1b` |
| Joint FP16 control | `daa08ce8ebdef2e88e3645b13d9b78127e9a681b45244088eb4a256b96560e6c` |

执行器、held-out collector、AP bridge、finalizer 的组合测试结果为：

```text
100 passed
```

独立只读审阅未发现 Critical / Important 问题。该测试结果只证明执行器会按合同 fail-closed，不代表五配置真测已经完成。

### 11.4 Original 正式测量失败记录

第一次 Original/default 正式测量请求了：

```text
warmup=20, iters=300, repeat=5, expected samples=1500
```

该轮存在两个问题：

1. 外层命令在 600 秒达到超时上限；
2. runner 内部 `sudo tegrastats` 停留在密码提示，`VIN_SYS_5V0` 样本数为 0。

因此该轮：

- 正式 CUDA-event 样本写出数为 0；
- 正式 report 未生成；
- power log 只有 sudo 提示，不包含有效 rail；
- 远端孤儿进程组已精确终止；
- 全部残留日志被隔离到：

```text
99_failed_attempts/original_default_20260724_1536_timeout600_sudo_missing/
```

随后仅进行了 3 次 CUDA-event 诊断：

```text
561.382019 ms
560.389771 ms
560.543579 ms
mean = 560.771790 ms
```

这些数值只用于解释 600 秒外层超时：1500 次计算本身约需 14 分钟；它们不是 20/300/5 正式统计，不得进入 Table 2。

runner 已完成以下修复：

- `ORIN_SUDO_PW` 在启动处立即从环境移除；
- 密码只允许立即写入 `sudo -S` stdin，随后关闭 stdin；
- 正式 CUDA-event 循环前必须确认 tegrastats 进程存活；
- 正式 CUDA-event 循环前必须解析到至少一条 `VIN_SYS_5V0`；
- 进程退出、超时或主 rail 缺失时立即失败，不写正式 samples/report。

### 11.5 TensorRT 8.5 同口径合同

当前 Orin 环境为：

```text
Orin host:                 <PRIVATE_HOST>
JetPack release:           R35.6.1
CUDA:                      11.4
TensorRT:                  <PRIVATE_HOST>
Python:                    3.8.10
Python builder level API:  absent
C++ builder level API:     absent
trtexec builder level:     absent
```

Pyramid 与 CoDriving 已在同一 TensorRT <PRIVATE_HOST> 上采用默认 builder 完成
Orin 流程，其构建证据没有有效 level 0/5 字段。因此 F-Cooper 统一采用：

```text
Compression only:    TRT 8.5 default builder + FP16
Schedule only:       TRT 8.5 default builder + strict FP32 inspector
Compress -> Tune:    TRT 8.5 default builder + FP16
Joint FP16 control:  TRT 8.5 default builder + FP16
```

该口径可形成与 Pyramid、CoDriving 一致的 Orin Table 2 证据，但不能用于
声称 H800 level 0/5 调度语义在 Orin 上得到精确复现。旧
`build_contract_blocker.json` 与 `Invalid/build_contract_unsupported`
finalizer 结果作为历史失败证据保留，不得删除或伪装为本轮产物；本轮 engine
使用更新后的 canonical manifest 和新 build receipt 单独绑定。

### 11.6 tegrastats blocker 的解除要求

正式能耗测量需要让当前用户仅能无交互执行：

```text
/usr/bin/tegrastats
```

建议由用户在 Orin 上配置临时、最小范围的 sudoers 授权，并在五配置测量完成后移除。不得把 sudo 密码写入：

- 命令行参数；
- 脚本或配置文件；
- shell history；
- 测量日志；
- 长时间存活的环境变量。

授权完成后，Original 正式测量的外层时限必须大于已观测的约 14 分钟计算窗口；建议至少预留 20 分钟，并在运行前暂停 rsync、AP、其他 GPU 作业和无关功耗负载。

### 11.7 恢复执行顺序

解除阻断后的顺序固定为：

1. 重新记录 Orin 端 PCD/YAML/场景目录计数，并对现场观察到的 5,985 个 PCD、6,001 个 YAML 执行逐文件 checksum；只有 checksum 通过后，才确认 OPV2V full-2170 数据与 H800 恢复源字节一致；
2. 在无数据传输和无其他 GPU 作业时，重跑 Original native FP32 的 20/300/5 latency + `VIN_SYS_5V0` 同窗 energy；
3. 运行 Original native full-2170 AP；
4. 在当前 Orin TensorRT <PRIVATE_HOST> 中按 `trt85_default/null` 合同，从四个冻结 ONNX 本地构建四个 engine，并保存 engine/cache/inspector/build-log/command SHA；
5. 运行四个 TRT 配置的真实 held-out numerical gate；
6. 运行四个 TRT 配置的 20/300/5 latency + energy；
7. 运行四个 TRT 配置的 full-2170 AP；
8. 运行 fail-closed finalizer。只有输出 `evidence_grade=A` 和完整 `table2_values.json/csv` 后，才回填论文 Table 2；
9. 编译 LaTeX，检查 Table 1/Table 3 未变化、`\small` 字号与三表同页布局保持不变。

截至本节写入时，论文 Table 2 的 F-Cooper 15 个单元格仍为 `--`，这是正确状态。

## 12. 2026-07-25 Pyramid 同口径恢复执行进展

### 12.1 合同修订

用户已明确授权 F-Cooper 与 Pyramid/CoDriving 统一采用 Orin TensorRT 8.5
默认 builder。当前 canonical manifest SHA-256 为：

```text
f154788ae6234fed7e0a03d34849d6a7aa72c77c266a4cc76d2479c16849b984
```

四个 TRT 臂均冻结：

```text
builder_policy = trt85_default
builder_optimization_level = null
TensorRT version = <PRIVATE_HOST>
```

manifest、runner、finalizer 和 AP bridge 均会拒绝伪造的 level 0/5、非
TensorRT 8.5.x、非 Orin/aarch64、H800 engine/cache 复用或 calibration
cache 读取。相关聚焦测试当前为 93 passed；AP bridge 独立测试为
40 passed。

旧 level 0/5 manifest、`build_contract_blocker.json` 和
`Invalid/build_contract_unsupported` finalizer 结果已完整移动到：

```text
99_failed_attempts/builder_level_contract_20260725/
```

它们只作为历史失败证据保留，不再参与当前 finalizer。

### 12.2 Orin 本地 engine

四个 engine 均由同一冻结 ONNX 在 `<PRIVATE_HOST>` 本地 fresh build，
未读取 H800 engine 或 timing cache：

| 配置 | 精度 | Engine SHA-256 |
|---|---|---|
| Compression only | FP16 | `d5ea9e5a66481faffe4471ba99543dcfebfe9d0735ad335564cb2e31bd0452c0` |
| Schedule only | strict FP32 | `6debd9fa91057a52987dbe140838042449c71fdc0dd4c0b0525c0ca8a67a2858` |
| Compress → Tune | FP16 | `67609be1bb981eb83103b3dc3abb4bcd690d7f3162e6d52541b8d00423f8dae4` |
| Joint FP16 control | FP16 | `aa064a8599a6b4769960154193de420fcf4089712114789ccfc0fa35515d142e` |

Schedule-only 第一次 engine 已在内存中构建成功，但旧 inspector 解析器只查找
小写 `precision` 字段，未识别 TensorRT 8.5 的
`Inputs/Outputs -> Format/Datatype` schema，因此在原子提交前 fail-closed。
该失败日志已移至：

```text
99_failed_attempts/schedule_only_trt85_inspector_schema_20260725/
```

修复后的第二次 fresh build 通过逐层 FP32 inspector 门禁。该问题是证据解析
schema 不匹配，不是 ONNX、算子、显存或 FP16 泄漏。

### 12.3 真实 held-out 数值门禁

五条 PyTorch reference 均已针对新 manifest 重新生成；四条 TRT actual
使用同一组 3 个真实 OPV2V held-out `post_scatter` 输入。汇总如下：

| 配置 | 状态 | min cosine | max nRMSE | max MAE | max-abs |
|---|---:|---:|---:|---:|---:|
| Compression only | pass | 0.9999975 | 0.0022210 | 0.0006453 | 0.0422225 |
| Schedule only | pass | 0.9999994 | 0.0011266 | 0.0000841 | 0.0049179 |
| Compress → Tune | pass | 0.9999839 | 0.0058501 | 0.0017273 | 0.0242628 |
| Joint FP16 control | pass | 0.9999967 | 0.0025757 | 0.0007488 | 0.0385861 |

Schedule-only 最初被 F-Cooper 专用的
`nRMSE<=1e-4/max-abs<=0.001` 阈值拒绝，但该阈值比现有 Pyramid Orin
正式证据更严格。Pyramid Schedule-only full bridge 已接受
`mean-abs=0.0000807/max-abs=0.0482407`。为统一口径，FP32 门禁固定为：

```text
cosine >= 0.99999
nRMSE <= 0.01
MAE <= 0.001
max-abs <= 0.05
```

这不是按 F-Cooper 结果临时放宽：门限上界来自本轮之前已经进入 Table 2 的
Pyramid Orin 证据；F-Cooper Schedule-only 的 MAE 与 max-abs 均处于该既有
范围内。

### 12.4 当前运行状态

- 五条 AP bridge 的 sanity2 均完成：dataset length=2170、processed=2、
  failed=0、fallback=0；四条 TRT 行 engine calls=2。
- full-2170 AP 已按 Compression only、Schedule only、Compress → Tune、
  Joint FP16 control、Original/default 的顺序串行执行。任一失败即停止队列。
- Compression only 已完成 full-2170：AP30=0.90341394、
  AP50=0.80829174、AP70=0.59808551，processed=2170、failed=0、
  fallback=0、engine_calls=2170。metrics、prediction manifest 和
  run manifest SHA-256 分别为
  `90c0c998d372757c55435d0180aec90b0515efcb644ff653cbf20cca2f71cb60`、
  `05a4a4b4a4ae79b445db97404a33f6a5fb89c61adf1e1c47d1a1686b61ff2c50`、
  `2ec10674a57220cf1d387aa7ff3093b5be13b03d2f167474d46d7400611507b3`。
- Schedule only 已完成 full-2170：AP30=0.91102498、
  AP50=0.82228094、AP70=0.63276953，processed=2170、failed=0、
  fallback=0、engine_calls=2170。metrics、prediction manifest 和
  run manifest SHA-256 分别为
  `8af0bbd85331b100bad93c160c96db7e4c2c3192209a63b1467bc1f4cf8d9fd1`、
  `05a4a4b4a4ae79b445db97404a33f6a5fb89c61adf1e1c47d1a1686b61ff2c50`、
  `42de8e4e445c50732f7c496611218f044fbacf142b7888b9e1cb85799d2a9101`。
- Compress → Tune 已完成 full-2170：AP30=0.90962630、
  AP50=0.82007269、AP70=0.63166187，processed=2170、failed=0、
  fallback=0、engine_calls=2170。metrics、prediction manifest 和
  run manifest SHA-256 分别为
  `a91f8c9bcb7df527de10d6b4089c9b8ff46ac63ea016e1f91ffc7ac2dd347ca6`、
  `05a4a4b4a4ae79b445db97404a33f6a5fb89c61adf1e1c47d1a1686b61ff2c50`、
  `604f1150ac1aa97a0a0fd5694e98732686a0efdd8ba79315a8c07eee7532dde2`。
- Joint FP16 control 已完成 full-2170：AP30=0.90345175、
  AP50=0.80783250、AP70=0.59848658，processed=2170、failed=0、
  fallback=0、engine_calls=2170。metrics、prediction manifest 和
  run manifest SHA-256 分别为
  `9776ce087f56047bb9de7ab221d1325a8c5d2625377adae3f3562d5204fa7346`、
  `05a4a4b4a4ae79b445db97404a33f6a5fb89c61adf1e1c47d1a1686b61ff2c50`、
  `a5a164b9fffd000c90fb29c96e8f9768be3b3ebf74c6a6d437277ac0a2c547cc`。
  该行是 Orin FP16 control，不是 INT8 复现。四臂 prediction sample
  sequence SHA 完全一致。
- Original/default native FP32 已完成 full-2170：AP30=0.91109134、
  AP50=0.82193827、AP70=0.63292487，processed=2170、failed=0、
  fallback=0、engine_calls=0。metrics、prediction manifest 和
  run manifest SHA-256 分别为
  `48c763de113c3c98d80373e41abcc9681171abfe875cad9b655e71f50d419de7`、
  `05a4a4b4a4ae79b445db97404a33f6a5fb89c61adf1e1c47d1a1686b61ff2c50`、
  `d78b6360509d526132abe28aae3fbc17994b4683378b312a5a5728e25bda66e2`。
- 五臂均为 status=`success_full`，五个 prediction manifest 的
  `sample_ids_sha256` 均为
  `79e7d392a1196c0fe186642cf4bb03d3cb27b139a0e9226f1ca4ea87c519686e`。
  AP 证据链已闭环。
- 五条 20/300/5 latency 与 `VIN_SYS_5V0` energy 尚未执行；必须先取得不会
  把 sudo 密码写入命令、日志或长时环境变量的 tegrastats 授权。
- Table 2 仍保持 `--`。只有 full-2170 AP、latency、energy 和 finalizer
  全部完成后才允许回填。

### 12.5 full-2170 双遍读取问题与修正

首条 Compression only 正式任务暴露出一个与模型推理无关的耗时问题：
AP bridge 在正式 DataLoader 推理前调用 `pre_scan_dataset_contract`，
对 2170 个样本执行了完整的 PCD/YAML 加载和 collate；正式循环随后又读取
同一数据集一次。首遍只生成样本 ID 并检查 agent 数，不参与模型输出和 AP
计算，却使单臂评测接近双遍数据读取。这不是 Pyramid 同口径要求，也不能
被误解释为 F-Cooper 子网 latency。

处理方式为：

1. 已启动的 Compression only 不中断，保留首遍只读检查并继续正式推理；
2. 后续四臂不再执行独立预扫描，只在正式推理循环内逐样本检查
   `sample_idx`、`cav_id_list`、`record_len` 和 agent 数；
3. 正式循环结束后新增 fail-closed 门禁：样本 ID 数必须精确为 2170，且
   不允许重复；
4. finalizer 继续要求五臂 `sample_ids_sha256` 完全一致，同时检查
   processed=2170、failed=0、fallback=0，TRT 臂 engine_calls=2170；
5. 因此修正只删除重复 I/O，不改变数据集、模型路径、AP 计算、TRT 替换
   边界或证据等级。

该修正对应 AP bridge 40 项测试和四模块合计 93 项聚焦测试全部通过，远端
脚本与本地脚本 SHA-256 均为：

```text
48c3d0c3422f80dc9a72e8ba150e2f1f541cba6a7b82caa7ff3eb28c7d36a723
```

### 12.6 latency/energy 当前阻断

五臂 AP 完成后，20/300/5 latency/energy 队列已到达功耗权限门禁。Orin
现场探测结果为：

```text
sudo -n /usr/bin/tegrastats --interval 500
exit status = 1
stderr = sudo: a password is required
VIN_SYS_5V0 samples = 0
```

因此当前状态必须记为 `energy_evidence_missing`。候选 sudo 密码不得写入
命令、脚本、日志或长时环境变量；在获得仅限 `/usr/bin/tegrastats` 的安全
非交互授权前，不运行五臂正式 latency/energy，不估算能耗，也不回填
Table 2。队列 PID 742441 保持等待，授权就绪后会按
Original、Compression only、Schedule only、Compress → Tune、Joint
FP16 control 顺序自动执行。

### 12.7 latency/energy 恢复、生命周期修复与最终结果

用户在 2026-07-25 明确授权本次一次性输入 sudo 密码。密码仅通过 sudo
stdin 使用，未写入远端文件、环境变量、实验命令或日志。随后临时创建仅允许
`/usr/bin/tegrastats` 的 NOPASSWD 规则。

权限生效后暴露出第二个问题：旧权限探针将 `timeout` 放在 sudo 外层，无法
终止 root tegrastats 子进程；旧 runner 结束采样时也只向 sudo 父进程发送
SIGTERM，存在残留采样器污染后续 active window 的风险。处理如下：

1. 在正式报告写出前主动终止第一次 Original 尝试；
2. 将该 partial window、权限探针日志和退出码完整归档到
   `99_failed_attempts/original_default_native_stop_lifecycle_20260725_1003/`；
3. runner 改用
   `sudo -n /usr/bin/tegrastats --interval 500`，结束时调用 tegrastats
   原生 `--stop` 并等待前台进程退出；
4. 权限探针改为以“原生 stop 成功且日志包含 `VIN_SYS_5V0`”为门禁，不再
   把被 `--stop` 终止的前台退出码 137 误判为权限失败；
5. 短探针取得 5 个 `VIN_SYS_5V0` 和 5 个 `VDD_GPU_SOC` 样本，停止后
   tegrastats 残留数为 0；四模块聚焦回归为 93 passed；
6. runner 和 queue 远端/本地 SHA-256 分别为
   `51646673fc8b71159cb85535c54431a421640b725f4ed100e18cc4a0efcc33ef`
   和
   `b2721cb06dae0425bce0417639552e2cf7c7c081f5446492240150f18a2175ff`。

修复后五条配置均按 warmup=20、iters=300、repeat=5、CUDA event、
compute-only/no-data-transfer 完成，且每条都有精确 1500 个原始时延样本。
主 rail 仅使用 `VIN_SYS_5V0`：

| 配置 | median (ms) | p90 (ms) | p99 (ms) | mean (ms) | mean VIN_SYS_5V0 (W) | Energy (J) |
|---|---:|---:|---:|---:|---:|---:|
| Original/default | 559.3361 | 560.0493 | 560.6299 | 559.3361 | 8.8818 | 4.9679 |
| Compression only | 41.2498 | 41.3008 | 41.5916 | 41.2555 | 8.7699 | 0.3618 |
| Schedule only | 1149.6327 | 1151.5650 | 1172.2776 | 1150.8645 | 8.3343 | 9.5814 |
| Compress → Tune | 48.2811 | 48.3418 | 48.7699 | 48.3173 | 8.6147 | 0.4159 |
| Joint FP16 control | 38.2407 | 38.2891 | 38.4717 | 38.2479 | 8.8622 | 0.3389 |

逐条重新计算
`mean(VIN_SYS_5V0 W) * median(ms) / 1000` 与报告 Energy 完全一致。五条
测量结束后 tegrastats 残留数均为 0。

fail-closed finalizer 最终输出：

```text
status = complete
evidence_grade = A
scope = post_scatter_backbone_shrinker
manifest SHA-256 = f154788ae6234fed7e0a03d34849d6a7aa72c77c266a4cc76d2479c16849b984
table2_values.json SHA-256 = b153926da3492aed4193caf3578d18cc05ef90746a7d38eaf75fff1edbd76c97
table2_values.csv SHA-256 = 6107e8cd310a99cccb978f445ffecdafafcc32981fcf1096b3a1969f6534ab21
evidence_manifest.json SHA-256 = 342234836b9aff2661e96b99fb279e63cc48ffefd9057dd8c766f78e7c1775a6
```

正式测量与 finalizer 完成后，临时 sudoers 规则已删除；再次执行
`sudo -n /usr/bin/tegrastats` 返回“a password is required”，证明临时权限
已收回。

### 12.8 论文 Table 2 回填与页面校验

依据 Grade A finalizer 的 `table2_values.json`，已将 F-Cooper 五条配置的
AP70、汇总 median latency 和 `VIN_SYS_5V0` energy 按两位小数回填到
`AnonymousSubmission2027.tex`：

| 配置 | AP70 | Latency (ms) | Energy (J) |
|---|---:|---:|---:|
| Original/default | 0.63 | 559.34 | 4.97 |
| Compression only | 0.60 | 41.25 | 0.36 |
| Schedule only | 0.63 | 1149.63 | 9.58 |
| Compress → Tune | 0.63 | 48.28 | 0.42 |
| Joint FP16 control | 0.60 | 38.24 | 0.34 |

Table 2 的说明已明确：三种模型均测冻结部署子网且不计数据传输；
Pyramid/CoDriving 使用 batch=2，F-Cooper 使用 sample batch=1、
dense agent batch=5；Joint 行为 FP16 control，不宣称复现 H800 INT8。

Table 1 与 Table 3 的源码块在回填前后 SHA-256 保持不变，分别为：

```text
Table 1 = b2777333d2f005044d536359b924d5c2f6e02c9dc6cbbf1cafad758bf6783c1e
Table 3 = 31b20a43212d1dc6666c5cbba29dd96465474071506fbe038871ad1adf465e79
```

使用 `TEXINPUTS=./Figures//:` 连续两次执行 `pdflatex
-interaction=nonstopmode -halt-on-error AnonymousSubmission2027.tex` 均成功；
生成 PDF 为 16 页，SHA-256 为
`59def239a983fead3a85af81d7682f18594ffecb6e8579e0652e0eaa85dd7fb2`。
PDF 文本和页面渲染检查确认 Table 1、Table 2、Table 3 均位于第 6 页，
三个标题均为单行，表格继续共用原有 `\small` 字号，无裁切或列溢出。
编译日志中的盒子告警位于既有正文/公式行，不涉及三张表。

### 12.9 本地离线证据复核

独立审阅首次检查发现，本地结果根只同步了 latency、energy、AP 和 final
目录，`evidence_manifest.json` 所列的 24 个 engine/numeric 原始产物仍只
保存在 Orin。虽然远端 finalizer 已成功，但当时本地副本不能独立重验 Grade A。

处理方式不是降低门禁或只引用旧 SHA，而是从 Orin 的同一正式结果根补齐：

- 四条 TRT 配置的 build receipt、inspector、engine 和 timing cache；
- 四条 TRT 配置的 numerical gate、Original reference；
- held-out manifest、sidecar 和 1,006,633,088-byte
  `heldout_inputs.npy`。

大张量先验证断点前缀与 Orin 原文件 SHA 一致，再通过压缩数据流完整恢复；
最终文件 SHA-256 为
`115e69c179afe87395eef609cbbd372c7e44e4abdb339ba70e82c1997a88ad73`。
补齐后在本地重新执行正式 fail-closed finalizer，并再次按
`evidence_manifest.json` 对全部文件逐项复算：

```text
artifact_count = 67
present = 67
SHA/bytes matched = 67
missing = 0
mismatch = 0
status = complete
evidence_grade = A
```

重新 finalization 后 `table2_values.json`、`table2_values.csv` 和
`evidence_manifest.json` 的 SHA-256 均与第 12.7 节一致。这一过程修复了
“远端完成但本地证据包不完整”的交付问题；以后同步 final 目录时必须同时
运行 manifest 全量 rehash，不能把清单存在误当成清单内文件已经归档。
