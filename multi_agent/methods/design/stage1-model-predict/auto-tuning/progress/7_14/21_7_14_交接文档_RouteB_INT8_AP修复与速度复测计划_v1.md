# 21_7_14 交接文档：RouteB INT8 AP 修复与速度复测计划 v1

**日期**: 2026-07-10  
**状态**: AP 根因已定位，代表宽度的三条量化臂已完成全量 AP 闭环；修复后 latency/energy 尚未闭环  
**适用范围**: CoDriving、H800、TVM RouteB、`backbone.resnet` 进入 TVM 且其余检测流程保持 PyTorch  
**核心判断**: 旧 `AP70=0` 是无有效 scale 的量化实现错误，不是 TVM INT8 的真实精度结论。当前目标不是把 INT8 AP 完全恢复到 FP16，而是在真实 scale-aware 语义下复现 `mixed_topK_flops` 的加速收益，同时保证 AP 非崩溃、可测量、可比较。旧 latency/energy 仍需同口径重测后才能与新 AP 合并。

---

## 1. 首要回答：修复 AP 的同时能否保证推理速度

当前答案是：**不能直接保证，但保留 mixed INT8 加速是本轮第一优化目标，而不是附属指标。**

可以确认的内容：

1. 修复后的 pure INT8、mixed top25、mixed top50 均真实进入 TVM 路径。
2. 三条路径在 `32x64x128` 上均完成 `1789/1789` 帧，fallback 为 0。
3. 编译结果均检测到 `12` 个 `mma_sync`，说明 TensorCore 路径存在。
4. AP70 已从错误的 0 恢复到 `0.3959-0.3979`。
5. 旧 G4-S3/S4 latency 实验已经观察到 `mixed_top25_flops` 在部分 width 上快于 FP16，证明“只量化高计算量 Conv”是值得保留的候选路径。

尚不能确认的内容：

1. 修复后的 scale-aware INT8 是否保持旧 RouteB INT8 latency。
2. pure INT8 或 mixed INT8 是否快于同口径 FP16。
3. 新路径的 energy、显存占用和尾延迟是否优于 FP16。
4. `32x64x128` 的结果能否推广到其余 11 个 width。

原因是 AP 修复不是只替换一个常数，而是改变了实际计算：

```text
旧错误路径:
FP32 value -> 直接 cast INT8 -> INT32 MMA -> scale=1 输出

修复后路径:
FP32 value / calibrated_scale
-> round
-> clamp[-127, 127]
-> INT8
-> INT32 MMA
-> FP32 dequant(input_scale * weight_scale)
-> bias/residual/ReLU
```

其中 `round`、`clamp`、真实 scale 乘除和 materialization 都可能改变 latency。旧速度只能作为 mixed policy 的候选加速信号，不能与新 AP 组合成同一个 gold 样本。下一轮的关键不是继续单独追高 AP，而是验证该加速信号在真实量化语义下是否仍成立。

---

## 2. AP70 全部为 0 的根因

旧实现直接对 FP32 activation/weight 执行 `astype(int8)`，没有校准 scale。该根因可以从两类已落盘证据交叉确认：

1. 旧 TIR 构造代码没有 activation/weight calibration scale，只执行直接 cast；旧三条 full-eval 报告的 AP70 均为 0。
2. 加入真实 train calibration、round/clamp 和 dequant 后，在相同 `32x64x128` checkpoint 上，三条 full-eval 报告的 AP70 同时恢复到 `0.3959-0.3979`。

旧 AP 报告位置：

- `${V2X_ROOT}/results/v2_gold_coldstart_96_20260708/codriving_tvm_routeb_resnet_ap_raw/32x64x128/tvm_routeb_int8_all_final.json`
- `${V2X_ROOT}/results/v2_gold_coldstart_96_20260708/codriving_tvm_routeb_resnet_ap_raw/32x64x128/tvm_routeb_int8_top25_final.json`
- `${V2X_ROOT}/results/v2_gold_coldstart_96_20260708/codriving_tvm_routeb_resnet_ap_raw/32x64x128/tvm_routeb_int8_top50_final.json`

调试过程中还执行过权重归零率和首层输出误差检查，但该次临时诊断没有形成独立落盘报告，因此本交接文档不采用其具体数值作为正式论文证据。若论文需要展示微观根因，应在后续 profiler/numerical audit 中重新执行并保存 JSON。当前可以正式使用的结论是：**旧 AP=0 与无 scale 的错误路径绑定；scale-aware 路径已经通过完整 1789 帧检测评估推翻该结果。**

---

## 3. 已完成的实现修复

### 3.1 scale-aware INT8 TIR

文件：

- `${V2X_ROOT}/scripts/stage2_codriving_whole_engine_tc_v1.py`

修复内容：

- INT8 PrimFunc 强制要求正的 `input_scale` 和 `weight_scale`。
- 使用 symmetric absmax、signed INT8 `[-127,127]`。
- quantize 顺序为 `divide -> round -> clamp -> cast int8`。
- TensorCore 主计算为 `int8 x int8 -> int32`。
- 输出使用 `input_scale * weight_scale` 反量化到 FP32，再执行 bias/residual/ReLU。
- 独立 latency probe 若使用 `scale=1`，会显式标记为 `dummy_scale_1_latency_tensorization_probe_only`，禁止作为 AP 证据。

### 3.2 train calibration 与来源证明

文件：

- `${V2X_ROOT}/scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py`
- `${V2X_ROOT}/scripts/stage2_codriving_int8_provenance.py`

当前有效校准口径：

| 字段 | 值 |
|---|---|
| calibration split | `train` |
| sample count | 精确 16 个 |
| tensor shape | `[16,2,64,256,512]` |
| quantization semantics | `symmetric_absmax_int8_dequant_fp32` |
| qrange | `[-127,127]` |
| train split source | `/data/lxf_data/DAIR-V2X-C/Full_Dataset/cooperative-vehicle-infrastructure/train.json` |
| calibration NPZ SHA256 | `3cdc40a69c0e4961333c903e892139fe1dde6f731cf5017d679f51f028c693e7` |
| summary SHA256 | `5d8c14a0025a49a5dddb92f7c01717fe002b7fab0d0bb3c3393a85d2a6133868` |

门禁会拒绝：

- val/test calibration；
- 非真实 JSON 整数的 sample count；
- 样本数或 tensor shape 不匹配；
- 缺少 NPZ/summary SHA256；
- 缺失任一实际 INT8 PrimFunc signature 的 scale；
- dummy-scale 或量化语义不匹配的旧报告。

### 3.3 AP runner、watcher 和 96 点 importer

文件：

- `${V2X_ROOT}/scripts/stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py`
- `${V2X_ROOT}/scripts/stage2_v2_gold_coldstart96_codriving_trt_ap_watcher.py`
- `${V2X_ROOT}/scripts/stage2_v2_gold_coldstart96.py`

新增约束：

- INT8/mixed AP runner 必须同时接收 calibration NPZ 和 summary。
- watcher 只把成对存在且通过哈希验证的 train calibration 交给 AP runner。
- importer 强制检查 `tag/mode/precision/pipeline_scope`。
- 历史 dummy-scale INT8 报告和缺少 digest 的报告 fail closed，不能进入 96 点训练表。

---

## 4. H800 最终 AP 实验结果

当前只将以下三个 `*_final_full1789.json` 视为最终有效结果。目录中的 `n20`、`traincalib`、`true_traincalib` 和不带 `final` 的文件均为中间诊断结果，不进入最终结论。

宽度：`32x64x128`

| strategy | AP30 | AP50 | AP70 | AP70 vs FP16 | TVM samples | fallback | INT8 signatures |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 0.718878 | 0.634915 | 0.413453 | 100.00% | 1789 | 0 | 0 |
| pure INT8 | 0.666936 | 0.596504 | **0.397915** | **96.24%** | 1789 | 0 | 12 |
| mixed top25 | 0.666802 | 0.595007 | **0.396823** | **95.98%** | 1789 | 0 | 3 |
| mixed top50 | 0.665397 | 0.594205 | **0.395889** | **95.75%** | 1789 | 0 | 6 |

最终结果文件：

- `${V2X_ROOT}/results/routeb_int8_ap_repair_20260710/32x64x128/int8_all_final_full1789.json`
- `${V2X_ROOT}/results/routeb_int8_ap_repair_20260710/32x64x128/top25_final_full1789.json`
- `${V2X_ROOT}/results/routeb_int8_ap_repair_20260710/32x64x128/top50_final_full1789.json`

本轮能支持的结论：

1. 三条量化臂 AP70=0 的旧结论作废。
2. scale-aware INT8 在代表宽度上保留了 FP16 AP70 的 `95.75%-96.24%`。
3. 当前 pure INT8 AP70 略高于 top25/top50，但差距只有约 `0.001-0.002`，不能据此写成跨 width 的固定排序。
4. AP30/AP50 的相对下降比 AP70 更明显，后续仍应研究 calibration sample 数、percentile/entropy scale 和 per-callsite scale。
5. 当前 AP 已经达到“非崩溃”要求；后续不应以恢复到 FP16 AP 为单一停止条件，而应转向速度-AP联合测量。

---

## 5. 与既有速度实验的关系

### 5.1 18 号文档归纳的三个慢因

此前围绕 INT8 为什么慢于 FP16，总结并验证了三个原因：

1. **H1 grouped convolution/shape 问题**：group conv、小 GEMM、通道剪枝后的 tile 不对齐和 padding 会降低 INT8 TensorCore 利用率，并放大 im2col/materialization 成本。
2. **H2 TVM fusion boundary 问题**：conv、cast、dequant/requant、layout transform、add 等阶段被拆开，会产生中间张量落地和额外 kernel launch。
3. **H3 INT8 本征与覆盖范围问题**：INT8 仍有 int32 accumulator、scale、requant、pack 等成本；并非所有 Conv 都有足够计算量抵消这些固定开销。

`mixed_topK_flops` 的作用主要是针对 H3：只把计算量最大的 eligible Conv 送入 INT8，把低收益的小 Conv 留在 FP16。它对 H2 的影响不能预设为“边界更少”：FP16/INT8 交替可能新增 cast/requant 边界，最终收益取决于大 Conv 的计算节省能否覆盖这些边界成本。它不是人为规定的最终规则，而是搜索空间中的一个逻辑策略候选。

### 5.2 已观察到的 mixed INT8 加速信号

G4-S3 的 fp32-boundary 对照中，`mixed_top25_flops` 在 4/8 个 width 上快于 pure FP16：

| width | FP16 ms | mixed top25 ms | top25/FP16 | 约加速幅度 |
|---|---:|---:|---:|---:|
| `16x32x64` | 2.535 | 2.480 | 0.979 | 2.2% |
| `32x32x128` | 4.162 | 4.085 | 0.981 | 1.9% |
| `32x64x128` | 4.358 | 4.270 | 0.980 | 2.0% |
| `48x96x192` | 6.418 | 6.270 | 0.977 | 2.3% |

G4-S4 的 fp16-boundary 对照中，`mixed_top25_flops` 在 2/8 个 width 上快于 pure FP16：

| width | FP16 ms | mixed top25 ms | top25/FP16 | 约加速幅度 |
|---|---:|---:|---:|---:|
| `16x32x64` | 1.793 | 1.729 | 0.965 | 3.6% |
| `32x64x128` | 2.597 | 2.578 | 0.993 | 0.7% |

这些结果证明了“选择高计算量 Conv 可以出现超过 FP16 的点”，因此 mixed policy 必须保留在搜索空间中。但加速幅度目前只有约 `0.7%-3.6%`，而且旧测量使用的是不能产生有效 AP 的旧 INT8 数值路径，所以它们是**候选加速证据**，还不是 scale-aware gold latency。

G4-S4 同时显示：

- pure INT8 在 8/8 CoDriving 点慢于 FP16；
- `median(INT8/FP16)=1.275x`；
- mixed top25 只有 2/8 点快于 FP16；
- FP16 graph boundary 能消除公共 materialization 开销，但没有翻转 INT8/FP16 排序。

因此正确目标不是让 pure INT8 整网强行超过 FP16，而是从 `top10/top25/top50/top75/all` 中找到在给定 width/backend 下真正产生速度收益且 AP 不崩溃的 mixed 点。

### 5.3 G4-S5 对后端路径的补充

G4-S5 的结论仍独立成立：

- FP16-CUTLASS full-engine 8/8 build/run 成功，且 8/8 快于 S4 FP16。
- CUTLASS INT8 最小 conv 可以 build/run。
- 当时的最小探针中，单层 INT8 CUTLASS 仍慢于 FP16，且中间 dtype boundary 会进一步放大劣势。

因此目前不能承诺“AP 修复后 mixed INT8 一定保持旧加速”。更严格的表述是：

> 已修复 INT8 的数值正确性并证明 TensorCore 执行入口成立；此前筛选高计算量 Conv 已出现超过 FP16 的 latency 点。下一步主任务是在相同量化语义、graph boundary、输入和计时方法下复现该加速，同时把 AP 损失作为 Pareto 代价记录，而不是要求 AP 完全恢复。

---

## 6. 当前 96 点数据集状态

在启用新 provenance 门禁后，旧 96 点导入审计结果为：

| 状态 | 行数 |
|---|---:|
| 可继续视为 complete | 60 |
| 失效并等待补测 | 36 |
| 失效原因 | 旧 TVM INT8/mixed 缺少有效 AP70 |

审计文件：

- `${V2X_ROOT}/results/routeb_int8_ap_repair_20260710/import_audit_before_full_rerun/v2_gold_coldstart_96_missing_failure_audit.json`

当前没有把 `32x64x128` 的新 AP 写入正式 96 点 raw 目录。这样处理是有意的：该点的新 latency/energy 尚未重测，提前写回会形成 AP 与性能指标来自不同 pipeline 的伪 gold 样本。

---

## 7. 下一步速度闭环实验

### 7.1 第一阶段：代表宽度 paired measurement

先只测 `32x64x128`。这里不能把 AP 与 latency 写成“同一 runner”：当前 AP runner 是 `TVM backbone.resnet + PyTorch 完整检测评估`，已有 latency runner 是 TVM backbone-only ONNX benchmark。正确要求是使用**同一导出模型、同一编译产物、同一 calibration、同一 graph boundary 和同一 pipeline fingerprint**，由两个同步入口分别测量：

1. AP runner：完成 1789 帧完整检测评估，输出 AP30/AP50/AP70。
2. performance runner：对同一个已编译 TVM backbone module 测量 backbone latency、energy 和 peak memory。

如果论文要宣称“端到端检测推理加速”，还必须增加第三个 full-pipeline latency runner；backbone-only latency 只能支持“backbone 加速”表述，不能替代端到端速度。

四条 paired arms：

```text
fp16
int8_all
mixed_top25_flops
mixed_top50_flops
```

每条臂必须由共享 fingerprint 关联以下产出：

- AP30/AP50/AP70；
- warmup 后 backbone latency p50/p90；
- backbone scope energy per inference；
- peak memory；
- `n_tvm_path/n_fallback_path`；
- `mma_sync` 与实际 INT8 signature 数；
- calibration NPZ/summary SHA256；
- 完整 pipeline fingerprint。

建议额外产出 full-pipeline latency p50/p90，作为端到端结论的唯一速度依据。该指标与 backbone latency 分列记录，不允许混用。

第一阶段停止条件：

1. 四条臂均使用修复后的 scale-aware pipeline。
2. 每条量化臂完成 1789 帧 full evaluation，AP30/AP50/AP70 均为有限非零值，不再发生量化崩溃。
3. 至少一个 mixed arm 的 paired backbone latency 低于 FP16，且三次独立进程重复后的置信区间/离散度足以排除测量噪声；这是“保留加速结论”的核心验收条件。
4. AP 不要求等于 FP16，只需如实记录 retention 和变化趋势，进入 latency-AP-energy Pareto；不能为了追回 AP 而人工固定 mixed 比例。
5. latency/energy 不能复用旧 JSON，必须生成新报告。
6. 若 scale-aware mixed INT8 失去加速，必须通过 profiler 分解 quantize、im2col/materialization、MMA、dequant/epilogue 的耗时，并继续优化 fusion/tensorization；不能回退到无有效 scale 的旧路径来保留速度数字。

### 7.2 第二阶段：其余 11 个 width 补测

第一阶段测量脚本和证据门禁稳定后，再补：

```text
11 widths x 3 INT8/mixed arms = 33 条修复后量化记录
```

当前旧 FP16 报告缺少完整 checkpoint/ONNX 哈希和 measurement fingerprint，无法严格证明与新路径一致。因此本轮默认同时重测 FP16；只有后续补齐并验证完整 fingerprint 后，才允许复用历史 FP16。

第二阶段还需要验证“AP 有趋势”而非只检查单点非零：

- 不同 width/mixed coverage 的 AP 必须存在可测的非零方差，不能再次退化为全零或同一异常常数。
- 同一配置重复评估应得到稳定 AP；只有超过测量波动的差异才用于训练排序模型。
- 记录 `mixed coverage -> AP loss` 和 `mixed coverage -> latency/energy` 的联合变化，让 cost model 从实测回流学习趋势。
- 不预先规定 top25 必须优于 top50，也不设置固定 AP retention 作为搜索规则。

最终重建 96 点表时，每一行 AP、latency、energy 必须来自同一个：

```text
model checkpoint
compiler pipeline
quantization semantics
calibration artifact
graph boundary
measurement version
```

任何字段不一致都不能合并为一个 gold 点。

---

## 8. 对自动搜索框架的当前含义

本轮结果不要求写入“INT8 必须启用”或“INT8 必须禁用”的人工规则。

搜索 genome 仍保持：

```text
([p1,p2,p3], q_mode, mixed_policy_id)
```

但训练数据必须更新为修复后的真实测量，且优化目标是联合 Pareto，不是“AP 恢复优先”：

- AP 模型学习不同 mixed policy 的精度损失；
- latency/energy cost model 学习 quantize、padding、materialization、fusion boundary 和 backend context 的实际代价；
- 搜索器根据冷启动数据和实测回流决定 FP16、pure INT8 或 mixed INT8；
- 不把当前 `32x64x128` 上 pure INT8 的 AP 略优写成固定策略。

AP 的正确角色：

- AP30/AP50/AP70 是需要最小化损失的质量目标，不是必须等于 FP16 的硬约束。
- “非崩溃”只作为数据有效性门禁：完整 1789 帧评估、指标有限且非零、没有 fallback 或错误量化语义。
- 不在搜索器中人工写入固定 AP retention 阈值；由真实 AP、latency、energy 共同形成 Pareto，让模型学习不同 mixed coverage 的代价。
- `mixed_top25_flops` 只是已有加速信号最强的候选之一，不能写死为最终策略；top10/top50/top75/all 仍由冷启动数据和回流竞争。

在速度补测完成前，36 条 TVM INT8/mixed 行应保持 invalid/pending，不允许进入最终 Pareto。

---

## 9. 验证状态

- 本地相关单元测试：`53/53` 通过。
- 本地与 H800 上六个生产脚本均通过 `py_compile`。
- 三份最终报告通过 `report_complete(min_samples=1789)`。
- 三份旧 dummy-scale AP 报告均被新 runner/importer 拒绝。
- 三份最终报告的 calibration source/summary SHA256 完全一致。
- H800 上已无遗留 AP/calibration 进程。

当前最紧急的未完成批次是：**先在 `32x64x128` 上重新测量修复后 scale-aware pipeline 的 backbone latency/energy，并补充 full-pipeline latency；随后完成其余 11 个 width 的 33 条 AP/性能补测，重建 AP 与速度同源的 96 点证据。**

---

## 10. 2026-07-10 速度重测后的自动 lowering 优化约束

### 10.1 重测结果对计划的修正

`32x64x128` 已完成四条臂、每条三次独立进程、每次 200 个 latency 样本的 scale-aware backbone 重测。结果显示：

| strategy | INT8 Conv | AP70 retention | p50 latency | latency/FP16 | energy/FP16 |
|---|---:|---:|---:|---:|---:|
| FP16 | 0/12 | 100.00% | 1.604 ms | 1.000x | 1.000x |
| pure INT8 | 12/12 | 96.24% | 8.976 ms | 5.595x | 3.287x |
| mixed top25 | 3/12 | 95.98% | 3.954 ms | 2.465x | 2.113x |
| mixed top50 | 6/12 | 95.75% | 6.907 ms | 4.305x | 3.023x |

AP 非崩溃目标已经达到，但速度门禁失败。延迟随 INT8 Conv 覆盖数显著增加，结合当前 TIR 实现可定位为：activation 和 constant weight 的 `divide -> round -> clamp -> cast` 都在运行时 rewritten Conv 内执行，权重没有编译期预量化，activation quantize 也没有在边界处复用/融合。

该批结果只标记为 `conditional_for_runtime_overhead_diagnosis`，不进入 final frontier。另一个证据限制是：旧 AP 报告尚未记录 ONNX 内容 SHA；新 runner 已修复该字段，后续 gold AP/performance 必须按 ONNX SHA 绑定后重跑。

结果位置：

- `${V2X_ROOT}/results/routeb_int8_scaleaware_perf_20260710/routeb_int8_scaleaware_perf_summary_32x64x128.json`
- `${V2X_ROOT}/results/routeb_int8_scaleaware_perf_20260710/routeb_int8_scaleaware_perf_summary_32x64x128.csv`

### 10.2 优化必须保持的论文叙事

下一步不以“手写一个更快的 CoDriving INT8 Conv”为目标，而是修复逻辑 mixed strategy 到可执行图之间的通用 compiler lowering：

```text
搜索器提出逻辑策略
([p1,p2,p3], q_mode, mixed_policy_id)

        -> compiler profile 自动 realization

编译期权重量化
+ 根据 mixed mask 自动生成 activation Q/DQ boundary
+ quantize/dequantize/epilogue fusion
+ 暴露标准 INT8 matmul
+ TVM tensorize/MetaSchedule

        -> AP/latency/energy 实测回流

cost model 自动学习 FP16、pure INT8、mixed INT8 的真实 Pareto
```

必须遵守以下边界：

1. **不能手工指定具体 Conv 名单。** 搜索变量只能是 `top10/top25/top50/top75/all` 等逻辑 mixed policy；具体 Conv 集合由通用 FLOPs 排序、图结构分析和 capability scan 自动产生。
2. **不能回退到 per-model 专用 MMA kernel。** 下一步应实现通用 per-PrimFunc/per-block rewrite：识别 constant weight、编译期量化、自动插入 activation boundary、暴露标准 INT8 matmul，再交给 TVM tensorize/tune。
3. **编译期权重量化不是外部先验。** 它属于常量折叠和 engine build realization，不告诉搜索器选择哪种 mixed policy，也不改变搜索 genome。
4. **不能写死 top25。** `top25_flops` 只是已有加速信号最明确的首轮验证点；最终仍由 FP16、top10、top25、top50、top75、all 通过冷启动和实测回流竞争。
5. **不能用错误数值路径保留速度。** 无有效 scale 的 dummy cast 只能作为 diagnosis，不能进入 AP、latency、energy gold 数据。
6. **backend 仍是测量上下文。** compiler profile/capability scan 决定当前环境能实现哪些策略，但 backend 名称不进入搜索 genome。

### 10.3 自动 lowering 三阶段消融

后续实现和论文实验按三个阶段推进：

| stage | realization | 验证问题 |
|---|---|---|
| L0 | runtime weight + activation quantize | 当前失败基线，量化覆盖率越高、运行时开销越大 |
| L1 | compile-time INT8 weight + runtime activation quantize | 常量权重量化 hoist 能消除多少开销 |
| L2 | compile-time weight + automatic/fused activation boundary | 完整自动 lowering 能否恢复 mixed INT8 相对 FP16 的加速 |

首轮仍使用 `32x64x128` 的 FP16/top25 paired measurement。L1/L2 的进入条件：

- 具体量化 Conv 集合由 `mixed_policy_id` 自动生成；
- 同一 pass 能处理多个 width，不读取 CoDriving 专用 Conv 名称；
- AP full evaluation 非零且稳定；
- performance/AP 报告按 ONNX、calibration、compiler pipeline SHA/fingerprint 绑定；
- 至少三次独立进程重复，速度差异必须超过测量离散度；
- 若 top25 通过，再扩展 top10/top50/top75/all，不能提前写入固定规则。

### 10.4 当前优化入口

优先审计并复用：

- `${V2X_ROOT}/scripts/stage2_route_b_int8_auto_decomp.py`

该入口已经包含 `uint8 activation + int8 prequantized weight + int32 accumulator` 的自动 BlockBuilder/PrimFunc 设计。下一步应把其中通用的 initializer prequantization、per-block decomposition 和 TensorCore exposure 接入 CoDriving mixed runner，而不是继续在 `stage2_codriving_whole_engine_tc_v1.py` 中增加模型专用分支。

第一优化停止点不是完成所有 width，而是：

> 在不写死 Conv 名称的条件下，让 `32x64x128 + mixed_top25_flops` 完成 L1 build/run，并用同口径 measurement 判断编译期权重量化是否显著降低当前 `3.954 ms` 延迟；若 L1 有效，再继续 L2 activation boundary fusion。

### 10.5 L1 通用 auto-decomp 首轮 smoke 结果

已将现有 `stage2_route_b_int8_auto_decomp.py` 直接运行在同一个 CoDriving `32x64x128` ONNX 上，作为“运行前预量化权重”机制验证：

| item | result |
|---|---:|
| ONNX Conv 数 | 27 |
| 自动构造 block 数 | 89 |
| Tensorized Conv | 25/27 |
| schedule failure | 0 |
| build/run | success |
| correctness vs 同量化 native direct reference | exact |
| p50 latency | 2.874 ms |
| 当前 L0 runtime-quant pure INT8 | 8.976 ms |
| 延迟下降 | 67.98% |

结果位置：

- `${V2X_ROOT}/results/codriving_mixed_auto_lowering_20260710/codriving_32x64x128_l1_prequant_smoke/route_b_int8_auto_decomp_result.json`

该结果只证明 L1 机制有效，不能进入 AP/Pareto，原因是：

1. 当前 auto-decomp 是全图 uint8 输入/输出，不是 FP16 主数据流上的 mixed top25。
2. 输入是 synthetic uint8，不是 AP runner 的真实 calibrated activation。
3. 当前 tensor scale 传播出现极小 accumulator scale，bias quantization 有 overflow/invalid cast 警告。
4. `2.874 ms` 与 `1.604 ms` FP16 的 boundary/scope 仍不完全同口径，不能据此宣称 pure INT8 已超过或接近 FP16。

下一实现不继续扩展该全图 uint8 builder，而是新建通用 mixed builder：

```text
FP32 graph input -> FP16 主数据流

unselected Conv:
  FP16 Conv -> FP16 output

selected Conv:
  FP16 activation
  -> 独立一次 symmetric signed-INT8 quantize
  -> INT8 MMA / INT32 accumulator + 预量化 INT8 weight
  -> dequant + FP16 bias
  -> FP16 output

FP16 Add/Relu -> FP32 graph outputs
```

其中 selected Conv 必须来自稳定的 ONNX Conv FLOPs 排序和落盘 selection manifest；FLOPs 相同时用 node ID 做稳定 tie-break，不能依赖遍历偶然顺序或模型专用名称。

### 10.6 FP16 BlockBuilder、rank-1 与 top25 fused build 结果

已完成下一实践节点的三个 build/run 对照：

| realization | selected INT8 Conv | p50 latency | vs 同构 FP16 | 数值状态 |
|---|---:|---:|---:|---|
| FP16 BlockBuilder baseline | 0/27 | 1.808 ms | 1.000x | mean abs err `0.00094/0.00491/0.00570` |
| rank-1, split accumulator/dequant | 1/27 | 1.960 ms | 1.084x | 非崩溃 |
| rank-1, fused dequant epilogue | 1/27 | 1.949 ms | 1.078x | 与 split 版本一致 |
| top25 fused | 7/27 | 2.031 ms | 1.123x | mean abs err `0.165/0.625/0.255`，不合格 |

FP16 baseline 的 27/27 Conv 均进入 FP16 `mma_sync`。rank-1 和 top25 的 selected Conv 也全部进入 INT8 `mma_sync`，说明 build/tensorization 已闭合，当前失败来自 realization 成本和 calibration，而不是“没有调用 TensorCore”。

关键发现：

1. rank-1 的输入为 `[2,64,256,512]`，独立 activation quantize 需要扫描约 16M 元素；即使 fused epilogue 消除约 32 MiB padded INT32 accumulator 的跨 PrimFunc 物化，仍不足以快于 FP16。
2. top25 的 7 个 Conv 全部 tensorize，但累计 quantize boundary 成本使其比同构 FP16 慢 12.3%。因此 FLOPs 最大不等价于净收益最大；需要让 cost model 学习 `compute saving - boundary/materialization cost`，不能人工写入排除规则。
3. top25 数值误差明显增大，原因是当前借用了 FuseTIR signature 聚合 scale；该 scale 不能代表 raw ONNX 27 个 Conv 的逐节点中间 activation。继续扩展 topK 会产生无效 AP。
4. 下一节点必须先采集 raw-ONNX per-node activation calibration，并在 selection manifest 中按 ONNX node ID 绑定 scale。相邻 selected Conv 应保留/复用 INT8 activation，避免每层重复 quantize/dequantize。

结果位置：

- `${V2X_ROOT}/results/codriving_mixed_auto_lowering_20260710/fp16_blockbuilder_baseline_32x64x128.json`
- `${V2X_ROOT}/results/codriving_mixed_auto_lowering_20260710/mixed_rank1_build_32x64x128.json`
- `${V2X_ROOT}/results/codriving_mixed_auto_lowering_20260710/mixed_rank1_fused_build_32x64x128.json`
- `${V2X_ROOT}/results/codriving_mixed_auto_lowering_20260710/mixed_top25_fused_build_32x64x128.json`

当前停止结论：不把该 top25 结果进入 AP 或 Pareto，也不继续扩到 top50/top75。下一实践节点转为 `per-node calibration + INT8 region boundary reuse`，完成后再重新执行 rank-1/top25 paired measurement。

---

## 11. 与 18 号文档三个慢因假设的最终对应解释

### 11.1 18 号文档原始假设

`18*_7_14_交接文档_INT8慢因验证_后三组对照实验设计_v1.md` 将 INT8 慢于 FP16 的原因归纳为：

| hypothesis | 原始问题 |
|---|---|
| H1 grouped convolution/shape | group conv、小 GEMM、padding 和 tile 不对齐是否使 INT8 难以利用 TensorCore |
| H2 TVM fusion boundary | quant/dequant/requant/layout/add 等边界是否产生额外 kernel 和中间张量物化 |
| H3 INT8 intrinsic overhead | int32 accumulator、scale、pack、requant 等固定成本是否超过 INT8 计算收益 |

当前实验表明三个假设不是互斥关系，而是按层次共同作用：

```text
network/operator shape (H1)
        x
backend realization/fusion (H2)
        x
INT8 fixed and shape-dependent cost (H3)
        -> final INT8/FP16 latency
```

### 11.2 H1 的结论：成立但不是当前主因

已有证据：

1. Pyramid 的 group conv、剪枝后小通道和 tile padding 会放大 INT8 的低利用率与 materialization。
2. 但换成以普通 `groups=1` Conv 为主的 CoDriving 后，当前 TVM scale-aware mixed INT8 仍慢于 FP16。
3. CoDriving rank-1 Conv 的 `Cout=32`，当前 INT8 tensorization 为满足 N 轴 tile 将 logical channel 扩展到 128；若 accumulator 跨 PrimFunc 物化，会产生明显放大。

因此 H1 是网络/shape 相关的放大因素，但不能单独解释 TRT 与 TVM 的差距，也不能解释 CoDriving 上的当前失败。

论文可用表述：

> Irregular grouped or narrow convolutions amplify INT8 padding and utilization losses, but ordinary convolutions alone do not guarantee speedup; backend realization remains decisive.

### 11.3 H2 的结论：当前最主要的系统差异

项目实测支持：

1. TRT INT8 在对应对照中可以快于 FP16，而 TVM RouteB pure INT8 在 Pyramid/CoDriving 上长期未形成同等级收益。
2. G4-S4 证明 graph boundary/materialization 是显著公共开销；它不是 INT8 恒弱的唯一原因，但会放大 INT8 路线成本。
3. 当前 TVM L0 在每个 selected Conv 内重复执行 activation/weight quantization，top25 为 `3.954 ms`，FP16 为 `1.604 ms`。
4. 将权重量化移到运行前后，pure INT8 auto-decomp 从 `8.976 ms` 降到 `2.874 ms`，下降约 68%，直接证明 build-time weight preparation 是关键。
5. rank-1 split accumulator/dequant 为 `1.960 ms`；fused epilogue 为 `1.949 ms`。融合有效但幅度有限，因为 activation quantize 仍是独立边界。

对 TRT 的机制解释需要区分证据等级：

- **项目实验证实**：TRT 路线能取得 INT8 加速；当前 TVM 路线存在明显 boundary/materialization 成本。
- **编译机制解释**：TRT 通常在 engine build 阶段完成 constant weight quantization/prepack、全图 tactic/layout 选择以及 Q/DQ/epilogue fusion，使多个连续 INT8 算子共享区域边界。该解释符合当前消融结果，但论文若需要声称某个 TRT engine 的具体 fusion，应补充 engine inspector/profile，而不能仅凭 latency 推断。

两条路线的核心差异可写为：

```text
当前 TVM per-Conv realization:
Q -> Conv -> DQ -> Q -> Conv -> DQ

目标/TRT-style region realization:
Q -> Conv -> Conv -> Conv -> DQ
```

因此 H2 当前是最主要的系统差异，但必须和 H3 一起解释，不能写成“只要融合就一定加速”。

### 11.4 H3 的结论：成立，决定哪些 Conv 值得进入 INT8

已有证据：

1. G4-S5 最小探针中，单层 CUTLASS INT8 Conv 本身仍慢于 FP16，说明 INT8 不天然更快。
2. 当前 rank-1 fused INT8 已检测到 `mma_sync`，但仍比同构 FP16 慢 7.8%。失败不是“没有 TensorCore”，而是 activation quantize、padding 和边界成本超过计算节省。
3. top25 的 7 个 Conv 全部进入 INT8 `mma_sync`，整体仍比同构 FP16 慢 12.3%。
4. 延迟不能只由 FLOPs 解释，还取决于：

```text
input/output bytes
activation quantize bytes
Cout/Cin tile padding
INT32 accumulator expansion
Q/DQ boundary count
selected Conv 是否形成连续 region
epilogue fusion
```

因此 `topK_flops` 应保留为搜索候选，但不能被写成固定最优规则。后续 cost model 应学习：

```text
predicted net gain
= estimated INT8 compute saving
- quantize/dequantize boundary cost
- padding/materialization cost
- fusion break cost
```

这些是从 capability scan 和实测回流得到的特征，不是人工禁用某些层的规则。

### 11.5 为什么 TRT 可以加速而当前 TVM 不能

TRT 并不是没有数值转换，而是更可能将转换成本在 engine 范围内摊薄：

| component | 当前 TVM mixed builder | TRT-style engine realization |
|---|---|---|
| constant weight | 已开始运行前预量化；早期 L0 每次推理量化 | engine build 阶段量化/prepack |
| activation Q/DQ | 每个 selected Conv 独立执行 | 在连续 INT8 region 边界执行并传播 |
| INT32 accumulator | split 版本可物化为大中间张量 | 通常保留在 kernel 内并由 epilogue 消费 |
| bias/ReLU/scale | 部分独立 PrimFunc/kernel | 尽可能进入 fused epilogue |
| layout/tactic | per-PrimFunc 通用 tensorization | engine 级 shape/tactic/layout 选择 |
| mixed decision | 当前按逻辑 policy 构图 | 由 engine capability 和精度约束实现 |

因此下一步不是放弃 TVM，也不是回退到 dummy cast，而是让 TVM 的自动 realization 接近 engine 级 lowering：编译前权重准备、per-node scale、INT8 region、边界复用和融合调度。

---

## 12. 下一步实验与实现计划

### 12.1 P0：raw-ONNX per-node calibration

目标：取代当前错误的 shape-signature scale 聚合。

实施：

1. 在 FP16 BlockBuilder/原始 ONNX 27 个 Conv 上注册逐节点 activation observer。
2. 使用固定的 16 个 train calibration samples。
3. 每个 Conv 按稳定 ONNX node ID 记录：

```text
node_id
node_index
input_name / weight_name / output_name
input absmax/scale
weight absmax/scale
observed tensor shape
sample count
ONNX SHA256
calibration NPZ/summary SHA256
```

4. 输出 schema：`codriving_raw_onnx_per_node_calibration_v1`。
5. mixed builder 默认只接受 `ONNX SHA + node_id + tensor names` 完全匹配的记录；legacy shape scale 只能显式用于 diagnosis。

P0 停止条件：27 个 Conv 均有唯一、有限、正数 scale；同形状不同 node 不再共享记录；重复校准结果稳定。

### 12.2 P1：自动 INT8 region formation

目标：避免每个 selected Conv 都执行 `FP16 -> INT8 -> FP16`。

实施：

1. 搜索器仍只输出 `mixed_policy_id`，不输出具体层名单。
2. compiler pass 根据 selection manifest 和 ONNX dataflow 自动合并连续 selected Conv。
3. 每个 region：

```text
region entry: quantize once
selected Conv chain: retain INT8 activation
region internal Add/Relu: 在 scale 可对齐时保持 INT8，否则自动切断 region
region exit: dequant once to FP16
```

4. selection manifest 同时记录 `region_id`、entry/exit、Q/DQ 数量、fusion break 原因。

P1 停止条件：不能出现模型专用 Conv 名单；相邻 selected Conv 的 Q/DQ 数量低于 per-Conv realization；所有 region 可 build/run。

### 12.3 P2：代表点逐级消融

固定 `32x64x128`，按顺序测量：

| experiment | purpose |
|---|---|
| FP16 BlockBuilder | 同构 baseline，当前约 1.808 ms |
| rank-1 + per-node scale | 验证 scale 正确性与单 Conv 净收益 |
| rank-1 + fused epilogue | 验证 accumulator/materialization |
| 自动形成的最小连续 INT8 region | 验证 boundary reuse |
| top25 region realization | 验证完整 logical policy |

每点至少记录：三次独立进程 latency p50/p90、energy、ORT numerical error、MMA 计数、Q/DQ count、region count、padding/materialization bytes 和 pipeline fingerprint。

P2 速度门禁：至少一个 mixed/region 点显著快于同构 FP16，差异大于独立进程离散度。若没有点通过，不进入 AP full evaluation，而是继续 profiler/region 修复。

### 12.4 P3：AP 与三目标闭环

只有 P2 速度门禁通过的 realization 才进入 1789 帧 AP：

- AP30/AP50/AP70 必须有限、非零、稳定；不要求完全恢复 FP16。
- AP、latency、energy 必须共享 ONNX/calibration/compiler pipeline fingerprint。
- 不设置固定 AP retention 作为搜索规则，真实结果进入 latency-AP-energy Pareto。
- 通过代表点后再扩展其余 width；当前不启动 11 width/33 条批量补测。

### 12.5 对自动搜索框架的回流

新测量表增加以下 derived features：

```text
selected_conv_count
int8_region_count
quantize_boundary_count
dequantize_boundary_count
selected_macs
selected_input_output_bytes
padding_ratio
accumulator_materialization_bytes
mma_sync_count
fusion_break_count
```

这些特征进入 latency/energy cost model；搜索 genome 仍保持：

```text
([p1,p2,p3], q_mode, mixed_policy_id)
```

不把 TRT/TVM 名称、具体 Conv ID、手工速度规则或 evidence trust 放入 genome。最终由冷启动数据和真实回流学习在当前 compiler profile 下选择 FP16、pure INT8 或 mixed INT8。

---

## 13. P0--P3 实现反思与下一轮双问题归因计划

### 13.1 上一轮 P0--P3 的实际完成度

上一轮计划不能整体写成“已完成”。当前准确状态如下：

| phase | 原目标 | 实际结果 | 当前判定 |
|---|---|---|---|
| P0 | raw-ONNX per-node calibration | 27 个 Conv 均生成按 `ONNX SHA + node_id + tensor names` 绑定的 train-16 scale；两次清单字节级一致；27 个节点得到 24 个不同 input scale | 已达到停止条件 |
| P1 | 自动 INT8 region formation | top25 的 7 个 selected Conv 由通用 dataflow 自动形成 4 个 region，Q/DQ 从 7/7 降为 4/4；最小 2-Conv region 的 Q/DQ 从 2/2 降为 1/1；均可 build/run | 已达到机制停止条件 |
| P2 | 代表点逐级消融并通过速度门 | 已完成首轮 FP16、rank-1、rank-2 per-Conv、最小 region、top25 per-Conv 和 top25 region 对照；尚未完成三独立进程、p90、energy 和 profiler 全项；所有已测 mixed/region 点均未快于同构 FP16 | 未通过速度门，不能写成完成 |
| P3 | 1789 帧 AP 与三目标闭环 | 因 P2 未通过速度门，且 rank-2/top25 未通过 numerical gate，未启动 full AP | 按门禁未启动，处理正确 |

关键实测汇总：

| realization | Q/DQ | p50 latency | numerical status | 相对 FP16 |
|---|---:|---:|---|---:|
| FP16 BlockBuilder | 0/0 | 1.8083 ms | pass | baseline |
| rank-1 per-node INT8 | 1/1 | 1.9490 ms | pass | 慢 7.78% |
| rank-2 per-Conv INT8 | 2/2 | 1.9577 ms | fail | 慢 8.26% |
| rank-2 automatic region | 1/1 | 1.9549 ms | fail | 慢 8.11% |
| top25 per-Conv INT8 | 7/7 | 2.0397 ms | fail | 慢 12.80% |
| top25 automatic region | 4/4 | 2.0282 ms | fail | 慢 12.16% |

上述结果支持三个有限结论：

1. per-node calibration 修复了旧 shape-signature 记录错误，但没有单独解决多层量化误差。
2. 自动 region formation 确实减少 Q/DQ；然而 rank-2 和 top25 的延迟只分别改善约 0.14% 和 0.57%，说明 Q/DQ boundary reuse 不是当前唯一或主要的性能决定因素。
3. rank-2 per-Conv 与 rank-2 region 的三路输出 mean error 几乎完全一致，说明当前误差突增不是 region 内 requant 新引入的，而是在第二个 selected Conv 进入 INT8 后已经出现。

因此，上一轮最重要的反思是：

> “per-node scale 正确”和“region 可以自动形成”只证明了数据绑定与 lowering 机制闭合，不能推出 INT8 已经具有速度或精度收益。下一轮必须分别定位计算路径的性能反转点和第二个 Conv 的数值误差来源。

### 13.2 当前需要分别回答的两个问题

问题 A：为什么当前 TVM 自动 INT8 仍慢于同构 FP16？

候选原因必须通过逐级剥离区分：

```text
INT8 MMA core 本身未形成优势
tile/padding 降低有效计算利用率
im2col/x_col/w_mat materialization
INT32 accumulator 与 requant epilogue
activation quantize 与 kernel launch
full-engine fusion boundary
```

问题 B：为什么加入第二个 selected Conv 后 numerical error 突增？

候选原因必须通过数值 oracle 和正交消融区分：

```text
TVM INT8 lowering/tensorization 数学错误
activation absmax 被 outlier 主导
per-tensor weight scale 造成通道级量化失衡
raw-FP32 calibration 与量化后 region 分布不匹配
该 Conv 对 INT8 本身敏感
```

两条问题链共享同一 `32x64x128` ONNX、同一 16 个 train calibration samples、同一输入和同一 pipeline fingerprint。当前不扩 width，不跑 1789 帧 AP，也不把任何诊断结论写成搜索器硬规则。

### 13.3 第一组：INT8 数学正确性 oracle

在执行 8 点精度矩阵之前，先固定 FLOPs 排名第一和第二的两个 Conv，构建与 TVM 完全同输入、同权重、同 scale 的独立 NumPy/CPU oracle：

```text
quantized input/weight
-> exact int32 convolution/accumulation
-> bias
-> requant/dequant
```

必须逐张量比较：

| comparison | purpose |
|---|---|
| quantized input: oracle vs TVM runtime input | 检查 rounding、clip 和 signed INT8 语义 |
| int32 accumulator: oracle vs TVM | 检查 im2col、group/channel mapping、padding 与 tensorization |
| requant output: oracle vs TVM | 检查 scale 传播、bias 和 clip |
| final three outputs: oracle-lowered graph vs TVM | 检查误差是否来自更后续传播 |

停止判定：

- 任一中间张量不一致：先修 lowering，禁止开展 scale 策略比较。
- oracle 与 TVM 一致：说明实现数学闭合，再进入 8 点精度矩阵。

### 13.4 第二组：8 点精度归因矩阵

诊断对象仍为 selection manifest 自动排序得到的前两个 Conv；层选择只用于固定实验锚点，不写成模型专用规则。三个二值轴为：

| axis | level 0 | level 1 |
|---|---|---|
| activation scale | symmetric absmax | fixed train-16 percentile `p99.99` |
| weight scale | per-tensor symmetric | per-output-channel symmetric |
| calibration source | raw-FP32 ONNX activation | region-aware activation，即上游 Conv 已量化后的真实输入分布 |

形成 `2 x 2 x 2 = 8` 个点。所有点共享 signed INT8、相同 rounding/clip 语义、相同样本和相同 TVM schedule，只改变上述三个因素。

每点至少记录：

```text
input saturation ratio
activation quantization MSE / cosine similarity
per-channel weight quantization error
second-Conv accumulator error
second-Conv requant output error
three backbone outputs max/mean error
numerical gate
latency p50
calibration and compiler fingerprint
```

归因规则：

| observation | supported explanation |
|---|---|
| per-output-channel 显著恢复，其余轴影响小 | per-tensor weight scale 是主因 |
| region-aware 显著恢复 | raw-FP32 calibration 存在 distribution shift |
| p99.99 显著恢复且 saturation 可控 | activation outlier/absmax 是主因 |
| 组合方案才恢复 | 多因素耦合，不允许归结为单一原因 |
| 8 点均失败且 oracle 正确 | 第二个 Conv 在当前 PTQ 约束下高度敏感，应由真实 AP/代价模型学习其收益，不人工禁用 |

精度阶段停止条件：至少一个点使三路 mean error 全部 `<= 0.05`；若没有，则输出有证据的失败归因和层敏感性结论，不启动 AP。

### 13.5 第三组：S0--S5 性能逐级剥离

固定同一 rank-2 逻辑工作负载，逐步恢复 INT8 实现成本：

| stage | measured realization | question |
|---|---|---|
| S0 | FP16 TensorCore Conv/region | 同构基线 |
| S1 | INT8 MMA core，输入已量化且已完成等价展开/预处理 | INT8 计算核心本身是否快于 FP16 |
| S2 | S1 + im2col/x_col/w_mat 形成 | materialization 和 layout 成本 |
| S3 | S2 + fused bias/requant epilogue | INT32 与 requant 成本 |
| S4 | S3 + activation quantize | Q 边界和额外 kernel 成本 |
| S5 | 自动 2-Conv INT8 region full-engine | 图级边界、launch 与融合后的最终成本 |

每个 stage 使用 3 个独立进程，固定 warmup/number/repeat；正式结果记录：

```text
latency p50/p90 and process dispersion
energy per inference
kernel count and launch duration
MMA/tensorization count
TensorCore utilization
DRAM throughput and bytes
padding ratio
im2col/materialization bytes
Q/DQ/requant time
pipeline fingerprint
```

必要时使用 Nsight Systems/Compute 采集一次 profiler 证据，但 profiler run 与正式 latency run 分离，不能用 profiler 下的时间写主表。

性能归因规则：

| observation | conclusion |
|---|---|
| S1 不快于 S0 | 当前 INT8 core schedule、tile/padding 或利用率是主因 |
| S1 快，S2 反转 | im2col/materialization 是主因 |
| S2 快，S3/S4 反转 | requant 或 activation quantize 是主因 |
| S4 快，S5 反转 | full-engine fusion/launch boundary 是主因 |
| S5 仍快于 S0 | INT8 路径具备进入 numerical/AP 门禁的性能基础 |

TRT 只作为机制上界对照：对同一 shape/precision 记录 engine inspector、layer profile 和 Q/DQ/fusion 信息，用于判断 TVM 与成熟 engine 的差距发生在哪一级；不能仅凭 TRT 总延迟反推其具体融合行为。

### 13.6 P2/P3 的新门禁

下一轮不再把“可 build/run”视为 P2 成功。进入 AP 必须同时满足：

1. oracle 证明 TVM 数学实现正确；
2. 至少一种自动 realization 通过 `mean error <= 0.05` numerical gate；
3. S0--S5 中定位到明确性能反转级别；
4. 最终 mixed/region realization 比同构 FP16 快，且差异大于三独立进程离散度；
5. latency、energy、numerical error 共享同一 pipeline fingerprint。

满足以上条件后，只选择最多两个通过门禁的 realization 运行 1789 帧 AP。AP30/AP50/AP70 只要求有限、非零且可重复，不预设固定 AP retention 规则；真实 AP、latency、energy 共同进入 Pareto。

### 13.7 对自动搜索叙事的约束

本轮诊断轴不立即全部加入 genome。处理原则为：

1. capability scan 自动枚举后端实际支持的 scale 粒度、region 和 fusion realization，并通过微基准记录能力与代价。
2. 若某种 realization 在全部诊断点上无额外代价且严格占优，可作为该 compiler profile 的默认 lowering 实现。
3. 若 realization 的收益随 width、shape 或 mixed policy 变化，再将其提升为搜索变量或条件特征。
4. 层敏感性、padding、saturation、materialization 和 fusion break 只作为冷启动/实测回流特征，不写入人工 `if-then` 禁用规则。
5. 主 genome 暂保持 `([p1,p2,p3], q_mode, mixed_policy_id)`；是否扩展 realization 轴由本轮证据决定。

### 13.8 执行顺序与本轮停止条件

严格顺序如下：

```text
E0: NumPy/CPU INT8 oracle
 -> E1: 8-point accuracy attribution
 -> E2: S0--S5 performance decomposition
 -> E3: at most two gated full-AP evaluations
 -> E4: measured evidence backflow
```

本轮在以下任一条件满足时停止并汇报，不自动扩展 width：

- oracle 发现 lowering 错误；
- 8 点均未通过 numerical gate；
- S1 已证明 INT8 core 不具备优势且需要新 schedule；
- S5 未显著快于 FP16；
- 同时通过速度和精度门，完成最多两个 AP 点并写入真实 Pareto 数据。

该顺序确保下一轮首先得到可证伪的原因结论，而不是继续用更多 full-engine 点重复“INT8 慢且精度受损”的现象。
