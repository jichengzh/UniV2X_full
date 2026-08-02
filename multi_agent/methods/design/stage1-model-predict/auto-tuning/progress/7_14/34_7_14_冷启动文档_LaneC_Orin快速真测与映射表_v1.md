# 34_7_14 冷启动文档：Lane C Orin 快速真测与映射表 v1

> 日期：2026-07-23  
> 总计划：[32 号三线并行计划](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md)  
> 并行窗口：[Lane B CPU/TVM 冷启动文档](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/33_7_14_冷启动文档_LaneB_4090服务器CPU_TVM固定点回放_v1.md)  
> 本窗口只负责 Orin 快速锚点和 preliminary mapping，不修改 H800 Table 1，不运行 F-Cooper 搜索。

## 1. 单一目标

在 Orin 上尽快获得两个 Pyramid 真实部署锚点，并利用真实锚点和既有分阶段数据生成带证据等级的边缘性能映射表，判断是否值得继续 full-chain 和 Orin few-shot 搜索。

快速表是内部决策与排版占位产物，不把估算值伪装成论文实测值。

## 2. 已知设备与历史证据

历史记录中的设备入口为：

```text
ssh ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

连接前先实查，不假设历史环境仍成立。需要记录：

- Jetson 型号和内存版本；
- JetPack、TensorRT、CUDA、cuDNN；
- `nvpmodel` 当前档位与可用档位；
- GPU/CPU/EMC 频率；
- 磁盘、温度、空闲进程和可用空间。

可参考但不能直接升级为本轮 Gold 的历史文件：

- `${V2X_ROOT}/results/E7_orin_e2e_baseline_vs_best.csv`；
- `${V2X_ROOT}/results/E8_orin_e2e_fullchain.csv`；
- `${V2X_ROOT}/results/E6_orin_p03_lat_energy.csv`；
- `${V2X_ROOT}/results/orin_e2e_baseline_vs_best_2026-06-05.md`。

历史关键边界：E7 body 行有真实测量，E8 只有 FP32 eager full-chain 行为真实单机全链；`*_synth` 行不得当作本轮 full-chain 实测。历史 Orin Nano CSV 为估算，不是真实 Nano 测量。

## 3. 两个固定真测锚点

### 3.1 Anchor M0：Pyramid original

```text
model = pyramid
structure = (64,128,256)
q_mode = fp32
role = original/default
```

### 3.2 Anchor M1：Pyramid H800 winner

```text
model = pyramid
structure = (16,32,64)
q_mode = int8
role = H800-GEAR winner zero-shot replay
```

H800 engine 不可复制到 Orin。必须使用同一 checkpoint/ONNX/genome 在 Orin 本地重新 build，保存 engine SHA、calibration cache SHA、TensorRT version 和构建日志。

如果旧 `base_fp32_orin.engine` 或 `p75_int8_orin.engine` 与当前 checkpoint/ONNX SHA 不一致，它们只能标为 `H`，不能标为本轮 `M`。

## 4. 快速 scope

第一停止点只测与当前优化区域一致的 body/dense scope：

- post-scatter/body 输入；
- backbone、neck/shrinker 和 heads；
- batch/agent 数与 H800 对照明确记录；
- 使用 `trtexec --noDataTransfers` 时必须把结果标为 engine-compute，不得写成 full-chain wall-clock。

full-chain、AP 和 K=4 Orin calibration 均为结果有利后的第二阶段，不阻塞 preliminary mapping 表。

## 5. 环境与测量合同

### 5.1 启动前

- 检查无其他 GPU 推理进程；
- 固定一个可复现 power mode，优先沿用历史 MODE_30W；
- 记录是否执行 `jetson_clocks`，同一表内必须一致；
- tegrastats 预热到稳定温度后开始；
- 记录 RAM、swap、GR3D、CPU、EMC 和温度。

### 5.2 数值

- FP32/FP16 先与服务器参考输出做 sanity；
- INT8 保存 Orin engine 的真实输出，不能复用 H800/4090 AP；
- 快速阶段允许 AP 状态为 `pending`，但不能填 AP 数值；
- 数值失败必须标为 feasibility failure，不允许回退 FP16 后仍写 INT8。

### 5.3 性能与功耗

- warmup 至少 200 次；
- 每个锚点至少 3 次独立进程重复；
- 每次持续时间至少 10 秒或 500 次 inference；
- 报告 median、p90、p99、mean、throughput 和 CV；
- tegrastats 原始日志与 timing 对齐，报告平均功率和 energy/frame；
- latency、功率和 energy 使用同一 power mode 和同一 scope。

## 6. Preliminary mapping 规则

每个表格单元必须携带 evidence status：

| 状态 | 含义 | 可否进入论文最终实测表 |
|---|---|---|
| `M` | 本轮按新合同真实测量 | 可以，仍需独立审计 |
| `H` | 历史真实测量，SHA/合同未完全对齐 | 不可以 |
| `E` | 由真测锚点和分阶段模型映射 | 不可以 |
| `NA` | 无合法映射或 build failure | 不可以 |

映射优先级：

1. 同模型、同结构、同 dtype、同 Orin power mode 的 `M`；
2. 同配置历史 `H`，只用于漂移检查；
3. 以 body 真实比率和 stage breakdown 分段映射为 `E`；
4. 不满足以上条件则 `NA`。

禁止直接按 TOPS 做单一比例缩放。低成本板卡预测至少拆分 engine compute、memory-bound、CPU preprocess/postprocess，并给出区间而不是单点伪精确值。

## 7. 本轮结果与时延口径审计

本节表格已按 31 号交接文档第 11.3、14.3 节重构。每个模型保留五个
**可执行代表配置**；`tune -> compress` 因冻结 policy 在 H800 阶段
`16/16` 迁移失败、没有合法配置，不伪造第六条数值行。F-Cooper 不属于这两个
来源章节，本轮不在本表中占位。

统一结果根目录：

```text
${V2X_ROOT}/results/lane_c_orin_stage6_five_config_20260724/
```

### 7.0.1 Pyramid：来源于第 11.3 节的五配置

| 方法 | 配置 | H800 源 AP70 / latency ms / energy J | Orin 部署路径 | Orin AP50 / AP70 | AP 合同/状态 | Orin latency median (ms) | Orin energy (J/batch2 invocation) | Evidence | Orin 终态 |
|---|---|---|---|---:|---|---:|---|---|---|
| original/default | `(64,128,256,fp32)` | `0.6311 / 3.2187 / 1.1636` | native PyTorch/CUDA FP32 | `0.790771 / 0.631360` | full_1789 success | 128.747826 | 1.060172; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| compression-only | `(24,128,64,fp16)` | `0.5377 / 0.5173 / 0.1318` | TensorRT FP16 standard build | `0.767906 / 0.540773` | full_1789; 0 failed; 0 fallback | 12.427008 | 0.107964; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| schedule-only | `(64,128,256,fp32)` | `0.6314 / 1.1038 / 0.4305` | TensorRT strict-FP32 tactic build | `0.790768 / 0.631390` | full_1789; 0 failed; 0 fallback | 120.944336 | 0.833228; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| compress -> tune | `(32,48,128,fp16)` | `0.5332 / 1.6008 / 0.4677` | TensorRT FP16 tuned/build | `0.739985 / 0.508246` | full_1789; 0 failed; 0 fallback | 36.531216 | 0.248038; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| joint SHCoSearch | H800 source `(16,32,64,int8)`；Orin control `(16,32,64,fp16)` | `0.6189 / 0.4524 / 0.1112`（H800 INT8） | TensorRT FP16；复用 calibration-locked Orin engine/AP | `0.785480 / 0.621029` | full_1789; 0 failed; 0 fallback | 10.622080 | 0.083540; `VIN_SYS_5V0` | M（跨精度控制替代） | Orin FP16 AP/latency/energy measured |

### 7.0.2 CoDriving：来源于第 14.3 节的五配置

| 方法 | 配置 | H800 源 AP70 / latency ms / energy J | Orin 部署路径 | Orin AP50 / AP70 | AP 合同/状态 | Orin latency median (ms) | Orin energy (J/batch2 invocation) | Evidence | Orin 终态 |
|---|---|---|---|---:|---|---:|---|---|---|
| original/default | `(64,128,256,fp32)` | `0.3975 / 1.8500 / 0.6999` | native PyTorch/CUDA FP32 | `0.389746 / 0.293111` | full_1789 success | 68.980675 | 0.562511; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| compression-only | `(32,64,96,fp16)` | `0.4190 / 0.3579 / 0.1013` | TensorRT FP16 standard build | `0.511936 / 0.323025` | full_1789; 0 failed; 0 fallback | 8.479584 | 0.071739; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| schedule-only | `(64,128,256,fp32)` | `0.3973 / 0.7686 / 0.3510` | TensorRT FP32 tactic build | `0.347966 / 0.211007` | full_1789; 0 failed; 0 fallback；输出对参考 mean/max abs error=`0.295438/26.965477` | 92.487297 | 0.709235; `VIN_SYS_5V0` | M（数值不等价） | AP/latency/energy measured; numerical parity failed |
| compress -> tune | `(32,80,128,fp16)` | `0.3984 / 0.3892 / 0.1391` | TensorRT FP16 tuned/build | `0.509579 / 0.317147` | full_1789; 0 failed; 0 fallback | 10.823024 | 0.087265; `VIN_SYS_5V0` | M | AP/latency/energy measured |
| joint SHCoSearch | H800 source `(16,32,64,int8)`；Orin control `(16,32,64,fp16)` | `0.3565 / 0.2976 / 0.0708`（H800 INT8） | TensorRT FP16 fresh build | `0.489677 / 0.268293` | full_1789; 0 failed; 0 fallback；输出 mean/max abs error=`0.002180/0.069327` | 5.719824 | 0.048895; `VIN_SYS_5V0` | M（跨精度控制替代） | FP16 substantially recovers INT8 AP, but remains below native FP32 |

上表中的 latency 统一要求为 multiscale-backbone
`engine-compute/no-data-transfer`；native 行使用相同输入/输出边界的 PyTorch
CUDA compute。AP 统一使用 full_1789 bridge，只替换
`get_multiscale_feature`；energy 必须来自与 latency 同窗口的 Orin
`tegrastats` 命名 rail。本轮通过临时 sudo 读取 `VIN_SYS_5V0` 和
`VDD_GPU_SOC`；主表 energy 固定采用
`mean(VIN_SYS_5V0 active window) × median CUDA-event latency / 1000`，
单位为一次 batch=2 backbone invocation。`VDD_GPU_SOC` 仅作为附加 rail，
不与 `VIN_SYS_5V0` 相加。

表内 latency 只展示主统计量 median；每条配置的 p90、p99、mean、1500 个
样本合同、AP50/AP70 原始值、raw SHA 和报告 SHA 见：

```text
${V2X_ROOT}/results/lane_c_orin_stage6_five_config_20260724/summary_fp16_joint_sudo_energy/
```

### 7.1 `133.5194702148 ms` 测到的部分

Pyramid original 的 `133.5194702148 ms` 是本轮 M0 在 Orin 上的三进程
median-of-medians。其边界为：

- 输入：post-scatter `spatial_features` 和 `t_ego`，agent batch = 2；
- TensorRT engine 内包含 Pyramid backbone、协同 fusion、deblock/neck、
  shrinker 和 `cls/reg/dir` heads；
- CUDA event 统计 engine compute；
- 不包含 voxelization、pillar/BEV encoder、数据传输、NMS、Python glue 和
  full-chain wall-clock。

因此该值是 **full body/dense engine-compute**，不是 H800 搜索中
multiscale-backbone subnet 的时延，也不是端到端时延。

### 7.2 `20.3105 ms` 测到的部分

Pyramid H800 winner 行中的 `20.3105 ms` 不是本轮 current M1 engine 的实测值。
它来自 E6 的旧 `p75_int8_orin.engine` 历史记录，口径为
`body_subnet_collab2_orin`，并与同一 E6 行的 `6.821 W`、`0.138796 J`
配套，所以只能标为 `H`。旧 engine SHA `a31a1056...`、旧 ONNX SHA
`2bd922ba...` 与本轮 current M1 engine/ONNX 不一致。

本轮 current M1 full-body INT8 engine 虽然在 Orin 本地 build 成功，但没有
填入 latency：它在性能测量前停止于数值门禁，故当前 anchor 终态为 `NA`。

### 7.3 与 H800 winner 时延的关系

H800 上 `(16,32,64,INT8)` 的 `0.456288 ms`（搜索测量）和
`0.452384 ms`（独立复测）只覆盖 `get_multiscale_feature` 对应的
multiscale-backbone subnet，输入为 `[2,64,128,256]`，输出为三层
multiscale feature。它不包含 fusion、shrinker 和检测 heads。

所以本节中的 `133.519 ms`、历史 `20.3105 ms` 和 H800 `0.452–0.456 ms`
不是同一计算图边界，禁止用于直接计算跨硬件加速比。下一轮必须先把 Orin
收缩到与 H800 完全相同的 multiscale-backbone scope。

## 8. 产物目录

统一根目录：

```text
${V2X_ROOT}/results/lane_c_orin_quick_mapping_20260723/
```

最低产物：

```text
capability/orin_profile.json
manifests/orin_anchor_manifest.json
build/{anchor}/build.log
build/{anchor}/artifact_sha.json
measurements/{anchor}/latency_repeats.csv
measurements/{anchor}/tegrastats.log
audits/numerical_and_fallback_audit.json
orin_preliminary_mapping.csv
orin_preliminary_mapping_audit.json
lane_c_orin_summary.md
```

## 9. 停止条件

快速停止点：

- 完成 Orin 环境审计；
- M0、M1 各获得本轮 `M`，或明确说明为什么只能保留 `H/NA`；
- 两个成功锚点具备 3 次独立 latency/energy 重复和 SHA；
- 生成完整的 `M/H/E/NA` preliminary 表和审计；
- 明确是否值得继续 full-chain、设备端 AP 和 K=4 calibration。

若 M1 INT8 数值失败，必须以失败收口，不自动改成 FP16。若历史 engine 可运行但 SHA 不匹配，先填 `H`，同时保留新 build blocker。

## 10. 禁止事项

1. 不把 H800 engine 当成 Orin engine；
2. 不把 E7 body 与 E8 full-chain 混成同一 latency；
3. 不把历史 synth 行升级为 `M`；
4. 不复用 H800/4090 INT8 AP；
5. 不把 `E/H` 行写进论文最终实测主表；
6. 不用单一 TOPS 比率输出低成本板精确延迟；
7. 不在本窗口启动 F-Cooper 或 CPU-TVM 工作。

## 11. 下一轮同口径 Orin 测试 `/goal`

> 历史命令：已由第 13 节完成。当前最新 `/goal` 见第 16 节。

```text
/goal 执行 Lane C 下一轮 Orin/H800 multiscale-backbone 同口径复测。严格依据 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/34_7_14_冷启动文档_LaneC_Orin快速真测与映射表_v1.md，只负责 Pyramid (16,32,64) 的 get_multiscale_feature/backbone subnet，不量化 fusion、shrinker、cls/reg/dir heads，不修改 H800 Table 1，不运行 F-Cooper 或 CPU/TVM 工作。以 H800 已验证源为唯一对照：checkpoint SHA d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279、multiscale ONNX SHA 8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5、输入 [2,64,128,256]、三层 multiscale feature 输出、TensorRT INT8+FP16 fallback。先恢复 H800 calibration manifest SHA 03ef12e30eb1ff1ac98f5d3a0a64007027c5bf79077e6299c13337762bb30015 对应的 15 个 batch2 NPY，并逐文件校验 SHA；若无法恢复或字节级重建一致，必须以 calibration_not_identical 阻塞“完全同口径”结论，不得静默换用 Lane C full-body calibration。禁止复制 H800 engine，必须在 <PRIVATE_HOST> 的 Orin 上由同一 ONNX 和同一 calibration 本地分别 build FP16、INT8 engine，保存 engine/cache/inspector/build-log SHA。数值检查必须使用未参与 calibration 的真实 DAIR held-out spatial_features，禁止以标准高斯张量作为唯一门禁；逐层比较 Orin FP16、Orin INT8 与服务器参考的三层 feature，报告 cosine、nRMSE、MAE 和饱和/裁剪统计。性能主口径严格复刻 H800：batch=2、warmup=20、iters=300、repeat=5、CUDA event、engine-compute/no-data-transfer，分别报告 FP16/INT8 median、p90、p99、mean；可附加 warmup>=200、每轮>=500 且>=10秒、3独立进程的 Orin 稳态审计，但必须单列，不能替换跨硬件主口径。AP 必须复刻 H800 full_1789 bridge：仅用 Orin TRT engine 替换 get_multiscale_feature，其余 fusion/shrinker/heads 保持同一 PyTorch checkpoint 和精度，成对报告 FP16/INT8 AP50、AP70 及 delta，不复用 H800 AP。功耗分别标注 H800 NVML board power 与 Orin tegrastats rail，禁止把不同 rail 当成同一物理量。统一产物写入 ${V2X_ROOT}/results/lane_c_orin_backbone_parity_20260723/。停止条件：source/calibration SHA 对齐审计、Orin FP16/INT8 本地 engine、真实 held-out 数值报告、H800 同协议 latency 回放、Orin 5-repeat latency、full_1789 AP bridge、功耗口径说明、跨硬件对照表和结论全部生成；任何 scope、calibration 或 AP bridge 不一致必须显式降级证据等级，不得输出伪跨硬件加速比。
```

## 12. 本轮研究问题与方法学反思

1. **量化 scope 漂移。** H800 AP/latency 的正式图只替换 multiscale backbone，
   本轮 Orin 却扩展到 fusion、shrinker 和 heads 的 full-body engine。两者不能
   用于直接归因“硬件差异”，这是本轮最核心的设计偏差。
2. **把扩展诊断误当成 winner 复现。** current M1 full-body 数值失败只能写成
   `full-body INT8 exploratory diagnostic failure`，不能表述为
   “H800 winner 在 Orin 上精度失败”。
3. **数值门禁输入分布不匹配。** 本轮确定性高斯输入约 50% 为负、几乎无零值；
   真实 DAIR calibration activation 非负且约 37.5% 为零。OOD 压力输入可保留，
   但不能作为阻断真实部署 AP 的唯一证据。
4. **full-body calibration 覆盖不足。** calibration 的 `t_ego` 全部使用 identity，
   没有覆盖真实协同变换分布；虽然现有 identity/轻微平移样本误差接近，尚不能
   将其认定为主因，但该设计不足必须记录。
5. **历史 `H` 与本轮 `M/NA` 混淆风险。** `20.3105 ms` 是旧 p75 engine 的
   E6 历史 body-subnet 数据，不是 current M1 的 Orin latency；表格必须同时给出
   scope、engine SHA 和 evidence status。
6. **TensorRT 与校准资产未对齐。** H800 使用 TensorRT 10.13、backbone ONNX
   `8f09b525...` 和旧 calibration manifest；Orin full-body 使用 TensorRT 8.5、
   ONNX `ab3eb36a...` 和重新生成的 calibration。engine tactic 差异不能在图和
   calibration 已不同的情况下单独归因于硬件。
7. **功耗只可同边界解释。** H800 NVML board power 与 Orin tegrastats rail
   不是同一电气边界；跨硬件主比较应优先使用同 scope latency 和同 bridge AP，
   energy 只能带采样接口与 rail 分列报告。

本轮结果仍有价值：它证明了 Orin 本地 full-body 构建链、SHA 审计、no-fallback
检查和重复测量流程可运行；但跨硬件结论必须由第 11 节的新一轮
multiscale-backbone 同口径实验重新建立。

## 13. 2026-07-23 multiscale-backbone 同口径复测结果

本节记录第 11 节 `/goal` 的实际执行结果。**不修改第 7 节历史 preliminary
mapping 表，也不修改 H800 Table 1。**

### 13.1 Scope 与证据等级

本轮 Orin engine 只包含 Pyramid `(16,32,64)` 的
`get_multiscale_feature` / multiscale-backbone subnet，输入
`[2,64,128,256]`，输出三层 multiscale feature。fusion、shrinker 和
`cls/reg/dir` heads 保持 PyTorch checkpoint 精度。

checkpoint SHA `d08fb16e...a279` 和 ONNX SHA `8f09b525...2a5` 均与 H800
已验证源一致。H800 calibration manifest SHA `03ef12e3...015` 也能校验，但
对应 15 个原始 batch2 NPY 已无法找回；按原命令在 GPU7 重建后 15/15 文件均
不是 byte-identical。因此终态为：

```text
calibration_status = calibration_not_identical
evidence_grade = degraded
cross_hardware_speedup = not_reported
```

本轮没有改用 Lane C full-body calibration，也没有复制 H800 engine。

### 13.2 Orin 本地 engine 与真实 held-out 数值

Orin TensorRT `<PRIVATE_HOST>` 本地构建结果：

| Precision | Engine SHA | Cache SHA | Layer precision |
|---|---|---|---|
| FP16 | `dd2da2815280...` | NA | FP16 40 / FP32 16 |
| INT8+FP16 fallback | `f161831b84d9...` | `96b104837008...` | INT8 59 / FP16 2 / FP32 5 |

真实 DAIR held-out 使用 calibration 前 16 个 scenes 之后的 16 个 scenes，
组成 16 个 batch2。与服务器 ORT FP32 三层 feature 比较：

| Precision | Output | Cosine | nRMSE | MAE |
|---|---|---:|---:|---:|
| FP16 | level0 | 0.99999964 | 0.00086055 | 0.00095097 |
| FP16 | level1 | 0.99999169 | 0.00416617 | 0.00141042 |
| FP16 | level2 | 0.99996413 | 0.00853298 | 0.00127107 |
| INT8 | level0 | 0.95063792 | 0.31728564 | 0.17854745 |
| INT8 | level1 | 0.86753786 | 0.49737193 | 0.17007230 |
| INT8 | level2 | 0.88467163 | 0.46844214 | 0.08799364 |

INT8 observed max 约为 `4.57 / 3.69 / 2.93`，服务器参考约为
`21.49 / 14.79 / 13.60`，三层均出现显著动态范围压缩。这里的 saturation /
clipping 是输出分布 proxy，不是 TensorRT 内部 quantizer counter。

### 13.3 同协议 latency 和功耗

主口径固定为 batch=2、warmup=20、iters=300、repeat=5、CUDA event、
engine-compute/no-data-transfer：

| Device | Precision | Median ms | p90 ms | p99 ms | Mean ms | Power interface |
|---|---|---:|---:|---:|---:|---|
| H800 archived replay | FP16 | 0.456800 | 0.459008 | 0.461312 | 0.456934 | 299.024 W, NVML board |
| H800 archived replay | INT8 | 0.456288 | 0.459424 | 0.463776 | 0.456501 | 297.371 W, NVML board |
| Orin local | FP16 | 10.528880 | 10.547456 | 10.573600 | 10.531149 | 7.763 W, tegrastats `VIN_SYS_5V0` |
| Orin local | INT8 | 9.303072 | 9.328192 | 9.359520 | 9.300231 | 7.401 W, tegrastats `VIN_SYS_5V0` |

H800 行是已验证归档证据回放，不是本窗口的新 H800 执行。NVML board power 与
Orin tegrastats rail 不是同一物理量；calibration payload 和 TensorRT 版本也
不一致，所以不计算 latency speedup 或 energy ratio。

### 13.4 Orin full_1789 成对 AP

两条 bridge 均只用 Orin TRT engine 替换 `get_multiscale_feature`，其余模型
保持同一 checkpoint 的 PyTorch FP32 参数：

| Precision | Samples | Failed | Fallback | AP50 | AP70 | Delta vs FP16 |
|---|---:|---:|---:|---:|---:|---|
| FP16 | 1789 | 0 | 0 | 0.78555450 | 0.62087188 | 0 / 0 |
| INT8 | 1789 | 0 | 0 | 0.00000000 | 0.00000000 | -0.78555450 / -0.62087188 |

两行 AP 均为 Orin engine 真测，没有复用 H800 AP。Orin 缺少服务器 spconv
voxel generator API，因此使用 pure-PyTorch compatibility voxelizer；它在
60 个真实 DAIR infrastructure frames 上与 spconv 达到 coordinate、point
content、`num_points` 全部 exact，但尚未对 full_1789 逐帧证明 byte identity。
该差异与 calibration mismatch 一并计入 degraded evidence。

### 13.5 对本路问题的更新反思

1. **原始 calibration payload 丢失是当前首要阻断项。** manifest SHA 正确并不
   代表 calibration 可复现；以后必须把 manifest 和全部 NPY 作为同一不可分割
   资产封存。
2. **相同生成命令不等于相同 calibration bytes。** GPU activation export 的
   非字节确定性足以阻断“完全同口径”结论，不能用“场景相同”替代逐文件 SHA。
3. **Orin INT8 的直接失败征象在 backbone feature。** 三层 nRMSE 达
   `0.317–0.497`，动态范围严重压缩，随后 full_1789 AP 归零；这说明问题发生在
   量化 backbone，而不是 fusion/heads 被误量化。
4. **尚不能把 AP 崩溃单独归因于 Orin 硬件。** calibration bytes、
   TensorRT `10.13 vs 8.5`、tactic/quantization implementation 同时不同。
   下一轮必须让 H800 也使用本轮同一 rebuilt calibration bytes 重新 build，
   才能区分 calibration 与 runtime/hardware 因素。
5. **时延边界现已对齐。** 本轮 Orin `9.30–10.53 ms` 和 H800 `0.456 ms`
   都只覆盖 multiscale-backbone engine compute；第 7 节 `133.519 ms` 和历史
   `20.3105 ms` 仍属于不同 full-body/body-subnet 边界，禁止混算。
6. **功耗 rail 与 AP preprocessing 都属于协议。** NVML 与 tegrastats 不可
   合并；preprocessing compatibility fallback 即使数值抽检 exact，也必须写入
   evidence downgrade。

完整产物：

```text
${V2X_ROOT}/results/lane_c_orin_backbone_parity_20260723/
```

关键入口为 `lane_c_orin_backbone_parity_summary.md`、
`cross_hardware_comparison.json`、`ap_summary.csv`、
`latency_power_summary.csv`、`numerical_summary.csv` 和 `audits/`。

## 14. 下一轮 calibration-locked 归因实验 `/goal`

> 历史命令：该轮已收口。当前最新 `/goal` 见第 16 节。

```text
/goal 执行 Lane C Pyramid (16,32,64) multiscale-backbone calibration-locked H800/Orin 归因复测。严格沿用 ${V2X_ROOT}/results/lane_c_orin_backbone_parity_20260723/ 的 source、held-out、latency 和 full_1789 bridge 合同，只量化 get_multiscale_feature/backbone subnet，不量化 fusion、shrinker、cls/reg/dir heads，不修改 H800 Table 1，不运行 F-Cooper、CPU/TVM 或 full-body 量化。以 checkpoint SHA d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279、ONNX SHA 8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5 和本轮 rebuilt_exact_command_gpu7 的 15 个 batch2 NPY 为新的共享 calibration payload；先生成 canonical manifest，逐文件记录 SHA，并确认 H800 与 Orin 接收端 15/15 byte-identical。禁止复用任何旧 engine，必须在 H800 与 <PRIVATE_HOST> Orin 上分别由同一 ONNX、同一 calibration bytes 本地 build FP16 和 INT8+FP16 fallback engine，保存 TensorRT 版本、engine/cache/inspector/build-log SHA 和逐层 precision。使用同一真实 held-out batch2 输入逐层报告 cosine、nRMSE、MAE、observed-range 与 clipping proxy；主 latency 固定 batch=2、warmup=20、iters=300、repeat=5、CUDA event、engine-compute/no-data-transfer；AP 固定 full_1789，仅替换 get_multiscale_feature，成对报告两台设备各自 FP16/INT8 AP50、AP70 和 delta。设置归因判据：若 H800 使用共享 rebuilt calibration 后同样 AP 崩溃，则主因归入 calibration；若 H800 保持而 Orin 崩溃，则进入 TensorRT 8.5 vs 10.13 的 layer-scale、dynamic-range、precision-placement 和 tactic 差异审计，仍不得直接写成纯硬件因果。禁止微调作为首轮补救；只有完成共享 calibration 的双端复测后，才能把 QAT/微调列为独立后续实验。功耗继续分列 H800 NVML board power 与 Orin tegrastats rail。统一产物写入 ${V2X_ROOT}/results/lane_c_calibration_locked_attribution_20260724/；任何 source、calibration、scope、held-out、latency 或 AP bridge 不一致必须显式降级，禁止输出伪跨硬件加速比。
```

## 15. 2026-07-24 Original `(64,128,256)` strict-FP32 Orin 实测

### 15.1 范围、源与严格精度

本轮只测试 Pyramid Original 的 `get_multiscale_feature` /
multiscale-backbone subnet；不量化 fusion、shrinker 和
`cls/reg/dir` heads。锁定源为：

- checkpoint SHA `4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2`；
- config SHA `f55a6ad8cff9fa522ce74fe9ef4d85d682314455278240ce7d5ec998d567f540`；
- ONNX SHA `3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff`；
- 输入 `[2,64,128,256]`，三层输出通道为 `(64,128,256)`。

Orin TensorRT `<PRIVATE_HOST>` 本地重新构建的 engine SHA 为
`69e208f14e1bd3ae55534cdf49fd4b149d1819bc26a55e6bb9806de4a7b6a6e7`。
inspector 中 54 层 compute 均为 FP32，FP16/INT8/TF32/BF16/other 均为 0，
因此这里的 FP32 是严格精度门禁后的结果，不是允许 TF32 或 fallback 的泛称。

### 15.2 真实 held-out 数值门禁

从同一 epoch-23 checkpoint 导出的真实 DAIR spatial features 为 float32；
固定 batch2 held-out SHA 为
`84fa0d43dd4c4a4f8a499ab6835bf3dfb09518e8b3e5d70fc378266dd7de1a2a`，
shape 为 `[30,2,64,128,256]`。Orin TRT 与服务器 ORT FP32 的逐层比较为：

| Output | Cosine | nRMSE | MAE |
|---|---:|---:|---:|
| level0 | 1.000000000000 | 0 | 0 |
| level1 | 1.000000000001 | 1.871853e-7 | 2.131440e-8 |
| level2 | 1.000000000000 | 1.587072e-7 | 5.705616e-9 |

全部输出 finite，未发现相对参考范围的越界/截断征象。首次在 Miniforge Python
中出现的 cuDNN 初始化失败被定位为解释器/动态库组合问题；切换到 Orin 系统
TensorRT 对应的 Python 环境后，新旧 engine 均可正常反序列化，因此没有通过
修改 engine 或放宽精度门禁来绕过该问题。

### 15.3 时延边界与 full_1789 AP

主时延协议为 batch=2、warmup=20、iters=300、repeat=5、CUDA event、
engine-compute/no-data-transfer，共 1500 个样本：

| Variant | Checkpoint / structure / precision | Median ms | p90 ms | p99 ms | Mean ms | AP50 | AP70 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original strict-FP32 | epoch23 / `(64,128,256)` / FP32 | 118.501488 | 118.593025 | 119.206368 | 118.517908 | 0.790983 | 0.631202 |
| compact FP16（既有部署结果） | pruned / `(16,32,64)` / FP16 | 10.539296 | 10.559008 | 10.728864 | 10.541548 | 0.785480 | 0.621029 |

这里的时延只包含 TRT `get_multiscale_feature` engine compute，不包含
voxelization、fusion、shrinker、检测 heads、NMS、主机/设备传输或完整链路
wall-clock。AP 则使用 full_1789 bridge：仅以 Orin TRT engine 替换
`get_multiscale_feature`，其余网络保持各自 checkpoint 的 PyTorch 精度。
Original strict-FP32 为 1789 成功、0 失败、0 fallback，5367/5367 个三层输出
均完成比较且 finite。

两行同时改变 checkpoint、结构和精度，只能解释为“部署模型变体对照”，不能
把时延差异或 AP 差异归因为纯通道数、纯剪枝或纯 FP32/FP16 效应。因此本节不
计算加速比，也不把 AP 差值写成单因素收益。full_1789 bridge 使用
pure-PyTorch compatibility voxelizer；它是当前 Orin 环境的可复现实测路径，
但与服务器 spconv preprocessing 的实现差异仍需作为证据边界保留。

### 15.4 功耗与本轮反思

此前非特权 `tegrastats` 没有输出命名功耗 rail，因此旧终态为
`missing_tegrastats_power_rails`。本轮获得用户授权后通过临时 sudo
重新运行同一 20/300×5 latency 窗口，十条配置均取得 `VIN_SYS_5V0` 和
`VDD_GPU_SOC`；密码没有写入脚本、报告或文档。主表只用
`VIN_SYS_5V0 mean W × median latency s` 计算 J/batch2 invocation，
没有用温度、利用率或历史 watt 替代。

本轮进一步确认：

1. Original 的“FP32”必须以逐层 inspector 排除 TF32/FP16/INT8 后才能成立；
2. held-out 必须与当前 checkpoint 一致，不能复用 pruned checkpoint 的
   activation 作为 Original 主门禁；
3. backbone engine 时延与 full-model AP bridge 是两个不同边界，表格必须
   明确区分；
4. 当前 Original FP32 与 compact FP16 不是单变量实验；下一轮应补齐同一部署
   变体内的精度控制，而不是回到 full-body INT8；
5. 功耗 rail 不可访问时应报告不可用；取得权限后也必须把 rail 名称、sudo
   状态、原始日志 SHA 和能量公式写入报告，不能只填一个无来源的 J 数值。

完整产物：

```text
${V2X_ROOT}/results/lane_c_orin_original_fp32_20260723/
```

关键入口为 `final_summary.json`、`comparison.csv`、`summary.md`、
`orin/fp32/numerical_comparison.json`、`orin/fp32/primary_latency_power.json`
和 `ap_full/orin/fp32/stage3_trt_multiscale_ap_bridge_report.json`。

## 16. 历史计划：部署变体 × 精度控制补测（已取消）

> **2026-07-24 纠偏：本节方案不再执行。** Original FP16 与 compact
> strict-FP32 的 `2×2` 控制矩阵不是当前补测目标。当前唯一有效任务是复刻
> 31 号交接文档第 11.3、14.3 节，在 Orin 上完成 Pyramid、CoDriving 各五个
> 可执行代表配置的 AP、multiscale-backbone latency 与同窗口 energy 测量，
> 并将终态写回本文第 7 节。以下命令仅作为被取消方案的历史记录，禁止继续据此
> 派发测试。

```text
/goal 执行 Lane C Orin Pyramid deployed-model-variant × precision 控制补测。严格依据 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/34_7_14_冷启动文档_LaneC_Orin快速真测与映射表_v1.md，只补齐当前 2×2 对照中缺失的两个单元：Original checkpoint SHA 4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2、ONNX SHA 3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff、结构 (64,128,256) 的 Orin FP16；以及 compact checkpoint SHA d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279、ONNX SHA 8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5、结构 (16,32,64) 的 Orin strict-FP32。只负责 get_multiscale_feature/multiscale-backbone subnet，不量化或替换 fusion、shrinker、cls/reg/dir heads，不运行 INT8、H800、F-Cooper、CPU/TVM 或 full-body 工作，不微调。禁止复制既有 engine，必须在 <PRIVATE_HOST> Orin 上由各自锁定 ONNX 本地 fresh build；Original FP16 必须保存逐层 FP16/FP32 placement，compact strict-FP32 必须关闭 TF32/FP16/INT8/BF16 并由 inspector 证明所有 compute layer 为 FP32。每个变体使用与其 checkpoint 一致、未参与 build 的真实 DAIR float32 held-out spatial_features，逐层对服务器参考报告 cosine、nRMSE、MAE、finite 和 observed-range/clipping proxy。性能统一采用 batch=2、warmup=20、iters=300、repeat=5、CUDA event、engine-compute/no-data-transfer，分别报告 median、p90、p99、mean；AP 统一采用 full_1789 bridge，仅替换 get_multiscale_feature，其余网络保持对应 checkpoint 的 PyTorch 精度，报告 AP50、AP70、1789/failed/fallback 和 5367 次输出比较。将本轮已完成的 Original strict-FP32 证据根 ${V2X_ROOT}/results/lane_c_orin_original_fp32_20260723/ 与 compact FP16 证据根 ${V2X_ROOT}/results/lane_c_calibration_locked_attribution_20260724/ 锁定为只读对照，最终生成四单元原始数值表；只允许在同一 checkpoint/结构内部解释 FP32-vs-FP16 精度效应，compact-vs-Original 仍标注为 checkpoint+structure 的部署变体差异，不得声称纯通道数或纯剪枝因果，不输出未经支持的跨硬件加速比。功耗仅在 tegrastats 命名 rail 可访问时报告；若仍缺少权限，必须以 missing_tegrastats_power_rails 阻断能耗比较且不得填入估算值。统一产物写入 ${V2X_ROOT}/results/lane_c_orin_variant_precision_matrix_20260724/；停止条件为两个新 engine、严格精度审计、真实 held-out 数值报告、两组 5-repeat latency、两组 full_1789 AP bridge、功耗状态、四单元对照表和因果边界说明全部生成。
```

## 17. 2026-07-24 Stage6 五配置 Orin 终态与方法学反思

### 17.1 完成状态

- 已严格从 31 号交接文档第 11.3、14.3 节取出 Pyramid、CoDriving 各五个
  可执行代表配置，共十条；没有执行已取消的 `2×2` 控制矩阵，也没有为
  `tune -> compress` 的 `16/16` policy-transfer 失败伪造配置。
- 两条 original/default 使用 native PyTorch/CUDA；其余八条 TensorRT engine
  均在 Orin 上由锁定 ONNX fresh build，没有复制 H800 engine。
- 十条 latency 均完成 batch=2、warmup=20、iters=300、repeat=5、CUDA
  event、multiscale-backbone compute/no-data-transfer，共 1500 个样本。
- 当前主表十条均完成 full_1789 AP。Pyramid joint 行按用户要求复用既有
  `(16,32,64,fp16)` calibration-locked 证据；CoDriving joint 由同宽度 ONNX
  在 Orin fresh build FP16，并重新运行 sanity16 与 full_1789。原 INT8
  失败证据保留在原目录，没有覆盖或改写。
- 十条均以 sudo `tegrastats` 重跑同一 latency 窗口，取得
  `VIN_SYS_5V0`/`VDD_GPU_SOC`。表中 energy 是前者均值乘 median CUDA
  latency，单位为 J/batch2 invocation；原始日志和报告 SHA 均已绑定。

### 17.2 本轮暴露的问题

1. **此前任务拆解错把五配置问题收缩成了两个控制单元。** 第 7 节真正对应的是
   11.3/14.3 的各五条可执行配置，不是 Original FP16 与 compact
   strict-FP32 的 `2×2`。第 16 节现已明确标记为取消，避免继续误派发。
2. **表内时延必须继续写清计算图边界。** 本轮数值只覆盖
   `get_multiscale_feature` / CoDriving `backbone.resnet` 的同边界计算，
   不包含 fusion、shrinker、检测 heads、NMS、数据传输或端到端 wall-clock；
   第 7.1、7.2 节历史 full-body/body-subnet 数值不得混入本表。
3. **FP16 控制证明 INT8 是主要风险源，但不是唯一差异。** Pyramid joint
   FP16 的 AP70=`0.621029`，替代了原 INT8 sanity 零预测行。CoDriving
   joint FP16 的输出 mean/max absolute error 降至
   `0.002180/0.069327`，AP70 从 INT8 的 `0.039415` 恢复到 `0.268293`；
   但仍低于同模型 native FP32 的 `0.293111`。因此可以说 INT8 是精度崩溃的
   主要因素，不能说 FP16 已与 native 完全等价。
4. **CoDriving 的问题不只发生在 INT8。** schedule-only FP32 的
   full_1789 AP70 为 `0.211007`，三层输出 mean/max absolute error 为
   `0.295438/26.965477`；同一 `(64,128,256)` native FP32 AP70 为
   `0.293111`。这证明当前 CoDriving TRT schedule 路径没有通过数值等价门禁。
   在进一步审计 ONNX 导出、输出排序、TensorRT precision placement 和 bridge
   输入前，不能把差异归因于 Orin 硬件。
5. **native original 与 TRT schedule 不是同一 runtime。** 即使宽度和
   checkpoint 相同，两行仍分别代表 native PyTorch/CUDA 与 TensorRT tactic
   build；其 latency 差异是部署路径差异，不能写成结构压缩收益。
6. **CoDriving AP 报告曾使用非标准文件摘要。** 旧
   `sha256_path(file)` 实际摘要为 `relative filename + NUL + raw bytes`，
   因而报告里的 engine/checkpoint SHA 与 build/manifest 的 raw-file SHA
   外观不同。四条 TRT 报告均已按旧算法复算一致，实际 engine raw SHA 与
   build、latency、同步文件一致，未发生 engine 替换；脚本现已修正为文件直接
   计算 raw SHA。历史 AP 报告保持原样，不能静默改写。
7. **功耗证据已经闭合，但物理量仍不能跨硬件直接等同。** Orin 主值来自
   `VIN_SYS_5V0`，H800 源值来自 NVML board power；二者 rail 边界不同，
   因此本轮仍不计算跨硬件 energy improvement 或 energy-delay 指标。
8. **CoDriving AP 使用当前 Orin compatibility preprocessing 路径。**
   full_1789 样本、checkpoint 和 bridge 均可复现，但其 preprocessing
   implementation 与 H800 历史环境不完全相同；H800 行在本节仅作为 31 号
   文档的锁定源值展示，不能据此声称纯硬件因果。

### 17.3 终态产物

```text
${V2X_ROOT}/results/lane_c_orin_stage6_five_config_20260724/
```

关键入口：

- `summary_fp16_joint_sudo_energy/final_summary.json`：十条最新终态、
  AP/latency/energy 与证据 SHA；
- `summary_fp16_joint_sudo_energy/orin_stage6_five_config.csv`：可直接制表的
  扁平结果；
- `summary_fp16_joint_sudo_energy/summary.md`：十条 Orin 原始值；
- `summary_fp16_joint_sudo_energy/artifact_sha256.json`：三份汇总产物 SHA；
- `contracts/five_config_manifest.json`：配置、H800 源值和 source SHA 合同；
- `contracts/fp16_joint_substitution_override.json`：两条 joint 的 Orin FP16
  替代合同；明确 H800 源行仍为 INT8；
- `pyramid/*`、`codriving/*`：fresh build、inspector、latency、AP 和失败证据。

本轮不提供新的 `/goal`。下一步若继续追求 CoDriving 与 native FP32 完全
对齐，应审计 FP16 bridge/preprocessing 的剩余 AP 差值及 schedule FP32
数值不等价；不应再把 joint INT8 的低时延单独作为可部署结论。
