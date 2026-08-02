# 4_7_2_frontier_01 异常策略与 cost_model 数据冻结口径

## 0. 本文用途
落定 P0 阶段两件事:① frontier_01 异常点在 cost-model 训练数据中的处理策略;② original60 三指标表冻结为 cost-model 训练/校准数据的口径。供后续 P1(cost model 训练)直接引用。

## 1. frontier_01 异常事实(已核实,权威 summary)
label=frontier_01,width=**24x64x128**(最窄 stage0=24 通道)。

| precision | AP70 | AP status | latency_ms | energy_j | 说明 |
|---|---|---|---|---|---|
| FP32 | **0.5924** | measured/true_eval | 21.357 | 4.070 | 正常 |
| FP16 | **0.0** | measured/true_eval | 4.486 | 0.872 | AP 塌缩 |
| INT8 | **0.0** | measured/true_eval | 3.477 | 0.669 | AP 塌缩,pred_nonempty_count=1480 |

## 2. 判定:真实"降精度精度悬崖",非测量 bug
证据:
1. FP32 在同宽度 AP 正常(0.592)→ 宽度本身不是问题,ONNX/权重/评测链路正常。
2. INT8 `pred_nonempty_count=1480`(1789 帧多数产出了非空预测)→ **前向确实执行、预测框确实生成**,只是与 GT 匹配后 AP=0(定位/置信度在降精度下退化到 0.3/0.5/0.7 IoU 阈值以下)。
3. FP16 与 INT8 **同时**塌缩(FP16 是 tensorcore-rewritten route)→ 指向"极窄 backbone(24 通道 stage0)+ 降精度"的共性数值退化,非某一路径的偶发 CUDA 错误。

结论:frontier_01 是 **reduced-precision detection collapse at extreme-narrow width**,是一个科学上真实的数据点(降精度精度悬崖),不是坏点/测量伪影。

## 3. cost-model 训练策略(落定)
1. **保留全部证据,不删**(rows/raw/summary 都保留 frontier_01 三精度 measured row)。
2. **标记为 anomaly/outlier**:在 cost-model 训练表构建时,给 frontier_01 打 `anomaly_flag=reduced_precision_ap_collapse`。
3. **AP 回归器训练**:frontier_01 的 FP16/INT8 AP=0 点**从平滑 AP 回归训练集中排除**(避免 0.0 异常值污染回归面),但**作为已知精度悬崖证据单列保留**;FP32 点正常保留入训练集。
4. **latency/energy 回归器**:frontier_01 三精度 latency/energy 均正常 measured,**全部保留入训练集**(悬崖只在 AP 轴,不在 lat/energy 轴)。
5. 论文叙事可用其作为"极窄宽度触发降精度精度悬崖"的定性证据点。

## 4. original60 三指标表冻结口径(P0-D 前置)
- 数据角色:original60 三精度表 = **cost-model 训练/校准数据**,不是直接 Pareto 前沿来源。
- 覆盖(权威 summary,180 cells):AP 180/180 measured;latency 待 lhc_17/s2_096 补齐后 180/180(当前 178/180);energy 180/180 measured。
- FP16 latency/energy 已用 tensorcore route 重写重测(fp16_rewritten_tensorcore_full60)。
- 重复行处理:rows 文件保留历史(含异常复核 rerun),canonical 选择在 coverage generator 构建时按 `is_compliant + score=(created_at,run_id)` 每 label 取最新合规行完成。**不在 rows 层删重复。**
- 冻结动作待 P0-A(latency 补点)+ P0-C(本异常策略)完成后执行。

## 5. 待办勾稽
- [ ] P0-A: lhc_17/s2_096 FP32 latency 补 measured(strict direct tuned 优先,失败则 default salvage)
- [ ] P0-B: coverage generator + validator PASS,latency 60/60
- [x] P0-C: frontier_01 异常策略(本文)
- [ ] P0-D: 冻结 cost-model 训练数据(引用本文 §3/§4 口径)
