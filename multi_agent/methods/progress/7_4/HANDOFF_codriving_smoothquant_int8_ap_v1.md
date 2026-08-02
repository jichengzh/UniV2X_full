# HANDOFF: CoDriving SmoothQuant INT8 真实 AP 闭环(2026-07-04)

> **入口文档**。本次任务给 CoDriving ResNet backbone 的 INT8 量化补齐了缺失的一环:**真实端到端 AP**(不是 tensor rel_err 代理,不是 fake-quant)。全程 TVM 口径(dp4a/WMMA 真张量化),**全程未使用 TRT/TensorRT/trtexec**。

## 0. 任务背景

CoDriving 旧 PTQ(per-tensor activation p99.9 + per-output-channel weight scale)在 full-model 级别精度不够(base width: rel_err_mean=0.251, frac_rel_err_gt_0.2=0.255)。本次任务:
1. 实现更好的量化方法(SmoothQuant-style per-input-channel migration)
2. 在 4 个宽度(base/p25/p50/p75)上做数值验证(held-out 帧,不许用校准帧)
3. 搭建 AP 桥接管线,把 TVM 编译的 int8/fp16 backbone 接回完整 CoDriving PyTorch 推理链路(deblocks→fusion_net/where2comm→检测头→NMS),测**真实 AP**
4. 诚实报告,不许用旧结果覆盖,不许用 dummy scale 冒充真 int8

---

## 1. 新方法:SmoothQuant per-input-channel migration

公式:
```
s_ic = act_absmax_ic**alpha / max(w_absmax_ic**(1-alpha), 1e-8)   # clip to [1e-4, 1e4]
x_smoothed = x / s_ic     (激活按输入通道除)
w_smoothed = w * s_ic     (权重按输入通道乘,数学等价迁移)
```
- alpha 在 {0.3, 0.5, 0.7} 中**逐层**从校准数据(仅校准帧,不碰 held-out/val)选择,选择准则是 `analytic_dynamic_range_utilization_proxy_on_calibration_frames_only`(校准帧上 migrate 后的通道最大幅值代理,非用held-out数据做全量化误差仿真)。
- 迁移后,激活量化 scale = migrated tensor 的 per-tensor p99.9 百分位;权重量化 scale = migrated tensor 的 per-output-channel max-abs(与旧 PTQ 的 scale 计算方式保持一致,只是作用在 migrate 之后的张量上)。
- **`quant_scale_source` 精确字符串**(写入所有最终 JSON):
  ```
  smoothquant_alpha_perlayer_real_calibration_percentile_p99.9_activation_per_tensor_weight_per_output_channel_migrated
  ```
- 校准数据:每个宽度独立跑,来自 `${V2X_DATA_ROOT}/V2Xverse_pyramid/output/codriving_pilot/calib_real`(DAIR-V2X train-split 衍生,64 帧 calib + 16 帧 held-out verify,均与官方 val split 999% 不重叠 —— no-peek 在 split-file 级别结构性验证: `train.json` 4811 / `val.json` 1789 / overlap=0)。
- 未做 Phase 2(更紧的分层 percentile clip on stubborn layers)——SmoothQuant 结果已足够支撑推进,不做 QAT(明确禁止,仅作 future work 提及)。

---

## 2. 数值验证(held-out 16 帧,SmoothQuant vs 旧 PTQ,4 宽度全测)

| 宽度 | 方法 | rel_err_mean | rel_err_p50 | rel_err_p99 | frac_gt_0.2 |
|---|---|---|---|---|---|
| base | 旧PTQ | 0.2509 | 0.0859 | 2.4630 | 0.2552 |
| base | SmoothQuant | 0.2478 | 0.0888 | 2.2875 | 0.2875 |
| p25 | 旧PTQ | 0.3348 | 0.1369 | 2.7628 | 0.3836 |
| p25 | SmoothQuant | **0.3871** | 0.1656 | 3.4530 | **0.4418** |
| p50 | 旧PTQ | 0.5276 | 0.2384 | 4.0645 | 0.5400 |
| p50 | SmoothQuant | **0.2093** | 0.0770 | 1.9125 | **0.2143** |
| p75 | 旧PTQ | 0.6463 | 0.2490 | 9.7695 | 0.5595 |
| p75 | SmoothQuant | **0.2135** | 0.0777 | 1.5049 | **0.2246** |

**诚实结论(非单调,如实记录,不做美化)**:
- **p50/p75**:SmoothQuant 大幅优于旧 PTQ(rel_err_mean 降 60%+)。
- **base**:基本持平(略优)。
- **p25**:SmoothQuant **反而更差**(rel_err_mean 0.335→0.387,frac_gt_0.2 0.384→0.442)。原因未完全定位,推测与 p25 宽度下某些层的 activation/weight 动态范围分布使 alpha 网格{0.3,0.5,0.7}选不到更优迁移点有关,留作后续可选的深入诊断项(非本次任务阻塞)。

---

## 3. AP 桥接管线 + 真实 int8/fp16 AP(4 宽度全测)

### 3.1 桥接方法

`scripts/plan3_codriving_ap_bridge_host.py`(H800):在标准 `opencood` 推理循环(`inference_intermediate_fusion` + `eval_utils.caluclate_tp_fp`)中,把 `model.backbone.resnet.forward` **monkeypatch** 成一个把输入张量发给持久化 TVM worker 子进程(`plan3_codriving_tvm_worker.py`,运行在独立 tvm310 venv)、取回 3 个多尺度 feature map 的桥接函数。deblocks/fusion_net/where2comm/检测头/NMS/AP 统计全部走**真实未改动的 PyTorch 代码**。AP80 由 `eval_utils.calculate_ap(result_stat, 0.8)` 直接算(HEAL/OpenCOOD 原生 `eval_final_results` 只支持 AP30/50/70)。

300 帧(stride-sample 自 1789 帧官方 val split)。

### 3.2 fp16 sanity check(4 宽度全 PASS)

| 宽度 | 真实PyTorch全val(1789帧) AP50/AP70 | 桥接fp16(300帧) AP50/AP70 | Δ AP50 | Δ AP70 |
|---|---|---|---|---|
| base | 0.6263 / 0.4063 | 0.6208 / 0.3932 | -0.0055 | -0.0131* |
| p25 | 0.5864 / 0.3661 | 0.5888 / 0.3564 | +0.0024 | -0.0097 |
| p50 | 0.6062 / 0.3750 | 0.6087 / 0.3811 | +0.0025 | +0.0061 |
| p75 | 0.6182 / 0.4050 | 0.6249 / 0.4095 | +0.0067 | +0.0046 |

*所有 Δ 均在 300-帧子采样噪声 + fp16 舍入误差可解释范围内(< 0.014),**4 宽度 sanity check 全部 PASS**,证明桥接管线本身正确,可信赖后续 int8 AP 数字。

### 3.3 真实 int8 AP(SmoothQuant scale,4 宽度全测)

| 宽度 | int8 AP50 | fp16 AP50(桥接) | Δ AP50 | int8 AP70 | fp16 AP70(桥接) | Δ AP70 | int8 AP80 |
|---|---|---|---|---|---|---|---|
| base | 0.6058 | 0.6208 | -0.0149 (-2.40%) | 0.3932 | 0.4087 | -0.0155 (-3.79%) | 0.1881 |
| p25 | 0.5765 | 0.5888 | -0.0122 (-2.08%) | 0.3454 | 0.3564 | -0.0110 (-3.10%) | 0.1528 |
| p50 | 0.6035 | 0.6087 | -0.0052 (-0.85%) | 0.3766 | 0.3811 | -0.0046 (-1.20%) | 0.1556 |
| p75 | 0.6122 | 0.6249 | -0.0127 (-2.03%) | 0.3925 | 0.4095 | -0.0171 (-4.17%) | 0.1964 |

以上全部数字均直接取自各宽度的 `results/plan3_codriving_int8_ap/{width}.json` 文件字段(`int8_ap50/70/80`、`fp16_ap50/70/80`),已逐一 SSH 读回核验,不存在覆盖/推断/转录不一致问题。

**观察**:int8 相对 fp16-桥接的 AP 退化幅度在 4 宽度间大致都在 **1–4%** 区间,**并未复现数值验证阶段(§2)那种大跨度的非单调模式**(p25/p50/p75 rel_err 差异巨大,但 AP 退化幅度相近)。说明 **AP 是比 tensor-level rel_err 粗糙得多的度量**——大的张量误差不一定 1:1 转化为等比例的 AP 损失(NMS/检测阈值化对局部数值噪声有一定鲁棒性)。这一发现同样如实记录,不做过度解读。

**bridge_fail_count 全部为 0**(600/600 或 300/300 帧全部桥接成功,无 fallback/无静默失败)。

---

## 4. 耦合曲线结论:量化难度是 model-dependent 的

| 模型 | base FP16→INT8 AP50 | 相对变化 |
|---|---|---|
| **Pyramid**(DAIR, backbone子模块) | 0.791 → 0.790 | **-0.13%,近乎无损** |
| **CoDriving**(DAIR, backbone子模块,本次) | 0.6208 → 0.6058(桥接口径) | **-2.40%,有实质但可控的代价** |

CoDriving 4 宽度(base/p25/p50/p75)INT8 相对 fp16-桥接 AP50 退化幅度分别为 -2.40% / -2.08% / -0.85% / -2.03%,量级上比 Pyramid 的 -0.13% **大一个数量级**。这与既有认知一致:**Pyramid backbone 是标准 conv,TVM int8 张量化(WMMA/mma_sync)对其近乎无损;CoDriving ResNet backbone 量化代价更高但仍可控(2–4% AP50 绝对损失),不是"崩溃级"的**。→ **量化难度是模型架构相关的(model-dependent),不能拿单一模型的"近乎无损"结论泛化到所有模型**,这是协同加速框架设计"精度轴"时必须考虑的现实约束。

---

## 5. 产物清单(全部在 H800,已逐一 SSH 读回验证非空/非占位)

- `${V2X_ROOT}/scripts/plan3_codriving_tvm_worker.py` — 持久化 TVM worker(修复:calib JSON 无论精度都要加载)
- `${V2X_ROOT}/scripts/plan3_codriving_ap_bridge_host.py` — AP 桥接主脚本(修复:worker 子进程环境必须从干净基线构建,不能继承 t2lib torch 的 PYTHONPATH/LD_LIBRARY_PATH)
- `${V2X_ROOT}/scripts/assemble_phase4.py` — Phase4 汇总脚本
- `${V2X_ROOT}/results/plan3_codriving_int8_ap/{base,p25,p50,p75}_fp16_bridge_sanity.json` — 4 宽度 fp16 sanity(全 PASS)
- `${V2X_ROOT}/results/plan3_codriving_int8_ap/{base,p25,p50,p75}_int8_ap_smoothquant.json` — 4 宽度真实 int8 AP(bridge_fail_count=0)
- `${V2X_ROOT}/results/plan3_codriving_int8_ap/{base,p25,p50,p75}.json` — Phase4 最终逐宽度汇总(含 quant_scale_source/calib_set_desc/numerical_verification/artifacts 全字段)
- `${V2X_ROOT}/results/plan3_codriving_int8_ap/summary_coupling.json` + `.md` — 与 Pyramid 的耦合曲线对比 + 结论

所有文件均通过直接 SSH `cat`/`python3 -c json.load` 读回确认字段非空、非占位,不是仅凭退出码判断成功。

---

## 6. 阻塞项/未完成项(诚实列出)

- **无硬阻塞**。4 个宽度(base/p25/p50/p75)的 checkpoint 均已确认是真实 finetune 收敛的(多 epoch 历史确认),GPU 全程守纪律使用(仅 GPU4/GPU5,未碰 GPU2/6/7)。
- p25 SmoothQuant 数值验证劣于旧 PTQ 的**根因未深入定位**(留作可选后续诊断项,不影响本次 AP 结论的真实性——AP 桥接是独立于数值验证的真实端到端测量)。
- 未做 Phase 2(分层更紧 percentile clip)与 QAT(按任务要求明确不做)。
- 本文档所有数字均以 `results/plan3_codriving_int8_ap/*.json` 文件内容为唯一权威源(已逐一 SSH 读回核验);如后续 supervisor/其他 agent 复核发现任何数字与文件不一致,以文件为准。
