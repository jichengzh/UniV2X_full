# P0.2 — 真 INT8 AP 闭环(Q1 + Q2):实施 + 结果 (v1)

> **日期**: 2026-05-31
> **目标**: 解决 plan v5 Phase C 的 INT8/FP16 AP=0 阻塞,拿到**真 INT8 AP**(+ 真 INT8 lat)。
> **关联**: `plan6_方法优化路线_交接_v1.md` §三/§六 P0.2、`P0_1_CUDA_NMS_结果_v1.md`。

---

## 一、结论(一句话)

**多输出 seam(Q2)+ 真校准(Q1)修复后,FP16 AP50 与 FP32 差 −0.0006、INT8 差 −0.0014(远在 0.02 门内),300 样本全成功。Phase C 的 AP=0 阻塞彻底解决,真 INT8 AP 可测且正确。**

## 二、实测 AP(p64 baseline,DAIR-V2X val 300 样本,GPU 7)

| Q | AP30 | AP50 | AP70 | ΔAP50 vs FP32 |
|---|---|---|---|---|
| FP32 (PyTorch) | 0.8401 | 0.7978 | 0.6420 | — |
| FP16 (TRT 3 输出) | 0.8399 | 0.7972 | 0.6417 | **−0.0006** |
| INT8 (TRT 3 输出 + 真校准) | 0.8385 | 0.7964 | 0.6389 | **−0.0014** |

n_ok = 300/300(全部成功,非 partial)。

## 三、两个根因怎么修的

### Q2 — 多输出 seam(解决 AP=0 主因)

plan v5 把 `get_multiscale_feature` + `decode_multiscale_feature` 合成**单输出 bev** export,旁路了两者之间的 V2X collab 融合(`single_head_i` + `warp_affine` + `weighted_fuse`)→ AP=0。

修法:TRT engine **只替换 `get_multiscale_feature`**,输出 **3 个 stage features**(feat0/1/2);融合 + decode 全留 PyTorch。
- `get_multiscale_feature(x) = self.resnet(x)` → tuple(feat0,feat1,feat2)
- ONNX 3 输出: feat0 (1,64,128,256) / feat1 (1,128,64,128) / feat2 (1,256,32,64)
- `TRTBackbone3` 逐 agent 跑 batch=1 引擎,拼回 batch → patch `get_multiscale_feature`

### Q1 — 真校准(解决 INT8 fallback)

plan v5 的 `MinMaxCalibrator.get_batch()` 直接 `return None`,只读一个错配的 stale cache → TRT 无 scale → fallback。

修法:`RealMinMaxCalibrator` 用**该 anchor 自己** harvest 的真实 `spatial_features`(186 个 batch,`p0_2_calibrate.py` 钩 `get_multiscale_feature` 输入采集)喂校准。INT8 AP 因此与 FP32 仅差 0.14%。

### 附带修复 — DAIR 几何

harvest 暴露真实输入是 **(64,128,256)**,plan v5 引擎用了错误的方形 **(64,256,256)** → 固定 256×256 引擎喂 128×256 真张量会 shape-mismatch,是 Phase C 失败的**又一原因**。本次按真实几何重导。

## 四、产物

- `scripts/phase2/p0_2_multiscale_trt.py` — 多输出 export + ONNX-runtime 数值 sanity(mean_abs 1.8e-4,PASS)
- `scripts/phase2/p0_2_calibrate.py` — Q1 真校准数据采集(钩 get_multiscale_feature 输入)
- `scripts/phase2/p0_2_build_eval.py` — FP16/INT8 引擎构建(真校准)+ TRTBackbone3 + AP 评估
- `data/p0_2_multiscale_ap.csv` — AP 结果
- ONNX `/tmp/plan6_p0_2_onnx/p64_baseline_multiscale.onnx`(20.95MB),引擎 `/tmp/plan6_p0_2_engines/p64_baseline_{fp16,int8}.engine`(FP16 11.91MB)

## 四·补、真 INT8 latency(2026-05-31,GPU 6 独占,get_multiscale 单 agent 1×64×128×256)

`p0_2d_int8_latency.py`,300 runs CUDA-Event:

| 变体 | mean (ms) | vs PyTorch fp32 |
|---|---|---|
| PyTorch fp32 | 2.939 | 1.00× |
| PyTorch fp16 (autocast) | 3.258 | **0.90×(反而慢)** |
| TRT fp16 | 0.468 | **6.28×** |
| TRT int8 | 0.364 | **8.08×** |
| **INT8 vs FP16 (TRT)** | — | **1.29×** |

**三个结论**:
1. **INT8 比 FP16 快 1.29×** → 量化轴有真实 lat 区分度(plan5 一直想拿、因 seam/cache bug 没拿到的数)。
2. **真正大杠杆是"进 TRT"本身**:TRT FP16 比 PyTorch fp32 快 **6.28×**;PyTorch→TRT 差距(6.28×)≫ FP16→INT8 差距(1.29×)。
3. **PyTorch fp16 autocast 比 fp32 还慢(0.90×)** → 坐实 plan5 之前 `int8_proxy_fp16`(autocast 当 INT8)无意义。

含义:加速故事应以"把 conv 模块编译进 TRT(FP16 即 6×)"为主、INT8(1.29×)为二阶精修。`data/p0_2d_int8_latency.json`。

## 四·补2、真实 e2e(pyramid get_multiscale 换进 TRT,DAIR 150 样本,GPU 6)

`p0_2e_e2e_with_trt.py`,基线已含 P0.1 的 CUDA NMS:

| mode | e2e mean (ms) | vs PyTorch |
|---|---|---|
| PyTorch | 21.25 | 1.00× |
| TRT fp16 | 15.15 | **1.40×** |
| TRT int8 | 14.47 | **1.47×** |

- 只把 pyramid ResNeXt 换进 TRT → e2e **1.40× (FP16) / 1.47× (INT8)**,AP 不变(−0.14%)。
- submodule 6.28× 在 e2e 缩到 1.40×:因 get_multiscale 只占 e2e ~38% + TRTBackbone3 逐 agent sync 开销。
- INT8 over FP16 在 e2e 仅 1.05×(submodule 1.29× 被稀释)。
- **剩余 14.5ms 现在由其他 PyTorch 模块主导**(encoder 2.4 / backbone 0.84 / shrink 1.4 / 融合 ~2.8 / postproc 2.85)→ 要再降 e2e,需把这些也搬进 TRT(全网 TRT 化)。`data/p0_2e_e2e_with_trt.json`。

## 五、状态:AP 闭环 ✅ / 真 INT8 lat ✅ / 真实 e2e ✅

- **已完成**: 真 INT8 **AP**(P0.2 核心阻塞)。框架现在能对 INT8 配置打分。
- **待办**: 真 INT8 **latency**(get_multiscale 子模块 fp16 vs int8 vs PyTorch),需 **C10 合规 idle GPU**;本次 GPU 6/7 跑到一半被其他用户占用,lat 暂缓。
- **待办**: 扩到 5 plane(p48/p32/p16/p8)——p64 已证管线,扩展是机械的。

## 六、对项目论点的意义

- **真 INT8 AP 可测**:量化轴从"AP 测不出"升级为"AP 可微分",搜索空间的量化维度成立。
- INT8 AP 仅降 0.14% → **再次印证该模型 AP 对量化极不敏感**(与项目核心 finding 一致):量化几乎"免费"保 AP。
- 诚实预期:get_multiscale 子模块 PyTorch ~3.3ms,INT8 即便 2× 也只省 ~1.5ms ≈ 新 e2e(~27ms,P0.1 后)的 6%。**P0.2 价值在"让量化轴可评估",非大幅 e2e 提速**。

## 七、复跑

```bash
cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=<idle> \
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  /home/jichengzhi/UniV2X/scripts/phase2/p0_2_build_eval.py --n-samples 300
```
