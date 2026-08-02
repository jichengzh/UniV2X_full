# 8_7_14 交接文档 — Phase 1 / E6 Pyramid-LiDAR RSU Pareto **v2**(fp16+int8_tc 全真测,1029 已搜)

> 承接 `7_7_14`(Phase 1 启动)。本轮:H800 恢复后主控直接驱动,在 4 个 gold 锚宽度上**真建+真测 fp16 WMMA 与 int8_tc lat/energy**,配 gold 收敛 AP,产**三极值全真测的部署 Pareto**;并对全 1029 空间跑 surrogate 穷举搜索。日期 2026-07-05。

---

## 0. 一句话状态

**Phase 1 / E6 Pareto v2 已交付,fp16 与 int8_tc 全部三轴(lat+energy+AP)真测**。`results/phase1_pyramid_pareto.{csv,json}` + `results/phase1_surrogate_search_1029.json` + `multi_agent/figure/fig_phase1_pyramid_pareto.png`(均真仓库 `${V2X_ROOT}`,主控 stat+真读核验)。**三极值全真测**:AP-max=[64,128,256]fp16、lat-min=E-min=[16,32,64]int8_tc。

---

## 1. 交付物(主控独立核验)

| 文件 | 内容 |
|---|---|
| `results/phase1_pyramid_pareto.json` (7.6KB) | 8 点(4宽×{fp16,int8_tc}全真测)+3 极值+GAP+搜索上下文 |
| `results/phase1_pyramid_pareto.csv` (1.9KB) | 8 点表 (w,q,s,lat,E,AP,pareto,status,source) |
| `results/phase1_surrogate_search_1029.json` (2.9KB) | 全 1029 穷举精确 Pareto 前沿(surrogate) |
| `multi_agent/figure/fig_phase1_pyramid_pareto.png` (121KB) | lat×AP70,色=energy,fp16+int8 双真前沿 |
| `scripts/phase1_assemble_pyramid_pareto.py` / `phase1_surrogate_search_1029.py` | 可复现 |

---

## 2. 口径(§2,统一,可比)

- **lat/energy = H800/TVM tensorcore,输入 [2,64,256,256] 真 AP-shape,batch=2,fresh 逐宽度独立进程,串行独占测**。
  - **fp16 = im2col+WMMA**(`--rewrite-dtype float16 --cast-fp16-source`,`tensorcore_gate=True`,144–180 wmma)。
  - **int8_tc = MatmulInt8Tensorization**(`--rewrite-dtype int8 --cast-fp16-source`,`gate=True`,latency-only dummy scale = **scale 无关的真 int8_tc kernel 延迟**)。
- **AP70 = gold 收敛**(`stage_a_ap_real`,DAIR val n=1789),逐锚点精确;int8 用 gold 真 int8 AP(非 fake-quant)。
- ★[v2 纠三坑]:①shape 双重口径(corpus fp16@256²/int8@128²)不可混轴→全 256² 重建;②fp16 **和** int8 张量化都必须 `--cast-fp16-source`(否则停 float32→gate=False→退化 40+ms);③latency 必须**全机串行独占**测(3 卡并发致同配置 5.93→11.88ms 2× 漂移)。

---

## 3. Pareto v2 结果(fp16+int8_tc 全真测,8 点全 Pareto)

| 宽度 | 精度 | latency | energy | AP70 | int8/fp16 |
|---|---|---|---|---|---|
| [16,32,64] | fp16 | 6.681ms | 0.604J | 0.5300 | — |
| [16,32,64] | int8_tc | **4.527ms** | **0.469J** | 0.5236 | lat 0.678 / E 0.776 |
| [32,64,128] | fp16 | 13.584ms | 1.486J | 0.5641 | — |
| [32,64,128] | int8_tc | 9.374ms | 1.255J | 0.5542 | lat 0.690 / E 0.845 |
| [48,96,192] | fp16 | 17.309ms | 1.967J | 0.5905 | — |
| [48,96,192] | int8_tc | 15.359ms | 1.743J | 0.5841 | lat 0.887 / E 0.886 |
| [64,128,256] | fp16 | 20.500ms | 2.525J | 0.6309 | — |
| [64,128,256] | int8_tc | 18.697ms | 2.168J | 0.6228 | lat 0.912 / E 0.858 |

**三极值点(全真测三轴)**:
- **AP-max** = [64,128,256] fp16:AP70 **0.6309** / 20.50ms / 2.525J。
- **lat-min** = [16,32,64] int8_tc:**4.527ms** / AP70 0.5236 / 0.469J。
- **energy-min** = [16,32,64] int8_tc:**0.469J** / AP70 0.5236 / 4.527ms。

**int8_tc Q 轴(真测)**:同宽度 int8 在 lat/energy 上真实左移;小宽度 int8 增益大(lat 0.68/0.69),最宽处小(0.89/0.91),印证 `[[project-precision-axis-double-unfairness]]` + memory "int8 最宽处不划算"。int8 加速真实(非 corpus 比率派生)。

---

## 4. 全 1029 空间 surrogate 搜索

- **空间** = 343 宽度(W0×W1×W2 各 7 档 {16,24,32,40,48,56,64})× 3 精度 {fp32,fp16,int8_tc} = **1029**。s=内层 tuned。
- **surrogate** = corpus-60 fp16 latency 二次 shape 拟合 + **4 锚真值仿射校准**;int8/fp16 lat/E 比率用 **4 锚真测比率**(线性 w0);fp32=未张量化 default 真比率。AP=gold w0-锚。
- **穷举精确 Pareto over 1029**(比 NSGA-II 采样更强;框架 NSGA-II run_pqs_ablation 是生产采样器):前沿 **n=10,int8_tc=7 / fp16=3 / fp32=0**(int8 占低延迟、fp16 占高 AP=Q 轴 co-design;fp32 全被支配),lat [2.87,10.77]ms,AP [0.522,0.631]。
- ★**重要发现**:surrogate 前沿 = **minimal-neck 配置 [w0,16,16]**(高 w0 高 AP + 最小 neck 低延迟),**支配我真测的对角线锚点 [w0,2w0,4w0]**。即 **[64,16,16]fp16 以 10.77ms 达 AP 0.6309**(vs 对角线 [64,128,256] 20.5ms)。这是 **w0-主导-AP** 先验结论下的**预测**,minimal-neck 的 AP 未 gold 实测。

---

## 5. GAP(诚实标注)

- **G-minimal-neck(最重要,下一步实测)**:1029 surrogate 预测 minimal-neck [w0,16,16] 支配对角线,**前提是 w0-only-AP 在 minimal-neck 处成立**(先验剪枝结论,未 gold 验证)。**下一步真测目标 = minimal-neck 前沿宽度的 AP finetune + lat/energy**,验证后即得比对角线更优的真 Pareto。对角线锚点是**保守的已验证前沿邻域**。
- **G-corpus-caliber**:180-LUT corpus(fp16@256/int8@128,tuned≈default)仅作 cost-model warm-start + surrogate shape 源,不并入统一部署 latency 轴。

★[已闭合] G-int8 直建(v1 的比率派生)已闭合 = int8_tc 现全 4 锚直接真测(`--rewrite-dtype int8 --cast-fp16-source`,gate=True)。

---

## 6. 复现命令

```bash
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
# 1) 导出锚宽度 onnx @256×256
$PY scripts/stage2_original60_export_onnx.py --candidate-queue <q.jsonl> --out-dir <onnx256> --input-hw 256,256
# 2) H800 逐宽度真建 (driver ${V2X_DATA_ROOT}/s2_tvm/knee_g1_run/build_one.sh, CAST 恒开):
#    probe stage2_fp16_tensorcore_convblock_and_engine_probe.py --mode full-engine-group-conv-rewrite
#      --rewrite-dtype {float16|int8} --cast-fp16-source --batch 2 --gpu 0 (CVD=物理卡); 核 gate=True
#    ★串行独占一次一卡
# 3) energy stage2_measure_fp16_rewritten_artifact.py --kind energy --gpu <物理索引,不设CVD> --artifact <so>
# 4) 组装+搜索: $PY scripts/phase1_assemble_pyramid_pareto.py; $PY scripts/phase1_surrogate_search_1029.py
```
H800 = <PRIVATE_HOST>:30001 <LOCAL_USER>(密码八进制永不明文);GPU3/4 空闲(GPU5 常被 user01);TVM `${V2X_DATA_ROOT}/tvm310/bin/python`。

---

## 7. 停止条件核对(全满足)

- [x] Pyramid-LiDAR 完整搜索跑完:全 1029 surrogate 穷举 Pareto + 前沿邻域(8 锚点 fp16+int8_tc)真测;三极值 AP/lat/energy 均 §2 口径真测确认。
- [x] 每点标 (w,q)+s=tuned+AP/lat/energy+来源;无 proxy 冒充真测(int8 lat/energy 直接真测、AP 用 gold 真)。
- [x] Pareto 图+csv+json 落真仓库 fs,主控独立 stat+数值真读核验。

---

## 8. 下一步(精化,非阻塞)

1. **G-minimal-neck 实测**:对 surrogate 前沿的 minimal-neck 宽度([w0,16,16] 系列)做 AP finetune(DAIR val n=1789)+ lat/energy 真测,验证 w0-only-AP 是否在 minimal-neck 成立 → 若成立则真 Pareto 从对角线推进到 minimal-neck(延迟腰斩)。
2. **前沿加密**:补非对角非最小 neck 的中间宽度真测,画出完整 2D 前沿面。
3. **Phase 2**:多模型/camera-Pyramid(留到 Phase 1 完全定稿后)。
