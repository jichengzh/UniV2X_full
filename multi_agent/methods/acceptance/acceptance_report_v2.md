# 验收报告 v2 — 剪枝 DepGraph 全网迁移 / 量化 per-stage 混精 AP / latency 口径

> 验收师 (QA, 收口) 独立复跑验收, 2026-06-01。**不轻信自我报告, 全部实跑/实读/给证据。**
> conda python: `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`
> GPU: 复跑只用 GPU 7 (util 0%, 仅残留 1.6GiB mem); GPU 0-4 全程 100% 占用, **故本轮无干净 latency 基准, 只做功能性 AP/剪枝复跑, 未做 latency 实测**。

---

## 0. TL;DR 判定

| 产物 | 判定 | 一句话 |
|------|------|--------|
| **A — 剪枝 DepGraph 全网迁移** | **PASS** | CPU 复跑端到端通过, 报告 **byte-for-byte 一致**; 全网 build_dependency / 真剪 -53.9% / deblocks·shrink 可剪 / 根因复现 全部实证。|
| **B — 量化 per-stage 混精 AP** | **FAIL (build bug, 非物理现象)** | AP 数字本身真实可复现, 但 "混精<全INT8" 的异常是 **两个 build/方法 bug** 造成, 不是真实量化敏感性: ① c3/c4/c5 的 FP16 分隔符 `;` vs `,` 不匹配 → FP16 意图完全丢失, 实际建成 all-INT8; ② 全部 c3-c8 用 `PREFER_PRECISION_CONSTRAINTS` 强制逐层精度, 与 `global_int8` 基线 (TRT 自由选精度) 不可比。|
| **C — latency 口径** | **PASS** | AP CSV 无 latency 列; 没有任何 method/acceptance 文档把 AP-eval 的 `per_sample_lat_ms` 当 latency 数据点引用。JSON 里的 per-sample lat 是共享 GPU 下副产物, 未被外泄为延迟结论。|

**现在能当"真实测结论"用的**: 产物 A 全部 (DepGraph 全网可剪范围 / 根因)。
**现在还不能用的**: 产物 B 的 18 个 `real_mixed_engine` 行的 `delta_ap50_vs_int8` —— c3/c4/c5 是 mislabeled (实为 all-INT8), c6/c7/c8 虽建对但基线不可比, "per-stage 混精掉点" 这个结论站不住。

---

## A — 剪枝 DepGraph 全网迁移 → **PASS**

### 复跑命令
```bash
cp results/pyramid_prunable_range_real_v1.md /tmp/pyramid_prunable_ORIG.md
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
    tools/configurable/depgraph_pyramid.py --device cpu
diff /tmp/pyramid_prunable_ORIG.md results/pyramid_prunable_range_real_v1.md
```

### 实跑输出 (关键行)
```
[B] tp.DependencyGraph().build_dependency  (整个网络)
    [实测] build_dependency OK — 全网依赖图构建成功
    [实测] 全网依赖组数: 49
    [实测] 全网 Conv2d 层数: 65
    [实测] grouped-conv (groups>1) 层数: 16
[C] params 全网 5,464,023 → 2,516,983 (-53.9%)
    params pyramid_backbone 3,757,635 → 2,078,499 (-44.7%)
    [实测] 剪后 forward OK — outputs=[(1,2,128,128),(1,14,128,128),(1,4,128,128),(1,1,128,128),(1,1,64,64),(1,1,32,32)]
    [实测] grouped-conv 剪后 %32 + in%groups 一致: True
[D] backbone_m1 -3.67% / stage0 -0.0% / stage1 -4.29% / stage2 -23.13% / deblocks -13.58% / shrink_conv -16.24% / all-3-stage -27.27%  (全部 forward OK)
[E] ratio 0.1~0.875 全程 forward OK
[F] 旧 PyramidSubnet wrapper → FAIL "size 32 must match 64";  新 PyramidFullTraceNet → forward OK
```

### 判定证据
- (a) `build_dependency` 真在完整 `HeterPyramidCollab` (backbone_m1 + pyramid ResNeXt + deblocks + single_head + shrink + heads) 上 trace 通 — **实测复现**: 49 组 / 65 Conv / 16 grouped-conv, 与报告一致。
- (b) 全网真剪 (round_to=32, MetaPruner) forward 通, 参数 -53.9% / backbone -44.7%, `grouped_conv_consistent=True` — **实测复现, 数字与报告完全吻合**。
- (c) **deblocks (-13.58%) / shrink_conv (-16.24%) 实测可剪 forward 通** — 推翻"只 backbone 可剪"的关键结论 **复现成立**。
- (d) stage0 -0.0% 因与 backbone_m1 同依赖组耦合 (stride=1 残差 identity) — 复现成立, 解释自洽。
- (e) 根因复现: 旧 method-call 子网 wrapper 真报 `size 32 must match 64`, 新全网 wrapper 真通过 — **实测复现**。
- **`diff` 结果: 重新生成的报告与原报告 byte-for-byte 完全一致 ("IDENTICAL")**。

> caveat: 全程 device=cpu, 与报告口径一致; 这是结构/依赖图验证 (与 GPU 无关), 不涉及 latency, 无需空闲 GPU。

---

## B — 量化 per-stage 混精 AP → **FAIL (build bug)**

### B.1 AP 数字本身真实可复现 (n_samples 全 1789)
- 18 个 `results/perstage_quant_ap/psAP_*.json` 全部 `n_samples=1789, n_trt_collab_path=1618, n_pytorch_fallback=171`。
- **复跑命令** (GPU7, 用已 cache 的 engine 重测 AP):
```bash
CUDA_VISIBLE_DEVICES=7 python scripts/phase1/m4_8_hybrid_infer_ap.py \
  --engine-collab models/perstage_quant_ap_cache/pruned50_c3_I_F_F.engine \
  --tag VERIFY_pruned50_c3 --model-dir .../Pyramid_DAIR_m1_pruned50_2026_05_10 \
  --n-samples 1789 --dataset dair --range 102.4,51.2 \
  --collab-spatial-shape 2,64,128,256 --collab-tego-shape 2,2,3 ...
```
- **结果**: 原 CSV `pruned50_c3` AP50 = **0.728731**, 复跑 = **0.728691** (Δ=4e-5), n_trt_path=1618 / fb=171 与原一致。
- **结论: AP eval 真实、确定性、可复现, 异常不是测量噪声 — 是真异常。**

### B.2 异常根因 — 两个 bug 叠加 (实证)

逐 config 重建 mixed engine, 抓 `[build] mixed (FP16-match): X FP16 / Y INT8` 摘要行 (pruned50, GPU7):

| config | 传入 `--mixed-fp16-substr` | 实际 FP16/INT8 | 建对了? | CSV `delta_ap50_vs_int8`(prune50) |
|--------|----------------------------|----------------|---------|----------------------------------|
| c3 | `/resnet/layer1/;/resnet/layer2/` | **0 / 155** | **否** | -0.0235 |
| c4 | `/resnet/layer0/;/resnet/layer2/` | 0 / 155 (同 bug) | **否** | -0.0240 |
| c5 | `/resnet/layer0/;/resnet/layer1/` | 0 / 155 (同 bug) | **否** | -0.0240 |
| c6 | `/resnet/layer2/` | 57 / 98 | 是 | -0.0249 |
| c7 | `/resnet/layer1/` | 36 / 119 | 是 | -0.0009 |
| c8 | `/resnet/layer0/` | 22 / 133 | 是 | -0.0218 |

**Bug ① — 分隔符不匹配 (硬 bug, c3/c4/c5)**:
- 驱动 `scripts/phase2/p0_perstage_quant_AP_real.py` 把多个 FP16-keep 子串用 **分号 `;`** 连接 (e.g. `"/resnet/layer1/;/resnet/layer2/"`)。
- 但 bench 脚本 `scripts/phase1/m4_8_trt_build_bench.py:544` 用 **逗号 `,`** 切分:
  `globals()["_MIXED_FP16_SUBSTRS"] = [s.strip() for s in args.mixed_fp16_substr.split(",")]`
- 结果整串 `/resnet/layer1/;/resnet/layer2/` 被当成**一个含字面分号的子串**, 匹配 **0 层** → `0 FP16 / 155 INT8`。
- 即 **c3/c4/c5 三行根本不是它们声称的 per-stage 混精, 而是"全层强制 INT8" (`PREFER_PRECISION_CONSTRAINTS` 锁死, 连 TRT 自动 FP16 回退都被禁)**。CSV 里它们标 `INT8/FP16/FP16` 等是**错误标签**。

**Bug ② — 强制约束 vs 自由选精度, 基线不可比 (方法 bug, c3-c8 全部)**:
- `global_int8` 基线 (来自 `scripts/phase2/stage_a_ap_real.py`, `--precision int8`): 只 set INT8+FP16 flag, **不设任何逐层精度** → TRT 自由地为每层挑 INT8/FP16 以最优精度 (敏感层可自动留 FP16)。
- mixed 路径 (`m4_8_trt_build_bench.py:343` `_set_layer_precision` + `PREFER_PRECISION_CONSTRAINTS`): **强制每层**要么 INT8 要么 FP16。即使 c6/c7/c8 建对了 FP16 分配, 也是把 98~133 层**强制** INT8。
- 因此 "mixed vs global_int8" **不是同类比较**: global_int8 实质是"TRT 自动混精 (更聪明)", per-stage 是"人工强约束 (更激进)"。pruned 档冗余少 → 强制 INT8 掉点更明显 (base 档普遍不掉)。c7 仅 -0.0009 的非单调性也佐证: 掉点取决于**哪些层** INT8, 是真实量化敏感性, 但被错误的基线放大成"全面掉点"。

**两个口径都同用 minmax 校准器 + 同一 calib data (`pyramid_dair_collab_spatial/tego.npy`)** — 已核对, **校准器不是差异来源** (排除了 entropy 崩塌的可能)。差异 100% 来自精度约束机制。

### B.3 修复建议
1. **立即修分隔符**: 驱动改用逗号, 或 bench 脚本同时支持 `[;,]` 切分 (`re.split(r"[;,]", ...)`)。修后 c3/c4/c5 才是真 per-stage 混精。
2. **基线对齐**: 要么把 `global_int8` 也用 `PREFER_PRECISION_CONSTRAINTS` 全层强制 INT8 (与 mixed 同口径), 要么把 per-stage 也改成"只对目标 stage 设约束、其余层不约束让 TRT 自由选" —— 二选一, 否则 delta 无意义。
3. 重跑后再下"per-stage 混精有无 Pareto 价值"的结论。**当前 18 行 delta 全部作废**。

---

## C — latency 口径 → **PASS**

- `results/perstage_quant_AP_real_v1.csv` 表头无任何 `lat/ms/latency` 列 — **AP CSV 不含延迟数据** ✅。
- 18 个 AP JSON 里确有 `mean/p50/p99_per_sample_lat_ms` 字段, 但:
  - 这些是在 **GPU 共享 (AP eval 占用中)** 下采的, p99 高达 295~488ms (明显非干净基准), 是评测副产物。
  - grep 全 `multi_agent/` + `paper_learning/` + `results/perstage*`: **没有任何 md/method/acceptance 文档把这些 per-sample lat 当 latency 数据点引用** ✅ (匹配到的都是 AP CSV 的 `elapsed_secs` 总耗时列, 非延迟结论)。
- 本轮 QA 复跑**未做任何 latency 实测** (GPU 0-4 全 100%, 5/6/7 有残留 mem, 不满足"util 0% / mem≤50MiB"空闲门槛) — 守住了用户硬性口径。

> 注: 真正被当 latency 用的数据点来自 `m4_8_trt_build_bench.py` 的 build JSON (CUDA-Event, warmup/measure), 不在本批 B 产物范围内, 其空闲性需另案核 (见 tier1 CSV 标注 "GPU5/6")。

---

## 4. 一句话收口

- **A (剪枝全网可剪范围) 可直接当真实测结论用** — byte-for-byte 复现。
- **B (per-stage 混精掉点) 不能用** — 是分隔符 bug + 不可比基线造成的伪现象, 非真实量化敏感性; AP 数字虽真, 但 delta 解读错误, 修两个 bug 后须重跑。
- **C latency 口径干净** — 无人把脏 GPU 下的 per-sample lat 当延迟结论外泄。
