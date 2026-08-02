# 59_6_29_交接文档_INT8近无损存疑问题_待codex核查

## 0. 本文用途

本文供 **codex(或其他独立审查者)** 独立核查一个存疑结论:

> 当前 native INT8 backbone full-val(1789 帧)测得的 AP **几乎不损失**(与 FP16 差异仅 ±1e-4 量级)。
> 用户直觉认为"INT8 量化几乎无损"不合常理,要求核查此结论是否可信、是否有测量/实现缺陷在制造假象。

本文给出:① 现象;② 公平性前提;③ 我(Claude)的全部调查证据与可复现命令;④ 我中途犯过的判断错误与更正;⑤ 我的当前结论;⑥ **我尚未能独立确认、请 codex 重点核查的不确定点**。

**请审查者不要默认本文结论正确——本文作者已在调查中出现过一次草率误判(见 §4.1),结论可信度本身就是被核查对象。**

---

## 1. 核心存疑现象

INT8 与 FP16 在 4 个已完成 label 上的 AP70 对比(全部 1789 帧 full-val,measurement_source=true_eval):

| label | FP16 AP70 | INT8 AP70 | diff(INT8−FP16) |
|---|---|---|---|
| s0_024 | 0.596145 | 0.596215 | +0.00007 |
| s0_040 | 0.589736 | 0.590200 | +0.00046 |
| s0_056 | 0.587067 | 0.586587 | −0.00048 |
| s1_048 | 0.599228 | 0.599864 | +0.00064 |

差异 ±1e-4 量级,且符号有正有负 → "几乎无损"。这是被质疑点。

数据来源(本地 4090 仓库 `${V2X_ROOT}`):
```
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/
  fp16_true_original60_ap_rows_v1.jsonl        # FP16 AP, 60/60
  native_int8_original60_ap_rows_v1.jsonl      # INT8 AP, 目前 4 行(s0_024/s0_040/s0_056/s1_048), s2_160 在跑
```
复现对比:
```bash
cd ${V2X_ROOT} && python3 - <<'PY'
import json
from pathlib import Path
R=Path('multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows')
fp16={json.loads(l)['label']:json.loads(l) for l in (R/'fp16_true_original60_ap_rows_v1.jsonl').read_text().splitlines() if l.strip()}
i8={json.loads(l)['label']:json.loads(l) for l in (R/'native_int8_original60_ap_rows_v1.jsonl').read_text().splitlines() if l.strip()}
for L in sorted(i8):
    print(L, fp16[L]['metric_value'], i8[L]['metric_value'])
PY
```

---

## 2. 公平性前提(已核实,codex 可复验)

1. **同一 checkpoint,同 digest**: FP16 与 INT8 两条 row 的 `ckpt_path` 均为
   `${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26/net_epoch_bestval_at29.pth`,
   `ckpt_digest` 均为 `74bdf34458d0972f519b57f6862a936a57e4ebeba7087e34b97af1dd9b6d7009`。
   → INT8 为训练后量化(PTQ),不额外训练,因此"finetune 轮次"与 FP16 天然一致。
2. **AP 评测算法一致**: 两条路径都用 opencood 标准评测:
   `build_dataset(hypes, visualize=True, train=False)` → `dataset.post_process(...)` → `eval_utils.caluclate_tp_fp(...)`,
   阈值 0.3/0.5/0.7,1789 帧(DAIR-V2X val)。
   - FP16 脚本: `scripts/stage2_h800_true_fp16_ap_eval.py`
   - INT8 脚本: `scripts/stage2_h800_native_int8_real_activation_bridge.py`(行 ~745-816)
3. **gate**: 4 个 INT8 点均 `status=success`、`processed_samples=1789`、`pred_nonempty_count=1789`、`ap_row_allowed=true`、`full_network_claim=false`。

---

## 3. 关键实现事实(决定结论的核心)

### 3.1 测量"前向路径"两者不同(评测算法相同,执行方式不同)
- **FP16**: 整个 PyTorch 模型 autocast(amp_fp16)一次性 GPU 前向。
- **INT8**: 仅 `pyramid_backbone.get_multiscale_feature` 被 hook 成 `NativeInt8BackboneBridge`,
  走 TVM native int8 graph;**VFE encoder / neck / head / post_process 仍是主进程 PyTorch FP32**。
  (见 bridge 脚本行 ~775-777 的 hook;`full_network_claim=false` 即承认非全网 int8。)

> **含义**: INT8 只量化了 backbone 一段,网络其余部分仍 FP32。这本身就会**低估**整网量化损失。

### 3.2 TIR 真实 dtype 构成(H800 远端,真 lowered 代码)
文件: `.../20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024/s0_024_native_int8_full_onnx_tir_module.txt`

| dtype | 计数 |
|---|---|
| uint8 | 231 |
| int8 | 282 |
| int32 | 817 |
| float16 | 0 |
| **float32** | **3044** |

operator inventory(图标称层): int8=276, int32=154, float32=1。

→ **图标称几乎全 int8,但真实 lowered TIR 中 float32 出现最多(3044)**。
float32 主要出现在 requantize 标量乘法(如 `T.float32(0.046088...)`、`T.float32(128.0)` 这类 scale/zero-point),
即 `int32 累加结果 × float32(scale) + zp → round → clamp → uint8`。

### 3.3 是否真整数 MAC?
- 我**直接阅读 TIR 头部**时见到: 输入 `spatial_features` 为 `uint8`,各 `weight_*_onnx_Conv_*` 为 `int8`,
  `bias_*` 为 `int32`,且卷积累加缓冲 `conv2d_nchw` / `group_conv2d_nchw` 标为 **`int32`**。
  → 据此判断卷积主计算是 `uint8 × int8 → int32 累加` 的**真整数 MAC**,requant 用 float32 scale(标准量化推理流程)。
- ⚠️ **但我在最终自动化 grep(更严正则)时未能复现 conv 累加 dtype 的匹配行**(正则过严),因此该点
  **请 codex 用更宽松方式独立确认**(例如 `grep -nE "conv2d_nchw|group_conv2d_nchw" tir_module.txt | head` 看 buffer dtype)。

### 3.4 无任何 int8 硬件加速内核
对 TIR grep `dp4a / __dp4a / int8x4 / conv2d_NCHWc_int8 / qnn / wmma / tensorcore / mma` → **全部 0 命中**。
→ 即使数值上是 int8,**也没有用 dp4a 等 int8 SIMD 指令加速**,而是普通标量整数 MAC。
这解释了运行时 GPU 利用率几乎 0、且比 FP16 慢约 10 倍(见 §3.5)。

### 3.5 测量极慢 + GPU util≈0 的根因
- bridge **每帧 spawn 一个新的 TVM worker 子进程**(`_run_worker` → `subprocess.run`,
  worker 脚本 `scripts/stage2_native_int8_tvm_worker.py`,bridge 行 494-520),每帧 load .so 跑 1 帧再退出(为 CUDA 隔离)。
- 结果: 单 label 1789 帧约需 3.5–4 小时(对照 FP16 约 10h 跑完 60 个 label,即 ~10min/label)。
- 用户后台观察 GPU util 几乎全程 0、偶尔突发,与"进程 spawn 主导 + 无 int8 加速内核 + batch=1"一致。

复现证据采集(H800: `${V2X_REMOTE_USER}@<PRIVATE_HOST> -p 30001`):
```bash
# /tmp/_evidence.sh 内容见本仓库 scripts 或下方
R=.../20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024
TIR=$R/s0_024_native_int8_full_onnx_tir_module.txt
for dt in uint8 int8 int32 float32; do echo "$dt = $(grep -oE "$dt" $TIR | grep -wc "$dt")"; done
for kw in dp4a int8x4 conv2d_NCHWc_int8 qnn wmma tensorcore; do echo "$kw = $(grep -oiE "$kw" $TIR | wc -l)"; done
```

---

## 4. 我(Claude)的调查过程与一次误判更正

### 4.1 ⚠️ 我犯过的错误(影响可信度评估,如实记录)
我第一轮看到"TIR float32=3044 占主导"就**草率断言"这不是真整数运算、计算走 float32"**。
随后阅读 TIR 头部(input uint8 / weight int8 / conv 累加 int32)后**更正**: 卷积是真整数 MAC,
那 3044 个 float32 是 requant scale,不是卷积主计算。
→ 这次反复说明: 仅凭 dtype 计数下结论不可靠;**请 codex 不要复制我任一中间结论,以原始 TIR 为准。**

### 4.2 我当前(更正后)的结论
1. **数值上是真 int8 量化**(uint8×int8→int32 累加 + float32 requant),**不是 fake/伪量化**。
2. **但不是 fully-integer、也无硬件加速**: requant 用 float32 标量;无 dp4a/tensorcore。
3. **量化范围仅 backbone**: VFE/neck/head 仍 FP32 → 低估整网量化损失。
4. AP≈FP16 的可能解释(多因素,均未被排他性证明):
   - (a) per-tensor scale 标定 + BN-aware 做得当,量化损失本就小;
   - (b) **仅量化 backbone**,损失被 FP32 的其余部分稀释;
   - (c) **模型对 DAIR 过参数化**——本项目已知"剪枝近无损"(见 `CLAUDE.md` 勘误 7),量化近无损可能同源;
   - (d) 协同检测 AP 对 backbone 特征的微小扰动不敏感。

---

## 5. 我尚未能独立确认 / 请 codex 重点核查的点

| # | 待核查问题 | 建议核查方法 |
|---|---|---|
| A | **卷积累加确实是 int32 整数 MAC**(我自动 grep 未复现) | 直接读 `*_tir_module.txt`,确认 `conv2d_nchw`/`group_conv2d_nchw` buffer dtype 与乘加表达式 |
| B | **运行时是否真的执行了 int8 .so**,而非 fallback 到某 FP32 路径 | 查 worker response json(`native_int8_worker_response.json`)的 engine_kind/输出 dtype;或对比 int8 .so 与 fp32 输出数值差异 |
| C | **AP≈FP16 是真无损,还是 backbone 量化在协同 AP 中不敏感** | 设极端量化对照(如故意用过小/过大 scale,或 4-bit),看 AP 是否如预期崩;若崩→说明评测能反映量化损失,当前无损可信;若不崩→评测/范围有问题 |
| D | **per-sample spawn 的 1789 帧 AP 累积是否正确**(det/gt 累积无遗漏/重复) | 对比 bridge 内嵌评测 与 用同 backbone 输出走 FP16 脚本同款评测器的结果 |
| E | **requant 的 float32 scale 是否等效引入了 fp 精度**,使 int8 名不副实 | 检查 scale 是否做了定点化;或对比"int8+fp32 requant" vs "纯 fp32" 的逐层输出 |
| F | **仅 backbone 量化是否 low-ball 了量化损失** | 量化更大范围(neck/head)再测 AP,看损失是否显著上升 |

---

## 6. 关键文件清单(codex 可独立验证)

本地 4090(`${V2X_ROOT}`):
```
multi_agent/data/.../rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/.../rows/native_int8_original60_ap_rows_v1.jsonl
multi_agent/data/.../exports/original60_quant_three_metric_summary_latest.{json,csv,md}
scripts/stage2_h800_native_int8_real_activation_bridge.py     # INT8 评测 bridge(hook 行 ~775, worker spawn 行 494-520, 评测 745-816)
scripts/stage2_h800_true_fp16_ap_eval.py                      # FP16 评测脚本(对照)
scripts/stage2_native_int8_tvm_worker.py                      # 每帧 spawn 的 TVM worker
scripts/stage2_import_native_int8_full_ap_row.py              # AP row 导入(含 gate 校验)
```
H800 远端(`${V2X_REMOTE_USER}@<PRIVATE_HOST> -p 30001`, 同路径前缀 `${V2X_ROOT}`):
```
.../raw/int8_native_route/20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024/
    s0_024_native_int8_full_onnx_tir_module.txt          # 真 lowered TIR(dtype 证据)
    tvm_operator_inventory.json                          # 图标称 dtype
    native_int8_route_manifest.json                      # route 元数据
    runtime_weights_int8.npz                             # int8 权重
    *_tvm_graph.so                                       # 编译产物
    ap_fullval_1789_bnaware_bias_v1_gpu3_20260629/full_ap_eval_report.json   # full-val 报告(ap30/50/70 等)
.../raw/int8_native_route/20260628_s0_024_bnaware_tensor_range_capture_v1_gpu7/
    tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json        # 量化 scale/zp
```

---

## 7. 当前实验状态(交接时)

- INT8 AP measured: **4/60**(s0_024/s0_040/s0_056/s1_048),总表已刷新(`original60_quant_three_metric_summary_latest`)。
- 旧的 1 个 s0_024 INT8 AP row(AP70≈0.0002,old centered-conv route 崩溃残留)已被新 BN-aware row 替换;原 4 行已备份(`*.bak_*` / `*.local_bak_*`)。
- s2_160 第 5 个点的 full-val 仍在 H800 GPU3 运行(交接时约 700/1789 帧)。
- INT8 的 latency / energy 本就 60/60 measured(本次仅涉及 AP 轴)。
- 监控(Monitor)与编排器已停。

## 8. 下一步(待用户/codex 裁决)
1. **若结论被判可信**: 优化测量管线(per-sample spawn → 持久 worker,目标提速至 FP16 量级),再批量补 P4 其余 55 label。
2. **若结论存疑被证伪**: 先修复实现/测量缺陷,再重测已有点。
3. 是否启用真 int8 硬件加速内核(dp4a/tensorcore)是另一独立议题(关系到 latency/energy,不影响 AP 数值)。

---

## 9. 补充(2026-06-29 晚):测量管线优化 + 一个关键的非确定性发现

### 9.1 持久 worker 优化(已实现 + 验证)
为解决 §3.5 的慢/util0,把"每帧 spawn 新进程"改成"持久 worker(一次 load .so/weights/cuda,循环跑)"。
- 实现: bridge 新增 `--persistent-worker`;worker 新增 `--server`(stdin 喂 request 路径)。
  改动文件(本地,尚未覆盖原文件,以 `_v2` 部署在 H800 验证): `scripts/stage2_native_int8_tvm_worker.py` / `scripts/stage2_h800_native_int8_real_activation_bridge.py`。
- 提速(s0_024, 20 帧, GPU4): per-sample 94s vs persistent 30s(含 ~25s 固定 model load);纯推理 ~14×;外推 1789 帧 3.5h → ~8min。

### 9.2 ★ 关键发现: AP 测量有 ~1e-4 的 run-to-run 非确定性
为确认持久 worker 是否引入偏差,做了对照(s0_024, 同 20 帧):

| 对比 | |Δap30| | |Δap50| | |Δap70| |
|---|---|---|---|
| per-sample 跑两次(同模式重复) | 9.8e-05 | 1.6e-04 | 4.4e-05 |
| persistent vs per-sample | 1.1e-04 | 1.0e-04 | 8.8e-05 |

**两者量级相同** → ① 持久 worker 无系统性偏差,与 per-sample 等价(差异落在重复噪声内);
② **AP 评测本身 run-to-run 非确定 ~1e-4**(应来自 head/postprocess 的 PyTorch FP32 cudnn 非确定算法)。

### 9.3 对"INT8 近无损"判断的直接影响(请 codex 重点考虑)
§1 表中 INT8 vs FP16 的差异为 ±7e-5 ~ ±6e-4,**与上面测得的测量噪声 ~1e-4 同量级**。
→ **当前 full-val 的 AP 分辨率不足以把"INT8 量化损失"与"测量非确定噪声"区分开**。
→ "INT8 几乎无损"这一表述,在当前测量精度下**既不能被确证、也不能被否证**;它可能是真无损,也可能是损失被噪声淹没。
→ 建议核查路径(对应 §5 核查点 C): 用一个**预期会显著掉 AP 的极端量化**(如故意破坏 scale / 更激进位宽)作阳性对照——
   若该对照 AP 明显下降(远超 1e-4),说明评测能反映量化损失,则"近无损"可信;
   若极端量化 AP 仍几乎不变,则说明评测/量化范围(仅 backbone)low-ball 了损失,"近无损"是假象。
