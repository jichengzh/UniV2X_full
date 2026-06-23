# B-Orin.3 流水线工程 plan (v2 修订: 预测器驱动 + 实测优先)

> 目标: 通过 multi-engine runtime 实测 + 扩展 LGB 训练空间,让搜索器**自动发现** Pareto frontier
> 状态: **真做(2026-05-11 启动)**
> v1 → v2 修订理由: v1 走向了"手写硬件约束 C12/C13"的错误路径,违反 framework 的核心论点(hardware-aware 应该 from data,not from rules)
> 关联: `搜索空间更新_plan2.md` Part B-Orin.3 / `results/B_Orin_3_pipeline_validation.md`

---

## 一、修订原则(v2)

### 1.1 框架价值的正确表述

framework 的价值不是 "我们手写了 C12a/b/c 让搜索器避开 DLA 不兼容算子",**而是**:
- bench infrastructure 自动测 (config, hw) → (lat, throughput, build_success)
- LGB 预测器从数据中学规律
- 搜索器(NSGA-II)在 LGB 上找 Pareto

**任何硬件特异知识都该由预测器学,不由 framework 预定义。**

### 1.2 手写约束 vs 预测器学习的边界

| 类型 | 例子 | 谁来管 |
|------|------|--------|
| **结构性物理约束** | 通道对齐(C8)、宽度 pow2(C7)、跨层梯度(C10)、2:4 sparse_tc 强制(C11) | **framework 手写**(`constraints.py`),适用所有硬件 |
| **硬件特异规律** | DLA banks 阈值、算子 whitelist、流水线 throughput 增益条件 | **LGB 预测器从 bench 数据学**,不写代码 |
| **运行时调度** | CUDA Graph on/off、TRT workspace 大小、tactic_sources | 不在搜索空间,部署时默认值 |

---

## 二、Step 1: ONNX 子图分解 — **已完成**

详见 `results/B_Orin_3_pipeline_validation.md` 第一节。产出:

| 文件 | 含义 |
|------|------|
| `onnx_064_128_256_backbone.onnx` | baseline backbone (114 nodes,纯 Conv/Relu/Add) |
| `onnx_032_064_136_backbone.onnx` | prune50 backbone |
| `onnx_016_032_064_backbone.onnx` | prune75 backbone |
| `onnx_016_032_064_collab.onnx` | prune75 collab (含 GridSample/Einsum) |

---

## 三、Step 2: 扩展 LGB 训练数据集(4090 + Orin 并行)— **核心步骤**

### 关键认知: 4090 D 维度同样需要 bench 扩展

`搜索空间一览.md` v4 修订后 D 空间扩到 3 维(D_scheme × D_tactic × D_workspace),**4090 也涨到 15**(虽然 D_scheme 退化为 1,但 D_tactic 5 × D_workspace 3 仍是 15 个 D 配置)。

当前 4090 LGB 训练数据缺口:

| 数据集 | 覆盖维度 | gap |
|--------|---------|-----|
| `data/pyramid_random_bench.parquet` (100 anchor) | B × Q,D 单点(default + 8GB) | LGB 完全没学 D 维度 |
| `data/perstage_quant_bench.parquet` (18 anchor) | per-stage Q,D 单点 | 同上 |
| `data/tactic_workspace_bench.parquet` (8 anchor) | tactic × workspace,但只 1 triplet × 2 prec | **不覆盖 triplet × tactic 交互**,LGB 无法学到剪枝/per-channel 跟 tactic 的耦合 |

→ 不补 4090 bench,LGB 对 4090 D 空间盲,framework 在 4090 上自动 Pareto 失效。

### Step 2a: 4090 D 维度 bench(~60-80 点,0.5 day)

#### Step 2a 采样设计

| 维度 | 采样值 |
|------|--------|
| Triplet | 5: baseline + 4 个 mid/high prune (s2 ∈ {256, 192, 136, 96, 64}) |
| Precision | {FP16, INT8} 2 种 |
| D_scheme | {单 IP GPU} 1 个(4090 无 DLA) |
| **D_tactic** | {default, default+CUDNN, default+CUBLAS_LT, CUBLAS_LT only, all enabled} 5 个 |
| **D_workspace** | {1GB, 4GB, 8GB} 3 个 |

笛卡尔: 5 × 2 × 1 × 5 × 3 = **150 理论点**

**优先级采样 ~60-80 点**:
- 必跑 40: 5 triplet × 2 prec × 4 D_tactic主轴(略 CUBLAS_LT only) × workspace=8GB
- 必跑 20: 选 2 个 sparse / per-channel 配置 × 5 D_tactic × 2 D_workspace
- 选跑 20: T_baseline × 2 prec × 3 ws × 多 tactic(workspace 资源轴)

输出: `data/perstage_quant_bench.parquet`(扩展)+ `data/tactic_workspace_bench_full.parquet`(新)

### 3.1 Step 2b 目标(原 Step 2)

让 LGB 在 Orin 上学到三个目标:
- `lat` (single-frame latency)
- `throughput` (FPS, **真实 multi-engine bench**)
- `build_success` (∈ {0, 1},失败的 config 也作为训练样本)

### 3.2 Step 2b: Orin bench 采样设计(1.5 day)

配置空间(120 数据点,覆盖三个 D 维度):

| 维度 | 采样值 |
|------|--------|
| Triplet | 5 个: baseline + 4 个 mid/high prune(s2 ∈ {256, 192, 136, 96, 64}) |
| Precision | {FP16, INT8} 两种 |
| **D_scheme** | {A: 单 IP GPU, B: 双 IP DLA0+GPU, B': 双 IP DLA1+GPU, C: 三 IP DLA0+DLA1+GPU} 4 种 |
| **D_tactic** | {default, default+CUDNN, default+CUBLAS_LT, all enabled} 4 种(TRT 8.5) |
| **D_workspace** | {1GB, 2GB} 2 种(Orin 内存约束) |

笛卡尔: 5 × 2 × 4 × 4 × 2 = **320 理论点**

**实际优先级采样 ~80-120 点**(由于跑 320 点 ~ 6-8 hour, 用 random_search 抽样):
- 必跑 80 点: 5 triplet × 2 prec × 4 D_scheme × 2 (default tactic × {1GB, 2GB} workspace) — 让 LGB 学到 D_scheme 主轴
- 选跑 40 点: 4 个稀疏量化 + per-channel 配置 × 4 D_tactic(让 LGB 学到 tactic 跟 Q 的耦合)

每个 bench 点产出:
```python
{
    "triplet_sig": "064_128_256",
    "precision": "fp16",
    "d_scheme": "B",
    "build_success": False,   # ← 关键: 失败也记
    "fail_reason": "DLA banks 1024 > 16",
    "lat_ms": None,
    "throughput_fps": None,
}

# 或
{
    "triplet_sig": "016_032_064",
    "precision": "fp16",
    "d_scheme": "B",
    "build_success": True,
    "lat_ms": 36.55,
    "throughput_fps": 45.2,    # ← 实测,不是数学组合
}
```

### 3.3 工程实现

新脚本 `scripts/phase2/p_orin_routing_bench.py`:
1. 输入: 5 triplet × 2 prec × 4 scheme list
2. 对每个 (triplet, prec, scheme):
   - 若 scheme = A: 单 IP GPU,直接 trtexec bench 整图
   - 若 scheme = B/B': 拆 backbone + collab 子图,build 各自 engine,Python TRT multi-engine runtime
   - 若 scheme = C: 进一步拆 backbone 为 stage0/1 vs stage2,build 三 engine,三 stream 并发
3. 记录 build_success / lat / throughput,即使失败也写入数据集
4. 输出: `data/orin_routing_bench.parquet`

---

## 四、Step 3: Python TRT Multi-engine Runtime — **核心工程**

### 4.1 实现要点

```python
# scripts/phase2/multi_engine_runtime.py (新)

import tensorrt as trt
import torch
import time

class MultiEnginePipeline:
    """流水线 runtime: backbone (DLA) + collab (GPU) 双 stream 并发."""

    def __init__(self, backbone_engine_path, collab_engine_path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.backbone_engine = runtime.deserialize_cuda_engine(open(backbone_engine_path,'rb').read())
        self.collab_engine = runtime.deserialize_cuda_engine(open(collab_engine_path,'rb').read())
        self.ctx_bb = self.backbone_engine.create_execution_context()
        self.ctx_cb = self.collab_engine.create_execution_context()
        self.stream_dla = torch.cuda.Stream()
        self.stream_gpu = torch.cuda.Stream()
        # 双缓冲: 当 backbone 处理 frame N+1 时, collab 处理 frame N
        self.bbout_buffers = [self._alloc_bb_out(), self._alloc_bb_out()]

    def bench_throughput(self, n_frames=200):
        """实测稳态 throughput (跳过 priming 阶段)."""
        # ... warm up + main loop
        t0 = time.perf_counter()
        for i in range(n_frames):
            slot = i % 2
            # backbone (DLA) on stream_dla
            with torch.cuda.stream(self.stream_dla):
                self.ctx_bb.execute_async_v3(self.stream_dla.cuda_stream)
            evt = torch.cuda.Event()
            evt.record(self.stream_dla)
            # collab (GPU) on stream_gpu, waiting for backbone slot
            self.stream_gpu.wait_event(evt)
            with torch.cuda.stream(self.stream_gpu):
                self.ctx_cb.execute_async_v3(self.stream_gpu.cuda_stream)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return n_frames / elapsed   # fps
```

### 4.2 实测目标

- 在 Orin 上跑 ~200 帧 warm-up + 200 帧测量
- 输出 throughput (fps) 和 per-frame latency p50/p99

---

## 五、Step 4: 重训 LGB + 自动 Pareto 发现

### 5.1 LGB 训练

输入 features:
```python
features = [
    "stage0_planes", "stage1_planes", "stage2_planes",
    "precision_fp16", "precision_int8",
    # D_scheme one-hot
    "d_scheme_A",      # 单 IP GPU
    "d_scheme_B",      # 双 IP DLA0+GPU
    "d_scheme_B_alt",  # 双 IP DLA1+GPU
    "d_scheme_C",      # 三 IP
    # D_tactic one-hot
    "tactic_default",
    "tactic_with_cudnn",
    "tactic_with_cublas_lt",
    "tactic_all",
    # D_workspace numeric
    "workspace_gb",    # 1 or 2 (Orin) / 1 or 4 or 8 (4090)
]
```

输出 targets:
- `lat_ms` (regression)
- `throughput_fps` (regression)
- `build_success` (binary classification)

三个 booster 独立训练,合并预测时:
```python
def predict(config):
    p_success = model_build_success.predict(config)
    if p_success < 0.5:
        return {"feasible": False}
    lat = model_lat.predict(config)
    fps = model_throughput.predict(config)
    return {"feasible": True, "lat_ms": lat, "throughput_fps": fps}
```

### 5.2 NSGA-II 跑 Pareto

二维目标 (lat, -throughput) 上跑 NSGA-II:
- 输入: 40+ training samples
- 搜索空间: 整个 D_scheme × B × Q
- 输出: Pareto frontier(预期含两支:lat-optimal 单 IP / throughput-optimal 流水线)

预期论文 §C 输出:
> "Without any hardcoded hardware rules, our framework discovered through bench-and-predict that pipelined deployment (D_scheme B/C) is Pareto-optimal in the throughput-constrained subspace, while single-IP GPU is Pareto-optimal in the latency-constrained subspace. The build_success predictor automatically pruned configurations exceeding DLA banks limits."

---

## 六、时间预算(修订)

| Step | 任务 | 时间 |
|------|------|------|
| 1 | ONNX 子图分解(已完成) | ✓ |
| **2a** | **4090 D bench 扩展(~60-80 点,本地)** | **0.5 day(可与 2b 并行)** |
| 2b | Orin D bench 扩展(~80-120 点 + multi-engine throughput) | 1.5 day |
| 3 | Multi-engine Python TRT runtime | 1-2 day(嵌在 2b 内或独立) |
| 4 | 重训 LGB(11 features + hardware one-hot × 3 booster,跨硬件训练) | 0.5 day |
| 5 | NSGA-II Pareto 发现(三维目标: lat, throughput, memory/workspace) | 0.5 day |
| 6 | 论文素材整合 | 0.5 day |
| **合计** | | **4.5-5.5 day**(并行后 = 4-4.5 day) |

---

## 七、与 v1 plan 的关键差异

| 维度 | v1 思路(已废弃) | v2 思路(本计划) |
|------|------------------|------------------|
| C12 算子兼容性 | framework 手写 C12a/b/c per-operator 约束 | 不写代码,LGB build_success 学 |
| C13 DLA banks | 手写阈值 s2 ≤ 136 过滤 | 不写代码,bench 失败样本喂 LGB |
| 流水线 throughput | 数学组合 `1/max(lat)` 估算 | **真实 multi-engine bench** |
| Pareto 发现 | 我观察实验然后手写规律 | NSGA-II 在 LGB 上自动跑 |

---

## 八、本计划开始时的下一步

立即启动 **Step 2a (4090 D bench) 优先**(轻量 0.5 day,完成后可立即喂 LGB 看 4090 端学习效果):
- 写 `scripts/phase2/p_4090_dspace_bench.py` 跑 60-80 个 (triplet × prec × D_tactic × D_workspace)
- 跑完后 dry-run LGB v2 训练,看 4090 D 维度是否被学到

并行 / 后续启动:
- **Step 2b (Orin D bench)**: SSH Orin 跑 80-120 配置
- **Step 3 (Multi-engine runtime)**: 写 Python TRT cuStream runtime 实测流水线 throughput

完成后回报数据 + 让用户决定是否接受 LGB 学到的 Pareto。
