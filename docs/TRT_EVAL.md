# TensorRT Acceleration & Evaluation

This document describes how to deploy UniV2X with TensorRT (TRT) for accelerated inference on NVIDIA GPUs, and how to reproduce the end-to-end accuracy results.

**Hardware target**: RTX 4090 (SM 89), CUDA 11.8, TensorRT 10.13
**Validated accuracy**: AMOTA 0.370 (full TRT) vs 0.338 (PyTorch baseline) on V2X-Seq-SPD cooperative val

---

## Table of Contents

1. [TRT Environment Setup](#1-trt-environment-setup)
2. [Build Custom TRT Plugins](#2-build-custom-trt-plugins)
3. [Export ONNX Models](#3-export-onnx-models)
4. [Build TRT Engines](#4-build-trt-engines)
5. [Run TRT Evaluation](#5-run-trt-evaluation)
6. [Accuracy Results](#6-accuracy-results)
7. [Inference Architecture](#7-inference-architecture)

---

## 1. TRT Environment Setup

The TRT pipeline requires a **separate conda environment** from the base UniV2X environment, using a newer PyTorch / CUDA stack.

### a. Create environment

```shell
conda create -n UniV2X_2.0 python=3.9 -y
conda activate UniV2X_2.0
```

### b. Install PyTorch (CUDA 11.8)

```shell
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### c. Install mmdet3d stack

```shell
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmdet==2.28.2 mmsegmentation==0.30.0
pip install mmdet3d==1.0.0rc6
```

### d. Install TensorRT 10.13 (Python wheel)

```shell
pip install tensorrt-cu11==10.13.0.35 \
    --extra-index-url https://pypi.nvidia.com
```

Then extract the C++ headers needed for plugin compilation:

```shell
# Download libnvinfer-headers-dev_10.13.0.35-1+cuda11.8_amd64.deb
# from https://developer.nvidia.com/tensorrt, then:
mkdir -p /tmp/trt_root
dpkg-deb -x libnvinfer-headers-dev_10.13.0.35-1+cuda11.8_amd64.deb /tmp/trt_root_deb
cp -r /tmp/trt_root_deb/usr/include /tmp/trt_root/
# TRT libs are already installed by the pip wheel:
ls $(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))")
```

### e. Install other requirements

```shell
cd UniV2X
pip install -r requirements.txt
pip install onnx onnxruntime scipy scikit-image
```

---

## 2. Build Custom TRT Plugins

UniV2X uses three custom TRT plugins:

| Plugin | Purpose |
|--------|---------|
| `MSDAPlugin` | Multi-scale deformable attention (BEV encoder) |
| `RotatePlugin` | Coordinate rotation (AgentQueryFusion) |
| `InversePlugin` | Batched matrix inversion (LaneQueryFusion) |

### Build

```shell
cd plugins
mkdir -p build && cd build

cmake .. \
    -DTRT_ROOT=/tmp/trt_root \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89   # SM 89 = RTX 4090; change for other GPUs

make -j$(nproc)
# Output: plugins/build/libuniv2x_plugins.so
cd ../..
```

> For other GPU architectures: SM 80 = A100, SM 86 = RTX 3090, SM 87 = Jetson Orin.
> Change `-DCMAKE_CUDA_ARCHITECTURES` accordingly.

---

## 3. Export ONNX Models

All export commands should be run from the UniV2X root directory with the `UniV2X_2.0` environment.

> **Note on V2X-Seq-SPD dataset**: Both ego and infra models use **1 camera** with resolution 1088×1920. Always pass `--num-cam 1 --img-h 1088 --img-w 1920` for this dataset.

### 3.1 BEV Encoder (Phase 1)

```shell
# Ego vehicle (1-cam, 200×200 BEV)
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego --backbone-only \
    --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920 \
    --out onnx/univ2x_ego_bev_encoder_200_1cam.onnx

# Infra roadside unit (1-cam, 200×200 BEV)
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model infra --backbone-only \
    --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920 \
    --out onnx/univ2x_infra_bev_encoder_200_1cam.onnx
```

### 3.2 Detection Head (Phase 2, non-V2X path, 901 queries)

```shell
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego --heads-only --bev-size 200 \
    --out onnx/univ2x_ego_heads_200.onnx
```

### 3.3 Detection Head — V2X path (Phase 3C, 1101 queries)

In the V2X cooperative path, `AgentQueryFusion` can append up to 200 unmatched infra instances to the ego query list (901 + up to 200 = 1101). A separate engine with fixed `N_PAD=1101` is needed:

```shell
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego --v2x-heads --bev-size 200 \
    --out onnx/univ2x_ego_heads_v2x_1101.onnx
```

### 3.4 Downstream Heads (Phase 3, Motion + Occ + Planning)

```shell
# Ego (with planning head)
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego --downstream --bev-size 200 \
    --out onnx/univ2x_ego_downstream.onnx

# Infra (no planning head)
conda run -n UniV2X_2.0 python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model infra --downstream --bev-size 200 \
    --out onnx/univ2x_infra_downstream.onnx
```

---

## 4. Build TRT Engines

All engines are built with `tools/build_trt_downstream.py`:

```shell
# BEV encoders
conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --out  trt_engines/univ2x_ego_bev_encoder_200_1cam.trt

conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_infra_bev_encoder_200_1cam.onnx \
    --out  trt_engines/univ2x_infra_bev_encoder_200_1cam.trt

# Detection heads
conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_heads_200.onnx \
    --out  trt_engines/univ2x_ego_heads_200.trt

conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_heads_v2x_1101.onnx \
    --out  trt_engines/univ2x_ego_heads_v2x_1101.trt

# Downstream heads
conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_downstream.onnx \
    --out  trt_engines/univ2x_ego_downstream.trt

conda run -n UniV2X_2.0 python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_infra_downstream.onnx \
    --out  trt_engines/univ2x_infra_downstream.trt
```

Engine build time is ~5–15 minutes per engine on RTX 4090. Expected sizes:

| Engine | Size |
|--------|------|
| `univ2x_ego_bev_encoder_200_1cam.trt` | ~75 MB |
| `univ2x_infra_bev_encoder_200_1cam.trt` | ~75 MB |
| `univ2x_ego_heads_200.trt` | ~33 MB |
| `univ2x_ego_heads_v2x_1101.trt` | ~34 MB |
| `univ2x_ego_downstream.trt` | ~153 MB |
| `univ2x_infra_downstream.trt` | ~134 MB |

---

## 5. Run TRT Evaluation

`tools/test_trt.py` is the TRT-accelerated evaluation script. TRT modules are injected via **monkey-patch hooks** — the backbone (ResNet-FPN) always runs in PyTorch, while the specified sub-modules run in TRT.

### Hook reference

| Flag | Hook | Module replaced |
|------|------|----------------|
| `--bev-engine-ego` | A | `pts_bbox_head.get_bev_features` (ego BEV encoder) |
| `--bev-engine-inf` | A | `pts_bbox_head.get_bev_features` (infra BEV encoder) |
| `--use-lane-trt` | B | `seg_head.cross_lane_fusion` → `LaneQueryFusionTRT` |
| `--use-agent-trt` | C | `cross_agent_query_interaction` → `AgentQueryFusionTRT` |
| `--heads-engine-ego` | D | `pts_bbox_head.get_detections` (V2X detection head, N_PAD=1101) |

### Step 1 — PyTorch baseline (reference)

```shell
CUDA_VISIBLE_DEVICES=0 bash tools/univ2x_dist_eval.sh \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    1
# Expected: AMOTA=0.338, mAP=0.0727
```

### Step 2 — Hook A only (ego BEV encoder TRT)

```shell
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200_1cam.trt \
    --eval bbox \
    --out output/trt_hook_a.pkl
# Expected: AMOTA=0.381, mAP=0.0766
```

### Step 3 — Hooks A + B + C (BEV TRT + V2X fusion vectorisation)

```shell
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200_1cam.trt \
    --use-lane-trt \
    --use-agent-trt \
    --eval bbox \
    --out output/trt_hooks_abc.pkl
# Expected: AMOTA=0.379, mAP=0.0763
```

### Step 4 — Hooks A + B + C + D (full TRT pipeline)

```shell
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200_1cam.trt \
    --use-lane-trt \
    --use-agent-trt \
    --heads-engine-ego trt_engines/univ2x_ego_heads_v2x_1101.trt \
    --eval bbox \
    --out output/trt_hooks_abcd.pkl
# Expected: AMOTA=0.370, mAP=0.0760
```

### Adding infra BEV TRT (optional)

Once `univ2x_infra_bev_encoder_200_1cam.trt` is built (see Section 4), add:

```shell
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt
```

> **Important**: Do NOT use `univ2x_infra_bev_encoder_200.trt` (6-camera build) with the V2X-Seq-SPD dataset — it will fail due to shape mismatch. Always use the `_1cam` variant.

---

## 6. Accuracy Results

Evaluated on **168 samples** from V2X-Seq-SPD cooperative validation set.

| Configuration | AMOTA | AMOTP | mAP | NDS |
|---------------|-------|-------|-----|-----|
| PyTorch (baseline) | 0.338 | 1.474 | 0.0727 | 0.0679 |
| Hook-A: ego BEV TRT | 0.381 | 1.450 | 0.0766 | 0.0700 |
| Hooks A+B+C: BEV + V2X fusion | 0.379 | 1.441 | 0.0763 | 0.0699 |
| Hooks A+B+C+D: full TRT | **0.370** | 1.446 | 0.0760 | 0.0697 |

**Notes**:
- Hook-A shows higher AMOTA than PyTorch baseline. This is a numerical artefact: TRT FP32 and PyTorch produce slightly different BEV features (cosine similarity 0.9999993), and the downstream nonlinearities happen to produce slightly better detections on this dataset. It does not indicate TRT is "more accurate".
- Hooks A+B+C vs Hook-A: only −0.002 AMOTA, confirming that V2X fusion vectorisation introduces negligible accuracy loss.
- Hook-D (1101-query detection head): −0.009 AMOTA vs A+B+C, caused by zero-padding self-attention contamination in the decoder (see `result.log` for details).

---

## 7. Inference Architecture

```
Infra side:
  Camera image
    → ResNet-FPN backbone        (PyTorch, DCNv2 not ONNX-exportable)
    → BEV encoder                [TRT]  univ2x_infra_bev_encoder_200_1cam.trt
    → Detection head             (PyTorch)
    → PansegformerHead           (PyTorch)
    ──────────────── other_agent_results ────────────────────▶

Ego side:
  Camera image
    → ResNet-FPN backbone        (PyTorch)
    → BEV encoder                [TRT]  univ2x_ego_bev_encoder_200_1cam.trt   ← Hook-A
    → AgentQueryFusionTRT        (vectorised PyTorch, Hook-C)
        CPU: coord transform + ego_selection + vectorised cost matrix + Hungarian
        GPU: MLP align + fuse + complementation  →  901~1101 queries
    → _get_coop_bev_embed        (vectorised PyTorch, index_add_ scatter)
    → Detection head             [TRT]  univ2x_ego_heads_v2x_1101.trt         ← Hook-D
        zero-pad to 1101 → TRT → slice back to N
    → PansegformerHead + LaneQueryFusionTRT  (Hook-B)
    → Downstream heads           [TRT]  univ2x_ego_downstream.trt
        Motion + Occupancy + Planning
    → Trajectory / Occupancy / Planning output
```

### Module summary

| Module | Infra | Ego | Note |
|--------|-------|-----|------|
| ResNet-FPN backbone | PyTorch | PyTorch | DCNv2 blocks ONNX export |
| BEV encoder | TRT | TRT | Hook-A |
| Detection head (non-V2X) | — | TRT (901-query) | `univ2x_ego_heads_200.trt` |
| Detection head (V2X path) | PyTorch | TRT (1101-query) | Hook-D, zero-pad |
| AgentQueryFusion | — | Vectorised PyTorch | Hook-C, eliminates GPU-CPU syncs |
| _get_coop_bev_embed | — | Vectorised PyTorch | `index_add_` scatter, no hook needed |
| LaneQueryFusion | — | Vectorised PyTorch | Hook-B, `forward_trt()` |
| Motion / Occ / Planning | TRT | TRT | Phase 3 |
