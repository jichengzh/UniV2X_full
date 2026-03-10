# Phase 1：基础设施复用实现计划

> 对应 `plan.md` 第六章 Phase 1，目标：将 uniad-trt 中与 UniV2X 同构的 TRT 插件和 ONNX 导出适配模块完整迁移，最终产出 **两个独立 ONNX 文件 + 两个独立 TRT engine**，分别对应 ego 车辆侧和基础设施侧，为分离部署做好准备。
>
> **环境确认**：
> - TensorRT 版本：**TRT 10.x**（待安装到当前主机）
> - 目标部署平台（验证阶段）：**本机 RTX 4090**（SM 架构 = 89，显存 24GB）
> - 验收标准：完整 TRT engine 构建 + 精度对比（非仅 ONNX 合法性检查）
> - **核心架构约束**：ego 侧和基础设施侧**必须分别**导出独立 ONNX 和 TRT engine，不可合并为单一模型
>
> **Phase 1 两阶段验证策略**（无 50×50 专用 checkpoint，不可直接用 200×200 权重缩小 BEV）：
> - **Step A — 图结构验证**：使用**随机初始化权重**（不加载任何 checkpoint）+ **50×50 BEV**，快速完成 ONNX 导出和 TRT engine 编译，验证计算图中 MSDAPlugin / RotatePlugin / InversePlugin 节点正确生成且 TRT 可识别
> - **Step B — 精度验证**：切换至 **200×200 BEV + 现有真实 checkpoint**（4090 显存 24GB 应可承载 backbone 导出），对比 PyTorch 输出与 TRT engine 推理输出误差，完成正式验收

---

## 一、两个项目库分析

### 1.1 uniad-trt 项目结构

```
uniad-trt/
├── patch/
│   ├── uniad-onnx-export.patch       # 核心：所有 *TRT 模块的 Python 实现
│   ├── bevformer_tensorrt.patch      # MSDAPlugin symbolic + DCN + FFNTRT 适配
│   ├── plugins-trt10-support.patch   # TRT 插件 TRT8→TRT10 兼容性补丁
│   └── mmdet3d.patch                 # mmdet3d 内部 API 暴露补丁
├── inference_app/
│   └── enqueueV3/                    # C++ 推理框架（TRT 10 enqueueV3 接口）
├── tools/
│   ├── export_onnx.py                # ONNX 导出主脚本
│   └── add_bevformer_tensorrt_support.sh
└── documents/
    └── proj_setup.md                 # 项目搭建文档（说明插件来源）
```

**插件来源**：TRT 插件（C++ 代码）来自 `BEVFormer_tensorrt` 仓库（通过 git submodule）。uniad-trt 自身仅提供 patch，不直接包含插件 C++ 源码。在 DL4AGX 同目录的 `vad-trt` 项目包含了可用的参考实现：

```
vad-trt/plugins/
├── multi_scale_deform_attn/    # MultiScaleDeformableAttentionPlugin 完整实现
│   ├── ms_deform_attn.h / .cpp
│   ├── ms_deform_attn_kernel.cu / .cuh / .hpp
│   └── common.h
├── rotate/                     # RotatePlugin 完整实现
│   ├── rotate_plugin.h / .cpp
│   └── rotateKernel.cu / .h
└── CMakeLists.txt              # 插件构建脚本
```

InverseTRT 插件来自 `BEVFormer_tensorrt/TensorRT/plugin/inverse/`，可通过克隆该仓库获取。

### 1.2 UniV2X 目标项目结构

```
UniV2X/projects/mmdet3d_plugin/univ2x/
├── modules/
│   ├── encoder.py                          # BEVFormerEncoder + BEVFormerLayer
│   ├── spatial_cross_attention.py          # SpatialCrossAttention + MSDeformableAttention3D
│   ├── temporal_self_attention.py          # TemporalSelfAttention
│   ├── decoder.py                          # DetectionTransformerDecoder + CustomMSDeformableAttention
│   ├── transformer.py                      # PerceptionTransformer
│   └── multi_scale_deformable_attn_function.py  # 底层 CUDA 调用（需扩充 MSDAPlugin）
├── dense_heads/
│   ├── track_head.py                       # BEVFormerTrackHead
│   └── ...
├── detectors/
│   ├── univ2x_track.py                     # UniV2XTrack
│   └── univ2x_e2e.py                       # UniV2X
└── ...

projects/mmdet3d_plugin/core/bbox/
└── util.py                                 # normalize_bbox / denormalize_bbox（需扩充 TRT 版本）
```

### 1.3 关键差异表

| 对比维度 | uniad-trt | UniV2X | 影响 |
|---|---|---|---|
| 项目角色 | 提供 patch + 工具，依赖上游仓库 | 独立完整项目 | 插件需从 vad-trt/BEVFormer_tensorrt 引入 |
| BEV 尺寸 | 50×50（tiny 版） | **200×200**（生产）/ **50×50**（Phase 1 验证） | Phase 1 先用 50×50 跑通流程，再切 200×200 |
| 图像分辨率 | 256×416（tiny 版） | **原始大尺寸**（待配置） |  |
| 模型入口 | `cfg.model`（单模型） | `cfg.model_ego_agent` + `cfg.model_other_agent_inf`（双模型） | **两个模型分别独立导出**，Phase 1 先各自验证 |
| MSDA 函数文件 | `functions/multi_scale_deformable_attn.py` | `modules/multi_scale_deformable_attn_function.py` | 路径不同，导入需适配 |
| `inverse` 函数 | `functions/inverse.py`（独立文件） | 不存在 | Phase 1 需新增 |
| `rotate` 函数 | `functions/rotate.py`（独立文件） | 不存在 | Phase 1 需新增 |

---

## 二、Phase 1 任务分解

Phase 1 共 5 个子任务，按依赖顺序执行：

```
T1: 建立 plugins/ 目录，引入三个 TRT 插件 C++ 源码
T2: 新增 Python 侧 functions/ 目录，添加 MSDAPlugin / InversePlugin / RotatePlugin symbolic 函数
T3: 添加 TRT 工具函数（custom_torch_atan2_trt / denormalize_bbox_trt）
T4: 在 modules/ 中添加 BEVFormer*TRT 变体类
T5: 编译插件 + ONNX 导出 + 构建完整 TRT engine + 精度对比验证（ego 和基础设施侧各一份）
```

---

## 三、详细实现步骤

### T1：建立 plugins/ 目录，引入三个 TRT 插件

#### T1.1 目录结构规划

```
UniV2X/
└── plugins/
    ├── CMakeLists.txt                       # 新建
    ├── multi_scale_deform_attn/             # 从 vad-trt/plugins 复制
    │   ├── common.h
    │   ├── ms_deform_attn.h
    │   ├── ms_deform_attn.cpp
    │   ├── ms_deform_attn_kernel.cu
    │   ├── ms_deform_attn_kernel.cuh
    │   └── ms_deform_attn_kernel.hpp
    ├── rotate/                              # 从 vad-trt/plugins 复制
    │   ├── common.h
    │   ├── rotate_plugin.h
    │   ├── rotate_plugin.cpp
    │   ├── rotateKernel.cu
    │   └── rotateKernel.h
    └── inverse/                             # 从 BEVFormer_tensorrt 克隆后复制
        ├── inversePlugin.h
        └── inversePlugin.cpp
```

#### T1.2 执行命令

```bash
# 创建目录
mkdir -p /home/jichengzhi/UniV2X/plugins

# 复制 MSDA 和 Rotate 插件（vad-trt 已有完整实现）
cp -r /home/jichengzhi/DL4AGX/AV-Solutions/vad-trt/plugins/multi_scale_deform_attn \
      /home/jichengzhi/UniV2X/plugins/
cp -r /home/jichengzhi/DL4AGX/AV-Solutions/vad-trt/plugins/rotate \
      /home/jichengzhi/UniV2X/plugins/

# 获取 InverseTRT 插件
cd /tmp
git clone https://github.com/DerryHub/BEVFormer_tensorrt.git --depth=1
cp -r /tmp/BEVFormer_tensorrt/TensorRT/plugin/inverse /home/jichengzhi/UniV2X/plugins/
```

#### T1.3 编写 plugins/CMakeLists.txt

基于 `vad-trt/plugins/CMakeLists.txt` 修改，适配 UniV2X 环境（CUDA 11.8，x86_64）：

```cmake
cmake_minimum_required(VERSION 3.16)
project(univ2x_plugins LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O2")
set(CMAKE_CXX_FLAGS "-fPIC -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr \
    --compiler-options -fPIC -O2")

# 配置路径（根据实际安装位置调整）
option(TRT_ROOT "Path to TensorRT" "/usr/local/tensorrt")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

find_package(CUDA REQUIRED)

# GPU 架构（A100=80, RTX3090=86, RTX4090=89）
# 验证平台：RTX 4090，SM 架构 = 89
set(ARCH 89)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TRT_ROOT}/include
    ${CMAKE_SOURCE_DIR}/multi_scale_deform_attn
    ${CMAKE_SOURCE_DIR}/rotate
    ${CMAKE_SOURCE_DIR}/inverse)

link_directories(${TRT_ROOT}/lib)

# CUDA kernel 静态库
set(plugins_cu_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotateKernel.cu)

cuda_add_library(plugins_cu STATIC ${plugins_cu_srcs} OPTIONS -arch=sm_${ARCH})
set_target_properties(plugins_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 最终动态库
add_library(univ2x_plugins SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotate_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/inverse/inversePlugin.cpp)

target_link_libraries(univ2x_plugins plugins_cu nvinfer ${CUDA_LIBRARIES} cublas)
```

---

### T2：新增 Python functions/ 目录，添加 Plugin symbolic 函数

UniAD 有独立的 `functions/` 目录存放 ONNX symbolic 函数，UniV2X 没有。需新建：

```
UniV2X/projects/mmdet3d_plugin/univ2x/functions/
├── __init__.py
├── multi_scale_deformable_attn.py   # 添加 _MSDAPlugin + MSDAPlugin()
├── inverse.py                        # 新增 InversePlugin symbolic
└── rotate.py                         # 新增 RotatePlugin symbolic
```

#### T2.1 `functions/multi_scale_deformable_attn.py`

在现有 `modules/multi_scale_deformable_attn_function.py` 基础上，新增 `_MSDAPlugin` 类和 `MSDAPlugin()` 函数。`_MSDAPlugin.symbolic()` 向 ONNX 图注册名为 `"MSDAPlugin"` 的自定义算子节点，`forward()` 复用现有 `ext_module.ms_deform_attn_forward` 实现（与原有逻辑一致，仅增加 symbolic 路径）。

来源：`uniad-trt/patch/bevformer_tensorrt.patch` 中 `_MSDAPlugin` 类定义（约 90 行）直接迁移，仅需将导入路径从 `from mmcv.utils import ext_loader` 调整为 UniV2X 的既有导入方式。

#### T2.2 `functions/inverse.py`

`InversePlugin` 的 Python symbolic 函数，向 ONNX 图注册名为 `"InversePlugin"` 的节点，`forward()` 直接调用 `torch.linalg.inv`（推理时走 TRT 插件，训练时走 PyTorch 原生实现）。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `from projects.mmdet3d_plugin.uniad.functions import inverse` 对应的 `inverse.py` 文件内容。

#### T2.3 `functions/rotate.py`

`RotatePlugin` 的 Python symbolic 函数，向 ONNX 图注册名为 `"RotatePlugin"` 的节点，`forward()` 调用 `torchvision.transforms.functional.rotate` 或等价实现。

来源：`uniad-trt/patch/bevformer_tensorrt.patch` 中 `rotate.py` 的修改部分。

#### T2.4 `functions/__init__.py`

```python
from .multi_scale_deformable_attn import MSDAPlugin, _MSDAPlugin
from .inverse import inverse, InversePlugin
from .rotate import rotate, RotatePlugin
```

---

### T3：添加 TRT 工具函数

修改文件：`projects/mmdet3d_plugin/core/bbox/util.py`

在现有 `normalize_bbox` / `denormalize_bbox` 函数之后追加两个 TRT 兼容版本：

#### T3.1 `custom_torch_atan2_trt`

TRT 不支持 `torch.atan2`（ONNX 算子不完整）。替换为基于条件判断的手动实现：

```python
import math

def custom_torch_atan2_trt(y, x):
    """TRT 兼容的 atan2 实现，替代 torch.atan2。
    参考: https://en.wikipedia.org/wiki/Atan2
    """
    eps = 1e-8
    atan = torch.atan(y / (x + eps))
    pi_div_2    = torch.ones_like(atan) * (math.pi / 2)
    neg_pi_div_2 = torch.ones_like(atan) * (-math.pi / 2)

    x_eq_0 = (x == 0)
    x_gt_0 = (x > 0)
    x_ls_0 = (x < 0)
    y_ge_0 = (y >= 0)
    y_gt_0 = (y > 0)
    y_ls_0 = (y < 0)

    atan2 = (neg_pi_div_2) * (x_eq_0 & y_ls_0).int() \
          + (pi_div_2)     * (x_eq_0 & y_gt_0).int() \
          + (atan - math.pi) * (x_ls_0 & y_ls_0).int() \
          + (atan + math.pi) * (x_ls_0 & y_ge_0).int() \
          + (atan)           *  x_gt_0.int()
    return atan2.float()
```

来源：`uniad-trt/patch/uniad-onnx-export.patch` `core/bbox/util.py` 修改部分，直接复制。

#### T3.2 `denormalize_bbox_trt`

现有 `denormalize_bbox` 内部调用 `torch.atan2`，需替换为 `custom_torch_atan2_trt`。新增 TRT 版本不覆盖原有函数（避免影响训练逻辑）：

```python
def denormalize_bbox_trt(normalized_bboxes, pc_range):
    """TRT 兼容的 bbox 反归一化，用 custom_torch_atan2_trt 替代 torch.atan2。"""
    rot_sine   = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = custom_torch_atan2_trt(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    w  = normalized_bboxes[..., 2:3].exp()
    l  = normalized_bboxes[..., 3:4].exp()
    h  = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized
```

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `denormalize_bbox_trt` 及 `if_normal_bboxes_size` 函数，合并适配。

---

### T4：添加 BEVFormer*TRT 变体类

Phase 1 仅实现 BEVFormer 骨干部分的 TRT 变体（不含 head）。各类均继承对应的 PyTorch 基类，**仅重写 `forward` 方法**使其调用 TRT 兼容版本的子模块（用 `MSDAPlugin`、`InversePlugin`、`RotatePlugin` 替换原生 CUDA 调用）。

#### T4.1 在 `modules/multi_scale_deformable_attn_function.py` 末尾追加

```python
# ── TRT 导出路径 ──────────────────────────────────────────────
from projects.mmdet3d_plugin.univ2x.functions import MSDAPlugin
_MSDAPlugin_gpu = MSDAPlugin  # alias，保持与 UniAD 命名一致
```

#### T4.2 在 `modules/spatial_cross_attention.py` 末尾追加

新增 `MSDeformableAttention3DTRT` 和 `SpatialCrossAttentionTRT` 两个类：

- `MSDeformableAttention3DTRT(MSDeformableAttention3D)`：重写 `forward()`，将内部的 `multi_scale_deformable_attn(...)` 调用替换为 `MSDAPlugin(...)` 调用。
- `SpatialCrossAttentionTRT(SpatialCrossAttention)`：重写 `forward()`，调用 `MSDeformableAttention3DTRT` 实例进行前向传播，其余逻辑与父类完全一致。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `SpatialCrossAttentionTRT`（约第 5962 行起）及 `MSDeformableAttention3DTRTP` 类，需将类名前缀从 `TRTP` 改为 `TRT` 并简化，去除 UniAD 独有的 `img_metas_scene_token` 参数。

#### T4.3 在 `modules/temporal_self_attention.py` 末尾追加

新增 `TemporalSelfAttentionTRT(TemporalSelfAttention)`：

- 重写 `forward()`，将内部的：
  - `torch.inverse(...)` → `InversePlugin(...)`
  - `rotate(...)` → `RotatePlugin(...)`
  - `multi_scale_deformable_attn(...)` → `MSDAPlugin(...)`

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `TemporalSelfAttentionTRT`（约第 6737 行起），约 150 行，直接复制后调整导入路径。

#### T4.4 在 `modules/encoder.py` 末尾追加

新增 `BEVFormerLayerTRT(BEVFormerLayer)` 和 `BEVFormerEncoderTRT(BEVFormerEncoder)`：

- `BEVFormerLayerTRT`：重写 `forward()`，将子模块中的 SCA 和 TSA 替换为 TRT 变体版本的调用。
- `BEVFormerEncoderTRT`：重写 `forward()`，循环调用 `BEVFormerLayerTRT`。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `BEVFormerEncoderTRT` 和 `BEVFormerLayerTRT`（约第 5257 行起），共约 400 行。

#### T4.5 注册到 mmdet3d Registry

在各文件末尾确保 `@ATTENTION.register_module()` / `@TRANSFORMER_LAYER_SEQUENCE.register_module()` 装饰器正确注册新类，使 config 可通过 `type="BEVFormerEncoderTRT"` 等字符串引用。

#### T4.6 更新各模块的 `__init__.py`

在 `modules/__init__.py` 中追加导出：

```python
from .encoder import BEVFormerEncoder, BEVFormerLayer, BEVFormerEncoderTRT, BEVFormerLayerTRT
from .spatial_cross_attention import (SpatialCrossAttention, MSDeformableAttention3D,
                                       SpatialCrossAttentionTRT, MSDeformableAttention3DTRT)
from .temporal_self_attention import TemporalSelfAttention, TemporalSelfAttentionTRT
```

---

### T5：编译验证与 ONNX 导出测试

#### T5.1 编译 TRT 插件

```bash
cd /home/jichengzhi/UniV2X/plugins
mkdir build && cd build
cmake .. \
    -DTRT_ROOT=/usr/local/tensorrt \
    -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 4090 SM 89
make -j$(nproc)
# 期望输出：libuniv2x_plugins.so
```

#### T5.2 编写 ONNX 导出脚本（支持双模型）

新建 `tools/export_onnx_univ2x.py`，基于 `uniad-trt/tools/export_onnx.py` 适配，主要改动：

1. 支持通过 `--model` 参数选择导出 `ego`（`cfg.model_ego_agent`）或 `infra`（`cfg.model_other_agent_inf`）
2. **Phase 1 BEV 尺寸：`bevh = 50`**（`prev_bev=[2500,1,256]`），后续切换为 200 时仅改此参数
3. Phase 1 仅导出 **BEVFormer 骨干部分**（`get_bevs()`），不导出完整模型

```python
# Phase 1 验证用：仅导出 BEVFormer 编码器（ego 和 infra 共用同一 Wrapper）
class BEVBackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pts_bbox_head = model.pts_bbox_head  # BEVFormerTrackHead

    def forward(self, img, img_metas_can_bus, img_metas_lidar2img,
                prev_bev, use_prev_bev):
        return self.pts_bbox_head.get_bev_features_trt(
            img, img_metas_can_bus, img_metas_lidar2img,
            prev_bev, use_prev_bev)
```

**Step A：随机权重 + 50×50，验证图结构和 TRT 编译**

```bash
# ego 侧（随机初始化，不加载 checkpoint）
CUDA_VISIBLE_DEVICES=0 python3 tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
    --model ego \
    --random-weights \
    --backbone-only \
    --bev-size 50 \
    --out onnx/univ2x_ego_bev_backbone_50_rand.onnx

# 基础设施侧（随机初始化，不加载 checkpoint）
CUDA_VISIBLE_DEVICES=0 python3 tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
    --model infra \
    --random-weights \
    --backbone-only \
    --bev-size 50 \
    --out onnx/univ2x_infra_bev_backbone_50_rand.onnx
```

**Step B：真实 checkpoint + 200×200，正式精度验收**

```bash
# ego 侧（真实权重，200×200）
CUDA_VISIBLE_DEVICES=0 python3 tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
    ckpts/univ2x_coop_e2e_stg1.pth \
    --model ego \
    --backbone-only \
    --bev-size 200 \
    --out onnx/univ2x_ego_bev_backbone_200.onnx

# 基础设施侧（真实权重，200×200）
CUDA_VISIBLE_DEVICES=0 python3 tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
    ckpts/univ2x_coop_e2e_stg1.pth \
    --model infra \
    --backbone-only \
    --bev-size 200 \
    --out onnx/univ2x_infra_bev_backbone_200.onnx
```

#### T5.3 创建 TRT 推理验证配置文件

新建 `projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py`，在现有 `univ2x_coop_e2e_track.py` 基础上将 BEVFormer 骨干相关模块替换为 TRT 变体（**对 `model_ego_agent` 和 `model_other_agent_inf` 均做相同替换**）：

```python
# 对 ego_agent 和 other_agent_inf 中的 BEVFormer 模块统一替换：
type="BEVFormerEncoderTRT",    # 原 BEVFormerEncoder
type="BEVFormerLayerTRT",      # 原 BEVFormerLayer
type="TemporalSelfAttentionTRT",   # 原 TemporalSelfAttention
type="SpatialCrossAttentionTRT",   # 原 SpatialCrossAttention
type="MSDeformableAttention3DTRT", # 原 MSDeformableAttention3D
```

---

## 四、迁移效果验证

### 4.1 单元验证——插件功能正确性

```bash
cd /home/jichengzhi/UniV2X
conda activate UniV2X_2.0

# 验证 MSDA 插件 Python symbolic 注册正常
python3 -c "
from projects.mmdet3d_plugin.univ2x.functions import MSDAPlugin
import torch
print('MSDAPlugin import OK')
"

# 验证 TRT 工具函数
python3 -c "
from projects.mmdet3d_plugin.core.bbox.util import custom_torch_atan2_trt, denormalize_bbox_trt
import torch
y = torch.tensor([1.0, -1.0, 0.0])
x = torch.tensor([1.0,  1.0, 0.0])
result = custom_torch_atan2_trt(y, x)
ref    = torch.atan2(y, x)
assert torch.allclose(result, ref, atol=1e-5), f'atan2 mismatch: {result} vs {ref}'
print('custom_torch_atan2_trt OK, max error:', (result - ref).abs().max().item())
"
```

### 4.2 模块导入验证——TRT 变体类注册正常

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from projects.mmdet3d_plugin.univ2x.modules.encoder import BEVFormerEncoderTRT, BEVFormerLayerTRT
from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import SpatialCrossAttentionTRT
from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import TemporalSelfAttentionTRT
print('All TRT module classes import OK')

# 验证 mmcv registry 注册
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER_SEQUENCE
assert 'BEVFormerEncoderTRT'       in TRANSFORMER_LAYER_SEQUENCE._module_dict
assert 'SpatialCrossAttentionTRT'  in ATTENTION._module_dict
assert 'TemporalSelfAttentionTRT'  in ATTENTION._module_dict
print('Registry check OK')
"
```

### 4.3 前向推理一致性验证——TRT 变体与原版输出对齐

```bash
python3 - << 'EOF'
import torch, sys
sys.path.insert(0, '.')
from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import (
    TemporalSelfAttention, TemporalSelfAttentionTRT)

# 构造最小化测试输入
B, N, C = 1, 40000, 256
query = torch.randn(B, N, C).cuda()
# ... (构造 img_metas 等必要输入)

tsa    = TemporalSelfAttention(embed_dims=256).cuda().eval()
tsa_trt = TemporalSelfAttentionTRT(embed_dims=256).cuda().eval()
tsa_trt.load_state_dict(tsa.state_dict())

with torch.no_grad():
    out     = tsa(query, ...)
    out_trt = tsa_trt(query, ...)

max_err = (out - out_trt).abs().max().item()
print(f'TemporalSelfAttention output max error: {max_err:.2e}')
assert max_err < 1e-4, f'Error too large: {max_err}'
print('TSA TRT consistency check PASSED')
EOF
```

### 4.4 ONNX 导出验证

**Step A**：验证随机权重 50×50 ONNX 图结构（确认 Plugin 节点存在）：

```bash
for ONNX_FILE in onnx/univ2x_ego_bev_backbone_50_rand.onnx \
                 onnx/univ2x_infra_bev_backbone_50_rand.onnx; do
    python3 -c "
import onnx
model = onnx.load('${ONNX_FILE}')
onnx.checker.check_model(model)
nodes = [n.op_type for n in model.graph.node]
assert 'MSDAPlugin'    in nodes, 'MSDAPlugin not found'
assert 'RotatePlugin'  in nodes, 'RotatePlugin not found'
assert 'InversePlugin' in nodes, 'InversePlugin not found'
print(f'Step A PASSED: {set(n for n in nodes if \"Plugin\" in n)} in ${ONNX_FILE}')
"
done
```

**Step B**：验证真实权重 200×200 ONNX 合法性：

```bash
for ONNX_FILE in onnx/univ2x_ego_bev_backbone_200.onnx \
                 onnx/univ2x_infra_bev_backbone_200.onnx; do
    python3 -c "
import onnx
model = onnx.load('${ONNX_FILE}')
onnx.checker.check_model(model)
nodes = [n.op_type for n in model.graph.node]
assert 'MSDAPlugin'    in nodes
assert 'RotatePlugin'  in nodes
assert 'InversePlugin' in nodes
print(f'Step B PASSED: ${ONNX_FILE}')
"
done
```

### 4.5 TRT engine 构建验证（Step A：图结构验证）

使用 Step A 生成的随机权重 ONNX 文件构建 engine，验证 TRT 可识别所有自定义 Plugin 节点：

```bash
TRT_ROOT=/usr/local/tensorrt    # TRT 10.x 安装路径
PLUGIN_LIB=/home/jichengzhi/UniV2X/plugins/build/libuniv2x_plugins.so

# ego 侧（随机权重 50×50 ONNX → engine）
${TRT_ROOT}/bin/trtexec \
    --onnx=onnx/univ2x_ego_bev_backbone_50_rand.onnx \
    --staticPlugins=${PLUGIN_LIB} \
    --saveEngine=engines/univ2x_ego_50_rand.engine \
    --verbose

# 基础设施侧（随机权重 50×50 ONNX → engine）
${TRT_ROOT}/bin/trtexec \
    --onnx=onnx/univ2x_infra_bev_backbone_50_rand.onnx \
    --staticPlugins=${PLUGIN_LIB} \
    --saveEngine=engines/univ2x_infra_50_rand.engine \
    --verbose
```

成功标志：两个 engine 均输出 `[TRT] Engine built successfully`，无 `unsupported layer` 报错。Step A 通过后进入 Step B。

### 4.6 TRT engine 精度对比（Step B：正式精度验收）

使用 Step B 生成的真实权重 200×200 ONNX 文件构建 engine，并与 PyTorch 输出对比：

```bash
# 构建正式 engine（真实权重，200×200）
${TRT_ROOT}/bin/trtexec \
    --onnx=onnx/univ2x_ego_bev_backbone_200.onnx \
    --staticPlugins=${PLUGIN_LIB} \
    --saveEngine=engines/univ2x_ego_200.engine \
    --verbose

${TRT_ROOT}/bin/trtexec \
    --onnx=onnx/univ2x_infra_bev_backbone_200.onnx \
    --staticPlugins=${PLUGIN_LIB} \
    --saveEngine=engines/univ2x_infra_200.engine \
    --verbose
```

精度对比（ego 和 infra 各执行）：

```bash
python3 - << 'EOF'
import torch, tensorrt as trt
# 构造与 ONNX export 完全相同的 dummy 输入（固定随机种子）
# 分别用 PyTorch 模型（真实权重）和 TRT engine 做推理
# 对比两者输出的最大误差
# 验收阈值：FP32 mode < 1e-4，FP16 mode < 1e-2
EOF
```

---

## 五、验收标准

| 验收项 | Step | 通过标准 | 备注 |
|---|---|---|---|
| 插件编译 | — | `libuniv2x_plugins.so` 生成，无编译错误 | TRT 10.x + SM89 |
| 工具函数精度 | — | `custom_torch_atan2_trt` vs `torch.atan2` 最大误差 < 1e-5 | |
| 模块类导入 | — | 5 个 TRT 变体类导入成功，mmcv Registry 注册正常 | |
| TSA/SCA 一致性 | — | TRT 变体与原版 PyTorch 输出最大误差 < 1e-4 | |
| ONNX 图结构 ×2 | **A** | ego + infra ONNX 中含 MSDAPlugin / RotatePlugin / InversePlugin 节点 | 随机权重，50×50 |
| TRT engine 编译 ×2 | **A** | ego + infra `trtexec` 成功生成 `.engine`，无不支持层报错 | 随机权重，50×50 |
| ONNX 合法性 ×2 | **B** | ego + infra `onnx.checker` 通过 | 真实权重，200×200 |
| TRT engine 编译 ×2 | **B** | ego + infra `trtexec` 成功生成正式 `.engine` | 真实权重，200×200 |
| TRT 精度对比 ×2 | **B** | ego + infra engine 输出与 PyTorch 最大误差 < 1e-4（FP32）/ < 1e-2（FP16） | **正式验收项** |

---

## 六、文件变更清单

| 文件路径 | 操作 | 来源 |
|---|---|---|
| `plugins/CMakeLists.txt` | 新建 | 基于 vad-trt/plugins/CMakeLists.txt 改写 |
| `plugins/multi_scale_deform_attn/*` | 复制 | vad-trt/plugins/multi_scale_deform_attn/ |
| `plugins/rotate/*` | 复制 | vad-trt/plugins/rotate/ |
| `plugins/inverse/*` | 复制 | BEVFormer_tensorrt/TensorRT/plugin/inverse/ |
| `projects/mmdet3d_plugin/univ2x/functions/__init__.py` | 新建 | — |
| `projects/mmdet3d_plugin/univ2x/functions/multi_scale_deformable_attn.py` | 新建 | uniad-trt bevformer_tensorrt.patch (_MSDAPlugin 部分) |
| `projects/mmdet3d_plugin/univ2x/functions/inverse.py` | 新建 | uniad-trt uniad-onnx-export.patch |
| `projects/mmdet3d_plugin/univ2x/functions/rotate.py` | 新建 | uniad-trt bevformer_tensorrt.patch |
| `projects/mmdet3d_plugin/core/bbox/util.py` | 追加 | uniad-trt uniad-onnx-export.patch (util.py 部分) |
| `projects/mmdet3d_plugin/univ2x/modules/multi_scale_deformable_attn_function.py` | 追加 | uniad-trt bevformer_tensorrt.patch |
| `projects/mmdet3d_plugin/univ2x/modules/spatial_cross_attention.py` | 追加 | uniad-trt uniad-onnx-export.patch (SCA 部分) |
| `projects/mmdet3d_plugin/univ2x/modules/temporal_self_attention.py` | 追加 | uniad-trt uniad-onnx-export.patch (TSA 部分) |
| `projects/mmdet3d_plugin/univ2x/modules/encoder.py` | 追加 | uniad-trt uniad-onnx-export.patch (Encoder 部分) |
| `projects/mmdet3d_plugin/univ2x/modules/__init__.py` | 修改 | 新增 TRT 类导出 |
| `projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py` | 新建 | 基于 univ2x_coop_e2e_track.py 替换模块类型 |
| `tools/export_onnx_univ2x.py` | 新建 | 基于 uniad-trt/tools/export_onnx.py 适配 |

---

## 七、注意事项

1. **BEV 尺寸两阶段策略**：现有 checkpoint 均为 200×200 训练，BEV queries 形状 `[40000, 256]` 与 50×50 的 `[2500, 256]` 不兼容，**不能用现有 checkpoint 直接切换 BEV 尺寸**。因此分两步：① 用**随机初始化权重 + 50×50** 验证图结构和 TRT 编译（Step A）；② 切换至**真实 checkpoint + 200×200** 完成精度验收（Step B）。所有 BEV 尺寸相关参数（`bev_h`、`bev_w`）必须通过 config 控制，不可 hard-code。

2. **双模型独立导出（核心约束）**：ego 车辆侧（`cfg.model_ego_agent`）和基础设施侧（`cfg.model_other_agent_inf`）**必须分别**导出独立的 ONNX 和 TRT engine，对应不同的硬件节点部署，不可合并。`MultiAgent` 外层封装仅用于联合训练，推理时不使用。

3. **导入路径适配**：uniad-trt 中的导入形如 `from projects.mmdet3d_plugin.uniad.functions import ...`，迁移到 UniV2X 后需改为 `from projects.mmdet3d_plugin.univ2x.functions import ...`。

4. **`img_metas_scene_token` 参数**：UniAD `TemporalSelfAttentionTRT` 额外接受 `img_metas_scene_token` 用于场景切换检测，UniV2X 的 TSA 使用不同的时序机制，迁移时需核查该参数是否存在及其处理逻辑。

5. **TRT 版本：已确认 TRT 10.x**。插件 C++ 代码中 `IPluginV2DynamicExt` 接口在 TRT 10 有变动（`enqueue` 签名改变），必须应用 `uniad-trt/patch/plugins-trt10-support.patch` 中的兼容性修改，不可跳过。

6. **Plugin 名称统一**：vad-trt C++ 代码注册名为 `"MultiScaleDeformableAttentionPlugin"`，uniad-trt Python symbolic 注册名为 `"MSDAPlugin"`，两者不一致会导致 TRT 找不到 plugin。**迁移时必须统一**：建议将 C++ 注册名改为 `"MSDAPlugin"` 与 Python 端对齐（改动量更小）。

7. **目标平台：RTX 4090（SM 89）**。CMakeLists.txt 中 `set(ARCH 89)` 已配置，ONNX export 和 trtexec 构建 engine 均在本机执行，无需 cross-compile。
