# TRT 测试脚本改进计划

> 参考：`/home/jichengzhi/DL4AGX/AV-Solutions/vad-trt/export_eval/test_tensorrt.py`

---

## 一、当前 `validate_downstream_trt.py` 的局限性

### 测试方法
- 用**随机/固定合成输入**（dummy inputs）做 TRT 引擎推理
- 与 PyTorch 正向传播做数值对比（余弦相似度、最大误差）

### 核心局限

| 问题 | 影响 |
|------|------|
| 输入非真实数据 | 无法测量真实场景下的精度退化 |
| 不走完整推理链路 | 跳过了图像预处理、帧间状态管理、后处理 |
| 只输出数值指标 | 无法得到 AMOTA、L2 Distance、碰撞率等任务级指标 |
| 推理路径与实际部署不一致 | 无法检测状态传递（prev_bev、track_instances）中的问题 |

---

## 二、VAD-TRT 的测试方法

VAD-TRT 的 `test_tensorrt.py` 核心思路：

```
真实 nuScenes 数据
    ↓
正常 PyTorch 评估循环（model.forward_test）
    ↓  （方法替换 / Hook 注入）
    ├─ extract_img_feat     → 骨干网络 TRT 引擎
    └─ pts_bbox_head.forward → 检测头 TRT 引擎（含/不含 prev_bev 两个版本）
    ↓
dataset.evaluate() → 真实 AMOTA / Planning 指标
```

关键设计：
1. **单 GPU 非分布式**（`MMDataParallel` + 简单 for 循环）
2. **`data_ptr()` 直接地址绑定**，零拷贝将 PyTorch 张量传给 TRT
3. **保留完整外层循环**（数据加载、帧间状态、后处理不变）
4. **仅替换内部计算密集型子模块**，其余照常 PyTorch 运行

---

## 三、UniV2X `tools/test_trt.py` 与 VAD-TRT 的对比

### 相同点
- 单 GPU + `MMDataParallel` + 非分布式 for 循环
- 保留完整 `MultiAgent.forward_test` 调用链
- 直接 monkey-patch 目标方法，不引入 HookHelper 依赖
- 最终调用 `dataset.evaluate()` 得到真实指标

### 关键差异

| 方面 | VAD-TRT | UniV2X test_trt.py |
|------|---------|-------------------|
| 骨干网络 | 已导出为 TRT（替换 `extract_img_feat`）| PyTorch 运行（DCNv2 无 ONNX symbolic） |
| BEV 编码器 | 与骨干合并为单 TRT 引擎 | 单独 TRT 引擎（Hook A，Phase 1） |
| 检测头 | 整个 `pts_bbox_head.forward` 为 TRT | PyTorch 运行（Phase 2 检测头 TRT 接口设计不同） |
| 下游头 | 不适用（VAD 无 Motion/Occ 头） | PyTorch 运行（Phase 2 下游 TRT 输入格式与推理路径不兼容，见下文说明） |
| TRT 接口 | pycuda + 自定义 InferTrt | tensorrt Python API + `execute_async_v3` |

### 下游 TRT 不直接接入评估循环的原因

`DownstreamHeadsWrapper`（Phase 2 导出）接收的是检测解码器的**完整输出**（901 个查询的全层特征），而实际评估循环中 `motion_head.forward_test` 接收的是经过跟踪过滤的**活跃实例特征**（动态数量，20~50 个）。两者接口不兼容，因此 Phase 2 下游 TRT 精度通过 `validate_downstream_trt.py`（合成输入 + 余弦相似度）单独验证。

---

## 四、`tools/test_trt.py` 实现计划

### 已完成

- [x] **TrtEngine 类**：加载 `.trt` 引擎，通过命名张量绑定地址，`infer(dict) → dict`
- [x] **Hook A（BEV 编码器）**：
  - `attach_bev_hook(model_agent, engine, label)` 注入到 `pts_bbox_head.get_bev_features`
  - `_extract_bev_inputs` 处理 `img_metas` → 兼容 `img_shape` 列表 / 单元组两种格式
  - 正确处理 `prev_bev`（由外层帧循环传入，首帧 `use_prev_bev=0`）
- [x] **评估循环**：`single_gpu_test_trt` 函数——完全复刻 `custom_multi_gpu_test` 逻辑：
  - 在循环内在线计算 planning / occ 指标（不依赖分布式通信）
  - 返回与 `dataset.evaluate()` 兼容的 `ret_results` dict（含 `bbox_results` / `occ_results_computed` / `planning_results_computed`）
  - 使用 `MMDataParallel`（单 GPU），正确访问 `model.module`（`MultiAgent`）
- [x] **同时支持 ego 和 infra BEV TRT**（`--bev-engine-ego` / `--bev-engine-inf`）

### 待验证与修复

1. **实际运行验证**：需用真实数据集和 checkpoint 运行，比对 AMOTA 与 PyTorch baseline 偏差是否在 ≤0.005 范围内。

2. **Hook B（检测头 TRT，可选）**：Phase 2 检测头 TRT（`HeadsWrapper`）接口需要额外的 `l2g` 参数，需要在更高层面（`_forward_single_frame_inference`）做 hook 才能正确传参。

3. **Hook C（下游头 TRT，可选）**：需先解决输入格式不兼容问题（当前 `DownstreamHeadsWrapper` 接收全量 901 查询，而实际路径使用过滤后的活跃实例），或重新导出以活跃实例为输入的下游 TRT 引擎。

---

## 五、运行方式

```bash
# 仅 ego BEV 编码器 TRT（基础验证）
python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200.trt \
    --eval bbox \
    --out output/trt_results.pkl

# ego + infra 双 BEV TRT（协同推理）
python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200.trt \
    --eval bbox \
    --out output/trt_coop_results.pkl
```

---

## 六、验收标准

与纯 PyTorch baseline（`tools/test.py`）对比：

| 指标 | 可接受偏差 |
|------|-----------|
| AMOTA | ≤ 0.005 |
| AMOTP | ≤ 0.005 |
| Planning L2 (1s/2s/3s) | ≤ 0.01 m |
| Planning 碰撞率 | ≤ 0.001 |
