# UniV2X INT8 量化 — 多智能体协作执行计划

**文档类型**：项目 PM 执行计划（Multi-Agent Collaboration Guide）  
**版本**：v1.0  
**日期**：2026-04-05  
**项目负责人**：PM（本文档）  
**执行目标**：完成 BEV Encoder 和下游头的 TRT 隐式 INT8 DataCalibrator 量化，并集成进端到端测试管线

---

## 一、背景与目标

### 1.1 现状

经过前序实验（见 `result_log.md` §11-13），已确认：
- AdaRound + Q/DQ 方案失败（AMOTA 0.137/0.190，均废弃）
- QuantV2X 研究表明（见 `quantv2x_learn.md`）：正确路径是 **TRT 隐式 INT8 DataCalibrator**，与 AdaRound 无关
- BEV Encoder INT8 v5（50 帧，ego only）已初步验证 AMOTA=0.364，引擎 43 MB

### 1.2 本次目标

| 目标 | 量化范围 | 预期引擎大小 | 成功指标 |
|------|---------|:-----------:|---------|
| **Stream A**：BEV Encoder INT8 扩容 + infra | ego + infra BEV encoder | ~43 MB × 2 | AMOTA ≥ 0.364，cosine ≥ 0.999 |
| **Stream B**：下游头 INT8 首次实现 | ego + infra downstream heads | ~80 MB × 2（预期） | cosine ≥ 0.999，AMOTA 不低于 FP16 基准 0.001 |

### 1.3 两个 Stream 的独立性

```
Stream A（BEV Encoder）              Stream B（下游头）
─────────────────────────            ─────────────────────────
输入：DataLoader 原始帧              输入：BEV + 检测头 中间激活
ONNX：univ2x_*_bev_encoder_*.onnx   ONNX：univ2x_*_downstream.onnx
工具：build_trt_int8_univ2x.py      工具：build_trt_int8_univ2x.py
                                     ⚠️ 需要新建 dump_downstream_calibration.py
```

两个 Stream **可以并行**执行，互不依赖（除最终集成阶段）。

---

## 二、现有资产清单（开始前必读）

### 2.1 已可用的工具

| 工具 | 路径 | 用途 | 状态 |
|------|------|------|------|
| 校准数据导出 | `tools/dump_univ2x_calibration.py` | 从 DataLoader 导出原始帧 PKL | ✅ 可直接使用 |
| INT8 引擎构建 | `tools/build_trt_int8_univ2x.py` | `--target bev_encoder/downstream` | ✅ 支持两个 target |
| 下游头精度验证 | `tools/validate_downstream_trt.py` | Step A（形状）+ Step B（精度） | ✅ 已验证 FP16 |
| 端到端测试 | `tools/test_trt.py` | Hook A/B/C/D，集成 AMOTA 测试 | ✅ 需新增 Hook E |

### 2.2 已有 ONNX 文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `onnx/univ2x_ego_bev_encoder_200_1cam.onnx` | 65 MB | **Stream A** ego BEV encoder |
| `onnx/univ2x_infra_bev_encoder_200.onnx` | 65 MB | **Stream A** infra BEV encoder |
| `onnx/univ2x_ego_downstream.onnx` | ~127 MB | **Stream B** ego 下游头 |
| `onnx/univ2x_infra_downstream.onnx` | ~127 MB | **Stream B** infra 下游头 |

### 2.3 已有校准数据

| 文件 | 大小 | 用途 | 备注 |
|------|------|------|------|
| `calibration/bev_encoder_calib_inputs.pkl` | ~4 GB | BEV encoder 校准数据 | 50 帧，含 temporal prev_bev |
| `calibration/univ2x_ego_bev_encoder_int8_int8.cache` | 68 KB | ego BEV encoder INT8 activation scale 缓存 | v5 实验结果 |

### 2.4 已有 TRT 引擎

| 文件 | 大小 | 状态 |
|------|------|------|
| `trt_engines/univ2x_ego_bev_encoder_int8.trt` | 43 MB | ✅ v5 已验证（AMOTA=0.364） |
| `trt_engines/univ2x_ego_downstream.trt` | 152 MB | FP16，待量化 |
| `trt_engines/univ2x_infra_downstream.trt` | 134 MB | FP16，待量化 |

---

## 三、Stream A：BEV Encoder INT8 扩容（含 infra）

### 3.1 任务分解

#### Task A-1：扩充 ego BEV encoder 校准数据（可选但推荐）

**目标**：将 50 帧扩充至 128 帧，提升 activation scale 的统计准确性。

**执行命令**：
```bash
cd /home/jichengzhi/UniV2X
conda run -n UniV2X_2.0 python tools/dump_univ2x_calibration.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --n-cal 128 \
    --out calibration/bev_encoder_calib_inputs_128.pkl
```

**预期输出**：`calibration/bev_encoder_calib_inputs_128.pkl`（约 10 GB）  
**是否阻塞**：否，A-2 可先使用已有 50 帧数据

---

#### Task A-2：重建 ego BEV encoder INT8 引擎（验证 DataCalibrator 路径）

**前提**：`calibration/bev_encoder_calib_inputs.pkl`（50 帧）已存在  
**目标**：验证 `build_trt_int8_univ2x.py --target bev_encoder` 走通，可复现 AMOTA=0.364

**执行命令**：
```bash
# 若 cache 已存在则直接复用（--cali-data 仍需指定，但实际读 cache）
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --out trt_engines/univ2x_ego_bev_encoder_int8_v6.trt \
    --target bev_encoder \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --calib-cache calibration/univ2x_ego_bev_encoder_int8_v6.cache \
    --plugin plugins/build/libuniv2x_plugins.so \
    --workspace-gb 8
```

**预期输出**：`trt_engines/univ2x_ego_bev_encoder_int8_v6.trt`（约 43 MB）

**⚠️ 已知风险（来自 v3 教训）**：  
`_force_msda_fp16()` 中**不得**显式设置 `layer.precision = trt.DataType.HALF`，否则引入级联 dequant/quant 误差。当前代码已规避（见 `build_trt_int8_univ2x.py:222-233`），执行时确认日志无 `Explicitly setting precision` 字样。

---

#### Task A-3：构建 infra BEV encoder INT8 引擎（新增）

**前提**：需要 infra 的校准数据（与 ego 分开，infra 用 infra 相机视角的数据）

**Step A-3a**：导出 infra 校准数据
```bash
# 修改 config 或通过 cfg-options 指定 infra 模型
# 注意：infra 模型键为 model_other_agents（infra agent），需要确认导出脚本支持
conda run -n UniV2X_2.0 python tools/dump_univ2x_calibration.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --n-cal 50 \
    --out calibration/bev_encoder_infra_calib_inputs.pkl
```
> **注意**：需要确认 `dump_univ2x_calibration.py` 是否同时 dump ego 和 infra 的输入，或者是否需要添加 `--agent infra` 参数。若脚本不支持，Agent 需修改脚本加入 infra 路径支持。

**Step A-3b**：构建 infra INT8 引擎
```bash
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_infra_bev_encoder_200.onnx \
    --out trt_engines/univ2x_infra_bev_encoder_int8.trt \
    --target bev_encoder \
    --cali-data calibration/bev_encoder_infra_calib_inputs.pkl \
    --calib-cache calibration/univ2x_infra_bev_encoder_int8.cache \
    --plugin plugins/build/libuniv2x_plugins.so \
    --workspace-gb 8
```

---

#### Task A-4：精度验证（BEV cosine + 端到端 AMOTA）

**Step A-4a**：BEV cosine 验证
```bash
# 复用 validate_quant_bev.py（已有），指向新的 INT8 引擎
conda run -n UniV2X_2.0 python tools/validate_quant_bev.py \
    --engine trt_engines/univ2x_ego_bev_encoder_int8_v6.trt \
    --plugin plugins/build/libuniv2x_plugins.so
# 验收标准：cosine ≥ 0.999，mean_abs < 0.01
```

**Step A-4b**：端到端 AMOTA（168 样本）
```bash
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --use-bev-trt trt_engines/univ2x_ego_bev_encoder_int8_v6.trt \
    --plugin plugins/build/libuniv2x_plugins.so \
    --out-dir output/int8_bev_v6
# 验收标准：AMOTA ≥ 0.364
```

---

### 3.2 Stream A 依赖图

```
A-1（128 帧扩充，可选）
    ↓（可跳过）
A-2（ego INT8 引擎，用 50 帧）────→ A-4a（cosine 验证）
A-3a（infra 校准数据）                    ↓
A-3b（infra INT8 引擎）         A-4b（端到端 AMOTA）
```

---

## 四、Stream B：下游头 INT8 PTQ（首次实现）

### 4.1 核心技术挑战

下游头的输入不是原始图像帧，而是**模型中间层激活**：

```
DownstreamHeadsWrapper 的输入（见 export_onnx_univ2x.py）：
  - bev_embed:      (H*W, 1, C) = (40000, 1, 256)
  - query_feats:    (6, 1, N, C) = (6, 1, 901, 256)
  - all_bbox_preds: (6, 1, N, 10)
  - all_cls_scores: (6, 1, N, 10)
  - lane_query:     (1, 300, C)
  - lane_query_pos: (1, 300, C)
  - command:        (int64)          [infra 无此输入]
```

这些张量**在实际推理中间产生**，需要 hook 截取，而非从 DataLoader 直接获取。

---

### 4.2 任务分解

#### Task B-1：编写 `tools/dump_downstream_calibration.py`（关键新工具）

**功能**：在真实推理流程中插入 hook，捕获下游头的输入张量，保存为 PKL 文件。

**设计方案**：
```python
# 伪代码结构
class DownstreamInputCapture:
    """Hook into pts_bbox_head.get_detections_trt() 和 seg_head 的输出，
    在 DownstreamHeadsWrapper.forward() 被调用之前截取所有输入张量。"""
    
    def __init__(self, model, save_dir, max_frames=50):
        self.captures = []
        # 在模型的下游头调用点 monkey-patch forward
        # 与 test_trt.py 中 Hook D 的做法一致
    
    def capture(self, bev_embed, query_feats, bbox_preds, cls_scores,
                lane_query, lane_query_pos, command):
        self.captures.append({
            'bev_embed': bev_embed.cpu().numpy(),
            'query_feats': query_feats.cpu().numpy(),
            # ...
        })
```

**参考文件**：
- `tools/test_trt.py`：Hook D 的实现方式（如何截取下游头输入）
- `export_onnx_univ2x.py`：`DownstreamHeadsWrapper` 的输入格式定义
- `tools/dump_univ2x_calibration.py`：PKL 保存格式参考

**执行命令（完成后）**：
```bash
conda run -n UniV2X_2.0 python tools/dump_downstream_calibration.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --n-frames 50 \
    --out-ego calibration/downstream_ego_calib_inputs.pkl \
    --out-infra calibration/downstream_infra_calib_inputs.pkl
```

**预期输出**：
- `calibration/downstream_ego_calib_inputs.pkl`（~X GB，含 bev_embed 等 7 个张量 × 50 帧）
- `calibration/downstream_infra_calib_inputs.pkl`（~X GB，含 bev_embed 等 6 个张量 × 50 帧，无 command）

**估算存储**：每帧约 `40000*256*4 + 6*901*256*4 ≈ 46 MB` → 50 帧约 2.3 GB

---

#### Task B-2：确认 `build_trt_int8_univ2x.py` 对 downstream 的支持

检查 `--target downstream` 路径的兼容性：
- 下游头 ONNX 包含 MSDAPlugin 邻层吗？（`validate_downstream_trt.py` 的 Step A 可检查）
- 若不含 MSDAPlugin，`_force_msda_fp16()` 的 plugin_count 应为 0，全层可参与 INT8

```bash
# 快速 smoke test（随机权重，验证引擎可构建）
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_downstream.onnx \
    --out trt_engines/univ2x_ego_downstream_int8_smoke.trt \
    --target downstream \
    --no-int8 \
    --plugin plugins/build/libuniv2x_plugins.so
# 先用 --no-int8 验证 FP16 路径可通，再开 INT8
```

---

#### Task B-3：构建 ego + infra 下游头 INT8 引擎

**前提**：Task B-1 完成，校准数据已就绪

```bash
# ego 下游头
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_downstream.onnx \
    --out trt_engines/univ2x_ego_downstream_int8.trt \
    --target downstream \
    --cali-data calibration/downstream_ego_calib_inputs.pkl \
    --calib-cache calibration/univ2x_ego_downstream_int8.cache \
    --plugin plugins/build/libuniv2x_plugins.so \
    --workspace-gb 8

# infra 下游头
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_infra_downstream.onnx \
    --out trt_engines/univ2x_infra_downstream_int8.trt \
    --target downstream \
    --cali-data calibration/downstream_infra_calib_inputs.pkl \
    --calib-cache calibration/univ2x_infra_downstream_int8.cache \
    --plugin plugins/build/libuniv2x_plugins.so \
    --workspace-gb 8
```

**预期输出大小**：ego ~80 MB，infra ~70 MB（目标：FP16 152/134 MB 的 50%）

---

#### Task B-4：精度验证（下游头 cosine + 端到端 AMOTA）

**Step B-4a**：下游头 cosine 验证（复用 validate_downstream_trt.py）
```bash
# ego Step B（真实 checkpoint）
conda run -n UniV2X_2.0 python tools/validate_downstream_trt.py \
    --engine trt_engines/univ2x_ego_downstream_int8.trt \
    --model ego \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth
# 验收标准：traj_scores/traj_preds/occ_logits/sdc_traj 的 cosine ≥ 0.999

# infra
conda run -n UniV2X_2.0 python tools/validate_downstream_trt.py \
    --engine trt_engines/univ2x_infra_downstream_int8.trt \
    --model infra \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth
```

**Step B-4b**：新增 Hook E 并运行端到端 AMOTA

在 `tools/test_trt.py` 中新增 Hook E（下游头 INT8 TRT），参照现有 Hook D 的结构：
```python
# 在 test_trt.py 中（Hook D 之后）
def attach_downstream_int8_hook(ego_model, infra_model,
                                 ego_engine_path, infra_engine_path,
                                 plugin_path):
    """Hook E: 替换下游头 forward 为 INT8 TRT 引擎调用"""
    # ...（参照 Hook D 实现）
```

```bash
# 全链路 AMOTA（使用 BEV INT8 + 下游头 INT8）
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --use-bev-trt trt_engines/univ2x_ego_bev_encoder_int8_v6.trt \
    --use-downstream-int8 trt_engines/univ2x_ego_downstream_int8.trt \
    --plugin plugins/build/libuniv2x_plugins.so \
    --out-dir output/int8_bev_downstream
# 验收标准：AMOTA ≥ (Hook-A+B+C+D FP16 0.370) - 0.005 = 0.365
```

---

### 4.3 Stream B 依赖图

```
B-1（新建 dump_downstream_calibration.py）
    ↓
B-2（smoke test，验证 build 路径）── B-3（构建 ego+infra INT8 引擎）
                                            ↓
                                     B-4a（cosine 验证）
                                            ↓
                                     B-4b（Hook E + 端到端 AMOTA）
```

---

## 五、Agent 分工与职责

### Agent 1（Stream A 负责人）

**职责**：BEV Encoder INT8 扩容与 infra 量化

| 任务 | 关键产出 | 验收标准 |
|------|---------|---------|
| A-1：128 帧校准数据导出 | `calibration/bev_encoder_calib_inputs_128.pkl` | 文件存在，128 帧 |
| A-2：ego INT8 引擎重建 | `trt_engines/univ2x_ego_bev_encoder_int8_v6.trt` | cosine ≥ 0.999 |
| A-3：infra INT8 引擎 | `trt_engines/univ2x_infra_bev_encoder_int8.trt` | cosine ≥ 0.999 |
| A-4：端到端验证 | AMOTA 数值 | ≥ 0.364 |

**关键注意事项**：
- `dump_univ2x_calibration.py` 是否支持 infra 模型的校准数据导出？若不支持需要告知 PM 并修改脚本
- 使用已有 cache 时（`univ2x_ego_bev_encoder_int8_int8.cache`），构建会跳过 calibration 直接读 cache，速度极快（< 5 分钟）
- **禁止**在 `_force_msda_fp16()` 中显式设置 `layer.precision = HALF`（v3 教训）

---

### Agent 2（Stream B 负责人）

**职责**：下游头 INT8 首次实现，包括新工具开发

| 任务 | 关键产出 | 验收标准 |
|------|---------|---------|
| B-1：新建 dump 脚本 | `tools/dump_downstream_calibration.py` | 成功导出 50 帧 PKL |
| B-2：smoke test | INT8 引擎可构建，无报错 | 构建成功 |
| B-3：ego+infra INT8 引擎 | 两个 `.trt` 文件 | cosine ≥ 0.999 |
| B-4a：cosine 验证 | 验证报告 | 所有输出 cosine ≥ 0.999 |
| B-4b：Hook E + 端到端 | AMOTA 数值 | ≥ 0.365 |

**关键注意事项**：
- `dump_downstream_calibration.py` 的设计要点：
  - 参照 `test_trt.py` 中 Hook D 的截取方式，在 `DownstreamHeadsWrapper.forward()` 调用前 hook
  - ego 和 infra 的输入格式不同（ego 有 command，infra 无），分别存储
  - 张量需要 `.detach().cpu().numpy()` 后存储，避免 GPU 内存泄漏
- 下游头无 MSDAPlugin，plugin_count 应为 0，全部层理论上可 INT8

---

### Agent 3（集成与验证负责人）

**职责**：汇总两个 Stream 产出，运行完整端到端测试，更新文档

| 任务 | 关键产出 | 验收标准 |
|------|---------|---------|
| C-1：集成 Hook E 到 test_trt.py | 代码修改 | 通过 smoke test |
| C-2：全链路端到端（BEV INT8 + 下游头 INT8） | AMOTA 结果 | ≥ 0.365 |
| C-3：更新 result_log.md | §12.x 更新 | 所有数值记录准确 |
| C-4：更新 MEMORY.md | 新引擎路径与大小 | 条目完整 |

**前提**：Stream A 的 A-4b 和 Stream B 的 B-4a 均通过，才能进入 C-2。

---

## 六、验收标准汇总

| 阶段 | 验收项 | 通过标准 | 失败处理 |
|------|--------|---------|---------|
| A-2/A-3 | BEV cosine | cosine ≥ 0.999 | 检查校准数据质量，扩充帧数 |
| A-4b | 端到端 AMOTA（BEV INT8） | ≥ 0.364 | 可回退到 50 帧 v5 引擎 |
| B-4a | 下游头 cosine | 所有输出 cosine ≥ 0.999 | 检查 PKL 数据中的 bev_embed 激活范围 |
| B-4b | 端到端 AMOTA（含下游头 INT8） | ≥ 0.365（vs 全 FP16 0.370，容差 -0.005） | 若 < 0.360，回退下游头到 FP16，仅保留 BEV INT8 |
| C-2 | 全链路 AMOTA | ≥ 0.360 | 分析 INT8 下游头贡献，决定是否保留 |

---

## 七、风险与缓解措施

| 风险 | 概率 | 影响 | 缓解方案 |
|------|:----:|:----:|---------|
| infra 校准数据导出失败（脚本不支持） | 中 | 中 | 修改 `dump_univ2x_calibration.py` 添加 `--agent infra` 参数；或手动 hook infra forward |
| 下游头 INT8 cosine < 0.99（激活分布异常） | 中 | 高 | 检查是否有激活值异常大的层（occ_head 的 logits 无界），考虑跳过特定层 INT8 |
| 存储不足（校准数据 ~12 GB 总计） | 低 | 中 | 先 dump 50 帧，验证精度后再考虑扩充 |
| `build_trt_int8_univ2x.py --target downstream` 无法解析 ONNX | 低 | 高 | 检查 ONNX 中是否有 TRT 不支持的算子；参照 `validate_downstream_trt.py` 排查 |
| Hook E 与 Hook D（V2X 路径）冲突 | 低 | 中 | Hook E 仅替换非 V2X 路径的下游头，V2X 路径单独处理 |

---

## 八、交付物列表

完成后应存在以下文件：

```
trt_engines/
├── univ2x_ego_bev_encoder_int8_v6.trt          # Stream A，ego BEV INT8（~43 MB）
├── univ2x_infra_bev_encoder_int8.trt           # Stream A，infra BEV INT8（~43 MB）
├── univ2x_ego_downstream_int8.trt              # Stream B，ego 下游头 INT8（~80 MB 预期）
└── univ2x_infra_downstream_int8.trt            # Stream B，infra 下游头 INT8（~70 MB 预期）

calibration/
├── bev_encoder_calib_inputs_128.pkl            # Stream A（可选扩充，~10 GB）
├── bev_encoder_infra_calib_inputs.pkl          # Stream A infra（~4 GB）
├── downstream_ego_calib_inputs.pkl             # Stream B（~2.3 GB）
├── downstream_infra_calib_inputs.pkl           # Stream B（~2.0 GB）
├── univ2x_ego_bev_encoder_int8_v6.cache        # Stream A cache
├── univ2x_infra_bev_encoder_int8.cache         # Stream A cache
├── univ2x_ego_downstream_int8.cache            # Stream B cache
└── univ2x_infra_downstream_int8.cache          # Stream B cache

tools/
├── dump_downstream_calibration.py              # Stream B 新建（⚠️ 需开发）
└── test_trt.py                                 # 新增 Hook E（--use-downstream-int8）
```

---

## 九、执行顺序建议

```
第 1 天（并行）：
  Agent 1: A-1（导出 128 帧，后台运行）
            A-2（用已有 50 帧 + cache 快速验证路径，< 5 分钟）
  Agent 2: B-1（编写 dump_downstream_calibration.py）
            B-2（smoke test，无需校准数据）

第 2 天（并行）：
  Agent 1: A-3a（导出 infra 校准数据）+ A-3b（构建 infra INT8 引擎）
            A-4a（cosine 验证）
  Agent 2: 运行 B-1 脚本导出 50 帧下游头数据
            B-3（构建 ego + infra 下游头 INT8 引擎）

第 3 天（串行）：
  Agent 1 + 2: A-4b + B-4a 均通过
  Agent 3: C-1（Hook E 集成）→ C-2（全链路 AMOTA）→ C-3/C-4（文档更新）
```

---

## 十、参考文件速查

| 需要了解什么 | 看哪里 |
|------------|--------|
| 下游头 ONNX 输入格式 | `tools/export_onnx_univ2x.py:DownstreamHeadsWrapper` |
| INT8 构建器的完整实现 | `tools/build_trt_int8_univ2x.py` |
| DataCalibrator 实现参考 | `QuantV2X/opencood/tools/build_trt_int8.py` |
| Hook D（下游头 FP16）的实现 | `tools/test_trt.py` 中 `attach_downstream_hook()` |
| 下游头精度验证 | `tools/validate_downstream_trt.py` |
| 当前全部 AMOTA 结果 | `result_log.md §八、十五` |
| QuantV2X 研究结论 | `quantv2x_learn.md` |
| CLAUDE.md 当前阶段说明 | `CLAUDE_quantv2x.md §五` |
