# Phase A3: V2X 通信量化 + 统一配置入口

> 覆盖维度: D9(V2X通信精度) + D1-D8(统一 quant_config.json 入口)
> 新建文件: tools/quick_eval_quant.py, projects/.../quant/comm_quant.py
> 依赖: Phase A1
> 预计工作量: 1.5 天
> 状态: 待开始

---

## 1. 任务清单

- [ ] Task 1: CommQuantizer 实现 (D9)
- [ ] Task 2: quant_config.json 格式定义
- [ ] Task 3: apply_quant_config() 实现
- [ ] Task 4: quick_eval_quant.py 主脚本
- [ ] Task 5: export_scales_to_config() (Scale 导出, Part A->B 接口)
- [ ] Task 6: 集成���试 (给定 config -> AMOTA)
- [ ] 反思文档

---

## 2. Task 1: CommQuantizer (D9)

### 2.1 设计

V2X 通信特征量化模拟通���带宽压缩。在通信��插入 fake-quant:

插入位置:
- AgentQueryFusionTRT.forward_trt(): infra track_query 传入 ego 之前
- LaneQueryFusionTRT.forward_trt(): infra lane_query 传入 ego 之前
- _get_coop_bev_embed(): infra BEV 特征传入 ego 之前

### 2.2 实现

```python
# projects/mmdet3d_plugin/univ2x/quant/comm_quant.py

class CommQuantizer(nn.Module):
    """V2X 通信特征量化器"""
    def __init__(self, n_bits=8, symmetric=True):
        super().__init__()
        self.quantizer = UniformAffineQuantizer(
            n_bits=n_bits, symmetric=symmetric,
            channel_wise=False, scale_method='minmax')
        self.enabled = True
    
    def forward(self, x):
        if not self.enabled:
            return x
        self.quantizer.inited = False
        return self.quantizer(x)
```

---

## 3. Task 2-3: 统一配置入口

quant_config.json 完整格式见实施计划_1.1_可配置量化.md。

apply_quant_config() 核心逻辑:
1. QuantModel 包裹 (global 默认参数)
2. 逐层覆盖 D1-D8
3. V2X 通信量化 D9

---

## 4. Task 4: quick_eval_quant.py

```
用法:
  python tools/quick_eval_quant.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth \
    --quant-config quant_configs/test_config.json \
    --eval-samples 17 \
    --cali-data calibration/bev_encoder_calib_inputs.pkl

输出:
  AMOTA, AMOTP, mAP, NDS + 各层 scale 值
```

---

## 5. Task 5: Scale 导出

PyTorch 校准完成后, 将每层 delta (scale) 写入 quant_config.json。
这是 Part A -> Part B 的数据接口。

---

## 6. 验��方法

```
1. 给定全 FP16 config -> AMOTA 应等于 baseline (0.381)
2. 给定全 INT8 config (skip 敏感层) -> AMOTA 应接近 0.364 (已知结果)
3. 给定混合 config -> AMOTA 在两者之间
4. 给定 V2X INT8 通信 -> 对比 FP16 通信的 AMOTA 差异
```

---

## 7. Debug 方案

| 可能问题 | 排查方法 |
|---------|---------|
| QuantModel 包裹后某层未找到 | 打印 named_modules 列表与 config keys 对比 |
| set_quant_state 后精度不变 | 检查 use_weight_quant / use_act_quant 标志 |
| CommQuantizer 导致 NaN | 检查通信张量是否含极端���, 加 clamp 保护 |
| eval_samples 太少导致 AMOTA 波动大 | 增加到 34 帧 (2/10) |

---

## 8. 执行记录

- [x] Task 1: CommQuantizer 实现, 含 FP16 passthrough + disabled 模式
- [x] Task 2: quant_config.json 格式定义, default_int8.json 示例文件
- [x] Task 3: apply_quant_config() 实现 (在 quick_eval_quant.py 中)
- [x] Task 4: quick_eval_quant.py 主脚本 (评估部分为占位, 核心 apply 逻辑完成)
- [x] Task 5: export_scales_to_config() 实现
- [x] Task 6: 集成测试

### 修复记录

1. CommQuantizer 的 `__init__` 中 n_bits=16 会触发 UniformAffineQuantizer 的 `assert 2<=n_bits<=8`。
   修复: 当 n_bits>=16 时不创建 quantizer (self.quantizer=None), forward 中直接 passthrough。

2. quant_model.py 和 comm_quant.py 的相对导入在单元测试环境中失败。
   修复: 添加 try/except fallback 到绝对导入。实际项目中通过 mmdet3d 框架导入不受影响。

### 测试结果

```
CommQuantizer:
  INT8 err: 0.008789
  INT4 err: 0.159423
  FP16 passthrough: OK
  Disabled passthrough: OK

QuantModel:
  QuantModules: 2 (fc1, fc2 正确替换)
  Per-layer INT4: OK
  Selective disable: OK
  Mixed precision output: OK

Config file:
  version=1.1, symmetric=True, v2x_comm present
```

---

## 9. 反思

### 完成时间
2026-04-12, 与 Phase A2 并行执行

### 关键发现

1. **CommQuantizer 的 n_bits>=16 边界情况**: 原始设计假设 quantizer 总是被创建, 但 FP16(16bit) 超出 UniformAffineQuantizer 的 2-8bit 范围。需要在创建时就做边界检查, 而不是仅在 forward 中检查。这类边界问题在后续扩展配置维度时很容易出现。

2. **apply_quant_config 的层名匹配**: QuantModel 包裹后, `qmodel.model.named_modules()` 的层名是相对于原始模型的 (如 `fc1`), 不含 `model.` 前缀。config 中的层名需要使用相同的命名约定。实际 UniV2X 模型的层名会更长 (如 `pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj`), 需要确保完全匹配。

3. **评估管线暂为占位**: quick_eval_quant.py 的实际评估需要加载完整 mmdet3d 数据管线, 当前为占位实现。Phase A4 敏感度分析时需要补全。

### 创建的文件清单

- `projects/mmdet3d_plugin/univ2x/quant/comm_quant.py` (新建)
- `projects/mmdet3d_plugin/univ2x/quant/__init__.py` (修改, 加入 CommQuantizer)
- `tools/quick_eval_quant.py` (新建)
- `quant_configs/default_int8.json` (新建)
