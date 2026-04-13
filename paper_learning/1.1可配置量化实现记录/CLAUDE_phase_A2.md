# Phase A2: Per-group 量化

> 覆盖维度: D3(per-group 量化粒度)
> 修改文件: projects/mmdet3d_plugin/univ2x/quant/quant_layer.py
> 依赖: Phase A1 (对称量化)
> 预计工作量: 1 天
> 状态: 待开始

---

## 1. 任务清单

- [ ] Task 1: UniformAffineQuantizer 增加 per-group 模式
- [ ] Task 2: QuantModule 适配 per-group
- [ ] Task 3: 单元测试
- [ ] Task 4: 集成验证
- [ ] 反思文档

---

## 2. 设计

### 2.1 per-group 量化原理

```
per-tensor:  整个张量共享 1 个 scale
per-channel: 每个输出通道 1 个 scale (out_channels 个)
per-group:   每 group_size 个元素共享 1 个 scale

示例: Linear(256, 1024), group_size=128
  权重形状: (1024, 256)
  per-tensor: 1 个 scale
  per-channel: 1024 个 scale  
  per-group: 1024 * (256/128) = 2048 个 scale
```

### 2.2 实现方案

�� `UniformAffineQuantizer` 中新增参数:
- `group_size: int = -1` (-1 表示不使用 group 量化)
- `use_group_quant: bool = False`

forward ��的 per-group 路径:
```python
if self.use_group_quant and self.group_size > 0:
    orig_shape = x.shape
    # reshape: (..., in_features) -> (..., n_groups, group_size)
    x = x.reshape(*orig_shape[:-1], -1, self.group_size)
    # 对最后一维做 per-group 量化 (每组独立 scale)
    # 利用 channel_wise 机制: reshape 后的倒数第二维就是 "通道"
    x_q = quantize_per_last_dim_group(x)
    x_q = x_q.reshape(orig_shape)
    return x_q
```

### 2.3 验证方法

```python
def test_per_group():
    q = UniformAffineQuantizer(n_bits=8, symmetric=True, 
                                group_size=64, use_group_quant=True)
    # 模拟 Linear(256, 1024) 的权重
    w = torch.randn(1024, 256)
    q.inited = False
    w_q = q(w)
    # scale 数量应为 1024 * (256/64) = 4096
    assert w_q.shape == w.shape
    err = (w - w_q).abs().mean().item()
    print(f'per-group(64) err: {err:.6f}')
    
    # 对比: per-tensor 误差应更大
    q_pt = UniformAffineQuantizer(n_bits=8, symmetric=True, channel_wise=False)
    q_pt.inited = False
    w_pt = q_pt(w)
    err_pt = (w - w_pt).abs().mean().item()
    print(f'per-tensor err: {err_pt:.6f}')
    assert err < err_pt  # per-group 应更精确
```

### 2.4 Debug 方案

| ���能问题 | 排查方法 |
|---------|---------|
| group_size 不整除 in_features | 对最后一组 pad ��报错 |
| reshape 后维度混乱 | 打印 reshape 前��� shape |
| 与 QuantModule 的交互问题 | 确保 fwd_func (F.linear) 的输入形状不变 |

---

## 3. 执行记录

- [x] Task 1: _quantize_per_group 方法实现, forward 中 early-return 分支
- [x] Task 2: QuantModule 无需额外修改 (per-group 在 UniformAffineQuantizer 内部完成)
- [x] Task 3: 5 个测试全部通过

### 测试结果

```
per-group(64) err: 0.005024  (delta shape: [4096])
per-tensor err:    0.008172
per-channel err:   0.005956
per-group better than both: True

group_size= 32: err=0.004472
group_size= 64: err=0.005024
group_size=128: err=0.005517
group_size=256: err=0.005992

INT4+per-group(64): err=0.091371
INT8+per-group(64): err=0.005024
```

---

## 4. 反思

### 完成时间
2026-04-12, 与 Phase A3 并行执行

### 关键发现

1. **per-group 精度排序正确**: group(64) < per-channel < per-tensor, 且 group_size 越小越精确。这意味着 per-group(32) 可以在几乎不增加计算量的情况下获得最好的量化精度。

2. **INT4+per-group 是有意义的组合**: INT4 per-group(64) 的误差 0.091 远大于 INT8 per-tensor 的 0.008, 但如果某层对量化不敏感, INT4+per-group 可以提供比 INT8+per-tensor 更低的有效带宽（4bit 存储）同时可能有更好的精度（per-group 补偿位宽损失）。

3. **delta shape 验证了分组正确性**: 1024x256 权重, group_size=64 -> 4096 个 scale 值 = 1024*(256/64), 完全正确。

### 对后续阶段的影响

- Phase B1 Q/DQ 注入: per-group 需要 reshape 技巧将 group 展开为 per-channel, 增加实现复杂度
- 搜索空间: group_size 可选 {32, 64, 128, 256, -1(disabled)}, 5 个离散值
