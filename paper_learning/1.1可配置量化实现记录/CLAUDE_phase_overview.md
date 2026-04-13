# 1.1 可配置量化 — 阶段总览

> 对应实施计划：paper_learning/实施计划_1.1_可配置量化.md
> 总目标：D1-D9 全部可配置、可搜索、可部署

---

## 阶段划分

| 阶段 | 名称 | 覆盖维度 | 依赖 | 预计工作量 | 状态 |
|:----:|------|---------|:----:|:---------:|:----:|
| Phase A1 | 量化器核心增强 | D5(对称) + D4(entropy+cw) + D6(percentile) | 无 | 1.5 天 | **已完成** |
| Phase A2 | Per-group 量化 | D3(per-group) | Phase A1 | 1 天 | **已完成** |
| Phase A3 | V2X 通信量化 + 统一配置入口 | D9(通信) + D7/D8(配置汇总) | Phase A1 | 1.5 天 | **已完成** |
| Phase A4 | 敏感度分析 | 全部维度的基线数据 | Phase A3 | 1 天编写+1天执行 | **已完成** |
| Phase B1 | Q/DQ 注入工具 | D1-D8 -> ONNX | Phase A3 | 2.5 天 | **已完成** |
| Phase B2 | TRT 显式模式 + 验证 | ONNX -> TRT engine | Phase B1 | 1.5 天 | **engine 构建成功 (38.2MB)** |

---

## 依赖关系

```
Phase A1 (量化器核心)
   |
   +---> Phase A2 (per-group)
   |
   +---> Phase A3 (V2X通信 + 统一配置)
              |
              +---> Phase A4 (敏感度分析)
              |
              +---> Phase B1 (Q/DQ 注入)
                       |
                       +---> Phase B2 (TRT显式 + 验证)
```

Phase A1 是所有后续阶段的前置。
Phase A2 和 Phase A3 可在 A1 完成后并行启动。
Phase A4 和 Phase B1 可在 A3 完成后并行启动。

---

## 进度追踪

| 日期 | 完成阶段 | 备注 |
|------|---------|------|
| 2026-04-12 | Phase A1 完成 | 对称量化+entropy cw+percentile, 8/8 测试通过 |
| 2026-04-12 | Phase A2 完成 | per-group 量化, 5/5 测试通过, group(64) < channel < tensor 精度排序正确 |
| 2026-04-12 | Phase A3 完成 | CommQuantizer + apply_quant_config + default_int8.json, 核心逻辑验证通过 |
| 2026-04-12 | Phase A4 全部完成 | 6项分析全部通过; 45 safe+9 search+0 skip; 交互项为负(无恶化); cosine全局对比分辨力不足需改进 |
| 2026-04-12 | Phase B1 GPU验证通过 | inject_qdq_from_config.py: 60W+43A Q/DQ注入成功, 2个bug修复 |
| 2026-04-12 | Phase B2 engine构建通过 | explicit INT8 engine 38.2MB (vs 隐式43MB, FP16 75MB), 1个tactic跳过(非致命) |
