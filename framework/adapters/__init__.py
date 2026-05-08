"""Framework adapters — 把 v1.5 Config schema 映射到具体网络架构.

每个 adapter 实现:
  - Config → 网络模块树构造 / 模块名映射
  - 网络架构特定的 prune/quant 可行性约束 (扩展 PHYSICAL constraints)
  - 4090 latency 数据加载 (参考 baseline 表)

当前支持:
  - univ2x_tiny: UniAD-tiny class (R50 + 50x50 BEV + 无 DCN)
  - pyramid_fusion: HEAL Pyramid Fusion (全 CNN)

Phase 3 扩展:
  - univ2x: 完整 UniV2X (R101 + DCNv2 + V2X comm)
"""
