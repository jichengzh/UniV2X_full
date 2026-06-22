"""Stage1 — 部署场景刻画 + 硬件扫描 + 网络计算图扫描与分区。

设计文档: multi_agent/methods/design/stage1_graph_scan_partition_v1.md
子模块:
  hardware_scan : 子流程 A (硬件 capability 归一化访问)
  graph_scan    : 子流程 B 模型无关核 (S1-S6)
  adapters      : 每模型 trace adapter
  run_scan      : CLI 入口
"""
