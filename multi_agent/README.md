# multi_agent — 软硬件协同优化框架 知识库 (★当前事实源)

本目录是 "软硬件协同加速框架" multi-agent team(`sw-hw-cooptim`)的知识库与产物落点(2026-05-31 起,持续维护)。

## 目录结构 (2026-06-04 doc-curator 重组后)
```
multi_agent/
├── background/                 研究总纲 + 实验数据可信度档案
│   └── 00_研究目标与实验档案_v1.md   最终目标 / 可扩展模型 / 实验档案分级 (★总纲)
├── methods/
│   ├── design/                 设计文档 — 收敛为论文骨架六文档 (新结论整合进对应骨架, 勿另开新文档)
│   │   ├── dims_pruning_v1.md          骨架1 剪枝维度 (含 §8 剪枝无悬崖定论)
│   │   ├── dims_quantization_v1.md     骨架2 量化维度 (含 Phase M 量化无信号定论)
│   │   ├── dims_hardware_v2.md         骨架3 硬件维度 (v1 已 archive; 含 §0.5 可搜性裁决 + H1/H2)
│   │   ├── doe_design_v1.md            骨架4 DOE 数据集设计
│   │   ├── pareto_definition_v1.md     骨架5 Pareto 前沿定义 (★§七=精度轴定位, 含原 ap_activation 定论)
│   │   └── search_space_size_v1.md     骨架6 搜索空间规模
│   ├── acceptance/             验收文档
│   │   └── acceptance_report_v1.md     逐 agent PASS/FAIL + 证据
│   └── progress/               进度/过程 (charter / issues_log / session 报告 / handoff)
│       ├── team_charter_v1.md          ★团队宪章 (纪律红线 + 勘误, 每个成员必读)
│       ├── issues_log_v1.md            问题落盘 (ISS-xxx)
│       └── ...                          session/编排/handoff 各报告
├── references/                 外部论文对照 / 文献笔记 (不属于 methods)
│   └── moe_paper_dim_review_v2.md      EMOS(TPAMI'26) 维度对照 (supervisor PASS)
├── archive/                    被取代的旧版文档 (完整保留可追溯, 不再维护; 见其 README)
├── data/                       有效数据
│   ├── dataset_v2.{csv,parquet} ★ 统一主表 (v2 schema)
│   ├── schema_v2.md            列定义; build_dataset_v2.py 构建脚本
│   └── sources/                构建输入 (原 tier1/tier2 文件, 合并前快照)
├── figure/                     描述性统计图 (可复跑)
└── output/
    └── README.md               engine/build 报告索引 (实物在 output/、results/)
```

## 文档归类规则 (按用户要求)
- **设计定论** → `methods/design/` 六骨架文档 (整合, 不追加; 过程性内容不留在 design)
- **验收文档** → `methods/acceptance/`
- **进度/过程** → `methods/progress/`
- **外部论文笔记** → `references/`;**被取代旧版** → `archive/`(原位留一行指针)

## 关联代码 (不在本目录, 在 repo 原位)
- `tools/configurable/{quant,prune,deploy}_config.py` — 三个配置驱动工具
- `tools/configurable/generate_dataset.py` — 数据生成管线 (有结构缺陷待修)
- `framework/config_schema.py` — 共享配置契约 (全程未改)

## 当前状态
~~(2026-06-01 快照: 7 行真测 latency, AP 全待补)~~ → ★[2026-06-04] 本行不再维护快照数字;数据现状以 `team_charter_v1.md §1`(完整点定义与计数)+ `background/00 §0.6` 为准, 进展以 `methods/progress/` 最新 session 报告为准。
