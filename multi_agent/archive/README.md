# archive — 被取代/降为过程记录的旧版文档

> 规则:文档被新版取代或定论被整合进骨架文档后,原文移入本目录**完整保留**(可追溯)。**指针规则**:原文档若被骨架文档吸收、或其旧路径仍被其它设计文档活引用,原位置留一行重定向指针(如 ap_activation / dims_hardware_v1);**整文件平移**(外部论文笔记迁 references、被打回版迁 archive)不留指针, 由 `multi_agent/README.md` 目录树与本表负责导航。本目录内容**不再维护**,引用定论请回到 `methods/design/` 六骨架文档。

| 文件 | 移入日期 | 取代/去向 | 完整性 | 为何归档 |
|---|---|---|---|---|
| `ap_activation_strategy_v1.md` | 2026-06-04 | 定论并入 `methods/design/pareto_definition_v1.md §七`(精度轴定位);原位有指针 | 与迁移前**逐字节一致**(md5 核验) | AP/mAOE 问题本质属 Pareto 精度轴定位, 不单独成文;原文大半是决策前的四路径 ROI 过程分析 |
| `dims_hardware_v1.md` | 2026-06-04 | 被 `methods/design/dims_hardware_v2.md` 取代;原位有指针 | 与迁移前**逐字节一致**(md5 核验) | v2 重组并勘误 v1 两处错误声明(DLA INT8 / CUDA Graph 幅度);v1 §6 档位计数推导(25/96)仍可查 |
| `moe_paper_dim_review_v1.md` | 2026-06-04 | 被 `references/moe_paper_dim_review_v2.md` 取代;无指针(整文件平移, 按上方指针规则) | **正文逐字节一致;尾注"文档路径"一行已就地更新**标注迁移去向 | v1 被 supervisor 打回(ISS-022:编造数字 496MB/8306ms/BEVDet + 6064→397 归因误读), 仅留作核验过程记录 |
