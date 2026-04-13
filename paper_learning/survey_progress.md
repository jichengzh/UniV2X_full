# 综述项目进程报告

## 当前状态: Phase 3 - 已完成
## 最后更新: 2026-04-09

## Phase 0: 预调研
- 状态: 已完成
- 结论: 确定5大分类方向 (LLM加速/模型压缩/编译器/新型架构/自动驾驶)

## Phase 1: 并行检索
| Agent | 方向 | 类型 | 状态 | 论文数 | 备注 |
|-------|------|------|------|--------|------|
| Agent-1 | LLM/Transformer加速 | general-purpose | 已完成 | 10 | 未能调用WebSearch,基于训练知识 |
| Agent-2 | 量化/剪枝/NAS协同 | general-purpose | 已完成 | 10 | 使用WebSearch,质量高 |
| Agent-3 | 编译器与系统栈 | general-purpose | 已完成 | 9 | 使用WebSearch,质量高 |
| Agent-4 | 新型计算架构 | general-purpose | 已完成 | 10 | 使用WebSearch,质量高 |
| Agent-5 | 自动驾驶/V2X | general-purpose | 已完成 | 10 | 使用WebSearch,质量高 |

## Phase 2: 汇总去重
- 去重前总数: 49
- 去重后总数: 49 / 30 (目标) ✅ 已超额完成
- 无跨Agent重复论文

## Phase 3: 撰写
- 状态: 已完成
- 交付物: paper_learning/survey_hw_sw_codesign.md

## 待验证事项
- [ ] Agent-1的10篇论文链接需人工验证(基于训练知识,非实时搜索)
- [ ] 部分arXiv preprint论文需确认是否已正式发表

## 项目统计
- 总论文数: 49篇
- Venue分布: ISCA(4), HPCA(5), MICRO(2), ASPLOS(5), OSDI(2), SOSP(1), MLSys(4), NeurIPS(3), ICLR(1), ICML(1), CVPR(3), ICCV(2), DAC(2), PLDI(1), SC(1), IEEE TCAD(1), IEEE TC(1), ACM TRETS(2), Sensors(1), DASIP(1), arXiv(6)
