# 工作流 Checklist

> 简单说明: 每段编码前要看什么, 编码完后要更新什么.

---

## 开始下一段编码前 — 必查文件

### 上下文 / 任务定位
- `CLAUDE.md` (项目根目录) — 当前 phase 的目标 + MUST/NEVER 规则
- `paper_learning/2. AAAI最终故事/00_故事评估与实验路线_v1.md` — 整体路线图
- `paper_learning/2. AAAI最终故事/05_论文局限性_living.md` — 已知 caveat, 避免重蹈
- `paper_learning/2. AAAI最终故事/reflection_mistakes.md` — 历史错误清单

### 当前实验状态
- `results/m4_9_framework_pareto_v2.csv` — 10-anchor 最新表
- `results/m4_9_framework_pareto_report_v2.md` — 解读
- `results/m4_9_closed_loop_v2_report.md` — framework 自动 search demo
- `git status` — 有哪些未提交改动
- `git log --oneline -10` — 最近做了什么

### 框架代码 (改框架前必看)
- `framework/capability_schema.py` — HW 字段 schema
- `framework/config_schema.py` — Config schema
- `framework/constraints.py` — 硬约束清单
- `framework/measure_pyramid.py` — closed-loop 测量入口

---

## 完成任务后 — 必更新文件

### 测量数据 (实验完必更新)
- `results/m4_8_*.json` — 原始 bench / AP eval 输出
- `results/m4_9_framework_pareto_v2.csv` — 如果产生新 anchor
- `results/m4_9_framework_pareto_report_v2.md` — 同上, 新增 anchor 行 + 更新 Pareto 段

### 论文局限性 (每次实验后必更新, 强制)
- `paper_learning/2. AAAI最终故事/05_论文局限性_living.md`
  - §0 元信息: 最后更新日期, 当前 anchor 总数
  - §6 更新日志: 追加一行 (不删旧)
  - 如果消除 limitation → §1 → §2 转移
  - 如果暴露新 limitation → §1 添加并引用 measurement file

### 框架代码 (改了框架才更新)
- `framework/constraints.py` — 新硬件 constraint 加到此处
- `configs/hardware/*.yaml` — 新平台 feature flag / capability
- `framework/capability_schema.py` — 新 schema 字段

### 项目层
- `CLAUDE.md` — 仅 phase 切换或重大方向变化时改
- 新文件如果产生 → `git add` + commit (commit message 用 feat/fix/docs 前缀, 见 ~/.claude/rules/common/git-workflow.md)

---

## 简单流程示例

**开新实验前**:
```
1. 看 CLAUDE.md 确认还在哪个 phase
2. 看 results/m4_9_framework_pareto_v2.csv 确认 baseline 数据
3. 看 05_论文局限性_living.md 看有没有相关 caveat
```

**实验完成后**:
```
1. 把 *.json / *.csv 写到 results/
2. 更新 05_论文局限性_living.md (§0 + §6 日志, 必填)
3. 如果是新 anchor → 更新 m4_9_framework_pareto_v2.csv + report
4. git add + commit (用户授权后)
```

---

## 不要做的事

- ❌ 不跑实验前不看 reflection_mistakes.md (会重复以前的坑)
- ❌ 不更新 05_论文局限性_living.md 就直接进下一个实验
- ❌ 不 commit 大批未追踪文件 (可能含 .env / 大模型权重)
- ❌ 不 amend commit (先建新 commit, 用户审完再决定)
