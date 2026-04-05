# Phase B-2 代码审查报告

> 由 PY-REV agent 生成，2026-04-02  
> 审查文件：`tools/export_onnx_adaround.py`  
> 结论：初始版本 BLOCK，修复后 PASS

---

## CRITICAL 问题（已修复）

### CRITICAL-1：soft_targets 检查路径——实现正确，文档有 typo
`arch_decision.md` 第29行的守卫是 `hasattr(m, 'soft_targets')`（检查 QuantModule 本身），
但 `soft_targets` 实际属于 `weight_quantizer`（AdaRoundQuantizer）。
**实现代码正确**，已在注释中标注此 typo，避免后续维护者"修复"引入 bug。

### CRITICAL-2：ref_model 死代码——已完全移除
`ref_model` 和 `ref_enc` 被 build 到 GPU（~数百 MB 显存）但从未使用，
整个 `with torch.no_grad():` 块是空操作。已删除，用单行注释替代。

---

## HIGH 问题（已修复）

### HIGH-2：sys.path 依赖运行目录——已改为 __file__ 绝对路径
从 `sys.path.insert(0, '.')` 改为：
```python
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tools_dir)
sys.path.insert(0, os.path.dirname(tools_dir))  # project root
```
现在脚本从任意工作目录都能找到 `export_onnx_univ2x.py`。

### HIGH-3（MEDIUM-1 升级）：双重加载 adaround_ckpt——已消除
`ckpt_data` 在 main() 中加载一次，传入 `apply_adaround_weights(qmodel, ckpt_data)`，
消除了第二次 `torch.load` 调用。

---

## MEDIUM 问题（知悉，未修改）

- `torch.load` 未加 `weights_only` 参数——内部 checkpoint 可接受，与项目其他脚本一致
- `print()` 而非 `logging`——与项目其他 tools/ 脚本风格一致，不改

---

## 修复后验收

- [x] CRITICAL-1 已修复（注释标注）
- [x] CRITICAL-2 已删除死代码
- [x] HIGH-2 已修复 sys.path
- [x] 双重加载已消除
- **最终结论：PASS** ✅
