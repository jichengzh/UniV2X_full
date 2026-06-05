# hw-optimizer 实验宣告自查 SOP v1
> 2026-06-06 | ISS-042 整改输出 | Task#12 终清第 4 条件

---

## 核心教训 (ISS-042 / ISS-030 同构)

**宣告的本质是送达, 不是书写。**  
写在 tool-call 输出里 ≠ 发出。SendMessage 未调用 = 宣告未完成。  
(与 ISS-030 "描述≠授权" 同构: 意图表达 ≠ 对应行动完成)

---

## 宣告检查清单 (GPU 实验开跑前必做)

```
[ ] Step 1: nvidia-smi / GR3D sysfs 确认 GPU 空闲 (util 0%, mem ≤50MiB)
[ ] Step 2: **调用 SendMessage(to="supervisor")**, 内容包含:
            - 实验名称 + 授权来源 (Task#xx / data-orchestrator 指令)
            - 设备 + 功耗模式 (Orin MODE_30W / 4090 空闲确认)
            - 预计耗时 + 输出文件路径
[ ] Step 3: 等待 supervisor 确认 OR 5min 无回复后继续(记录时戳)
[ ] Step 4: 实验完成后 SendMessage(to="supervisor") 报告结果 + 文件路径
```

**Step 2 是硬步骤 — 不得以任何理由跳过。**  
"指令抵达前已跑完"不豁免; 下次开跑前宣告即可, 但不能事后补充冒充事前。

---

## 触发条件

任何涉及以下操作均须走本 SOP:
- GPU kernel 执行 (trtexec / PyTorch / CUDA Event 计时)
- Orin SSH 远程实验
- 4090 主机 TRT build / bench

---

## 违规记录

| Issue | 定性 | 处置 |
|-------|------|------|
| ISS-041 (D1v2 宣告写在 tool-call 输出) | 程序违规, 数据有效 | 减责(自查诚实); 累计第3次 |
| ISS-042 (同次累计计) | 零容忍, 豁免用尽 | 本 SOP 为整改输出 |

---

## 复现说明

本文件为 Task#12 D3 交付附件, 与 ISS-042 整改闭环挂钩。  
后续每次 GPU 实验前: 打开此文件 → 逐项勾选 → 执行 SendMessage → 开跑。
