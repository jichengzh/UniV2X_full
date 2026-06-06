# hw-optimizer 实验宣告自查 SOP v1.2
> 2026-06-06 | ISS-042 整改输出 | Task#12 终清第 4 条件  
> v1.1: 补全收件人(+team-lead)、授权原文、板卡快照、"未送达不开跑"硬规则  
> v1.2: ISS-043 后追加 Step 0 — 引用数据前先查冻结清单(supervisor 明确建议)

---

## 核心教训 (ISS-042 / ISS-030 同构)

**宣告的本质是送达, 不是书写。**  
写在 tool-call 输出里 ≠ 发出。SendMessage 未调用 = 宣告未完成。  
(与 ISS-030 "描述≠授权" 同构: 意图表达 ≠ 对应行动完成)

**未送达不开跑 — 无豁免条款。**

---

## 宣告检查清单 (GPU 实验开跑前必做, 逐项勾选)

```
[ ] Step 0: 【v1.2新增】任何引用内部数据前先查冻结清单
            - 冻结数据集: Phase H (H1/H2/H3-5全冻结, Task#1 FROZEN占位)
            - 可用替代: ISS-014 canonical (P0_1_p25_*); data/tactic_workspace_bench.csv
            - 冻结数据可"标注透明"但标注不豁免冻结 — 必须替换为非冻结替代源
            - 无非冻结替代 → 标"Phase H 冻结中, 待用户裁定"且不给数值

[ ] Step 1: 查 nvidia-smi (4090) 或 GR3D sysfs (Orin) 确认设备空闲
            - 4090: GPU util 0%, mem ≤50MiB, 截取 nvidia-smi 输出备用
            - Orin: cat /sys/devices/gpu.0/load = 0, tegrastats 确认 GR3D@0%
            ⇒ 把板卡快照(原始文本输出)放入宣告消息

[ ] Step 2: 调用 SendMessage(to="supervisor") 且 CC to="team-lead", 消息包含:
            ① 实验名称 + 编号 (如 "D1v2 B=2 PyTorch FP32 全链路测试")
            ② 授权原文 — 引用授权者原话或 Task#xx 编号
               例: "授权: Task#12 team-lead 原文 '路线A Orin全链路真测'"
            ③ 板卡快照 — Step 1 的 nvidia-smi/GR3D 输出(复制进消息)
            ④ 预计耗时 + 输出文件路径
               例: "预计 ~5min; 结果落 results/d1_v2_b2_fp32_orin.csv"

[ ] Step 3: 等待 supervisor 或 team-lead 明确确认后开跑
            ⚠️ 无"5min超时自动继续" — 未收到确认 = 不开跑
            ⚠️ 若长时无响应, SendMessage 再次催确认, 不得自行开跑

[ ] Step 4: 实验完成后 SendMessage(to="supervisor") 报告:
            - 结果摘要(关键数字)
            - 原始文件路径
            - 任何异常(冷启/util非零/超时等)
```

---

## 触发条件

任何涉及以下操作均须走本 SOP:

| 操作类型 | 示例 |
|---------|------|
| GPU kernel 计时 | CUDA Event / trtexec bench |
| TRT build | trtexec --buildOnly |
| Orin SSH 远程实验 | 任何 GPU 占用操作 |
| 4090 主机实验 | PyTorch / TRT inference |

---

## 违规记录

| Issue | 定性 | 处置 |
|-------|------|------|
| ISS-041 (D1v2 宣告写在 tool-call 输出未 SendMessage) | 程序违规, 数据有效 | 减责(自查诚实); 累计第3次 |
| ISS-042 (同次累计计, 豁免用尽) | 零容忍 | 本 SOP 为整改输出 |

---

## 核心原则再强调

1. **送达 = SendMessage 工具调用成功返回** (不是写在文字里, 不是注释里)  
2. **收件人: supervisor + team-lead 双抄** (单抄不够)  
3. **未送达不开跑** — 任何"先跑再宣告"均构成违规  
4. **授权原文必须在宣告里** — "有授权"不等于在消息里写出来了

---

本文件为 Task#12 D3 交付附件, 与 ISS-042 整改闭环挂钩。  
后续每次 GPU 实验前: 打开此文件 → 逐步勾选 → SendMessage 发出 → 等确认 → 开跑。
