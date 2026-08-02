# AAAI-27 附录格式依据与翻译说明

## 官方依据

1. [AAAI-27 Author Kit](https://aaai.org/authorkit27/)：本目录中的 `aaai2027.sty` 和 `aaai2027.bst` 于 2026-07-28 从该官方压缩包直接提取，未作修改。
2. [AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)：要求使用 AAAI 双栏 camera-ready 样式、US Letter（8.5 × 11 英寸）页面、Type 1 或 TrueType 字体，并在评审阶段保持双盲匿名。
3. [AAAI-27 Supplementary Material](https://aaai.org/conference/aaai/aaai-27/supplementary-material/)：技术附录应作为独立的 Supplementary Document PDF 提交；附录必须匿名，正文应保持自洽，审稿人没有义务阅读补充材料。

## 当前文件如何落实要求

- `appendix_a_aaai27.tex` 是独立源文件，不引用、拼接或修改现有论文 PDF。
- 本目录使用的是本次从 AAAI-27 官网重新下载的样式文件；现有 `multi_agent/paper/Latex/aaai2027.sty` 的校验值不同，未参与本附录编译。
- 文档使用官方 `\documentclass[letterpaper]{article}` 和 `\usepackage[submission]{aaai2027}`。
- `submission` 选项自动隐藏作者与单位信息，并使用匿名评审页脚。
- 未修改官方 `aaai2027.sty`，也未使用 `geometry`、`fullpage`、`titlesec`、`float`、`\resizebox` 等官方禁止的版式工具。
- 表格跨双栏排版；表格正文采用官方允许的 9 pt 字号，并通过官方允许的 `\tabcolsep` 调整控制列间距。
- 官方样式文件 SHA-256：
  - `aaai2027.sty`: `391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623`
  - `aaai2027.bst`: `5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c`

## 术语表

| 中文术语 | 英文固定写法 | 说明 |
|---|---|---|
| 自车 | ego vehicle | 不与 host vehicle 混用 |
| 路侧单元 | roadside unit | 本段不重复引入缩写 |
| 自车观测时延注入模块 | ego-observation latency module | 保留“观测滞后”而非“计算阻塞”的机制含义 |
| 名义感知时延 | nominal perception latency | 记为 $\tau_{\mathrm{perc}}$ |
| 目标帧延迟 | target frame delay | 记为 $\Delta_{\mathrm{perc}}$ |
| Honest Driving Score | Honest Driving Score (Honest DS) | 首次出现时定义缩写 |
| 输入编码 | input encoding | Figure 2 第一阶段 |
| Backbone + Neck | Backbone + Neck | 保持正文模块名 |
| 特征融合 | feature fusion | Figure 2 第三阶段 |
| 检测（预测头 + 后处理） | detection (prediction head + post-processing) | 合并为正文使用的第四阶段 |
| 已归因总时延 | attributed total latency | 与完整墙钟时延区分 |

## 对中文稿中两个排版引用的处理

- 中文稿中的未解析占位符 `Figure~ref{fig}` 已根据上下文明确为主文 `Figure 1(a)`。
- 中文稿“结果见下图”后实际接续表格，英文版改为对本附录表格的交叉引用，不改变任何实验数据或结论。

## 编译与校验

在本目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error appendix_a_aaai27.tex
```

最终 PDF 已通过以下检查：

- 共 2 页，页面尺寸为 612 × 792 pt，即 US Letter。
- 所有字体均为嵌入式 Type 1 字体，无 Type 3 字体。
- LaTeX 日志中无 overfull box、未解析引用、错误或 float overflow。
- PDF 元数据中无作者、单位、邮箱或本地路径。
- LaTeX 依赖记录中没有输入任何现成 PDF。
- 已逐页渲染检查，表格位于第二页顶部，没有越过页边距或栏间距。
- 最终 PDF SHA-256：`cfb1de366d8b210f715a173186f55201d7e0520cc8f4eff858004ee4095b2102`。
