# multi_agent/figure — 数据描述性统计图

> 生成脚本: `make_data_figures.py`(读 `../data/dataset_v2.csv`)。
> **数据更新后**(重跑 `../data/build_dataset_v2.py`)**再跑本脚本即可刷新图**:
> `python multi_agent/figure/make_data_figures.py`
> 图用英文标签(避免 matplotlib 中文缺字)。输出在 `data/`。

## data/ 下的图
| 文件 | 内容 | 看什么 |
|---|---|---|
| `fig1_pareto_latAP_collab2.png` | 3 triplet 的 lat-AP 散点 + Pareto 前沿(collab2 口径)| **前沿只有 int8-auto / fp16-auto 两绿点;蓝(per-stage 混精)红(forced)全被支配** |
| `fig2_ap_vs_prune.png` | AP@0.5 vs prune_rate(单 agent body,FP16 vs INT8)| **INT8 近无损**(与 FP16 几乎重合);AP 随剪枝率下降 |
| `fig3_delta_ap_perstage.png` | per-stage 混精 ΔAP vs forced-all-INT8(分 triplet 柱)| **T_prune50p 上 c3/c5/c7(保 stage1 FP16)+0.022~0.025;c4/c6/c8(stage1 INT8)贴噪声线** |
| `fig4_inventory.png` | 数据清单(按 latency_kind / q_mode / dataset_src)| 33 完整点构成;**两种 latency 口径**(6 body_subnet + 27 collab2) |

## 口径提醒
- fig1/fig3 用 `body_subnet_collab2`(双 agent);fig2 用 `body_subnet`(单 agent)。两口径不可混比(见 `../data/schema_v2.md`)。
