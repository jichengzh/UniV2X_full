# HANDOFF: AP x latency DS map + coupling evidence (v1, 2026-06-24)

> 目的: 在清理上下文前, 固化当前 CoDriving `AP x latency` 闭环实测状态。本文把 **DS 地图** 与 **碰撞崖口前移证据** 分开记录, 防止后续把 composed DS 结论和 safety/collision 结论混用。

---

## 0. 当前一句话结论

1. **DS 地图已经有完整高原补齐数据**: `latency={0,100,200,300,400,450,500,550,600,800}ms x AP={0.841,0.783,0.559,0.360,0.281}` 均有实测 DS; `650/700/750ms` 目前只有 `AP=0.841` 和 `AP=0.360` 两条交互臂有实测 DS。
2. **DS 表本身没有清楚显示"高 AP 崖口后移"**: 650ms 后高 AP/低 AP 的 composed DS 都掉到约 27-31。
3. **之前"崖口前移(750 -> 650ms)"的证据来自碰撞率, 不是 DS**: 低 AP 在 650ms 已到 37.9% 有碰撞局, 高 AP 到 750-800ms 才约 25% 有碰撞局。
4. **下一步最该补的是崖口带缺口**: `650/700/750ms x AP={0.783,0.559,0.281}` 的 DS/RC/collision, 再决定是否做更细安全阈值和 RSU/ego latency 轴。

---

## 1. 数据源索引

### 1.1 本地源文件

- 主 5x5 DS 网格: `multi_agent/real_test/ap_tau_grid_ds.csv`
- 主网格分析脚本: `multi_agent/real_test/ap_tau_grid_analyze.py`
- 交互实验子指标 CSV: `multi_agent/real_test/ap_tau_interaction_results.csv`
- 交互实验碰撞图: `multi_agent/real_test/ap_tau_interaction_collision.png`
- 旧 DS 地图脚本: `multi_agent/real_test/ds_map_build.py`
- 旧 DS 地图合并 CSV: `multi_agent/real_test/ds_ap_latency_merged.csv`
- 旧 DS 地图说明: `multi_agent/real_test/README_ds_map.md`

### 1.2 交接文档证据

- 主网格结论: `multi_agent/methods/progress/HANDOFF_ap_tau_grid_v1.md`
  - §3c: 完整 5x5 DS 网格。
  - 关键 caveat: composed DS 在 AP 轴上非单调, 需要拆 RC/collision。
- 交互实验结论: `multi_agent/methods/progress/HANDOFF_ap_tau_interaction_v1.md`
  - §0.5: 低 AP 让 **碰撞崖口** 提前, 不是 DS 崖口。
  - §1: 实验主指标明确为 **碰撞率 + RC**, 不用 composed DS。

### 1.3 H800 raw results 路径

统一路径模式:

```text
/exdata/jichengzhi/V2Xverse_apknob/results/
  results_driving_grid_g{code}_r{route}_n{rep}/
  v2x_final/town05_short_collab/*/ego_vehicle_0/results.json
```

DS 字段:

```text
_checkpoint.global_record.scores.score_composed
```

honest DS 口径:

- `TIMEOUT_SKIP` 或缺少 `score_composed` -> DS=0。
- 其它状态使用 `score_composed`。
- clean6 routes: `[3,17,18,104,136,317]`。

---

## 2. 本轮新高原补齐实验

### 2.1 实验范围

- latency: `100, 300, 450, 500, 550ms`
- AP/drop:
  - drop 0.00 -> AP50 0.841
  - drop 0.25 -> AP50 0.783
  - drop 0.50 -> AP50 0.559
  - drop 0.70 -> AP50 0.360
  - drop 0.85 -> AP50 0.281
- route: clean6 `[3,17,18,104,136,317]`
- N: 3 per route, 即 18 episode/cell
- 总量: `5 latency x 5 AP x 6 routes x 3 reps = 450`
- 启动标签/log: `/exdata/jichengzhi/dsmap_plateau_v2.log`
- 最终完成: `450/450 new + 0 skipped`

### 2.2 完成状态

| 总数 | Completed | Failed | TIMEOUT_SKIP | Missing | Other |
|---:|---:|---:|---:|---:|---:|
| 450 | 440 | 4 | 6 | 0 | 0 |

异常 episode:

- Timeout:
  - `(30025,136,2)`
  - `(45025,18,1)`
  - `(45070,17,1)`
  - `(45070,18,2)`
  - `(45070,18,3)`
  - `(50025,17,1)`
- Failed:
  - `(30050,18,3)`
  - `(50025,317,3)`
  - `(50070,3,1)`
  - `(55025,136,1)`

### 2.3 本轮新增 DS 表

表内为 honest DS。

| AP50 \ latency ms | 100 | 300 | 450 | 500 | 550 |
|---:|---:|---:|---:|---:|---:|
| 0.841 | 91.7 | 68.1 | 64.2 | 73.3 | 81.6 |
| 0.783 | 77.8 | 67.8 | 63.1 | 65.1 | 61.9 |
| 0.559 | 72.2 | 68.6 | 63.4 | 56.0 | 54.9 |
| 0.360 | 75.0 | 58.3 | 48.1 | 64.3 | 58.6 |
| 0.281 | 75.0 | 62.5 | 72.9 | 82.6 | 85.5 |

---

## 3. 当前完整 AP x latency DS 地图

表内为 raw results 汇总出的 honest DS。`-` 表示该 AP/latency cell 目前没有实测, 不能当作插值结果。

注意: `600/800ms` 的高 AP/低 AP 端点在主网格和交互实验里都有重复测量。下表为了和 `650/700/750ms` 交互臂一致, 对 `AP=0.841/0.360` 的 `600/800ms` 采用交互实验 N=36 版本; 其它格为主网格或高原补齐 N=18。

| AP50 \ latency ms | 0 | 100 | 200 | 300 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.841 | 85.3 | 91.7 | 84.7 | 68.1 | 67.2 | 64.2 | 73.3 | 81.6 | 77.9 | 28.0 | 28.2 | 30.2 | 26.6 |
| 0.783 | 68.5 | 77.8 | 83.3 | 67.8 | 66.9 | 63.1 | 65.1 | 61.9 | 65.9 | - | - | - | 37.2 |
| 0.559 | 56.4 | 72.2 | 75.0 | 68.6 | 68.5 | 63.4 | 56.0 | 54.9 | 76.8 | - | - | - | 23.1 |
| 0.360 | 63.8 | 75.0 | 69.4 | 58.3 | 54.6 | 48.1 | 64.3 | 58.6 | 58.5 | 26.9 | 31.3 | 29.9 | 27.5 |
| 0.281 | 66.7 | 75.0 | 60.0 | 62.5 | 62.5 | 72.9 | 82.6 | 85.5 | 81.3 | - | - | - | 25.6 |

重复端点对照:

| Cell | 主网格 N=18 | 交互 N=36 |
|---|---:|---:|
| AP0.841, 600ms | 77.4 | 77.9 |
| AP0.360, 600ms | 59.2 | 58.5 |
| AP0.841, 800ms | 26.2 | 26.6 |
| AP0.360, 800ms | 27.7 | 27.5 |

结论: 重复端点一致性很好, 可以合并使用; 但 `650/700/750ms` 仍只有两条 AP 臂。

---

## 4. 交互实验的碰撞耦合证据

这部分是之前“低 AP 让崖口前移”的正式证据。它不是 DS 表结论。

来源: `multi_agent/real_test/ap_tau_interaction_results.csv` 和 `HANDOFF_ap_tau_interaction_v1.md`。

### 4.1 碰撞局率

| AP50 \ latency ms | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|
| 0.841 高 AP | 0.0 | 15.6 | 13.3 | 25.8 | 25.0 |
| 0.360 低 AP | 5.6 | 37.9 | 25.8 | 36.4 | 38.7 |

### 4.2 碰撞数 / episode

| AP50 \ latency ms | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|
| 0.841 高 AP | 0.000 | 0.061 | 0.024 | 0.093 | 0.076 |
| 0.360 低 AP | 0.008 | 0.163 | 0.111 | 0.177 | 0.197 |

### 4.3 RC

| AP50 \ latency ms | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|
| 0.841 高 AP | 97.7 | 61.2 | 66.4 | 64.8 | 65.7 |
| 0.360 低 AP | 100.0 | 67.9 | 66.7 | 64.5 | 68.0 |

### 4.4 正确解读

- `DS` 视角: 650ms 后两条 AP 臂都已经掉到约 27-31, 所以 composed DS 不显示清楚的 AP-dependent cliff shift。
- `collision/safety` 视角: 低 AP 在 650ms 已经到 37.9% 有碰撞局; 高 AP 到 750-800ms 仍约 25%。因此“崖口前移”应写成:

```text
低 AP 让碰撞/安全崖口从约 750ms 提前到 650ms;
但 composed DS 崖口没有显示清晰 AP 位移。
```

---

## 5. 当前耦合地图存在的问题

### 5.1 崖口带覆盖不足

`650/700/750ms` 目前只测了:

- AP0.841 / drop0.00
- AP0.360 / drop0.70

缺失:

- AP0.783 / drop0.25
- AP0.559 / drop0.50
- AP0.281 / drop0.85

缺口总数: `3 latency x 3 AP = 9 cells`。这些正好位于最重要的 cliff band。

### 5.2 DS 与 safety/collision 指标讲的是不同事情

- composed DS 混合了 RC、罚分、碰撞、超时。
- AP 退化会造成 planner 过度保守, 形成“堵车不撞”的 DS 地板。
- 所以 DS 可以做综合驾驶分地图, 但不适合单独支持“崖口前移”的 safety 论断。
- “崖口前移”应绑定 collision/collision-episode-rate。

### 5.3 高原补齐后仍存在非单调和噪声

新高原数据中出现多个非单调点, 例如:

- AP0.281 在 500/550ms 反而 DS 很高: 82.6 / 85.5。
- AP0.360 在 450ms 掉到 48.1, 但 500ms 又回到 64.3。

这与既有主网格结论一致: AP 轴在 composed DS 上不干净, 需要拆 RC/collision/timeout。

### 5.4 当前图像产物可能已过期

`multi_agent/real_test/ds_ap_latency_map.png` 与 `ds_map_build.py` 是高原补齐前的产物, 只包含:

- 主网格 25 点
- 交互细延迟 6 点

它没有包含本轮新测的 `100/300/450/500/550ms` 25 个高原点。后续出图前必须更新脚本/CSV。

### 5.5 latency 注入有 50ms 帧量化

`tau_perc_ms` 实际通过 `ceil(tau_ms / 50)` 转换为延迟帧数。因此:

- 600ms -> 12 帧
- 650ms -> 13 帧
- 625ms 也会变成 13 帧, 等价于 650ms

如果想分辨 600-650ms 之间更细的真实拐点, 不能只加 625ms; 需要改延迟注入机制或仿真 tick/队列逻辑。

### 5.6 尚未覆盖的维度

- RSU latency arm 未并入。
- `tau_ego` 控制延迟未并入。
- 只在 Town05 / `town05_short_collab` / full traffic `_1` 下测。
- AP 范围只到 0.281-0.841。
- 模型只覆盖 CoDriving, 尚非 Pyramid/V2X-ViT 专属 DS 曲面。

---

## 6. 建议的下一阶段实验

### 6.1 Phase A: 崖口带 AP 补齐, 优先级最高

目标: 把最关键的 cliff band 从“两条 AP 臂”补成完整 5 AP。

待测 cells:

| latency ms | drop | AP50 | code |
|---:|---:|---:|---:|
| 650 | 0.25 | 0.783 | g65025 |
| 650 | 0.50 | 0.559 | g65050 |
| 650 | 0.85 | 0.281 | g65085 |
| 700 | 0.25 | 0.783 | g70025 |
| 700 | 0.50 | 0.559 | g70050 |
| 700 | 0.85 | 0.281 | g70085 |
| 750 | 0.25 | 0.783 | g75025 |
| 750 | 0.50 | 0.559 | g75050 |
| 750 | 0.85 | 0.281 | g75085 |

建议 N:

- 最低: clean6 x N=6 = 36/cell, 与现有交互实验对齐。
- 总量: `9 x 36 = 324 episodes`。
- 指标: 同时输出 DS / RC / collision episode rate / collision per episode / timeout。

### 6.2 Phase B: 更新地图产物

完成 Phase A 后再更新以下文件, 不建议现在就把插值图当最终图:

- `ds_ap_latency_all_measured.csv`
- `ds_ap_latency_map_v2.png`
- `ap_tau_collision_map_v2.png`
- `README_ds_map.md`
- `ds_map_build.py` 或新建 `ds_map_build_v2.py`

地图应至少分两张:

1. `DS(AP, latency)` 综合驾驶分地图。
2. `collision(AP, latency)` 安全崖口地图。

这两张图的结论不可混写。

### 6.3 Phase C: 如果要精确拐点, 先修注入分辨率

当前 `tau_perc_ms` 是 50ms 帧量化, 625ms 不会提供新信息。若要定位 600-650ms 内真实拐点:

1. 审计/修改 delay buffer, 支持亚帧或更高频采样;
2. 或把仿真/感知输入 tick 调整到更小步长;
3. 再测 610/620/630/640ms。

否则只应说“崖口位于 600-650ms 之间”, 不应声称精确到 625ms。

### 6.4 Phase D: 扩展到 PQS 全维

在 AP x tau_perc 曲面稳定后, 再扩:

- RSU latency arm: `latency_inject_ms`
- ego control latency: `tau_ego`
- 模型/结构轴: Pyramid / V2X-ViT / CoDriving variants
- 压缩轴: pruning / quantization / schedule

---

## 7. 后续接手检查清单

1. 不要把 `ap_tau_interaction_results.csv` 当 DS 表; 它是 collision/RC 表。
2. 650/700/750 的 DS 是从交互 raw `results.json` 里按 `score_composed` 重算出来的, 本地只保存于 `ds_map_build.py` / `ds_ap_latency_merged.csv`。
3. “崖口提前”只引用 collision 证据: `HANDOFF_ap_tau_interaction_v1.md` §0.5。
4. 目前 `ds_ap_latency_map.png` 旧, 未包含本轮高原补齐 25 点。
5. 下一步不要先跑高原; 高原已补齐。应先跑 `650/700/750 x AP0.783/0.559/0.281`。
6. 若要继续用 GPU, 之前 GPU6 CARLA smoke 不稳; 高原补齐实际使用 GPU1-5 完成。

