# CoDriving DS 预测地图 — DS = f(AP, latency)

**用途**: 拿到某感知算法的 (vehicle AP50, 感知延迟 τ_perc ms),快速预测它在 CoDriving 闭环里的驾驶分 DS。

## 文件
- `ds_ap_latency_map.png` — 地图(填色等高线,真实坐标轴 latency×AP50,DS 颜色,实测点标注)。
- `ds_ap_latency_merged.csv` — 31 个实测点(主网格 25 + 交互细延迟 6)。
- `ds_map_build.py` — 建图脚本 + **查询函数 `predict_ds(ap50, latency_ms)`**。

## 查询
```python
from ds_map_build import predict_ds
predict_ds(0.84, 400)   # -> ~67  (高AP, 中延迟, 安全高原)
predict_ds(0.36, 700)   # -> ~31  (崖口右侧, 灾难台地)
```
凸包外(AP<0.28 或 >0.84,latency>800)回退最近实测点。

## 地图怎么读(核心结构)
1. **延迟是主导轴,崖口 ≈625ms**: 延迟 ≤600ms = 健康高原(DS 55-85);延迟 ≥650ms = 灾难台地(DS ~26-37),catastrophe(DS<10)率 39-56%。崖口陡(600→650 之间 DS 从 ~78 掉到 ~28),不是缓降。
2. **AP 是高原上的二级调制**: 延迟安全时,AP50 0.84→0.28 使 DS 在 ~85→55 间变化(噪声大、非单调,过度保守地板)。**过了崖口 AP 无关**(延迟灾难压倒一切,DS ~28 无论 AP)。
3. **预测要点**: 先看延迟落在崖口哪侧 —— <625ms 则 DS≈高原值(查 AP);≥625ms 则 DS≈28(灾难,AP 无关)。

## 口径与 caveat(必读)
- **DS = honest composed DS**: `score_composed`,**超时(ego 卡死跑不完)记 DS=0**,clean6 路(3/17/18/104/136/317)× N=18(主网格)/36(交互)。
- **延迟轴 = τ_perc(车端自身感知延迟)**;AP 轴 = **融合后** AP50(ego+RSU 融合输出退化,非 ego-only)。drop→AP50 标定: drop 0/.25/.5/.7/.85 = AP50 0.841/0.783/0.559/0.360/0.281。
- **AP 轴在 composed DS 上信号弱**(N 内方差大 + 过度保守地板掩盖)。AP 的**干净**安全信号在**碰撞率**上(见 `ap_tau_interaction_collision.png`: 低AP 让碰撞崖口 750→650ms 提前),composed DS 把它和"堵车不撞"的地板混在一起了。
- 仅 Town05、town05_short_collab、满交通 `_1`;RSU 延迟臂(latency_inject_ms)与 τ_ego 未并入(=0)。
- 未测区间为线性插值,非物理模型;崖口精确位置在 600-650 间(只采到这两端 + 650),真实拐点可能在 610-640。

## 复跑
`ds_map_build.py` 数据内联,直接 `python ds_map_build.py` 重生成图 + 查询样例。源数据: 主网格 `ap_tau_grid_ds.csv`,交互 H800 `/tmp/ix_ds.py`(honest_DS 重算)。
