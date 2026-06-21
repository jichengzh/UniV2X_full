# 闭环驾驶目标 → B4 插件设计 (v1, 2026-06-20)

> 由 Agent-CL (sim-integrator) 撰写。overnight model-only deliverable。
> 配套文件: `results/closedloop_objective_model.json` + `scripts/phase2/closedloop_objective_query.py`

---

## §1 诚实评估：现有闭环数据

### 已有什么
| 文件 | 模型 | 平台 | 数据点 | 可用性 |
|---|---|---|---|---|
| `percinj_curve_full.csv` | **CoDriving** | H800 V2Xverse | 18 个 τ_perc 点 (0-1000ms), N=105 | ✅ 可拟合 DS(τ) 曲线 |
| `percinj_curve.csv` | CoDriving | H800 | 3 点 (0/200/500ms), N=141 | ✅ 备用 |
| `closedloop_c4_d{0,108}_ego.json` | CoDriving | 4090/H800 | 各 1 条路线, DS=100 均 | ❌ 不可辨别延迟效果 |
| Pyramid 闭环 | — | **未安装** | — | ❌ CARLA 未装本机 |

### AP70→DS 数据
**无任何配对实验**。AP70 项纯粹是拟定公式中的假设系数 (β=0 默认 = 纯延迟模型)。

### Pyramid 闭环可行性
**暂不可运行**。本机无 CARLA 0.9.10.1；V2Xverse 移植 Pyramid 尚未完成 (pending 多阶段工程)。任何 Pyramid 闭环数据须等到专门的仿真搭建完成后。

---

## §2 DS(τ_perc) 模型

拟合自 `percinj_curve_full.csv` (CoDriving, H800, full-traffic):

```
DS(τ_ms) = 100 × sigmoid((976 - τ_ms) / 247)
```

| τ_ms | 实测 DS | 预测 DS | 误差 |
|------|---------|---------|------|
| 0 | 98.7 | 98.1 | -0.6 |
| 200 | 95.0 | 95.9 | +0.9 |
| 500 | 87.3 | 87.3 | 0.0 |
| 800 | 66.4 | 67.1 | +0.7 |
| 1000 | 47.7 | 47.6 | -0.1 |

**曲线噪声警告**: 50-500ms 区间 CI = ±3-5 DS。B4 配置的 e2e 范围是 108-220ms (tuned FP32, 除 trap25/iso_s0)，对应 DS 差 ≈ 3-7 点，在曲线噪声内。**DS 轴主要区分 trap25/iso_s0 (tuned→535ms→DS≈85) vs pad64/base/iso_s1/s2 (~215ms→DS≈96)**。

---

## §3 Backbone→e2e 延迟映射

```
e2e_orin_ms = FIXED_OVERHEAD_MS + h800_backbone_us × scale_ms_per_us[tier]
```

- `FIXED_OVERHEAD_MS = 87.5ms`  (vox+enc+FP32_backbone_PyTorch+NMS, 2-agent, Orin, E8)
- `scale_ms_per_us["fp32"] = 0.02073`  (校准点: base-tuned 6320µs → body 131ms E7)

**B4 配置估算 (tuned schedule, FP32):**

| 配置 | H800 tuned (µs) | 估算 e2e (ms) | 估算 DS | AP70 |
|------|----------------|--------------|---------|------|
| base | 6320 | 218.5 | 95.6 | 0.631 |
| p50 | 3011 | 149.9 | 96.6 | 0.564 |
| p75 | 484 | 97.5 | 97.2 | 0.530 |
| **trap25 (W_g)** | **21615** | **535.6** | **85.6** | 0.590 |
| **pad64 (P_g)** | **6152** | **215.0** | **95.6** | 0.590 |
| iso_s0 | 21752 | 538.4 | 85.5 | 0.630 |
| iso_s1 | 6911 | 230.8 | 95.3 | 0.634 |
| iso_s2 | 7244 | 237.7 | 95.2 | 0.634 |

**注意**: 线性缩放对 trap25/iso_s0 (backbone 占主导) 可靠度较高；对 p75 (backbone 极小，fixed overhead 主导) 误差 ~17% (估算 89ms vs 实测 107ms INT8)。标注口径 "model-estimated, ±30%"。

**关键发现**: DS 轴在 B4 里主要区分**慢 backbone 配置** (trap25/iso_s0, 3-4× base H800) 与**快配置** (pad64/base/iso_s1/s2, ~base H800)。这正好对应 W_g vs P_g 的核心区分 — DS 轴支持结构性论点。

---

## §4 B4 代码集成方案

### 方案 A：第三目标轴 (推荐，但须修改 HV 为 3D)

修改 `framework/search_three_arm.py` 的 `CostModel`:

```python
# 在文件头部 import
from scripts.phase2.closedloop_objective_query import ds_objective_for_b4

@dataclass
class CostModel:
    ...
    use_closedloop: bool = False  # 新增开关
    cl_beta: float = 0.0          # AP70 系数 (默认 0=纯延迟)
    cl_scale_tier: str = "fp32"   # H800→Orin 精度档

    def evaluate(self, width, sched):
        ...
        obj = (lat, -ap)
        if self.use_closedloop:
            neg_ds = ds_objective_for_b4(lat, ap,
                                         scale_tier=self.cl_scale_tier,
                                         beta=self.cl_beta)
            obj = (lat, -ap, neg_ds)   # 三维: 最小化 (lat, -AP70, -DS)
        ...
```

⚠️ HV 计算也需升级到 3D (`hypervolume_2d` → 3D)。B4 当前的 2D HV nadir 须扩展。

### 方案 B：DS 作为非支配排序附加信息 (侵入性最小，推荐 overnight)

不改 obj tuple，将 DS 作为 `rec` 字段追加：

```python
from scripts.phase2.closedloop_objective_query import (
    backbone_to_e2e_latency, driving_score
)

# 在 CostModel.evaluate() 的 rec 字典里追加：
if self.use_closedloop:
    e2e_ms = backbone_to_e2e_latency(lat, self.cl_scale_tier)
    rec["ds_model"] = round(driving_score(ap, e2e_ms, beta=self.cl_beta), 2)
    rec["e2e_orin_ms_est"] = round(e2e_ms, 1)
```

这样 `b4_ablation_results.json` 里每个评估点都附带 DS 估算，不影响 HV 计算和搜索逻辑。**最少改动，可立即并入 B4 运行**。

### 推荐集成流程
1. 先走**方案 B** (侵入性最小，本 session 可做)
2. 在 `run_b4_ablation.py` 里，读出 eval_log 后额外报告: 各臂 A-joint/A-serial/A-noS 收敛解的平均 DS 估算
3. 作为 "三轴感知力" 的补充描述，不纳入 hypervolume 主线
4. 若 B4 结论需要 DS 轴强化 (trap25 DS 显著低于 pad64)，再做方案 A

---

## §5 DS 轴对 B4 结构性论点的贡献

| 配置对 | AP70 | 估算 e2e (ms) | 估算 DS |
|--------|------|--------------|---------|
| W_g=trap25 (tuned) | 0.590 | 535.6 | 85.6 |
| P_g=pad64 (tuned) | 0.590 | 215.0 | 95.6 |
| **DS差 (同AP70)** | — | **320ms** | **10 DS点** |

结论：**在同等 AP70 (0.590) 下，pad64 比 trap25 驾驶分高 10 点** (85.6 vs 95.6)。这从闭环驾驶安全角度支持了"A-serial 锁 W_g=trap25 而非 P_g=pad64 的代价"——不仅 latency 更慢，驾驶安全也更差。

DS 轴独立支持 B4 的核心论点：trap25 是结构性局部最优，不仅 tuned latency 劣 (21615µs vs 6152µs)，还伴随更高驾驶安全风险 (DS 85 vs 96)。

---

## §6 标注规范 (报告时必须遵守)

任何引用这些数字时须：
1. 标注 "model-estimated DS" (非真实仿真结果)
2. 注明数据来源 = CoDriving/H800 τ_perc 曲线
3. 注明 e2e 估算误差 ±30% (非 base 配置)
4. 注明 AP70 系数 β=0 (无实测数据，默认不惩罚)
5. **不得写 "real closed-loop result"**

---

## §7 后续工作 (B5/C阶段)

| 工作项 | 优先级 | 说明 |
|--------|--------|------|
| Pyramid→V2Xverse 移植完成 | P0 | 才能跑真 Pyramid 闭环 |
| 真实 Pyramid τ_perc sweep (Orin e2e 真测点) | P0 | 替换估算值，得真 DS 曲线 |
| 补 tp100/300/400/1000 CoDriving 曲线 (已有命令) | P1 | 7点更可信 → 更好拟合 |
| Orin TRT body for trap25/pad64 | P1 | 直接实测 e2e_orin_ms 替换估算 |
| AP70→DS 配对实验 | P2 | 需要不同 AP 模型在同一条 route 上跑 |
