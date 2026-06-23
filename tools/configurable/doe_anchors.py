"""DoE anchor 设计 (数据生成师交付物 2).

小规模高信息量实验设计 (Design of Experiments). 不纯随机, 用三类策略:

  1. **基线锚 (baseline anchor)**: FP32 / FP16 / INT8 三档 baseline (prune_rate=0),
     校准 latency/AP/engine_size 的绝对参照点. 没有它, 所有"加速倍率"无锚.
  2. **敏感度分层 (sensitivity-stratified)**: 沿每个维度轴 (prune_rate / q_bits /
     calibrator / granularity / W-only) 单独扫, 其余维度锚定 baseline. 这是
     OAT (one-at-a-time) 灵敏度分析 — 用最少点拿到每个轴的边际效应, 喂 LGB 的主效应.
  3. **Pareto 边界加密 (Pareto-edge densification)**: 在已知 Pareto 拐点附近
     (prune50+INT8, prune75+FP16) 加密交互锚 — 这些是 latency 预测器跨数量级
     崩塌 (MEMORY: f_lat ceiling ~0.73) 最需要的交互项样本.
  4. **跨硬件镜像 (cross-hw mirror)**: 同一 B+Q 配置在 4090 / Orin 各一份, 给
     latency 预测器跨硬件外推的配对样本 (带宽 5× 差距是已知外推难点).

为什么"小而高信息" (验收师会问):
  - **信息增益**: OAT 主效应 + 拐点交互项覆盖了 LGB GBDT 最需要的低阶项;
    全笛卡尔积里 99% 的点是冗余的内部插值点 (LGB 能内插, 不需真测).
  - **覆盖边角**: baseline + 最激进剪枝 (p75) + 最低精度 (INT8) 锚定了响应面的
    四个角, 避免预测器在边界外推时无约束发散.
  - **避免 selection bias**: 纯随机采样会过采样"中间配置" (因约束过滤后中间区
    通过率最高), 欠采样边界. 分层设计强制覆盖边界.

每个 anchor 要测的标签 (对齐 e2e_bench_v1_schema.md):
  - latency: throughput_fps (+ bench JSON 三段 lat_trt/postproc/e2e)
  - AP: ap30 / ap50 / ap70 (DAIR val sweep, INT8 走 e2e engine 实测)
  - 资源: engine_size_mb / build_secs / params (剪枝 manifest)

注意: 标签的真实采集需 GPU 跑 TRT build + AP eval; 本文件只定义 anchor 矩阵.
真采集由 generate_dataset.py 调三个工具完成 (跑不动标 dry_run).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from framework.config_schema import Config, PYRAMID_M1_MODULES  # noqa: E402


# Pyramid 单模块名 (M4.9 反思 #21: backbone+heads 合并为 "model")
M = "model"

# 剪枝率 → triplet 标签 (对齐 schema §一 triplet 命名)
# 这些是已有真实 ckpt 的剪枝档 (checkpoints/stage1/Pyramid_DAIR_m1_pruned{25,50,75})
PRUNE_TRIPLET = {
    0.0: ("T1_base", [64, 128, 256]),
    0.25: ("T2_p25", [48, 96, 192]),
    0.50: ("T4_p50", [32, 64, 128]),
    0.75: ("T6_p75", [16, 32, 64]),
}


@dataclass(frozen=True)
class Anchor:
    """一个 DoE 锚点 = (Config, 元数据). Config 喂三个工具, 元数据写入数据行."""

    anchor_id: str
    config: Config
    hardware: str          # rtx4090 / orin_agx
    strategy: str          # baseline / sens_prune / sens_quant / sens_calib /
                           # sens_gran / sens_wonly / pareto_edge / cross_hw
    triplet: str
    note: str = ""


def _cfg(prune_rate: float, bits: str, *, gran: str = "none", obj: str = "none",
         calib: str = "none", prune_obj: str = "channel", crit: str = "L1",
         routing: str = "GPU", cid: str = "") -> Config:
    """构造单模块 Pyramid Config (省去重复 dict 样板)."""
    po = prune_obj if prune_rate > 0 else "none"
    pc = crit if prune_rate > 0 else "none"
    return Config(
        prune_rate={M: prune_rate},
        prune_object=po,
        prune_criterion={M: pc},
        q_bits={M: bits},
        q_granularity={M: gran},
        q_object={M: obj},
        q_calibrator={M: calib},
        d_routing={M: routing},
        config_id=cid,
        source="doe_v1",
    )


def build_doe(hardware: str = "rtx4090") -> list[Anchor]:
    """构造一份 hardware 的 DoE anchor 清单 (单模型 Pyramid).

    返回 anchor 列表. 跨硬件镜像由调用方对两个 hardware 各跑一次 build_doe 实现.
    """
    anchors: list[Anchor] = []

    def add(aid, cfg, strategy, triplet, note=""):
        cfg = cfg.with_field(config_id=f"{hardware}_{aid}")
        anchors.append(Anchor(f"{hardware}_{aid}", cfg, hardware, strategy, triplet, note))

    # --- 策略 1: 基线锚 (3 档精度, 不剪枝) ---
    add("base_fp32", _cfg(0.0, "FP32"), "baseline", "T1_base", "FP32 绝对参照")
    add("base_fp16", _cfg(0.0, "FP16", gran="per-tensor", obj="W+A"),
        "baseline", "T1_base", "FP16 主路径参照")
    add("base_int8", _cfg(0.0, "INT8", gran="per-channel", obj="W+A", calib="minmax"),
        "baseline", "T1_base", "INT8 minmax 参照")

    # --- 策略 2a: 剪枝率灵敏度 (OAT, FP16 锚定) ---
    for r in (0.25, 0.50, 0.75):
        trip = PRUNE_TRIPLET[r][0]
        add(f"sens_prune_{int(r*100)}", _cfg(r, "FP16", gran="per-tensor", obj="W+A"),
            "sens_prune", trip, f"剪枝率轴 OAT @ FP16 (p{int(r*100)})")

    # --- 策略 2b: 校准器灵敏度 (OAT, INT8 base, minmax vs percentile) ---
    add("sens_calib_pct", _cfg(0.0, "INT8", gran="per-channel", obj="W+A",
                               calib="percentile_99_99"),
        "sens_calib", "T1_base", "校准器轴: percentile_99_99 (entropy 禁用)")

    # --- 策略 2c: 量化粒度灵敏度 (OAT, INT8 base, per-tensor weight) ---
    add("sens_gran_pt", _cfg(0.0, "INT8", gran="per-tensor", obj="W+A", calib="minmax"),
        "sens_gran", "T1_base", "粒度轴: weight per-tensor (默认 per-channel)")

    # --- 策略 2d: W-only 灵敏度 (OAT, INT8 base) ---
    add("sens_wonly", _cfg(0.0, "INT8", gran="per-channel", obj="W-only", calib="minmax"),
        "sens_wonly", "T1_base", "量化对象轴: W-only (默认 W+A)")

    # --- 策略 3: Pareto 边界加密 (剪枝 × 量化 交互项) ---
    add("pareto_p50_int8", _cfg(0.50, "INT8", gran="per-channel", obj="W+A",
                                calib="minmax"),
        "pareto_edge", "T4_p50", "拐点: 50% 剪枝 + INT8 (协同加速主卖点)")
    add("pareto_p75_int8", _cfg(0.75, "INT8", gran="per-channel", obj="W+A",
                                calib="minmax"),
        "pareto_edge", "T6_p75", "拐点: 75% 剪枝 + INT8 (最激进, AP 风险)")
    add("pareto_p25_int8", _cfg(0.25, "INT8", gran="per-channel", obj="W+A",
                                calib="minmax"),
        "pareto_edge", "T2_p25", "拐点: 25% 剪枝 + INT8 (低损区)")

    # --- 策略 4: 跨硬件镜像 (仅 Orin 加 DLA 路由档, 4090 无 DLA) ---
    if hardware == "orin_agx":
        add("cross_dla_fp16", _cfg(0.0, "FP16", gran="per-tensor", obj="W+A",
                                   routing="DLA0"),
            "cross_hw", "T1_base", "Orin DLA0 路由 (4090 无此轴)")

    return anchors


def build_all_doe() -> list[Anchor]:
    """两个硬件的完整 DoE (4090 + Orin), 用于跨硬件配对."""
    return build_doe("rtx4090") + build_doe("orin_agx")


if __name__ == "__main__":
    import json
    all_a = build_all_doe()
    print(f"DoE 总 anchor 数: {len(all_a)}")
    by_strat: dict[str, int] = {}
    by_hw: dict[str, int] = {}
    for a in all_a:
        by_strat[a.strategy] = by_strat.get(a.strategy, 0) + 1
        by_hw[a.hardware] = by_hw.get(a.hardware, 0) + 1
    print("按策略:", json.dumps(by_strat, ensure_ascii=False))
    print("按硬件:", json.dumps(by_hw, ensure_ascii=False))
    print("\nanchor 清单:")
    for a in all_a:
        c = a.config
        print(f"  [{a.anchor_id:28s}] {a.strategy:12s} {a.triplet:9s} "
              f"pr={c.prune_rate[M]:.2f} bits={c.q_bits[M]:5s} "
              f"gran={c.q_granularity[M]:11s} obj={c.q_object[M]:7s} "
              f"calib={c.q_calibrator[M]:15s} route={c.d_routing[M]}")
