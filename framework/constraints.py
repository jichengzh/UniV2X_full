"""硬约束 DSL (Phase 1A.4)

对应 v1.5 §0.3 三类约束 + Phase1_2 §2.3 关键设计.

约束分类:
- physical: 物理硬约束 (违反 build 失败) — 不写它也存在,搜索器**必须**硬过滤
- empirical: 经验硬约束 (违反能跑但性能差) — 来自实测拐点,可硬过滤,Phase 1B 后可能放宽
- soft: 软约束 (偏好引导) — 搜索器作 penalty 而非过滤

每条约束实现为可调用对象:
    check(config, hw) -> (is_ok: bool, reason: str)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config

ConstraintKind = Literal["physical", "empirical", "soft"]


@dataclass(frozen=True)
class Constraint:
    name: str
    kind: ConstraintKind
    check: Callable[[Config, HardwareCapability], tuple[bool, str]]


# ============================================================
# 物理硬约束 (PHYSICAL) — 违反就 build 失败
# ============================================================

def _check_trt_gpu_symmetric(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """TRT GPU 强制对称量化 (来自 v1.5 §0.2 锁定维度 / 1.1 实验结论)."""
    return True, ""  # Config 当前没建模 symmetric 字段,默认对称,所以总是通过


def _check_dla_per_tensor(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """DLA 路由的模块强制 per-tensor (DLA 不支持 per-channel)."""
    if not hw.has_dla:
        return True, ""
    for m, route in cfg.d_routing.items():
        if route.startswith("DLA") and cfg.q_granularity.get(m, "none") == "per-channel":
            return False, f"模块 {m} 路由到 {route} 但量化粒度=per-channel,DLA 不支持"
    return True, ""


def _check_dla_int8_only(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """DLA 模块若用了 FP32,直接非法 (DLA 只支持 FP16/INT8)."""
    if not hw.has_dla:
        return True, ""
    dla_precs = set(hw.ips.get("dla").precisions) if "dla" in hw.ips else set()
    for m, route in cfg.d_routing.items():
        if route.startswith("DLA"):
            bits = cfg.q_bits.get(m, "FP32")
            if bits not in dla_precs:
                return False, f"模块 {m} 路由到 {route} 但精度={bits} 不在 DLA 支持集 {sorted(dla_precs)}"
    return True, ""


def _check_precision_supported(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """每个模块的位宽都得在所路由 IP 的精度集合中."""
    for m, bits in cfg.q_bits.items():
        route = cfg.d_routing.get(m, "GPU")
        ip_key = "dla" if route.startswith("DLA") else "gpu"
        if ip_key not in hw.ips:
            return False, f"硬件 {hw.name} 不存在 IP {ip_key} (模块 {m} 路由 {route})"
        if bits not in hw.ips[ip_key].precisions:
            return False, f"模块 {m} 精度 {bits} 不在 {ip_key} 支持集 {hw.ips[ip_key].precisions}"
    return True, ""


def _check_dla_count(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """请求的 DLA core 数不能超过硬件实际数."""
    requested = {r for r in cfg.d_routing.values() if r.startswith("DLA")}
    if not requested:
        return True, ""
    needed = max((int(r.replace("DLA", "")) for r in requested), default=-1) + 1
    # v1.1 schema 兼容: 优先 features.dla_count, fallback ips.dla.count (extra=allow)
    actual = hw.features.dla_count
    if actual == 0 and "dla" in hw.ips:
        dla_extra = hw.ips["dla"].model_extra or {}
        actual = dla_extra.get("count", 0)
    if needed > actual:
        return False, f"配置请求 DLA{needed-1} 但硬件只有 {actual} 个 DLA"
    return True, ""


def _check_sparse_tc_support(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """2:4 N:M 剪枝必须有 Sparse Tensor Core 支持."""
    if cfg.prune_object == "2:4" and not hw.features.sparse_tc:
        return False, f"prune_object=2:4 但硬件 {hw.name} 不支持 Sparse Tensor Core"
    return True, ""


PHYSICAL_CONSTRAINTS: list[Constraint] = [
    Constraint("trt_gpu_symmetric", "physical", _check_trt_gpu_symmetric),
    Constraint("dla_per_tensor_only", "physical", _check_dla_per_tensor),
    Constraint("dla_precision_subset", "physical", _check_dla_int8_only),
    Constraint("ip_precision_supported", "physical", _check_precision_supported),
    Constraint("dla_count_within_hw", "physical", _check_dla_count),
    Constraint("sparse_tc_support", "physical", _check_sparse_tc_support),
]


# ============================================================
# 经验硬约束 (EMPIRICAL) — 违反能跑但性能差,Phase 1B 后可能放宽
# ============================================================

def _check_channel_alignment(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """通道数对齐 INT8 TC (默认 32).

    规则: 假设原始通道数 C,剪枝率 r,则 round(C*(1-r)) 必须是 32 的倍数.
    实现简化: 检查 (1 - prune_rate) * 256 % 32 == 0 (假定原始 256 通道)
    Phase 1B.6 拐点测量后会改写为更精确的版本.
    """
    align = hw.alignment.int8_channel
    # 简化:剪枝率必须落在 (32k/256) 这种网格上,k 为整数
    # 即 1 - rate 必须能写成 k/8 的形式 (256/32 = 8)
    for m, rate in cfg.prune_rate.items():
        if rate == 0.0:
            continue
        kept = 1.0 - rate
        # 容差 1e-6
        if abs(round(kept * 8) - kept * 8) > 1e-6:
            return False, f"模块 {m} 剪枝率 {rate:.3f} 后通道数未对齐到 {align} 倍数"
    return True, ""


def _check_k_dim_for_2to4(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """2:4 剪枝要求 K 维度 >= 64 (来自 survey_raw_2 / cuSPARSELt 实测)."""
    if cfg.prune_object != "2:4":
        return True, ""
    # Config 当前未细到逐层 K 维度,这里仅占位返回通过
    # Phase 1B.6 测完拐点后用模块级 K_dim 元数据填充
    return True, ""


def _check_no_entropy_calibrator(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """禁用 INT8 entropy 校准 (KL散度) — 在 self-trained ckpt 上触发 AP 崩塌.

    决议背景 (2026-05-15, 见 paper_learning/2. AAAI最终故事/data/问题.md §问题 1):
        - 4 个自训 Pyramid ckpt (T3/T5/T7/T8) 用 trt.IInt8EntropyCalibrator2 后 AP50 = 0.02
        - 同 ckpt 用 trt.IInt8MinMaxCalibrator 后 AP50 = 0.62-0.65 (正常)
        - 根因: 自训 ckpt 的 shrink_conv 输出激活分布比官方 ckpt 宽 2-3×, entropy KL
          算法对宽分布过敏, 算出的 clip 过宽 → INT8 量化 step 过粗 → cls head 精度崩塌
        - 实测确认: 多训 30 epoch 不能修复 (方案 3 已验证无效)
        - 决议: 论文主路径用 minmax; entropy 节点禁用 (或仅作 LGB 负样本对照)

    Config.q_calibrator 默认空 dict, 此时该函数返回通过 (fallback 到 minmax).
    显式指定 "entropy" 才拒绝.
    """
    for m, calib in cfg.q_calibrator.items():
        if calib == "entropy":
            return False, (
                f"模块 {m} 选用 INT8 entropy 校准 — 在自训 ckpt 上触发 AP 崩塌 "
                f"(见问题.md §问题 1). 主路径使用 minmax, 备选 percentile_99_99."
            )
    return True, ""


def _check_cross_layer_gradient(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """跨**已剪枝**模块的剪枝率梯度 < 30% (避免打断 fusion).

    v1.5 §0.3 列为经验硬约束. M4.9 v2 修正: 只在多个模块同时被剪枝时检查
    它们之间的差异 (避免单模块 backbone-only 剪枝被错误拒绝).

    Phase 1B.6 + Stage 1.3 拐点验证后可能改判.
    """
    pruned_rates = [r for r in cfg.prune_rate.values() if r > 0.0]
    if len(pruned_rates) < 2:
        # 全 0 或单模块剪枝 — 不存在跨层 gradient
        return True, ""
    spread = max(pruned_rates) - min(pruned_rates)
    if spread > 0.30:
        return False, f"已剪枝模块间率差 {spread:.2f} 超过 0.30"
    return True, ""


# ResNeXt32x4d backbone (Pyramid_m1) baseline planes per stage.
# planes -> width = int(planes * 4 / 64) * 32, groups = 32
# width_per_group = width / 32 = int(planes * 4 / 64)
# IMMA fast-path requires width_per_group ∈ {1, 2, 4, 8, 16, 32}
RESNEXT_BASE_PLANES = (64, 128, 256)
_POW2 = {1, 2, 4, 8, 16, 32, 64}


def _check_resnext_width_pow2(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """ResNeXt32x4d grouped-conv width_per_group 必须是 2 的幂 (Tensor Core IMMA fast-path).

    实证 (M4.9 v2 Pyramid_DAIR_m1, RTX 4090):
        prune  baseline_planes  new_planes  width  width_per_group  hit_fast_path  lat_p50
        0%     64,128,256       64,128,256  128,256,512  4,8,16    ✓             0.81 ms
        25%    64,128,256       48,96,192   96,192,384   3,6,12    ✗             2.71 ms (3.35× 慢!)
        50%    64,128,256       32,64,128   64,128,256   2,4,8     ✓             0.78 ms

    width_per_group=3/6/12 出 cuDNN/TRT 优化的 IMMA grouped-conv 路径,
    fall back 到通用 group conv kernel, 实测 3× 慢 — 算法 -27.8% FLOPs 反而 lat 变长.

    本约束: 只对带 Tensor Core 的硬件激活, 只对 backbone 模块的 channel 剪枝有意义.
    """
    # 没 TC 的硬件不关心 (CPU / DLA per-tensor only)
    has_tc = bool(getattr(hw.features, "tensor_core", False) or
                  any(ip.tensor_core_gen for ip in hw.ips.values() if ip.tensor_core_gen))
    if not has_tc:
        return True, ""
    if cfg.prune_object != "channel":
        return True, ""

    # 只检查 ResNeXt backbone 的剪枝率 (兼容 1-module Pyramid 用 "model" 名,
    # 也兼容 5-module UniV2X 用 "backbone" 名). M4.9 v2 反思 #21.
    rate = float(cfg.prune_rate.get("model",
                                     cfg.prune_rate.get("backbone", 0.0)))
    if rate == 0.0:
        return True, ""

    for planes_base in RESNEXT_BASE_PLANES:
        new_planes = max(8, round(planes_base * (1.0 - rate)))
        # snap to nearest 8-multiple (consistent with HEAL conventions)
        new_planes = (new_planes // 8) * 8
        wpg = int(new_planes * 4 / 64)
        if wpg not in _POW2:
            return False, (
                f"backbone prune_rate={rate:.3f} → planes={new_planes} → "
                f"width_per_group={wpg} 不是 2 的幂 (IMMA fast-path 要求 ∈ "
                f"{sorted(_POW2)}); 实测 3× 慢"
            )
    return True, ""


EMPIRICAL_CONSTRAINTS: list[Constraint] = [
    Constraint("channel_align_32", "empirical", _check_channel_alignment),
    Constraint("k_dim_ge_64", "empirical", _check_k_dim_for_2to4),
    Constraint("cross_layer_gradient_30pct", "empirical", _check_cross_layer_gradient),
    Constraint("resnext_width_pow2", "empirical", _check_resnext_width_pow2),
    Constraint("no_entropy_calibrator", "empirical", _check_no_entropy_calibrator),
]


# ============================================================
# 软约束 (SOFT) — 不致命,作为搜索器 penalty
# ============================================================

def _check_module_precision_consistency(cfg: Config, hw: HardwareCapability) -> tuple[bool, str]:
    """模块间精度一致性偏好: 极端混合 (FP32+INT8) 不致命但通常更差."""
    bits = set(cfg.q_bits.values())
    if "FP32" in bits and "INT8" in bits:
        return False, "FP32+INT8 极端混合,建议至少加 FP16 过渡"
    return True, ""


SOFT_CONSTRAINTS: list[Constraint] = [
    Constraint("module_precision_consistency", "soft", _check_module_precision_consistency),
]


# ============================================================
# 公共 API
# ============================================================

ALL_CONSTRAINTS = PHYSICAL_CONSTRAINTS + EMPIRICAL_CONSTRAINTS + SOFT_CONSTRAINTS


def is_legal(
    config: Config,
    hw: HardwareCapability,
    kinds: tuple[ConstraintKind, ...] = ("physical", "empirical"),
) -> tuple[bool, str]:
    """检查 Config 是否满足指定类型的所有约束 (静态全局清单版本).

    ⚠️ 这是 v1.0 静态版本; v1.1 起推荐改用 ``is_legal_for_hardware``,
    它会根据 capability YAML 字段动态调整经验约束级别 (N2v2 实证机制).

    默认只检查 physical + empirical (作为硬过滤);soft 约束留给搜索器做 penalty.

    返回:
        (True, "")            — 合法
        (False, "reason..")    — 第一条违反的约束
    """
    for c in ALL_CONSTRAINTS:
        if c.kind not in kinds:
            continue
        ok, reason = c.check(config, hw)
        if not ok:
            return False, f"[{c.kind}/{c.name}] {reason}"
    return True, ""


def list_violations(
    config: Config,
    hw: HardwareCapability,
    kinds: tuple[ConstraintKind, ...] = ("physical", "empirical", "soft"),
) -> list[tuple[ConstraintKind, str, str]]:
    """列出所有违反的约束 (用于诊断). 返回 [(kind, name, reason), ...]."""
    out: list[tuple[ConstraintKind, str, str]] = []
    for c in ALL_CONSTRAINTS:
        if c.kind not in kinds:
            continue
        ok, reason = c.check(config, hw)
        if not ok:
            out.append((c.kind, c.name, reason))
    return out


# ============================================================
# v1.1 — 硬件实证驱动的动态约束清单 (N2v2 实证机制)
# ============================================================
#
# 核心创新: 经验约束的"硬度"由硬件 capability YAML 的实证字段动态决定.
# 这把 v1.5 §3 第 9 条 "硬约束如何从人工规则演进为学到的规则" 实证化.
#
# 例: channels % 32 == 0 这条规则,
#   - 在旧硬件 / 旧 TRT 上确实是经验硬约束 (违反性能差 > 5%)
#   - 但 N2v2 (2026-05-06) 在 Orin AGX sm87 + TRT 8.5 上实测,
#     63 vs 64 仅差 0.27%, 90 vs 96 反向差 -0.93% — 几乎失效
#   - 因此 orin_agx.yaml 写入 alignment_enforcement="soft",
#     本函数据此把对齐约束从 "empirical" 降级为 "soft"
#   - 结果: 搜索空间扩大 ~30% (恢复 90/126/200 这类近似对齐通道数的剪枝率)
# ============================================================

def build_constraints_for_hardware(hw: HardwareCapability) -> list[Constraint]:
    """根据硬件 capability YAML 实证字段动态构造约束清单.

    物理约束总是包含 (硬件事实不可改写).
    经验约束的级别由 capability YAML 字段动态决定:
        - alignment_enforcement="hard"  → 保持 empirical (默认)
        - alignment_enforcement="soft"  → 降级为 soft (N2v2 实证后用)
        - alignment_enforcement="auto"  → 完全移除 (让预测器学)

    返回的约束清单可直接用于 is_legal_for_hardware / list_violations_for_hardware.
    """
    constraints: list[Constraint] = list(PHYSICAL_CONSTRAINTS)

    # 通道对齐约束: 由 hw.alignment.alignment_enforcement 字段决定级别
    align_mode = getattr(hw.alignment, "alignment_enforcement", "hard")
    if align_mode == "hard":
        constraints.append(
            Constraint("channel_align_32", "empirical", _check_channel_alignment)
        )
    elif align_mode == "soft":
        # 同一 check 函数,但级别降为 soft (作 penalty 不过滤)
        constraints.append(
            Constraint("channel_align_32_soft", "soft", _check_channel_alignment)
        )
    # "auto" 模式: 不加,让 LightGBM 等预测器从数据中学

    # K_dim ≥ 64 (2:4 sparse 门槛) — 与 sparse_tc 能力绑定
    if hw.features.sparse_tc:
        constraints.append(
            Constraint("k_dim_ge_64_for_2to4", "empirical", _check_k_dim_for_2to4)
        )

    # 跨层剪枝率梯度 — 当前默认仍 empirical,留待后续实证
    constraints.append(
        Constraint("cross_layer_gradient_30pct", "empirical", _check_cross_layer_gradient)
    )

    # ResNeXt grouped-conv width_per_group power-of-2 — TC 硬件硬约束
    # M4.9 v2 实证 (Pyramid prune25): 违反时 lat 3.35× 慢于不违反
    has_tc = bool(getattr(hw.features, "tensor_core", False) or
                  any(ip.tensor_core_gen for ip in hw.ips.values() if ip.tensor_core_gen))
    if has_tc:
        constraints.append(
            Constraint("resnext_width_pow2", "empirical", _check_resnext_width_pow2)
        )

    # 软约束 (跨硬件通用)
    constraints.extend(SOFT_CONSTRAINTS)

    return constraints


def is_legal_for_hardware(
    config: Config,
    hw: HardwareCapability,
    kinds: tuple[ConstraintKind, ...] = ("physical", "empirical"),
) -> tuple[bool, str]:
    """v1.1 推荐 API: 用硬件实证驱动的动态约束清单做合法性检查."""
    for c in build_constraints_for_hardware(hw):
        if c.kind not in kinds:
            continue
        ok, reason = c.check(config, hw)
        if not ok:
            return False, f"[{c.kind}/{c.name}] {reason}"
    return True, ""


def list_violations_for_hardware(
    config: Config,
    hw: HardwareCapability,
    kinds: tuple[ConstraintKind, ...] = ("physical", "empirical", "soft"),
) -> list[tuple[ConstraintKind, str, str]]:
    """v1.1 推荐 API: 列出所有违反的约束 (用动态约束清单)."""
    out: list[tuple[ConstraintKind, str, str]] = []
    for c in build_constraints_for_hardware(hw):
        if c.kind not in kinds:
            continue
        ok, reason = c.check(config, hw)
        if not ok:
            out.append((c.kind, c.name, reason))
    return out


def soft_penalty_for_hardware(
    config: Config,
    hw: HardwareCapability,
) -> tuple[float, list[str]]:
    """计算 soft 约束的惩罚总和 (供搜索器 fitness 用).

    每违反一条 soft 约束 +1.0 (后续可用更细的权重).
    返回 (penalty, [violated_names]).
    """
    penalty = 0.0
    names: list[str] = []
    for c in build_constraints_for_hardware(hw):
        if c.kind != "soft":
            continue
        ok, _ = c.check(config, hw)
        if not ok:
            penalty += 1.0
            names.append(c.name)
    return penalty, names


__all__ = [
    "Constraint",
    "ConstraintKind",
    "PHYSICAL_CONSTRAINTS",
    "EMPIRICAL_CONSTRAINTS",
    "SOFT_CONSTRAINTS",
    "ALL_CONSTRAINTS",
    "is_legal",                          # v1.0 静态版 (向后兼容)
    "list_violations",                    # v1.0 静态版
    "build_constraints_for_hardware",     # v1.1 动态版 (推荐)
    "is_legal_for_hardware",              # v1.1 动态版 (推荐)
    "list_violations_for_hardware",       # v1.1 动态版
    "soft_penalty_for_hardware",          # v1.1 动态版
]


# ============================================================
# Demo / 自测 (展示 N2v2 实证机制如何让同一 Config 在不同硬件上得到不同判定)
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    ORIN_YAML = REPO_ROOT / "configs/hardware/orin_agx.yaml"
    RTX4090_YAML = REPO_ROOT / "configs/hardware/rtx4090.yaml"

    print("=" * 70)
    print("A3 Demo: 硬件实证驱动的约束级别动态调整 (N2v2 → constraints DSL)")
    print("=" * 70)

    # 准备一个"剪枝率不对齐到 32"的测试 Config
    # 256 通道 backbone, 剪枝率 0.297 → 保留 ~180 通道 (180 % 32 != 0)
    test_cfg = Config.fp32_baseline()
    test_cfg = test_cfg.with_field(
        prune_rate={"backbone": 0.30, "encoder": 0.30, "decoder": 0.0,
                    "heads": 0.0, "v2x_comm": 0.0},
        prune_object="channel",
    )
    print(f"\n测试 Config: backbone/encoder 剪枝 30% (保留 ~180/256 通道, 不对齐 32)")

    for name, yaml_path in [("Orin AGX (alignment=soft, N2v2 实证)", ORIN_YAML),
                             ("RTX 4090 (alignment=hard, 默认)", RTX4090_YAML)]:
        print(f"\n--- {name} ---")
        if not yaml_path.exists():
            print(f"  [skip] {yaml_path} 不存在")
            continue
        try:
            hw = HardwareCapability.from_yaml(yaml_path)
        except Exception as e:
            print(f"  [load failed] {e}")
            continue

        print(f"  hw.alignment.alignment_enforcement = "
              f"{getattr(hw.alignment, 'alignment_enforcement', 'hard')}")

        constraints = build_constraints_for_hardware(hw)
        kinds_count = {"physical": 0, "empirical": 0, "soft": 0}
        for c in constraints:
            kinds_count[c.kind] += 1
        print(f"  动态展开约束: {kinds_count}")

        ok, reason = is_legal_for_hardware(test_cfg, hw)
        print(f"  is_legal_for_hardware: {ok}  {reason if not ok else ''}")

        violations = list_violations_for_hardware(test_cfg, hw)
        if violations:
            print(f"  违反清单 ({len(violations)} 条):")
            for kind, cname, msg in violations:
                print(f"    [{kind:9s}] {cname}: {msg}")
        else:
            print(f"  无违反 (此 Config 在该硬件上完全合法)")

        penalty, soft_names = soft_penalty_for_hardware(test_cfg, hw)
        print(f"  软约束 penalty: {penalty}  ({soft_names if soft_names else '无'})")

    print("\n" + "=" * 70)
    print("关键观察:")
    print("  - Orin AGX 上 channel_align 被 N2v2 实证降级为 soft → 不过滤，")
    print("    180 通道剪枝率合法 (旧 HW-NAS 会硬过滤掉)")
    print("  - RTX 4090 上保持 alignment_enforcement=hard (待 4090 端实证后调整)")
    print("  - 这就是 v1.5 §3 第 9 条 '硬约束 → 学到的规则' 的代码落地")
    print("=" * 70)
