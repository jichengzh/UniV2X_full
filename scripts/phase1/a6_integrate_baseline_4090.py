"""A6: 整合 1.1/1.2/1.3 baseline 数据 → data/baseline_4090.parquet

对应 v1.5 §6.1 Phase 1A.6, 与 framework/config_schema.py Config 字段对齐.

数据源:
1. data/phase1/baseline_unified.parquet — 22 行 1.1/1.2 各类基线
2. data/phase4/stage5_baseline_v3.csv — 23 行 UniV2X plan_b_active 联合搜索 4090 实测
3. calibration/latency_lut.json — 1.3 D 空间真实测量点 (D1-D4)

输出:
    data/baseline_4090.parquet — 真实实测 + 标注 source/is_real_measured/d_runtime
    data/baseline_4090.csv — 同内容 CSV (人读)

Schema (核心列, 见 BASELINE_4090_SCHEMA):
    主键: config_id (str)
    元数据: source, is_real_measured, notes
    度量: amota, amotp, mAP, NDS, recall, mota, lat_e2e_ms, params_after_M, mem_peak_mb
    Config 5 模块 × 6 字段: prune_rate/prune_criterion/q_bits/q_granularity/q_object/d_routing
    全局 Config: prune_object
    D v1.1 列: d_runtime, d_pipelined, d_temporal_cache_int8
    细粒度可选: prune_rate__encoder_ffn/encoder_attn/encoder_heads/decoder_ffn/decoder_attn/decoder_heads
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = ("backbone", "encoder", "decoder", "heads", "v2x_comm")

# 默认填充 (FP32 + GPU + 不剪枝)
DEFAULT_FP32 = "FP32"
DEFAULT_NONE_GRAN = "none"
DEFAULT_NONE_OBJ = "none"
DEFAULT_GPU = "GPU"


def empty_config_row() -> dict:
    """一行符合 Config schema 的全 FP32/不剪/GPU 默认值."""
    row: dict = {"prune_object": "none"}
    for m in MODULES:
        row[f"prune_rate__{m}"] = 0.0
        row[f"prune_criterion__{m}"] = "none"
        row[f"q_bits__{m}"] = DEFAULT_FP32
        row[f"q_granularity__{m}"] = DEFAULT_NONE_GRAN
        row[f"q_object__{m}"] = DEFAULT_NONE_OBJ
        row[f"d_routing__{m}"] = DEFAULT_GPU
    # D v1.1
    row["d_runtime"] = "pytorch_fp32"
    row["d_pipelined"] = 0
    row["d_temporal_cache_int8"] = 0
    return row


def empty_metric_row() -> dict:
    return {
        "amota": None, "amotp": None, "mAP": None, "NDS": None,
        "recall": None, "mota": None, "lat_e2e_ms": None,
        "params_after_M": None, "mem_peak_mb": None,
    }


def base_row(config_id: str, source: str, is_real: bool, notes: str = "") -> dict:
    row: dict = {
        "config_id": config_id,
        "source": source,
        "is_real_measured": is_real,
        "notes": notes,
    }
    row.update(empty_config_row())
    row.update(empty_metric_row())
    return row


# ---------- Source 1: baseline_unified.parquet ----------

def load_baseline_unified() -> list[dict]:
    """转换 22 行 1.1/1.2 baseline → 标准 schema."""
    df = pd.read_parquet(REPO_ROOT / "data/phase1/baseline_unified.parquet")
    out = []
    for _, src in df.iterrows():
        row = base_row(
            config_id=src["config_id"],
            source=src["source"],
            is_real=True,
            notes=str(src.get("notes", "") or ""),
        )
        # B1 prune
        po = src.get("prune_object", "none")
        row["prune_object"] = po if isinstance(po, str) and po else "none"
        for m in MODULES:
            v = src.get(f"prune_rate__{m}", 0.0)
            row[f"prune_rate__{m}"] = float(v) if pd.notna(v) else 0.0
        crit = src.get("prune_criterion__encoder", "none")
        if isinstance(crit, str) and crit and crit != "none":
            for m in MODULES:
                if row[f"prune_rate__{m}"] > 0:
                    row[f"prune_criterion__{m}"] = crit
        # B2 quant
        for m in MODULES:
            qb = src.get(f"q_bits__{m}", DEFAULT_FP32)
            row[f"q_bits__{m}"] = qb if isinstance(qb, str) and qb else DEFAULT_FP32
        gran = src.get("q_granularity__encoder", DEFAULT_NONE_GRAN)
        qobj = src.get("q_object__encoder", DEFAULT_NONE_OBJ)
        for m in MODULES:
            if row[f"q_bits__{m}"] in ("INT8", "FP16"):
                row[f"q_granularity__{m}"] = gran if isinstance(gran, str) else "per-tensor"
                row[f"q_object__{m}"] = qobj if isinstance(qobj, str) else "W+A"
        # D
        d_bb = src.get("d_routing__backbone", DEFAULT_GPU)
        for m in MODULES:
            row[f"d_routing__{m}"] = d_bb if isinstance(d_bb, str) else DEFAULT_GPU
        # 当 q_bits 是 INT8/FP16 时, d_runtime 应该是 trt_*
        if any(row[f"q_bits__{m}"] == "INT8" for m in MODULES):
            row["d_runtime"] = "trt_int8"
        elif any(row[f"q_bits__{m}"] == "FP16" for m in MODULES):
            row["d_runtime"] = "trt_fp16"
        # 度量
        if pd.notna(src.get("accuracy_amota")):
            row["amota"] = float(src["accuracy_amota"])
        if pd.notna(src.get("latency_ms")):
            row["lat_e2e_ms"] = float(src["latency_ms"])
        if pd.notna(src.get("mAP")):
            row["mAP"] = float(src["mAP"])
        out.append(row)
    return out


# ---------- Source 2: stage5_baseline_v3.csv (23 行 plan_b_active) ----------

def load_stage5_baseline_v3() -> list[dict]:
    df = pd.read_csv(REPO_ROOT / "data/phase4/stage5_baseline_v3.csv")
    out = []
    for _, src in df.iterrows():
        row = base_row(
            config_id=f"v3_{src['config_id']}",
            source=src["source"],
            is_real=True,
            notes="UniV2X plan_b_active 4090 联合搜索实测",
        )
        # 细粒度 → module 粒度: encoder = max(ffn, attn, heads), decoder 同理
        enc_rates = [
            float(src.get(f"prune_rate__encoder_{sub}", 0.0) or 0.0)
            for sub in ("ffn", "attn", "heads")
        ]
        dec_rates = [
            float(src.get(f"prune_rate__decoder_{sub}", 0.0) or 0.0)
            for sub in ("ffn", "attn", "heads")
        ]
        row["prune_rate__backbone"] = float(src.get("prune_rate__backbone", 0.0) or 0.0)
        row["prune_rate__encoder"] = max(enc_rates) if enc_rates else 0.0
        row["prune_rate__decoder"] = max(dec_rates) if dec_rates else 0.0
        row["prune_rate__heads"] = float(src.get("prune_rate__heads_mid", 0.0) or 0.0)
        row["prune_rate__v2x_comm"] = 0.0
        if any(r > 0 for r in [
            row["prune_rate__backbone"], row["prune_rate__encoder"],
            row["prune_rate__decoder"], row["prune_rate__heads"]
        ]):
            row["prune_object"] = "channel"
            for m in MODULES:
                if row[f"prune_rate__{m}"] > 0:
                    row[f"prune_criterion__{m}"] = "L1"

        # B2: bits int (32/16/8) → string
        bit2str = {32: "FP32", 16: "FP16", 8: "INT8"}
        for m in MODULES:
            qb_int = int(src.get(f"q_bits__{m}", 32) or 32)
            row[f"q_bits__{m}"] = bit2str.get(qb_int, "FP32")
        gran_w = str(src.get("q_granularity_w", "per_tensor") or "per_tensor")
        gran_str = "per-tensor" if gran_w == "per_tensor" else "per-channel"
        qtarget = str(src.get("q_target", "none") or "none")
        qobj_str = "W+A" if "A" in qtarget else ("W-only" if qtarget == "W" else "none")
        for m in MODULES:
            if row[f"q_bits__{m}"] in ("INT8", "FP16"):
                row[f"q_granularity__{m}"] = gran_str
                row[f"q_object__{m}"] = qobj_str if qobj_str != "none" else "W+A"
        # 23 configs 都是 GPU only (plan_b_active 没接入 D 维度)
        # d_runtime 由 q_bits 推
        if any(row[f"q_bits__{m}"] == "INT8" for m in MODULES):
            row["d_runtime"] = "trt_int8"
        elif any(row[f"q_bits__{m}"] == "FP16" for m in MODULES):
            row["d_runtime"] = "trt_fp16"

        # 细粒度列 (保留)
        for sub in ("ffn", "attn", "heads"):
            row[f"prune_rate__encoder_{sub}"] = float(src.get(f"prune_rate__encoder_{sub}", 0.0) or 0.0)
            row[f"prune_rate__decoder_{sub}"] = float(src.get(f"prune_rate__decoder_{sub}", 0.0) or 0.0)
        row["decoder_num_layers"] = int(src.get("decoder_num_layers", 6) or 6)

        # 度量
        for k_src, k_dst in [
            ("amota", "amota"), ("amotp", "amotp"), ("recall", "recall"),
            ("mota", "mota"), ("mAP", "mAP"), ("NDS", "NDS"),
            ("lat_e2e_ms", "lat_e2e_ms"), ("params_after_M", "params_after_M"),
        ]:
            v = src.get(k_src)
            if pd.notna(v):
                row[k_dst] = float(v)
        out.append(row)
    return out


# ---------- Source 3: 1.3 D空间 latency_lut.json ----------

def load_d_space_lut() -> list[dict]:
    """从 latency_lut.json 抽 1.3 真实测量点 (D2/D3/D4 关键 row)."""
    with open(REPO_ROOT / "calibration/latency_lut.json", "r", encoding="utf-8") as f:
        lut = json.load(f)
    out = []

    # D2 流水重叠 (baseline + pruned)
    d2 = lut.get("D2_pipeline_overlap", {})
    for variant, d in d2.items():
        if variant.startswith("_"):
            continue
        if "no_overlap" in d:
            row = base_row(
                config_id=f"d2_{variant}_no_overlap",
                source="1.3_d", is_real=True,
                notes="D2 流水未重叠 (baseline)",
            )
            row["d_pipelined"] = 0
            v = d["no_overlap"]
            row["lat_e2e_ms"] = float(v["actual_latency_ms"])
            row["mem_peak_mb"] = float(v.get("peak_memory_mb", 0))
            if "pruned" in variant:
                _apply_prune_d14_enc10_dec07(row)
            out.append(row)
        if "backbone_bev_overlap" in d:
            row = base_row(
                config_id=f"d2_{variant}_pipelined",
                source="1.3_d", is_real=True,
                notes="D2 backbone-BEV 重叠 (理论稳态)",
            )
            row["d_pipelined"] = 1
            v = d["backbone_bev_overlap"]
            row["lat_e2e_ms"] = float(v["theoretical_steady_state_ms"])
            row["mem_peak_mb"] = float(v.get("peak_memory_mb", 0))
            if "pruned" in variant:
                _apply_prune_d14_enc10_dec07(row)
            out.append(row)

    # D3 时序缓存 (baseline + pruned60)
    d3 = lut.get("D3_temporal_cache", {})
    for variant, d in d3.items():
        if variant.startswith("_"):
            continue
        for cfg_name, v in d.items():
            if not isinstance(v, dict) or "amota" not in v:
                continue
            row = base_row(
                config_id=f"d3_{variant}_{cfg_name}",
                source="1.3_d", is_real=True,
                notes=f"D3 时序缓存 {cfg_name} ({v.get('_note', '')})",
            )
            if "int8" in cfg_name:
                row["d_temporal_cache_int8"] = 1
                for m in MODULES:
                    row[f"q_bits__{m}"] = "INT8"
                    row[f"q_granularity__{m}"] = "per-tensor"
                    row[f"q_object__{m}"] = "W+A"
                row["d_runtime"] = "trt_int8"
            elif "fp16" in cfg_name:
                for m in MODULES:
                    row[f"q_bits__{m}"] = "FP16"
                    row[f"q_granularity__{m}"] = "per-tensor"
                    row[f"q_object__{m}"] = "W+A"
                row["d_runtime"] = "trt_fp16"
            if "pruned60" in variant:
                # decoder FFN 60% 剪枝 (1.2_prune p1_ffn_60pct 等价)
                row["prune_object"] = "channel"
                row["prune_rate__decoder"] = 0.6
                row["prune_rate__encoder"] = 0.6
                row["prune_criterion__decoder"] = "L1"
                row["prune_criterion__encoder"] = "L1"
            row["amota"] = float(v["amota"])
            if pd.notna(v.get("mAP")):
                row["mAP"] = float(v["mAP"])
            row["lat_e2e_ms"] = float(v["latency_ms"])
            row["mem_peak_mb"] = float(v.get("peak_memory_mb", 0))
            out.append(row)

    # D4 内存策略 (baseline only, 默认 vs defrag)
    d4 = lut.get("D4_memory_strategy", {}).get("baseline", {})
    for strategy in ("dynamic", "defrag"):
        v = d4.get(strategy)
        if not isinstance(v, dict):
            continue
        row = base_row(
            config_id=f"d4_baseline_{strategy}",
            source="1.3_d", is_real=True,
            notes=f"D4 内存策略 {strategy}",
        )
        row["lat_e2e_ms"] = float(v["latency_mean_ms"])
        row["mem_peak_mb"] = float(v.get("peak_allocated_mb", 0))
        out.append(row)

    return out


def _apply_prune_d14_enc10_dec07(row: dict) -> None:
    """应用 D2 'pruned_d14_enc10_dec07' 配置 (decoder 14 层→6 层等价 + enc/dec ffn)."""
    row["prune_object"] = "channel"
    row["prune_rate__encoder"] = 0.0
    row["prune_rate__decoder"] = 0.3
    row["prune_criterion__encoder"] = "L1"
    row["prune_criterion__decoder"] = "L1"


# ---------- 主流程 ----------

def main() -> None:
    rows: list[dict] = []
    rows.extend(load_baseline_unified())
    rows.extend(load_stage5_baseline_v3())
    rows.extend(load_d_space_lut())

    df = pd.DataFrame(rows)

    # 列顺序: 元数据 → 度量 → Config → D v1.1 → 细粒度
    fixed = ["config_id", "source", "is_real_measured", "notes"]
    metrics = ["amota", "amotp", "mAP", "NDS", "recall", "mota",
               "lat_e2e_ms", "params_after_M", "mem_peak_mb"]
    config_cols = ["prune_object"]
    for m in MODULES:
        for prefix in ("prune_rate", "prune_criterion", "q_bits",
                       "q_granularity", "q_object", "d_routing"):
            config_cols.append(f"{prefix}__{m}")
    d_v11 = ["d_runtime", "d_pipelined", "d_temporal_cache_int8"]
    fine = [c for c in df.columns if c.startswith("prune_rate__encoder_")
            or c.startswith("prune_rate__decoder_")
            or c == "decoder_num_layers"]

    ordered = fixed + metrics + config_cols + d_v11 + fine
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]

    out_dir = REPO_ROOT / "data"
    out_parquet = out_dir / "baseline_4090.parquet"
    out_csv = out_dir / "baseline_4090.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"✅ Wrote {out_parquet} ({len(df)} rows × {len(df.columns)} cols)")
    print(f"✅ Wrote {out_csv}")
    print()
    print("Source 分布:")
    print(df["source"].value_counts().to_string())
    print()
    print("is_real_measured:")
    print(df["is_real_measured"].value_counts().to_string())
    print()
    print(f"有 amota 度量的行数: {df['amota'].notna().sum()} / {len(df)}")
    print(f"有 lat_e2e_ms 度量的行数: {df['lat_e2e_ms'].notna().sum()} / {len(df)}")
    print(f"有 mAP 度量的行数: {df['mAP'].notna().sum()} / {len(df)}")


if __name__ == "__main__":
    main()
