"""Merge A.2 row JSONs into e2e_bench_v1.csv + parquet.

读 results/a2_d_expand/*.row.json → 列对齐 e2e_bench_v1 schema → 追加.
v1.4: dedup 按 (triplet, q_tag, d_tag) — d_tag 含 BL 后缀, 让噪声地板控制重测不被去掉.
v1.4: 老 csv 无 d_builder_opt_level 列 → 自动回填 BL=3; 无 d_tag → 根据 (tactic, ws) 推断.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REPO = Path("/home/jichengzhi/UniV2X")
ROW_DIR = REPO / "results/a2_d_expand"
DATA = REPO / "paper_learning/2. AAAI最终故事/data"
CSV = DATA / "e2e_bench_v1.csv"
PQ = DATA / "e2e_bench_v1.parquet"

# e2e_bench_v1.csv schema (v1.4: 33 列, +d_builder_opt_level +d_tag)
SCHEMA_COLS = [
    "triplet", "stage0_planes", "stage1_planes", "stage2_planes",
    "prune_object", "sparse_mask",
    "q_tag", "prec_flag", "q_bits", "q_bits_per_stage",
    "q_granularity", "q_object",
    "d_scheme", "d_tactic", "d_workspace_gb", "d_builder_opt_level",  # v1.4 +BL
    "d_tag",                                                            # v1.4 显式 D-tag
    "hardware", "device", "max_voxels",
    "n_collected", "n_skipped",
    "real_voxels_mean", "real_voxels_p99",
    "throughput_fps",
    "ap30", "ap50", "ap70",
    "engine_size_mb", "build_secs",
    "build_success", "fail_reason", "ts",
]


_LEGACY_D_TAG = {
    ("default", 4): "D1_default_4gb",
    ("with_cudnn", 8): "D2_with_cudnn_8gb",
    ("cublas_lt", 16): "D3_cublas_lt_16gb",
    ("all_enabled", 1): "D4_all_enabled_1gb",
    ("default", 1): "D5_default_1gb",
    ("default", 8): "D6_default_8gb",
    ("default", 16): "D7_default_16gb",
    ("with_cudnn", 4): "D8_with_cudnn_4gb",
    ("cublas_lt", 4): "D9_cublas_lt_4gb",
    ("cublas_lt", 8): "D10_cublas_lt_8gb",
    ("all_enabled", 4): "D11_all_enabled_4gb",
    ("edge_only", 4): "D12_edge_only_4gb",
}


def infer_d_tag_legacy(row):
    """老 csv 无 d_tag 时, 根据 (tactic, ws) 反推原 D1-D12 tag."""
    tactic = row["d_tactic"]
    ws = int(row["d_workspace_gb"])
    return _LEGACY_D_TAG.get((tactic, ws), f"D_unknown_{tactic}_{ws}gb")


def main():
    rows = []
    for jf in sorted(ROW_DIR.glob("*.row.json")):
        d = json.loads(jf.read_text())
        row = {c: d.get(c) for c in SCHEMA_COLS}
        # v1.4: 新 row 应该都有 d_builder_opt_level + d_tag, 老 row.json 兜底
        if row.get("d_builder_opt_level") is None:
            row["d_builder_opt_level"] = 3
        if row.get("d_tag") is None:
            row["d_tag"] = infer_d_tag_legacy(row)
        rows.append(row)
    print(f"loaded {len(rows)} a2 rows")

    a2_df = pd.DataFrame(rows, columns=SCHEMA_COLS)

    # 读现有 csv (v1.4 兼容: 老 csv 无 BL/d_tag 列 → 回填)
    if CSV.exists():
        cur = pd.read_csv(CSV)
        if "d_builder_opt_level" not in cur.columns:
            cur["d_builder_opt_level"] = 3
            print(f"[backfill] {len(cur)} 老 csv 行 d_builder_opt_level=3")
        if "d_tag" not in cur.columns:
            cur["d_tag"] = cur.apply(infer_d_tag_legacy, axis=1)
            print(f"[backfill] {len(cur)} 老 csv 行 d_tag 根据 (tactic, ws) 推断")
        cur = cur.reindex(columns=SCHEMA_COLS)
        print(f"existing csv: {len(cur)} rows")
    else:
        cur = pd.DataFrame(columns=SCHEMA_COLS)

    # v1.4 dedup 按 (triplet, q_tag, d_tag) — d_tag 已含 BL 后缀,
    # D25_default_4gb_BL3 vs D1_default_4gb 不会去重 → 保留噪声地板控制重测
    key_cols = ["triplet", "q_tag", "d_tag"]
    cur_keys = set(map(tuple, cur[key_cols].values)) if len(cur) else set()
    a2_keys = set(map(tuple, a2_df[key_cols].values))
    dups = cur_keys & a2_keys
    if dups:
        print(f"WARN: {len(dups)} duplicates would be replaced (a2 wins)")
    cur_dedup = cur[~cur.set_index(key_cols).index.isin(a2_keys)] if dups else cur

    merged = pd.concat([cur_dedup, a2_df], ignore_index=True)
    print(f"merged: {len(merged)} rows (was {len(cur)}, +{len(merged) - len(cur_dedup)})")

    merged.to_csv(CSV, index=False)
    merged.to_parquet(PQ, index=False)
    print(f"saved -> {CSV.name}, {PQ.name}")


if __name__ == "__main__":
    main()
