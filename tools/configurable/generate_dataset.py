"""generate_dataset.py — DoE anchor → 三工具管线 → 预测器训练数据 (交付物 3).

数据生成师交付物 3. 把 DoE 的每个 anchor Config 依次喂给:

    prune_config.resolve_prune_plan / execute_pyramid   (B 剪枝维度 → 真实小模型 + 通道数/params)
        ↓ (剪后小模型 → ONNX, 复用已有 models/*.onnx)
    quant_config.resolve_quant_plan                      (Q 量化维度 → effective_bits/calibrator/manifest)
        ↓ (FP32 ONNX + 量化 plan)
    deploy_config.build_plan / build_engine              (D 部署维度 → TRT engine + engine_size/build_secs/int8%)

收集每行标签, 对齐 e2e_bench_v1_schema.md 的 33 列, 标注:
  - is_real_measured: True 仅当该行的 latency/AP/size 是真测的
  - source: 每个标签的来源 (trtexec / deploy_config_build / 估算 / dry_run)

运行模式:
  --mode dry_run   : 只走 plan 决议, 不真 rebuild / 不真 build (无 GPU 也能跑, 接通管线)
  --mode prune_real: 真 rebuild 剪枝小模型 (CPU 可跑), TRT build 仍 dry_run
  --mode full      : prune rebuild + TRT build (需空闲 GPU, GPU 忙时单 anchor 会标 build failed)

设计原则 (与三工具一致): 不改 config_schema; 复用工具; 跑不通标 dry_run/未实测, 不假装.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from framework.config_schema import Config  # noqa: E402
from tools.configurable.doe_anchors import Anchor, build_doe, build_all_doe, M  # noqa: E402
from tools.configurable.prune_config import (  # noqa: E402
    resolve_prune_plan, resolve_pyramid_num_filters, INT8_ROUND_TO, DEFAULT_ROUND_TO,
)
from tools.configurable.quant_config import resolve_quant_plan  # noqa: E402
from tools.configurable.deploy_config import build_plan  # noqa: E402


# ---------------------------------------------------------------------------
# 资源路径 (复用已有 artifacts, 不重造)
# ---------------------------------------------------------------------------
ORIG_CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
HW_YAML = {
    "rtx4090": str(_REPO / "configs/hardware/rtx4090.yaml"),
    "orin_agx": str(_REPO / "configs/hardware/orin_agx.yaml"),
}
# 已有的剪枝档 subnet ONNX (FP32, 输入 1,64,128,256). 用于喂 quant/deploy.
# prune_rate → 已存在的 ONNX (复用; 没有的档走 dry_run).
PRUNE_ONNX = {
    0.0: str(_REPO / "models/pyramid_dair_m1_subnet_fp32.onnx"),
    0.50: str(_REPO / "models/pyramid_dair_m1_pruned50_ft_subnet_fp32.onnx"),
}
CALIB_NPY = str(_REPO / "calibration/pyramid_dair_calib.npy")
BASE_NUM_FILTERS = [64, 128, 256]

# e2e_bench_v1.csv 的 33 列 (顺序对齐 schema)
SCHEMA_COLUMNS = [
    "triplet", "stage0_planes", "stage1_planes", "stage2_planes",
    "prune_object", "sparse_mask", "q_tag", "prec_flag", "q_bits",
    "q_bits_per_stage", "q_granularity", "q_object", "d_scheme", "d_tactic",
    "d_workspace_gb", "d_builder_opt_level", "d_tag", "hardware", "device",
    "max_voxels", "n_collected", "n_skipped", "real_voxels_mean",
    "real_voxels_p99", "throughput_fps", "ap30", "ap50", "ap70",
    "engine_size_mb", "build_secs", "build_success", "fail_reason", "ts",
]
# 生成数据集额外追踪列 (schema 之外, 标来源)
EXTRA_COLUMNS = [
    "anchor_id", "strategy", "is_real_measured", "source", "prune_rate",
    "calibrator", "effective_bits", "int8_layer_pct", "params_total_new",
    "trt_precision", "exec_path", "pipeline_note",
]


def _q_tag(bits: str, calib: str, gran: str, obj: str) -> str:
    """精度配置短标签 (对齐 schema §二 q_tag)."""
    if bits == "FP32":
        return "Q_fp32"
    if bits == "FP16":
        return "Q_fp16"
    # INT8
    suffix = {"minmax": "mm", "percentile_99_99": "pct", "entropy": "ent"}.get(calib, "mm")
    base = f"Q_int8_{suffix}"
    if obj == "W-only":
        base += "_wo"
    if gran == "per-tensor":
        base += "_pt"
    return base


def _planes_for_rate(prune_rate: float, round_to: int) -> tuple[int, int, int]:
    """剪率 → (s0,s1,s2) planes (复用 prune_config 的对齐逻辑)."""
    if prune_rate <= 0:
        return tuple(BASE_NUM_FILTERS)
    try:
        nf, _wpg, _notes = resolve_pyramid_num_filters(
            BASE_NUM_FILTERS, prune_rate, round_to=round_to)
        return tuple(nf)
    except Exception:
        return tuple(BASE_NUM_FILTERS)


@dataclass
class RowResult:
    """单 anchor 经管线后的一行 (含 schema 列 + extra 列 + 各工具 manifest 引用)."""
    row: dict = field(default_factory=dict)
    prune_manifest: dict = field(default_factory=dict)
    quant_manifest: dict = field(default_factory=dict)
    deploy_report: dict = field(default_factory=dict)


def run_anchor(anchor: Anchor, mode: str, out_dir: Path,
               verbose: bool = False) -> RowResult:
    """把一个 anchor 串过三个工具, 返回 RowResult."""
    cfg = anchor.config
    hw = anchor.hardware
    pr = cfg.prune_rate[M]
    bits = cfg.q_bits[M]
    calib = cfg.q_calibrator.get(M, "minmax")
    gran = cfg.q_granularity.get(M, "none")
    obj = cfg.q_object.get(M, "none")
    routing = cfg.d_routing.get(M, "GPU")
    pipeline_notes: list[str] = []
    sources: list[str] = []
    is_real = False

    # ---------- 阶段 1: prune_config (B 维度) ----------
    round_to = INT8_ROUND_TO if bits == "INT8" else DEFAULT_ROUND_TO
    prune_plan = resolve_prune_plan(cfg, arch="pyramid")
    rm = next(m for m in prune_plan.modules if m.module == M)
    s0, s1, s2 = _planes_for_rate(pr, round_to)
    prune_manifest = {
        "config_id": cfg.config_id,
        "prune_rate": pr, "exec_path": rm.exec_path, "round_to": rm.round_to,
        "num_filters_new": [s0, s1, s2], "criterion": rm.criterion,
        "notes": list(rm.notes),
    }
    params_total_new = None
    if mode in ("prune_real", "full") and pr > 0:
        try:
            from tools.configurable.prune_config import execute_pyramid
            pm = execute_pyramid(prune_plan, orig_dir=ORIG_CKPT_DIR,
                                 out_dir=str(out_dir / f"pruned_{anchor.anchor_id}"),
                                 dry_run=False)
            prune_manifest.update(pm)
            params_total_new = pm.get("params_total_new")
            sources.append("prune:real_rebuild")
            pipeline_notes.append(
                f"prune 真 rebuild: params {pm.get('params_total_old')}→"
                f"{pm.get('params_total_new')} (-{pm.get('params_total_reduction_pct')}%)")
        except Exception as e:  # noqa: BLE001
            pipeline_notes.append(f"prune rebuild 失败: {e}")
            sources.append("prune:plan_only")
    else:
        prune_manifest["status"] = "dry_run (plan only)" if pr > 0 else "no_prune (baseline)"
        sources.append("prune:plan_only" if pr > 0 else "prune:none")

    # ---------- 阶段 2: quant_config (Q 维度) ----------
    onnx_path = PRUNE_ONNX.get(pr, PRUNE_ONNX[0.0])
    onnx_exists = Path(onnx_path).exists()
    if pr not in PRUNE_ONNX:
        pipeline_notes.append(
            f"prune_rate={pr} 无现成 ONNX, quant/deploy 用 baseline ONNX 占位 (标 dry_run)")
    engine_path = str(out_dir / f"{anchor.anchor_id}.engine")
    quant_plan = resolve_quant_plan(cfg, onnx_path=onnx_path, engine_path=engine_path)
    quant_manifest = quant_plan.to_manifest()
    effective_bits = quant_manifest["modules"][0]["effective_bits"]
    trt_precision = quant_manifest["trt_precision"]
    sources.append("quant:plan")
    for w in quant_plan.warnings:
        pipeline_notes.append(f"quant WARN: {w}")

    # ---------- 阶段 3: deploy_config (D 维度) ----------
    deploy_report: dict = {}
    engine_size_mb = None
    build_secs = None
    int8_layer_pct = None
    build_success = False
    fail_reason = ""
    device = ""

    # build_plan 永远能跑 (纯推导, 无 GPU)
    prec_for_deploy = {"int8": "int8", "mixed": "int8", "fp16": "fp16",
                       "fp32": "fp32"}.get(trt_precision, "fp16")
    try:
        dplan = build_plan(onnx_path=onnx_path, engine_path=engine_path,
                           hw_yaml=HW_YAML[hw], config=cfg, precision=prec_for_deploy,
                           calib_data_path=CALIB_NPY if effective_bits == "INT8" else None)
        deploy_report["plan"] = {
            "want_int8": dplan.want_int8, "int8_mode": dplan.int8_mode,
            "use_dla": dplan.use_dla, "workspace_gb": dplan.workspace_gb,
            "alignment": dplan.alignment_enforcement, "routing": dplan.routing,
        }
    except Exception as e:  # noqa: BLE001
        pipeline_notes.append(f"build_plan 失败: {e}")

    can_real_build = (mode == "full" and onnx_exists and pr in PRUNE_ONNX
                      and hw == "rtx4090")  # 真 build 只在 4090 (本机有 TRT10)
    if can_real_build:
        try:
            from tools.configurable.deploy_config import build_engine
            t0 = time.time()
            rep = build_engine(dplan, verbose=verbose)
            deploy_report["build"] = rep
            if rep.get("status") == "ok":
                engine_size_mb = rep.get("engine_size_mb")
                build_secs = rep.get("build_sec")
                int8_layer_pct = rep.get("int8_layer_pct")
                build_success = True
                is_real = True
                device = rep.get("hardware", "")
                sources.append("deploy:real_trt_build")
                pipeline_notes.append(
                    f"TRT build OK: {engine_size_mb}MB int8%={int8_layer_pct} "
                    f"({round(time.time()-t0,1)}s)")
            else:
                fail_reason = f"{rep.get('stage')}: {rep.get('errors')}"
                sources.append("deploy:build_failed")
                pipeline_notes.append(f"TRT build 失败: {fail_reason[:120]}")
        except Exception as e:  # noqa: BLE001
            fail_reason = str(e)[:200]
            sources.append("deploy:build_exception")
            pipeline_notes.append(f"TRT build 异常: {fail_reason[:120]}")
    else:
        sources.append("deploy:dry_run")
        if hw == "orin_agx":
            pipeline_notes.append("Orin 无本地 TRT, deploy 标 dry_run (engine 在 Orin 上 build)")
        elif mode != "full":
            pipeline_notes.append(f"mode={mode}: deploy 不真 build")
        else:
            pipeline_notes.append("无现成剪枝 ONNX 或硬件非 4090, deploy dry_run")

    # ---------- 组装数据行 (对齐 schema 33 列) ----------
    row = {c: "" for c in SCHEMA_COLUMNS}
    row.update({
        "triplet": anchor.triplet,
        "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
        "prune_object": cfg.prune_object,
        "sparse_mask": "dense",
        "q_tag": _q_tag(bits, calib, gran, obj),
        "prec_flag": prec_for_deploy,
        "q_bits": bits,
        "q_bits_per_stage": "",
        "q_granularity": quant_manifest["modules"][0]["granularity_w"],
        "q_object": quant_manifest["modules"][0]["q_object"],
        "d_scheme": routing,
        "d_tactic": "default",
        "d_workspace_gb": int(dplan.workspace_gb) if "dplan" in dir() else 4,
        "d_builder_opt_level": "",
        "d_tag": f"D_{routing}_default_ws4",
        "hardware": hw,
        "device": device,
        "max_voxels": 32000,
        "n_collected": "", "n_skipped": "",
        "real_voxels_mean": "", "real_voxels_p99": "",
        "throughput_fps": "",   # latency 需真跑 e2e bench (本管线 build_engine 不含 e2e timing)
        "ap30": "", "ap50": "", "ap70": "",  # AP 需 DAIR eval, 本管线不含
        "engine_size_mb": engine_size_mb if engine_size_mb is not None else "",
        "build_secs": build_secs if build_secs is not None else "",
        "build_success": build_success,
        "fail_reason": fail_reason,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    extra = {
        "anchor_id": anchor.anchor_id,
        "strategy": anchor.strategy,
        "is_real_measured": is_real,
        "source": ";".join(sources),
        "prune_rate": pr,
        "calibrator": quant_manifest["modules"][0]["calibrator"],
        "effective_bits": effective_bits,
        "int8_layer_pct": int8_layer_pct if int8_layer_pct is not None else "",
        "params_total_new": params_total_new if params_total_new is not None else "",
        "trt_precision": trt_precision,
        "exec_path": rm.exec_path,
        "pipeline_note": " | ".join(pipeline_notes),
    }
    row.update(extra)
    return RowResult(row=row, prune_manifest=prune_manifest,
                     quant_manifest=quant_manifest, deploy_report=deploy_report)


def generate(mode: str, hardware: Optional[str], out_dir: Path,
             verbose: bool = False) -> list[RowResult]:
    """跑整套 DoE → 返回 RowResult 列表."""
    if hardware:
        anchors = build_doe(hardware)
    else:
        anchors = build_all_doe()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)

    results: list[RowResult] = []
    for i, a in enumerate(anchors):
        print(f"[{i+1}/{len(anchors)}] {a.anchor_id} ({a.strategy}) ...", flush=True)
        rr = run_anchor(a, mode, out_dir, verbose=verbose)
        # 落 manifest (审计 / 给硬件师复跑)
        with open(manifest_dir / f"{a.anchor_id}_prune.json", "w", encoding="utf-8") as f:
            json.dump(rr.prune_manifest, f, ensure_ascii=False, indent=2)
        with open(manifest_dir / f"{a.anchor_id}_quant.json", "w", encoding="utf-8") as f:
            json.dump(rr.quant_manifest, f, ensure_ascii=False, indent=2)
        with open(manifest_dir / f"{a.anchor_id}_deploy.json", "w", encoding="utf-8") as f:
            json.dump(rr.deploy_report, f, ensure_ascii=False, indent=2)
        print(f"     → real={rr.row['is_real_measured']} "
              f"eff_bits={rr.row['effective_bits']} trt={rr.row['trt_precision']} "
              f"size={rr.row['engine_size_mb']} src={rr.row['source']}", flush=True)
        results.append(rr)
    return results


def write_csv(results: list[RowResult], path: Path) -> None:
    cols = SCHEMA_COLUMNS + EXTRA_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rr in results:
            w.writerow({c: rr.row.get(c, "") for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description="DoE → 三工具管线 → 预测器训练数据")
    ap.add_argument("--mode", choices=["dry_run", "prune_real", "full"],
                    default="dry_run")
    ap.add_argument("--hardware", choices=["rtx4090", "orin_agx"], default=None,
                    help="只跑单硬件 (默认两硬件全跑)")
    ap.add_argument("--out-dir", default=str(_REPO / "output/doe_dataset_v1"))
    ap.add_argument("--csv", default=None, help="输出数据 CSV (默认 <out_dir>/doe_dataset_v1.csv)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "doe_dataset_v1.csv"

    print(f"=== generate_dataset mode={args.mode} hardware={args.hardware or 'both'} ===")
    results = generate(args.mode, args.hardware, out_dir, verbose=args.verbose)
    write_csv(results, csv_path)

    n_real = sum(1 for r in results if r.row["is_real_measured"])
    print(f"\n=== 汇总 ===")
    print(f"总行数: {len(results)}  真测行: {n_real}  dry_run/plan: {len(results)-n_real}")
    print(f"数据 CSV: {csv_path}")
    print(f"manifest 目录: {out_dir / 'manifests'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
