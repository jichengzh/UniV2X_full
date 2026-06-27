"""Stage1 S2 low-cost schedule anchor runner.

Runs small H800/TVM probes for the two measurement-backlog S2 anchors:

- std_basebev_backbone_schedule_anchor: representative standard Conv2d.
- standard_neck_deconv_anchor: representative ConvTranspose2d neck/deblock op.

The runner intentionally uses synthetic representative shapes instead of full
model export. It answers the S2 acceptance questions cheaply:

- P: base / moderate / boundary width points.
- Q: fp16 / int8.
- S: default dlight lowering, own MetaSchedule-tuned, and int8 with the
  fp16-tuned database applied as a schedule-swap attempt.
- H: batch 1 and a throughput batch.

Usage on H800:

  CUDA_VISIBLE_DEVICES=6 /exdata/jichengzhi/tvm310/bin/python \
    scripts/phase2/stage1_s2_anchor_runner.py \
    --out-json /exdata/jichengzhi/s2_tvm/results/stage1_s2_anchor_scan_v1.json \
    --work-root /exdata/jichengzhi/s2_tvm/ms_work/stage1_s2_anchor_scan_v1 \
    --trials 8 --reps 50 --batches 1,2
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchorSpec:
    probe_id: str
    op_kind: str
    input_hw: tuple[int, int]
    kernel: int
    stride: int
    padding: int
    p_points: tuple[tuple[str, int], ...]
    scope_note: str


ANCHORS: tuple[AnchorSpec, ...] = (
    AnchorSpec(
        probe_id="std_basebev_backbone_schedule_anchor",
        op_kind="conv2d",
        input_hw=(32, 64),
        kernel=3,
        stride=1,
        padding=1,
        p_points=(("boundary", 32), ("moderate", 48), ("base", 64)),
        scope_note="Synthetic OpenCOOD BaseBEVBackbone-like standard Conv2d representative.",
    ),
    AnchorSpec(
        probe_id="standard_neck_deconv_anchor",
        op_kind="conv2d_transpose",
        input_hw=(16, 32),
        kernel=2,
        stride=2,
        padding=0,
        p_points=(("boundary", 128), ("moderate", 192), ("base", 256)),
        scope_note=(
            "Synthetic neck/deblock ConvTranspose2d representative. The real neck search group "
            "is mixed ConvTranspose2d + Conv2d, so this is a deconv-subgroup anchor."
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", default="1,2")
    parser.add_argument("--anchors", default="all")
    return parser.parse_args()


def _selected_anchors(value: str) -> list[AnchorSpec]:
    if value == "all":
        return list(ANCHORS)
    names = {v.strip() for v in value.split(",") if v.strip()}
    selected = [a for a in ANCHORS if a.probe_id in names]
    missing = names - {a.probe_id for a in selected}
    if missing:
        raise SystemExit(f"Unknown anchors: {sorted(missing)}")
    return selected


def _build_mod(anchor: AnchorSpec, precision: str, batch: int, width: int):
    import tvm
    from tvm import relax

    dtype = "int8" if precision == "int8" else "float16"
    out_dtype = "int32" if precision == "int8" else "float16"
    h, w = anchor.input_hw
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((batch, width, h, w), dtype))
    if anchor.op_kind == "conv2d":
        wt = relax.Var(
            "wt",
            relax.TensorStructInfo((width, width, anchor.kernel, anchor.kernel), dtype),
        )
        with bb.function("main", [x, wt]):
            with bb.dataflow():
                y = bb.emit(
                    relax.op.nn.conv2d(
                        x,
                        wt,
                        strides=(anchor.stride, anchor.stride),
                        padding=(anchor.padding, anchor.padding),
                        groups=1,
                        out_dtype=out_dtype,
                    )
                )
                gv = bb.emit_output(y)
            bb.emit_func_output(gv)
    elif anchor.op_kind == "conv2d_transpose":
        wt = relax.Var(
            "wt",
            # Relax conv2d_transpose default kernel_layout is IOHW.
            relax.TensorStructInfo((width, width, anchor.kernel, anchor.kernel), dtype),
        )
        with bb.function("main", [x, wt]):
            with bb.dataflow():
                y = bb.emit(
                    relax.op.nn.conv2d_transpose(
                        x,
                        wt,
                        strides=(anchor.stride, anchor.stride),
                        padding=(anchor.padding, anchor.padding),
                        output_padding=(0, 0),
                        groups=1,
                        out_dtype=out_dtype,
                    )
                )
                gv = bb.emit_output(y)
            bb.emit_func_output(gv)
    else:
        raise ValueError(anchor.op_kind)
    return bb.finalize()


def _prepare_mod(mod, target):
    import tvm
    from tvm import relax

    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        return seq(mod)


def _apply_dlight(mod, target):
    import tvm
    import tvm.s_tir.dlight as dl

    with target, tvm.transform.PassContext(opt_level=3):
        return dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)


def _compile(mod, target):
    import tvm

    with target, tvm.transform.PassContext(opt_level=3):
        return tvm.compile(mod, target=target)


def _make_inputs(anchor: AnchorSpec, precision: str, batch: int, width: int, seed: int, dev):
    import tvm

    rng = np.random.RandomState(seed)
    h, w = anchor.input_hw
    if precision == "int8":
        x_np = rng.randint(-64, 64, (batch, width, h, w)).astype("int8")
        wt_np = rng.randint(-64, 64, (width, width, anchor.kernel, anchor.kernel)).astype("int8")
    else:
        x_np = rng.standard_normal((batch, width, h, w)).astype("float16")
        wt_np = rng.standard_normal((width, width, anchor.kernel, anchor.kernel)).astype("float16")
    return [tvm.runtime.tensor(x_np, device=dev), tvm.runtime.tensor(wt_np, device=dev)]


def _time_vm(ex, args, dev, reps: int) -> dict[str, float]:
    from tvm import relax

    vm = relax.VirtualMachine(ex, dev)
    vm["main"](*args)
    dev.sync()
    timer = vm.time_evaluator("main", dev, number=reps, repeat=5)
    result = timer(*args)
    return {
        "mean_us": float(result.mean * 1e6),
        "min_us": float(min(result.results) * 1e6),
        "std_us": float(np.std(result.results) * 1e6),
    }


def _tir_products(mod) -> dict[str, Any]:
    try:
        text = mod.script()
    except Exception:
        return {"script_available": False}
    lower = text.lower()
    if "tvm_mma_sync" in text or "mma.sync" in lower or "wmma" in lower:
        d1_path = "WMMA"
    elif "dp4a" in lower:
        d1_path = "DP4A"
    else:
        d1_path = "SCALAR_OR_FALLBACK"
    return {
        "script_available": True,
        "d1_path": d1_path,
        "n_tvm_mma_sync": text.count("tvm_mma_sync"),
        "n_wmma": lower.count("wmma"),
        "n_dp4a": lower.count("dp4a"),
        "tir_len": len(text),
    }


def _safe_run(fn):
    try:
        value = fn()
        return {"ok": True, "value": value}
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "traceback_tail": traceback.format_exc()[-2000:],
        }


def _measure_cell(
    *,
    anchor: AnchorSpec,
    p_label: str,
    width: int,
    precision: str,
    batch: int,
    work_dir: Path,
    fp16_swap_work_dir: Path | None,
    trials: int,
    reps: int,
    seed: int,
    target,
    dev,
) -> dict[str, Any]:
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri

    cell: dict[str, Any] = {
        "probe_id": anchor.probe_id,
        "op_kind": anchor.op_kind,
        "p_label": p_label,
        "width": width,
        "precision": precision,
        "batch": batch,
        "input_hw": list(anchor.input_hw),
        "kernel": anchor.kernel,
        "stride": anchor.stride,
        "padding": anchor.padding,
        "trials": trials,
        "reps": reps,
        "work_dir": str(work_dir),
    }
    mod0 = _build_mod(anchor, precision, batch, width)
    modt = _prepare_mod(mod0, target)
    args = _make_inputs(anchor, precision, batch, width, seed, dev)

    def default_arm():
        try:
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(mod0, target="cuda")
            return {"schedule": "default_relax_build", **_time_vm(ex, args, dev, reps)}
        except Exception as exc:
            fallback_mod = _apply_dlight(modt, target)
            ex = _compile(fallback_mod, target)
            return {
                "schedule": "default_dlight_fallback",
                "relax_build_error": repr(exc),
                **_time_vm(ex, args, dev, reps),
                "tir_products": _tir_products(fallback_mod),
            }

    cell["default"] = _safe_run(default_arm)

    def own_tuned_arm():
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        db = ri.tune_relax(
            mod=modt,
            params={},
            target=target,
            work_dir=str(work_dir),
            max_trials_global=trials,
            seed=seed,
        )
        tune_s = time.time() - t0
        with target, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=str(work_dir))(modt)
            sched = _apply_dlight(sched, target)
            ex = tvm.compile(sched, target=target)
        recs = []
        try:
            recs = db.get_all_tuning_records()
        except Exception:
            recs = []
        return {
            "schedule": "own_metaschedule_tuned",
            "tune_s": tune_s,
            "n_records": len(recs),
            **_time_vm(ex, args, dev, reps),
            "tir_products": _tir_products(sched),
        }

    cell["own_tuned"] = _safe_run(own_tuned_arm)

    if precision == "int8" and fp16_swap_work_dir is not None:
        cell["schedule_swap"] = _safe_run(
            lambda: _schedule_swap_arm(
                modt=modt,
                args=args,
                dev=dev,
                reps=reps,
                target=target,
                fp16_work_dir=fp16_swap_work_dir,
            )
        )
    else:
        cell["schedule_swap"] = {
            "ok": False,
            "error": "not_applicable_for_non_int8_or_missing_fp16_workdir",
        }
    return cell


def _schedule_swap_arm(*, modt, args, dev, reps: int, target, fp16_work_dir: Path) -> dict[str, Any]:
    import tvm
    from tvm import relax

    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=str(fp16_work_dir))(modt)
        sched = _apply_dlight(sched, target)
        ex = tvm.compile(sched, target=target)
    return {
        "schedule": "int8_graph_with_fp16_tuned_database_then_dlight_fallback",
        "fp16_work_dir": str(fp16_work_dir),
        **_time_vm(ex, args, dev, reps),
        "tir_products": _tir_products(sched),
        "validity_note": (
            "TVM database matching is workload-hash based; cross-dtype matches may be zero. "
            "Use this as an attempted schedule-swap latency plus script/product evidence, not "
            "as proof that every fp16 trace matched int8."
        ),
    }


def _rank(values: list[dict[str, Any]], arm: str) -> list[str]:
    usable = []
    for row in values:
        arm_result = row.get(arm, {})
        if arm_result.get("ok") and arm_result.get("value", {}).get("mean_us", -1) > 0:
            usable.append((arm_result["value"]["mean_us"], row["p_label"]))
    return [label for _, label in sorted(usable)]


def _analyze(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_anchor: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_anchor.setdefault(cell["probe_id"], []).append(cell)
    out: dict[str, Any] = {}
    for probe_id, rows in by_anchor.items():
        probe_summary: dict[str, Any] = {"comparisons": [], "cell_failures": []}
        for row in rows:
            for arm in ("default", "own_tuned"):
                if not row.get(arm, {}).get("ok"):
                    probe_summary["cell_failures"].append(
                        {
                            "p": row["p_label"],
                            "precision": row["precision"],
                            "batch": row["batch"],
                            "arm": arm,
                            "error": row.get(arm, {}).get("error"),
                        }
                    )
        for precision in sorted({r["precision"] for r in rows}):
            for batch in sorted({r["batch"] for r in rows}):
                subset = [r for r in rows if r["precision"] == precision and r["batch"] == batch]
                default_rank = _rank(subset, "default")
                tuned_rank = _rank(subset, "own_tuned")
                gains = []
                for row in subset:
                    d = row.get("default", {})
                    t = row.get("own_tuned", {})
                    if d.get("ok") and t.get("ok"):
                        d_us = d["value"].get("mean_us", -1)
                        t_us = t["value"].get("mean_us", -1)
                        if d_us > 0 and t_us > 0:
                            gains.append({"p": row["p_label"], "gain": d_us / t_us})
                probe_summary["comparisons"].append(
                    {
                        "precision": precision,
                        "batch": batch,
                        "default_rank": default_rank,
                        "tuned_rank": tuned_rank,
                        "default_vs_tuned_rank_flip": default_rank != tuned_rank,
                        "schedule_gains": gains,
                    }
                )
        for batch in sorted({r["batch"] for r in rows}):
            int8_rows = [r for r in rows if r["precision"] == "int8" and r["batch"] == batch]
            gaps = []
            for row in int8_rows:
                own = row.get("own_tuned", {})
                swp = row.get("schedule_swap", {})
                if own.get("ok") and swp.get("ok"):
                    own_us = own["value"].get("mean_us", -1)
                    swp_us = swp["value"].get("mean_us", -1)
                    if own_us > 0 and swp_us > 0:
                        gaps.append(
                            {
                                "p": row["p_label"],
                                "own_tuned_us": own_us,
                                "fp16_tuned_swap_us": swp_us,
                                "swap_over_own": swp_us / own_us,
                            }
                        )
            probe_summary.setdefault("schedule_swap_gaps", []).append({"batch": batch, "gaps": gaps})
        out[probe_id] = probe_summary
    return out


def main() -> None:
    args = _parse_args()
    batches = [int(v.strip()) for v in args.batches.split(",") if v.strip()]
    work_root = Path(args.work_root)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    import tvm
    import tvm.s_tir.tensor_intrin.cuda  # noqa: registers cuda tensor intrinsics

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    anchors = _selected_anchors(args.anchors)

    cells: list[dict[str, Any]] = []
    started = time.time()
    for anchor in anchors:
        fp16_work_dirs: dict[tuple[str, int], Path] = {}
        for batch in batches:
            for p_label, width in anchor.p_points:
                for precision in ("fp16", "int8"):
                    work_dir = work_root / anchor.probe_id / f"b{batch}_{p_label}_w{width}_{precision}"
                    fp16_swap_work_dir = fp16_work_dirs.get((p_label, batch))
                    print(
                        f"[S2] {anchor.probe_id} batch={batch} p={p_label}:{width} q={precision}",
                        flush=True,
                    )
                    cell = _measure_cell(
                        anchor=anchor,
                        p_label=p_label,
                        width=width,
                        precision=precision,
                        batch=batch,
                        work_dir=work_dir,
                        fp16_swap_work_dir=fp16_swap_work_dir,
                        trials=args.trials,
                        reps=args.reps,
                        seed=args.seed,
                        target=target,
                        dev=dev,
                    )
                    cells.append(cell)
                    if precision == "fp16" and cell.get("own_tuned", {}).get("ok"):
                        fp16_work_dirs[(p_label, batch)] = work_dir
                    partial = {
                        "schema": "stage1_s2_anchor_scan_v1",
                        "status": "running",
                        "created_unix": started,
                        "updated_unix": time.time(),
                        "tvm_version": tvm.__version__,
                        "target": str(target),
                        "trials": args.trials,
                        "reps": args.reps,
                        "batches": batches,
                        "anchors": [
                            {
                                "probe_id": a.probe_id,
                                "op_kind": a.op_kind,
                                "scope_note": a.scope_note,
                                "p_points": list(a.p_points),
                            }
                            for a in anchors
                        ],
                        "cells": cells,
                        "analysis": _analyze(cells),
                    }
                    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
                    tmp.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                    os.replace(tmp, out_json)

    final = {
        "schema": "stage1_s2_anchor_scan_v1",
        "status": "completed",
        "created_unix": started,
        "updated_unix": time.time(),
        "elapsed_s": time.time() - started,
        "tvm_version": tvm.__version__,
        "target": str(target),
        "trials": args.trials,
        "reps": args.reps,
        "batches": batches,
        "anchors": [
            {
                "probe_id": a.probe_id,
                "op_kind": a.op_kind,
                "scope_note": a.scope_note,
                "p_points": list(a.p_points),
                "input_hw": list(a.input_hw),
                "kernel": a.kernel,
                "stride": a.stride,
                "padding": a.padding,
            }
            for a in anchors
        ],
        "cells": cells,
        "analysis": _analyze(cells),
    }
    out_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[S2] DONE -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
