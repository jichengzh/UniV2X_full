"""S2.2b screening — sweep ONE (precision,width) point, extract ALL-dimension
schedule products from scheduled-TIR + a coarse latency, append one CSV row.

This is the stage-A scanner of the coupling-discovery campaign
(dims_hardware_v4 §4.3). A bash driver loops it over the S1-width × S3-precision
grid. We extract, per point, the schedule products the campaign's signal
detectors need:
  D1 intrinsic  -> tvm_mma_sync / ptx mma / dp4a / scalar  (path class)
  L1 SMEM       -> total shared-scope buffer bytes
  L2 bind       -> threadIdx/blockIdx extents
  L3 bank       -> storage_align / buffer_dim_align factors
  L4 pipeline   -> software_pipeline stage count
  L5 reduction  -> reduction-loop / rfactor presence
  D3 tiling     -> loop-split factor signature

Always dumps the full scheduled-TIR to <work_root>/tir_<label>.txt so the parser
can be validated/refined against reality (we have not yet seen this build's exact
TVMScript text format -> dump-and-refine, do not trust regex blind).

Usage:
  python s2_2b_screen.py <precision int8|fp16> <Cin> <Cout> <H> <W> <groups> \
                         <work_root> <trials> <out_csv> <label>
"""
from __future__ import annotations
import sys, os, re, time, traceback
import numpy as np

PREC   = sys.argv[1]
CIN    = int(sys.argv[2]); COUT = int(sys.argv[3]); H = int(sys.argv[4]); W = int(sys.argv[5])
GROUPS = int(sys.argv[6])
WORK_ROOT = sys.argv[7]
TRIALS = int(sys.argv[8])
OUT_CSV = sys.argv[9]
LABEL  = sys.argv[10]
SEED   = int(sys.argv[11]) if len(sys.argv) > 11 else 0   # vary to test cross-seed stability (R4)
KSIZE  = int(sys.argv[12]) if len(sys.argv) > 12 else 1   # 1x1 dense or 3x3 grouped role


def build_conv(prec, cin, cout, h, w, groups, ksize=1):
    import tvm
    from tvm import relax
    dt_in = "int8" if prec == "int8" else "float16"
    dt_out = "int32" if prec == "int8" else "float16"
    pad = ksize // 2
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, cin, h, w), dt_in))
    # grouped conv: weight is (cout, cin//groups, kh, kw)
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin // groups, ksize, ksize), dt_in))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y = bb.emit(relax.op.nn.conv2d(x, wt, groups=groups, padding=(pad, pad),
                                           out_dtype=dt_out))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def parse_products(txt):
    """Extract all-dimension schedule products from this RENAMED Unity build's
    scheduled-TIR text (T.sblock_alloc_buffer / T.sblock_attr /
    T.thread_binding(T.int64(N), thread="...") / T.axis.reduce(T.int64(N)) /
    scope="shared.dyn"). Tolerant: returns 0/'' when a feature is absent.
    Validated against dumped tir_*.txt (2026-06-17)."""
    # --- D1 intrinsic / tensor-core path ---
    n_wmma_mma  = txt.count("tvm_mma_sync")
    n_wmma_load = txt.count("tvm_load_matrix_sync")
    n_wmma_fill = txt.count("tvm_fill_fragment")
    n_ptx_mma   = txt.count("ptx_mma") + txt.count("mma.sync") + txt.count("tvm_mma_ptx")
    n_dp4a      = txt.lower().count("dp4a")
    if n_wmma_mma > 0:
        d1_path = "WMMA"
    elif n_ptx_mma > 0:
        d1_path = "PTX_MMA"
    elif n_dp4a > 0:
        d1_path = "DP4A"
    else:
        d1_path = "SCALAR"
    # fragment shape from wmma.matrix match_buffer: (T.int64(16), T.int64(16))
    frags = re.findall(r'scope="wmma\.matrix_[ab]"', txt)
    n_frag = len(frags)
    # the fragment tile is the (16,16) in match_buffer lines tagged wmma.matrix
    fshapes = re.findall(r'match_buffer\([^\n]*?\(T\.int64\((\d+)\),\s*T\.int64\((\d+)\)\)[^\n]*?wmma\.matrix_[ab]', txt)
    frag = sorted(set(fshapes))[:2]

    # --- L5 reduction: T.axis.reduce(T.int64(N)) extents (N = K/16 for WMMA) ---
    red_ext = re.findall(r'T\.axis\.reduce\(T\.int64\((\d+)\)', txt)
    n_rfactor = txt.lower().count("rfactor")

    # --- L1 SMEM: shared.dyn buffers (shapes signature) ---
    shared_bufs = []
    for m in re.finditer(r'T\.sblock_alloc_buffer\((\(.*?\))\s*,\s*"(\w+)"\s*,\s*scope="(shared\.dyn|shared)"', txt):
        dims = re.findall(r'T\.int64\((\d+)\)', m.group(1)); dt = m.group(2)
        n = 1
        for d in dims:
            n *= int(d)
        b = {"int8": 1, "int32": 4, "float16": 2, "float32": 4}.get(dt, 2)
        shared_bufs.append((tuple(int(d) for d in dims), dt, n * b))
    n_shared = len(shared_bufs)
    smem_bytes = sum(b for _, _, b in shared_bufs)  # logical (dynamic) buffer size, coarse
    smem_sig = ";".join("x".join(str(d) for d in s) + f":{dt}" for s, dt, _ in shared_bufs)[:140]

    # --- L2 thread/block bind extents (grid signature) ---
    binds = {}
    for ext, name in re.findall(r'T\.thread_binding\(T\.int64\((\d+)\),\s*thread="([\w.]+)"', txt):
        binds.setdefault(name, []).append(int(ext))
    # primary extents (max per axis, since launch-level bind is the outermost)
    grid_sig = "|".join(f"{k}:{max(v)}" for k, v in sorted(binds.items()))

    # --- L3 bank-conflict padding: buffer_dim_align factors ---
    aligns = re.findall(r'buffer_dim_align":\s*\[\[([0-9,\s]+)\]\]', txt)
    align_factors = "|".join(a.replace(" ", "") for a in aligns)[:80]
    n_align = len(aligns)

    # --- L4 software pipeline ---
    n_pipe_stage = txt.count("software_pipeline_stage")
    n_unroll = txt.count("pragma_auto_unroll_max_step")

    # --- D3 tiling signature: all spatial loop extents (non-reduce) ---
    # captured indirectly via grid_sig + shared tile shapes above

    return {
        "d1_path": d1_path, "n_mma": n_wmma_mma, "n_ldmat": n_wmma_load,
        "n_fill": n_wmma_fill, "n_ptx": n_ptx_mma, "n_dp4a": n_dp4a,
        "n_frag": n_frag, "frag": "|".join("x".join(f) for f in frag),
        "red_ext": "|".join(red_ext[:4]), "n_rfactor": n_rfactor,
        "n_shared": n_shared, "smem_bytes": smem_bytes, "smem_sig": smem_sig,
        "grid_sig": grid_sig, "n_align": n_align, "align_factors": align_factors,
        "n_pipe_stage": n_pipe_stage, "n_unroll": n_unroll,
        "tir_len": len(txt),
    }


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa: registers wmma/mma intrins

    k = (CIN // GROUPS) * KSIZE * KSIZE  # contraction dim = (Cin/g)*kh*kw
    print(f"[{LABEL}] prec={PREC} Cin={CIN} Cout={COUT} g={GROUPS} ksize={KSIZE} "
          f"K={k} k32={k%32==0} k16={k%16==0} {H}x{W} trials={TRIALS} seed={SEED}", flush=True)

    work_dir = os.path.join(WORK_ROOT, LABEL)
    os.makedirs(work_dir, exist_ok=True)
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_conv(PREC, CIN, COUT, H, W, GROUPS, KSIZE)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    t_tune0 = time.time()
    db = ri.tune_relax(mod=mod, params={}, target=target, work_dir=work_dir,
                       max_trials_global=TRIALS, seed=SEED)
    tune_s = time.time() - t_tune0
    print(f"[{LABEL}] tune done {tune_s:.0f}s", flush=True)

    # scheduled TIR (static products, GPU-noise immune)
    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(mod)
    txt = sched.script()
    dump_path = os.path.join(WORK_ROOT, f"tir_{LABEL}.txt")
    with open(dump_path, "w") as f:
        f.write(txt)
    prod = parse_products(txt)
    print(f"[{LABEL}] products {prod}", flush=True)

    # clean latency = best measured run_secs from the tuning DB (LocalRunner on
    # the real GPU). Far more reliable than a VM loop on a us-scale kernel, whose
    # dispatch overhead dominates (pilot: VM gave 56us vs tuner 4.2us). Still a
    # screening signal only; publishable latency comes on idle 4090/Orin (S2.3).
    best_us = -1.0
    try:
        recs = db.get_all_tuning_records()
        best = None
        for r in recs:
            rs = getattr(r, "run_secs", None)
            if not rs:
                continue
            vals = [float(x) for x in rs if x is not None]
            if vals:
                m = sum(vals) / len(vals)
                if best is None or m < best:
                    best = m
        if best is not None:
            best_us = best * 1e6
        print(f"[{LABEL}] best_run_us={best_us:.3f} over {len(recs)} recs", flush=True)
    except Exception as e:
        print(f"[{LABEL}] lat-from-db err {repr(e)[:120]}", flush=True)

    # append CSV row
    cols = ["label", "prec", "cin", "cout", "groups", "k", "k32", "k16", "hw",
            "trials", "tune_s", "best_us", "d1_path", "n_mma", "n_ldmat", "n_fill",
            "n_ptx", "n_dp4a", "n_frag", "frag", "red_ext", "n_rfactor",
            "n_shared", "smem_bytes", "smem_sig", "grid_sig", "n_align",
            "align_factors", "n_pipe_stage", "n_unroll", "tir_len"]
    row = [LABEL, PREC, CIN, COUT, GROUPS, k, int(k % 32 == 0), int(k % 16 == 0),
           f"{H}x{W}", TRIALS, f"{tune_s:.0f}", f"{best_us:.3f}", prod["d1_path"],
           prod["n_mma"], prod["n_ldmat"], prod["n_fill"], prod["n_ptx"],
           prod["n_dp4a"], prod["n_frag"], prod["frag"], prod["red_ext"],
           prod["n_rfactor"], prod["n_shared"], prod["smem_bytes"],
           prod["smem_sig"], prod["grid_sig"], prod["n_align"],
           prod["align_factors"], prod["n_pipe_stage"], prod["n_unroll"],
           prod["tir_len"]]
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        # quote fields that may contain commas
        safe = ['"%s"' % c if ("," in str(c) or ";" in str(c) or "|" in str(c)) else str(c) for c in row]
        f.write(",".join(safe) + "\n")
    print(f"[{LABEL}] RESULT path={prod['d1_path']} red_ext={prod['red_ext']} "
          f"grid={prod['grid_sig']} align={prod['align_factors']} "
          f"best_us={best_us:.3f} -> {OUT_CSV}", flush=True)
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
