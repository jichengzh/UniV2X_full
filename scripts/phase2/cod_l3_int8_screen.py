"""L3 Step-3: INT8 alignment screen for CoDriving 3x3 conv shapes.

Key research question: Does standard-conv CoDriving become COUPLED under INT8?
Mechanism: INT8 WMMA requires K (reduction dim) ÷ 32.
For 3x3 conv: K = Cin * 9, so K÷32 ⟺ Cin÷32.

Test Cin = 16 (p75), 32 (p50), 48 (p25), 64 (base), 96, 128:
  Cin=16: K=144, 144÷32=4.5 ✗ → expect WMMA FAIL or degraded
  Cin=32: K=288, 288÷32=9 ✓ → expect WMMA OK
  Cin=48: K=432, 432÷32=13.5 ✗ → expect WMMA FAIL or degraded
  Cin=64: K=576, 576÷32=18 ✓ → expect WMMA OK
  Cin=96: K=864, 864÷32=27 ✓ → expect WMMA OK
  Cin=128: K=1152, 1152÷32=36 ✓ → expect WMMA OK

Output CSV columns: label, prec, cin, k, k_div32, k_mod32, d1_path, lat_us, trials, tc_frac

Usage: python cod_l3_int8_screen.py <out_csv> [gpu=7] [trials=200]
"""
from __future__ import annotations
import sys, os, re, time, traceback, json
import numpy as np

OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/results/cod_int8_screen.csv"
GPU_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 7
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 200

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# CoDriving 3x3 conv: stage0 inner conv Cin = num_filters[0]
# H,W for stage0 output at various backbone widths (after stride-2 downsample from 256x512)
# stage0: spatial 128x256 (stride=2 from H=256,W=512)
CONFIGS = [
    # (cin, cout, hw_h, hw_w, label, pruning)
    (16, 16, 128, 256, "cod_p75_s0", "p75"),   # stage0 of p75 backbone
    (32, 32, 128, 256, "cod_p50_s0", "p50"),   # stage0 of p50 backbone
    (48, 48, 128, 256, "cod_p25_s0", "p25"),   # stage0 of p25 backbone (MISALIGNED)
    (64, 64, 128, 256, "cod_base_s0", "base"), # stage0 of base backbone
    (96, 96, 64, 128, "cod_p50_s1", "p50_s1"), # stage1 of p50 backbone
    (128, 128, 64, 128, "cod_base_s1", "base_s1"), # stage1 of base backbone
]

BATCH = 2
KSIZE = 3  # 3x3 conv
KTYPE = "int8"


def build_3x3_int8_conv(cin, cout, h, w, batch=2):
    """Build a 3x3 int8 conv as TVM relax module."""
    import tvm
    from tvm import relax
    pad = 1  # same padding for 3x3
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((batch, cin, h, w), "int8"))
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, KSIZE, KSIZE), "int8"))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y = bb.emit(relax.op.nn.conv2d(x, wt, padding=(pad, pad), out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


TC_KEYS = ("tvm_mma_sync", "tvm_load_matrix_sync", "tvm_fill_fragment",
           "ptx_mma", "mma.sync", "dp4a", "wmma")


def tune_and_measure(cin, cout, h, w, label, work_dir, trials, batch=2):
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa

    dev = tvm.cuda(0)
    tgt = tvm.target.Target.from_device(dev)

    mod = build_3x3_int8_conv(cin, cout, h, w, batch)

    # Prepare input data (int8)
    rng = np.random.RandomState(42)
    x_data = rng.randint(-64, 64, (batch, cin, h, w)).astype("int8")
    w_data = rng.randint(-64, 64, (cout, cin, KSIZE, KSIZE)).astype("int8")

    k = cin * KSIZE * KSIZE
    k32 = (k % 32 == 0)
    k32_val = k % 32

    print(f"\n[{label}] Cin={cin} Cout={cout} H={h}xW={w} K={k} K÷32={'OK ✓' if k32 else f'FAIL ✗ (mod={k32_val})'}", flush=True)

    # Legalize and fuse
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with tgt, tvm.transform.PassContext(opt_level=3):
        modt = seq(mod)

    # Tune
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=work_dir,
                  max_trials_global=trials, seed=42)
    tune_s = time.time() - t0
    print(f"[{label}] Tuning done in {tune_s:.1f}s", flush=True)

    # Apply and compile
    import tvm.s_tir.dlight as dl
    with tgt, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(modt)
        sched = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
        ex = tvm.compile(sched, target=tgt)

    # Measure latency
    vm = relax.VirtualMachine(ex, dev)
    x_tvm = tvm.runtime.tensor(x_data, device=dev)
    w_tvm = tvm.runtime.tensor(w_data, device=dev)
    args = [x_tvm, w_tvm]

    vm["main"](*args); dev.sync()
    vf = vm.time_evaluator("main", dev, number=300, repeat=5)
    r = vf(*args)
    lat_us = r.mean * 1e6
    lat_min = min(r.results) * 1e6
    print(f"[{label}] lat_us={lat_us:.2f} min_us={lat_min:.2f}", flush=True)

    # Check TC path from scheduled TIR
    try:
        from tvm.s_tir.meta_schedule import database as db_mod
        db = db_mod.JSONDatabase(work_dir)
        records = db.get_all_tuning_records()
        tc_count = 0
        for rec in records[:min(50, len(records))]:
            try:
                tir_txt = str(rec.trace)
                if any(k.lower() in tir_txt.lower() for k in TC_KEYS):
                    tc_count += 1
            except Exception:
                pass
        tc_frac = tc_count / max(1, min(50, len(records)))
        print(f"[{label}] TC traces: {tc_count}/{min(50, len(records))} ({tc_frac:.1%})", flush=True)
    except Exception as e:
        tc_frac = -1.0
        print(f"[{label}] TC check failed: {e}", flush=True)

    # Detect d1 path from compiled module
    d1_path = "UNKNOWN"
    try:
        src = ex.mod.get_source() if hasattr(ex, 'mod') else ""
        src_l = src.lower()
        if "tvm_mma_sync" in src_l or "mma.sync" in src_l or "wmma" in src_l:
            d1_path = "WMMA"
        elif "dp4a" in src_l:
            d1_path = "DP4A"
        elif "ptx_mma" in src_l:
            d1_path = "PTX_MMA"
        else:
            d1_path = "SCALAR"
    except Exception:
        d1_path = "UNKNOWN"

    print(f"[{label}] d1_path={d1_path}", flush=True)

    return {
        "label": label,
        "prec": "int8",
        "cin": cin,
        "cout": cout,
        "k": k,
        "k_div32": int(k32),
        "k_mod32": int(k32_val),
        "d1_path": d1_path,
        "lat_us": lat_us,
        "lat_min_us": lat_min,
        "tune_s": tune_s,
        "tc_frac": tc_frac,
        "trials": trials,
    }


def main():
    print(f"[L3 INT8 SCREEN] GPU={GPU_ID} trials={TRIALS} ksize={KSIZE} out={OUT_CSV}", flush=True)

    # Write CSV header
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("label,prec,cin,cout,k,k_div32,k_mod32,d1_path,lat_us,lat_min_us,tune_s,tc_frac,trials\n")

    for (cin, cout, h, w, label, pruning) in CONFIGS:
        work_dir = f"/exdata/jichengzhi/s2_tvm/ms_work_cod_int8_{label}"
        # Remove existing workdir for fresh isolation
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        try:
            row = tune_and_measure(cin, cout, h, w, label, work_dir, TRIALS)
            with open(OUT_CSV, "a") as f:
                f.write(f"{row['label']},{row['prec']},{row['cin']},{row['cout']},"
                        f"{row['k']},{row['k_div32']},{row['k_mod32']},"
                        f"{row['d1_path']},{row['lat_us']:.3f},{row['lat_min_us']:.3f},"
                        f"{row['tune_s']:.1f},{row['tc_frac']:.3f},{row['trials']}\n")
            print(f"[{label}] DONE: d1={row['d1_path']} lat={row['lat_us']:.2f}µs k32={bool(row['k_div32'])}", flush=True)
        except Exception:
            traceback.print_exc()
            with open(OUT_CSV, "a") as f:
                f.write(f"{label},int8,{cin},{cout},ERROR,-1,-1,ERROR,-1,-1,-1,-1,{TRIALS}\n")

    print("\n[L3 INT8 SCREEN] ALL DONE", flush=True)
    print(f"Results: {OUT_CSV}", flush=True)
    with open(OUT_CSV) as f:
        print(f.read(), flush=True)


if __name__ == "__main__":
    main()
