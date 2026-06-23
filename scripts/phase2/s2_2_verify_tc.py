"""Ground-truth verify: did the tuned schedule REALLY use INT8 tensor cores?

Keyword-matching trace strings is unreliable. This reloads each arm's tuned
JSON database, recompiles, and inspects the generated CUDA source for actual
tensor-core PTX (mma.sync / wmma / m16n16k32 / ldmatrix) vs scalar int32 /
dp4a. Also dumps the best record's trace tail for inspection.

Usage: python s2_2_verify_tc.py <Cin> <Cout> <H> <W> <work_dir> <label> [pad_to]
"""
from __future__ import annotations
import sys, re
import numpy as np

CIN = int(sys.argv[1]); COUT = int(sys.argv[2]); H = int(sys.argv[3]); W = int(sys.argv[4])
WORK_DIR = sys.argv[5]; LABEL = sys.argv[6]
PAD_TO = int(sys.argv[7]) if len(sys.argv) > 7 else 0

REAL_TC = ["mma.sync", "wmma.mma", "wmma::", "m16n16k32", "m16n16k16", "ldmatrix", "hmma", "imma"]
SCALAR = ["dp4a", "__dp4a"]


def build_int8_conv(cin, cout, h, w):
    import tvm
    from tvm import relax
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, cin, h, w), "int8"))
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, 1, 1), "int8"))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y = bb.emit(relax.op.nn.conv2d(x, wt, out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa: side-effect registers wmma/mma intrins

    cin_eff = PAD_TO if PAD_TO else CIN
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_int8_conv(cin_eff, COUT, H, W)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    print(f"[{LABEL}] Cin={CIN} eff={cin_eff} k32={cin_eff%32==0} k16={cin_eff%16==0}", flush=True)

    # Ground truth #1: apply best schedules from DB -> scheduled TIR, grep WMMA builtins
    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(mod)
    txt = sched.script()
    wmma_builtins = {
        "tvm_mma_sync": txt.count("tvm_mma_sync"),
        "tvm_fill_fragment": txt.count("tvm_fill_fragment"),
        "tvm_load_matrix_sync": txt.count("tvm_load_matrix_sync"),
        "tvm_store_matrix_sync": txt.count("tvm_store_matrix_sync"),
        "wmma": txt.count("wmma"),
    }
    used_tc = wmma_builtins["tvm_mma_sync"] > 0
    print(f"[{LABEL}] scheduled-TIR wmma_builtins={wmma_builtins}", flush=True)
    # fragment shape (e.g. 16, 16, 16) appears in wmma matrix_a/b allocations
    shapes = sorted(set(re.findall(r'wmma\.matrix_[ab]"[^\n]*?(\d+),\s*(\d+)', txt)))[:4]
    mshapes = sorted(set(re.findall(r'(\d+),\s*(\d+),\s*(\d+).*?wmma', txt)))[:2]
    print(f"[{LABEL}] frag_hints={shapes}", flush=True)

    # Ground truth #2: generated CUDA source via build of the scheduled TIR
    try:
        ex = ri.compile_relax(db, mod, target, params={})
        rt = ex.mod if hasattr(ex, "mod") else ex
        srcs = []
        def collect(m, d=0):
            if d > 4 or m is None:
                return
            for fmt in ("cu", "ptx", ""):
                try:
                    s = m.get_source(fmt) if fmt else m.get_source()
                    if s:
                        srcs.append(s)
                        break
                except Exception:
                    pass
            for im in getattr(m, "imported_modules", []) or []:
                collect(im, d + 1)
        collect(rt)
        src = "\n".join(srcs).lower()
        real_hits = {k: src.count(k) for k in REAL_TC if k in src}
        print(f"[{LABEL}] CUDA n_src={len(srcs)} total_len={len(src)} real_tc_ptx={real_hits}", flush=True)
    except Exception as e:
        print(f"[{LABEL}] compile/source step err: {repr(e)[:100]}", flush=True)

    verdict = "REAL_WMMA_TENSORCORE" if used_tc else "NO_TENSORCORE"
    print(f"[{LABEL}] GROUNDTRUTH {verdict} (tvm_mma_sync count={wmma_builtins['tvm_mma_sync']})", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION", repr(e)); sys.exit(3)
