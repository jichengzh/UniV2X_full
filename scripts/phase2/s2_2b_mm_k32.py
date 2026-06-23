"""S2.2b option-3 retry: force PTX-MMA K32 on a clean int8 MATMUL (not conv).

The conv force-K32 attempt failed because MLT-TC(MMA group) did not engage even
for aligned w64 (conv tensorize has extra reindex/layout structure). A plain
GEMM is the canonical MMA-tensorize case, far more likely to engage. We build
int8 matmul (M,K) x (K,N) -> (M,N) int32 and restrict the space to the MMA-K32
intrin group:
  K=64 (64%32==0) -> expect MMA-K32 engages -> PTX_MMA path
  K=48 (48%32==16)-> expect tensorize FAILS  -> SCALAR (the controlled cliff)
  K=48 pad->64    -> expect MMA re-engages (pad mitigation)

If K=64 engages PTX_MMA while K=48 does not, the alignment cliff on the
32-granularity path is demonstrated (the counterfactual that the default
WMMA-K16 path dissolves).

Usage: python s2_2b_mm_k32.py <M> <K> <N> <work_root> <label> [trials] [pad_to]
"""
from __future__ import annotations
import sys, os, re, time, traceback

M = int(sys.argv[1]); K = int(sys.argv[2]); N = int(sys.argv[3])
WORK_ROOT = sys.argv[4]; LABEL = sys.argv[5]
TRIALS = int(sys.argv[6]) if len(sys.argv) > 6 else 64
PAD_TO = int(sys.argv[7]) if len(sys.argv) > 7 else 0


def build_int8_matmul(m, k, n):
    import tvm
    from tvm import relax
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, k), "int8"))
    w = relax.Var("w", relax.TensorStructInfo((k, n), "int8"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w, out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def build_k32_space():
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import schedule_rule as sr
    from tvm.s_tir.tensor_intrin.cuda import get_mma_intrin_group, get_mma_store_intrin
    from tvm.tirx import TensorIntrin
    store_name = "mma_store_16x16_i32_shared_simple_"
    if TensorIntrin.get(store_name, allow_missing=True) is None:
        TensorIntrin.register(store_name,
                              *get_mma_store_intrin("int32", 8, "shared", use_mma_store_intrinic=False),
                              override=True)
    mma_group = get_mma_intrin_group(
        load_scope="shared", store_scope="shared",
        in_dtype="int8", out_dtype="int32", trans_a=False, trans_b=False,
        not_use_mma_store_intrinic=True)
    mlt_tc = sr.MultiLevelTilingTensorCore(
        intrin_groups=[mma_group], structure="SSSRRSRS",
        tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
        max_innermost_factor=4, vector_load_lens=[1, 2, 3, 4, 8, 16],
        reuse_read=sr.ReuseType(req="must", levels=[4], scope="shared"),
        reuse_write=sr.ReuseType(req="must", levels=[2], scope="shared"),
        use_software_pipeline=False)
    base = list(sr.ScheduleRule.create("cuda"))
    rules = [mlt_tc] + base
    return ms.space_generator.PostOrderApply(
        sch_rules=rules, postprocs=list(ms.postproc.Postproc.create("cuda")),
        mutator_probs=ms.mutator.Mutator.create("cuda"))


def parse_path(txt):
    n_wmma = txt.count("tvm_mma_sync")
    n_ptx = txt.lower().count("ptx_mma")
    n_dp4a = txt.lower().count("dp4a")
    if n_ptx > 0:
        path = "PTX_MMA_K32"
    elif n_wmma > 0:
        path = "WMMA_K16"
    elif n_dp4a > 0:
        path = "DP4A"
    else:
        path = "SCALAR"
    red = re.findall(r'T\.axis\.reduce\(T\.int64\((\d+)\)', txt)
    return path, n_ptx, n_wmma, n_dp4a, "|".join(red[:4])


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa

    k_eff = PAD_TO if PAD_TO else K
    print(f"[{LABEL}] MM-FORCE-K32 M={M} K={K} eff={k_eff} N={N} "
          f"k32={k_eff%32==0} k16={k_eff%16==0} trials={TRIALS}", flush=True)
    work_dir = os.path.join(WORK_ROOT, LABEL)
    os.makedirs(work_dir, exist_ok=True)
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_int8_matmul(M, k_eff, N)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    space = build_k32_space()
    print(f"[{LABEL}] K32-only space built", flush=True)
    t0 = time.time()
    try:
        db = ri.tune_relax(mod=mod, params={}, target=target, work_dir=work_dir,
                           max_trials_global=TRIALS, space=space, seed=0)
    except Exception as e:
        print(f"[{LABEL}] TUNE_FAILED {repr(e)[:160]}", flush=True)
        print(f"[{LABEL}] RESULT label={LABEL} K={K} eff={k_eff} k32={k_eff%32==0} path=TUNE_FAILED", flush=True)
        return
    print(f"[{LABEL}] tune done {time.time()-t0:.0f}s", flush=True)
    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(mod)
    txt = sched.script()
    with open(os.path.join(WORK_ROOT, f"tir_{LABEL}.txt"), "w") as f:
        f.write(txt)
    path, n_ptx, n_wmma, n_dp4a, red = parse_path(txt)
    best = None
    for r in db.get_all_tuning_records():
        rs = getattr(r, "run_secs", None)
        if rs:
            vals = [float(x) for x in rs if x is not None]
            if vals:
                mm = sum(vals) / len(vals); best = mm if best is None or mm < best else best
    best_us = best * 1e6 if best else -1.0
    print(f"[{LABEL}] path={path} n_ptx={n_ptx} n_wmma={n_wmma} n_dp4a={n_dp4a} "
          f"red={red} best_us={best_us:.3f}", flush=True)
    print(f"[{LABEL}] RESULT label={LABEL} M={M} K={K} eff={k_eff} N={N} "
          f"k32={k_eff%32==0} path={path} best_us={best_us:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
