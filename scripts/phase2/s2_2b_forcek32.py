"""S2.2b stage-B deepen: FORCE the PTX-MMA K32 tensor-core path and test the
alignment cliff that the default WMMA-K16 path dissolves.

Default int8 tuning picks WMMA m16n16k16 (K=16) -> 48 (=3x16) aligns, no cliff
(screening proved this across all widths). But TRT's INT8 NCHW32 and the
high-throughput PTX path use K=32 granularity. Here we build a MetaSchedule
space whose ONLY tensor-core rule uses get_mma_intrin_group (MMA_i8i8i32 =
m16n8k32, K=32). Then:
  w64  (K=64,  64%32==0) -> expect tensorize on MMA-K32  -> TC schedule
  w48  (K=48,  48%32==16)-> expect tensorize FAILS       -> fall through to
                            non-TC (scalar) -> the controlled cliff
  w48 pad->64             -> expect TC re-engages (FAST/ALT pad mitigation)

This is the TVM-exclusive co-configuration demo for S2.2c: pad / intrinsic
choice is a knob TRT does not expose to a search.

Usage: python s2_2b_forcek32.py <Cin> <Cout> <H> <W> <work_root> <label> [trials] [pad_to]
"""
from __future__ import annotations
import sys, os, re, time, traceback

CIN = int(sys.argv[1]); COUT = int(sys.argv[2]); H = int(sys.argv[3]); W = int(sys.argv[4])
WORK_ROOT = sys.argv[5]; LABEL = sys.argv[6]
TRIALS = int(sys.argv[7]) if len(sys.argv) > 7 else 64
PAD_TO = int(sys.argv[8]) if len(sys.argv) > 8 else 0


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


def build_k32_space():
    """PostOrderApply with a K32-MMA-only tensor-core rule prepended to the
    standard cuda rule set, so the block tensorizes ONLY if its K divides 32."""
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import schedule_rule as sr
    from tvm.s_tir.tensor_intrin.cuda import get_mma_intrin_group, get_mma_store_intrin
    from tvm.tirx import TensorIntrin
    # The shared-scope i32 "simple" store intrin (BufferStore variant) is not
    # pre-registered in this build (only global i32 + f16/f32 shared.dyn are).
    # Register it under the exact name get_mma_intrin_group requests.
    store_name = "mma_store_16x16_i32_shared_simple_"
    if TensorIntrin.get(store_name, allow_missing=True) is None:
        TensorIntrin.register(store_name,
                              *get_mma_store_intrin("int32", 8, "shared", use_mma_store_intrinic=False),
                              override=True)
    # non-dyn 'shared' scope: MMA load_a/load_b/compute/init intrins registered
    # there (the *_dyn variants are not in this build).
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
    base = list(sr.ScheduleRule.create("cuda"))  # 7 default cuda rules (non-TC)
    rules = [mlt_tc] + base
    postprocs = list(ms.postproc.Postproc.create("cuda"))
    mutators = ms.mutator.Mutator.create("cuda")
    return ms.space_generator.PostOrderApply(
        sch_rules=rules, postprocs=postprocs, mutator_probs=mutators)


def parse_path(txt):
    n_wmma = txt.count("tvm_mma_sync")        # WMMA K16 builtin
    n_ptx = txt.count("ptx_mma") + txt.count("mma_sync") - n_wmma  # ptx K32 (mma_sync substring minus wmma)
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

    cin_eff = PAD_TO if PAD_TO else CIN
    K = cin_eff
    print(f"[{LABEL}] FORCE-K32 Cin={CIN} eff={cin_eff} Cout={COUT} K={K} "
          f"k32={K%32==0} k16={K%16==0} trials={TRIALS}", flush=True)

    work_dir = os.path.join(WORK_ROOT, LABEL)
    os.makedirs(work_dir, exist_ok=True)
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_int8_conv(cin_eff, COUT, H, W)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    space = build_k32_space()
    print(f"[{LABEL}] K32-only space generator built", flush=True)

    t0 = time.time()
    try:
        db = ri.tune_relax(mod=mod, params={}, target=target, work_dir=work_dir,
                           max_trials_global=TRIALS, space=space, seed=0)
    except Exception as e:
        print(f"[{LABEL}] TUNE_FAILED (K32 cannot apply?) {repr(e)[:160]}", flush=True)
        print(f"[{LABEL}] RESULT label={LABEL} K={K} k32={K%32==0} path=TUNE_FAILED", flush=True)
        return
    print(f"[{LABEL}] tune done {time.time()-t0:.0f}s", flush=True)

    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(mod)
    txt = sched.script()
    with open(os.path.join(WORK_ROOT, f"tir_{LABEL}.txt"), "w") as f:
        f.write(txt)
    path, n_ptx, n_wmma, n_dp4a, red = parse_path(txt)

    best_us = -1.0
    recs = db.get_all_tuning_records()
    best = None
    for r in recs:
        rs = getattr(r, "run_secs", None)
        if rs:
            vals = [float(x) for x in rs if x is not None]
            if vals:
                m = sum(vals) / len(vals)
                best = m if best is None or m < best else best
    if best is not None:
        best_us = best * 1e6

    print(f"[{LABEL}] path={path} n_ptx={n_ptx} n_wmma={n_wmma} n_dp4a={n_dp4a} "
          f"red_ext={red} best_us={best_us:.3f}", flush=True)
    print(f"[{LABEL}] RESULT label={LABEL} Cin={CIN} eff={cin_eff} K={K} "
          f"k32={K%32==0} path={path} best_us={best_us:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
