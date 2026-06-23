"""S2.2c — per-knob ablation: prove each T1 hardware knob's effect on latency.

Methodology = MetaSchedule RULE-STACK ablation (Ansor/MetaSchedule standard):
take the full cuda schedule-rule stack as baseline, then DROP exactly one rule
and re-tune the SAME workload on an idle GPU. The OFF/ON latency ratio = that
knob's contribution. Avoids hand-written tensorize (which is brittle, see
force-K32). Rule <-> dimension map:
  tc                       -> default TC-enabled space (tensorize ON)   [D1]
  cuda_full                -> PostOrderApply(create("cuda"))  (no TC, all CUDA-core rules ON)
  no_MultiLevelTiling      -> drop tiling+shared-staging                 [D3 + L1]
  no_AutoBind              -> drop thread/block binding                  [L2]
  no_ParallelizeVectorizeUnroll -> drop vectorize/unroll/pipeline        [L4]
  no_CrossThreadReduction  -> drop cross-thread reduction                [L5]
delta(D1)  = cuda_full / tc
delta(knob)= no_<rule> / cuda_full

Usage:
  python s2_2c_knob_ablation.py <conv|matmul> <prec> <a> <b> <c> <d> <spec> \
                                <work_root> <trials> <out_csv> <label>
  conv:   a=Cin b=Cout c=ksize d=spatial(H), W=2*H  (groups=1)
  matmul: a=M   b=K    c=N     d=ignored
"""
from __future__ import annotations
import sys, os, re, time, traceback

WL = sys.argv[1]; PREC = sys.argv[2]
A = int(sys.argv[3]); B = int(sys.argv[4]); C = int(sys.argv[5]); D = int(sys.argv[6])
SPEC = sys.argv[7]; WORK_ROOT = sys.argv[8]; TRIALS = int(sys.argv[9])
OUT_CSV = sys.argv[10]; LABEL = sys.argv[11]


def build_workload():
    import tvm
    from tvm import relax
    di = "int8" if PREC == "int8" else "float16"
    do = "int32" if PREC == "int8" else "float16"
    bb = relax.BlockBuilder()
    if WL == "conv":
        cin, cout, ks, h = A, B, C, D
        w_ = 2 * h
        x = relax.Var("x", relax.TensorStructInfo((1, cin, h, w_), di))
        wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, ks, ks), di))
        with bb.function("main", [x, wt]):
            with bb.dataflow():
                y = bb.emit(relax.op.nn.conv2d(x, wt, padding=(ks // 2, ks // 2), out_dtype=do))
                gv = bb.emit_output(y)
            bb.emit_func_output(gv)
    else:  # matmul M,K,N
        m, k, n = A, B, C
        x = relax.Var("x", relax.TensorStructInfo((m, k), di))
        wt = relax.Var("wt", relax.TensorStructInfo((k, n), di))
        with bb.function("main", [x, wt]):
            with bb.dataflow():
                y = bb.emit(relax.op.matmul(x, wt, out_dtype=do))
                gv = bb.emit_output(y)
            bb.emit_func_output(gv)
    return bb.finalize()


def build_space(spec):
    """Return (space_or_None). None => default TC-enabled space.

    specs:
      tc                       -> default TC space
      cuda_full                -> full non-TC rule+postproc stack
      no_<RuleName>            -> drop a schedule rule (D3+L1 / L2 / L4 / L5)
      no_pp_<PostprocName>     -> drop a postproc (L3 = no_pp_RewriteCooperativeFetch)
      mlt_noreuse              -> replace MultiLevelTiling with reuse=no (isolate L1 staging)
    """
    if spec == "tc":
        return None
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import schedule_rule as sr
    rules = list(sr.ScheduleRule.create("cuda"))
    postprocs = list(ms.postproc.Postproc.create("cuda"))
    if spec.startswith("no_pp_"):
        drop = spec[len("no_pp_"):]
        postprocs = [p for p in postprocs if type(p).__name__ != drop]
    elif spec == "mlt_noreuse":
        # replace MultiLevelTiling with a no-shared-reuse variant (L1 staging OFF, tiling ON)
        noreuse = sr.MultiLevelTiling(
            structure="SSSRRSRS", tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            max_innermost_factor=64, vector_load_lens=[1, 2, 3, 4],
            reuse_read=sr.ReuseType(req="no", levels=[], scope=""),
            reuse_write=sr.ReuseType(req="no", levels=[], scope=""))
        rules = [noreuse if type(r).__name__ == "MultiLevelTiling" else r for r in rules]
    elif spec != "cuda_full":
        drop = spec[len("no_"):]
        rules = [r for r in rules if type(r).__name__ != drop]
    return ms.space_generator.PostOrderApply(
        sch_rules=rules, postprocs=postprocs,
        mutator_probs=ms.mutator.Mutator.create("cuda"))


def parse_path(txt):
    n_mma = txt.count("tvm_mma_sync"); n_dp4a = txt.lower().count("dp4a")
    n_ptx = txt.lower().count("ptx_mma")
    path = "WMMA" if n_mma > 0 else ("PTX_MMA" if n_ptx > 0 else ("DP4A" if n_dp4a > 0 else "SCALAR"))
    binds = set(re.findall(r'thread="([\w.]+)"', txt))
    n_shared = txt.count('scope="shared')
    n_align = txt.count("buffer_dim_align")
    n_pipe = txt.count("software_pipeline_stage")
    return path, n_mma, "+".join(sorted(binds)), n_shared, n_align, n_pipe


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa

    print(f"[{LABEL}] WL={WL} prec={PREC} dims=({A},{B},{C},{D}) spec={SPEC} trials={TRIALS}", flush=True)
    work_dir = os.path.join(WORK_ROOT, LABEL); os.makedirs(work_dir, exist_ok=True)
    dev = tvm.cuda(0); target = tvm.target.Target.from_device(dev)
    mod = build_workload()
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    space = build_space(SPEC)

    t0 = time.time()
    path, n_mma, binds, n_shared, n_align, n_pipe, best_us = "TUNE_FAIL", 0, "", 0, 0, 0, -1.0
    try:
        kw = dict(mod=mod, params={}, target=target, work_dir=work_dir,
                  max_trials_global=TRIALS, seed=0)
        if space is not None:
            kw["space"] = space
        db = ri.tune_relax(**kw)
        tune_s = time.time() - t0
        try:
            with target, tvm.transform.PassContext(opt_level=3):
                sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(mod)
            txt = sched.script()
            with open(os.path.join(WORK_ROOT, f"tir_{LABEL}.txt"), "w") as f:
                f.write(txt)
            path, n_mma, binds, n_shared, n_align, n_pipe = parse_path(txt)
        except Exception as e:
            print(f"[{LABEL}] apply-db err {repr(e)[:100]}", flush=True)
        best = None
        for r in db.get_all_tuning_records():
            rs = getattr(r, "run_secs", None)
            if rs:
                vals = [float(x) for x in rs if x is not None]
                if vals:
                    m = sum(vals) / len(vals); best = m if best is None or m < best else best
        if best is not None:
            best_us = best * 1e6
    except Exception as e:
        tune_s = time.time() - t0
        print(f"[{LABEL}] TUNE/RUN FAILED {repr(e)[:140]}", flush=True)

    cols = ["label", "wl", "prec", "a", "b", "c", "d", "spec", "trials", "tune_s",
            "best_us", "path", "n_mma", "binds", "n_shared", "n_align", "n_pipe"]
    row = [LABEL, WL, PREC, A, B, C, D, SPEC, TRIALS, f"{tune_s:.0f}",
           f"{best_us:.3f}", path, n_mma, binds, n_shared, n_align, n_pipe]
    hdr = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if hdr:
            f.write(",".join(cols) + "\n")
        safe = ['"%s"' % c if ("," in str(c) or "+" in str(c)) else str(c) for c in row]
        f.write(",".join(safe) + "\n")
    print(f"[{LABEL}] RESULT spec={SPEC} path={path} binds=[{binds}] n_shared={n_shared} "
          f"n_align={n_align} n_pipe={n_pipe} best_us={best_us:.3f}", flush=True)
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
