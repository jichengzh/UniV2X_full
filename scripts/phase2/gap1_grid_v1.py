"""
W_g 探针 Phase 1 — prune×schedule (AP, latency) Pareto 重排 grid (TVM, H800)
generalizes gap1_run_v2.py to a full width grid.

每个宽度: default(dlight) + MetaSchedule-tuned latency (min-of-N), 复用已有 ms_work_2e_* db,
新宽度 fresh tune. 产出 gap1_grid_lut.json: 每宽度 default/tuned/ratio/ap70
+ default-Pareto vs tuned-Pareto on (ap70, lat).

信号: 各宽度 schedule 调优余量(ratio)不均匀 → default-(AP,lat)-Pareto ≠ tuned-(AP,lat)-Pareto
      ⇒ 单侧贪心(按 default 估计选宽度)锁 W_g(default-Pareto/tuned 掉出), 错过 P_g.

H800 环境:
  export PATH=/usr/local/cuda-12.2/bin:$PATH
  export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
  export CUDA_VISIBLE_DEVICES=0
  /exdata/jichengzhi/tvm310/bin/python /exdata/jichengzhi/s2_tvm/gap1_grid_v1.py
"""
import os, time, json, traceback
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir.meta_schedule import relax_integration as ri
import tvm.s_tir.tensor_intrin.cuda  # noqa
import onnx

MODELS_DIR = "/exdata/jichengzhi/s2_tvm/models"
OUT_DIR = "/exdata/jichengzhi/s2_tvm/results"
WORK_ROOT = "/exdata/jichengzhi/s2_tvm"
os.makedirs(OUT_DIR, exist_ok=True)

TRIALS = 1000
REPS = 500
SEED = 0

# (label, onnx_fn, work_dir_name, do_fresh_tune, num_filters, ap70_or_None, in_per_g, note)
# in_per_g = 2*plane/32 per stage; power-of-2 => aligned (fast grouped kernel)
configs = [
    # --- reuse existing (tuned db present) ---
    ("base",     "base_backbone.onnx",     "ms_work_2e_base",   False, [64, 128, 256], 0.631,
     "aligned [64,128,256] in_per_g=[4,8,16] all 2^k"),
    ("p50",      "p50_backbone.onnx",      "ms_work_2e_p50",    False, [32, 64, 128],  0.564,
     "aligned [32,64,128] in_per_g=[2,4,8] all 2^k"),
    ("trap25",   "trap25_backbone.onnx",   "ms_work_2e_trap25", False, [48, 96, 192],  0.590,
     "ALL-misaligned [48,96,192] in_per_g=[3,6,12] none 2^k"),
    ("trap25_pad64", "trap25_pad64_backbone.onnx", "ms_work_2e_pad64", False, [64, 96, 192], 0.590,
     "s0 padded /32 but s1/s2 still misaligned in_per_g=[4,6,12]"),
    # --- new (latency-only, fresh tune) ---
    ("p75",      "p75_backbone.onnx",      "ms_work_grid_p75",   True, [16, 32, 64],  0.530,
     "aligned [16,32,64] in_per_g=[1,2,4] all 2^k"),
    ("iso_s0",   "iso_s0_backbone.onnx",   "ms_work_grid_iso_s0", True, [48, 128, 256], None,
     "ONLY s0 misaligned [48,128,256] in_per_g=[3,8,16]"),
    ("iso_s1",   "iso_s1_backbone.onnx",   "ms_work_grid_iso_s1", True, [64, 96, 256],  None,
     "ONLY s1 misaligned [64,96,256] in_per_g=[4,6,16]"),
    ("iso_s2",   "iso_s2_backbone.onnx",   "ms_work_grid_iso_s2", True, [64, 128, 192], None,
     "ONLY s2 misaligned [64,128,192] in_per_g=[4,8,12]"),
]


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in m.graph.input if i.name not in init}


def time_vm(vm, args, dev, reps=REPS):
    vm["main"](*args)
    dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)
results = []

for label, onnx_fn, work_subdir, do_tune, nf, ap70, note in configs:
    onnx_path = os.path.join(MODELS_DIR, onnx_fn)
    if not os.path.exists(onnx_path):
        print(f"\n=== {label}: ONNX MISSING ({onnx_path}) -> SKIP ===", flush=True)
        results.append({"label": label, "onnx": onnx_fn, "num_filters": nf,
                        "default_us": -1, "tuned_us": -1, "ratio": -1,
                        "ap70": ap70, "note": note + " [ONNX MISSING]"})
        continue
    work_dir = os.path.join(WORK_ROOT, work_subdir)
    os.makedirs(work_dir, exist_ok=True)
    print(f"\n=== {label} nf={nf} (tune={do_tune}) ===", flush=True)

    m_onnx = onnx.load(onnx_path)
    SH = read_inputs(m_onnx)
    print(f"  inputs: {SH}", flush=True)
    rng = np.random.RandomState(0)
    feeds_np = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
    mod0 = from_onnx(m_onnx, shape_dict=SH, keep_params_in_input=False)

    def_us = -1.0
    try:
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in SH]
        def_us, def_mn = time_vm(vm, args, dev)
        print(f"  DEFAULT mean={def_us:.1f}us min={def_mn:.1f}us", flush=True)
    except Exception:
        traceback.print_exc()
        print("  DEFAULT FAILED", flush=True)
        def_mn = -1.0

    tun_us = -1.0
    tun_mn = -1.0
    tune_s = -1
    try:
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR()])
        with tgt, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)

        db_file = os.path.join(work_dir, "database_tuning_record.json")
        if do_tune or not os.path.exists(db_file):
            print(f"  Tuning ({TRIALS} trials)...", flush=True)
            t0 = time.time()
            ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=work_dir,
                          max_trials_global=TRIALS, seed=SEED)
            tune_s = time.time() - t0
            print(f"  Tune done in {tune_s:.0f}s", flush=True)
        else:
            print("  Reusing existing work_dir", flush=True)
            tune_s = 0

        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(modt)
            ex2 = tvm.compile(sched, target=tgt)
        vm2 = relax.VirtualMachine(ex2, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in SH]
        tun_us, tun_mn = time_vm(vm2, args, dev)
        print(f"  TUNED mean={tun_us:.1f}us min={tun_mn:.1f}us", flush=True)
    except Exception:
        traceback.print_exc()
        print("  TUNED FAILED", flush=True)

    ratio = def_us / tun_us if (def_us > 0 and tun_us > 0) else -1.0
    print(f"  ratio={ratio:.3f}x", flush=True)
    results.append({
        "label": label, "onnx": onnx_fn, "num_filters": nf,
        "default_us": round(def_us, 2), "default_min_us": round(def_mn, 2),
        "tuned_us": round(tun_us, 2), "tuned_min_us": round(tun_mn, 2),
        "ratio": round(ratio, 3), "tune_s": tune_s,
        "ap70": ap70, "note": note,
    })


# ---- Pareto on (ap70, latency): lower lat + higher ap dominates ----
def pareto(points):
    # points: list of dict with 'label','ap','lat'; return labels on Pareto front
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["ap"] >= p["ap"] and q["lat"] <= p["lat"] and (q["ap"] > p["ap"] or q["lat"] < p["lat"]):
                dominated = True
                break
        if not dominated:
            front.append(p["label"])
    return front


have_ap = [r for r in results if r.get("ap70") is not None and r["default_us"] > 0 and r["tuned_us"] > 0]
def_pts = [{"label": r["label"], "ap": r["ap70"], "lat": r["default_us"]} for r in have_ap]
tun_pts = [{"label": r["label"], "ap": r["ap70"], "lat": r["tuned_us"]} for r in have_ap]
def_front = pareto(def_pts)
tun_front = pareto(tun_pts)
# W_g candidates: on default-Pareto but dropped from tuned-Pareto
wg = sorted(set(def_front) - set(tun_front))
pg = sorted(set(tun_front) - set(def_front))

out = {
    "experiment": "W_g probe Phase 1 — prune x schedule (AP,lat) Pareto reshaping grid",
    "hardware": "H800 Hopper, CUDA 12.2, TVM 0.20.dev1070",
    "caveat": [
        "backbone-only subnet (NOT e2e); Amdahl applies",
        "relative latency valid across widths (same platform/protocol, min-of-N)",
        "AP70 from 4090/DAIR stage_a (finetuned); iso_* widths latency-only (ap70=null, need finetune for Phase 2)",
        "in_per_g = 2*plane/32; power-of-2 => aligned fast grouped kernel",
    ],
    "grid": results,
    "pareto": {
        "default_front_labels": def_front,
        "tuned_front_labels": tun_front,
        "W_g_candidates_on_default_not_tuned": wg,
        "P_g_candidates_on_tuned_not_default": pg,
        "note": "Pareto over widths WITH ap70 only; iso_* excluded until finetuned",
    },
}
json_path = os.path.join(OUT_DIR, "gap1_grid_lut.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE. Written: {json_path}")
print("default-Pareto:", def_front)
print("tuned-Pareto:  ", tun_front)
print("W_g (default-Pareto, tuned-dropped):", wg)
print("P_g (tuned-Pareto, default-dropped):", pg)
print("\nratios (tuning headroom) by width:")
for r in results:
    print(f"  {r['label']:14s} nf={r['num_filters']} ratio={r['ratio']}x "
          f"def={r['default_us']}us tuned={r['tuned_us']}us ap70={r['ap70']}")
