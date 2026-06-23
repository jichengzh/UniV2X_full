"""
W_g 探针 Phase 1 v2 — prune x schedule (AP,lat) Pareto 重排 grid (TVM, H800)
v2 改进 (针对 H800 ssh 不稳 + 慢 reuse re-timing):
  - 4 个已测宽度(base/p50/trap25/pad64)直接复用 gap1_schedule_lut.json 的 default/tuned 数, 不再慢测。
  - 只测 4 个新宽度(p75/iso_s0/iso_s1/iso_s2): default + fresh MetaSchedule tune + tuned。
  - 每测完一个宽度就增量写 gap1_grid_lut.json(进度防丢, 死了能续)。
  - REPS 降到 200/repeat 3, min-of-N 仍稳。
nohup 后台跑:
  cd /exdata/jichengzhi/s2_tvm && export PATH=/usr/local/cuda-12.2/bin:$PATH \
    && export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path) && export CUDA_VISIBLE_DEVICES=0 \
    && nohup /exdata/jichengzhi/tvm310/bin/python gap1_grid_v2.py > grid_v2.log 2>&1 &
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
PRIOR_LUT = os.path.join(OUT_DIR, "gap1_schedule_lut.json")
OUT_JSON = os.path.join(OUT_DIR, "gap1_grid_lut.json")
os.makedirs(OUT_DIR, exist_ok=True)

TRIALS = 1000
REPS = 200
SEED = 0

# in_per_g = 2*plane/32 per stage; power-of-2 => aligned (fast grouped kernel)
# (label, onnx_fn, work_dir, num_filters, ap70, note)  -- only the 4 NEW widths measured here
NEW = [
    ("p75",    "p75_backbone.onnx",    "ms_work_grid_p75",    [16, 32, 64],  0.530,
     "aligned [16,32,64] in_per_g=[1,2,4] all 2^k"),
    ("iso_s0", "iso_s0_backbone.onnx", "ms_work_grid_iso_s0", [48, 128, 256], None,
     "ONLY s0 misaligned [48,128,256] in_per_g=[3,8,16]"),
    ("iso_s1", "iso_s1_backbone.onnx", "ms_work_grid_iso_s1", [64, 96, 256],  None,
     "ONLY s1 misaligned [64,96,256] in_per_g=[4,6,16]"),
    ("iso_s2", "iso_s2_backbone.onnx", "ms_work_grid_iso_s2", [64, 128, 192], None,
     "ONLY s2 misaligned [64,128,192] in_per_g=[4,8,12]"),
]

# reuse prior measured 4 widths
prior = json.load(open(PRIOR_LUT))
prior_by = {r["label"]: r for r in prior["schedule_lut"]}
REUSE_AP = {"base": 0.631, "p50": 0.564, "trap25": 0.590, "trap25_pad64": 0.590}
REUSE_NF = {"base": [64, 128, 256], "p50": [32, 64, 128],
            "trap25": [48, 96, 192], "trap25_pad64": [64, 96, 192]}
results = []
for lab in ["base", "p50", "trap25", "trap25_pad64"]:
    p = prior_by[lab]
    results.append({"label": lab, "onnx": p["onnx"], "num_filters": REUSE_NF[lab],
                    "default_us": p["default_us"], "tuned_us": p["tuned_us"],
                    "ratio": p["ratio"], "ap70": REUSE_AP[lab],
                    "note": p.get("note", "") + " [REUSED from gap1_schedule_lut.json]",
                    "source": "reused"})


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in m.graph.input if i.name not in init}


def time_vm(vm, args, dev, reps=REPS):
    vm["main"](*args)
    dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=3)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


def pareto(points):
    front = []
    for p in points:
        dom = False
        for q in points:
            if q is p:
                continue
            if q["ap"] >= p["ap"] and q["lat"] <= p["lat"] and (q["ap"] > p["ap"] or q["lat"] < p["lat"]):
                dom = True
                break
        if not dom:
            front.append(p["label"])
    return front


def write_out(results):
    have_ap = [r for r in results if r.get("ap70") is not None and r["default_us"] > 0 and r["tuned_us"] > 0]
    def_pts = [{"label": r["label"], "ap": r["ap70"], "lat": r["default_us"]} for r in have_ap]
    tun_pts = [{"label": r["label"], "ap": r["ap70"], "lat": r["tuned_us"]} for r in have_ap]
    df, tf = pareto(def_pts), pareto(tun_pts)
    out = {
        "experiment": "W_g probe Phase 1 v2 — prune x schedule (AP,lat) Pareto reshaping grid",
        "hardware": "H800 Hopper, CUDA 12.2, TVM 0.20.dev1070",
        "caveat": [
            "backbone-only subnet (NOT e2e); Amdahl applies",
            "relative latency valid across widths (same platform/protocol, min-of-N)",
            "base/p50/trap25/pad64 default/tuned REUSED from gap1_schedule_lut.json (REPS=500); new widths REPS=200",
            "AP70 from 4090/DAIR stage_a (finetuned); iso_* latency-only (ap70=null until Phase2 finetune)",
            "in_per_g=2*plane/32; power-of-2 => aligned fast grouped kernel",
        ],
        "grid": results,
        "pareto": {
            "default_front_labels": df, "tuned_front_labels": tf,
            "W_g_candidates_on_default_not_tuned": sorted(set(df) - set(tf)),
            "P_g_candidates_on_tuned_not_default": sorted(set(tf) - set(df)),
            "note": "Pareto over widths WITH ap70 only; iso_* excluded until finetuned",
        },
    }
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT_JSON)


write_out(results)  # initial: 4 reused points
print("Reused 4 widths. Now measuring 4 new widths.", flush=True)

dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)

for label, onnx_fn, work_subdir, nf, ap70, note in NEW:
    onnx_path = os.path.join(MODELS_DIR, onnx_fn)
    work_dir = os.path.join(WORK_ROOT, work_subdir)
    os.makedirs(work_dir, exist_ok=True)
    print(f"\n=== {label} nf={nf} ===", flush=True)
    rec = {"label": label, "onnx": onnx_fn, "num_filters": nf, "ap70": ap70,
           "note": note, "source": "measured", "default_us": -1, "tuned_us": -1, "ratio": -1}
    try:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)
        m_onnx = onnx.load(onnx_path)
        SH = read_inputs(m_onnx)
        rng = np.random.RandomState(0)
        feeds_np = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
        mod0 = from_onnx(m_onnx, shape_dict=SH, keep_params_in_input=False)

        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in SH]
        def_us, def_mn = time_vm(vm, args, dev)
        rec["default_us"] = round(def_us, 2)
        rec["default_min_us"] = round(def_mn, 2)
        print(f"  DEFAULT mean={def_us:.1f}us min={def_mn:.1f}us", flush=True)

        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(), relax.transform.FuseTIR()])
        with tgt, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
        db_file = os.path.join(work_dir, "database_tuning_record.json")
        if not os.path.exists(db_file):
            print(f"  Tuning ({TRIALS} trials)...", flush=True)
            t0 = time.time()
            ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=work_dir,
                          max_trials_global=TRIALS, seed=SEED)
            rec["tune_s"] = round(time.time() - t0)
            print(f"  Tune done {rec['tune_s']}s", flush=True)
        else:
            rec["tune_s"] = 0
            print("  reuse db", flush=True)
        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(modt)
            ex2 = tvm.compile(sched, target=tgt)
        vm2 = relax.VirtualMachine(ex2, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in SH]
        tun_us, tun_mn = time_vm(vm2, args, dev)
        rec["tuned_us"] = round(tun_us, 2)
        rec["tuned_min_us"] = round(tun_mn, 2)
        rec["ratio"] = round(def_us / tun_us, 3) if tun_us > 0 else -1
        print(f"  TUNED mean={tun_us:.1f}us min={tun_mn:.1f}us ratio={rec['ratio']}x", flush=True)
    except Exception:
        traceback.print_exc()
        rec["note"] += " [MEASURE FAILED]"
    results.append(rec)
    write_out(results)  # incremental write after each width
    print(f"  written {OUT_JSON} ({len(results)} widths)", flush=True)

# final summary
have_ap = [r for r in results if r.get("ap70") is not None and r["default_us"] > 0 and r["tuned_us"] > 0]
print("\n=== DONE ===")
print("default-Pareto:", pareto([{"label": r["label"], "ap": r["ap70"], "lat": r["default_us"]} for r in have_ap]))
print("tuned-Pareto:  ", pareto([{"label": r["label"], "ap": r["ap70"], "lat": r["tuned_us"]} for r in have_ap]))
for r in results:
    print(f"  {r['label']:14s} nf={r['num_filters']} ratio={r['ratio']}x "
          f"def={r['default_us']}us tuned={r['tuned_us']}us ap70={r['ap70']} ipg={[2*p//32 for p in r['num_filters']]}")
