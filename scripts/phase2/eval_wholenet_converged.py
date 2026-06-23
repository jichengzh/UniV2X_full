"""重跑 wholenet 收敛 bestval 的 AP eval (orchestrator 被误杀后由主控接管)。
对 light/p50/aggr: 保留最高 bestval(其余 .bak)→ inference.py(DAIR val 1789, intermediate)
→ 产出 eval_intermediate.yaml。3 个并行 GPU 0/1/2。复用 run_ap 的口径。"""
import os, glob, re, subprocess, json
from pathlib import Path

HEAL = "/home/jichengzhi/heal_research/HEAL"
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CKROOT = "/home/jichengzhi/heal_research/checkpoints/stage1"
REPO = Path("/home/jichengzhi/UniV2X")
CFG = {"light": "0", "p50": "1", "aggr": "2"}

def highest_bestval(d):
    fs = glob.glob(f"{d}/net_epoch_bestval_at*.pth")
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"at(\d+)\.pth", f).group(1)))

procs = {}
for tag, gpu in CFG.items():
    d = f"{CKROOT}/Pyramid_DAIR_m1_wholenet_{tag}_2026_06_01"
    best = highest_bestval(d)
    if best is None:
        print(f"[{tag}] no bestval, skip", flush=True); continue
    # 保留最高 bestval, 其余 .bak (HEAL 要求单一 bestval)
    for f in glob.glob(f"{d}/net_epoch_bestval_at*.pth"):
        if f != best:
            os.rename(f, f + ".bak")
    yml = f"{d}/eval_intermediate.yaml"
    if os.path.exists(yml):
        os.remove(yml)
    log = open(REPO / f"results/ap_wholenet_{tag}_converged.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": HEAL}
    p = subprocess.Popen([PY, "opencood/tools/inference.py", "--model_dir", d,
                          "--fusion_method", "intermediate"],
                         cwd=HEAL, env=env, stdout=log, stderr=subprocess.STDOUT)
    procs[tag] = (p, d, os.path.basename(best))
    print(f"[{tag}] launched on GPU{gpu}, bestval={os.path.basename(best)}, pid={p.pid}", flush=True)

print("=== waiting for 3 inferences ===", flush=True)
results = {}
for tag, (p, d, best) in procs.items():
    rc = p.wait()
    yml = f"{d}/eval_intermediate.yaml"
    ap = {}
    if os.path.exists(yml):
        import yaml
        y = yaml.safe_load(open(yml)) or {}
        ap = {"ap30": y.get("ap30") or y.get("ap_30"),
              "ap50": y.get("ap_50") or y.get("ap50"),
              "ap70": y.get("ap_70") or y.get("ap70")}
    results[tag] = {"rc": rc, "bestval": best, "ap": ap}
    print(f"[{tag}] done rc={rc} bestval={best} AP={ap}", flush=True)

(REPO / "results/wholenet_converged_ap.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("=== ALL DONE, wrote results/wholenet_converged_ap.json ===", flush=True)
