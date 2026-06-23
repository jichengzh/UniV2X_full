"""Stage B Phase-2 — sequential AP eval on GPU 0, no contention.

Discovers all engines in models/stage_b_cache/*_ft_{fp16,int8}.engine, runs AP
eval one at a time on GPU 0 so dataloader I/O isn't thrashed by parallel evals.

Stops cleanly when all known engines have JSON reports.
Loops over the cache directory so newly arrived engines (built by the parallel
finetune script) are picked up automatically — terminates when no progress made
for 10 minutes.
"""
from __future__ import annotations
import json, os, signal, subprocess, time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
B_CACHE = REPO_ROOT / "models/stage_b_cache"
B_OUT = REPO_ROOT / "results/stage_b"
B_OUT.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = 30
NO_PROGRESS_TIMEOUT = 600  # 10 min — if no new engines and queue empty, exit


def _ensure_single_bestval(ft_dir: Path) -> None:
    """HEAL's load_saved_model asserts exactly one bestval@*.pth.

    If ft_dir contains both at23 (baseline) and a post-finetune bestval@N>23,
    move the at23 ckpt aside so HEAL loads the finetuned one.
    """
    bestvals = sorted(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if len(bestvals) <= 1:
        return
    # Find post-finetune bestval (highest epoch number)
    keep = None
    keep_epoch = -1
    for p in bestvals:
        import re
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) > keep_epoch:
            keep_epoch = int(m.group(1))
            keep = p
    for p in bestvals:
        if p == keep: continue
        backup = ft_dir / f"_baseline_{p.name}"
        if not backup.exists():
            p.rename(backup)
            print(f"  [setup] stashed {p.name} → {backup.name} (keeping bestval@{keep_epoch})")


def run_one(engine: Path, gpu: str = "0") -> dict | None:
    name = engine.stem  # e.g. "032_072_128_ft_fp16"
    parts = name.split("_")
    sig = "_".join(parts[:3])
    prec = parts[-1]
    tag = f"sB_{sig}_{prec}"
    report = B_OUT / f"{tag}.json"
    if report.exists():
        return json.loads(report.read_text())
    ft_dir = B_CACHE / f"ft_{sig}"
    if not ft_dir.exists():
        print(f"  [skip] {tag}: ft_dir missing")
        return None
    _ensure_single_bestval(ft_dir)
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
           "--engine-collab", str(engine),
           "--tag", tag,
           "--model-dir", str(ft_dir),
           "--n-samples", "1789", "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256",
           "--collab-tego-shape", "2,2,3",
           "--report", str(report)]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": os.environ.get("PATH", "")}
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(HEAL_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=3600)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        # Kill entire process group so dataloader workers don't leak GPU memory
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        time.sleep(3)
        print(f"  [timeout] {tag} > 3600s — killed pgroup, moving on")
        return None
    elapsed = time.time() - t0
    if returncode != 0 or not report.exists():
        print(f"  [FAIL] {tag} ({elapsed:.0f}s): {stderr[-300:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"  [done] {tag}  ap30={rep['ap30']:.4f}  ap50={rep['ap50']:.4f}  ap70={rep['ap70']:.4f}  ({elapsed:.0f}s)")
    return rep


def discover_engines() -> list[Path]:
    return sorted(B_CACHE.glob("*_ft_*.engine"))


def main():
    print("=" * 72)
    print("Stage B Sequential AP eval (GPU 0, no parallel contention)")
    print("=" * 72)
    t0 = time.time()
    rows = []
    seen = set()
    last_progress = time.time()

    while True:
        engines = discover_engines()
        pending = [e for e in engines if e.name not in seen]
        if not pending:
            elapsed_idle = time.time() - last_progress
            if elapsed_idle > NO_PROGRESS_TIMEOUT:
                print(f"\n[idle] no new engines for {NO_PROGRESS_TIMEOUT}s, exiting")
                break
            print(f"[wait] queue empty, polling for new engines... ({elapsed_idle:.0f}s idle)")
            time.sleep(POLL_INTERVAL)
            continue

        for eng in pending:
            seen.add(eng.name)
            print(f"\n[{len(seen)}/?] {eng.name}")
            rep = run_one(eng)
            if rep is not None:
                name = eng.stem
                parts = name.split("_")
                sig = "_".join(parts[:3])
                prec = parts[-1]
                rows.append({
                    "triplet_sig": sig, "precision": prec,
                    "stage0_planes": int(parts[0]),
                    "stage1_planes": int(parts[1]),
                    "stage2_planes": int(parts[2]),
                    "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
                    "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
                    "elapsed_secs": rep["elapsed_secs"],
                })
                last_progress = time.time()
            # Save incremental parquet after each anchor
            if rows:
                df = pd.DataFrame(rows)
                out = REPO_ROOT / "data/stage_b_ap_real.parquet"
                df.to_parquet(out); df.to_csv(out.with_suffix(".csv"), index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n[done] {len(rows)} anchors in {elapsed:.1f} min")
    if rows:
        df = pd.DataFrame(rows)
        out = REPO_ROOT / "data/stage_b_ap_real.parquet"
        df.to_parquet(out); df.to_csv(out.with_suffix(".csv"), index=False)
        print(f"saved → {out}")
        print(df[["triplet_sig","precision","ap50","ap70"]].to_string(index=False))


if __name__ == "__main__":
    main()
