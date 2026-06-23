"""Track A 后处理 — ONNX export for 13 new triplets.

每个新 triplet (T10-T22) 训完后:
  1. 找最新 bestval ckpt (net_epoch_bestval_at{N}.pth, N 最大)
  2. tools/export_onnx_pyramid_e2e.py 导出 ONNX
  3. 输出 models/e2e_cache/{tag}.onnx

CPU 工作 (load ckpt + export), 不抢 GPU. 13 个串行 ≈ 5-10 min.

用法:
  python a4_export_onnx_new_triplets.py            # 全 13 triplet
  python a4_export_onnx_new_triplets.py --only T10_p11   # 单个
"""
from __future__ import annotations
import argparse, re, subprocess, time
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
EXPORT_SCRIPT = REPO / "tools/export_onnx_pyramid_e2e.py"
CACHE = REPO / "models/dataset_a_cache"
ONNX_OUT = REPO / "models/e2e_cache"
MAX_VOX = 32000

# (tag, planes) — 13 new triplets (T9 (64,256,128) excluded, 不可行)
TRIPLETS = [
    ("T10_p11", (16, 128, 256)),
    ("T11_p14", (64,  64, 256)),
    ("T12_p21", (32,  64, 256)),
    ("T13_p29", (32,  32, 256)),
    ("T14_p36", (16,  16, 256)),
    ("T15_p39", (16, 128, 128)),
    ("T16_p54", (16,  64, 128)),
    ("T17_p57", (32,  32, 128)),
    ("T18_p64", (16,  16, 128)),
    ("T19_p71", (32,  32,  64)),
    ("T20_p79", (16,  16,  64)),
    ("T21_p82", (16,  32,  32)),
    ("T22_p89", (16,  16,  16)),
]


def find_latest_bestval(ft_dir: Path) -> Path | None:
    """Find net_epoch_bestval_at{N}.pth with largest N."""
    cands = list(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if not cands:
        return None
    def epoch_of(p):
        m = re.search(r"at(\d+)", p.name)
        return int(m.group(1)) if m else -1
    cands.sort(key=epoch_of, reverse=True)
    return cands[0]


def export_one(tag, planes):
    s0, s1, s2 = planes
    sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
    ft_dir = CACHE / f"ft_{sig}"
    if not ft_dir.is_dir():
        print(f"[skip] {tag} ({sig}): ft_dir missing")
        return False
    ckpt = find_latest_bestval(ft_dir)
    if ckpt is None:
        print(f"[skip] {tag} ({sig}): no bestval ckpt yet (still training)")
        return False
    cfg = ft_dir / "config.yaml"
    out = ONNX_OUT / f"{tag}.onnx"
    if out.exists():
        print(f"[skip] {tag}: ONNX exists at {out.name}")
        return True

    cmd = [PYTHON, str(EXPORT_SCRIPT),
           "--ckpt", str(ckpt),
           "--hypes", str(cfg),
           "--out", str(out),
           "--max-voxels", str(MAX_VOX)]
    t0 = time.time()
    print(f"[export] {tag}: {ckpt.name} → {out.name}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-500:]}")
        return False
    if not out.exists():
        print(f"  FAILED: ONNX not produced")
        return False
    sz = out.stat().st_size / 1024 / 1024
    print(f"  OK ({time.time()-t0:.0f}s, {sz:.1f}MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="单个 tag")
    args = ap.parse_args()

    targets = TRIPLETS if not args.only else [t for t in TRIPLETS if t[0] == args.only]
    print(f"[a4] 导出 {len(targets)} 个 ONNX")
    n_ok = 0
    for tag, planes in targets:
        if export_one(tag, planes):
            n_ok += 1
    print(f"\n[a4] {n_ok}/{len(targets)} OK")


if __name__ == "__main__":
    main()
