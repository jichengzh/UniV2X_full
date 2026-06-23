"""Parse tp_errors_v1_run.log to extract results as they come in.

Can be run at any point during or after the eval to see partial results.
"""
import re
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "results/eval_tp_errors_v1_run.log"


def parse_log(log_path=LOG_PATH):
    text = log_path.read_text() if log_path.exists() else ""

    # Extract RESULT blocks
    # Pattern: "anchor=X  precision=Y  engine=Z"
    # Then later: "RESULT X Y:\n  AP30=...\n  mATE=...\n"

    configs = re.findall(r"anchor=(\w+)\s+precision=(\w+)\s+engine=", text)
    results = re.findall(
        r"RESULT (\w+) (\w+):\n\s+AP30=([\d.]+)\s+AP50=([\d.]+)\s+AP70=([\d.]+)\n"
        r"\s+mATE=([\d.]+)m\s+mASE=([\d.]+)\s+mAOE=([\d.]+)rad\n"
        r"\s+n_tp=(\d+)\s+n_gt=(\d+)\s+elapsed=([\d.]+)s",
        text
    )

    rows = []
    for r in results:
        anchor, prec, ap30, ap50, ap70, mATE, mASE, mAOE, n_tp, n_gt, elapsed = r
        rows.append({
            "anchor": anchor,
            "precision": prec,
            "ap30": float(ap30),
            "ap50": float(ap50),
            "ap70": float(ap70),
            "mATE": float(mATE),
            "mASE": float(mASE),
            "mAOE": float(mAOE),
            "n_tp": int(n_tp),
            "n_gt": int(n_gt),
            "elapsed_secs": float(elapsed),
        })
    return rows


def main():
    rows = parse_log()
    if not rows:
        print("No RESULT blocks found in log yet.")
        return

    print(f"{'anchor':<12} {'prec':<6} {'AP50':<8} {'AP70':<8} {'mATE(m)':<10} {'mASE':<8} {'mAOE(rad)':<10} {'n_tp'}")
    print("-" * 75)
    for r in rows:
        print(f"{r['anchor']:<12} {r['precision']:<6} {r['ap50']:.4f}   {r['ap70']:.4f}   "
              f"{r['mATE']:.4f}     {r['mASE']:.4f}   {r['mAOE']:.4f}      {r['n_tp']}")

    print(f"\n({len(rows)}/8 configs done)")

    if len(rows) >= 4:
        fp16 = [r for r in rows if r["precision"] == "fp16"]
        fp16.sort(key=lambda x: ["base", "pruned25", "pruned50", "pruned75"].index(x["anchor"])
                   if x["anchor"] in ["base", "pruned25", "pruned50", "pruned75"] else 99)
        if len(fp16) >= 2:
            print("\nFP16 trend (pruning rate →):")
            for col in ["ap70", "mATE", "mASE", "mAOE"]:
                vals = [r[col] for r in fp16]
                diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
                monotone = "↑" if all(d >= 0 for d in diffs) else "↓" if all(d <= 0 for d in diffs) else "~"
                print(f"  {col}: {' → '.join(f'{v:.4f}' for v in vals)} {monotone}")


if __name__ == "__main__":
    main()
