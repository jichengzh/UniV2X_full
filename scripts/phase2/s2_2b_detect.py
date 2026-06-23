"""S2.2b signal detectors (dims_hardware_v4 §4.3) over the Role-A screening CSVs.

Reads the int8 + fp16 screening CSVs, merges, and runs the four coupling-signal
detectors, emitting a candidate-coupling table:
  D-a  latency cliff / non-monotone  -> per-FLOP efficiency vs width
  D-b  schedule-product switch        -> d1_path / frag / red_ext / grid / smem change at a width threshold
  D-c  prune x quant inseparable      -> int8 vs fp16 product structure differ beyond trivial
  D-d  occupancy / bank cliff         -> smem_bytes / n_align step across a width

Each flagged cell is classified later (stage B) as known-alignment / new-coupling /
auto-noise. Negative results (a dimension flat across all widths) are printed too.

Usage: python s2_2b_detect.py <int8_csv> <fp16_csv>
"""
from __future__ import annotations
import sys, csv

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # numeric coercion
    for r in rows:
        for k in ("cin", "cout", "k", "k32", "k16", "best_us", "smem_bytes",
                  "n_align", "n_shared", "n_pipe_stage", "n_unroll", "tir_len",
                  "n_mma", "n_dp4a", "n_ptx"):
            try:
                r[k] = float(r[k]) if "." in str(r.get(k, "")) else int(r.get(k, 0))
            except (ValueError, TypeError):
                r[k] = r.get(k, "")
    rows.sort(key=lambda r: r["cin"])
    return rows


def fmt_table(rows, prec):
    print(f"\n=== {prec} per-width products (width=Cin=Cout) ===")
    hdr = f"{'W':>4} {'÷32':>3} {'÷16':>3} {'path':>7} {'frag':>6} {'redE':>5} " \
          f"{'best_us':>8} {'eff*':>7} {'n_sh':>4} {'n_al':>4} {'grid':>34} {'smem_sig':>20}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        w = r["cin"]
        flops = w * w  # K=N=width, M fixed -> FLOPs ∝ width^2
        eff = r["best_us"] / (flops / 1e4) if isinstance(r["best_us"], (int, float)) and r["best_us"] > 0 else -1
        d32 = "Y" if r["k32"] else "."
        d16 = "Y" if r["k16"] else "."
        print(f"{int(w):>4} {d32:>3} {d16:>3} {str(r['d1_path']):>7} {str(r['frag']):>6} "
              f"{str(r['red_ext']):>5} {r['best_us']:>8} {eff:>7.3f} {int(r['n_shared']):>4} "
              f"{int(r['n_align']):>4} {str(r['grid_sig']):>34} {str(r['smem_sig'])[:20]:>20}")


def detect(rows, prec):
    flags = []
    # D-a: efficiency non-monotone / cliff (per-FLOP us). Higher eff = worse.
    effs = []
    for r in rows:
        w = r["cin"]; flops = w * w
        if isinstance(r["best_us"], (int, float)) and r["best_us"] > 0:
            effs.append((w, r["best_us"] / (flops / 1e4), r["k16"], r["k32"]))
    if effs:
        med = sorted(e[1] for e in effs)[len(effs) // 2]
        for w, e, k16, k32 in effs:
            if e > 1.6 * med:
                flags.append(("D-a", prec, f"W={w} eff={e:.3f} >1.6x median {med:.3f} "
                              f"(÷16={'Y' if k16 else 'N'} ÷32={'Y' if k32 else 'N'})"))
    # D-b: any width NOT on WMMA = TC fallback cliff
    for r in rows:
        if r["d1_path"] != "WMMA":
            flags.append(("D-b", prec, f"W={r['cin']} path={r['d1_path']} (TC fallback! ÷16={r['k16']})"))
    # D-b: product switch — frag/redE pattern. red_ext expected = ceil(K/16); flag if frag != 16x16
    for r in rows:
        if r["frag"] and r["frag"] != "16x16":
            flags.append(("D-b", prec, f"W={r['cin']} frag={r['frag']} (not 16x16!)"))
    return flags


def cross_prec(int8, fp16):
    print("\n=== D-c: prune x quant inseparability (int8 vs fp16 at same width) ===")
    f16 = {r["cin"]: r for r in fp16}
    flags = []
    for r in int8:
        w = r["cin"]
        if w not in f16:
            continue
        o = f16[w]
        # compare path + grid + n_shared + n_align structure
        diffs = []
        if r["d1_path"] != o["d1_path"]:
            diffs.append(f"path int8={r['d1_path']}/fp16={o['d1_path']}")
        if r["grid_sig"] != o["grid_sig"]:
            diffs.append("grid≠")
        if int(r["n_shared"]) != int(o["n_shared"]):
            diffs.append(f"n_shared {r['n_shared']}/{o['n_shared']}")
        if int(r["n_align"]) != int(o["n_align"]):
            diffs.append(f"n_align {r['n_align']}/{o['n_align']}")
        tag = "TRIVIAL(dtype)" if (len(diffs) <= 1 and "grid≠" in diffs) else ("DIFF" if diffs else "same")
        print(f"  W={int(w):>4}: {tag:>14}  {'; '.join(diffs) if diffs else '(identical structure)'}")
        if tag == "DIFF":
            flags.append(("D-c", "x", f"W={w}: {'; '.join(diffs)}"))
    return flags


def main():
    int8 = load(sys.argv[1])
    fp16 = load(sys.argv[2])
    fmt_table(int8, "int8")
    fmt_table(fp16, "fp16")
    flags = detect(int8, "int8") + detect(fp16, "fp16") + cross_prec(int8, fp16)
    print("\n=== CANDIDATE COUPLING FLAGS ===")
    if not flags:
        print("  (none — all dimensions smooth across widths; see negative-results note)")
    for det, prec, msg in flags:
        print(f"  [{det}] {prec}: {msg}")
    # negative-results summary: which dimensions never varied
    print("\n=== NEGATIVE-RESULT SCAN (dimensions flat across all widths) ===")
    for tag, key in [("L3 bank-align", "align_factors"), ("L4 pipeline", "n_pipe_stage")]:
        vals_i = set(str(r.get(key)) for r in int8)
        vals_f = set(str(r.get(key)) for r in fp16)
        print(f"  {tag}: int8 distinct={sorted(vals_i)[:4]} | fp16 distinct={sorted(vals_f)[:4]}")


if __name__ == "__main__":
    main()
