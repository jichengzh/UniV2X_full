"""Variant of dump_activation_histograms.py — point to net_epoch77.pth.

Goal: verify whether extended fine-tune (epoch 47 → 77) narrowed the
shrink_conv / cls_head activation distributions on the 4 self-trained ckpts.
If p99 drops from 2.48-3.53 toward official 1.16-1.41 range, Method 3 worked.

Reuses dump_activation_histograms but swaps the 4 self-trained ckpt paths
from net_epoch_bestval_at33.pth -> net_epoch77.pth, and the official ones
are unchanged.

Output: results/activation_histograms_epoch77.json
"""
from __future__ import annotations
from pathlib import Path

import dump_activation_histograms as base

REPO_ROOT = Path("/home/jichengzhi/UniV2X")

# Swap 4 self-trained tags to epoch77 ckpts
SELF_DIRS = {
    "T3_p37": "ft_040_080_160",
    "T5_p62": "ft_024_056_128",
    "T7_wide_shallow": "ft_048_064_128",
    "T8_narrow_deep": "ft_024_048_192",
}

new_ckpts = []
for tag, ckpt, hypes, source in base.CKPTS:
    if tag in SELF_DIRS:
        d = SELF_DIRS[tag]
        new_ckpt = str(REPO_ROOT / f"models/dataset_a_cache/{d}/net_epoch77.pth")
        new_ckpts.append((tag, new_ckpt, hypes, source))
    else:
        new_ckpts.append((tag, ckpt, hypes, source))
base.CKPTS = new_ckpts

# Override output path
_orig_main = base.main


def main():
    out = REPO_ROOT / "results/activation_histograms_epoch77.json"
    # Monkey-patch the out_path inside _orig_main by writing to a wrapper
    import json
    all_results = {}
    for tag, ckpt, hypes_path, source in base.CKPTS:
        print(f"\n========= {tag} ({source}) =========")
        print(f"  ckpt: {ckpt}")
        try:
            model = base.build_pyramid_from_ckpt(hypes_path, ckpt, device="cuda")
            model.eval()
            ckpt_dir = str(Path(ckpt).parent)
            loader, _ = base.load_dair_loader(ckpt_dir, hypes_path, batch_size=1)
            stats, n_done = base.collect_activations(model, loader, n_samples=30)
            all_results[tag] = {
                "source": source, "ckpt": ckpt, "n_samples": n_done,
                "layers": stats,
            }
            print(f"  collected {n_done} samples, {len(stats)} layers")
            if "cls_head" in stats and "out" in stats["cls_head"]:
                cs = stats["cls_head"]["out"]
                print(f"  cls_head OUT: max={cs['max']:.3f}, p99={cs['p99']:.3f}, "
                      f"tail_ratio={cs['long_tail_ratio']:.2f}, DR={cs['dynamic_range']:.1f}")
            if "shrink_conv" in stats and "out" in stats["shrink_conv"]:
                ss = stats["shrink_conv"]["out"]
                print(f"  shrink_conv OUT: max={ss['max']:.3f}, p99={ss['p99']:.3f}, "
                      f"tail_ratio={ss['long_tail_ratio']:.2f}")
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [err] {tag}: {e}")
            import traceback; traceback.print_exc()
            all_results[tag] = {"source": source, "error": str(e)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")

    print("\n\n========= Summary: shrink_conv OUTPUT (= cls_head INPUT) =========")
    print(f"{'tag':18s} {'src':12s} {'max':>8s} {'p99':>8s} {'p99.9':>8s} "
          f"{'tail_ratio':>12s} {'DR':>8s}")
    for tag, _, _, source in base.CKPTS:
        if tag not in all_results or "layers" not in all_results[tag]:
            continue
        layers = all_results[tag]["layers"]
        if "shrink_conv" not in layers or "out" not in layers["shrink_conv"]:
            continue
        s = layers["shrink_conv"]["out"]
        print(f"{tag:18s} {source:12s} {s['max']:8.3f} {s['p99']:8.3f} "
              f"{s['p99_9']:8.3f} {s['long_tail_ratio']:12.2f} {s['dynamic_range']:8.1f}")

    print("\n========= Summary: cls_head OUTPUT =========")
    print(f"{'tag':18s} {'src':12s} {'max':>8s} {'p99':>8s} {'tail_ratio':>12s}")
    for tag, _, _, source in base.CKPTS:
        if tag not in all_results or "layers" not in all_results[tag]:
            continue
        layers = all_results[tag]["layers"]
        if "cls_head" not in layers or "out" not in layers["cls_head"]:
            continue
        s = layers["cls_head"]["out"]
        print(f"{tag:18s} {source:12s} {s['max']:8.3f} {s['p99']:8.3f} "
              f"{s['long_tail_ratio']:12.2f}")


if __name__ == "__main__":
    main()
