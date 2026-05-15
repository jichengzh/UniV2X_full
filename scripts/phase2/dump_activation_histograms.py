"""Compare activation distributions: official ckpts vs self-trained ckpts.

For each of 8 ckpts (4 official: T1/T2/T4/T6, 4 self-trained: T3/T5/T7/T8):
  1. Load HEAL Pyramid model
  2. Hook key Conv layers (cls_head, reg_head, dir_head, shrink_conv,
     pyramid_backbone.single_head_0/1/2)
  3. Forward 30 DAIR test samples
  4. Aggregate per-layer activation statistics:
     mean, std, p50, p99, p99.9, max, max/p99 (long-tail ratio), max/p50 (DR)

Output: results/activation_histograms.json + summary print
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/phase2"))
os.chdir(str(HEAL_ROOT))

# Patch missing label files for DAIR loader
import opencood.utils.common_utils as _cu
_orig_read_json = _cu.read_json


def _safe_read_json(p):
    if not os.path.exists(p) and ("backup" in str(p) or "label" in str(p).split("/")[-2:]):
        return []
    return _orig_read_json(p)


_cu.read_json = _safe_read_json
import opencood.data_utils.datasets.basedataset.dairv2x_basedataset as _dair_mod
_dair_mod.read_json = _safe_read_json

from opencood.hypes_yaml import yaml_utils
from opencood.utils.common_utils import update_dict
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402

DAIR_ROOT = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

# (tag, ckpt_path, hypes_path, source)
CKPTS = [
    ("T1_base",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml",
     "official"),
    ("T2_p25",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/config.yaml",
     "official"),
    ("T4_p50",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/config.yaml",
     "official"),
    ("T6_p75",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/config.yaml",
     "official"),
    ("T3_p37",
     str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/config.yaml"),
     "self-train"),
    ("T5_p62",
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/config.yaml"),
     "self-train"),
    ("T7_wide_shallow",
     str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/config.yaml"),
     "self-train"),
    ("T8_narrow_deep",
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/config.yaml"),
     "self-train"),
]

# Layers to hook (by attribute path from model root)
HOOK_LAYERS = [
    "cls_head",
    "reg_head",
    "dir_head",
    "shrink_conv.0" if False else "shrink_conv",  # nn.Sequential — hook the whole module's output
    "pyramid_backbone.single_head_0",
    "pyramid_backbone.single_head_1",
    "pyramid_backbone.single_head_2",
]


def get_module(model, attr_path):
    obj = model
    for p in attr_path.split("."):
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def collect_activations(model, loader, n_samples=30):
    """Forward n_samples through model, collect activation stats per hooked layer."""
    device = next(model.parameters()).device
    # Bucket of all activation values per layer (flat numpy)
    buckets_in: dict[str, list] = defaultdict(list)
    buckets_out: dict[str, list] = defaultdict(list)

    handles = []
    for path in HOOK_LAYERS:
        try:
            m = get_module(model, path)
        except (AttributeError, KeyError):
            print(f"  [warn] no such module: {path}")
            continue

        def make_hook(p):
            def h(mod, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    buckets_in[p].append(inp[0].detach().float().abs().flatten().cpu().numpy())
                if isinstance(out, torch.Tensor):
                    buckets_out[p].append(out.detach().float().abs().flatten().cpu().numpy())
            return h
        handles.append(m.register_forward_hook(make_hook(path)))

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= n_samples:
                break
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            try:
                _ = model(batch["ego"])
            except Exception as e:
                print(f"  [skip] forward failed: {e}")
                continue
            n_done += 1
            if n_done % 10 == 0:
                print(f"    sample {n_done}/{n_samples}", flush=True)

    for h in handles:
        h.remove()

    # Aggregate stats per layer
    stats = {}
    for layer in set(list(buckets_in.keys()) + list(buckets_out.keys())):
        s = {}
        for tag, bucket in [("in", buckets_in.get(layer, [])),
                            ("out", buckets_out.get(layer, []))]:
            if not bucket:
                continue
            # Subsample to avoid OOM (heads output is huge tensor)
            arrs = []
            for a in bucket:
                if a.size > 50000:
                    idx = np.random.choice(a.size, 50000, replace=False)
                    arrs.append(a[idx])
                else:
                    arrs.append(a)
            allv = np.concatenate(arrs)
            s[tag] = {
                "n": int(allv.size),
                "mean": float(allv.mean()),
                "std": float(allv.std()),
                "p50": float(np.percentile(allv, 50)),
                "p90": float(np.percentile(allv, 90)),
                "p99": float(np.percentile(allv, 99)),
                "p99_9": float(np.percentile(allv, 99.9)),
                "max": float(allv.max()),
                "long_tail_ratio": float(allv.max() / max(np.percentile(allv, 99), 1e-9)),
                "dynamic_range": float(allv.max() / max(np.percentile(allv, 50), 1e-9)),
                "outlier_mass_6sigma": float((allv > 6 * allv.std()).mean()),
            }
        stats[layer] = s
    return stats, n_done


def load_dair_loader(ckpt_dir, hypes_path, batch_size=1):
    class _Opt:
        model_dir = ckpt_dir

    hypes = yaml_utils.load_yaml(None, _Opt())
    new_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_range,
        "lidar_range": new_range,
        "gt_range": new_range,
    })
    val_split = f"{DAIR_ROOT}/val.json"
    hypes["data_dir"] = DAIR_ROOT
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    import importlib
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"),
                          hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["data_dir"] = DAIR_ROOT
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2,
                        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False)
    return loader, hypes


def main():
    out_path = REPO_ROOT / "results/activation_histograms.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tag, ckpt, hypes_path, source in CKPTS:
        print(f"\n========= {tag} ({source}) =========")
        print(f"  ckpt: {ckpt}")
        try:
            model = build_pyramid_from_ckpt(hypes_path, ckpt, device="cuda")
            model.eval()
            ckpt_dir = str(Path(ckpt).parent)
            loader, hypes = load_dair_loader(ckpt_dir, hypes_path, batch_size=1)
            stats, n_done = collect_activations(model, loader, n_samples=30)
            all_results[tag] = {
                "source": source, "ckpt": ckpt, "n_samples": n_done,
                "layers": stats,
            }
            print(f"  collected {n_done} samples, {len(stats)} layers")
            # Quick print: cls_head output stats
            if "cls_head" in stats and "out" in stats["cls_head"]:
                cs = stats["cls_head"]["out"]
                print(f"  cls_head OUT: max={cs['max']:.3f}, p99={cs['p99']:.3f}, "
                      f"long_tail_ratio={cs['long_tail_ratio']:.2f}, "
                      f"DR={cs['dynamic_range']:.1f}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [err] {tag} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[tag] = {"source": source, "error": str(e)}

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}")

    # ─── Final comparison table ──────────────────────────────────────────
    print("\n\n========= Summary: cls_head OUTPUT activation =========")
    print(f"{'tag':18s} {'src':12s} {'max':>8s} {'p99':>8s} {'p99.9':>8s} "
          f"{'tail_ratio':>12s} {'DR':>8s} {'outlier_6σ%':>12s}")
    for tag, _, _, source in CKPTS:
        if tag not in all_results or "layers" not in all_results[tag]:
            continue
        layers = all_results[tag]["layers"]
        if "cls_head" not in layers or "out" not in layers["cls_head"]:
            continue
        s = layers["cls_head"]["out"]
        print(f"{tag:18s} {source:12s} {s['max']:8.3f} {s['p99']:8.3f} "
              f"{s['p99_9']:8.3f} {s['long_tail_ratio']:12.2f} {s['dynamic_range']:8.1f} "
              f"{s['outlier_mass_6sigma']*100:12.3f}")

    print("\n========= Summary: shrink_conv OUTPUT (= cls_head INPUT) =========")
    print(f"{'tag':18s} {'src':12s} {'max':>8s} {'p99':>8s} {'p99.9':>8s} "
          f"{'tail_ratio':>12s} {'DR':>8s}")
    for tag, _, _, source in CKPTS:
        if tag not in all_results or "layers" not in all_results[tag]:
            continue
        layers = all_results[tag]["layers"]
        if "shrink_conv" not in layers or "out" not in layers["shrink_conv"]:
            continue
        s = layers["shrink_conv"]["out"]
        print(f"{tag:18s} {source:12s} {s['max']:8.3f} {s['p99']:8.3f} "
              f"{s['p99_9']:8.3f} {s['long_tail_ratio']:12.2f} {s['dynamic_range']:8.1f}")


if __name__ == "__main__":
    main()
