"""Stage1 扫描 CLI — 对模型跑 graph_scan, 产出 partition yaml。

用法 (UniV2X_2.0 env):
  PYTHONPATH=/home/jichengzhi/V2X \
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
      -m framework.stage1.run_scan --model all --device cpu

  单模型: --model codriving | pyramid_lidar | pyramid_camera | v2xvit
  --hw 指定硬件 capability yaml (默认 configs/hardware/rtx4090.yaml)
  --out-dir 默认 framework/partitions/

诚实纪律: 每模型独立 try; 失败记录原因到 <model>_partition.yaml(scan_status=fail),
不伪装成功。供用户次日晨检查。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

_REPO = Path("/home/jichengzhi/V2X")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from framework.stage1.adapters import REGISTRY, get_adapter
from framework.stage1.hardware_scan import HwCapability


def _write_yaml(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"  [write] {path}")


def run_one(model: str, hw: HwCapability, device: str, out_dir: Path,
            profile_latency: str = "auto", lat_warmup: int = 30,
            lat_measure: int = 100) -> str:
    from framework.stage1 import graph_scan
    out_path = out_dir / f"{model}_partition.yaml"
    try:
        adapter = get_adapter(model)
        manifest = graph_scan.scan(adapter, hw, device=device,
                                   profile_latency_mode=profile_latency,
                                   lat_warmup=lat_warmup, lat_measure=lat_measure)
        _write_yaml(manifest, out_path)
        return manifest.get("scan_status", "unknown")
    except Exception as e:  # noqa: BLE001 — 失败如实落盘, 不抛断整批
        tb = traceback.format_exc()
        print(f"  [FAIL] {model}: {type(e).__name__}: {e}")
        fail = {
            "stage": "stage1_partition",
            "model": model,
            "scan_status": "fail",
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb.splitlines()[-15:],
            "hw_capability": hw.summary(),
        }
        _write_yaml(fail, out_path)
        return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    help="all | " + " | ".join(REGISTRY))
    ap.add_argument("--hw", default=str(_REPO / "configs/hardware/rtx4090.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(_REPO / "framework/partitions"))
    ap.add_argument("--profile-latency", default="auto", choices=["auto", "on", "off"],
                    help="S5b 逐层时延: auto=仅cuda开 | on=总开(cpu估算) | off=关")
    ap.add_argument("--lat-warmup", type=int, default=30)
    ap.add_argument("--lat-measure", type=int, default=100)
    args = ap.parse_args()

    hw = HwCapability.from_yaml(args.hw)
    out_dir = Path(args.out_dir)
    models = list(REGISTRY) if args.model == "all" else [args.model]

    print("=" * 78)
    print(f"Stage1 scan — models={models}  hw={hw.name}  device={args.device}")
    print(f"  hw schema_validated={hw.validated}  has_dla={hw.has_dla}  "
          f"int8_align={hw.int8_align}  legal_bits={hw.legal_bits}")
    print("=" * 78)

    print(f"  profile_latency={args.profile_latency}  "
          f"warmup={args.lat_warmup}  measure={args.lat_measure}")

    results = {}
    for m in models:
        print(f"\n--- {m} ---")
        results[m] = run_one(m, hw, args.device, out_dir,
                             profile_latency=args.profile_latency,
                             lat_warmup=args.lat_warmup, lat_measure=args.lat_measure)

    print("\n" + "=" * 78)
    print("Stage1 scan 汇总:")
    for m, st in results.items():
        print(f"  {m:16s} : {st}")
    print("=" * 78)


if __name__ == "__main__":
    main()
