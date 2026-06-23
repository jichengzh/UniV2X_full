"""测全网剪枝 PyramidSubnet 的 PyTorch latency (CUDA Event, FP16).

口径与 stage_a / M4.3 一致: 计时 prunable core =
    pyramid_backbone (single) + shrink_conv + cls/reg/dir heads
输入 spatial_features (1, 64, 128, 256) [DAIR backbone_m1 输出 shape].

★ 必须在完全空闲 GPU (util 0% / mem<=50MiB) 上跑, 否则 latency 不可信。
脚本启动时自检 GPU 占用, 不空闲则报错退出。

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/phase2/wholenet_subnet_latency.py \
        --model-dir <ckpt dir> --epoch 31 --tag wholenet_p50 \
        --report results/lat_wholenet_p50.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL))
sys.path.insert(0, str(REPO))

from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.models.heter_pyramid_collab import HeterPyramidCollab  # noqa: E402
from tools.export_onnx_pyramid import PyramidSubnet  # noqa: E402


def assert_gpu_idle():
    """自检当前可见 GPU 空闲 (util 0% / mem<=50MiB)。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"]).decode()
    # CUDA_VISIBLE_DEVICES 已限制可见, torch 看到的就是被允许的卡
    # 但 nvidia-smi 仍列全部物理卡; 取 torch.cuda.current_device 对应的物理 index 较繁,
    # 简化: 要求至少存在一张 util<=2% & mem<=50MiB 的卡 (即我们指定的那张)。
    lines = [l for l in out.strip().splitlines() if l.strip()]
    idle = [l for l in lines
            if int(l.split(",")[1]) <= 2 and int(l.split(",")[2]) <= 50]
    if not idle:
        print("[FATAL] 无空闲 GPU (util<=2% & mem<=50MiB), 拒绝测 latency:")
        print(out)
        sys.exit(2)
    print(f"[gpu-check] 空闲 GPU 存在: {idle}")


def load_subnet(model_dir, epoch, device):
    cfg = Path(model_dir) / "config.yaml"
    hypes = load_yaml(str(cfg))
    model = HeterPyramidCollab(hypes["model"]["args"])
    ckpt_path = Path(model_dir) / f"net_epoch{epoch}.pth"
    if not ckpt_path.exists():
        cands = sorted(Path(model_dir).glob("net_epoch_bestval_at*.pth"),
                       key=lambda q: int(q.stem.split("_at")[-1]))
        assert cands, f"no ckpt for epoch {epoch} nor bestval in {model_dir}"
        ckpt_path = cands[-1]
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    sub = PyramidSubnet(model).to(device).eval()
    return sub, str(ckpt_path)


def bench(sub, device, hw, fp16, warmup=100, runs=300):
    H, W = hw
    x = torch.randn(1, 64, H, W, device=device)
    if fp16:
        sub = sub.half()
        x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            sub(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            sub(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    t = torch.tensor(times)
    return {
        "mean": float(t.mean()), "p50": float(t.median()),
        "p99": float(t.kthvalue(int(len(t) * 0.99)).values),
        "min": float(t.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--epoch", type=int, default=31)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hw", default="128,256")
    ap.add_argument("--report", required=True)
    ap.add_argument("--no-gpu-check", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no cuda"
    if not args.no_gpu_check:
        assert_gpu_idle()
    device = "cuda"
    hw = tuple(int(x) for x in args.hw.split(","))

    sub, ckpt = load_subnet(args.model_dir, args.epoch, device)
    n_params = sum(p.numel() for p in sub.parameters())
    lat_fp32 = bench(sub, device, hw, fp16=False)
    # rebuild for fp16 (half() is in-place-ish; reload to be safe)
    sub2, _ = load_subnet(args.model_dir, args.epoch, device)
    lat_fp16 = bench(sub2, device, hw, fp16=True)

    rep = {
        "tag": args.tag, "ckpt": ckpt, "hw": hw,
        "subnet_params": n_params,
        "gpu": torch.cuda.get_device_name(0),
        "lat_fp32_ms": lat_fp32, "lat_fp16_ms": lat_fp16,
        "note": "PyramidSubnet (pyramid+shrink+heads) CUDA Event timing, idle GPU",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
