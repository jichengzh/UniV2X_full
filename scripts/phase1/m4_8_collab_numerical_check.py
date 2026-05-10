"""Phase A.5 — Numerical sanity for collab N=2 wrapper vs HEAL forward_collab.

Force same input through both paths and compare outputs.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from tools.export_onnx_pyramid_collab import PyramidCollabSubnetN2  # noqa: E402
import tensorrt as trt  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def trt_run_collab(engine_path: str, spatial: torch.Tensor, t_ego: torch.Tensor):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        eng = runtime.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    in_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
                if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    out_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
                 if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    s_name = next(n for n in in_names if "spatial" in n.lower())
    t_name = next(n for n in in_names if "ego" in n.lower())
    ctx.set_input_shape(s_name, tuple(spatial.shape))
    ctx.set_input_shape(t_name, tuple(t_ego.shape))
    bufs = {s_name: spatial.float().contiguous(), t_name: t_ego.float().contiguous()}
    for n in out_names:
        shape = tuple(ctx.get_tensor_shape(n))
        bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")
    for n in in_names + out_names:
        ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return tuple(bufs[n] for n in out_names)


def main():
    model_dir = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
    hypes = yaml_utils.load_yaml(str(Path(model_dir) / "config.yaml"))
    rng = "102.4,51.2"
    if "heter" in hypes:
        x_max, y_max = [float(x) for x in rng.split(",")]
        new_range = [-x_max, -y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
                     x_max, y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range,
                                    "lidar_range": new_range,
                                    "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print("[1] build model + load")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.cuda().eval()
    wrapper = PyramidCollabSubnetN2(model, align_corners=False).cuda().eval()

    print("[2] build dataset + grab one N=2 sample")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            agent_modality_list = ego["agent_modality_list"]
            modality_count = Counter(agent_modality_list)
            mod_feat = {}
            for m in model.modality_name_list:
                if m not in modality_count: continue
                f = getattr(model, f"encoder_{m}")(ego, m)
                f = getattr(model, f"backbone_{m}")({"spatial_features": f})["spatial_features_2d"]
                f = getattr(model, f"aligner_{m}")(f)
                mod_feat[m] = f
            counting = {m: 0 for m in model.modality_name_list}
            heter = []
            for m in agent_modality_list:
                heter.append(mod_feat[m][counting[m]])
                counting[m] += 1
            heter_2d = torch.stack(heter)  # (N, C, H, W)
            n_agents = heter_2d.shape[0]
            print(f"  batch {batch_idx}: n_agents={n_agents}")
            if n_agents != 2:
                continue
            record_len = ego["record_len"]
            affine = normalize_pairwise_tfm(ego["pairwise_t_matrix"], model.H, model.W,
                                             model.fake_voxel_size)
            t_ego = affine[0, 0, :2, :, :].contiguous()
            print(f"  spatial: {tuple(heter_2d.shape)}  t_ego: {tuple(t_ego.shape)}")

            # Path A: HEAL native forward_collab
            fused_a, _ = model.pyramid_backbone.forward_collab(
                heter_2d, record_len, affine, agent_modality_list, model.cam_crop_info,
            )
            fused_a = model.shrink_conv(fused_a)
            cls_a = model.cls_head(fused_a)
            reg_a = model.reg_head(fused_a)
            dir_a = model.dir_head(fused_a)

            # Path B: my wrapper
            cls_b, reg_b, dir_b = wrapper(heter_2d, t_ego)

            print("\n  === HEAL vs PyTorch wrapper ===")
            for name, A, B in [("cls", cls_a, cls_b), ("reg", reg_a, reg_b), ("dir", dir_a, dir_b)]:
                d = (A - B).abs()
                amax = A.abs().max()
                rel = d.max() / max(amax.item(), 1e-6)
                print(f"  {name}: |A|max={amax:.3f}  max|Δ|={d.max():.4e}  rel_max={rel:.2%}")

            # Path C: TRT engines (FP32 + FP16) — also check on multiple samples
            for tag, eng_path in [
                ("FP32", str(REPO_ROOT / "models/pyramid_dair_m1_collab_n2_fp32.engine")),
                ("FP16", str(REPO_ROOT / "models/pyramid_dair_m1_collab_n2_fp16.engine")),
            ]:
                print(f"\n  === HEAL vs TRT {tag} engine ===")
                cls_t, reg_t, dir_t = trt_run_collab(eng_path, heter_2d, t_ego)
                for name, A, T in [("cls", cls_a, cls_t), ("reg", reg_a, reg_t), ("dir", dir_a, dir_t)]:
                    d = (A - T).abs()
                    amax = A.abs().max()
                    rel = d.max() / max(amax.item(), 1e-6)
                    print(f"  {name}: |A|max={amax:.3f}  max|Δ|={d.max():.4e}  "
                          f"mean|Δ|={d.mean():.4e}  rel_max={rel:.2%}")
            # Sample multiple N=2 batches to look for outlier drift
            seen_n2 = 1
            print("\n  === checking 4 more N=2 samples for drift outliers ===")
            for batch_idx2, b2 in enumerate(loader):
                if seen_n2 >= 5: break
                if batch_idx2 <= batch_idx: continue
                if b2 is None: continue
                b2 = train_utils.to_device(b2, "cuda")
                ego2 = b2["ego"]
                aml2 = ego2["agent_modality_list"]
                mc2 = Counter(aml2)
                mf2 = {}
                for m in model.modality_name_list:
                    if m not in mc2: continue
                    f = getattr(model, f"encoder_{m}")(ego2, m)
                    f = getattr(model, f"backbone_{m}")({"spatial_features": f})["spatial_features_2d"]
                    f = getattr(model, f"aligner_{m}")(f)
                    mf2[m] = f
                cnt2 = {m: 0 for m in model.modality_name_list}
                hl2 = []
                for m in aml2:
                    hl2.append(mf2[m][cnt2[m]])
                    cnt2[m] += 1
                h2 = torch.stack(hl2)
                if h2.shape[0] != 2: continue
                rl2 = ego2["record_len"]
                af2 = normalize_pairwise_tfm(ego2["pairwise_t_matrix"], model.H, model.W,
                                              model.fake_voxel_size)
                te2 = af2[0, 0, :2, :, :].contiguous()
                fa, _ = model.pyramid_backbone.forward_collab(h2, rl2, af2, aml2, model.cam_crop_info)
                fa = model.shrink_conv(fa)
                ca, ra, da = model.cls_head(fa), model.reg_head(fa), model.dir_head(fa)
                ct, rt, dt = trt_run_collab(str(REPO_ROOT / "models/pyramid_dair_m1_collab_n2_fp32.engine"),
                                            h2, te2)
                rel_c = (ca - ct).abs().max() / max(ca.abs().max().item(), 1e-6)
                rel_r = (ra - rt).abs().max() / max(ra.abs().max().item(), 1e-6)
                print(f"  sample {batch_idx2}: cls rel={rel_c:.2%}, reg rel={rel_r:.2%}, "
                      f"|cls|={ca.abs().max():.2f}")
                seen_n2 += 1
            break


if __name__ == "__main__":
    main()
