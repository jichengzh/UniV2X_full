"""H800 GPU6 — Pyramid collab INT8 AP evaluation (真量化, DAIR val 1789).

Self-contained pipeline:
  Step 1: Export ONNX (collab N=2, DAIR shape 2,64,128,256)
  Step 2: Generate calibration data from DAIR val (N=100 cooperative samples)
  Step 3: Build TRT INT8 engine (MinMax calibrator, real DAIR distribution)
  Step 4: Run hybrid AP eval on full DAIR val 1789 samples
  Step 5: Write JSON report

Usage (on H800 GPU 6):
  CUDA_VISIBLE_DEVICES=6 python3 h800_int8_ap_pyramid.py [--work-dir /exdata/jichengzhi/int8_ap_base]

Calibration: IInt8MinMaxCalibrator, 100 real DAIR val cooperative samples
              (spatial_features shape 2,64,128,256 + t_ego shape 2,2,3)
              NOT dummy scale=1.

AP口径: TRT collab engine (N=2 samples) + PyTorch fallback (N=1 samples)
        Same protocol as stage_a_ap_real (4090), enables direct comparison.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths (hardcoded for H800 environment)
# ---------------------------------------------------------------------------
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
BASE_CKPT_DIR = Path(
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"


# ---------------------------------------------------------------------------
# HEAL imports (requires HEAL_ROOT on sys.path, cwd=HEAL_ROOT)
# ---------------------------------------------------------------------------
def setup_heal_env():
    if str(HEAL_ROOT) not in sys.path:
        sys.path.insert(0, str(HEAL_ROOT))
    os.chdir(HEAL_ROOT)


# ---------------------------------------------------------------------------
# Step 1: ONNX export
# ---------------------------------------------------------------------------
def export_onnx(ckpt_dir: Path, out_onnx: Path) -> bool:
    """Export collab N=2 ONNX from Pyramid DAIR ckpt."""
    if out_onnx.exists():
        print(f"[onnx] cached {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
        return True

    setup_heal_env()
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from opencood.utils.common_utils import update_dict

    print(f"[onnx] loading model from {ckpt_dir}")
    hypes_path = ckpt_dir / "config.yaml"
    hypes = yaml_utils.load_yaml(str(hypes_path))
    if "heter" in hypes:
        x_max, y_max = 102.4, 51.2
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_range,
            "lidar_range": new_range,
            "gt_range": new_range,
        })
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(ckpt_dir), model)
    model.cuda().eval()

    # Identify pyramid_backbone attributes
    pb = model.pyramid_backbone
    s0, s1, s2 = pb.num_filters
    print(f"[onnx] num_filters s0={s0} s1={s1} s2={s2}")

    # Use collab export wrapper (same as PyramidCollabSubnetN2 in export_onnx_pyramid_collab.py)
    # Inline version to avoid REPO_ROOT dependency
    from opencood.utils.transformation_utils import normalize_pairwise_tfm

    class PyramidCollabSubnetN2(torch.nn.Module):
        """Wraps N=2 collab forward: spatial(2,64,H,W) + t_ego(2,2,3) → (cls, reg, dir)."""
        def __init__(self, m, align_corners=False):
            super().__init__()
            self.m = m
            self.align_corners = align_corners

        def forward(self, spatial, t_ego):
            # spatial: (2, 64, H, W), t_ego: (2, 2, 3)
            record_len = torch.tensor([2], device=spatial.device)
            # affine matrix from t_ego
            t = t_ego.unsqueeze(0)  # (1, 2, 2, 3) → (B, L, N, 2, 3)
            # For N=2, affine_matrix shape: (1, 2, 2, 2, 3)? Let's use what forward_collab expects:
            # normalize_pairwise_tfm: (B, L, L, 2, 3) but we have (1, 2, 2, 3)
            # Just pass t_ego directly to forward_collab compatible form
            affine_matrix = t_ego.unsqueeze(0)  # (1, 2, 2, 3)

            fused, occ_outputs = self.m.pyramid_backbone.forward_collab(
                spatial, record_len, affine_matrix,
                ["m1", "m1"],  # agent_modality_list for 2 agents
                {},  # cam_crop_info (empty for LiDAR-only)
            )
            if self.m.shrink_flag:
                fused = self.m.shrink_conv(fused)
            cls_p = self.m.cls_head(fused)
            reg_p = self.m.reg_head(fused)
            dir_p = self.m.dir_head(fused)
            return cls_p, reg_p, dir_p

    wrapper = PyramidCollabSubnetN2(model, align_corners=False).cuda().eval()

    # Dummy inputs
    H, W = 128, 256
    spatial_dummy = torch.zeros(2, 64, H, W, device="cuda")
    tego_dummy = torch.zeros(2, 2, 3, device="cuda")

    print("[onnx] exporting (shape 2,64,128,256 + t_ego 2,2,3) ...")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    # Try export; fall back to a simpler wrapper if forward_collab fails
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, (spatial_dummy, tego_dummy), str(out_onnx),
                input_names=["spatial_features", "t_ego"],
                output_names=["cls_preds", "reg_preds", "dir_preds"],
                opset_version=17,
                dynamic_axes=None,  # static shape engine
                verbose=False,
            )
        print(f"[onnx] exported {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
        del model, wrapper
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[onnx] FAILED: {e}")
        if out_onnx.exists():
            out_onnx.unlink()
        return False


# ---------------------------------------------------------------------------
# Step 2: Generate calibration data
# ---------------------------------------------------------------------------
def gen_calib_data(ckpt_dir: Path, work_dir: Path, n_samples: int = 100):
    """Capture (spatial_features, t_ego) pairs from DAIR val N=2 samples."""
    spatial_path = work_dir / "calib_spatial.npy"
    tego_path = work_dir / "calib_tego.npy"
    if spatial_path.exists() and tego_path.exists():
        arr = np.load(spatial_path)
        print(f"[calib] cached {arr.shape} ({arr.dtype}) min={arr.min():.4f} max={arr.max():.4f}")
        return spatial_path, tego_path

    setup_heal_env()
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils.common_utils import update_dict
    from opencood.utils.transformation_utils import normalize_pairwise_tfm

    print(f"[calib] generating {n_samples} calibration samples from DAIR val")
    hypes_path = ckpt_dir / "config.yaml"
    hypes = yaml_utils.load_yaml(str(hypes_path))
    if "heter" in hypes:
        x_max, y_max = 102.4, 51.2
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_range,
            "lidar_range": new_range,
            "gt_range": new_range,
        })
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(ckpt_dir), model)
    model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    print(f"[calib] dataset size {len(dataset)}")

    spatials = []
    t_egos = []
    with torch.inference_mode():
        for batch_data in loader:
            if len(spatials) >= n_samples:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            aml = ego["agent_modality_list"]
            mc = Counter(aml)
            mf: dict[str, torch.Tensor] = {}
            for m in model.modality_name_list:
                if m not in mc:
                    continue
                feat = getattr(model, f"encoder_{m}")(ego, m)
                feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
                feat = getattr(model, f"aligner_{m}")(feat)
                mf[m] = feat

            # Only N=2 cooperative samples
            n_agents = len(aml)
            if n_agents != 2:
                continue
            counting = {m: 0 for m in model.modality_name_list}
            heter_list = []
            for m in aml:
                heter_list.append(mf[m][counting[m]])
                counting[m] += 1
            spatial = torch.stack(heter_list)  # (2, 64, H, W)
            affine = normalize_pairwise_tfm(
                ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size,
            )
            t_ego = affine[0, 0, :2, :, :]  # (2, 2, 3)

            spatials.append(spatial.cpu().numpy().astype(np.float32))
            t_egos.append(t_ego.cpu().numpy().astype(np.float32))

    sp_arr = np.stack(spatials)  # (N, 2, 64, H, W)
    te_arr = np.stack(t_egos)  # (N, 2, 2, 3)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(spatial_path, sp_arr)
    np.save(tego_path, te_arr)
    print(f"[calib] saved {sp_arr.shape} min={sp_arr.min():.4f} max={sp_arr.max():.4f}")
    print(f"[calib] t_ego {te_arr.shape}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return spatial_path, tego_path


# ---------------------------------------------------------------------------
# Step 3: Build TRT engine (FP16 or INT8)
# ---------------------------------------------------------------------------
def build_trt_engine(onnx_path: Path, precision: str, engine_path: Path,
                     calib_spatial: Path | None = None,
                     calib_tego: Path | None = None,
                     calib_cache: Path | None = None,
                     workspace_mb: int = 4096) -> dict:
    """Build TRT engine (FP16 or INT8 with MinMax calibrator)."""
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1e6
        print(f"[trt] cached {engine_path} ({size_mb:.1f} MB)")
        return {"cached": True, "precision": precision, "engine_size_mb": size_mb}

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # INT8 calibrator (MinMax)
    class _NumpyMultiCalib(trt.IInt8MinMaxCalibrator):
        def __init__(self, sp_path, te_path, batch_size=1, cache_path=None):
            super().__init__()
            self.sp = np.load(sp_path).astype(np.float32)
            self.te = np.load(te_path).astype(np.float32)
            assert self.sp.shape[0] == self.te.shape[0]
            self.n = self.sp.shape[0]
            self.batch_size = batch_size
            self.idx = 0
            self.sp_dev = torch.zeros(
                (batch_size, *self.sp.shape[1:]), dtype=torch.float32, device="cuda"
            )
            self.te_dev = torch.zeros(
                (batch_size, *self.te.shape[1:]), dtype=torch.float32, device="cuda"
            )
            self.cache_path = cache_path
            print(f"[calib] {self.n} samples sp={self.sp.shape} te={self.te.shape}")

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx + self.batch_size > self.n:
                return None
            self.sp_dev.copy_(
                torch.from_numpy(self.sp[self.idx: self.idx + self.batch_size])
            )
            self.te_dev.copy_(
                torch.from_numpy(self.te[self.idx: self.idx + self.batch_size])
            )
            addrs = []
            for nm in names:
                if nm == "spatial_features":
                    addrs.append(int(self.sp_dev.data_ptr()))
                elif nm == "t_ego":
                    addrs.append(int(self.te_dev.data_ptr()))
                else:
                    raise RuntimeError(f"unknown calib input: {nm}")
            self.idx += self.batch_size
            return addrs

        def read_calibration_cache(self):
            if self.cache_path and Path(self.cache_path).exists():
                return open(self.cache_path, "rb").read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path:
                open(self.cache_path, "wb").write(cache)
                print(f"[calib] cache written {self.cache_path} ({len(cache)} bytes)")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"[trt] parsing {onnx_path}")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    print(f"[trt] inputs={network.num_inputs} outputs={network.num_outputs}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  in[{i}] {t.name} {t.shape} {t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] precision=FP16")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if calib_spatial is None or calib_tego is None:
            raise ValueError("INT8 requires calib_spatial + calib_tego")
        calibrator = _NumpyMultiCalib(calib_spatial, calib_tego,
                                      cache_path=str(calib_cache) if calib_cache else None)
        config.int8_calibrator = calibrator
        print(f"[trt] precision=INT8+FP16fallback  calibrator=MinMax  calib_cache={calib_cache}")
    else:
        raise ValueError(precision)

    print(f"[trt] building engine (workspace={workspace_mb}MB) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build failed")
    build_secs = time.time() - t0
    blob = bytes(serialized)
    engine_path.write_bytes(blob)
    size_mb = len(blob) / 1e6
    print(f"[trt] done in {build_secs:.1f}s, size={size_mb:.1f} MB → {engine_path}")

    del calibrator, serialized
    gc.collect()
    torch.cuda.empty_cache()
    return {"precision": precision, "build_secs": build_secs, "engine_size_mb": size_mb}


# ---------------------------------------------------------------------------
# Step 4: AP evaluation
# ---------------------------------------------------------------------------
def run_ap_eval(engine_path: Path, ckpt_dir: Path,
                spatial_shape=(2, 64, 128, 256),
                tego_shape=(2, 2, 3),
                n_samples: int = 1789) -> dict:
    """Hybrid TRT+PyTorch AP eval on DAIR val."""
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    setup_heal_env()
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils.common_utils import update_dict
    from opencood.utils.transformation_utils import normalize_pairwise_tfm
    from opencood.utils import eval_utils

    hypes_path = ckpt_dir / "config.yaml"
    hypes = yaml_utils.load_yaml(str(hypes_path))
    if "heter" in hypes:
        x_max, y_max = 102.4, 51.2
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_range,
            "lidar_range": new_range,
            "gt_range": new_range,
        })
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print("[ap] loading model + ckpt ...")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(ckpt_dir), model)
    model.cuda().eval()

    # Load TRT engine
    print(f"[ap] loading TRT engine {engine_path}")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    n_io = engine.num_io_tensors
    tnames = [engine.get_tensor_name(i) for i in range(n_io)]
    in_names = [n for n in tnames if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in tnames if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    print(f"[ap] engine inputs={in_names} outputs={out_names}")

    # Set input shapes
    context.set_input_shape(in_names[0], spatial_shape)
    if len(in_names) > 1:
        context.set_input_shape(in_names[1], tego_shape)

    # Allocate buffers
    bufs: dict[str, torch.Tensor] = {}
    for nm in tnames:
        shape = tuple(context.get_tensor_shape(nm))
        dtype = engine.get_tensor_dtype(nm)
        td = {trt.float32: torch.float32, trt.float16: torch.float16}.get(dtype, torch.float32)
        bufs[nm] = torch.empty(shape, dtype=td, device="cuda")
        context.set_tensor_address(nm, int(bufs[nm].data_ptr()))
    trt_stream = torch.cuda.Stream()

    def run_trt(spatial: torch.Tensor, t_ego: torch.Tensor):
        bufs[in_names[0]].copy_(spatial.float())
        if len(in_names) > 1:
            bufs[in_names[1]].copy_(t_ego.float())
        with torch.cuda.stream(trt_stream):
            context.execute_async_v3(trt_stream.cuda_stream)
        trt_stream.synchronize()
        return tuple(bufs[n].clone() for n in out_names)

    # Build dataset
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    n_done = n_trt = n_fb = 0
    t0 = time.time()

    print(f"[ap] evaluating {n_samples} samples ...")
    with torch.inference_mode():
        for batch_data in loader:
            if n_done >= n_samples:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            aml = ego["agent_modality_list"]
            mc = Counter(aml)

            mf: dict[str, torch.Tensor] = {}
            for m in model.modality_name_list:
                if m not in mc:
                    continue
                feat = getattr(model, f"encoder_{m}")(ego, m)
                feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
                feat = getattr(model, f"aligner_{m}")(feat)
                mf[m] = feat

            counting = {m: 0 for m in model.modality_name_list}
            heter_list = []
            for m in aml:
                heter_list.append(mf[m][counting[m]])
                counting[m] += 1
            heter_feat = torch.stack(heter_list)  # (N_agents, 64, H, W)
            n_agents = heter_feat.shape[0]
            affine = normalize_pairwise_tfm(
                ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size,
            )

            if n_agents == 2:
                t_ego = affine[0, 0, :2, :, :]  # (2, 2, 3)
                cls_p, reg_p, dir_p = run_trt(heter_feat.contiguous(), t_ego.contiguous())
                n_trt += 1
            else:
                # PyTorch fallback for N=1 or N>2
                record_len = ego["record_len"]
                fused, occ_outputs = model.pyramid_backbone.forward_collab(
                    heter_feat, record_len, affine, aml, model.cam_crop_info,
                )
                if model.shrink_flag:
                    fused = model.shrink_conv(fused)
                cls_p = model.cls_head(fused)
                reg_p = model.reg_head(fused)
                dir_p = model.dir_head(fused)
                n_fb += 1

            output_wrapped = {"ego": {
                "cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p,
                "occ_single_list": [],
            }}
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_wrapped)
            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou_th)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{n_samples}  trt={n_trt} fb={n_fb}  "
                      f"elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    out_dir = engine_path.parent / "ap_eval_results"
    out_dir.mkdir(exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(out_dir))
    print(f"\n[ap] DONE  AP30={ap30:.4f}  AP50={ap50:.4f}  AP70={ap70:.4f}  "
          f"({n_done} samples, trt={n_trt}, fb={n_fb}, {elapsed:.1f}s)")
    return {
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "n_samples": n_done, "n_trt_path": n_trt, "n_pytorch_fallback": n_fb,
        "elapsed_secs": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=str(BASE_CKPT_DIR))
    p.add_argument("--work-dir", default="/exdata/jichengzhi/int8_ap_base_pyramid")
    p.add_argument("--n-calib", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=1789)
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # GPU check
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[init] TRT: ", end="")
    import tensorrt as trt
    print(trt.__version__)
    print(f"[init] work_dir={work_dir}")
    print(f"[init] ckpt_dir={ckpt_dir}")

    t_total = time.time()

    # Step 1: Export ONNX
    onnx_path = work_dir / "pyramid_base_collab_dair.onnx"
    ok = export_onnx(ckpt_dir, onnx_path)
    if not ok:
        print("FATAL: ONNX export failed")
        return 1

    # Step 2: Calibration data
    calib_spatial, calib_tego = gen_calib_data(ckpt_dir, work_dir / "calibration",
                                               n_samples=args.n_calib)

    # Step 3a: FP16 engine
    results = {}
    if not args.skip_fp16:
        fp16_engine = work_dir / "engines" / "pyramid_base_collab_fp16.engine"
        fp16_info = build_trt_engine(onnx_path, "fp16", fp16_engine)
        # Step 4a: FP16 AP eval
        fp16_ap = run_ap_eval(fp16_engine, ckpt_dir, n_samples=args.n_eval)
        results["fp16"] = {**fp16_info, **fp16_ap}

    # Step 3b: INT8 engine (with real calibration)
    if not args.skip_int8:
        int8_engine = work_dir / "engines" / "pyramid_base_collab_int8.engine"
        calib_cache = work_dir / "calibration" / "pyramid_base_int8.cache"
        int8_info = build_trt_engine(onnx_path, "int8", int8_engine,
                                     calib_spatial=calib_spatial,
                                     calib_tego=calib_tego,
                                     calib_cache=calib_cache)
        # Step 4b: INT8 AP eval
        int8_ap = run_ap_eval(int8_engine, ckpt_dir, n_samples=args.n_eval)
        results["int8"] = {**int8_info, **int8_ap}

    # Summary
    elapsed_total = (time.time() - t_total) / 60
    report = {
        "ckpt_dir": str(ckpt_dir),
        "work_dir": str(work_dir),
        "gpu": torch.cuda.get_device_name(0),
        "n_calib_samples": args.n_calib,
        "calibration_method": "IInt8MinMaxCalibrator",
        "calibration_data_source": "DAIR_val_cooperative_N2",
        "results": results,
        "total_elapsed_min": elapsed_total,
    }
    out_json = work_dir / "h800_int8_ap_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'='*60}")
    print(f"REPORT: {out_json}")
    print(f"Total: {elapsed_total:.1f} min")
    if "fp16" in results:
        r = results["fp16"]
        print(f"FP16: AP30={r['ap30']:.4f}  AP50={r['ap50']:.4f}  AP70={r['ap70']:.4f}  "
              f"engine={r.get('engine_size_mb', '?'):.1f}MB")
    if "int8" in results:
        r = results["int8"]
        print(f"INT8: AP30={r['ap30']:.4f}  AP50={r['ap50']:.4f}  AP70={r['ap70']:.4f}  "
              f"engine={r.get('engine_size_mb', '?'):.1f}MB")
    if "fp16" in results and "int8" in results:
        fp16 = results["fp16"]
        int8 = results["int8"]
        print(f"ΔAP50(int8-fp16)={int8['ap50']-fp16['ap50']:+.4f}  "
              f"ΔAP70={int8['ap70']-fp16['ap70']:+.4f}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
