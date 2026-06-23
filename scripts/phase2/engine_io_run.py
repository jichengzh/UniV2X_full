"""Portable TRT engine runner (TRT10 4090 + TRT8.5 Orin) for ISS-017 output match.

Loads cached collab2 input frames (spatial_features + t_ego), runs the engine,
saves cls/reg/dir outputs. Same script runs on both platforms; output .npz are
compared offline (raw cosine/L2 + decoded box sets) to test cross-platform INT8
numerical equivalence (ap_valid gate, basis=orin_int8_output_match_4090).

Usage:
  python engine_io_run.py --engine X.engine --inputs inputs_8.npy \
         --tego tego_8.npy --out outputs_PLATFORM.npz
"""
import argparse
import numpy as np
import tensorrt as trt
import torch

LOG = trt.Logger(trt.Logger.ERROR)


def load_engine(path):
    with open(path, "rb") as f:
        return trt.Runtime(LOG).deserialize_cuda_engine(f.read())


def run(engine, spatial, tego):
    """spatial: (2,64,128,256), tego: (2,2,3). Returns dict name->np array."""
    ctx = engine.create_execution_context()
    # name -> (is_input, shape)
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    dev = {}
    out_names = []
    for nm in names:
        is_in = engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
        if is_in:
            if "spatial" in nm or (len(engine.get_tensor_shape(nm)) == 4):
                arr = spatial.astype(np.float32)
            else:
                arr = tego.astype(np.float32)
            ctx.set_input_shape(nm, arr.shape)
            t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
            dev[nm] = t
            ctx.set_tensor_address(nm, int(t.data_ptr()))
        else:
            out_names.append(nm)
    for nm in out_names:
        shp = tuple(ctx.get_tensor_shape(nm))
        t = torch.empty(shp, dtype=torch.float32, device="cuda")
        dev[nm] = t
        ctx.set_tensor_address(nm, int(t.data_ptr()))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    return {nm: dev[nm].detach().cpu().numpy() for nm in out_names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--inputs", required=True)   # (N,2,64,128,256)
    ap.add_argument("--tego", required=True)      # (N,2,2,3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spatial = np.load(args.inputs)
    tego = np.load(args.tego)
    eng = load_engine(args.engine)
    N = spatial.shape[0]
    acc = {}
    for i in range(N):
        o = run(eng, spatial[i], tego[i])
        for k, v in o.items():
            acc.setdefault(k, []).append(v)
    save = {k: np.stack(v) for k, v in acc.items()}
    np.savez(args.out, **save)
    print(f"[io_run] {N} frames -> {args.out}  outputs: "
          f"{ {k: save[k].shape for k in save} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
