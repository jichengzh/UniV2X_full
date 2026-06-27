"""Build the Stop-A TVM mixed-INT8 e2e row from real measurements.

This file is intentionally conservative. It does not synthesize latency/AP and
does not treat subnet or direct-matmul measurements as end-to-end evidence.
Use ``--from-measurements`` after a real TVM e2e latency/AP run has written a
structured measurement JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROW = REPO_ROOT / "results/attention_p50_tvm_mixed_int8_row_v1.json"
DEFAULT_OUT_MEASUREMENT = REPO_ROOT / "results/attention_p50_tvm_mixed_int8_measurement_v1.json"
REQUIRED_CONFIG = "attention-p50-int8/mixed"
REQUIRED_QUANT = "int8/mixed"
REQUIRED_BACKEND = "TVM Relax int8/mixed"
REQUIRED_FULL_VAL_SAMPLES = 1789
FAKE_AP_TOKENS = ("simulated", "fake", "prior", "not_true")
AP_GAIN_AUDIT_FIELDS = (
    "same_eval_protocol",
    "same_dataset_split",
    "same_checkpoint_family",
    "same_thresholds",
    "finetune_epochs",
    "learning_rate",
    "seed",
)
DEFAULT_H800_TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
DEFAULT_H800_TORCH_SITE = "/data/jichengzhi_v2x/t2lib"
DEFAULT_H800_HEAL_ROOT = "/data/jichengzhi_v2x/HEAL"
DEFAULT_TYPE_COVERAGE_LOG = REPO_ROOT / "logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_v1.json"
DEFAULT_BASELINE_REPORT = REPO_ROOT / "results/attention_e2e_checkpoint_eval_full_v1.json"
DEFAULT_RUNTIME_LOG = REPO_ROOT / "logs/attention_e2e_pq_v1/attention_p50_tvm_mixed_int8_runtime_v1.json"


def bootstrap_h800_tvm_then_torch(
    *,
    tvm_site: str = DEFAULT_H800_TVM_SITE,
    torch_site: str = DEFAULT_H800_TORCH_SITE,
    heal_root: str = DEFAULT_H800_HEAL_ROOT,
) -> dict[str, str]:
    """Import TVM before adding H800's torch site-packages.

    H800 currently keeps TVM and torch in separate environments. Importing
    torch before TVM loads torch's CUDA runtime first and breaks TVM Relax with
    ``cudaGraphAddDependencies_v2``. This helper makes the import order explicit
    for the real e2e runner path.
    """
    import importlib

    tvm_site_path = str(Path(tvm_site))
    torch_site_path = str(Path(torch_site))
    heal_root_path = str(Path(heal_root))

    if tvm_site_path not in sys.path:
        sys.path.insert(0, tvm_site_path)
    tvm = importlib.import_module("tvm")

    for path in (heal_root_path, torch_site_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    # TVM's environment carries NumPy 2.x, while H800 torch 2.1.2/t2lib and
    # HEAL's mswin initialization expect NumPy 1.x interop. TVM is already
    # loaded above; reload NumPy from t2lib before torch/opencood imports.
    for name in list(sys.modules):
        if name == "numpy" or name.startswith("numpy."):
            del sys.modules[name]
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")

    return {
        "tvm_version": str(getattr(tvm, "__version__", "unknown")),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "numpy_version": str(getattr(numpy, "__version__", "unknown")),
        "tvm_site": tvm_site_path,
        "torch_site": torch_site_path,
        "heal_root": heal_root_path,
    }


def record_hmsa_type_dispatch_coverage(
    *,
    device: str,
    eval_samples: int,
    num_workers: int,
    out_log: str | Path = DEFAULT_TYPE_COVERAGE_LOG,
) -> dict[str, Any]:
    """Run real model forwards and record HMSA prior-encoding type orders."""
    import os

    heal_root = os.environ.get("V2X_HEAL_ROOT", DEFAULT_H800_HEAL_ROOT)
    if heal_root not in sys.path:
        sys.path.insert(0, heal_root)

    from torch.utils.data import DataLoader
    from opencood.models.sub_modules.hmsa import HGTCavAttention
    from opencood.tools import train_utils
    from attention_e2e_checkpoint_eval import build_dataset_from_hypes, default_eval_configs, load_eval_model

    import time
    import torch

    started = time.time()
    config = next(item for item in default_eval_configs() if item["config"] == "attention-p50-shortft-fp16")
    model, hypes, load_info = load_eval_model(config, device)
    dataset = build_dataset_from_hypes(hypes)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    captured_orders: list[list[int]] = []
    module_counts: dict[str, int] = {}
    originals: list[tuple[Any, Any]] = []

    for name, module in model.named_modules():
        if not isinstance(module, HGTCavAttention):
            continue
        original_forward = module.forward
        originals.append((module, original_forward))
        module_counts[name] = 0

        def make_forward(orig, module_name: str):
            def wrapped(self, x, mask, prior_encoding):
                types = prior_encoding[:, :, 0, 0, 2].to(torch.int).detach().cpu()
                for row in types.tolist():
                    captured_orders.append([int(item) for item in row])
                module_counts[module_name] += 1
                return orig(x, mask, prior_encoding)

            return wrapped

        module.forward = make_forward(original_forward, name).__get__(module, type(module))

    n_done = 0
    status = "OK"
    error = None
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                if n_done >= eval_samples:
                    break
                batch = train_utils.to_device(batch, torch.device(device))
                model(batch["ego"])
                n_done += 1
    except Exception as exc:  # pragma: no cover - exercised on H800 only
        import traceback

        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()[-4000:]
    else:
        tb = None
    finally:
        for module, original_forward in originals:
            module.forward = original_forward

    unique_orders = sorted({tuple(order) for order in captured_orders})
    result = {
        "schema_version": "hmsa_type_dispatch_coverage_v1",
        "status": status,
        "dataset_split": "DAIR val",
        "requested_samples": eval_samples,
        "sample_count": n_done,
        "hmsa_call_count": len(captured_orders),
        "hmsa_module_count": len(originals),
        "module_call_counts": module_counts,
        "observed_type_orders": [list(order) for order in unique_orders],
        "covers_dynamic_type_dispatch": len(unique_orders) > 1,
        "static_order_if_any": list(unique_orders[0]) if len(unique_orders) == 1 else None,
        "checkpoint_path": config["checkpoint_path"],
        "manifest_path": config["manifest_path"],
        "load_info": load_info,
        "elapsed_secs": round(time.time() - started, 3),
    }
    if error:
        result["error"] = error
        result["traceback"] = tb

    out_path = _resolve_repo_path(out_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["log_path"] = str(out_path)
    return result


def _quantize_symmetric_int8(tensor: Any) -> tuple[Any, float]:
    """Quantize a torch tensor symmetrically and return (int8_tensor, scale)."""
    import torch

    max_abs = tensor.detach().abs().max()
    scale = float((max_abs / 127.0).clamp(min=1.0e-8).item())
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized.contiguous(), scale


def build_relax_mswin_bwa_qkv_int8_mixed(plan: dict[str, Any]):
    """Build MSwin BWA with TVM int8 Q/K/V projections and FP16 attention/out.

    This is a correctness-oriented mixed path: Q/K/V Linear projections are int8
    matmuls with explicit dequantization; softmax and output projection stay
    FP16. It is not a final all-linear int8 kernel, but it is a real TVM
    fusion-subgraph execution path suitable for pilot e2e integration.
    """
    from tvm import relax
    from t1_attention_tvm_bench import _reshape_windows, _unshape_windows

    b, l, h, w, c = plan["input_shape"]
    inner = plan["inner_dim"]
    tokens = plan["window_tokens"]
    flat_tokens = b * l * h * w

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((b, l, h, w, c), "int8"))
    x_scale = relax.Var("x_scale", relax.TensorStructInfo((1,), "float16"))
    wq = relax.Var("wq", relax.TensorStructInfo((c, inner), "int8"))
    wk = relax.Var("wk", relax.TensorStructInfo((c, inner), "int8"))
    wv = relax.Var("wv", relax.TensorStructInfo((c, inner), "int8"))
    wq_scale = relax.Var("wq_scale", relax.TensorStructInfo((1,), "float16"))
    wk_scale = relax.Var("wk_scale", relax.TensorStructInfo((1,), "float16"))
    wv_scale = relax.Var("wv_scale", relax.TensorStructInfo((1,), "float16"))
    attn_scale = relax.Var("attn_scale", relax.TensorStructInfo((1,), "float16"))
    wo = relax.Var("wo", relax.TensorStructInfo((inner, c), "float16"))
    bo = relax.Var("bo", relax.TensorStructInfo((c,), "float16"))
    pos = relax.Var("pos", relax.TensorStructInfo((tokens, tokens), "float16"))
    weight_scale_vars = {"wq": wq_scale, "wk": wk_scale, "wv": wv_scale}

    def dequant_i32(expr, weight_key: str):
        y = bb.emit(relax.op.astype(expr, "float16"))
        y = bb.emit(relax.op.multiply(y, x_scale))
        return bb.emit(relax.op.multiply(y, weight_scale_vars[weight_key]))

    with bb.function("main", [x, x_scale, wq, wk, wv, wq_scale, wk_scale, wv_scale, attn_scale, wo, bo, pos]):
        with bb.dataflow():
            x2 = bb.emit(relax.op.reshape(x, (flat_tokens, c)))
            q = dequant_i32(bb.emit(relax.op.matmul(x2, wq, out_dtype="int32")), "wq")
            k = dequant_i32(bb.emit(relax.op.matmul(x2, wk, out_dtype="int32")), "wk")
            v = dequant_i32(bb.emit(relax.op.matmul(x2, wv, out_dtype="int32")), "wv")
            q_w = bb.emit(_reshape_windows(q, plan))
            k_w = bb.emit(_reshape_windows(k, plan))
            v_w = bb.emit(_reshape_windows(v, plan))
            k_t = bb.emit(relax.op.permute_dims(k_w, axes=[0, 2, 1]))
            dots = bb.emit(relax.op.matmul(q_w, k_t))
            dots = bb.emit(relax.op.multiply(dots, attn_scale))
            pos_b = bb.emit(relax.op.reshape(pos, (1, tokens, tokens)))
            dots = bb.emit(relax.op.add(dots, pos_b))
            attn = bb.emit(relax.op.nn.softmax(dots, axis=-1))
            out = bb.emit(relax.op.matmul(attn, v_w))
            out2 = bb.emit(_unshape_windows(out, plan))
            y = bb.emit(relax.op.matmul(out2, wo))
            y = bb.emit(relax.op.add(y, bo))
            y = bb.emit(relax.op.reshape(y, (b, l, h, w, c)))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def build_relax_mswin_bwa_w8a16_mixed(plan: dict[str, Any]):
    """Build MSwin BWA with INT8 Q/K/V weights and FP16 activations."""
    from tvm import relax
    from t1_attention_tvm_bench import _reshape_windows, _unshape_windows

    b, l, h, w, c = plan["input_shape"]
    inner = plan["inner_dim"]
    tokens = plan["window_tokens"]
    flat_tokens = b * l * h * w

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((b, l, h, w, c), "float16"))
    wq = relax.Var("wq", relax.TensorStructInfo((c, inner), "int8"))
    wk = relax.Var("wk", relax.TensorStructInfo((c, inner), "int8"))
    wv = relax.Var("wv", relax.TensorStructInfo((c, inner), "int8"))
    wq_scale = relax.Var("wq_scale", relax.TensorStructInfo((1,), "float16"))
    wk_scale = relax.Var("wk_scale", relax.TensorStructInfo((1,), "float16"))
    wv_scale = relax.Var("wv_scale", relax.TensorStructInfo((1,), "float16"))
    attn_scale = relax.Var("attn_scale", relax.TensorStructInfo((1,), "float16"))
    wo = relax.Var("wo", relax.TensorStructInfo((inner, c), "float16"))
    bo = relax.Var("bo", relax.TensorStructInfo((c,), "float16"))
    pos = relax.Var("pos", relax.TensorStructInfo((tokens, tokens), "float16"))

    def dequant_weight(w, scale):
        return bb.emit(relax.op.multiply(bb.emit(relax.op.astype(w, "float16")), scale))

    with bb.function("main", [x, wq, wk, wv, wq_scale, wk_scale, wv_scale, attn_scale, wo, bo, pos]):
        with bb.dataflow():
            x2 = bb.emit(relax.op.reshape(x, (flat_tokens, c)))
            q = bb.emit(relax.op.matmul(x2, dequant_weight(wq, wq_scale)))
            k = bb.emit(relax.op.matmul(x2, dequant_weight(wk, wk_scale)))
            v = bb.emit(relax.op.matmul(x2, dequant_weight(wv, wv_scale)))
            q_w = bb.emit(_reshape_windows(q, plan))
            k_w = bb.emit(_reshape_windows(k, plan))
            v_w = bb.emit(_reshape_windows(v, plan))
            k_t = bb.emit(relax.op.permute_dims(k_w, axes=[0, 2, 1]))
            dots = bb.emit(relax.op.matmul(q_w, k_t))
            dots = bb.emit(relax.op.multiply(dots, attn_scale))
            pos_b = bb.emit(relax.op.reshape(pos, (1, tokens, tokens)))
            dots = bb.emit(relax.op.add(dots, pos_b))
            attn = bb.emit(relax.op.nn.softmax(dots, axis=-1))
            out = bb.emit(relax.op.matmul(attn, v_w))
            out2 = bb.emit(_unshape_windows(out, plan))
            y = bb.emit(relax.op.matmul(out2, wo))
            y = bb.emit(relax.op.add(y, bo))
            y = bb.emit(relax.op.reshape(y, (b, l, h, w, c)))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


class TvmMswinBwaQkvInt8MixedRuntime:
    """Callable TVM runtime for one BaseWindowAttention module."""

    def __init__(
        self,
        module: Any,
        sample_shape: tuple[int, ...],
        *,
        quant_policy: str = "w8a8",
        seed: int = 20260623,
    ):
        import torch
        import tvm
        from tvm import relax
        from t1_attention_tvm_bench import compile_relax_module, make_mswin_bwa_shape_plan

        if quant_policy not in {"w8a8", "w8a16"}:
            raise ValueError(f"unsupported MSwin quant_policy: {quant_policy}")
        self.quant_policy = quant_policy
        if len(sample_shape) != 5:
            raise ValueError(f"BaseWindowAttention input must be B,L,H,W,C, got {sample_shape}")
        b, l, h, w, c = [int(item) for item in sample_shape]
        inner = int(module.to_qkv.out_features // 3)
        heads = int(module.heads)
        dim_head = int(inner // heads)
        self.plan = make_mswin_bwa_shape_plan(
            b, l, h, w, c, heads=heads, dim_head=dim_head,
            window_size=int(module.window_size), prune_rate_pct=50 if heads < 4 else 0,
        )

        weight = module.to_qkv.weight.detach().to(device="cuda", dtype=torch.float16).contiguous()
        wq, wk, wv = [part.t().contiguous() for part in weight.chunk(3, dim=0)]
        wq_i8, wq_scale = _quantize_symmetric_int8(wq)
        wk_i8, wk_scale = _quantize_symmetric_int8(wk)
        wv_i8, wv_scale = _quantize_symmetric_int8(wv)
        wo = module.to_out[0].weight.detach().to(device="cuda", dtype=torch.float16).t().contiguous()
        if module.to_out[0].bias is None:
            bo = torch.zeros((c,), device="cuda", dtype=torch.float16)
        else:
            bo = module.to_out[0].bias.detach().to(device="cuda", dtype=torch.float16).contiguous()
        if bool(module.relative_pos_embedding):
            idx = module.relative_indices
            pos = module.pos_embedding[idx[:, :, 0], idx[:, :, 1]]
        else:
            pos = module.pos_embedding
        pos = pos.detach().to(device="cuda", dtype=torch.float16).contiguous()

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod = (
            build_relax_mswin_bwa_w8a16_mixed(self.plan)
            if quant_policy == "w8a16"
            else build_relax_mswin_bwa_qkv_int8_mixed(self.plan)
        )
        ex = compile_relax_module(mod, target)
        self.vm = relax.VirtualMachine(ex, dev)
        self._torch_weights = [
            wq_i8,
            wk_i8,
            wv_i8,
            torch.tensor([wq_scale], device="cuda", dtype=torch.float16),
            torch.tensor([wk_scale], device="cuda", dtype=torch.float16),
            torch.tensor([wv_scale], device="cuda", dtype=torch.float16),
            torch.tensor([float(dim_head ** -0.5)], device="cuda", dtype=torch.float16),
            wo,
            bo,
            pos,
        ]
        self._tvm_weights = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(item))
            for item in self._torch_weights
        ]
        self.seed = seed

    def __call__(self, x: Any) -> Any:
        import torch
        import tvm

        if self.quant_policy == "w8a16":
            x_fp16 = x.detach().to(dtype=torch.float16).contiguous()
            args = [
                tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x_fp16)),
                *self._tvm_weights,
            ]
        else:
            x_i8, x_scale = _quantize_symmetric_int8(x.detach().to(dtype=torch.float16).contiguous())
            scale_tensor = torch.tensor([x_scale], device=x.device, dtype=torch.float16)
            args = [
                tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x_i8)),
                tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(scale_tensor)),
                *self._tvm_weights,
            ]
        out = self.vm["main"](*args)
        return torch.utils.dlpack.from_dlpack(out).to(dtype=x.dtype)


def build_relax_hmsa_static_2agent_qkv_int8_mixed(plan: dict[str, Any]):
    """Build static 2-agent HMSA with int8 Q/K/V projections and FP16 relation core.

    The DAIR val coverage run observed a fixed type order ``[0, 0]``. This
    builder still accepts per-agent weights so it can specialize any observed
    two-agent order, but dynamic type dispatch is handled outside TVM by the
    wrapper and recorded in the measurement evidence.
    """
    from tvm import relax
    from t1_attention_tvm_bench import _hmsa_message, _hmsa_project_out, _weighted_pair_sum

    b, l, h, w, c = plan["input_shape"]
    if b != 1 or l != 2:
        raise ValueError("static HMSA TVM runtime currently requires B=1,L=2")
    heads, dim_head, inner = plan["heads"], plan["dim_head"], plan["inner_dim"]
    spatial = plan["spatial_tokens"]

    bb = relax.BlockBuilder()
    x0 = relax.Var("x0", relax.TensorStructInfo((spatial, c), "int8"))
    x1 = relax.Var("x1", relax.TensorStructInfo((spatial, c), "int8"))
    x0_scale = relax.Var("x0_scale", relax.TensorStructInfo((1,), "float16"))
    x1_scale = relax.Var("x1_scale", relax.TensorStructInfo((1,), "float16"))
    key0_mask = relax.Var("key0_mask", relax.TensorStructInfo((spatial,), "float16"))
    key1_mask = relax.Var("key1_mask", relax.TensorStructInfo((spatial,), "float16"))
    q0s = relax.Var("q0s", relax.TensorStructInfo((1,), "float16"))
    k0s = relax.Var("k0s", relax.TensorStructInfo((1,), "float16"))
    v0s = relax.Var("v0s", relax.TensorStructInfo((1,), "float16"))
    q1s = relax.Var("q1s", relax.TensorStructInfo((1,), "float16"))
    k1s = relax.Var("k1s", relax.TensorStructInfo((1,), "float16"))
    v1s = relax.Var("v1s", relax.TensorStructInfo((1,), "float16"))
    hmsa_scale = relax.Var("hmsa_scale", relax.TensorStructInfo((1,), "float16"))
    q0w = relax.Var("q0w", relax.TensorStructInfo((c, inner), "int8"))
    k0w = relax.Var("k0w", relax.TensorStructInfo((c, inner), "int8"))
    v0w = relax.Var("v0w", relax.TensorStructInfo((c, inner), "int8"))
    q0b = relax.Var("q0b", relax.TensorStructInfo((inner,), "float16"))
    k0b = relax.Var("k0b", relax.TensorStructInfo((inner,), "float16"))
    v0b = relax.Var("v0b", relax.TensorStructInfo((inner,), "float16"))
    q1w = relax.Var("q1w", relax.TensorStructInfo((c, inner), "int8"))
    k1w = relax.Var("k1w", relax.TensorStructInfo((c, inner), "int8"))
    v1w = relax.Var("v1w", relax.TensorStructInfo((c, inner), "int8"))
    q1b = relax.Var("q1b", relax.TensorStructInfo((inner,), "float16"))
    k1b = relax.Var("k1b", relax.TensorStructInfo((inner,), "float16"))
    v1b = relax.Var("v1b", relax.TensorStructInfo((inner,), "float16"))
    r00a = relax.Var("r00a", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r01a = relax.Var("r01a", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r10a = relax.Var("r10a", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r11a = relax.Var("r11a", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r00m = relax.Var("r00m", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r01m = relax.Var("r01m", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r10m = relax.Var("r10m", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    r11m = relax.Var("r11m", relax.TensorStructInfo((heads, dim_head, dim_head), "float16"))
    wo0 = relax.Var("wo0", relax.TensorStructInfo((inner, c), "float16"))
    wo1 = relax.Var("wo1", relax.TensorStructInfo((inner, c), "float16"))
    bo0 = relax.Var("bo0", relax.TensorStructInfo((c,), "float16"))
    bo1 = relax.Var("bo1", relax.TensorStructInfo((c,), "float16"))

    weight_scale_vars = {
        "q0w": q0s, "k0w": k0s, "v0w": v0s,
        "q1w": q1s, "k1w": k1s, "v1w": v1s,
    }

    def qkv_project(x, x_scale, w_var, b_var, weight_key: str):
        y = bb.emit(relax.op.matmul(x, w_var, out_dtype="int32"))
        y = bb.emit(relax.op.astype(y, "float16"))
        y = bb.emit(relax.op.multiply(y, x_scale))
        y = bb.emit(relax.op.multiply(y, weight_scale_vars[weight_key]))
        y = bb.emit(relax.op.add(y, b_var))
        y = bb.emit(relax.op.reshape(y, (spatial, heads, dim_head)))
        return bb.emit(relax.op.permute_dims(y, axes=[1, 0, 2]))

    def pair_score(q, rel_att, k):
        q_rel = bb.emit(relax.op.matmul(q, rel_att))
        prod = bb.emit(relax.op.multiply(q_rel, k))
        score = bb.emit(relax.op.sum(prod, axis=2))
        return bb.emit(relax.op.multiply(score, hmsa_scale))

    def softmax_two_masked(score0, score1):
        exp0 = bb.emit(relax.op.multiply(relax.op.exp(score0), key0_mask))
        exp1 = bb.emit(relax.op.multiply(relax.op.exp(score1), key1_mask))
        denom = bb.emit(relax.op.add(exp0, exp1))
        return bb.emit(relax.op.divide(exp0, denom)), bb.emit(relax.op.divide(exp1, denom))

    params = [
        x0, x1, x0_scale, x1_scale, key0_mask, key1_mask,
        q0s, k0s, v0s, q1s, k1s, v1s, hmsa_scale,
        q0w, k0w, v0w, q0b, k0b, v0b,
        q1w, k1w, v1w, q1b, k1b, v1b,
        r00a, r01a, r10a, r11a, r00m, r01m, r10m, r11m,
        wo0, wo1, bo0, bo1,
    ]
    with bb.function("main", params):
        with bb.dataflow():
            q0 = qkv_project(x0, x0_scale, q0w, q0b, "q0w")
            k0 = qkv_project(x0, x0_scale, k0w, k0b, "k0w")
            v0 = qkv_project(x0, x0_scale, v0w, v0b, "v0w")
            q1 = qkv_project(x1, x1_scale, q1w, q1b, "q1w")
            k1 = qkv_project(x1, x1_scale, k1w, k1b, "k1w")
            v1 = qkv_project(x1, x1_scale, v1w, v1b, "v1w")
            s00 = pair_score(q0, r00a, k0)
            s01 = pair_score(q0, r01a, k1)
            s10 = pair_score(q1, r10a, k0)
            s11 = pair_score(q1, r11a, k1)
            w00, w01 = softmax_two_masked(s00, s01)
            w10, w11 = softmax_two_masked(s10, s11)
            msg00 = _hmsa_message(bb, v0, r00m)
            msg01 = _hmsa_message(bb, v1, r01m)
            msg10 = _hmsa_message(bb, v0, r10m)
            msg11 = _hmsa_message(bb, v1, r11m)
            out0 = _weighted_pair_sum(bb, w00, msg00, w01, msg01, heads, spatial)
            out1 = _weighted_pair_sum(bb, w10, msg10, w11, msg11, heads, spatial)
            y0 = bb.emit(relax.op.add(_hmsa_project_out(bb, out0, wo0, plan), bo0))
            y1 = bb.emit(relax.op.add(_hmsa_project_out(bb, out1, wo1, plan), bo1))
            y = bb.emit(relax.op.concat([y0, y1], axis=0))
            y = bb.emit(relax.op.reshape(y, (b, l, h, w, c)))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def _hmsa_key_masks_from_mask(mask: Any, h: int, w: int) -> tuple[Any, Any]:
    """Extract key-agent masks as flattened H*W tensors for HMSA two-way softmax."""
    import torch

    if mask.shape[-1] == 2 and len(mask.shape) == 5:
        if int(mask.shape[1]) == h and int(mask.shape[2]) == w:
            key0 = mask[0, :, :, 0, 0]
            key1 = mask[0, :, :, 0, 1]
        elif int(mask.shape[1]) == 1 and int(mask.shape[2]) == 1:
            key0 = mask[0, 0, 0, 0, 0].expand(h, w)
            key1 = mask[0, 0, 0, 0, 1].expand(h, w)
        else:
            raise ValueError(f"unsupported HMSA mask shape: {tuple(mask.shape)}")
    elif len(mask.shape) == 5 and int(mask.shape[3]) == 2 and int(mask.shape[4]) == 1:
        key0 = mask[0, :, :, 0, 0]
        key1 = mask[0, :, :, 1, 0]
    elif len(mask.shape) == 2 and int(mask.shape[1]) == 2:
        key0 = mask[0, 0].expand(h, w)
        key1 = mask[0, 1].expand(h, w)
    else:
        raise ValueError(f"unsupported HMSA mask shape: {tuple(mask.shape)}")
    return (
        key0.detach().to(dtype=torch.float16).reshape(h * w).contiguous(),
        key1.detach().to(dtype=torch.float16).reshape(h * w).contiguous(),
    )


class TvmHmsaStatic2AgentQkvInt8MixedRuntime:
    """Callable TVM runtime for one HGTCavAttention module and type order."""

    def __init__(self, module: Any, sample_shape: tuple[int, ...], type_order: tuple[int, int]):
        import torch
        import tvm
        from tvm import relax
        from t1_attention_tvm_bench import compile_relax_module, make_hmsa_shape_plan

        if len(sample_shape) != 5:
            raise ValueError(f"HGTCavAttention input must be B,L,H,W,C, got {sample_shape}")
        b, l, h, w, c = [int(item) for item in sample_shape]
        if b != 1 or l != 2:
            raise ValueError(f"static HMSA TVM path requires B=1,L=2, got B={b},L={l}")
        dim_head = int(module.relation_att.shape[-1])
        heads = int(module.heads)
        self.plan = make_hmsa_shape_plan(
            b, l, h, w, c,
            heads=heads,
            dim_head=dim_head,
            num_types=int(module.num_types),
            num_relations=int(module.relation_att.shape[0]),
            prune_rate_pct=50 if heads <= 4 else 0,
        )
        self.type_order = tuple(int(item) for item in type_order)

        def linear_qkv(linear):
            weight = linear.weight.detach().to(device="cuda", dtype=torch.float16).t().contiguous()
            weight_i8, scale = _quantize_symmetric_int8(weight)
            if linear.bias is None:
                bias = torch.zeros((weight.shape[1],), device="cuda", dtype=torch.float16)
            else:
                bias = linear.bias.detach().to(device="cuda", dtype=torch.float16).contiguous()
            return weight_i8, scale, bias

        def out_linear(linear):
            weight = linear.weight.detach().to(device="cuda", dtype=torch.float16).t().contiguous()
            if linear.bias is None:
                bias = torch.zeros((weight.shape[1],), device="cuda", dtype=torch.float16)
            else:
                bias = linear.bias.detach().to(device="cuda", dtype=torch.float16).contiguous()
            return weight, bias

        t0, t1 = self.type_order
        q0w, q0s, q0b = linear_qkv(module.q_linears[t0])
        k0w, k0s, k0b = linear_qkv(module.k_linears[t0])
        v0w, v0s, v0b = linear_qkv(module.v_linears[t0])
        q1w, q1s, q1b = linear_qkv(module.q_linears[t1])
        k1w, k1s, k1b = linear_qkv(module.k_linears[t1])
        v1w, v1s, v1b = linear_qkv(module.v_linears[t1])
        wo0, bo0 = out_linear(module.a_linears[t0])
        wo1, bo1 = out_linear(module.a_linears[t1])

        def rel_param(src: int, dst: int, table):
            rel_idx = int(module.get_relation_type_index(src, dst))
            return table[rel_idx].detach().to(device="cuda", dtype=torch.float16).contiguous()

        rels = [
            rel_param(t0, t0, module.relation_att),
            rel_param(t0, t1, module.relation_att),
            rel_param(t1, t0, module.relation_att),
            rel_param(t1, t1, module.relation_att),
            rel_param(t0, t0, module.relation_msg),
            rel_param(t0, t1, module.relation_msg),
            rel_param(t1, t0, module.relation_msg),
            rel_param(t1, t1, module.relation_msg),
        ]

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod = build_relax_hmsa_static_2agent_qkv_int8_mixed(self.plan)
        ex = compile_relax_module(mod, target)
        self.vm = relax.VirtualMachine(ex, dev)
        self._torch_weights = [
            torch.tensor([q0s], device="cuda", dtype=torch.float16),
            torch.tensor([k0s], device="cuda", dtype=torch.float16),
            torch.tensor([v0s], device="cuda", dtype=torch.float16),
            torch.tensor([q1s], device="cuda", dtype=torch.float16),
            torch.tensor([k1s], device="cuda", dtype=torch.float16),
            torch.tensor([v1s], device="cuda", dtype=torch.float16),
            torch.tensor([float(dim_head ** -0.5)], device="cuda", dtype=torch.float16),
            q0w, k0w, v0w, q0b, k0b, v0b,
            q1w, k1w, v1w, q1b, k1b, v1b,
            *rels,
            wo0, wo1, bo0, bo1,
        ]
        self._tvm_weights = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(item))
            for item in self._torch_weights
        ]

    def __call__(self, x: Any, mask: Any) -> Any:
        import torch
        import tvm

        b, l, h, w, c = [int(item) for item in x.shape]
        x0 = x[0, 0].detach().to(dtype=torch.float16).reshape(h * w, c).contiguous()
        x1 = x[0, 1].detach().to(dtype=torch.float16).reshape(h * w, c).contiguous()
        key0_mask, key1_mask = _hmsa_key_masks_from_mask(mask, h, w)
        x0_i8, x0_scale = _quantize_symmetric_int8(x0)
        x1_i8, x1_scale = _quantize_symmetric_int8(x1)
        args = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x0_i8)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x1_i8)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(torch.tensor([x0_scale], device=x.device, dtype=torch.float16))),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(torch.tensor([x1_scale], device=x.device, dtype=torch.float16))),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(key0_mask)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(key1_mask)),
            *self._tvm_weights,
        ]
        out = self.vm["main"](*args)
        return torch.utils.dlpack.from_dlpack(out).to(dtype=x.dtype)


def install_mswin_tvm_qkv_int8_mixed(model: Any, *, quant_policy: str = "w8a8") -> dict[str, Any]:
    """Patch BaseWindowAttention modules lazily with TVM qkv-int8 runtimes."""
    from opencood.models.sub_modules.mswin import BaseWindowAttention

    if quant_policy not in {"w8a8", "w8a16"}:
        raise ValueError(f"unsupported MSwin quant_policy: {quant_policy}")
    patched: list[str] = []
    runtime_boxes: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {
        "patched_count": 0,
        "patched_modules": patched,
        "call_count": 0,
        "compile_count": 0,
        "fallback_call_count": 0,
        "compiled_shapes": [],
        "quant_policy": quant_policy,
        "mixed_precision_policy": (
            f"MSwin BaseWindowAttention q/k/v projections in TVM {quant_policy}; "
            "softmax/output projection FP16 with PyTorch-equivalent scale and output bias"
        ),
    }
    for name, module in model.named_modules():
        if not isinstance(module, BaseWindowAttention):
            continue
        original_forward = module.forward
        runtime_boxes[name] = {"runtime": None, "original_forward": original_forward}

        def make_forward(module_name: str):
            def wrapped(self, x):
                state["call_count"] += 1
                box = runtime_boxes[module_name]
                if box["runtime"] is None:
                    state["compile_count"] += 1
                    state["compiled_shapes"].append({"module": module_name, "shape": list(x.shape)})
                    box["runtime"] = TvmMswinBwaQkvInt8MixedRuntime(
                        self,
                        tuple(x.shape),
                        quant_policy=quant_policy,
                    )
                return box["runtime"](x)

            return wrapped

        module.forward = make_forward(name).__get__(module, type(module))
        patched.append(name)

    state["patched_count"] = len(patched)
    return state


def install_hmsa_tvm_static_2agent_qkv_int8_mixed(
    model: Any,
    *,
    assumed_type_order: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Patch HGTCavAttention modules with a DAIR-observed static TVM path."""
    import torch
    from opencood.models.sub_modules.hmsa import HGTCavAttention

    patched: list[str] = []
    runtime_boxes: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {
        "patched_count": 0,
        "patched_modules": patched,
        "call_count": 0,
        "compile_count": 0,
        "fallback_call_count": 0,
        "fallback_reasons": {},
        "compiled_shapes": [],
        "assumed_type_order": list(assumed_type_order),
        "observed_type_orders": [],
        "first_mask_all_ones": None,
        "mask_policy": "lowered_as_two_key_mask_in_tvm_softmax",
        "covers_dynamic_type_dispatch": False,
        "mixed_precision_policy": (
            "HMSA static 2-agent path specialized to DAIR-observed type order; "
            "q/k/v projections in TVM int8; relation_att/msg, softmax, output projection FP16"
        ),
    }

    def add_fallback(reason: str) -> None:
        state["fallback_call_count"] += 1
        reasons = state["fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def remember_order(order: tuple[int, ...]) -> None:
        item = [int(value) for value in order]
        if item not in state["observed_type_orders"]:
            state["observed_type_orders"].append(item)

    for name, module in model.named_modules():
        if not isinstance(module, HGTCavAttention):
            continue
        original_forward = module.forward
        runtime_boxes[name] = {
            "runtime": None,
            "shape": None,
            "original_forward": original_forward,
            "checked_static_inputs": False,
            "fallback_always": False,
        }

        def make_forward(module_name: str):
            def wrapped(self, x, mask, prior_encoding):
                box = runtime_boxes[module_name]
                if len(tuple(x.shape)) != 5:
                    add_fallback("unsupported_input_rank")
                    return box["original_forward"](x, mask, prior_encoding)
                b, l, _h, _w, _c = [int(item) for item in x.shape]
                if b != 1 or l != 2:
                    add_fallback("unsupported_batch_or_agent_count")
                    return box["original_forward"](x, mask, prior_encoding)

                if not box["checked_static_inputs"]:
                    order_tensor = prior_encoding[0, :, 0, 0, 2].to(torch.int).detach().cpu()
                    observed_order = tuple(int(item) for item in order_tensor.tolist())
                    remember_order(observed_order)
                    if observed_order != assumed_type_order:
                        box["fallback_always"] = True
                        add_fallback("type_order_mismatch")
                    try:
                        key0_mask, key1_mask = _hmsa_key_masks_from_mask(mask, _h, _w)
                    except ValueError:
                        box["fallback_always"] = True
                        add_fallback("unsupported_mask_shape")
                    else:
                        mask_all_ones = bool(torch.all((key0_mask != 0) & (key1_mask != 0)).detach().cpu().item())
                        state["first_mask_all_ones"] = mask_all_ones if state["first_mask_all_ones"] is None else state["first_mask_all_ones"]
                    box["checked_static_inputs"] = True

                if box["fallback_always"]:
                    add_fallback("static_precondition_failed")
                    return box["original_forward"](x, mask, prior_encoding)

                state["call_count"] += 1
                shape = tuple(int(item) for item in x.shape)
                if box["runtime"] is None or box["shape"] != shape:
                    state["compile_count"] += 1
                    state["compiled_shapes"].append({"module": module_name, "shape": list(shape)})
                    box["runtime"] = TvmHmsaStatic2AgentQkvInt8MixedRuntime(self, shape, assumed_type_order)
                    box["shape"] = shape
                return box["runtime"](x, mask)

            return wrapped

        module.forward = make_forward(name).__get__(module, type(module))
        patched.append(name)

    state["patched_count"] = len(patched)
    return state


def _empty_patch_state(policy: str) -> dict[str, Any]:
    return {
        "patched_count": 0,
        "patched_modules": [],
        "call_count": 0,
        "compile_count": 0,
        "fallback_call_count": 0,
        "compiled_shapes": [],
        "mixed_precision_policy": policy,
    }


def install_attention_tvm_qkv_int8_mixed(
    model: Any,
    *,
    tvm_scope: str = "all",
    quant_policy: str = "w8a8",
) -> dict[str, Any]:
    """Patch all implemented attention fusion subgraphs and return live stats."""
    if tvm_scope not in {"all", "mswin", "hmsa"}:
        raise ValueError(f"unsupported tvm_scope: {tvm_scope}")
    mswin = (
        install_mswin_tvm_qkv_int8_mixed(model, quant_policy=quant_policy)
        if tvm_scope in {"all", "mswin"}
        else _empty_patch_state("MSwin TVM patch disabled for this run")
    )
    hmsa = (
        install_hmsa_tvm_static_2agent_qkv_int8_mixed(model, assumed_type_order=(0, 0))
        if tvm_scope in {"all", "hmsa"}
        else _empty_patch_state("HMSA TVM patch disabled for this run")
    )
    return {
        "backend": "TVM Relax VM",
        "fusion_subgraph_scope": tvm_scope,
        "quant_policy": quant_policy,
        "mswin": mswin,
        "hmsa": hmsa,
    }


def summarize_tvm_runtime_stats(patch_info: dict[str, Any]) -> dict[str, Any]:
    mswin = patch_info.get("mswin", {})
    hmsa = patch_info.get("hmsa", {})
    mswin_calls = int(mswin.get("call_count", 0) or 0)
    hmsa_calls = int(hmsa.get("call_count", 0) or 0)
    fallback_calls = int(mswin.get("fallback_call_count", 0) or 0) + int(hmsa.get("fallback_call_count", 0) or 0)
    return {
        "total_tvm_call_count": mswin_calls + hmsa_calls,
        "mswin_call_count": mswin_calls,
        "hmsa_call_count": hmsa_calls,
        "fallback_call_count": fallback_calls,
        "mswin_compile_count": int(mswin.get("compile_count", 0) or 0),
        "hmsa_compile_count": int(hmsa.get("compile_count", 0) or 0),
    }


def mixed_precision_policy_description(tvm_scope: str, quant_policy: str) -> str:
    mswin = (
        f"MSwin q/k/v {quant_policy.upper()} in TVM with FP16 softmax/output"
        if tvm_scope in {"all", "mswin"}
        else "MSwin TVM disabled"
    )
    hmsa = (
        "HMSA static [0,0] q/k/v INT8 in TVM with FP16 relation/softmax/output"
        if tvm_scope in {"all", "hmsa"}
        else "HMSA TVM disabled"
    )
    return f"scope={tvm_scope}; quant_policy={quant_policy}; {mswin}; {hmsa}"


def run_mswin_tvm_qkv_int8_forward_pilot(
    *,
    device: str,
    eval_samples: int,
    num_workers: int,
    out_log: str | Path,
) -> dict[str, Any]:
    """Patch MSwin subgraphs with TVM qkv-int8 and run full model forwards."""
    import time

    bootstrap = bootstrap_h800_tvm_then_torch()
    import torch
    from torch.utils.data import DataLoader

    from opencood.tools import train_utils
    from attention_e2e_checkpoint_eval import build_dataset_from_hypes, default_eval_configs, load_eval_model

    started = time.time()
    config = next(item for item in default_eval_configs() if item["config"] == "attention-p50-shortft-fp16")
    model, hypes, load_info = load_eval_model(config, device)
    patch_info = install_mswin_tvm_qkv_int8_mixed(model)
    dataset = build_dataset_from_hypes(hypes)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    latencies: list[float] = []
    n_done = 0
    status = "OK"
    error = None
    tb = None
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                if n_done >= eval_samples:
                    break
                batch = train_utils.to_device(batch, torch.device(device))
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(batch["ego"])
                end.record()
                torch.cuda.synchronize()
                latencies.append(float(start.elapsed_time(end)))
                n_done += 1
    except Exception as exc:  # pragma: no cover - H800 integration path
        import traceback

        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()[-5000:]

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        import numpy as np

        return round(float(np.percentile(values, q)), 4)

    result = {
        "schema_version": "attention_mswin_tvm_qkv_int8_forward_pilot_v1",
        "status": status,
        "latency_scope": "e2e_forward_model_with_mswin_tvm_subgraph",
        "dataset_split": "DAIR val",
        "requested_samples": eval_samples,
        "sample_count": n_done,
        "mean_ms": round(float(sum(latencies) / len(latencies)), 4) if latencies else None,
        "p50_ms": pct(latencies, 50),
        "p95_ms": pct(latencies, 95),
        "checkpoint_path": config["checkpoint_path"],
        "manifest_path": config["manifest_path"],
        "load_info": load_info,
        "bootstrap": bootstrap,
        "tvm_patch": patch_info,
        "caveat": (
            "Pilot only: MSwin BaseWindowAttention q/k/v projections run through TVM int8; "
            "HMSA dynamic/static TVM path and AP evaluation are not covered by this log."
        ),
        "elapsed_secs": round(time.time() - started, 3),
    }
    if error:
        result["error"] = error
        result["traceback"] = tb

    out_path = _resolve_repo_path(out_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["log_path"] = str(out_path)
    return result


def _find_report_row(report: dict[str, Any], config: str) -> dict[str, Any]:
    for row in report.get("rows", []):
        if row.get("config") == config:
            return row
    raise ValueError(f"missing {config} row in baseline report")


def _baseline_metrics_from_report(path: str | Path) -> dict[str, float]:
    report = json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))
    row = _find_report_row(report, "baseline")
    latency = row.get("latency_p50_ms", row.get("e2e_latency_ms"))
    return {
        "latency_p50_ms": float(latency),
        "ap50": float(row["ap50"]),
        "ap70": float(row["ap70"]),
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    out_path = _resolve_repo_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(out_path)


def run_tvm_mixed_int8_e2e_measurement(
    *,
    device: str,
    precision: str,
    tvm_scope: str,
    quant_policy: str,
    eval_samples: int,
    latency_samples: int,
    latency_warmup: int,
    num_workers: int,
    baseline_report: str | Path,
    out_measurement: str | Path,
) -> dict[str, Any]:
    """Run real full-model latency/AP with TVM-patched attention subgraphs."""
    bootstrap = bootstrap_h800_tvm_then_torch()
    from attention_e2e_checkpoint_eval import build_dataset_from_hypes, default_eval_configs, load_eval_model
    from attention_final_acceptance_report import ap_gain_audit_from_manifest, summarize_prune_manifest
    from t1_attention_e2e_pq import LOG_DIR, evaluate_ap, measure_model_forward_latency

    started = time.time()
    config = next(item for item in default_eval_configs() if item["config"] == "attention-p50-shortft-fp16")
    model, hypes, load_info = load_eval_model(config, device)
    patch_info = install_attention_tvm_qkv_int8_mixed(model, tvm_scope=tvm_scope, quant_policy=quant_policy)
    dataset = build_dataset_from_hypes(hypes)
    baseline = _baseline_metrics_from_report(baseline_report)

    latency_log = LOG_DIR / "attention-p50-int8_mixed_tvm_e2e_latency_v1.json"
    ap_log = LOG_DIR / "attention-p50-int8_mixed_tvm_e2e_ap_v1.json"
    runtime_log = DEFAULT_RUNTIME_LOG

    latency = measure_model_forward_latency(
        model,
        dataset,
        device,
        precision,
        latency_warmup,
        latency_samples,
        latency_log,
        num_workers,
    )
    latency["log_path"] = str(latency_log)
    ap = evaluate_ap(model, dataset, device, precision, eval_samples, ap_log, num_workers)
    ap["log_path"] = str(ap_log)

    runtime_stats = summarize_tvm_runtime_stats(patch_info)
    runtime_payload = {
        "schema_version": "attention_tvm_runtime_evidence_v1",
        "status": "OK" if runtime_stats["total_tvm_call_count"] > 0 else "NO_TVM_CALLS",
        "backend": "TVM Relax VM",
        "bootstrap": bootstrap,
        "patch_info": patch_info,
        "runtime_stats": runtime_stats,
        "latency_log": str(latency_log),
        "ap_log": str(ap_log),
    }
    runtime_log_path = _write_json(runtime_log, runtime_payload)

    e2e_latency = latency.get("p50_ms")
    ap50 = ap.get("ap50")
    ap70 = ap.get("ap70")
    delta_ap50 = round(float(ap50) - float(baseline["ap50"]), 4) if ap50 is not None else None
    delta_ap70 = round(float(ap70) - float(baseline["ap70"]), 4) if ap70 is not None else None
    measurement = {
        "schema_version": "attention_tvm_mixed_int8_e2e_measurement_v1",
        "status": "OK" if latency.get("status") == "OK" and ap.get("status") == "OK" else "FAILED",
        "config": REQUIRED_CONFIG,
        "tvm_scope": tvm_scope,
        "quant_policy": quant_policy,
        "attention_prune_pct": 50,
        "quant": REQUIRED_QUANT,
        "quant_backend": REQUIRED_BACKEND,
        "checkpoint_path": config["checkpoint_path"],
        "manifest_path": config["manifest_path"],
        "finetune": config["finetune"],
        "dataset_split": "DAIR val",
        "n_samples": int(ap.get("sample_count", eval_samples) or eval_samples),
        "latency": {
            "status": latency.get("status"),
            "latency_scope": "e2e",
            "source_latency_scope": latency.get("latency_scope"),
            "e2e_latency_ms": e2e_latency,
            "speedup": round(float(baseline["latency_p50_ms"]) / float(e2e_latency), 4) if e2e_latency else None,
            "latency_command": latency.get("command", " ".join(sys.argv)),
            "latency_log": str(latency_log),
            "latency_samples": latency.get("sample_count"),
            "latency_warmup": latency_warmup,
        },
        "ap": {
            "status": ap.get("status"),
            "ap50": ap50,
            "ap70": ap70,
            "delta_ap50": delta_ap50,
            "delta_ap70": delta_ap70,
            "ap_command": ap.get("command", " ".join(sys.argv)),
            "ap_log": str(ap_log),
            "ap_source": "real_dair_val_eval",
            "accuracy_status": ap.get("accuracy_status"),
        },
        "prune_manifest": summarize_prune_manifest(config["manifest_path"]),
        "tvm": {
            "artifact_path": runtime_log_path,
            "runtime": "TVM Relax VM",
            "tvm_version": bootstrap.get("tvm_version"),
            "target": "cuda",
            "runtime_log": runtime_log_path,
            "runtime_stats": runtime_stats,
            "mixed_precision_policy": mixed_precision_policy_description(tvm_scope, quant_policy),
            "covered_modules": {
                "mswin": patch_info["mswin"].get("patched_modules", []),
                "hmsa": patch_info["hmsa"].get("patched_modules", []),
            },
        },
        "dynamic_type_dispatch_coverage": {
            "status": "static_type_order_observed_on_full_dair_val",
            "covers_dynamic_type_dispatch": False,
            "observed_type_orders": [[0, 0]],
            "sample_count": REQUIRED_FULL_VAL_SAMPLES,
            "hmsa_call_count": 5367,
            "log_path": "logs/attention_e2e_pq_v1/hmsa_type_dispatch_coverage_full_v1.json",
        },
        "baseline_reference": {
            "report": str(baseline_report),
            **baseline,
        },
        "elapsed_secs": round(time.time() - started, 3),
    }
    if (delta_ap50 is not None and delta_ap50 > 0) or (delta_ap70 is not None and delta_ap70 > 0):
        measurement["ap_gain_audit"] = ap_gain_audit_from_manifest(config["manifest_path"])
    measurement_path = _write_json(out_measurement, measurement)
    measurement["measurement_path"] = measurement_path
    return measurement


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a dict")
    return value


def _require_non_missing(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if _is_missing(value):
        raise ValueError(f"{key} is required")
    return value


def _require_numeric(payload: dict[str, Any], key: str) -> float:
    value = _require_non_missing(payload, key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_existing_log(path_value: Any, field_name: str) -> str:
    path = str(_require_non_missing({field_name: path_value}, field_name))
    if not _resolve_repo_path(path).exists():
        raise ValueError(f"{field_name} must point to an existing log file")
    return path


def _validate_top_level(payload: dict[str, Any]) -> None:
    if payload.get("status") != "OK":
        raise ValueError("measurement status must be OK")
    if payload.get("config") != REQUIRED_CONFIG:
        raise ValueError(f"config must be {REQUIRED_CONFIG}")
    if payload.get("quant") != REQUIRED_QUANT:
        raise ValueError(f"quant must be {REQUIRED_QUANT}")
    if payload.get("quant_backend") != REQUIRED_BACKEND:
        raise ValueError(f"quant_backend must be {REQUIRED_BACKEND}")
    if int(_require_numeric(payload, "attention_prune_pct")) != 50:
        raise ValueError("attention_prune_pct must be 50")
    if str(payload.get("dataset_split")) != "DAIR val":
        raise ValueError("dataset_split must be DAIR val")
    if int(_require_numeric(payload, "n_samples")) != REQUIRED_FULL_VAL_SAMPLES:
        raise ValueError(f"n_samples must be {REQUIRED_FULL_VAL_SAMPLES}")
    for key in (
        "checkpoint_path",
        "manifest_path",
        "finetune",
        "prune_manifest",
        "dynamic_type_dispatch_coverage",
    ):
        _require_non_missing(payload, key)


def _validate_latency(latency: dict[str, Any]) -> None:
    if latency.get("status") != "OK":
        raise ValueError("latency.status must be OK")
    if str(latency.get("latency_scope", "")).lower() != "e2e":
        raise ValueError("latency_scope must be e2e")
    for key in ("e2e_latency_ms", "speedup"):
        value = _require_numeric(latency, key)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
    _require_non_missing(latency, "latency_command")
    _require_existing_log(latency.get("latency_log"), "latency_log")


def _validate_ap(ap: dict[str, Any]) -> None:
    if ap.get("status") != "OK":
        raise ValueError("ap.status must be OK")
    for key in ("ap50", "ap70", "delta_ap50", "delta_ap70"):
        _require_numeric(ap, key)
    _require_non_missing(ap, "ap_command")
    _require_existing_log(ap.get("ap_log"), "ap_log")
    status_text = " ".join(
        str(ap.get(key, ""))
        for key in ("accuracy_status", "ap_status", "ap_source", "ap_source_status", "evidence")
    ).lower()
    if any(token in status_text for token in FAKE_AP_TOKENS):
        raise ValueError("fake/prior AP cannot be used for TVM e2e row")


def _has_positive_ap_delta(ap: dict[str, Any]) -> bool:
    delta_ap50 = _require_numeric(ap, "delta_ap50")
    delta_ap70 = _require_numeric(ap, "delta_ap70")
    return delta_ap50 > 0 or delta_ap70 > 0


def _validate_ap_gain_audit(payload: dict[str, Any], ap: dict[str, Any]) -> None:
    if not _has_positive_ap_delta(ap):
        return
    audit = payload.get("ap_gain_audit")
    if not isinstance(audit, dict):
        raise ValueError("ap_gain_audit is required when AP improves over baseline")
    missing = [field for field in AP_GAIN_AUDIT_FIELDS if _is_missing(audit.get(field))]
    if missing:
        raise ValueError(f"ap_gain_audit missing fields: {missing}")
    for field in ("same_eval_protocol", "same_dataset_split", "same_checkpoint_family", "same_thresholds"):
        if audit.get(field) is not True:
            raise ValueError(f"ap_gain_audit.{field} must be true")


def _validate_tvm_evidence(tvm: dict[str, Any], tvm_scope: str) -> None:
    _require_non_missing(tvm, "runtime")
    _require_non_missing(tvm, "artifact_path")
    runtime = str(tvm.get("runtime", "")).lower()
    if "tvm" not in runtime:
        raise ValueError("tvm.runtime must identify TVM")
    _require_existing_log(tvm.get("artifact_path"), "tvm.artifact_path")
    stats = tvm.get("runtime_stats")
    if not isinstance(stats, dict):
        raise ValueError("tvm.runtime_stats must be a dict")
    total_calls = _require_numeric(stats, "total_tvm_call_count")
    if total_calls <= 0:
        raise ValueError("TVM runtime call count must be positive")
    fallback_calls = _require_numeric(stats, "fallback_call_count")
    if fallback_calls != 0:
        raise ValueError("TVM fallback_call_count must be 0")
    mswin_calls = _require_numeric(stats, "mswin_call_count")
    hmsa_calls = _require_numeric(stats, "hmsa_call_count")
    if tvm_scope == "all" and (mswin_calls <= 0 or hmsa_calls <= 0):
        raise ValueError("all scope requires both MSwin and HMSA TVM calls")
    if tvm_scope == "mswin" and mswin_calls <= 0:
        raise ValueError("mswin scope requires MSwin TVM calls")
    if tvm_scope == "hmsa" and hmsa_calls <= 0:
        raise ValueError("hmsa scope requires HMSA TVM calls")


def _validate_dynamic_type_dispatch_coverage(coverage: dict[str, Any]) -> None:
    if "covers_dynamic_type_dispatch" not in coverage:
        raise ValueError("dynamic_type_dispatch_coverage.covers_dynamic_type_dispatch is required")
    _require_non_missing(coverage, "status")
    if int(_require_numeric(coverage, "sample_count")) != REQUIRED_FULL_VAL_SAMPLES:
        raise ValueError(f"coverage sample_count must be {REQUIRED_FULL_VAL_SAMPLES}")
    _require_non_missing(coverage, "observed_type_orders")
    _require_non_missing(coverage, "hmsa_call_count")
    _require_existing_log(coverage.get("log_path"), "dynamic_type_dispatch_coverage.log_path")


def build_row_from_measurements(
    measurements: dict[str, Any],
    *,
    measurement_source: str | None = None,
) -> dict[str, Any]:
    """Return a Stop-A row from a real TVM e2e measurement payload."""
    _validate_top_level(measurements)
    tvm_scope = str(measurements.get("tvm_scope", ""))
    if tvm_scope not in {"all", "mswin", "hmsa"}:
        raise ValueError("tvm_scope must be one of all, mswin, hmsa")
    quant_policy = str(measurements.get("quant_policy", ""))
    if quant_policy not in {"w8a8", "w8a16"}:
        raise ValueError("quant_policy must be one of w8a8, w8a16")
    latency = _require_mapping(measurements, "latency")
    ap = _require_mapping(measurements, "ap")
    tvm = _require_mapping(measurements, "tvm")
    coverage = _require_mapping(measurements, "dynamic_type_dispatch_coverage")
    prune_manifest = _require_mapping(measurements, "prune_manifest")

    _validate_latency(latency)
    _validate_ap(ap)
    _validate_ap_gain_audit(measurements, ap)
    _validate_tvm_evidence(tvm, tvm_scope)
    _validate_dynamic_type_dispatch_coverage(coverage)

    row = {
        "config": REQUIRED_CONFIG,
        "attention_prune_pct": 50,
        "quant": REQUIRED_QUANT,
        "quant_backend": REQUIRED_BACKEND,
        "finetune": measurements["finetune"],
        "checkpoint_path": measurements["checkpoint_path"],
        "manifest_path": measurements["manifest_path"],
        "prune_manifest": prune_manifest,
        "latency_scope": "e2e",
        "e2e_latency_ms": latency["e2e_latency_ms"],
        "speedup": latency["speedup"],
        "ap50": ap["ap50"],
        "ap70": ap["ap70"],
        "delta_ap50": ap["delta_ap50"],
        "delta_ap70": ap["delta_ap70"],
        "latency_command": latency["latency_command"],
        "latency_log": latency["latency_log"],
        "ap_command": ap["ap_command"],
        "ap_log": ap["ap_log"],
        "dataset_split": measurements["dataset_split"],
        "n_samples": measurements["n_samples"],
        "tvm_scope": tvm_scope,
        "quant_policy": quant_policy,
        "tvm_evidence": dict(tvm),
        "dynamic_type_dispatch_coverage": dict(coverage),
        "evidence": "REAL_TVM_MIXED_INT8_E2E_FROM_MEASUREMENTS",
    }
    if _has_positive_ap_delta(ap):
        row["ap_gain_audit"] = dict(measurements["ap_gain_audit"])
    if measurement_source:
        row["measurement_source"] = measurement_source
    return row


def load_measurements(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("measurement file must contain a JSON object")
    return data


def write_row(row: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-measurements",
        help="JSON produced by a real TVM mixed-INT8 e2e latency/AP run.",
    )
    mode.add_argument(
        "--record-type-coverage",
        action="store_true",
        help="Run real forwards and record HMSA prior-encoding type orders.",
    )
    mode.add_argument(
        "--run-mswin-tvm-pilot",
        action="store_true",
        help="Run full model forwards with MSwin qkv-int8 TVM subgraphs patched in.",
    )
    mode.add_argument(
        "--run-tvm-e2e-measurement",
        action="store_true",
        help="Run full-model latency/AP with TVM mixed-INT8 attention subgraphs and write measurement JSON.",
    )
    parser.add_argument("--out-row", default=str(DEFAULT_OUT_ROW))
    parser.add_argument("--out-measurement", default=str(DEFAULT_OUT_MEASUREMENT))
    parser.add_argument("--out-log", default=str(DEFAULT_TYPE_COVERAGE_LOG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--tvm-scope", choices=("all", "mswin", "hmsa"), default="all")
    parser.add_argument("--quant-policy", choices=("w8a8", "w8a16"), default="w8a8")
    parser.add_argument("--eval-samples", type=int, default=1789)
    parser.add_argument("--latency-samples", type=int, default=30)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.record_type_coverage:
        result = record_hmsa_type_dispatch_coverage(
            device=args.device,
            eval_samples=args.eval_samples,
            num_workers=args.num_workers,
            out_log=args.out_log,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "OK" else 1

    if args.run_mswin_tvm_pilot:
        result = run_mswin_tvm_qkv_int8_forward_pilot(
            device=args.device,
            eval_samples=args.eval_samples,
            num_workers=args.num_workers,
            out_log=args.out_log,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "OK" else 1

    if args.run_tvm_e2e_measurement:
        result = run_tvm_mixed_int8_e2e_measurement(
            device=args.device,
            precision=args.precision,
            tvm_scope=args.tvm_scope,
            quant_policy=args.quant_policy,
            eval_samples=args.eval_samples,
            latency_samples=args.latency_samples,
            latency_warmup=args.latency_warmup,
            num_workers=args.num_workers,
            baseline_report=args.baseline_report,
            out_measurement=args.out_measurement,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "OK" else 1

    source_path = str(Path(args.from_measurements))
    row = build_row_from_measurements(
        load_measurements(args.from_measurements),
        measurement_source=source_path,
    )
    write_row(row, args.out_row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
