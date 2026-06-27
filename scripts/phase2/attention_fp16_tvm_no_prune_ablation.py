"""Measure no-prune/no-quant FP16 TVM acceleration for V2X-ViT attention.

This runner is deliberately separate from the Stop-A mixed-INT8 path.  It
answers a narrower ablation question:

* no structured pruning
* no quantization
* compare PyTorch FP16 vs TVM Relax FP16 for attention fusion subgraphs

The e2e latency scope matches the existing T1 reports: model forward only,
``model(batch["ego"])``.  It excludes dataloading, AP post-processing/NMS and
metric aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_JSON = REPO_ROOT / "results/attention_fp16_tvm_no_prune_ablation_v1.json"
DEFAULT_OUT_CSV = REPO_ROOT / "results/attention_fp16_tvm_no_prune_ablation_v1.csv"
DEFAULT_OUT_MD = REPO_ROOT / "results/attention_fp16_tvm_no_prune_ablation_v1.md"
DEFAULT_LOG_DIR = REPO_ROOT / "logs/attention_fp16_tvm_no_prune_v1"


def _torch_dtype_for_precision(precision: str):
    import torch

    return torch.float32 if precision == "fp32" else torch.float16


def _tvm_dtype_for_precision(precision: str) -> str:
    return "float32" if precision == "fp32" else "float16"


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    out = _resolve_repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(out)


def build_relax_mswin_bwa_fp_exact(plan: dict[str, Any], tvm_dtype: str):
    """TVM floating-point MSwin target matching BaseWindowAttention eval forward."""
    from tvm import relax
    from t1_attention_tvm_bench import _reshape_windows, _unshape_windows

    b, l, h, w, c = plan["input_shape"]
    inner = plan["inner_dim"]
    tokens = plan["window_tokens"]
    flat_tokens = b * l * h * w

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((b, l, h, w, c), tvm_dtype))
    wq = relax.Var("wq", relax.TensorStructInfo((c, inner), tvm_dtype))
    wk = relax.Var("wk", relax.TensorStructInfo((c, inner), tvm_dtype))
    wv = relax.Var("wv", relax.TensorStructInfo((c, inner), tvm_dtype))
    attn_scale = relax.Var("attn_scale", relax.TensorStructInfo((1,), tvm_dtype))
    wo = relax.Var("wo", relax.TensorStructInfo((inner, c), tvm_dtype))
    bo = relax.Var("bo", relax.TensorStructInfo((c,), tvm_dtype))
    pos = relax.Var("pos", relax.TensorStructInfo((tokens, tokens), tvm_dtype))

    with bb.function("main", [x, wq, wk, wv, attn_scale, wo, bo, pos]):
        with bb.dataflow():
            x2 = bb.emit(relax.op.reshape(x, (flat_tokens, c)))
            q = bb.emit(relax.op.matmul(x2, wq))
            k = bb.emit(relax.op.matmul(x2, wk))
            v = bb.emit(relax.op.matmul(x2, wv))
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


def build_relax_hmsa_static_2agent_fp_exact(plan: dict[str, Any], tvm_dtype: str):
    """TVM floating-point static two-agent HMSA target used for e2e patching."""
    from tvm import relax
    from t1_attention_tvm_bench import _hmsa_message, _hmsa_project_out, _weighted_pair_sum

    b, l, h, w, c = plan["input_shape"]
    if b != 1 or l != 2:
        raise ValueError("static HMSA FP16 TVM path currently requires B=1,L=2")
    heads, dim_head, inner = plan["heads"], plan["dim_head"], plan["inner_dim"]
    spatial = plan["spatial_tokens"]

    bb = relax.BlockBuilder()
    x0 = relax.Var("x0", relax.TensorStructInfo((spatial, c), tvm_dtype))
    x1 = relax.Var("x1", relax.TensorStructInfo((spatial, c), tvm_dtype))
    key0_mask = relax.Var("key0_mask", relax.TensorStructInfo((spatial,), tvm_dtype))
    key1_mask = relax.Var("key1_mask", relax.TensorStructInfo((spatial,), tvm_dtype))
    hmsa_scale = relax.Var("hmsa_scale", relax.TensorStructInfo((1,), tvm_dtype))
    q0w = relax.Var("q0w", relax.TensorStructInfo((c, inner), tvm_dtype))
    k0w = relax.Var("k0w", relax.TensorStructInfo((c, inner), tvm_dtype))
    v0w = relax.Var("v0w", relax.TensorStructInfo((c, inner), tvm_dtype))
    q0b = relax.Var("q0b", relax.TensorStructInfo((inner,), tvm_dtype))
    k0b = relax.Var("k0b", relax.TensorStructInfo((inner,), tvm_dtype))
    v0b = relax.Var("v0b", relax.TensorStructInfo((inner,), tvm_dtype))
    q1w = relax.Var("q1w", relax.TensorStructInfo((c, inner), tvm_dtype))
    k1w = relax.Var("k1w", relax.TensorStructInfo((c, inner), tvm_dtype))
    v1w = relax.Var("v1w", relax.TensorStructInfo((c, inner), tvm_dtype))
    q1b = relax.Var("q1b", relax.TensorStructInfo((inner,), tvm_dtype))
    k1b = relax.Var("k1b", relax.TensorStructInfo((inner,), tvm_dtype))
    v1b = relax.Var("v1b", relax.TensorStructInfo((inner,), tvm_dtype))
    r00a = relax.Var("r00a", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r01a = relax.Var("r01a", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r10a = relax.Var("r10a", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r11a = relax.Var("r11a", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r00m = relax.Var("r00m", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r01m = relax.Var("r01m", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r10m = relax.Var("r10m", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    r11m = relax.Var("r11m", relax.TensorStructInfo((heads, dim_head, dim_head), tvm_dtype))
    wo0 = relax.Var("wo0", relax.TensorStructInfo((inner, c), tvm_dtype))
    wo1 = relax.Var("wo1", relax.TensorStructInfo((inner, c), tvm_dtype))
    bo0 = relax.Var("bo0", relax.TensorStructInfo((c,), tvm_dtype))
    bo1 = relax.Var("bo1", relax.TensorStructInfo((c,), tvm_dtype))

    def qkv_project(x, w_var, b_var):
        y = bb.emit(relax.op.matmul(x, w_var))
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
        x0, x1, key0_mask, key1_mask, hmsa_scale,
        q0w, k0w, v0w, q0b, k0b, v0b,
        q1w, k1w, v1w, q1b, k1b, v1b,
        r00a, r01a, r10a, r11a, r00m, r01m, r10m, r11m,
        wo0, wo1, bo0, bo1,
    ]
    with bb.function("main", params):
        with bb.dataflow():
            q0 = qkv_project(x0, q0w, q0b)
            k0 = qkv_project(x0, k0w, k0b)
            v0 = qkv_project(x0, v0w, v0b)
            q1 = qkv_project(x1, q1w, q1b)
            k1 = qkv_project(x1, k1w, k1b)
            v1 = qkv_project(x1, v1w, v1b)
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


def _hmsa_key_masks_from_mask(mask: Any, h: int, w: int, *, torch_dtype: Any | None = None) -> tuple[Any, Any]:
    import torch

    dtype = torch_dtype or torch.float16
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
        key0.detach().to(dtype=dtype).reshape(h * w).contiguous(),
        key1.detach().to(dtype=dtype).reshape(h * w).contiguous(),
    )


class TvmMswinBwaFpRuntime:
    def __init__(self, module: Any, sample_shape: tuple[int, ...], *, precision: str):
        import torch
        import tvm
        from tvm import relax
        from t1_attention_tvm_bench import compile_relax_module, make_mswin_bwa_shape_plan

        self.precision = precision
        self.torch_dtype = _torch_dtype_for_precision(precision)
        self.tvm_dtype = _tvm_dtype_for_precision(precision)
        if len(sample_shape) != 5:
            raise ValueError(f"BaseWindowAttention input must be B,L,H,W,C, got {sample_shape}")
        b, l, h, w, c = [int(item) for item in sample_shape]
        inner = int(module.to_qkv.out_features // 3)
        heads = int(module.heads)
        dim_head = int(inner // heads)
        self.plan = make_mswin_bwa_shape_plan(
            b, l, h, w, c,
            heads=heads,
            dim_head=dim_head,
            window_size=int(module.window_size),
            prune_rate_pct=0,
        )

        weight = module.to_qkv.weight.detach().to(device="cuda", dtype=self.torch_dtype).contiguous()
        wq, wk, wv = [part.t().contiguous() for part in weight.chunk(3, dim=0)]
        wo = module.to_out[0].weight.detach().to(device="cuda", dtype=self.torch_dtype).t().contiguous()
        if module.to_out[0].bias is None:
            bo = torch.zeros((c,), device="cuda", dtype=self.torch_dtype)
        else:
            bo = module.to_out[0].bias.detach().to(device="cuda", dtype=self.torch_dtype).contiguous()
        if bool(module.relative_pos_embedding):
            idx = module.relative_indices
            pos = module.pos_embedding[idx[:, :, 0], idx[:, :, 1]]
        else:
            pos = module.pos_embedding
        pos = pos.detach().to(device="cuda", dtype=self.torch_dtype).contiguous()
        attn_scale = torch.tensor([float(module.scale)], device="cuda", dtype=self.torch_dtype)

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod = build_relax_mswin_bwa_fp_exact(self.plan, self.tvm_dtype)
        ex = compile_relax_module(mod, target)
        self.vm = relax.VirtualMachine(ex, dev)
        self._torch_weights = [wq, wk, wv, attn_scale, wo, bo, pos]
        self._tvm_weights = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(item))
            for item in self._torch_weights
        ]

    def __call__(self, x: Any) -> Any:
        import torch
        import tvm

        x_fp = x.detach().to(dtype=self.torch_dtype).contiguous()
        args = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x_fp)),
            *self._tvm_weights,
        ]
        out = self.vm["main"](*args)
        return torch.utils.dlpack.from_dlpack(out).to(dtype=x.dtype)


class TvmHmsaStatic2AgentFpRuntime:
    def __init__(self, module: Any, sample_shape: tuple[int, ...], type_order: tuple[int, int], *, precision: str):
        import torch
        import tvm
        from tvm import relax
        from t1_attention_tvm_bench import compile_relax_module, make_hmsa_shape_plan

        self.precision = precision
        self.torch_dtype = _torch_dtype_for_precision(precision)
        self.tvm_dtype = _tvm_dtype_for_precision(precision)
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
            prune_rate_pct=0,
        )
        self.type_order = tuple(int(item) for item in type_order)

        def linear_fp(linear):
            weight = linear.weight.detach().to(device="cuda", dtype=self.torch_dtype).t().contiguous()
            if linear.bias is None:
                bias = torch.zeros((weight.shape[1],), device="cuda", dtype=self.torch_dtype)
            else:
                bias = linear.bias.detach().to(device="cuda", dtype=self.torch_dtype).contiguous()
            return weight, bias

        t0, t1 = self.type_order
        q0w, q0b = linear_fp(module.q_linears[t0])
        k0w, k0b = linear_fp(module.k_linears[t0])
        v0w, v0b = linear_fp(module.v_linears[t0])
        q1w, q1b = linear_fp(module.q_linears[t1])
        k1w, k1b = linear_fp(module.k_linears[t1])
        v1w, v1b = linear_fp(module.v_linears[t1])
        wo0, bo0 = linear_fp(module.a_linears[t0])
        wo1, bo1 = linear_fp(module.a_linears[t1])

        def rel_param(src: int, dst: int, table):
            rel_idx = int(module.get_relation_type_index(src, dst))
            return table[rel_idx].detach().to(device="cuda", dtype=self.torch_dtype).contiguous()

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
        hmsa_scale = torch.tensor([float(module.scale)], device="cuda", dtype=self.torch_dtype)

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod = build_relax_hmsa_static_2agent_fp_exact(self.plan, self.tvm_dtype)
        ex = compile_relax_module(mod, target)
        self.vm = relax.VirtualMachine(ex, dev)
        self._torch_weights = [
            hmsa_scale,
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
        x0 = x[0, 0].detach().to(dtype=self.torch_dtype).reshape(h * w, c).contiguous()
        x1 = x[0, 1].detach().to(dtype=self.torch_dtype).reshape(h * w, c).contiguous()
        key0_mask, key1_mask = _hmsa_key_masks_from_mask(mask, h, w, torch_dtype=self.torch_dtype)
        args = [
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x0)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x1)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(key0_mask)),
            tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(key1_mask)),
            *self._tvm_weights,
        ]
        out = self.vm["main"](*args)
        return torch.utils.dlpack.from_dlpack(out).to(dtype=x.dtype)


def _empty_patch_state(policy: str) -> dict[str, Any]:
    return {
        "patched_count": 0,
        "patched_modules": [],
        "call_count": 0,
        "compile_count": 0,
        "fallback_call_count": 0,
        "compiled_shapes": [],
        "policy": policy,
    }


def install_mswin_tvm_fp(model: Any, *, precision: str) -> dict[str, Any]:
    from opencood.models.sub_modules.mswin import BaseWindowAttention

    patched: list[str] = []
    runtime_boxes: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {
        "patched_count": 0,
        "patched_modules": patched,
        "call_count": 0,
        "compile_count": 0,
        "fallback_call_count": 0,
        "compiled_shapes": [],
        "precision_policy": f"{precision.upper()} activations and {precision.upper()} weights in TVM Relax",
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
                    box["runtime"] = TvmMswinBwaFpRuntime(self, tuple(x.shape), precision=precision)
                return box["runtime"](x)

            return wrapped

        module.forward = make_forward(name).__get__(module, type(module))
        patched.append(name)

    state["patched_count"] = len(patched)
    return state


def install_hmsa_tvm_fp_static_2agent(
    model: Any,
    *,
    precision: str,
    assumed_type_order: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
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
        "covers_dynamic_type_dispatch": False,
        "precision_policy": f"{precision.upper()} activations and {precision.upper()} weights in TVM Relax; static two-agent type order",
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
                        _hmsa_key_masks_from_mask(mask, _h, _w)
                    except ValueError:
                        box["fallback_always"] = True
                        add_fallback("unsupported_mask_shape")
                    box["checked_static_inputs"] = True

                if box["fallback_always"]:
                    add_fallback("static_precondition_failed")
                    return box["original_forward"](x, mask, prior_encoding)

                state["call_count"] += 1
                shape = tuple(int(item) for item in x.shape)
                if box["runtime"] is None or box["shape"] != shape:
                    state["compile_count"] += 1
                    state["compiled_shapes"].append({"module": module_name, "shape": list(shape)})
                    box["runtime"] = TvmHmsaStatic2AgentFpRuntime(self, shape, assumed_type_order, precision=precision)
                    box["shape"] = shape
                return box["runtime"](x, mask)

            return wrapped

        module.forward = make_forward(name).__get__(module, type(module))
        patched.append(name)

    state["patched_count"] = len(patched)
    return state


def install_attention_tvm_fp(model: Any, *, tvm_scope: str, precision: str) -> dict[str, Any]:
    if tvm_scope not in {"all", "mswin", "hmsa"}:
        raise ValueError(f"unsupported tvm_scope: {tvm_scope}")
    mswin = (
        install_mswin_tvm_fp(model, precision=precision)
        if tvm_scope in {"all", "mswin"}
        else _empty_patch_state("MSwin TVM patch disabled for this run")
    )
    hmsa = (
        install_hmsa_tvm_fp_static_2agent(model, precision=precision, assumed_type_order=(0, 0))
        if tvm_scope in {"all", "hmsa"}
        else _empty_patch_state("HMSA TVM patch disabled for this run")
    )
    return {
        "backend": "TVM Relax VM",
        "fusion_subgraph_scope": tvm_scope,
        "precision": precision,
        "pruning": "none",
        "quantization": "none",
        "mswin": mswin,
        "hmsa": hmsa,
    }


def summarize_runtime_stats(patch_info: dict[str, Any]) -> dict[str, int]:
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


def _torch_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return round(float(np.percentile(values, q)), 6)


def _time_torch_callable(fn, *, warmup: int, number: int, repeat: int) -> dict[str, Any]:
    import torch
    import numpy as np

    values_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(repeat):
            started = time.perf_counter()
            for _ in range(number):
                fn()
            torch.cuda.synchronize()
            values_ms.append((time.perf_counter() - started) * 1000.0 / float(number))
    return {
        "status": "OK",
        "p50_ms": round(float(np.percentile(values_ms, 50)), 6),
        "mean_ms": round(float(np.mean(values_ms)), 6),
        "min_ms": round(float(np.min(values_ms)), 6),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 6),
        "number": number,
        "repeat": repeat,
        "warmup": warmup,
        "timer": "manual_wall_time_with_torch_cuda_sync",
    }


def _time_tvm_callable(vm, args: list[Any], dev: Any, *, warmup: int, number: int, repeat: int) -> dict[str, Any]:
    import numpy as np

    for _ in range(warmup):
        vm["main"](*args)
    dev.sync()
    values_ms: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        for _ in range(number):
            vm["main"](*args)
        dev.sync()
        values_ms.append((time.perf_counter() - started) * 1000.0 / float(number))
    return {
        "status": "OK",
        "p50_ms": round(float(np.percentile(values_ms, 50)), 6),
        "mean_ms": round(float(np.mean(values_ms)), 6),
        "min_ms": round(float(np.min(values_ms)), 6),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 6),
        "number": number,
        "repeat": repeat,
        "warmup": warmup,
        "timer": "manual_wall_time_with_tvm_dev_sync",
    }


def _torch_reshape_windows(expr: Any, plan: dict[str, Any]) -> Any:
    b, l, h, w = plan["batch"], plan["agents"], plan["h"], plan["w"]
    heads, dim_head = plan["heads"], plan["dim_head"]
    ws, new_h, new_w = plan["window_size"], plan["new_h"], plan["new_w"]
    expr = expr.reshape(b, l, new_h, ws, new_w, ws, heads, dim_head)
    expr = expr.permute(0, 1, 6, 2, 4, 3, 5, 7)
    return expr.reshape(plan["window_batches"], plan["window_tokens"], dim_head)


def _torch_unshape_windows(expr: Any, plan: dict[str, Any]) -> Any:
    b, l, heads, dim_head = plan["batch"], plan["agents"], plan["heads"], plan["dim_head"]
    ws, new_h, new_w = plan["window_size"], plan["new_h"], plan["new_w"]
    expr = expr.reshape(b, l, heads, new_h, new_w, ws, ws, dim_head)
    expr = expr.permute(0, 1, 3, 5, 4, 6, 2, 7)
    return expr.reshape(b * l * plan["h"] * plan["w"], plan["inner_dim"])


def _torch_mswin_forward(args: list[Any], plan: dict[str, Any]) -> Any:
    import torch

    x, wq, wk, wv, attn_scale, wo, bo, pos = args
    b, l, h, w, c = plan["input_shape"]
    tokens = plan["window_tokens"]
    x2 = x.reshape(b * l * h * w, c)
    q = torch.matmul(x2, wq)
    k = torch.matmul(x2, wk)
    v = torch.matmul(x2, wv)
    q_w = _torch_reshape_windows(q, plan)
    k_w = _torch_reshape_windows(k, plan)
    v_w = _torch_reshape_windows(v, plan)
    dots = torch.matmul(q_w, k_w.permute(0, 2, 1)) * attn_scale
    dots = dots + pos.reshape(1, tokens, tokens)
    attn = torch.softmax(dots, dim=-1)
    out = torch.matmul(attn, v_w)
    out2 = _torch_unshape_windows(out, plan)
    y = torch.matmul(out2, wo) + bo
    return y.reshape(b, l, h, w, c)


def _hmsa_qkv_project(x: Any, w: Any, b: Any, plan: dict[str, Any]) -> Any:
    y = x.matmul(w) + b
    y = y.reshape(plan["spatial_tokens"], plan["heads"], plan["dim_head"])
    return y.permute(1, 0, 2)


def _hmsa_pair_score(q: Any, rel_att: Any, k: Any, scale: Any) -> Any:
    return (q.matmul(rel_att) * k).sum(dim=2) * scale


def _hmsa_project_out(out: Any, wo: Any, bo: Any, plan: dict[str, Any]) -> Any:
    out = out.permute(1, 0, 2).reshape(plan["spatial_tokens"], plan["inner_dim"])
    return out.matmul(wo) + bo


def _torch_hmsa_forward(args: list[Any], plan: dict[str, Any]) -> Any:
    import torch

    (
        x0, x1, key0_mask, key1_mask, scale,
        q0w, k0w, v0w, q0b, k0b, v0b,
        q1w, k1w, v1w, q1b, k1b, v1b,
        r00a, r01a, r10a, r11a, r00m, r01m, r10m, r11m,
        wo0, wo1, bo0, bo1,
    ) = args
    q0 = _hmsa_qkv_project(x0, q0w, q0b, plan)
    k0 = _hmsa_qkv_project(x0, k0w, k0b, plan)
    v0 = _hmsa_qkv_project(x0, v0w, v0b, plan)
    q1 = _hmsa_qkv_project(x1, q1w, q1b, plan)
    k1 = _hmsa_qkv_project(x1, k1w, k1b, plan)
    v1 = _hmsa_qkv_project(x1, v1w, v1b, plan)
    s00 = _hmsa_pair_score(q0, r00a, k0, scale)
    s01 = _hmsa_pair_score(q0, r01a, k1, scale)
    s10 = _hmsa_pair_score(q1, r10a, k0, scale)
    s11 = _hmsa_pair_score(q1, r11a, k1, scale)
    exp00 = torch.exp(s00) * key0_mask
    exp01 = torch.exp(s01) * key1_mask
    exp10 = torch.exp(s10) * key0_mask
    exp11 = torch.exp(s11) * key1_mask
    w00 = exp00 / (exp00 + exp01)
    w01 = exp01 / (exp00 + exp01)
    w10 = exp10 / (exp10 + exp11)
    w11 = exp11 / (exp10 + exp11)
    msg00 = v0.matmul(r00m)
    msg01 = v1.matmul(r01m)
    msg10 = v0.matmul(r10m)
    msg11 = v1.matmul(r11m)
    out0 = w00.reshape(plan["heads"], plan["spatial_tokens"], 1) * msg00 + w01.reshape(plan["heads"], plan["spatial_tokens"], 1) * msg01
    out1 = w10.reshape(plan["heads"], plan["spatial_tokens"], 1) * msg10 + w11.reshape(plan["heads"], plan["spatial_tokens"], 1) * msg11
    y0 = _hmsa_project_out(out0, wo0, bo0, plan)
    y1 = _hmsa_project_out(out1, wo1, bo1, plan)
    y = torch.cat([y0, y1], dim=0)
    b, l, h, w, c = plan["input_shape"]
    return y.reshape(b, l, h, w, c)


def _randn(shape: tuple[int, ...], *, device: str, dtype: Any):
    import torch

    return torch.randn(shape, device=device, dtype=dtype).contiguous()


def run_subnet_ablation(*, device: str, precision: str, number: int, repeat: int, warmup: int, seed: int) -> dict[str, Any]:
    import torch
    import tvm
    from tvm import relax
    from t1_attention_tvm_bench import compile_relax_module, make_hmsa_shape_plan, make_mswin_bwa_shape_plan

    torch.manual_seed(seed)
    torch_dtype = _torch_dtype_for_precision(precision)
    tvm_dtype = _tvm_dtype_for_precision(precision)
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)

    mswin_plan = make_mswin_bwa_shape_plan(
        1, 2, 64, 128, 256,
        heads=4,
        dim_head=64,
        window_size=16,
        prune_rate_pct=0,
    )
    hmsa_plan = make_hmsa_shape_plan(
        1, 2, 64, 128, 256,
        heads=8,
        dim_head=32,
        num_types=2,
        num_relations=4,
        prune_rate_pct=0,
    )
    mswin_args = [
        _randn(tuple(mswin_plan["input_shape"]), device=device, dtype=torch_dtype),
        _randn((256, mswin_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, mswin_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, mswin_plan["inner_dim"]), device=device, dtype=torch_dtype),
        torch.tensor([64 ** -0.5], device=device, dtype=torch_dtype),
        _randn((mswin_plan["inner_dim"], 256), device=device, dtype=torch_dtype),
        _randn((256,), device=device, dtype=torch_dtype),
        _randn((mswin_plan["window_tokens"], mswin_plan["window_tokens"]), device=device, dtype=torch_dtype),
    ]
    hmsa_args = [
        _randn((hmsa_plan["spatial_tokens"], 256), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["spatial_tokens"], 256), device=device, dtype=torch_dtype),
        torch.ones((hmsa_plan["spatial_tokens"],), device=device, dtype=torch_dtype),
        torch.ones((hmsa_plan["spatial_tokens"],), device=device, dtype=torch_dtype),
        torch.tensor([32 ** -0.5], device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((256, hmsa_plan["inner_dim"]), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"],), device=device, dtype=torch_dtype),
    ]
    hmsa_args.extend(_randn((hmsa_plan["heads"], hmsa_plan["dim_head"], hmsa_plan["dim_head"]), device=device, dtype=torch_dtype) for _ in range(8))
    hmsa_args.extend([
        _randn((hmsa_plan["inner_dim"], 256), device=device, dtype=torch_dtype),
        _randn((hmsa_plan["inner_dim"], 256), device=device, dtype=torch_dtype),
        _randn((256,), device=device, dtype=torch_dtype),
        _randn((256,), device=device, dtype=torch_dtype),
    ])

    mswin_torch = _time_torch_callable(
        lambda: _torch_mswin_forward(mswin_args, mswin_plan),
        warmup=warmup,
        number=number,
        repeat=repeat,
    )
    hmsa_torch = _time_torch_callable(
        lambda: _torch_hmsa_forward(hmsa_args, hmsa_plan),
        warmup=warmup,
        number=number,
        repeat=repeat,
    )

    mswin_mod = build_relax_mswin_bwa_fp_exact(mswin_plan, tvm_dtype)
    mswin_ex = compile_relax_module(mswin_mod, target)
    mswin_vm = relax.VirtualMachine(mswin_ex, dev)
    mswin_tvm_args = [tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(item)) for item in mswin_args]
    mswin_tvm = _time_tvm_callable(
        mswin_vm,
        mswin_tvm_args,
        dev,
        warmup=warmup,
        number=number,
        repeat=repeat,
    )

    hmsa_mod = build_relax_hmsa_static_2agent_fp_exact(hmsa_plan, tvm_dtype)
    hmsa_ex = compile_relax_module(hmsa_mod, target)
    hmsa_vm = relax.VirtualMachine(hmsa_ex, dev)
    hmsa_tvm_args = [tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(item)) for item in hmsa_args]
    hmsa_tvm = _time_tvm_callable(
        hmsa_vm,
        hmsa_tvm_args,
        dev,
        warmup=warmup,
        number=number,
        repeat=repeat,
    )

    torch_subnet = float(mswin_torch["p50_ms"]) + float(hmsa_torch["p50_ms"])
    tvm_subnet = float(mswin_tvm["p50_ms"]) + float(hmsa_tvm["p50_ms"])
    return {
        "schema_version": "attention_fp_subnet_no_prune_ablation_v1",
        "scope": "attention_subnet_mswin_full_window_plus_hmsa_static_2agent_qkv_relation_out",
        "pruning": "none",
        "quantization": "none",
        "precision": precision,
        "device": device,
        "number": number,
        "repeat": repeat,
        "warmup": warmup,
        "rows": {
            "mswin": {"pytorch": mswin_torch, "tvm": mswin_tvm, "speedup": _round(mswin_torch["p50_ms"] / mswin_tvm["p50_ms"], 4)},
            "hmsa_static": {"pytorch": hmsa_torch, "tvm": hmsa_tvm, "speedup": _round(hmsa_torch["p50_ms"] / hmsa_tvm["p50_ms"], 4)},
            "subnet_sum": {
                "pytorch_p50_ms": _round(torch_subnet, 6),
                "tvm_p50_ms": _round(tvm_subnet, 6),
                "speedup": _round(torch_subnet / tvm_subnet, 4) if tvm_subnet > 0 else None,
            },
        },
        "caveat": (
            "Subnet timing is a synthetic attention-subgraph runtime benchmark with the same tensor shapes "
            "used by prior T1 TVM reports. It is not full-model e2e latency and HMSA uses a static two-agent target."
        ),
    }


def run_e2e_ablation(
    *,
    device: str,
    precision: str,
    tvm_scope: str,
    latency_samples: int,
    latency_warmup: int,
    num_workers: int,
    log_dir: Path,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    from attention_e2e_checkpoint_eval import build_dataset_from_hypes, default_eval_configs, load_eval_model
    from t1_attention_e2e_pq import measure_model_forward_latency

    config = next(item for item in default_eval_configs() if item["config"] == "baseline")
    log_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, hypes, baseline_load = load_eval_model(config, device)
    dataset = build_dataset_from_hypes(hypes)
    baseline_log = log_dir / f"baseline_no_tvm_{precision}_latency_v1.json"
    baseline_latency = measure_model_forward_latency(
        baseline_model,
        dataset,
        device,
        precision,
        latency_warmup,
        latency_samples,
        baseline_log,
        num_workers,
    )
    baseline_latency["log_path"] = str(baseline_log)

    tvm_model, hypes_tvm, tvm_load = load_eval_model(config, device)
    tvm_dataset = build_dataset_from_hypes(hypes_tvm)
    patch_info = install_attention_tvm_fp(tvm_model, tvm_scope=tvm_scope, precision=precision)
    tvm_log = log_dir / f"baseline_tvm_{precision}_{tvm_scope}_latency_v1.json"
    tvm_latency = measure_model_forward_latency(
        tvm_model,
        tvm_dataset,
        device,
        precision,
        latency_warmup,
        latency_samples,
        tvm_log,
        num_workers,
    )
    tvm_latency["log_path"] = str(tvm_log)

    runtime_stats = summarize_runtime_stats(patch_info)
    speedup = None
    if baseline_latency.get("p50_ms") and tvm_latency.get("p50_ms"):
        speedup = _round(float(baseline_latency["p50_ms"]) / float(tvm_latency["p50_ms"]), 4)

    return {
        "schema_version": "attention_fp_e2e_no_prune_ablation_v1",
        "status": "OK" if baseline_latency.get("status") == "OK" and tvm_latency.get("status") == "OK" else "FAILED",
        "checkpoint_config": "baseline",
        "pruning": "none",
        "quantization": "none",
        "precision": precision,
        "tvm_scope": tvm_scope,
        "latency_scope": "model_forward_e2e",
        "source_latency_scope": "pytorch_model_forward_e2e",
        "dataset_split": "DAIR val",
        "latency_samples": latency_samples,
        "latency_warmup": latency_warmup,
        "baseline_no_tvm": {
            "latency_p50_ms": baseline_latency.get("p50_ms"),
            "latency_mean_ms": baseline_latency.get("mean_ms"),
            "latency_p95_ms": baseline_latency.get("p95_ms"),
            "latency_log": str(baseline_log),
            "load_info": baseline_load,
        },
        "tvm_fp": {
            "latency_p50_ms": tvm_latency.get("p50_ms"),
            "latency_mean_ms": tvm_latency.get("mean_ms"),
            "latency_p95_ms": tvm_latency.get("p95_ms"),
            "latency_log": str(tvm_log),
            "load_info": tvm_load,
            "runtime_stats": runtime_stats,
            "patch_info": patch_info,
        },
        "speedup": speedup,
        "bootstrap": bootstrap,
        "caveat": (
            "E2E timing is model forward only: model(batch['ego']). It excludes dataloading, "
            "post_process/NMS and AP metric aggregation. HMSA TVM path is a static two-agent "
            "specialization and records fallback counts in runtime_stats."
        ),
    }


def render_csv(report: dict[str, Any], path: str | Path) -> None:
    out = _resolve_repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    e2e = report.get("e2e", {})
    tvm_e2e = e2e.get("tvm_fp", e2e.get("tvm_fp16", {}))
    subnet = report.get("subnet", {}).get("rows", {})
    rows = [
        {
            "scope": "e2e_model_forward",
            "component": "full_model",
            "pytorch_no_tvm_p50_ms": e2e.get("baseline_no_tvm", {}).get("latency_p50_ms"),
            "tvm_fp_p50_ms": tvm_e2e.get("latency_p50_ms"),
            "speedup": e2e.get("speedup"),
        },
        {
            "scope": "subnet",
            "component": "mswin",
            "pytorch_no_tvm_p50_ms": subnet.get("mswin", {}).get("pytorch", {}).get("p50_ms"),
            "tvm_fp_p50_ms": subnet.get("mswin", {}).get("tvm", {}).get("p50_ms"),
            "speedup": subnet.get("mswin", {}).get("speedup"),
        },
        {
            "scope": "subnet",
            "component": "hmsa_static",
            "pytorch_no_tvm_p50_ms": subnet.get("hmsa_static", {}).get("pytorch", {}).get("p50_ms"),
            "tvm_fp_p50_ms": subnet.get("hmsa_static", {}).get("tvm", {}).get("p50_ms"),
            "speedup": subnet.get("hmsa_static", {}).get("speedup"),
        },
        {
            "scope": "subnet",
            "component": "mswin_plus_hmsa_static_sum",
            "pytorch_no_tvm_p50_ms": subnet.get("subnet_sum", {}).get("pytorch_p50_ms"),
            "tvm_fp_p50_ms": subnet.get("subnet_sum", {}).get("tvm_p50_ms"),
            "speedup": subnet.get("subnet_sum", {}).get("speedup"),
        },
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any], path: str | Path) -> None:
    out = _resolve_repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    e2e = report.get("e2e", {})
    precision = str(report.get("precision", e2e.get("precision", "fp16"))).upper()
    tvm_e2e = e2e.get("tvm_fp", e2e.get("tvm_fp16", {}))
    subnet = report.get("subnet", {}).get("rows", {})
    stats = tvm_e2e.get("runtime_stats", {})
    lines = [
        f"# Attention {precision} TVM No-Prune Ablation v1",
        "",
        f"No pruning, no quantization. TVM rows use {precision} Relax runtime.",
        "",
        "## E2E Model Forward",
        "",
        "| config | p50 ms | speedup | scope |",
        "|---|---:|---:|---|",
        f"| PyTorch {precision} no TVM | {e2e.get('baseline_no_tvm', {}).get('latency_p50_ms')} | 1.0x | model(batch['ego']) |",
        f"| TVM {precision} {e2e.get('tvm_scope')} | {tvm_e2e.get('latency_p50_ms')} | {e2e.get('speedup')}x | model(batch['ego']) with TVM attention subgraphs |",
        "",
        "## Attention/Fusion Subnet",
        "",
        f"| component | PyTorch {precision} p50 ms | TVM {precision} p50 ms | speedup |",
        "|---|---:|---:|---:|",
        f"| MSwin | {subnet.get('mswin', {}).get('pytorch', {}).get('p50_ms')} | {subnet.get('mswin', {}).get('tvm', {}).get('p50_ms')} | {subnet.get('mswin', {}).get('speedup')}x |",
        f"| HMSA static | {subnet.get('hmsa_static', {}).get('pytorch', {}).get('p50_ms')} | {subnet.get('hmsa_static', {}).get('tvm', {}).get('p50_ms')} | {subnet.get('hmsa_static', {}).get('speedup')}x |",
        f"| Subnet sum | {subnet.get('subnet_sum', {}).get('pytorch_p50_ms')} | {subnet.get('subnet_sum', {}).get('tvm_p50_ms')} | {subnet.get('subnet_sum', {}).get('speedup')}x |",
        "",
        "## TVM Runtime Evidence",
        "",
        f"- TVM scope: `{e2e.get('tvm_scope')}`",
        f"- TVM calls: total `{stats.get('total_tvm_call_count')}`, MSwin `{stats.get('mswin_call_count')}`, HMSA `{stats.get('hmsa_call_count')}`, fallback `{stats.get('fallback_call_count')}`",
        "",
        "Caveat: E2E latency excludes dataloading, post-process/NMS and AP metric aggregation. HMSA TVM uses a static two-agent specialization.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--tvm-scope", choices=("all", "mswin", "hmsa"), default="all")
    parser.add_argument("--latency-samples", type=int, default=30)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subnet-number", type=int, default=10)
    parser.add_argument("--subnet-repeat", type=int, default=5)
    parser.add_argument("--subnet-warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-subnet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    status = "OK"
    error = None
    tb = None
    try:
        from attention_tvm_mixed_int8_e2e_runner import bootstrap_h800_tvm_then_torch

        bootstrap = bootstrap_h800_tvm_then_torch()
        os.environ.setdefault("V2X_HEAL_ROOT", str(bootstrap.get("heal_root", "")))
        default_ckpt_dir = REPO_ROOT / "checkpoints/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26"
        if default_ckpt_dir.exists():
            os.environ.setdefault("V2XVIT_CKPT_DIR", str(default_ckpt_dir))
            os.environ.setdefault("V2XVIT_CONFIG_YAML", str(default_ckpt_dir / "config.yaml"))
            os.environ.setdefault("V2XVIT_CKPT_FILE", str(default_ckpt_dir / "net_epoch_bestval_at17.pth"))
        report: dict[str, Any] = {
            "schema_version": "attention_fp_tvm_no_prune_ablation_v1",
            "status": "OK",
            "pruning": "none",
            "quantization": "none",
            "precision": args.precision,
            "tvm_scope": args.tvm_scope,
            "bootstrap": bootstrap,
        }
        if not args.skip_subnet:
            report["subnet"] = run_subnet_ablation(
                device=args.device,
                precision=args.precision,
                number=args.subnet_number,
                repeat=args.subnet_repeat,
                warmup=args.subnet_warmup,
                seed=args.seed,
            )
        if not args.skip_e2e:
            report["e2e"] = run_e2e_ablation(
                device=args.device,
                precision=args.precision,
                tvm_scope=args.tvm_scope,
                latency_samples=args.latency_samples,
                latency_warmup=args.latency_warmup,
                num_workers=args.num_workers,
                log_dir=_resolve_repo_path(args.log_dir),
                bootstrap=bootstrap,
            )
            if report["e2e"].get("status") != "OK":
                report["status"] = "FAILED"
        report["elapsed_secs"] = round(time.time() - started, 3)
    except Exception as exc:  # pragma: no cover - H800 integration path
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()[-8000:]
        report = {
            "schema_version": "attention_fp_tvm_no_prune_ablation_v1",
            "status": status,
            "error": error,
            "traceback": tb,
            "elapsed_secs": round(time.time() - started, 3),
        }

    _write_json(args.out_json, report)
    if report.get("status") == "OK":
        render_csv(report, args.out_csv)
        render_markdown(report, args.out_md)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
