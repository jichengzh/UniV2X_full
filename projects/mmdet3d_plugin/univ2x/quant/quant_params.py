"""
Weight / activation quantization parameter helpers for UniV2X.
Ported from QuantV2X/opencood/quant/set_weight_quantize_params.py and
set_act_quantize_params.py.
Changes:
  - Removed QuantSpconvModule references (UniV2X has no spconv)
  - Both helpers merged into one file to avoid excessive small files
"""

import torch
from typing import Union
from .quant_layer import QuantModule
from .quant_model import QuantModel, BaseQuantBlock
from .data_utils import save_inp_oup_data, save_dc_fp_data


# ---------------------------------------------------------------------------
# Weight-side helpers
# ---------------------------------------------------------------------------

def get_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool = True):
    cached_inps = save_inp_oup_data(model, block, cali_data, batch_size,
                                    input_prob=input_prob, keep_gpu=keep_gpu)
    return cached_inps


def get_dc_fp_init(model, block, cali_data, batch_size, input_prob: bool = False,
                   keep_gpu: bool = True, lamb=50, bn_lr=1e-3):
    cached_outs, cached_outputs, cached_sym = save_dc_fp_data(
        model, block, cali_data, batch_size,
        input_prob=input_prob, keep_gpu=keep_gpu, lamb=lamb, bn_lr=bn_lr)
    return cached_outs, cached_outputs, cached_sym


def set_weight_quantize_params(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)
            module.weight_quantizer(module.weight)
            module.weight_quantizer.set_inited(True)


def save_quantized_weight(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight.data = module.weight_quantizer(module.weight)


# ---------------------------------------------------------------------------
# Activation-side helpers
# ---------------------------------------------------------------------------

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cached_inps: Union[list, torch.Tensor],
                            batch_size: int = 8):
    module.set_quant_state(True, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(False)

    if isinstance(cached_inps, torch.Tensor):
        with torch.no_grad():
            for i in range(cached_inps.size(0)):
                module(cached_inps[i].cuda())
    else:
        for i in range(0, min(len(cached_inps), batch_size)):
            with torch.no_grad():
                module(cached_inps[i].cuda())

    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(True)
