"""
Calibration data capture utilities for UniV2X.
Ported from QuantV2X/opencood/quant/data_utils.py.
Changes:
  - Removed `from opencood.tools import train_utils`; replaced train_utils.to_device()
    with a local _to_device() that handles dict / tensor / list inputs
  - Removed output_fp['reg_preds'] key assumption in GetDcFpLayerInpOut.__call__;
    full model output is stored directly (callers that need a specific key can slice it)
  - cali_data items are expected to be whatever the UniV2X DataLoader yields (dicts);
    GetLayerInpOut.__call__ passes them straight to model.forward()
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from typing import Union

from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel, BaseQuantBlock
from tqdm import trange


# ---------------------------------------------------------------------------
# Utility: move arbitrary nested structure to device
# ---------------------------------------------------------------------------

def _to_device(data, device):
    """Recursively move tensors inside dict/list/tuple/Tensor to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        moved = [_to_device(v, device) for v in data]
        return type(data)(moved)
    return data  # leave non-tensor scalars etc. as-is


# ---------------------------------------------------------------------------
# Core data-capture functions
# ---------------------------------------------------------------------------

def save_dc_fp_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: list,
                    batch_size: int = 32, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """Activation after distribution-correction (DC)."""
    device = torch.device('cuda')
    get_inp_out = GetDcFpLayerInpOut(model, layer, device=device,
                                     input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    print("Start correcting {} batches of data!".format(len(cali_data)))
    for i in range(len(cali_data)):
        if input_prob:
            cur_out, out_fp, cur_sym = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym.cpu()))
        else:
            cur_out, out_fp = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu()))
    cached_outs = torch.cat([x[0] for x in cached_batches])
    cached_outputs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_outs = cached_outs.to(device)
        cached_outputs = cached_outputs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        cached_outs.requires_grad = False
        cached_sym.requires_grad = False
        return cached_outs, cached_outputs, cached_sym
    return cached_outs, cached_outputs


def save_inp_oup_data(model: QuantModel, block: Union[QuantModule, BaseQuantBlock], cali_data: list,
                      batch_size: int = 1, keep_gpu: bool = True,
                      input_prob: bool = False):
    """
    Save input data of a particular layer/block over the calibration dataset.

    :param model: QuantModel
    :param block: QuantModule or BaseQuantBlock
    :param cali_data: calibration data set (list — each element is what the DataLoader yields)
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input data (tensor or list)
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, block, device=device, input_prob=input_prob)
    cached_batches = []
    is_dict = False
    channel_sizes = []

    for i in range(len(cali_data)):
        cur_inp = get_inp_out(cali_data[i])
        if cur_inp is None:
            continue
        if type(cur_inp) not in [torch.Tensor, dict]:
            continue
        if isinstance(cur_inp, dict):
            cached_batches.append(cur_inp)
            is_dict = True
        else:
            channel_sizes.append(cur_inp.size(0))
            cur_inp = cur_inp.unsqueeze(0)
            cached_batches.append(cur_inp.cpu())
    if len(cached_batches) == 0:
        raise RuntimeError("No valid calibration inputs for this layer; check calibration data.")
    if is_dict or len(set(channel_sizes)) != 1:
        torch.cuda.empty_cache()
        return cached_batches
    else:
        cached_inps = torch.cat([x for x in cached_batches])
        torch.cuda.empty_cache()
        if keep_gpu:
            cached_inps = cached_inps.to(device)
        return cached_inps


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


class DataSaverHook:
    """Forward hook that stores the input and output of a block."""

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class input_hook:
    """Forward hook used to get the output of an intermediate layer."""

    def __init__(self, stop_forward=False):
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input

    def clear(self):
        self.inputs = None


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            self.model.set_quant_state(weight_quant=True, act_quant=True)
            try:
                model_input = _to_device(model_input, self.device)
                _ = self.model(model_input)
            except StopForwardException:
                pass

        handle.remove()
        if self.data_saver.input_store is None:
            return None
        if len(self.data_saver.input_store) == 0:
            return None
        if type(self.data_saver.input_store[0]) == dict:
            return self.data_saver.input_store[0]
        return self.data_saver.input_store[0].detach()


class GetDcFpLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False, lamb=50, bn_lr=1e-3):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        self.input_prob = input_prob
        self.bn_stats = []
        self.eps = 1e-6
        self.lamb = lamb
        self.bn_lr = bn_lr
        for n, m in self.layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_stats.append(
                    (m.running_mean.detach().clone().flatten().cuda(),
                     torch.sqrt(m.running_var + self.eps).detach().clone().flatten().cuda()))

    def own_loss(self, A, B):
        return (A - B).norm() ** 2 / B.size(0)

    def relative_loss(self, A, B):
        return (A - B).abs().mean() / A.abs().mean()

    def __call__(self, model_input):
        self.model.set_quant_state(False, False)
        handle = self.layer.register_forward_hook(self.data_saver)
        hooks = []
        hook_handles = []
        for name, module in self.layer.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = input_hook()
                hooks.append(hook)
                hook_handles.append(module.register_forward_hook(hook.hook))
        assert len(hooks) == len(self.bn_stats)

        with torch.no_grad():
            try:
                model_input = _to_device(model_input, self.device)
                output_fp = self.model(model_input)
                # NOTE: Unlike QuantV2X which accessed output_fp['reg_preds'],
                # we keep output_fp as-is. Callers that need a specific key should
                # slice after this function returns, or override this method.
            except StopForwardException:
                pass
            if self.data_saver.input_store is None:
                handle.remove()
                return None
            if len(self.data_saver.input_store) == 0:
                handle.remove()
                return None
            if self.input_prob:
                if isinstance(self.data_saver.input_store[0], dict):
                    input_sym = self.data_saver.input_store[0]
                else:
                    input_sym = self.data_saver.input_store[0].detach()

        handle.remove()
        para_input = input_sym.data.clone()
        para_input = _to_device(para_input, self.device)
        para_input.requires_grad = True
        optimizer = optim.Adam([para_input], lr=self.bn_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-5,
                                                         verbose=False,
                                                         patience=100)
        iters = 500
        for iter in range(iters):
            self.layer.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            _ = self.layer(para_input)
            mean_loss = 0
            std_loss = 0
            for num, (bn_stat, hook) in enumerate(zip(self.bn_stats, hooks)):
                tmp_input = hook.inputs[0]
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_input.view(tmp_input.size(0),
                                                     tmp_input.size(1), -1), dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_input.view(tmp_input.size(0),
                                             tmp_input.size(1), -1), dim=2) + self.eps)
                mean_loss += self.own_loss(bn_mean, tmp_mean)
                std_loss += self.own_loss(bn_std, tmp_std)
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb
            total_loss = mean_loss + std_loss + constraint_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            if (iter + 1) % 500 == 0:
                print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
                    float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))

        with torch.no_grad():
            out_fp = self.layer(para_input)

        out_fp = out_fp.unsqueeze(0)
        if self.input_prob:
            para_input = para_input.unsqueeze(0)

        if self.input_prob:
            return out_fp.detach(), output_fp.detach(), para_input.detach()
        return out_fp.detach(), output_fp.detach()
