"""
BatchNorm folding utilities for UniV2X.
Ported from QuantV2X/opencood/quant/fold_bn.py.
Changes:
  - Removed spconv imports and _fold_bn_spconv branch (UniV2X has no spconv)
  - Removed specials_unquantized_names import from quant_block (avoids circular dep);
    search_fold_and_remove_bn now takes a skip_names set (default empty)
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    """
    BatchNorm folding for Conv2d, ConvTranspose2d, and Linear layers.
    """
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)

    if isinstance(conv_module, nn.Conv2d):
        return _fold_bn_conv(conv_module, bn_module, w, y_mean, safe_std)
    elif isinstance(conv_module, nn.ConvTranspose2d):
        return _fold_bn_transpose(conv_module, bn_module, w, y_mean, safe_std)
    elif isinstance(conv_module, nn.Linear):
        return _fold_bn_linear(conv_module, bn_module, w, y_mean, safe_std)
    else:
        raise TypeError(f"Unsupported module type {type(conv_module)} in BN folding")


def _fold_bn_conv(conv_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for standard Conv2d layers.
    """
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def _fold_bn_transpose(conv_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for ConvTranspose2d layers.
    """
    w_view = (1, conv_module.out_channels, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def _fold_bn_linear(linear_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for Linear layers.
    """
    w_view = (linear_module.out_features, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if linear_module.bias is not None:
            bias = bn_module.weight * linear_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if linear_module.bias is not None:
            bias = linear_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1 - module.eps)
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))


def search_fold_and_remove_bn(model, skip_names=None):
    """
    Recursively fold BN layers into preceding Conv/Linear layers and replace BN with StraightThrough.

    :param model: nn.Module to process in-place
    :param skip_names: set of child attribute names to skip (e.g. unquantized special blocks)
    """
    if skip_names is None:
        skip_names = set()
    model.eval()
    prev = None
    for n, m in model.named_children():
        if n in skip_names:
            continue
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m, skip_names)
    return prev


def search_fold_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
        else:
            search_fold_and_reset_bn(m)
        prev = m
