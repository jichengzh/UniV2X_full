"""
Quantization primitives for UniV2X.
Ported from QuantV2X/opencood/quant/quant_layer.py.
Changes:
  - Removed spconv imports (UniV2X does not use spconv)
  - Removed QuantSpconvModule class
  - Removed matplotlib import (debug visualizations stripped)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'mse',
                 leaf_param: bool = False, prob: float = 1.0,
                 group_size: int = -1):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        if self.sym:
            self.qmin = -(2 ** (self.n_bits - 1) - 1)
            self.qmax = 2 ** (self.n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = self.n_levels - 1
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''per-group quantization'''
        self.group_size = group_size
        self.use_group_quant = (group_size > 0)

        '''mse params'''
        self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def _quantize_per_group(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize with per-group granularity by reshaping to 2D and applying
        independent scale/zero_point to each group of ``group_size`` elements."""
        orig_shape = x.shape
        gs = self.group_size
        assert x.shape[-1] % gs == 0, (
            f"group_size {gs} does not divide last dim {x.shape[-1]}"
        )

        # Reshape to 2D: (total_groups, group_size)
        x_2d = x.reshape(-1, gs)

        if self.sym:
            abs_max = x_2d.abs().amax(dim=1, keepdim=True)  # (total_groups, 1)
            scale = abs_max / self.qmax
            scale = torch.max(scale, self.eps.to(scale.device))
            x_int = (x_2d / scale).round()
            x_q = x_int.clamp(self.qmin, self.qmax)
            x_dq = x_q * scale
        else:
            x_min = x_2d.amin(dim=1, keepdim=True)
            x_max = x_2d.amax(dim=1, keepdim=True)
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            scale = torch.max(scale, self.eps.to(scale.device))
            zp = self.qmin - (x_min / scale).round()
            zp = zp.clamp(self.qmin, self.qmax)
            x_int = (x_2d / scale).round() + zp
            x_q = x_int.clamp(self.qmin, self.qmax)
            x_dq = (x_q - zp) * scale

        # Store scale for later export
        self.delta = scale.squeeze(-1)  # (total_groups,)
        self.zero_point = (
            torch.zeros_like(self.delta) if self.sym else zp.squeeze(-1)
        )

        return x_dq.reshape(orig_shape)

    def forward(self, x: torch.Tensor):
        # Per-group quantization shortcut
        if self.use_group_quant and self.group_size > 0:
            x_dq = self._quantize_per_group(x)
            if self.is_training and self.prob < 1.0:
                x_dq = torch.where(torch.rand_like(x) < self.prob, x_dq, x)
            return x_dq

        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        if self.sym:
            x_int = round_ste(x / self.delta)
            x_quant = torch.clamp(x_int, self.qmin, self.qmax)
            x_dequant = x_quant * self.delta
        else:
            x_int = round_ste(x / self.delta) + self.zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta

        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        if self.sym:
            abs_max = torch.max(min_val.abs(), max_val.abs())
            scale = abs_max / self.qmax
            scale = torch.max(scale, self.eps)
            zero_point = torch.zeros_like(scale)
            return scale, zero_point
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        if self.sym:
            abs_max = torch.max(x_min.abs(), x_max.abs())
            delta = abs_max / self.qmax
            delta = torch.max(delta, self.eps)
            if self.channel_wise:
                new_shape = [1] * len(x.shape)
                new_shape[0] = x.shape[0]
                delta = delta.reshape(new_shape)
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int, self.qmin, self.qmax)
            x_float_q = x_quant * delta
            return x_float_q
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.sym:
            return self.perform_1D_search(x)
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        if self.scale_method == 'minmax':  # return x_min, x_max directly without the search
            return x_min, x_max
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        if self.scale_method == 'minmax':  # return x_min, x_max directly without the search
            return x_min, x_max
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method not in ['mse', 'minmax', 'entropy', 'percentile']:
            raise NotImplementedError
        if self.scale_method == 'entropy':
            best_min, best_max = self.perform_entropy_search(x)
        elif self.scale_method == 'percentile':
            best_min, best_max = self.perform_percentile_search(x)
        elif self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
            if self.one_side_dist != 'no' or self.sym:
                best_min, best_max = self.perform_1D_search(x)
            else:
                best_min, best_max = self.perform_2D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def _entropy_search_1d(self, x_flat, num_bins, num_quant_bins):
        """Core entropy calibration on a single 1-D tensor.

        Returns (best_min, best_max) as scalar tensors.
        """
        x_min, x_max = x_flat.min(), x_flat.max()
        hist = torch.histc(x_flat, bins=num_bins, min=x_min.item(), max=x_max.item())
        bin_width = (x_max - x_min) / num_bins

        best_kl = float('inf')
        best_min = x_min
        best_max = x_max

        # Try several clipping thresholds
        for i in range(num_bins // 2, num_bins):
            clip_max = x_min + bin_width * i
            clipped_hist = hist.clone()
            clipped_hist[i:] = clipped_hist[i - 1:].sum()  # tail into last bin

            # Rebin histogram into quant bins
            rebin_ratio = i // num_quant_bins
            if rebin_ratio < 1:
                continue
            n = num_quant_bins * rebin_ratio  # use only cleanly divisible prefix
            quant_hist = clipped_hist[:n].reshape(num_quant_bins, rebin_ratio).sum(dim=1)

            # Expand back to full precision bins
            expanded = quant_hist.repeat_interleave(rebin_ratio)
            if expanded.shape[0] < i:
                expanded = F.pad(expanded, (0, i - expanded.shape[0]))

            # Normalize both distributions
            p = clipped_hist[:i]
            q = expanded + 1e-6  # avoid zero
            p = p / p.sum()
            q = q / q.sum()

            # KL divergence
            kl_div = (p * (p / q).log()).sum()
            if kl_div < best_kl:
                best_kl = kl_div
                best_min = x_min
                best_max = clip_max

        return best_min, best_max

    def perform_entropy_search(self, x, num_bins=2048, num_quant_bins=None):
        if num_quant_bins is None:
            num_quant_bins = self.n_levels

        x = x.detach().float()
        if self.channel_wise:
            y = torch.flatten(x, 1)  # shape: [out_channels, -1]
            mins = []
            maxs = []
            for c in range(y.shape[0]):
                c_min, c_max = self._entropy_search_1d(y[c], num_bins, num_quant_bins)
                mins.append(c_min)
                maxs.append(c_max)
            best_min = torch.stack(mins)
            best_max = torch.stack(maxs)
            return best_min, best_max

        return self._entropy_search_1d(x, num_bins, num_quant_bins)

    def perform_percentile_search(self, x, percentile=99.99):
        """Clip to percentile range to reduce outlier impact on scale."""
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min = torch.quantile(y, (100.0 - percentile) / 100.0, dim=1)
            x_max = torch.quantile(y, percentile / 100.0, dim=1)
        else:
            x_min = torch.quantile(x.flatten(), (100.0 - percentile) / 100.0)
            x_max = torch.quantile(x.flatten(), percentile / 100.0)
        return x_min, x_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits
        if self.sym:
            self.qmin = -(2 ** (self.n_bits - 1) - 1)
            self.qmax = 2 ** (self.n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = self.n_levels - 1

    @torch.jit.export
    def extra_repr(self):
        s = 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )
        if self.use_group_quant:
            s += ', group_size={}'.format(self.group_size)
        return s


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.ConvTranspose2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.ConvTranspose2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   output_padding=org_module.output_padding, groups=org_module.groups,
                                   dilation=org_module.dilation)
            self.fwd_func = F.conv_transpose2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight).to(input.device)
            bias = self.bias.to(input.device) if self.bias is not None else None
        else:
            weight = self.org_weight.to(input.device)
            bias = self.org_bias.to(input.device) if self.org_bias is not None else None
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if type(self.norm_function) == nn.BatchNorm1d:
            out = self.norm_function(out.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            out = self.norm_function(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )
