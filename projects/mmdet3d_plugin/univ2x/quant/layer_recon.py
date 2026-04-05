"""
Layer-level AdaRound reconstruction for UniV2X.
Ported from QuantV2X/opencood/quant/layer_recon.py.
Changes:
  - LinearTempDecay defined inline (was in block_recon.py)
  - Imports from quant_params instead of separate set_*_params files
  - specials_unquantized replaced with an empty set (no spconv in UniV2X)
  - find_unquantized_module kept as-is but references local BaseQuantBlock
"""

import torch
import torch.nn.functional as F
from tqdm import trange

from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel, BaseQuantBlock
from .adaptive_rounding import AdaRoundQuantizer
from .quant_params import get_init, get_dc_fp_init, set_act_quantize_params

# Modules that are deliberately NOT quantized (extend this set as needed).
# Analogous to QuantV2X's specials_unquantized — empty for the base port.
specials_unquantized: set = set()


# ---------------------------------------------------------------------------
# Temperature decay scheduler (inlined from block_recon.py)
# ---------------------------------------------------------------------------

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


# ---------------------------------------------------------------------------
# Utility: find untrained QuantModules in traversal order
# ---------------------------------------------------------------------------

include = False


def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = [],
                            parent_name=""):
    """Store subsequent unquantized modules in a list with full hierarchical names."""
    global include
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, torch.nn.Sequential):
            find_unquantized_module(module, module_list, name_list, full_name)

        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False, False)
                name_list.append(full_name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(full_name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list, full_name)

    return module_list[1:], name_list[1:]


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def layer_reconstruction(model: QuantModel, fp_model: QuantModel,
                         layer: QuantModule, fp_layer: QuantModule,
                         cali_data: list, batch_size: int = 1, iters: int = 20000,
                         weight: float = 0.001, opt_mode: str = 'mse',
                         b_range: tuple = (20, 2), warmup: float = 0.0,
                         p: float = 2.0, lr: float = 4e-5, input_prob: float = 1.0,
                         keep_gpu: bool = True, lamb_r: float = 0.2, T: float = 7.0,
                         bn_lr: float = 1e-3, lamb_c: float = 0.02):
    """
    Reconstruction to optimize the output from each layer via AdaRound.

    :param model: quantized QuantModel (W+A quant)
    :param fp_model: full-precision QuantModel (used for DC-FP target)
    :param layer: QuantModule inside model to optimize
    :param fp_layer: corresponding QuantModule inside fp_model
    :param cali_data: list of calibration batches (DataLoader items)
    :param batch_size: mini-batch size for calibration data capture
    :param iters: AdaRound optimization iterations
    :param weight: rounding regularization weight
    :param opt_mode: reconstruction loss type ('mse')
    :param b_range: temperature annealing range (start_b, end_b)
    :param warmup: fraction of iters before temperature decay starts
    :param p: Lp norm for reconstruction loss
    :param lr: learning rate for activation delta
    :param input_prob: probability of using noisy (quantized) input vs clean
    :param keep_gpu: keep calibration tensors on GPU
    :param lamb_r: AdaRound regularization weight
    :param T: KL divergence temperature
    :param bn_lr: learning rate for distribution correction
    :param lamb_c: distribution correction constraint weight
    """

    # ---- gather calibration inputs and DC-corrected FP targets ----
    cached_inps = get_init(model, layer, cali_data, batch_size=batch_size,
                           input_prob=True, keep_gpu=keep_gpu)
    cached_outs, cached_output, cur_syms = get_dc_fp_init(
        fp_model, fp_layer, cali_data, batch_size=batch_size,
        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c)
    set_act_quantize_params(layer, cached_inps=cached_inps[:min(256, cached_inps.size(0))])

    # ---- set quantization state ----
    layer.set_quant_state(True, True)
    for para in model.parameters():
        para.requires_grad = False

    # ---- replace weight quantizer with AdaRound ----
    round_mode = 'learned_hard_sigmoid'
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    a_scheduler = None

    layer.weight_quantizer = AdaRoundQuantizer(
        uaq=layer.weight_quantizer,
        round_mode=round_mode,
        weight_tensor=layer.org_weight.data)
    layer.weight_quantizer.soft_targets = True
    w_para += [layer.weight_quantizer.alpha]

    # ---- activation delta as learnable parameter ----
    if layer.act_quantizer.delta is not None:
        layer.act_quantizer.delta = torch.nn.Parameter(
            torch.tensor(layer.act_quantizer.delta))
        a_para += [layer.act_quantizer.delta]
    layer.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt, T_max=iters, eta_min=0.)

    loss_func = LossFunction(layer, round_loss='relaxation', weight=weight,
                             max_count=iters, rec_loss=opt_mode,
                             b_range=b_range, decay_start=0, warmup=warmup,
                             p=p, lam=lamb_r, T=T)
    device = 'cuda'
    sz = cached_inps.size(0)

    for i in range(iters):
        idx = torch.randint(0, sz, () if isinstance(cached_inps, torch.Tensor) else (1,))
        cur_inp = cached_inps[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        output_fp = cached_output[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        else:
            drop_inp = cur_inp

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()

        out_drop = layer(drop_inp)
        out_quant = layer(cur_inp)
        err = loss_func(out_drop, cur_out, out_quant, output_fp)

        err.backward(retain_graph=True)
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if a_scheduler:
            a_scheduler.step()

    torch.cuda.empty_cache()
    layer.weight_quantizer.soft_targets = False
    layer.act_quantizer.is_training = False
    layer.trained = True


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class LossFunction:
    def __init__(self,
                 block: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count,
                                          rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.hetero_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def __call__(self, pred, tgt, output, output_fp):
        """
        :param pred: output from quantized layer (noisy input)
        :param tgt: DC-FP corrected output (target)
        :param output: quantized layer output (clean input)
        :param output_fp: full-precision model output
        :return: total loss
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 200 == 0:
            print('Total loss:\t{:.5f} (rec:{:.5f}, round:{:.5f})\tb={:.5f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
