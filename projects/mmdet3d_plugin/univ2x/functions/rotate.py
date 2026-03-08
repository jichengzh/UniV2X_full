"""ONNX symbolic wrapper for RotatePlugin (image rotation TRT plugin).

During torch.onnx.export() the symbolic() emits a RotatePlugin ONNX node.
Otherwise falls back to torchvision grid_sampler-based rotation.
"""
import numpy as np
import torch
from torch.autograd import Function


_MODE = {"bilinear": 0, "nearest": 1}


class _RotatePlugin(Function):
    @staticmethod
    def symbolic(g, img, angle, center, interpolation):
        return g.op("custom_op::RotatePlugin", img, angle, center,
                    interpolation_i=interpolation)

    @staticmethod
    def forward(ctx, img, angle, center, interpolation):
        assert img.ndim == 3
        oh, ow = img.shape[-2:]
        if isinstance(center, (list, tuple)):
            center = torch.FloatTensor(center).to(img.device)
        cx = center[0] - center[0].new_tensor(ow * 0.5)
        cy = center[1] - center[1].new_tensor(oh * 0.5)
        if isinstance(angle, float):
            angle_ = torch.FloatTensor([angle]).to(img.device)
        else:
            angle_ = angle.to(img.device).float()

        angle_ = -angle_ * np.pi / 180
        theta = torch.stack([
            torch.cos(angle_),
            torch.sin(angle_),
            -cx * torch.cos(angle_) - cy * torch.sin(angle_) + cx,
            -torch.sin(angle_),
            torch.cos(angle_),
            cx * torch.sin(angle_) - cy * torch.cos(angle_) + cy,
        ]).view(1, 2, 3)

        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
        base_grid[..., 0] = torch.linspace(
            -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
        ).expand(1, oh, ow)
        base_grid[..., 1] = torch.linspace(
            -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
        ).unsqueeze_(-1).expand(1, oh, ow)
        base_grid[..., 2].fill_(1)

        rescaled_theta = 2 * theta.transpose(1, 2)
        rescaled_theta[..., 0] /= ow
        rescaled_theta[..., 1] /= oh

        grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta).view(1, oh, ow, 2)
        grid = grid.expand(img.unsqueeze(0).shape[0], oh, ow, 2)
        img = torch.grid_sampler(img.unsqueeze(0), grid, interpolation, 0, False)
        return img.squeeze(0)


_rotate_plugin = _RotatePlugin.apply


def rotate(img, angle, center, interpolation="nearest"):
    """Rotate image; emits RotatePlugin ONNX node during export."""
    if torch.onnx.is_in_onnx_export():
        center_t = torch.cuda.FloatTensor(center).to(img.device)
        return _rotate_plugin(img, angle, center_t, _MODE[interpolation])
    from torchvision.transforms.functional import rotate as tv_rotate, InterpolationMode
    return tv_rotate(img, angle.item(), center=center,
                     interpolation=InterpolationMode.NEAREST)
