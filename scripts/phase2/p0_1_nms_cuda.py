"""P0.1 — CUDA rotated-NMS drop-in replacement for HEAL's Shapely `nms_rotated`.

Background (see methods/plan6 §五): HEAL `opencood/utils/box_utils.py:nms_rotated`
is a pure-Python + Shapely O(N^2) CPU loop that costs ~72% of PyramidFusion e2e
(77.6 ms mean, measured 2026-05-31). This module provides a numerically-equivalent
GPU replacement built on `mmcv.ops.nms_rotated`.

Equivalence argument:
  HEAL's `convert_format` builds a BEV polygon from the FIRST 4 corners' (x, y)
  of each box (shape (N,8,3) or (N,4,2)) and computes IoU = intersection/union.
  Those 4 corners come from `boxes_to_corners_3d` (a rigid rotation+translation of
  an axis-aligned rectangle), so they form an EXACT rectangle. Therefore the
  rectangle is losslessly describable as (cx, cy, w, h, angle), and mmcv's rotated
  IoU over that parametrization equals the Shapely polygon IoU up to float epsilon.

Drop-in contract (matches box_utils.nms_rotated):
  nms_rotated_gpu(boxes, scores, threshold) -> np.ndarray[int32]  (kept indices,
  score-descending order, capped at top=1000 like the original).
"""
from __future__ import annotations

import numpy as np
import torch
from mmcv.ops import nms_rotated as _mmcv_nms_rotated

# Match HEAL's hard cap: nms_rotated only considers the top-1000 scored boxes.
_TOP = 1000


def corners_to_rotated_bev(corners: torch.Tensor) -> torch.Tensor:
    """Convert box corners to mmcv rotated-box form (cx, cy, w, h, angle).

    Parameters
    ----------
    corners : torch.Tensor
        (N, 8, 3) or (N, 4, 2). Only the first 4 corners' (x, y) are used,
        exactly mirroring HEAL `common_utils.convert_format`.

    Returns
    -------
    torch.Tensor
        (N, 5) float tensor [cx, cy, w, h, angle(rad)].
    """
    c = corners[:, :4, :2].float()              # (N, 4, 2) BEV bottom face
    center = c.mean(dim=1)                       # (N, 2)
    edge_a = c[:, 1, :] - c[:, 0, :]             # one side
    edge_b = c[:, 3, :] - c[:, 0, :]             # adjacent (perpendicular) side
    w = edge_a.norm(dim=1)
    h = edge_b.norm(dim=1)
    angle = torch.atan2(edge_a[:, 1], edge_a[:, 0])
    return torch.stack([center[:, 0], center[:, 1], w, h, angle], dim=1)


def nms_rotated_gpu(boxes, scores, threshold, top: int = _TOP) -> np.ndarray:
    """GPU drop-in for box_utils.nms_rotated. Same args, same return type.

    Returns kept indices as np.int32, in score-descending order.
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)

    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes)
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)

    device = boxes.device if boxes.is_cuda else torch.device("cuda")
    boxes = boxes.to(device).float()
    scores = scores.to(device).float()

    n = boxes.shape[0]
    if n > top:
        # replicate HEAL's argsort()[::-1][:top] pre-filter, then map back
        top_scores, top_idx = scores.topk(top)
        dets = corners_to_rotated_bev(boxes[top_idx])
        _, keep = _mmcv_nms_rotated(dets, top_scores, float(threshold))
        kept = top_idx[keep]
    else:
        dets = corners_to_rotated_bev(boxes)
        _, keep = _mmcv_nms_rotated(dets, scores, float(threshold))
        kept = keep

    return kept.detach().cpu().numpy().astype(np.int32)


def patch_heal(verbose: bool = True):
    """Monkey-patch HEAL's box_utils.nms_rotated -> nms_rotated_gpu.

    Returns the original callable so callers can restore it.
    """
    from opencood.utils import box_utils

    original = box_utils.nms_rotated
    box_utils.nms_rotated = nms_rotated_gpu
    if verbose:
        print("[p0.1] patched box_utils.nms_rotated -> CUDA nms_rotated_gpu", flush=True)
    return original
