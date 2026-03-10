#-------------------------------------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                                      #
# Copyright (c) DAIR-V2X. All rights reserved.                                                    #
#-------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class LaneQueryFusion(nn.Module):
    def __init__(self, pc_range, embed_dims=256):
        super(LaneQueryFusion, self).__init__()

        self.pc_range = pc_range
        self.embed_dims = embed_dims

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _lidar2norm(self, locs, pc_range, norm_mode='sigmoid'):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        if norm_mode == 'inverse_sigmoid':
            locs = inverse_sigmoid(locs)

        return locs
    
    def _norm2lidar(self, ref_pts, pc_range, norm_mode='sigmoid'):
        """
        normalized (x,y) ---> absolute (x,y) in inf lidar coordinate system
        """
        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))
        if norm_mode == 'inverse_sigmoid':
            locs = ref_pts.sigmoid().clone()
        else:
            locs = ref_pts.clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs


    def filter_other_lanes(self, cls_score, threshold=0.05, num_things_classes=3):
        '''
        refer to _get_bboxes_single in panseg_head.py
        cls_score = other_outputs_classes[-1, bs]
        '''
        cls_score = cls_score.sigmoid()
        indexes = list(torch.where(cls_score.view(-1) > threshold))[0]
        det_labels = indexes % num_things_classes
        bbox_index = indexes // num_things_classes
        
        return bbox_index

    def forward(self, other_outputs_classes, other_outputs_coords, other_query, other_query_pos, other_reference,
                                    veh_outputs_classes, veh_outputs_coords, veh_query, veh_query_pos, veh_reference,
                                    ego2other_rt, other_agent_pc_range, threshold=0.05):
        '''
        reference: (x, y, w, h), reference = inverse_sigmoid(reference)
        outputs_coords: (x, y, w, h), outputs_coords = outputs_coords.sigmoid()
        '''
        calib_other2ego = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_other2ego = torch.tensor(calib_other2ego).to(other_query)

        # UniV2X TODO: hardcode for filtering inf queries with scores
        # UniV2X TODO: supposed that img num = 1
        other_cls_scores = other_outputs_classes[-1]
        other_bbox_index = self.filter_other_lanes(other_cls_scores, threshold=threshold)
        other_outputs_classes = other_outputs_classes[:, :, other_bbox_index, :]
        other_outputs_coords = other_outputs_coords[:, :, other_bbox_index, :]
        other_query = other_query[:, other_bbox_index, :]
        other_query_pos = other_query_pos[:, other_bbox_index, :]
        other_reference = other_reference[:, other_bbox_index, :]

        # other_reference: other2ego
        other_ref_pts = torch.zeros(other_reference.shape[0],
                                                            other_reference.shape[1],
                                                            3).to(other_query)
        other_ref_pts[..., :2] = other_reference[..., :2]
        for ii in range(other_ref_pts.shape[0]):
            other_ref_pts[ii] = self._norm2lidar(other_ref_pts[ii], other_agent_pc_range, norm_mode='inverse_sigmoid')
            other_ref_tmp = torch.cat((other_ref_pts[ii], torch.ones_like(other_ref_pts[ii][..., :1])), -1).unsqueeze(-1)
            other_ref_pts[ii] = torch.matmul(calib_other2ego, other_ref_tmp).squeeze(-1)[..., :3]

            other_ref_pts[ii] = self._lidar2norm(other_ref_pts[ii], self.pc_range, norm_mode='inverse_sigmoid')
        other_reference[..., :2] = other_ref_pts[..., :2]

        # other_bboxes: other2ego
        other_bboxes = torch.zeros(other_outputs_coords.shape[0],
                                                        other_outputs_coords.shape[1],
                                                        other_outputs_coords.shape[2],
                                                        3).to(other_query)
        other_bboxes[..., :2] = other_outputs_coords[..., :2]
        for ii in range(other_bboxes.shape[0]):
            for jj in range(other_bboxes.shape[1]):
                other_bboxes[ii, jj] = self._norm2lidar(other_bboxes[ii, jj], other_agent_pc_range, norm_mode='sigmoid')
                other_ref_tmp = torch.cat((other_bboxes[ii, jj], torch.ones_like(other_bboxes[ii, jj][..., :1])), -1).unsqueeze(-1)
                other_bboxes[ii, jj] = torch.matmul(calib_other2ego, other_ref_tmp).squeeze(-1)[..., :3]

                other_bboxes[ii, jj] = self._lidar2norm(other_bboxes[ii, jj], self.pc_range, norm_mode='sigmoid')
        other_outputs_coords[..., :2] = other_bboxes[..., :2]

        # cross-agent feature alignment
        for ii in range(other_query.shape[0]):
            inf2veh_r = calib_other2ego[:3,:3].reshape(1,9).repeat(other_query[ii].shape[0], 1)
            other_query[ii] = self.cross_agent_align(torch.cat([other_query[ii], inf2veh_r], -1))
            other_query_pos[ii] = self.cross_agent_align_pos(torch.cat([other_query_pos[ii], inf2veh_r], -1))

        # UniV2X TODO: directly concat other-agent queries and veh queries
        # UniV2X TODO: supposed that img num = 1
        other_outputs_classes = torch.cat((veh_outputs_classes, other_outputs_classes), dim=2)
        other_outputs_coords = torch.cat((veh_outputs_coords, other_outputs_coords), dim=2)
        other_query = torch.cat((veh_query, other_query), dim=1)
        other_query_pos = torch.cat((veh_query_pos, other_query_pos), dim=1)
        other_reference = torch.cat((veh_reference, other_reference), dim=1)

        return other_outputs_classes, other_outputs_coords, other_query, other_query_pos, other_reference


class LaneQueryFusionTRT(LaneQueryFusion):
    """TRT-compatible variant of LaneQueryFusion.

    Key changes vs the original:

    1. ``np.linalg.inv(ego2other_rt)`` → pre-computed on CPU before calling,
       passed as ``calib_other2ego`` (4×4 float tensor).
    2. ``filter_other_lanes`` (dynamic boolean index) → done on CPU before
       calling; inputs are already filtered to fixed size.
    3. Python ``for`` loops over layers/images → replaced with vectorised
       batch-matmul (no Python control-flow in the traced graph).

    CPU preprocessing before calling ``forward_trt``
    -------------------------------------------------
    ::

        calib = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_t = torch.tensor(calib, dtype=dtype, device=device)
        bbox_index = fusion.filter_other_lanes(other_outputs_classes[-1])
        other_query       = other_query[:, bbox_index, :]
        other_query_pos   = other_query_pos[:, bbox_index, :]
        other_reference   = other_reference[:, bbox_index, :]
        other_outputs_cls = other_outputs_classes[:, :, bbox_index, :]
        other_outputs_crd = other_outputs_coords[:, :, bbox_index, :]
        results = fusion.forward_trt(other_query, ..., calib_t,
                                     other_agent_pc_range_t)
    """

    def __init__(self, pc_range, other_pc_range, embed_dims=256):
        super().__init__(pc_range=pc_range, embed_dims=embed_dims)
        # Store the infra model's pc_range as a buffer so ONNX can fold it
        # as a constant.
        self.register_buffer(
            'other_pc_range_buf',
            torch.tensor(other_pc_range, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Helpers (vectorised, no Python loops)
    # ------------------------------------------------------------------

    @staticmethod
    def _norm2lidar_vec(pts, pc_range, sigmoid_first: bool):
        """Vectorised denormalisation (any leading dims).

        Args:
            pts:          (..., 3) normalised coordinates.
            pc_range:     (6,) tensor.
            sigmoid_first: True  → apply sigmoid before denorm (inverse_sigmoid input).
                           False → pts are already in [0,1] (sigmoid input).
        """
        if sigmoid_first:
            pts = pts.sigmoid()
        else:
            pts = pts.clone()
        pts = pts * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
        return pts

    @staticmethod
    def _lidar2norm_vec(pts, pc_range, apply_inverse_sigmoid: bool):
        """Vectorised normalisation (any leading dims)."""
        from mmdet.models.utils.transformer import inverse_sigmoid
        pts = (pts - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])
        if apply_inverse_sigmoid:
            pts = inverse_sigmoid(pts)
        return pts

    # ------------------------------------------------------------------
    # TRT-compatible forward
    # ------------------------------------------------------------------

    def forward_trt(self,
                    other_query, other_query_pos, other_reference,
                    veh_query, veh_query_pos, veh_reference,
                    veh_outputs_classes, veh_outputs_coords,
                    other_outputs_classes, other_outputs_coords,
                    calib_other2ego):
        """TRT-compatible forward (all inputs pre-filtered, calib pre-computed).

        Args:
            other_query:            (L, N_inf, C)
            other_query_pos:        (L, N_inf, C)
            other_reference:        (L, N_inf, 4)  inverse-sigmoid normalised
            veh_query:              (L, N_veh, C)
            veh_query_pos:          (L, N_veh, C)
            veh_reference:          (L, N_veh, 4)
            veh_outputs_classes:    (num_dec, L, N_veh, num_cls)
            veh_outputs_coords:     (num_dec, L, N_veh, 4)
            other_outputs_classes:  (num_dec, L, N_inf, num_cls)  pre-filtered
            other_outputs_coords:   (num_dec, L, N_inf, 4)        pre-filtered
            calib_other2ego:        (4, 4)  pre-computed float tensor

        Returns:
            Tuple matching ``LaneQueryFusion.forward()`` return signature.
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        pc_ego   = other_query.new_tensor(self.pc_range)          # (6,)
        pc_other = self.other_pc_range_buf.to(other_query)        # (6,)

        L, N_inf, _ = other_query.shape

        # ── 1. Transform other_reference: other→ego ──────────────────────
        # other_reference[..., :2] are in inverse-sigmoid space; z=0
        other_ref_pts = other_reference.new_zeros(L, N_inf, 3)
        other_ref_pts[..., :2] = other_reference[..., :2]

        locs = self._norm2lidar_vec(other_ref_pts, pc_other, sigmoid_first=True)

        ones = locs.new_ones(L, N_inf, 1)
        pts_h = torch.cat([locs, ones], dim=-1)                   # (L, N, 4)
        pts_ego = torch.matmul(
            calib_other2ego,
            pts_h.reshape(L * N_inf, 4).unsqueeze(-1)             # (L*N, 4, 1)
        ).squeeze(-1).reshape(L, N_inf, 4)[..., :3]               # (L, N, 3)

        pts_ego = self._lidar2norm_vec(pts_ego, pc_ego, apply_inverse_sigmoid=True)

        other_reference = other_reference.clone()
        other_reference[..., :2] = pts_ego[..., :2]

        # ── 2. Transform other_outputs_coords: other→ego ─────────────────
        num_dec = other_outputs_coords.shape[0]
        other_bboxes = other_outputs_coords.new_zeros(num_dec, L, N_inf, 3)
        other_bboxes[..., :2] = other_outputs_coords[..., :2]

        bbox_locs = self._norm2lidar_vec(other_bboxes, pc_other, sigmoid_first=False)

        ones_b = bbox_locs.new_ones(num_dec, L, N_inf, 1)
        pts_h_b = torch.cat([bbox_locs, ones_b], dim=-1)          # (D, L, N, 4)
        D = num_dec
        pts_ego_b = torch.matmul(
            calib_other2ego,
            pts_h_b.reshape(D * L * N_inf, 4).unsqueeze(-1)
        ).squeeze(-1).reshape(D, L, N_inf, 4)[..., :3]

        pts_ego_b = self._lidar2norm_vec(pts_ego_b, pc_ego, apply_inverse_sigmoid=False)

        other_outputs_coords = other_outputs_coords.clone()
        other_outputs_coords[..., :2] = pts_ego_b[..., :2]

        # ── 3. Cross-agent feature alignment (vectorised) ─────────────────
        # inf2veh rotation: (1, 1, 9) → broadcast to (L, N_inf, 9)
        inf2veh_r = calib_other2ego[:3, :3].reshape(1, 1, 9).expand(L, N_inf, 9)
        other_query = self.cross_agent_align(
            torch.cat([other_query, inf2veh_r], dim=-1))
        other_query_pos = self.cross_agent_align_pos(
            torch.cat([other_query_pos, inf2veh_r], dim=-1))

        # ── 4. Concat with vehicle queries ───────────────────────────────
        out_cls   = torch.cat([veh_outputs_classes, other_outputs_classes], dim=2)
        out_coord = torch.cat([veh_outputs_coords,  other_outputs_coords],  dim=2)
        out_q     = torch.cat([veh_query,    other_query],    dim=1)
        out_qp    = torch.cat([veh_query_pos, other_query_pos], dim=1)
        out_ref   = torch.cat([veh_reference, other_reference], dim=1)

        return out_cls, out_coord, out_q, out_qp, out_ref