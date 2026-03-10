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

from ..dense_heads.track_head_plugin import Instances


class AgentQueryFusion(nn.Module):

    def __init__(self, pc_range, embed_dims=256):
        super(AgentQueryFusion, self).__init__()

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
    
    def _loc_norm(self, locs, pc_range):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        locs = inverse_sigmoid(locs)

        return locs
    
    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        locs = ref_pts.sigmoid().clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _dis_filt(self, veh_pts, inf_pts, veh_dims):
        """
        filter according to distance
        """
        diff = torch.abs(veh_pts - inf_pts) / veh_dims
        return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1
    
    def _query_matching(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums))
        cost_matrix.fill(1e6)

        for i in veh_mask:
            # for j in range(i,inf_nums):
            for j in range(inf_nums):
                cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
                if not self._dis_filt(veh_ref_pts[i], inf_ref_pts[j], veh_pred_dims[i]):
                    cost_matrix[i][j] = 1e6
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        return idx_veh, idx_inf, cost_matrix
    
    def _query_fusion(self, inf, veh, inf_idx, veh_idx, cost_matrix):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """

        veh_accept_idx = []
        inf_accept_idx = []

        for i in range(len(veh_idx)):
            if cost_matrix[veh_idx[i]][inf_idx[i]] < 1e5:
                veh_accept_idx.append(veh_idx[i])
                inf_accept_idx.append(inf_idx[i])
                veh.query[veh_idx[i], self.embed_dims:] = veh.query[veh_idx[i], self.embed_dims:] + self.cross_agent_fusion(inf.query[inf_idx[i], self.embed_dims:])
        
        return veh, veh_accept_idx, inf_accept_idx
    

    def _query_complementation(self, inf, veh, inf_accept_idx):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        # supply_idx = -1
        for i in range(inf.ref_pts.shape[0]):
            if i not in inf_accept_idx:
                veh = Instances.cat([veh, inf[i]])

        return veh

    
    def forward(self, inf, veh, ego2other_rt, other_agent_pc_range, threshold=0.3):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        ego2other_rt: calibration parameters from infrastructure to vehicle
        """
        inf_mask = torch.where(inf.obj_idxes>=0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh
        inf_mask_new = torch.where(inf.obj_idxes>=0)
        
        #not care obj_idxes of inf
        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1
                
        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, other_agent_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
            
        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # ego_selection
        remove_ego_ins = True
        if remove_ego_ins:
            H_B, H_F = -2.04, 2.04 # H = 4.084
            W_L, W_R = -0.92, 0.92 # W = 1.85
            def del_tensor_ele(arr,index):
                arr1 = arr[0:index]
                arr2 = arr[index+1:]
                return torch.cat((arr1,arr2),dim=0)

            inf_mask_new = list(inf_mask_new)
            for ii in range(len(inf_ref_pts)):
                xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
                if xx >= H_B and xx <= H_F and yy >= W_L and yy <= W_R:
                    inf_mask_new[0] = del_tensor_ele(inf_mask_new[0], ii)
                    break
            inf_mask_new = tuple(inf_mask_new)
            inf = inf[inf_mask_new]
            inf_ref_pts = inf_ref_pts[inf_mask_new]

        # matching
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching(inf_ref_pts, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2,3,5]]) # veh.pred_boxes x,y,dx,dy,z,dz

        # ref_pts normalization
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.pc_range)
        veh_ref_pts = self._loc_norm(veh_ref_pts, self.pc_range)
        inf.ref_pts = inf_ref_pts
        veh.ref_pts = veh_ref_pts

        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query.shape[0], 1)
        inf.query[..., :self.embed_dims] = self.cross_agent_align_pos(torch.cat([inf.query[..., :self.embed_dims],inf2veh_r], -1))
        inf.query[..., self.embed_dims:] = self.cross_agent_align(torch.cat([inf.query[..., self.embed_dims:],inf2veh_r], -1))

        # cross-agent query fusion
        veh, veh_accept_idx, inf_accept_idx = self._query_fusion(inf, veh, inf_idx, veh_idx, cost_matrix)

        # cross-agent query complementation
        veh = self._query_complementation(inf, veh, inf_accept_idx)

        return veh


class AgentQueryFusionTRT(AgentQueryFusion):
    """TRT-compatible variant of AgentQueryFusion.

    Key changes vs the original:
    1. ego_selection: vectorised boolean mask (removes first ego instance)
       instead of Python for-loop + del_tensor_ele.
    2. _query_fusion: vectorised index_add on query tensor instead of
       Python for-loop over matched pairs.
    3. CPU preprocessing (scipy Hungarian) remains on CPU — identical to
       original.

    Usage (drop-in replacement for forward())::

        veh = fusion_trt.forward_trt(inf, veh, ego2other_rt, other_pc_range)
    """

    N_EGO = 901
    N_INF_MAX = 200
    N_TOTAL = 1101  # N_EGO + N_INF_MAX

    def _query_matching_vec(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """Vectorised cost-matrix build (GPU) + scipy Hungarian (CPU).

        Replaces the O(N_veh × N_inf) Python for-loop in ``_query_matching``
        with a single batched GPU operation followed by one GPU→CPU transfer.

        Args:
            inf_ref_pts:   (N_inf, 3) absolute xyz in veh frame.
            veh_ref_pts:   (N_veh, 3)
            veh_mask:      1-D LongTensor of active veh indices (scores≥0.05).
            veh_pred_dims: (N_veh, 3) predicted dx/dy/dz.

        Returns:
            idx_veh, idx_inf, cost_matrix  (same as _query_matching)
        """
        veh_nums = veh_ref_pts.shape[0]
        inf_nums = inf_ref_pts.shape[0]

        # Start with all-1e6 cost matrix
        cost_gpu = veh_ref_pts.new_full((veh_nums, inf_nums), 1e6)

        if len(veh_mask) > 0:
            # Active veh pts: (M, 3), inf pts: (N_inf, 3)
            veh_pts = veh_ref_pts[veh_mask]              # (M, 3)
            dims    = veh_pred_dims[veh_mask]             # (M, 3)

            # Pairwise differences: (M, N_inf, 3)
            diff = veh_pts.unsqueeze(1) - inf_ref_pts.unsqueeze(0)

            # L2 distance: (M, N_inf)
            l2 = diff.pow(2).sum(-1).sqrt()

            # dis_filt: |diff| / veh_dims <= 1 for all 3 axes
            rel = diff.abs() / dims.unsqueeze(1).clamp(min=1e-6)   # (M, N_inf, 3)
            keep = rel.le(1.0).all(dim=-1)                           # (M, N_inf) bool

            cost_active = torch.where(keep, l2,
                                      l2.new_full((), 1e6))          # (M, N_inf)
            cost_gpu[veh_mask] = cost_active

        cost_matrix = cost_gpu.cpu().numpy()
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)
        return idx_veh, idx_inf, cost_matrix

    def forward(self, inf, veh, ego2other_rt, other_agent_pc_range,
                threshold=0.3):
        """Override forward to call forward_trt (drop-in replacement)."""
        return self.forward_trt(inf, veh, ego2other_rt, other_agent_pc_range,
                                threshold)

    def forward_trt(self, inf, veh, ego2other_rt, other_agent_pc_range,
                    threshold=0.3):
        """TRT-friendly forward: CPU matching + vectorised GPU MLPs.

        Semantics identical to AgentQueryFusion.forward().
        """
        inf_mask = torch.where(inf.obj_idxes >= 0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh

        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1

        # ref_pts norm → absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, other_agent_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)

        # inf → veh coordinate transform
        calib_inf2veh = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_inf2veh_t = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts_h = torch.cat(
            (inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1
        ).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh_t,
                                   inf_ref_pts_h).squeeze(-1)[..., :3]

        # ego_selection: vectorised — remove first infra instance inside
        # the ego vehicle bounding box (matches original loop + break logic)
        H_B, H_F, W_L, W_R = -2.04, 2.04, -0.92, 0.92
        in_ego = (
            (inf_ref_pts[:, 0] >= H_B) & (inf_ref_pts[:, 0] <= H_F) &
            (inf_ref_pts[:, 1] >= W_L) & (inf_ref_pts[:, 1] <= W_R)
        )
        first_ego = in_ego.nonzero(as_tuple=False)
        if len(first_ego) > 0:
            keep = torch.ones(len(inf), dtype=torch.bool,
                              device=inf_ref_pts.device)
            keep[first_ego[0, 0]] = False
            inf = inf[keep]
            inf_ref_pts = inf_ref_pts[keep]

        if len(inf) == 0:
            return veh

        # matching: vectorised GPU cost matrix + CPU scipy Hungarian
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching_vec(
            inf_ref_pts, veh_ref_pts, veh_mask,
            veh.pred_boxes[..., [2, 3, 5]])

        # ref_pts absolute → norm
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.pc_range)
        veh_ref_pts = self._loc_norm(veh_ref_pts, self.pc_range)
        inf.ref_pts = inf_ref_pts
        veh.ref_pts = veh_ref_pts

        # cross-agent feature alignment (vectorised)
        N_inf = inf.query.shape[0]
        inf2veh_r = calib_inf2veh_t[:3, :3].reshape(1, 9).expand(N_inf, 9)
        inf_q = inf.query.clone()
        inf_q[..., :self.embed_dims] = self.cross_agent_align_pos(
            torch.cat([inf_q[..., :self.embed_dims], inf2veh_r], -1))
        inf_q[..., self.embed_dims:] = self.cross_agent_align(
            torch.cat([inf_q[..., self.embed_dims:], inf2veh_r], -1))
        inf.query = inf_q

        # cross-agent query fusion (vectorised index_add — no Python loop)
        accept_mask = np.array(
            [cost_matrix[veh_idx[k]][inf_idx[k]] < 1e5
             for k in range(len(veh_idx))])
        inf_accept_idx = []
        if accept_mask.any():
            veh_acc = veh_idx[accept_mask]
            inf_acc = inf_idx[accept_mask]
            inf_accept_idx = inf_acc.tolist()

            veh_acc_t = torch.tensor(veh_acc, dtype=torch.long,
                                     device=inf.query.device)
            inf_acc_t = torch.tensor(inf_acc, dtype=torch.long,
                                     device=inf.query.device)
            fused = self.cross_agent_fusion(
                inf.query[inf_acc_t, self.embed_dims:])
            veh_q = veh.query.clone()
            veh_q[veh_acc_t, self.embed_dims:] = (
                veh_q[veh_acc_t, self.embed_dims:] + fused)
            veh.query = veh_q

        # cross-agent query complementation (dynamic Instances.cat)
        inf_accept_set = set(inf_accept_idx)
        for i in range(inf.ref_pts.shape[0]):
            if i not in inf_accept_set:
                veh = Instances.cat([veh, inf[i]])

        return veh