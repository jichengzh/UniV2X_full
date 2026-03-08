# TRT Phase-2 config: inherits from Phase-1 (BEV encoder already TRT-ified)
# and additionally replaces detection decoder + all downstream heads with
# TRT-compatible variants.
#
# Stages covered:
#   Stage C  – detection decoder  (BEVFormerTrackHeadTRT)
#   Stage D  – motion prediction  (MotionHeadTRTP)
#   Stage E  – occupancy forecast (OccHeadTRTP)
#   Stage F  – planning           (PlanningHeadSingleModeTRT)

_base_ = ["./univ2x_coop_e2e_track_trt_p.py"]  # BEV encoder already TRT

# ── Shared constants (mirror univ2x_coop_e2e.py) ──────────────────────────────
_point_cloud_range_      = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_inf_point_cloud_range_  = [0, -51.2, -5.0, 102.4, 51.2, 3.0]
_occflow_grid_conf_ego_  = dict(xbound=[-50.0, 50.0, 0.5],
                                ybound=[-50.0, 50.0, 0.5],
                                zbound=[-10.0, 10.0, 20.0])
_occflow_grid_conf_inf_  = dict(xbound=[0, 100.0, 0.5],
                                ybound=[-50.0, 50.0, 0.5],
                                zbound=[-10.0, 10.0, 20.0])
_bev_conf_ego_           = dict(xbound=[-51.2, 51.2, 0.512],
                                ybound=[-51.2, 51.2, 0.512],
                                zbound=[-10.0, 10.0, 20.0])
_bev_conf_inf_           = dict(xbound=[0, 102.4, 0.512],
                                ybound=[-51.2, 51.2, 0.512],
                                zbound=[-10.0, 10.0, 20.0])

# ── Detection decoder TRT override ───────────────────────────────────────────
_decoder_trt_override_ = dict(
    type="DetectionTransformerDecoderTRTP",
    num_layers=6,
    return_intermediate=True,
    transformerlayers=dict(
        type="DetrTransformerDecoderLayer",
        attn_cfgs=[
            dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
            dict(type="CustomMSDeformableAttentionTRT", embed_dims=256, num_levels=1),
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ),
)

# ── MotionHead TRT layers (shared by ego + infra) ─────────────────────────────
_motion_transformer_trt_ = dict(
    type="MotionTransformerDecoderTRTP",
    pc_range=_point_cloud_range_,
    embed_dims=256,
    num_layers=3,
    transformerlayers=dict(
        type="MotionTransformerAttentionLayerTRTP",
        batch_first=True,
        attn_cfgs=[
            dict(
                type="MotionDeformableAttentionTRTP",
                num_steps=12,
                embed_dims=256,
                num_levels=1,
                num_heads=8,
                num_points=4,
                sample_index=-1,
            ),
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=("cross_attn", "norm", "ffn", "norm"),
    ),
)

# ── OccHead transformer decoder (shared) ─────────────────────────────────────
_occ_transformer_decoder_ = dict(
    type="DetrTransformerDecoder",
    return_intermediate=True,
    num_layers=5,
    transformerlayers=dict(
        type="DetrTransformerDecoderLayer",
        attn_cfgs=dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            dropout_layer=None,
            batch_first=False,
        ),
        ffn_cfgs=dict(
            embed_dims=256,
            feedforward_channels=2048,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
            ffn_drop=0.0,
            dropout_layer=None,
            add_identity=True,
        ),
        feedforward_channels=2048,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ),
    init_cfg=None,
)

# ── Override ego agent ────────────────────────────────────────────────────────
model_ego_agent = dict(
    pts_bbox_head=dict(
        type="BEVFormerTrackHeadTRT",
        transformer=dict(decoder=_decoder_trt_override_),
    ),
    motion_head=dict(
        type="MotionHeadTRTP",
        bev_h=200,
        bev_w=200,
        num_query=300,
        num_classes=10,
        predict_steps=12,
        predict_modes=6,
        embed_dims=256,
        loss_traj=dict(
            type="TrajLoss",
            use_variance=True,
            cls_loss_weight=0.5,
            nll_loss_weight=0.5,
            loss_weight_minade=0.0,
            loss_weight_minfde=0.25,
        ),
        num_cls_fcs=3,
        pc_range=_point_cloud_range_,
        group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
        num_anchor=6,
        use_nonlinear_optimizer=False,  # casadi not needed for TRT
        anchor_info_path="data/others/motion_anchor_infos_mode6.pkl",
        transformerlayers=_motion_transformer_trt_,
    ),
    occ_head=dict(
        type="OccHeadTRTP",
        bev_h=200,
        bev_w=200,
        pc_range=_point_cloud_range_,
        inf_pc_range=_inf_point_cloud_range_,
        is_cooperation=True,
        is_ego_agent=True,
        n_future=4,
        grid_conf=_occflow_grid_conf_ego_,
        bevformer_bev_conf=_bev_conf_ego_,
        ignore_index=255,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        attn_mask_thresh=0.3,
        transformer_decoder=_occ_transformer_decoder_,
        query_dim=256,
        query_mlp_layers=3,
        aux_loss_weight=1.0,
        loss_mask=dict(
            type="FieryBinarySegmentationLoss",
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0,
            ignore_index=255,
        ),
        loss_dice=dict(
            type="DiceLossWithMasks",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            ignore_index=255,
            loss_weight=1.0,
        ),
        pan_eval=True,
        test_seg_thresh=0.1,
        test_with_track_score=True,
    ),
    planning_head=dict(
        type="PlanningHeadSingleModeTRT",
        embed_dims=256,
        planning_steps=10,
        loss_planning=dict(type="PlanningLoss"),
        loss_collision=[
            dict(type="CollisionLoss", delta=0.0, weight=2.5),
            dict(type="CollisionLoss", delta=0.5, weight=1.0),
            dict(type="CollisionLoss", delta=1.0, weight=0.25),
        ],
        use_col_optim=False,  # casadi-based optimizer disabled for TRT
        planning_eval=True,
        with_adapter=True,
    ),
)

# ── Override infra agent (detection + motion + occ; no planning) ──────────────
model_other_agent_inf = dict(
    pts_bbox_head=dict(
        type="BEVFormerTrackHeadTRT",
        transformer=dict(decoder=_decoder_trt_override_),
    ),
    motion_head=dict(
        type="MotionHeadTRTP",
        bev_h=200,
        bev_w=200,
        num_query=300,
        num_classes=10,
        predict_steps=12,
        predict_modes=6,
        embed_dims=256,
        loss_traj=dict(
            type="TrajLoss",
            use_variance=True,
            cls_loss_weight=0.5,
            nll_loss_weight=0.5,
            loss_weight_minade=0.0,
            loss_weight_minfde=0.25,
        ),
        num_cls_fcs=3,
        pc_range=_point_cloud_range_,
        group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
        num_anchor=6,
        use_nonlinear_optimizer=False,
        anchor_info_path="data/others/motion_anchor_infos_mode6.pkl",
        transformerlayers=_motion_transformer_trt_,
    ),
    occ_head=dict(
        type="OccHeadTRTP",
        bev_h=200,
        bev_w=200,
        pc_range=_point_cloud_range_,
        inf_pc_range=_inf_point_cloud_range_,
        is_cooperation=False,
        is_ego_agent=False,
        n_future=4,
        grid_conf=_occflow_grid_conf_inf_,
        bevformer_bev_conf=_bev_conf_inf_,
        ignore_index=255,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        attn_mask_thresh=0.3,
        transformer_decoder=_occ_transformer_decoder_,
        query_dim=256,
        query_mlp_layers=3,
        aux_loss_weight=1.0,
        loss_mask=dict(
            type="FieryBinarySegmentationLoss",
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0,
            ignore_index=255,
        ),
        loss_dice=dict(
            type="DiceLossWithMasks",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            ignore_index=255,
            loss_weight=1.0,
        ),
        pan_eval=True,
        test_seg_thresh=0.1,
        test_with_track_score=True,
    ),
)
