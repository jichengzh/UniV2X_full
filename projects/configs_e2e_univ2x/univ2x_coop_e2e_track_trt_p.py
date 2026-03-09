# TRT Phase-1 config: inherits from univ2x_coop_e2e_track.py and replaces
# BEVFormer module types with TRT-compatible variants.
#
# Usage:
#   python tools/export_onnx_univ2x.py \
#       projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
#       [ckpt] --model ego/infra ...
#
# Only the encoder-level module types are changed; all other config values
# (bev_h, bev_w, embed_dims, etc.) are inherited unchanged.

_base_ = ["./univ2x_coop_e2e_track.py"]

# ── Helper: recursively replace BEVFormer encoder module types ────────────
# mmdet3d config inheritance: re-assign the relevant sub-keys in model_ego_agent
# and model_other_agent_inf using config overrides.
#
# mmcv cfg_from_base merges dicts, so we set only the keys we want to change.

_encoder_trt_override_ = dict(
    type="BEVFormerEncoderTRT",
    transformerlayers=dict(
        type="BEVFormerLayerTRT",
        attn_cfgs=[
            dict(type="TemporalSelfAttentionTRT", embed_dims=256, num_levels=1),
            dict(
                type="SpatialCrossAttentionTRT",
                deformable_attention=dict(
                    type="MSDeformableAttention3DTRT",
                    embed_dims=256,
                    num_levels=4,
                ),
            ),
        ],
    ),
)

# Override ego agent encoder
model_ego_agent = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=_encoder_trt_override_,
        ),
    ),
)

# Override infra agent encoder (same structure)
model_other_agent_inf = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=_encoder_trt_override_,
        ),
    ),
)
