# TRT Phase-1 config for UniV2X-tiny (R50 + no DCN).
# Inherits from univ2x_coop_tiny.py and replaces BEVFormer module types
# with TRT-compatible variants for ONNX export + TRT engine build.
#
# Usage:
#   python tools/export_onnx_univ2x.py \
#       projects/configs_e2e_univ2x/univ2x_coop_tiny_trt.py \
#       projects/work_dirs_e2e_univ2x/univ2x_coop_tiny/epoch_30.pth \
#       --model ego --backbone-only --bev-size 200 \
#       --out onnx/univ2x_tiny_ego_bev_200.onnx

_base_ = ["./univ2x_coop_tiny.py"]

# Same TRT override as univ2x_coop_e2e_track_trt_p.py
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

model_ego_agent = dict(
    pts_bbox_head=dict(
        type="BEVFormerTrackHeadTRT",
        transformer=dict(
            encoder=_encoder_trt_override_,
        ),
    ),
)

model_other_agent_inf = dict(
    pts_bbox_head=dict(
        type="BEVFormerTrackHeadTRT",
        transformer=dict(
            encoder=_encoder_trt_override_,
        ),
    ),
)
