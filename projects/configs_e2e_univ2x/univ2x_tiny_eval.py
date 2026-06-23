"""阶段 4 评估 wrapper config — 用 tiny 训出的 ckpt 在 val 全集(168 sample)评估"""

_base_ = ["./univ2x_tiny_e2e_track.py"]

# val 集用全集 (168 samples / 12 clips), 不要 sanity 子集
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=dict(ann_file="data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl"),
)
