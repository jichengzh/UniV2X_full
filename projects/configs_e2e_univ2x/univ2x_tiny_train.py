"""阶段 4.6 正式训练 wrapper — 全集 365 sample × 3 epoch × 6 卡"""

_base_ = ["./univ2x_tiny_e2e_track.py"]

total_epochs = 3
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # 0 to avoid 'cannot pickle dict_keys' in distributed (1.2 known bug)
    train=dict(ann_file="data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_train.pkl"),
    val=dict(ann_file="data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl"),
    test=dict(ann_file="data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl"),
)

# 6 卡分布式 — 适当 scale lr (sanity 用 5e-5 单卡,这里 6 卡用 1e-4)
optimizer = dict(lr=1e-4)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
)

log_config = dict(
    interval=5,  # 训练慢(data_time 17s),5 iter 打一次便于看信号
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
checkpoint_config = dict(interval=1)

# 训练完后跑一次 eval (只在 ego_agent 上)
evaluation = dict(interval=99, pipeline=None)
