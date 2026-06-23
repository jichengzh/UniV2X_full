"""阶段 4 sanity wrapper config — 继承 univ2x_tiny_e2e_track,
只覆盖训练规模相关字段,做最小可行 sanity (~5-10 min).

sanity 阶段禁用 aux heads (seg/motion/occ/planning),
只验证 backbone+neck+BEVFormer+pts_bbox_head 主干能否前向 + loss 下降.
完整功能在 stage1 真实训练时启用.
"""

_base_ = ["./univ2x_tiny_e2e_track.py"]

# 训练规模缩减
total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# sanity 现在保留 BEV 200×200(与 base 一致),所有 head 都启用

# 数据子集
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,         # sanity 用少量 workers 避免 OOM
    train=dict(ann_file="data/phase4/spd_train_sanity_50.pkl"),
    val=dict(ann_file="data/phase4/spd_train_sanity_50.pkl"),    # 同 train 避免缺数据
    test=dict(ann_file="data/phase4/spd_train_sanity_50.pkl"),
)

# 学习率 (1 epoch 直接小一些, 避免大 lr 让 loss 跳)
optimizer = dict(lr=5e-5)

# 频繁 log,便于看 loss 曲线
log_config = dict(
    interval=5,
    hooks=[dict(type="TextLoggerHook")],
)

# 关掉 mid-epoch eval (sanity 时 epoch 间 eval 即可)
evaluation = dict(interval=99, pipeline=_base_["test_pipeline"] if False else None)

# checkpoint 每 epoch 存
checkpoint_config = dict(interval=1)
