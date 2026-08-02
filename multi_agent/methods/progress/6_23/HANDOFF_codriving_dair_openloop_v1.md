# HANDOFF: CoDriving 迁入开环评价体系 + DAIR-V2X AP/mAOE 对比 (v1, 2026-06-16)

> 自足交接文档。读完能接续。配套 memory: `project-codriving-dair-openloop-ap`(结论+复跑), `project-pyramid-p4-training`(那是 V2Xverse 闭环, 别混)。
> 本阶段独立于 V2Xverse 闭环线: 这是 **开环检测 benchmark (DAIR-V2X)** 上的 Pyramid vs CoDriving 对比。

---

## 0. 一句话现状
把 CoDriving(源自 V2Xverse, `center_point_codriving`+`codriving_attn` ATTEN 融合)**迁入开环评价体系**, 在 **DAIR-V2X-C** 上训练+评测, 与 HEAL/Pyramid 同口径对比。**现场真测定论: Pyramid 全面强于 CoDriving**(AP50 0.79 vs 0.56, AP70 0.63 vs 0.37, mAOE 0.060 vs 0.099)。差距经同 data/同范围/各自 native test 现场复现, 非欠拟合/旧数/范围伪影。

## 1. 任务背景 (用户原意)
- 手上两个模型: **HEAL/PyramidFusion** 与 **CoDriving(V2Xverse)**, 都已进 V2Xverse 闭环。
- HEAL 已做过充分**开环**评估(DAIR-V2X), CoDriving 没有 → 要把 CoDriving 也迁入开环体系拿 mAP/mAOE, 与 Pyramid 公平对比。
- 关键前提(已确认): **HEAL 开环基线 = DAIR-V2X-C**(`Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml`: dataset dairv2x, train/val.json, LiDAR 单类车 anchor_number=2, 范围 [-102.4,-51.2,-3.5,102.4,51.2,1.5])。

## 2. 方法路线 (为什么这么做)
- **不**把 codriving 移植进 HEAL: HEAL 的 center_point 路径休眠(有 `center_point_loss.py`+`center_point_where2comm.py` 但无 config 接, postprocessor 未测), 风险高。
- **改用 V2Xverse 自己的 opencood**: 原生支持 DAIR(`dairv2x_basedataset.py`)+ CenterPoint + `center_point_codriving` 模型 + `codriving_attn` + `center_point_loss`, 还有模板 `dair_centerpoint_where2comm.yaml`(CoDriving 孪生)。
- 公平: CoDriving 训练+评估范围对齐 HEAL 的 102.4×51.2, 同 train4811/val1789 切分, 单类车, 同 OpenCOOD eval_utils AP + 同 TP 误差定义。

## 3. 环境与路径 (★必守)
- **H800**: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认)。训练/评估都在 H800。
- **隔离副本(所有 codriving 工作在此)**: `${V2X_DATA_ROOT}/V2Xverse_pyramid`(已含 codriving 模型+dair 适配器; 主框架 `/data/jichengzhi_v2x/V2Xverse` 零污染)。环境 **t2lib**: `PYTHONPATH=/data/jichengzhi_v2x/t2lib:. python3`(torch2.1)。
- **4090**: HEAL 在 `${V2X_HOME}/heal_research/HEAL`, env `UniV2X_2.0`(torch2.0); Pyramid ckpt `${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29`; DAIR 数据 `/data/lxf/DAIR-V2X`。另有 4090 隔离副本 `/data/jichengzhi_dair/V2Xverse_codriving`(最初建配置+smoke 处, 后切 H800)。
- **DAIR 数据 (H800)**: data_dir(有点云)=`/data/jichengzhi_v2x/dair/cooperative-vehicle-infrastructure`(vehicle/infra-side velodyne 全); split json(4811/1789, 与 HEAL 同切分)=`/data/lxf_data/DAIR-V2X-C/Full_Dataset/cooperative-vehicle-infrastructure/{train,val}.json`。⚠️ `/data/lxf_data/.../Full_Dataset` 自身缺 velodyne, 点云只在 jichengzhi_v2x/dair。
- GPU: 训练用 H800 GPU5,6(2卡 DDP); 评估单卡 GPU5; Pyramid 现场重测在 4090 GPU4。跑前 nvidia-smi 确认空闲。

## 4. 关键产物
- **CoDriving DAIR 配置**: `${V2X_DATA_ROOT}/V2Xverse_pyramid/opencood/hypes_yaml/dairv2x/lidar_only/dair_centerpoint_codriving.yaml`(从 dair_centerpoint_where2comm.yaml 改: 模型 `center_point_codriving`, fusion `ATTEN` feature_dim256, 范围对齐 HEAL 102.4×51.2, voxel [0.4,0.4,5], 单类车 anchor_number 1)。
- **训练 log 目录**: `opencood/logs/dair_centerpoint_codriving_2026_06_15_21_19_14`(rank0, 干净, **单一 bestval@11**)。同时存在 `_21_19_13`(rank1, 多 bestval, 弃用)。
- **mAOE 脚本**: `${V2X_DATA_ROOT}/V2Xverse_pyramid/eval_tp_errors_codriving.py`(口径逐函数对齐 Pyramid `V2X/scripts/phase2/eval_tp_errors_corrected_full.py`); 结果 `<logdir>/tp_errors_result.json`。
- Pyramid 现场 live log(4090): `${V2X_ROOT}/pyramid_dair_r51.log`。

## 5. ★最终结果 (现场真测, DAIR-V2X-C val 1789, 范围 102.4×51.2, IoU0.5 TP)
| 指标 | Pyramid (bestval@23, 现场) | CoDriving (bestval@11, 现场) |
|---|---|---|
| AP30/50/70 ↑ | 0.83 / 0.79 / 0.63 | 0.67 / 0.56 / 0.37 |
| mATE ↓ (m) | 0.231 | 0.265 [0.262,0.268] |
| mASE ↓ | 0.130 | 0.133 [0.132,0.135] |
| mAOE ↓ (rad) | 0.060 | 0.099 [0.097,0.101] |
| n_TP | 26686 | 17532 |
- Pyramid 现场 = dataset_v2 金标准精确复现(0.833/0.791/0.631)。CoDriving epoch29 更低(0.523/0.450/0.343)=过拟合佐证。
- 差距集中: **召回(n_TP)+ 朝向 mAOE(1.66×)**; mASE 持平。

## 6. 为何 CoDriving 感知不如 Pyramid (架构核实)
1. ★**Pyramid 有多尺度 occ 前景监督**(`point_pillar_pyramid_loss` 的 calc_occ_loss + pyramid_weight); CoDriving `center_point_loss` 只有 cls+loc 无前景辅助 → 召回/朝向差(最主因)。
2. 融合: Pyramid `weighted_fuse` 逐位置置信度+多尺度前景加权; CoDriving where2comm ScaledDotProductAttention(+可选 Communication 阈值稀疏, 本配置未启)。
3. 设计目标: Pyramid/HEAL(ICLR24)=协同检测SOTA; CoDriving(TPAMI25)=端到端驾驶, 感知为规划服务非冲 AP。
4. 朝向: Pyramid anchor-based+dir_args 朝向 bin; CoDriving anchor-free center_point sin/cos 纯回归 → mAOE 差。
5. DAIR 上更快过拟合(bestval@11 vs Pyramid@23), 缺 occ 辅助正则。
- caveat: Pyramid 是作者完整调优 recipe, CoDriving 是 where2comm DAIR 模板 30ep; recipe 可缩小部分差距, 但 occ 监督缺失是架构本质。

## 7. 复跑命令
```
# 训练 (H800, 2卡DDP)
cd ${V2X_DATA_ROOT}/V2Xverse_pyramid
CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=/data/jichengzhi_v2x/t2lib:. python3 -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29517 opencood/tools/train_ddp.py \
  -y opencood/hypes_yaml/dairv2x/lidar_only/dair_centerpoint_codriving.yaml

# AP 评估 (单卡; 用 model_dir 的 bestval)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:. python3 opencood/tools/inference.py \
  --model_dir opencood/logs/dair_centerpoint_codriving_2026_06_15_21_19_14 --fusion_method intermediate

# mAOE/mATE/mASE
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:. python3 eval_tp_errors_codriving.py \
  opencood/logs/dair_centerpoint_codriving_2026_06_15_21_19_14

# Pyramid 现场重测 (4090) — ★必须 --range 102.4,51.2
cd ${V2X_HOME}/heal_research/HEAL
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=${V2X_HOME}/heal_research/HEAL \
  ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python opencood/tools/inference.py \
  --model_dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
  --fusion_method intermediate --range 102.4,51.2
```

## 8. 踩过的坑
- ★**HEAL inference.py `--range` 默认 `102.4,102.4`(正方形)** → 不传会评估错范围(给 0.76 偏低); 对齐 CoDriving/DAIR 标准必须 `--range 102.4,51.2`。对比前务必核范围。
- `codriving_attn.py` 第1行 `from turtle import update` IDE 误导入 → 删(副本已删 3 文件)。
- train_ddp 出**双 log 目录**(双 rank 差1秒): rank0 的 `_14` 干净单 bestval, rank1 `_13` 多 bestval 弃用。
- inference `load_saved_model` 断言**唯一 bestval** → 多 bestval 需移走只留最佳。
- DAIR 点云不在 lxf_data/Full_Dataset(只有 calib/label), 在 `/data/jichengzhi_v2x/dair`; data_dir 指有 velodyne 的, split json 指 4811/1789 的。
- 训练末尾 `python: not found` = 自动 inference 钩子(用 python 非 python3), 无害, 训练已完成。
- SSH 长命令避免内联 python 多层引号(易截断); 写文件用 scp 不用 heredoc。

## 9. 可选下一步
- 排除 recipe 因素(若仍质疑): CoDriving 用原生范围 100.8×40 或调 recipe(LR/aug)重训, 看上限。但 bestval@11 即过拟合, 证据指向架构非时长。
- 把这套对比(AP+TP误差+反思)整理进论文对比表 / 并入 dataset_v2 体系(注意 model_class 区分, 别和 Pyramid 行混)。
- 给 CoDriving 也补闭环↔开环的一致性分析(已有 V2Xverse 闭环 DS, 见 [[project-pyramid-p4-training]] / HANDOFF_pyramid_v2xverse_migration_v1)。
