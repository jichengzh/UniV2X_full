# HANDOFF: HEAL PyramidFusion → V2Xverse 闭环移植 + 公平对比 (v1, 2026-06-14)

> 自足交接文档。读完能接续：P1–P5 已完成，公平对比已出，下一步可做 CoDriving 同条件重训 / 多 repeat / τ_ego 曲线。
> 配套 memory: `project-pyramid-p4-training`, `project-h800-docker-training-stack`, `project-real-repo-is-v2x`。

---

## 0. 一句话现状
HEAL PyramidFusion 已移植进 V2Xverse、在其 CARLA 数据上训练(感知40ep AP@0.5车0.91 + 专属planner 2ep)、接入闭环驾驶、与 CoDriving 做了**公平 τ=0 对比**(Pyramid自训planner DS74.6/RC100 vs CoDriving DS85.3/RC90.3)。全程隔离在 H800 `${V2X_DATA_ROOT}/V2Xverse_pyramid`，**仿真主框架/t2lib 零污染**。

## 1. 环境与硬约束 (★必守)
- **H800 SSH**: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>` (密码每会话确认)。
- **隔离副本(所有工作只在这)**: `${V2X_DATA_ROOT}/V2Xverse_pyramid` (从sim副本rsync, 含闭环代码+Pyramid opencood+训练产物)。
- **绝不碰**: 仿真主框架 `/data/jichengzhi_v2x/V2Xverse`、torch2 环境 `/data/jichengzhi_v2x/t2lib`、GPU3 上的外部 CARLA 进程。改这些必须先问用户。
- **GPU 策略(用户给定)**: 训练用 GPU0,1,2(3卡); 闭环eval用 GPU5。跑前 nvidia-smi 确认空闲。
- **训练/推理栈 = t2lib**(torch2.1+cu121, sm90原生, numpy1.26.4, python3系统): `PYTHONPATH=/data/jichengzhi_v2x/t2lib:<repo> python3`。注意是 python3 不是 python。
- **数据集**: 163GB HF gjliu/V2Xverse(仅weather-0/96route)下到 `${V2X_DATA_ROOT}/hf_cache`, 解压到 `${V2X_DATA_ROOT}/v2xverse_dataset/weather-0/data/<route>/`, 索引 dataset_index.txt(绝对路径)。实际可用 **83 完整route**(4条下载中断残缺已排除)。

## 2. 已完成 P1–P5 (均主控复核, 非agent自报)
- **P1 模型搬运**: HEAL 5文件入opencood(heter_pyramid_collab/single.py, fuse_modules/pyramid_fuse.py, loss/point_pillar_pyramid_loss.py, visualization/debug_plot.py)。修复(仅副本): mmcv改惰性(feature_alignnet*), 删`from turtle import update`(where2comm_attn/codriving_attn), debug_plot强制matplotlib Agg, box_overlaps cython重编。
- **P2 多类头**: 新建 `opencood/models/heter_pyramid_collab_multiclass.py`(HeterPyramidCollabMulticlass), 对齐V2Xverse 3类(car/ped/cyclist), forward适配V2Xverse batch(合成agent_modality_list=['m1']*N, processed_lidar→inputs_m1)。
- **P3 混合loss(方案B)**: 新建 `opencood/loss/center_point_pyramid_loss_multiclass.py` = center_point_loss_multiclass + HEAL occ前景监督(occ_weight=0.1)。
- **P4 感知训练**: 4卡DDP(torchrun, train_ddp.py, **find_unused_parameters=False**关键提速4.5x), 40epoch, config `opencood/hypes_yaml/v2xverse/pyramid_multiclass_config.yaml`(multi_class:true)。最佳ckpt `opencood/logs/v2xverse_pyramid_multiclass_2026_06_13_01_43_57/net_epoch_bestval_at33.pth`。AP(val=train乐观): car 0.94/0.91/0.85, ped 0.65/0.40/0.08, cyclist 0.56/0.46/0.27 @IoU0.3/0.5/0.7。
- **P5 闭环接入**: process B用 `--pnp-config pnp_config_pyramid_norsu.yaml`(perception_model_dir→pyramid logs, planner段)加载Pyramid驱动CARLA。修复: 闭环 team_code/utils/yaml_utils.py 补 `load_general_params` 解析器(Pyramid config的yaml_parser需要)。

## 3. ★公平对比结果 (τ=0, 9route, norsu, GPU5, 各用自训planner)
为公平: V2Xverse协议是**每感知方法单独训planner**(感知冻结, train_planner_e2e.sh)。给Pyramid训了专属planner(end2end_pyramid.yaml=copy codriving版weathers改[0], 2epoch loss0.022收敛, ckpt `.../planner_e2e-<PRIVATE_HOST>-20260614-093449/models/epoch_1.ckpt`)。eval配置 `pnp_config_pyramid_eval.yaml`(planner指向自训ckpt)。

| route | Pyramid(自训planner) DS/RC | CoDriving DS/RC |
|---|---|---|
| r0/2/5/31 | 100/100 | 100/100 |
| r1 | 36/100 | 60/100 |
| r3 | 50/100 | 100/100 |
| r10 | **100/100** | **7.4/12.3** |
| r18 | 25/100 | 100/100 |
| r112 | 60/100 | 100/100 |
| **均值** | **74.6/100** | **85.3/90.3** |

结论(有取舍): CoDriving平均DS高(碰撞少); Pyramid完成率100%(不卡死)+r10鲁棒(100 vs 7.4)。**caveat: 单repeat n=1方差大; Pyramid planner仅2ep; Pyramid感知val=train; CoDriving用的是原作者全量数据checkpoint(占训练数据优势)**。

## 4. ★下一步训练计划: CoDriving 同条件重训(真正公平)
**为什么**: 当前CoDriving=原作者全量数据训练的checkpoint, 而Pyramid=我们83route子集训练。要隔离"架构差异"而非"训练数据差异", CoDriving需在**与Pyramid完全相同条件**下我们自己重训。

**计划(全在/exdata副本, t2lib)**:
1. **CoDriving感知重训**: 4卡DDP(GPU0,1,2,3 或按用户给的卡), 同数据(83route weather-0, dataset_index.txt), 同40epoch, config `opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml`:
   ```
   cd ${V2X_DATA_ROOT}/V2Xverse_pyramid
   CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=/data/jichengzhi_v2x/t2lib:. python3 -m torch.distributed.run \
     --nproc_per_node=3 --master_port=29515 opencood/tools/train_ddp.py \
     -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml > codriving_retrain.log 2>&1
   ```
   注意: train_ddp.py 需确认 find_unused_parameters=False(已改); 若codriving config缺multi_class等字段参考pyramid config补; 留意numpy/版本坑(np.int等, 见§6)。
2. **CoDriving planner重训**: 用重训的感知ckpt, 同2-3epoch:
   ```
   bash scripts/train_planner_e2e.sh 0,1,2 3 <codriving_retrain_logs> codriving
   ```
   (train_planner_e2e.sh 已改 python→python3)
3. **闭环对比**: 用重训的codriving(感知+planner)配置跑同9route(或更多)τ0, **建议每route≥3 repeat降方差**, 对比重训Pyramid vs 重训CoDriving。

**预算**: 感知40ep~6h(4卡)/~8h(3卡); planner 2-3ep~2-3h; eval 9route×3rep×2模型~6-9h。总~1-1.5天。

## 5. 关键命令速查
- **起CARLA(GPU5)**: `bash /data/jichengzhi_v2x/h800_carla_launch.sh 5 <port>` (port用全新如4400避TM残留)。等 `nc -z <PRIVATE_HOST> <port>`。
- **起process B(感知server)**: `cd ${V2X_DATA_ROOT}/V2Xverse_pyramid; CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:.:simulation/leaderboard setsid nohup python3 simulation/leaderboard/team_code/closedloop/process_b_server.py --port 5557 --gpu 5 --pnp-config <cfg> >log 2>&1 </dev/null &` 等 PROCESS_B_READY。
- **起process A(闭环eval)**: `CUDA_VISIBLE_DEVICES=5 USE_INFER_SERVER=1 INFER_SERVER_PORT=5557 PATH=/data/jichengzhi_v2x/envs/v2xverse/bin:$PATH bash scripts/eval_driving_e2e.sh <route> <port> <tag> 0 <agent_cfg> _1`。
- **读DS/RC**: results.json的 `_checkpoint.global_record.scores.{score_composed(DS), score_route(RC), score_penalty}`。路径 `results/results_driving_<tag>_r<route>/v2x_final/town05_short_collab/r<route>_repeat0/ego_vehicle_0/results.json`。
- **对比sweep脚本范本**: `${V2X_DATA_ROOT}/sweep_fair2.sh`(端口4400, process B切换+route循环, 可改route集/config)。

## 6. 踩过的坑 (省时间)
- **SSH挂2min(正常,进程已起,另开命令核验)。写文件用scp/base64不用heredoc。
- **TM端口冲突**: leaderboard TM端口=world-port+5; CARLA重启后残留TM不释放→下次bind error→route秒退(~12s无results)。**用全新world-port(如4400)+顺序跑route(TM自然释放), 先单route验证再全sweep**。
- **numpy版本**: t2lib numpy1.26.4移除np.int/np.float/np.bool→旧代码崩(如common/heatmap.py)。改成int/float/bool。
- **python vs python3**: 脚本里`python`命令不存在, 改python3。
- **find_unused_parameters=True**是DDP大瓶颈(慢4x), Pyramid模型无未用参数→改False提速。
- **多bestval文件**: train_ddp不删旧bestval, 但inference的load_saved_model断言只1个→eval前移走多余只留最佳。
- **Scenario13(ManeuverOppositeDirection)** 实例化失败, 影响含它的route(r146等)setup, 与模型无关→用不含它的route(0/1/2/3/5/10/18/31/112等)。
- **PFS(/exdata)解压几万小文件慢**(~6min/route串行)→6并行worker。
- 不轻信agent自报: 之前sim-integrator agent跑偏去看旧tau_curve实验, 还烂尾改了yaml_utils。凡"已改/已测"主控复跑核验。

## 7. 其他可选下一步
- planner多训epoch + 多repeat → 更硬的对比结论(当前最值得补)。
- τ_ego时延注入 × Pyramid → "时延×驾驶分"曲线(项目核心研究目标, 时延机制见 sim_test_design_v1.md §3)。
- 精度提升: occ用Gaussian heatmap(非二值投影); held-out测试集划分(现val=train); 补全4残缺route训完整87。
- 修Scenario13跑对抗场景route(更有挑战的DS)。
