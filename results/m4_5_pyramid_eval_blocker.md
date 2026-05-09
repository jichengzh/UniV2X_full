# M4.5 Pyramid AP 评估 — 诊断报告

**日期**: 2026-05-09 02:55
**状态**: 🚧 BLOCKED — 数据集依赖缺失

## 已完成步骤

| Step | 状态 | 说明 |
|------|------|------|
| 1. OPV2V-H LiDAR 下载 | ✅ | 40 分片 / 38.77 GB (HF `yifanlu/OPV2V-H`) |
| 2. cat 合并分片 | ✅ | OPV2V_Hetero.tar.gz 39 GB |
| 3. 文件类型识别 | ⚠️ | 实际是 ZIP (HEAL README 写错为 tar.gz) |
| 4. unzip test/ | ✅ | 18 GB / 16 scenarios / 11970 .pcd 文件 |
| 5. 修 config.yaml 路径 | ✅ | sed 直接改 root_dir/test_dir/validate_dir |
| 6. 装 pypcd + cython box_overlaps | ✅ | klintan/pypcd + setup.py build_ext |
| 7. HEAL inference 启动 | ❌ | `len: 0` — 数据集找不到样本 |

## 根因

**OPV2V-H 是补充数据集，不是独立数据集**。OPV2V-H 仅包含每个 agent 的 16-line / 32-line LiDAR PCD，**没有标签 yaml** 也没有 64-line LiDAR。

来自 OPV2V-H README:
> Based on the original OPV2V dataset, we supplemented 16-line, 32-line lidar data...
> **Annotations will be shared with the original OPV2V**, so please download the original OPV2V dataset as well.

而 Pyramid_m1_base ckpt 的 config 设定 `m1` modality = PointPillar + 64-line LiDAR (默认 OPV2V), 所以评估必须有原版 OPV2V。

unzip 内容核实: 11970 个文件全部是 `.pcd` (16/32 后缀), 0 个 `.yaml`.

## 三条路径

### A. 使用 HEAL 论文已发表的 AP (不再跑评估)
- HEAL 论文 Table 2 已公开 Pyramid m1_base 在 OPV2V 上的 AP30/AP50/AP70
- 我们用的就是 HEAL 原版 ckpt (`net_epoch_bestval_at23.pth`), 数值可直接引用
- **优**: 0 工时 + 0 数据下载
- **劣**: 不是我们自测, 但本来就是引用 HEAL 论文的 baseline
- **建议**: 在论文里 cite HEAL paper 的 AP, 4090/Orin latency 是我们自测

### B. 下载原版 OPV2V test split (~50 GB)
- OPV2V 官方在 Google Drive (大文件慢) 或可能找 HF/学术镜像
- 估算: 1-3 小时下载 + 30 min 解压 + 30 min HEAL inference
- HF rate-limit 已多次触发, 不可靠
- **优**: 我们自测的 AP, 完整可控
- **劣**: 时间成本大, 可能再次踩坑

### C. 跳过 AP 实测, 仅交付 latency 维度
- M4.7 latency-only Pareto 已 commit (50 候选 + 1 Pareto 点)
- AP 维度作为论文 "future work" 或借用 HEAL 论文数值
- **优**: 不再阻塞主流程, 可继续 M5.6 / Phase 2.5
- **劣**: 论文实验完整度略弱

## 已就位但未用的资源

- `/home/jichengzhi/heal_research/dataset/OPV2V_Hetero/` (78 GB: 39GB zip + 18GB test + 21GB partitions, 可保留待 OPV2V 下完后用)
- `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/` (HEAL ckpt 已就位)
- pypcd / box_overlaps.so 已编译进 UniV2X_2.0 conda env
- `opencood/modality_assign/opv2v_4modality.json` 已存在 (heter assignment 路径修正即可)

## 用户反馈 (2026-05-09)

> "但是我们的框架优化之后还要测试精度啊, 不然怎么找到帕累托最优前沿呢"

A 错了. AP 是 Pareto 必要维度, 不只 baseline.

## 修正路径 (Phase 2.4 v5.1 设计)

**两层 AP**:
1. **Baseline AP**: Pyramid_m1_base 自测一次 → 验证 ckpt 在我们环境正常工作 + 提供 anchor
2. **Per-config AP**: M4.7 50 候选剪枝/量化后的 AP

50 候选都跑实测不现实. 走代理 + 抽样验证:
- 精度预测器 v5.1 (LightGBM) 给 50 候选预测 AP (in-the-loop)
- 抽样 5-10 个 Pareto-optimal 候选实测 AP (out-of-loop, 验证)
- 这是 Phase 2.4 的初衷

## 实际执行 (今天)

**已发现可用资源**: HF `gqk/opv2v` 数据集
- test.tar.gz.part00-03 = 4 × ~5GB = **~21 GB** (远小于全 50GB)
- 已启动下载 → /home/jichengzhi/heal_research/dataset/OPV2V_orig

**步骤**:
1. ⏳ 下载 OPV2V test (~21GB, est 30-70min) - PID 3757675
2. cat parts > test.tar.gz, tar xzf
3. 修 Pyramid_m1_base config.yaml (root_dir/test_dir → OPV2V_orig 路径)
4. 修 heter assignment 路径 (`opencood/logs/heter_modality_assign/` ← `opencood/modality_assign/opv2v_4modality.json`)
5. HEAL inference → AP30/50/70

**输出**:
- results/m4_5_pyramid_baseline_eval.txt (AP30/50/70)
- baseline_4090.parquet 加 amota=AP50 列 (覆盖之前的 NaN)

## 后续 (M4.6+)

- Pyramid 剪枝/量化 hook 实装 (~2-3 天)
- 抽样 5-10 个 Pareto 候选实测 AP
- Phase 2.5: 加 Pyramid 数据点重训精度预测器
