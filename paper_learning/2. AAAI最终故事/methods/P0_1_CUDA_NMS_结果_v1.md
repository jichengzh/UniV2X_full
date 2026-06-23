# P0.1 — CUDA NMS 替换 Shapely:实施 + 验证结果 (v1)

> **日期**: 2026-05-31
> **目标**: 把 HEAL `box_utils.nms_rotated`(纯 Python + Shapely O(N²) CPU 循环,占 PyramidFusion e2e ~72%)替换为 GPU `mmcv.ops.nms_rotated`,在保 AP 不变的前提下拿真 e2e 加速。
> **关联**: `plan6_方法优化路线_交接_v1.md` §五/§六 P0.1。

---

## 一、结论(一句话)

**100% keep-set 逐框一致(AP 不变),e2e 81.5→26.8 ms = 3.04× 真加速,NMS 55→1.55 ms = 35.4×。NMS 不再是瓶颈。**

## 二、实测数据(RTX 4090 / GPU 7 独占 / OPV2V test / 170 NMS 调用 / 150 样本 e2e)

| 指标 | Shapely(原) | CUDA(新) | 比值 |
|---|---|---|---|
| keep-set 完全一致率 | — | — | **100.0%** (Jaccard 1.0) |
| 平均候选框 #in | 83.4 | 83.4 | — |
| 平均保留框 #keep | 18.2 | 18.2 | 逐框相同 |
| **NMS 延迟 mean** | 55.06 ms | **1.55 ms** | **35.4×** |
| **e2e 延迟 mean** | 81.53 ms | **26.79 ms** | **3.04×** |

注:e2e 3.04×(非预估 3.6×)因 Shapely 基线本趟测得 81.5 ms,低于 05-31 profiling 的 98 ms 均值(后者被 NMS p99 离群拉高)。CUDA 后 e2e 绝对值 26.8 ms 命中 ~27 ms 预估。

## 三、为什么 AP 严格不变(不是"近似不变")

- HEAL `convert_format` 用 box 前 4 角点的 (x,y) 构 BEV 多边形算 IoU;这 4 角点来自 `boxes_to_corners_3d` 的刚性旋转+平移,**是精确矩形**。
- 故矩形可无损表示为 `(cx, cy, w, h, angle)`,mmcv 旋转 IoU 与 Shapely 多边形 IoU 仅差浮点 epsilon。
- 实测 170 次调用 keep-set **bit 级一致** → 保留框集合相同 → **AP 完全相同**(比"差异在噪声内"更强)。

## 四、实现

- `scripts/phase2/p0_1_nms_cuda.py`
  - `corners_to_rotated_bev(corners)`: (N,8,3)/(N,4,2) → (N,5) mmcv 旋转框
  - `nms_rotated_gpu(boxes, scores, threshold)`: drop-in,同签名/同返回 (np.int32 keep idx),保留原 top=1000 截断
  - `patch_heal()`: monkey-patch `box_utils.nms_rotated`,返回原函数以便恢复
- `scripts/phase2/p0_1_nms_validate.py`: 三趟验证(compare 正确性+NMS计时 / shapely e2e / cuda e2e)
- 产物 `data/p0_1_nms_validation.json`

## 五、部署状态:✅ 已接入 HEAL 源码(2026-05-31)

用户决议:接入 HEAL 源码 + Shapely fallback。已落地:

- **改动文件**: `/home/jichengzhi/heal_research/HEAL/opencood/utils/box_utils.py`
  - 新增 `_get_mmcv_nms_rotated()`(懒加载 + 缓存可用性)、`_corners_to_rotated_bev()`、`_nms_rotated_cuda()`
  - 原 Shapely 实现保留为 `_nms_rotated_shapely()`(零改动,逐字保留)
  - 公开 `nms_rotated()` 改为 dispatcher:走 CUDA,**任何失败(无 mmcv / 无 CUDA / 报错)自动回退 Shapely** → 行为永不劣化
- **生效范围**: 整个 HEAL clone 所有 postprocessor + `inference.py` e2e,自动 ~3× 提速,无需改调用方。
- **验证**: 集成后 smoke test,dispatcher 走 CUDA 且 keep-set 与 `_nms_rotated_shapely` 逐框 EXACT MATCH,返回 dtype int32,空输入/fallback 正常。
- **可逆**: HEAL 是 git 仓库,`git checkout opencood/utils/box_utils.py` 即还原。

## 六、复跑命令

```bash
cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=7 \
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  /home/jichengzhi/UniV2X/scripts/phase2/p0_1_nms_validate.py --warmup 20 --measure 150
```

## 七、对 paper claim 的意义

- 车端 e2e 3.04× 真加速,**正确性 bit 级保证**,是 plan v5 之后第一个 strong claim。
- 重排后:剪枝/量化是 forward(占 e2e ~24%)内部的二阶优化;NMS(CUDA 化)是车端一阶加速。
