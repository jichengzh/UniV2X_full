# Stage1 扫描产物索引 (晨检用) — 2026-06-12 00:13

> 本目录 = Stage1「软硬件扫描」对 4 个模型的产出。每份 `<model>_partition.yaml` 含三视图(B1剪枝组/B2量化单元/D路由) + 硬件 capability 汇合 + 一致性校验。
> 代码: `framework/stage1/{hardware_scan,graph_scan,adapters,run_scan}.py`。设计: `multi_agent/methods/design/stage1_graph_scan_partition_v1.md`。
> 硬件: `configs/hardware/rtx4090.yaml`(无 DLA → D 路由退化为仅 GPU; int8_align=32; legal_bits=[INT8,FP16])。
> env: `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python` (torch 2.0.1 + torch_pruning 1.6.0)。

## 复跑命令 (每模型独立进程, ★V2Xverse 与 HEAL 都有 `opencood` 包, 同进程会撞, 必须分开)
```bash
cd /home/jichengzhi/V2X
for M in codriving pyramid_lidar pyramid_camera v2xvit; do
  PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
    -m framework.stage1.run_scan --model $M --device cpu
done
```

## 结果汇总 (全部 scan_status=ok, dry-run 0.5 真剪 forward 全通, grouped-conv 一致)

| 模型 | ckpt | trace 入口 | B1组 | B2单元 | D节点 | dry-run 剪幅 | 主参数桶 |
|------|------|-----------|------|--------|-------|-------------|---------|
| **codriving** | ✅ V2Xverse at16 | (1,64,192,704) | 20 | 3 | 64 | -74.8% | backbone 85.4% / neck 14.5% |
| **pyramid_lidar** | ✅ DAIR at23 | (1,64,128,256) | 43 | 4 | 128 | -53.9% | bev_encoder 57.8% / neck 38.0% |
| **pyramid_camera** | ⚠️ 仅 OPV2V(无DAIR) | (1,128,256,512) | 46 | 5 | 137 | -51.3% | bev_encoder 56.3% / neck 36.9% |
| **v2xvit** | ✅ DAIR at17 | (1,64,256,512) | 23 | 3 | 49 | -73.0% | backbone 74.2% / neck 25.7% |

## 每模型要点 (晨检关注)

1. **codriving**: ResNetBEV(BasicBlock, **无 grouped-conv**), backbone 独占 85% 参数 → 剪枝主战场在 backbone。heads(cls3/reg24)锁 FP16。
2. **pyramid_lidar**: ResNeXt **g=32** grouped-conv, round_to 自动取 lcm(32,32)=32。bev_encoder(pyramid_backbone)57.8% + neck(deblocks+shrink)38%。复用 `depgraph_pyramid.PyramidFullTraceNet`。与手工分组对照见设计文档 §3 Q1 教训框(DepGraph 抓到 backbone↔stage0 残差耦合)。
3. **pyramid_camera** ⚠️ **两个 caveat**:
   - **ckpt 仅 OPV2V**(`m2_alignto_m1/net_epoch25.pth`), 盘上**无 DAIR camera ckpt** → 该 partition 的结构有效, 但若要 DAIR AP 数据需另训/找 ckpt。
   - **aligner_m2(ConvNeXt channel_align 含 LayerNorm)被排除出可剪集**: torch_pruning 结构化剪枝不更新 ConvNeXt LayerNorm 的 normalized_shape(实测 dry-run 崩 `weight[32] vs normalized_shape[64]`), 且仅占 1.94% 参数(`other` 桶) → v0 冻结(其上游 backbone_m2 因耦合一并冻结, 4.76%)。主可剪路径 = bev_encoder 56% + neck 37%。**这是真实发现, 非掩盖** —— 若要剪 aligner 需 LayerNorm-aware 剪枝。
4. **v2xvit**: backbone(BaseBEVBackbone)**无 grouped-conv**; **transformer 融合(V2XTransformer: HMSA+MSwin, 多 agent 不可 trace)整体不入图, 仅剪 backbone**(依据 A-2 授权 + dims_pruning §8.6)。backbone 占 74%。

## 视图→搜索空间对齐 (每份 yaml 内含)
- `view_b1_prune_groups[*].max_rate` → 剪枝率上界; `.criterion_pool` → 准则候选; `.round_to_*` → 对齐。
- `view_b2_quant_units[*].legal_bits` → 量化位宽合法集(heads 锁 FP16); `.member_groups` → 量化单元=完整依赖组并集。
- `view_d_routing[*].dla_able` → 路由合法集(4090 无 DLA 全 False); `.propagate_b2` → D↔B2 传播边(有 DLA 时生效)。
- `prune_object: channel`(锁定, 不搜 2:4/element)。

## 已知共性
- 全部 heads `quantizable=False`(精度敏感锁 FP16); 输出通道 anchor 固定, 剪枝 ignored(输入侧随依赖图跟剪)。
- 4090 无 DLA → 所有 `dla_able=False`, D.L1 退化仅 GPU(设计 §1 A.3 退化规则)。换 Orin capability YAML 即可激活 DLA 路由视图 + D↔B2 传播。
