# 本次 session 我搞错的事情清单

## 判断 / 分析层面 (核心)

1. **建议跳过 AP 评估**: M4.5 阻塞时建议 option A (引用 HEAL 论文 AP), 被用户纠正——剪枝/量化后的 AP 是 Pareto 必要维度, 不是只测 baseline 就行.

2. **加速空间方向搞反**: 初次分析说 "univ2x_full 加速空间大 (5-10×), Pyramid 小 (1.4×)", 被用户指出: univ2x_full 因 **DCN 不可剪/不可量化** 实际 framework 控制空间反而小; Pyramid 纯 CNN 才是 framework 真正能 demo 加速的目标.

3. **Pyramid 加速天花板算错**: 把 M4.6.0 实测 FP16 1.32× 当成天花板, 实际 1.32× 只是 FP16 单维度起点, 还能叠加 INT8 (1.5-2×) + 剪枝 (1.5×) = 2-3× 真实可达加速.

4. **抽样矩阵 budget 算错**: 第一版给 Pyramid 15 + univ2x_full 5, 理由是 "univ2x 加速空间大", 跟 #2 同源错误, 实际应该相反.

5. **没事先识别 mask-based pruning 不真减 latency**: M4.6.1-3 设计时没想到 PyTorch `prune.ln_structured` 只是 mask + dense kernel 跑全, 不真减 latency. 应该提前在实验设计文档写明 "M4.6.1-3 是 AP-only validation, latency 数据不可信", 让用户知情.

6. **预测 latency anchor 选错**: M4.7 用 PyramidFusion 子模块 4.5ms (dummy 1×64×256×256) 作 KNN anchor, 跟实测完整 e2e 35ms 差 8×. 这导致 M4.7 Pareto candidate 0040 预测 lat 2.46ms 实测 26.8ms (10× 偏差), framework Pareto 完全失效.

7. **AP 预测器没接到 framework**: M4.7/M5.6/A7 都用 latency × params 双轴, 没用 Phase 2.4 v5.1 LGB 预测器估 AP. 直到 M4.6.3 实测出 Pareto candidate 最差才暴露这个设计 gap.

8. **表格数据呈现没标 caveat**: 跟用户报告"univ2x_full 244-2297ms" 看似真数据, 实际是 KNN over **3 个 sparse anchor 点** (90/?/5640ms) 的插值, 不可信. 应该标 [KNN-estimated / 3 anchors only].

9. **PyramidFusion 子模块 vs 完整 e2e 没标口径**: M4.3/M4.5_6 测的 4.49/3.27 ms 是 PyramidFusion 子模块 (dummy input), M4.6.0 测的 35/27 ms 是完整模型 (含 voxelization). 直接放进同一表格比较是错的, 用户问到才解释.

## 工程 / 实装层面

10. **数据集理解错**: 以为 OPV2V-H 是独立可评估数据集, 实际只是补充 16/32 线 LiDAR, 无 yaml 标签. 浪费 39GB 下载 + 18GB 解压, 后才明白要 OPV2V 原版.

11. **HEAL README 文档错信**: README 说 `cat parts > tar.gz`, 实际是 zip 格式. 第一次 tar 解压失败才发现.

12. **FP16 实装走 model.half() 路径错**: 直接 `model.half()` + input.half() 后 post_process anchor (fp32) dtype mismatch. 改用 `torch.cuda.amp.autocast` 仍错, 因为 inference_intermediate_fusion 内部 model→post_process 一气呵成. 最后必须 inline split.

13. **M4.6.0 FP32/FP16 timing 不一致**: v1 FP32 timing 含 post-process (119ms), v2 FP16 timing 只 model fwd (27ms). 不公平, 重跑 v2 一致.

14. **API signature 没查**: `calculate_ap(result_stat, 0.3, False)` 调用时多传一个参数, 实际只 take 2 个 (result_stat, iou). `eval_final_results` 已经返 ap30/50/70 不必再调用.

15. **YAML safe_load 不支持 OrderedDict tag**: HEAL config.yaml 有 `python/object/apply:collections.OrderedDict` tag, safe_load 失败. 改 sed 直接改路径.

## 流程 / 沟通层面

16. **没在动手前做"该 model 适不适合 framework"的 sanity check**: M4.6 启动前应该先算 latency profile + 加速天花板, 用户问"是否先做判断"才意识到漏了这步.

17. **wakeup 调度跟 monitor 重叠 + 早醒**: schedule wakeup 60min 后, 但实际 30 min 就 fired. 加上多个 monitor 并发, 导致旧 monitor 通知混入对话. 应该一种机制统一.

18. **多次重复跑同一脚本调试**: M4.6.0 FP16 实装错了 3 次 (model.half → autocast → autocast+split), 每次都跑 30 samples smoke test, 浪费 ~10min. 应该先静态分析 inference_utils 内部结构再写 wrapper.

## 结论

最重要的 3 个: **#2** (加速方向搞反), **#5** (mask 不真减 latency 没事先说), **#7** (AP 预测器没接 framework). 这 3 个是论文 framework 真实状态的 gap, 修这些才能让 M4.6.3 Pareto 验证成立.
