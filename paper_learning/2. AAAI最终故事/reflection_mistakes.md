# 本次 session 我搞错的事情清单
## 最核心的，多次出错的地方
1. 我们的评估是要找到模型的帕累托前沿，一定是多指标评估，不能只测速度。


## 判断 / 分析层面 (核心)

1. **建议跳过 AP 评估是错的**: M4.5 阻塞时建议 option A (引用 HEAL 论文 AP), 被用户纠正——剪枝/量化后的 AP 是 Pareto 必要维度, 不是只测 baseline 就行.

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

## 2026-05-10 (M4.9 v2 + framework methodology 反思)

19. **搜索空间 ^5 over-claim 错误**: 跟用户报告 "4090 完整搜索空间 ~10^14" 是 schema 笛卡尔积, 不是实际有意义空间. 实际 prune_criterion 通常 fix (model-class property), q_granularity / q_object 通常是全局而非 per-module 搜. 修正后 4090 完整空间 ~10^5 (10^9 倍 over-claim). Pyramid 1-module 后 ~432.

20. **q_granularity / q_object 当 1 是错的**: 第一次跟用户讨论时把它们标记为 1, 用户立刻指出"这是全局但仍要搜". 两者各 2-3 值的全局选项不能 fix.

21. **5-module schema 套到 Pyramid 是过度复杂**: 用 UNIV2X_MODULES (5 模块) 套 Pyramid_m1, 但实证 heads 量化不敏感 (A7 mixed precision 反而 -0.07pp 比 A2 全 INT8 差). 5-module schema 引入大量无意义自由度. **正确做法**: model-specific 模块声明, Pyramid 用 1-module ("model") 即可, 跨模型时按 model class 调整.

22. **200 anchor 引用错**: 跟用户说"LGB v6 训了 200 anchor", 实际 `baseline_4090.parquet` 只 83 行 (LGB 实际用 amota=72 / latency=64). 应该实证查 parquet 而不是凭印象引用.

23. **"5% LGB 覆盖够用" 低估**: 跟用户说"5% 训练覆盖足够" (~22 anchor on 432 space), 用户立刻反问"至少 20%". 我应该参照 LGB v6 实证 (72 anchor → CV MAE 0.047, in-sample 0.010) 推算单模型至少需要 ~80 anchor (20% 覆盖) 才能让 CV MAE 接近 in-sample. 应该 P0 目标 ~86 anchor 而非 22.

24. **Hand-pick anchor 拿来训 LGB 是 selection bias**: 当前 12 anchor 是 hand-picked Pareto 验证用, 不能拿来训 LGB (因为 selection bias 偏向 boundary cases, 中间区域无样本). **训练数据必须 random_search 均匀采样**, **验证 Pareto 才用 hand-pick or NSGA-II Top-K**.

25. **没区分 "validation hand-pick" vs "training random"**: 两者目的完全不同 — train 要均匀, validation 要选 Pareto-relevant. 之前混淆这两个用例, 把 12 anchor 同时当 train + validation, 是错的.

26. **NSGA-II 和 LGB 混淆**: 用户问 "它们是不是一回事", 我才意识到第一次 explain 没明确两者角色 (LGB = predictor / fitness function, NSGA-II = optimizer / search algorithm). 它们是 orthogonal complementary, 不能混为一谈.

27. **"prune25 width_per_group 慢" 实证早该写到 framework constraint**: M4.9 v2 才把 `_check_resnext_width_pow2` 加进 `framework/constraints.py`. 这个 hardware-aware 知识 (Tensor Core IMMA fast-path) 早期 survey_raw_2 已经有, 但没及时落地到 constraint DSL, 导致 prune25 实测才暴露. 应该 hardware capability 调研后立刻 codify 到 constraint, 不等实测踩坑.

28. **Closed-loop demo 用 enumeration 而非 random_search**: M4.9 v2 closed_loop_v2.py 是 7 prune × 3 prec = 21 enumerate, 不是真正 random_search. 跟 paper claim "framework 在大空间内自动搜 Pareto" 形象不一致. 正确做法: random_search 100+ candidates + 大空间 + LGB 评估.

## 2026-05-11 反思(Plan2 验证后两个根本概念错误)

29. **搜索空间 ≠ Pareto frontier — 误把"实测不优"当"不纳入"理由**: Part A 验证 per-stage 量化 18/18 都比全 INT8 慢, v1 报告就直接给"Q1' 不纳入搜索空间"的决策. 用户指出: **搜索空间是所有物理可行 + 语义合法的配置集合**, Pareto frontier 是其中的最优子集. 如果先把"不优"删了再搜, 等于不需要搜索 — 18 个 per-stage 数据点恰恰是 LGB 的负样本, 是搜索空间必须包含的. **修正**: 4090 协同 954 → 14416 (Q 9→17 加 per-stage; D 1→8 加 D2/D3/D4). 搜索空间维度的验证目的是**给预测器提供数据**, 不是"删维"决策.

30. **整图测试结果错误归因为模型级**: Part B-Orin.2 整模型 DLA 慢 3.6× → v1 报告概括"V2X 模型与 DLA 不兼容". 用户指出: Pyramid 算法核心 = ResNeXt backbone (~99.9% 参数, 纯标准 Conv), **DLA 完全支持**. 3.6× 慢是因为整图含 V2X collab module (GridSample/Einsum/IsNaN/Where) 不在 DLA whitelist, 触发 massive fallback. **修正 C12** 必须 per-operator 级 (C12a V2X collab→GPU / C12b Conv→任选 / C12c VFE sparse→GPU), 而非 per-model. 整图测试结论不能直接概括为模型属性, 必须先做**子图分解**再归因.

31. **B-Orin.3 流水线 deferred 是因为 #30 错误归因 + 工程懒**: v1 报告 defer B-Orin.3 的两个理由 ("TRT 8.5 不支持 layerDeviceTypes" + "B-Orin.2 已证 DLA 不可行") 都被推翻. Python TRT API (`ILayer.set_device_type`) 仍可用, 整模型不行 ≠ backbone 子图不行. 正确做法: 先拆 ONNX backbone-only 子图, 再 strict DLA build, 才能验证 plan2 B-Orin.3 流水线方案 B/C 是否物理可行. v1 直接 defer 是工程惰性 + 概念错误叠加.

## 结论 (2026-05-10 更新)

最重要的 4 个反思 (按 framework 论点严重性):

- **#19 / #20** (搜索空间 over-claim): 影响 paper §1 "search space ~10^X" 数字, 必须修正
- **#21** (5-module schema 错位): 影响 framework adapter 设计, Pyramid 应该 1-module
- **#23 / #24** (LGB 训练数据需求): 影响 P0 工程量估计, 86 anchor 而非 22
- **#28** (closed-loop 用 enumeration): 影响 paper §C demo 的 evidence 等级, 必须真跑 random_search + LGB

**矫正方向**: P0 (Pyramid 单模型 86 anchor LGB) → P1 (跨模型 240 anchor) → P2 (NSGA-II + LGB 真 search) → P3 (制品化). 这才是 paper level minimum.
