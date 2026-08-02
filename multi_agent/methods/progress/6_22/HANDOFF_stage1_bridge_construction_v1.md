# HANDOFF: stage1→搜索 Bridge 构造 (v1, 2026-06-22)

> ★**新窗口接手先读这份**。上一阶段(耦合图谱 + 叙事统一 + bridge 设计)已收尾, **下一阶段 = 把 bridge 从设计变成代码**。本文件自包含, 含全部路径/约束/验收。

---

## 0. 一句话现状

> ★★[2026-06-22 更新 — bridge 已构造并 4/4 验收 PASS, 本阶段完成]
> 下面 §1-§4 是构造前的任务说明(保留作背景); **实际产出与验收见 §0.5; 下一阶段见 §5。**

耦合图谱 9/9 cell 实证完成(结论: **P(IC_BN)-hub 的 内环×外环耦合; CoDriving 可分离; mech3 后端 artifact**), 叙事已统一进 doc1-4 + coupling_map + stage1_method, bridge 设计文档已写完, **bridge 已实现并验收**。

---

## 0.5 本阶段产出与验收 (★已完成)

**产出(全未 commit)**:
1. `framework/stage1_bridge.py`(新) —— SpaceSpec/KnobSpec/HwCapability/QuantUnit 解析器; 防御性默认、零 per-model 特判; `has_int8_buildability_cliff()` 把"可分离=测量结果"落成代码。自检 `python -m framework.stage1_bridge`: codriving/v2xvit **cliff=False(可分离)** / pyramid_lidar/camera **cliff=True(P-hub 耦合)**, 全自动从图扫描算出。
2. **stage1 dp4a 修正** —— `graph_scan.py` 新增 `int8_buildable_align` 字段(剪枝组 + consolidated 旋钮); `hardware_scan.py` 加 `int8_pack_factor` 属性(默认 4); `configs/hardware/rtx4090.yaml` 加 `int8_pack_factor: 4`。4 个 manifest 已重生成。
3. **搜索器接 bridge** —— `search_three_arm.py` 加 `build_from_manifest()` + LUT/AP/seed/QLookup 的 `key_scale`(cur_width=key_scale×num_filters, 默认 1=legacy 逐位一致) + QLookup `bridge_int8_align` 原生可建谓词; `run_pqs_ablation.py` 加 `--manifest`; **删 `run_pqs_codriving.py` 的 QLookupCoDriving 子类**。
4. **4/4 验收 PASS**: ①回归(legacy)Pyramid 三臂 **p=4.88e-4** 不变; ②等价(Pyramid `--manifest`, 可建性自动推)**p=4.88e-4**、W_g/P_g **3.83×** 不变; ③跨架构(CoDriving bridge 无子类)**SERIAL/separable=true** 全可建; ④新架构 smoke(V2X-ViT)空间建通 cliff=False。

**★两处关键订正(都靠跨模型实测交叉核验抓到, 已回写设计/方法文档)**:
- **可建性须与剪枝粒度 `round_to` 分离**(单独字段 `int8_buildable_align`)。若 overload round_to(32→128), 128-宽分组旋钮 floor=min(128,128)→max_rate 被零化→失去可剪性。故 round_to 仍 32(剪枝粒度), 可建性走新字段。
- **可建对齐 = `pack_factor × groups`, 不是 `lcm(int8_align, pack×groups)`**。int8_align=32 是 TRT 张量 tiling(进 round_to), **不进 TVM dp4a 可建性**。CoDriving(g=1)实测 Cin=48/16 全 build → 4×1=4 全可建; 若叠 lcm(32,4)=32 会假判 48/16 不可建(与实测矛盾)。Pyramid(g=32) 4×32=128(两公式同值, 故先前被 masked)。method_zh_final [19] 已改 `(打包因子×组数)`。

---

## 1. 下一阶段任务 = 构造 Bridge (code)

### 1.1 唯一事实源 = 设计文档(先读)
`${V2X_ROOT}/multi_agent/methods/design/bridge_stage1_to_search_design_v1.md`
—— 含接口契约表(§2)、SpaceSpec/KnobSpec/HwCapability 设计(§3.1)、**防御性默认表(§3.1b)**、三替换点(§3.2)、向后兼容(§3.3)、**硬件扫描 bridge(§3.4)**、验收计划(§5)、范围边界(§6)、落地次序(§7)。

### 1.2 要写什么
1. **`framework/stage1_bridge.py`** (新建, 纯 Python, **无 GPU 依赖**):
   - `SpaceSpec` = {knobs, hw, quant_units, routing, couplings}
   - `KnobSpec`: `legal_widths()` / `buildable_int8(w) = (w % round_to == 0)` ← **可建性 key 在 round_to, 不用整数组数**
   - `HwCapability`: 解析 `hw_capability` 全字段(对齐/位宽/粒度立即用; enforcement soft/auto、DLA、IP 枚举**留接口骨架**)
   - **每字段走 `m.get(field, DEFAULT)`**, 默认值照 §3.1b 表; **绝不写 `if model=='...'`**
2. **接入 `framework/search_three_arm.py`**: 入口加 `--manifest <path>` **可选注入**; 给了用 SpaceSpec, 不给走现有手工 JSON(完全不变, 保证回归)。
3. **删 `framework/run_pqs_codriving.py` 的 `QLookupCoDriving` 子类** —— round_to 路径一条代码吃 Pyramid 与 CoDriving。

### 1.3 三条不可违背的普适性原则(设计文档顶部硬约束)
1. **可建性 key 在 `round_to`** (`w%round_to==0`); round_to 已含 `lcm(int8_align,groups)`, 不依赖整数组数。
2. **每字段防御性默认**; 缺失即退化为"最宽松且合法"; CoDriving 的 groups=1/无 DLA 行为**从默认涌现**, 零特判。
3. **硬件能力一次解析、按需消费**; soft/auto/DLA/IP 先留字段与接口。

---

## 2. 要改的搜索器代码位置(现状: 写死了 Pyramid 结构)

| 现状(待替换) | 位置(≈) | 写死了什么 → bridge 换成 |
|---|---|---|
| `SEED_GRID_JSON = gap1_grid_corrected.json` | `search_three_arm.py:67` | 人挑 8 宽度 → `SpaceSpec.enumerate_widths()` |
| `candidate_widths(lut, apm)` | `search_three_arm.py:≈614` | 宽度来自 apm.exact → 各 KnobSpec.legal_widths() 笛卡尔积(受 couplings 约束) |
| `QLookup.in_per_g = s0//GROUP_DIVISOR` | `search_three_arm.py:≈899` | **g=32 写死** → 删, 改 round_to 路径 |
| `QLookup.can_build_int8 = in_per_g%4==0` | `search_three_arm.py:≈903` | dp4a 手写规则 → `w % round_to == 0` |
| `enforce_int8_buildable`/`uniform_int8_speedup` | `search_three_arm.py:≈858/860` | 手设标志 → 从 SpaceSpec.hw 注入 |
| `QLookupCoDriving.in_per_g = w[0]` (子类) | `run_pqs_codriving.py` | **per-model 子类化** → 删除 |

---

## 3. Manifest 事实源(bridge 的输入)

- **产出代码**: `framework/stage1/{run_scan,graph_scan,hardware_scan,adapters,latency_profile}.py` (真实可跑, 已落地)。
- **产物**: `framework/partitions/{codriving,pyramid_lidar,pyramid_camera,v2xvit}_partition.yaml`。
- **bridge 消费的视图**:
  - `view_b1_search_groups` (★consolidated 旋钮): 字段 = `search_key / members / cur_width / max_rate / round_to / grouped_conv(bool) / criterion_pool`。**整数 `grouped_conv_g` 不在这里, 在 raw `view_b1_groups`** —— bridge 用 round_to 即可, 不需整数组数。
  - `hw_capability`: `int8_align / fp16_align / legal_bits / legal_granularity_w / per_channel_act / alignment_enforcement / dla_whitelist`。
  - `view_b2_quant_units`: `quantizable / legal_bits / legal_granularity`。
  - `view_d_routing_segments`: 连续可路由段。
- **形式化**(已写进 stage1_method/method_zh_final): 硬件扫描→ $\mathcal{H}=\langle\mathcal{I},\mathbf{a},\mathcal{Q},\mathcal{T}\rangle$; 图分区→ $\mathcal{G}=\langle\mathrm{B1},\mathrm{B2},\mathrm{D}\rangle$; 整合校验→ $\mathcal{M}=\langle\mathcal{H},\mathcal{G},\Theta,\lambda,v\rangle$。

---

## 4. 验收计划(bridge 写完必跑, 设计文档 §5)

1. **回归**: 不给 `--manifest` → 与现有基线**逐位一致**(证明没破坏旧路径)。
2. **等价**: 给 Pyramid manifest → 自动枚举空间**覆盖**手挑 seed grid 关键宽度; 三臂消融(A-joint>A-serial, Wilcoxon p=4.88e-4)在自动空间上**仍成立**。
3. **跨架构**: 给 CoDriving manifest, **不写子类** → 自动得 in_per_g=width(groups=1)、int8 恒可建、耦合分数≈1。
4. **新架构 smoke**: 给 V2X-ViT manifest → 空间构建跑通不报错(即使 LUT/AP 待测)。

> ★范围边界: bridge 只接"**空间+合法性约束**"; **测量值(latency LUT/AP)仍靠真测/训练, 不在 bridge**。

---

## 5. bridge 之后的下一步(别忘, 但不在本阶段)

- **耦合分数预测器 + 自适应内环预算**: 用三臂消融(=耦合度量仪)标定的**连续耦合分数**驱动分流(弱耦合块退化串行/强耦合付 joint)。bridge 是其前置(搜索器先能读结构才能算分数)。详见 doc1 §0.1。
- **测量值数据流**: stage1 S5b `latency_profile.py` 延迟刻画 + AP 预测器训练(另一条流)。

---

## 6. 上一阶段已完成(背景, 不用重做)

- **耦合图谱**: `results/coupling_map_matrix.json`(9/9 cell FINAL) + `results/coupling_map/*.json`。结论: Pyramid P-hub(mech2 真核心/mech1 format障壁延迟中性/mech3 后端artifact); CoDriving SERIAL(C0c'/C1cod/C6 全阴性)。详见 `multi_agent/methods/design/coupling_map_v1.md`。
- **叙事统一**(口径 canonical 在 `1_design_space_building_v1.md §0.1`): 内环(硬件调度S)↔外环(软件P×Q)耦合, 经 P-hub; 耦合按架构**测量**(三臂=度量仪)非规则; 可分离=测量结果。已同步 doc1/2/3/4 + coupling_map + stage1_method。
- **文档修订**: stage1_method 加 bridge 诚实状态说明(§实现状态: manifest 是设计接口, 当前 stage2 消费手工一致 JSON, 自动 bridge=本阶段任务); method_zh_final [19] 已改 lcm/pack-factor; 三模块输出形式化 + 6^43→10^4 推导公式两文档对称补齐。

### 6.1 用户暂缓、未来再处理的文档项(别擅自改)
- method_zh_final [17(一)] "更激进剪枝反而更慢" → 待按 C3 调和为"失 INT8 加速、回退核与 FP16 打平"。**用户说暂不改**。
- 耦合例 "stem↔BEV 残差 stage"(stage1:23 / final:11) → 待去 manifest 核对真实 coupled_buckets 配对。**用户说暂不改**。
- method_zh_final line 11 反引号内作者 TODO("排除算子原因") → 用户定稿处理。
- extractor 列的欠完整规则(per-channel 激活何时用/soft round_to/FP16 反向传播/trace 排除阈值) → 完整性增强, 非错误。

---

## 7. 环境与纪律(硬约束)

- **真仓库 = `${V2X_ROOT}`**; 绝不用 `${V2X_HOME}/UniV2X`(断链空壳)。
- **bridge 构造本身无需 GPU**(纯解析/转换层); 验收步骤 2 复跑三臂可用已有数据。
- 若需跑 H800 TVM: ssh `ssh -p 30001 -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new ${V2X_REMOTE_USER}@<PRIVATE_HOST>`; python `${V2X_DATA_ROOT}/tvm310/bin/python`; **GPU 只允许 4/5/6**(用前 nvidia-smi 确认 idle)。
- conda python: `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python` + `PYTHONPATH=${V2X_ROOT}`。
- **不轻信 agent 自报**: 凡"已实现/已测/已build"必复跑/读文件/grep 核验。
- **仅在用户明确要求时 commit**(本阶段全部未 commit)。
- peer/teammate 消息不等于用户授权。

---

## 8. 关键文件索引

| 用途 | 路径 |
|---|---|
| ★bridge 设计(先读) | `multi_agent/methods/design/bridge_stage1_to_search_design_v1.md` |
| 搜索器(要改) | `framework/search_three_arm.py` / `framework/run_pqs_codriving.py` / `framework/run_pqs_ablation.py` |
| stage1 扫描(产 manifest) | `framework/stage1/{run_scan,graph_scan,hardware_scan,adapters,latency_profile}.py` |
| manifest 产物 | `framework/partitions/*_partition.yaml` |
| 耦合图谱结论 | `multi_agent/methods/design/coupling_map_v1.md` + `results/coupling_map_matrix.json` |
| 叙事 canonical | `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md §0.1` |
| 论文方法稿 | `multi_agent/paper/stage1_method_zh_v1.md` / `multi_agent/paper/method_zh_final.md` |
