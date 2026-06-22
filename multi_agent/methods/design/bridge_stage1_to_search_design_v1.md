# Bridge 设计: stage1 partition manifest → 阶段二搜索器 (v1, 仅设计不含实现)

> 目的: 把当前"人工保持口径一致"的 stage1↔搜索连接, 升级为**自动数据流** —— 让 `framework/stage1/run_scan.py` 产出的 partition manifest 直接驱动阶段二搜索器的**搜索空间与合法性约束**, 人工策展出环。
>
> ★**范围边界(诚实)**: 本 bridge 只接通"**空间定义 + 合法性约束**"这半边(manifest → 候选宽度/round_to/grouped_conv/coupled_buckets/routing)。**另一半"测量值"(latency LUT / AP model)仍来自真测/训练**(stage1 S5b 延迟刻画 + AP 预测器训练), 不在本 bridge 自动生成范围内。即: bridge 决定"**搜什么、哪些合法**", 真测决定"**每个配置多快/多准**"。
>
> ★★**普适性原则(本设计的硬约束, 回应"别只为某个模型")**: bridge **不得包含任何 per-model 分支**(`if model=='...'`)。普适靠三件事: ①**可建性 key 在 `round_to`** (`w%round_to==0`), round_to 已含 `lcm(int8_align,groups)`, 故不依赖整数组数等模型特定字段(§2/§3.1); ②**每字段防御性默认**, 缺失即退化为"最宽松且合法"语义, 缺字段的模型/未来扫描器照跑(§3.1b); ③**硬件能力一次解析、按需消费**, 软硬件特性先留接口(§3.4)。CoDriving 的"groups=1 / 无 DLA"行为**从默认自然涌现**, 不写一行特判 —— 这也是 §0.1 "可分离=测量非规则"在工程上的对应: 框架对任意架构走同一条路径。

---

## 1. 现状: 搜索器把单模型结构硬编码了 (代码证据)

阶段二搜索器 (`framework/search_three_arm.py`) 当前**不读 manifest**, 而是消费人工构建的 JSON, 且把 **Pyramid 的结构常量写死**:

| 搜索器现状 | 代码位置 | 写死了什么 |
|---|---|---|
| `SEED_GRID_JSON = gap1_grid_corrected.json` | `search_three_arm.py:67` | 人手挑的 8 个宽度点 (W_g 探针) |
| `candidate_widths(lut, apm)` | `:614` | 候选宽度来自 `apm.exact` keys + seed grid (人选的) |
| `QLookup.in_per_g(w) = s0 // GROUP_DIVISOR` | `:899` | **Pyramid 的 g=32 写死**; in_per_g = 每组输入通道 |
| `QLookup.can_build_int8(w) = in_per_g % 4 == 0` | `:903` | NCHWc dp4a 的 IC_BN≥4 规则**手写** |
| `enforce_int8_buildable` / `uniform_int8_speedup` | `:860 / :858` | 手设标志位 |
| **CoDriving 走子类重写** `QLookupCoDriving.in_per_g = w[0]` (groups=1) | `run_pqs_codriving.py` | **per-model 子类化** = 每换一个架构就手写一个子类 |

⇒ 病根: **结构信息(groups / IC_BN / round_to / 哪些通道同剪)在搜索器里是 per-model 手写的**, 而这些信息 **manifest 已经有了**。bridge 就是把"手写常量/子类"换成"读 manifest"。

---

## 2. manifest 已有字段 → 搜索器输入 的接口契约

stage1 `run_scan.py` 产出的 manifest (`framework/partitions/<model>_partition.yaml`) 相关视图 (均由 `graph_scan.py` 真实生成):

| manifest 字段 (已有) | graph_scan 出处 | → 替换搜索器的 | bridge 动作 |
|---|---|---|---|
| `view_b1_search_groups[].search_key` | `consolidate_search_groups` (S2b) | 旋钮身份 (per-stage / whole) | 旋钮枚举来源 |
| `view_b1_search_groups[].cur_width` + `.round_to_default` + `.max_rate` | S2/S2b | seed grid 的宽度候选 | **按 round_to 步长在 [floor, cur_width] 枚举合法宽度** → 取代 `gap1_grid_corrected` 手挑网格 |
| `view_b1_search_groups[].grouped_conv` (**bool**) | `consolidate_search_groups` (S2b, `grouped = any grouped`) | `QLookup` 写死的 GROUP_DIVISOR | 仅作"是否分组"标志; **不依赖整数组数**(整数 `grouped_conv_g` 只在 raw `view_b1_groups`, consolidated 不带) |
| `view_b1_search_groups[].round_to` (★**已 = `lcm(int8_align, groups)`**) | S2 `align_int8 = lcm(int8_align, g)` → S2b `round_to = max` | `can_build_int8` 手写 `in_per_g % 4` | ★**可建性 key 在 round_to**: `buildable ⟺ width % round_to == 0`。round_to 已把 grouped lcm 烤进去 → **无需整数组数, 天然跨模型** (CoDriving round_to=int8_align, Pyramid round_to=lcm)。取代 Pyramid 专用的 `in_per_g//32 % 4` |
| `view_b1_search_groups[].coupled_buckets` | `semantic_bucket` 并集 (S2) | (当前忽略) | **跨桶"必须同剪"约束** → 同一 coupled bucket 的旋钮联动 |
| `hw_capability.{int8_align,fp16_align,legal_bits,alignment_enforcement}` | `hardware_scan` | 手设常量 | `QLookup` 的合法位宽/对齐/hard-vs-soft |
| `view_b2_quant_units[].{quantizable,legal_granularity_w,per_channel_act}` | S3 | (当前 Q 轴手写) | Q 轴每单元合法粒度; heads 锁 FP16 |
| `view_d_routing_segments` | `routing_segments` (S4b) | (当前 D 轴未自动) | D 轴候选段 (GPU/DLA), op-blacklist 已在扫描判 |

---

## 3. bridge 模块设计 (新增, 不改搜索算法)

新增一个**纯加载/转换层** `framework/stage1_bridge.py` (建议名), 单向: manifest → 搜索器可消费的内存结构。**不触碰** `search_three_arm.py` 的核心搜索逻辑(三臂/NSGA/block-LUT), 只替换它的**输入构造**。

### 3.1 核心数据结构 (从 manifest 派生)

```
SpaceSpec (bridge 产出, 搜索器消费):
  knobs:        list[KnobSpec]          # 来自 view_b1_search_groups (consolidated)
  hw:           HwCapability            # 来自 hw_capability
  quant_units:  list[QuantUnitSpec]     # 来自 view_b2_quant_units
  routing:      list[RoutingSegment]    # 来自 view_d_routing_segments
  couplings:    list[set[knob_id]]      # 来自 coupled_buckets (必须同剪组)

KnobSpec:
  knob_id, search_key, cur_width, floor, round_to, grouped_conv(bool), max_rate, enforcement
  legal_widths():  [w for w in range(floor, cur_width+1, round_to) if (cur_width-w)/cur_width <= max_rate]
  buildable_int8(w): w % round_to == 0          # ★round_to 已含 lcm(int8_align,groups); 不需整数组数, 跨模型普适
```

### 3.1b ★防御性默认 (普适关键: 任何字段缺失/视图不齐都不崩)

bridge **不假设任何字段存在**。从 manifest 取每个字段都走"取值 or 默认", 缺失即退化为"最宽松且合法"的语义, 使**缺字段的模型/未来扫描器也能跑**:

| 字段 | 缺失/null 时默认 | 默认的语义(为何安全) |
|---|---|---|
| `round_to` | `hw.int8_align` (再缺 → 1) | 退化为硬件最小对齐; =1 即"无对齐约束" |
| `grouped_conv` | `false` | 标准卷积; 配合 round_to=int8_align ⇒ 可建性只看 `w%int8_align` |
| `max_rate` | `1.0` | 不额外限剪 (仍受 floor 约束) |
| `floor` | `round_to` (再缺 → 1) | 退化为最小合法宽度 |
| `coupled_buckets` | `[]` / `null` → 无耦合 | 该旋钮独立, 不与他者联动 |
| `enforcement` | `"hard"` | 保守: 默认严格对齐(宁可少建也不出错核) |
| `legal_bits` (hw) | `[16]` | 缺硬件能力 → 只允 FP16 (最保守) |
| `view_d_routing_segments` | 整网单段 GPU | 无 DLA → 路由轴退化(与硬件扫描 DLA-absent 一致) |

> 实现守则: bridge 解析层用 `m.get(field, DEFAULT)` 而非 `m[field]`; 每个默认在此表登记并在代码注释引用本表。**绝不 per-model 分支** —— CoDriving 的"groups=1/无 DLA"行为**从默认自然涌现**, 不写任何 `if model=='codriving'`。

### 3.2 三个替换点 (搜索器侧, 最小改动)

1. **候选宽度**: `candidate_widths(lut, apm)` → `SpaceSpec.enumerate_widths()` (笛卡尔积 over knobs.legal_widths(), 受 couplings 约束). 保留旧函数作 fallback。
2. **int8 可建性**: `QLookup.in_per_g/can_build_int8` 改为 `KnobSpec.buildable_int8(w) = (w % round_to == 0)`, 删掉写死的 `GROUP_DIVISOR` 与 `% 4`。**删除 `run_pqs_codriving.py` 的 QLookupCoDriving 子类** —— round_to 路径同时正确处理 Pyramid(round_to=lcm) 与 CoDriving(round_to=int8_align), 无需任何整数组数或子类。
3. **hw 约束**: `QLookup` 的对齐/合法位宽/粒度从 `SpaceSpec.hw` 注入(见 §3.4), 不再手设。

### 3.3 向后兼容 (关键, 降风险)

- bridge 是**可选注入**: 搜索器入口加 `--manifest <path>`; 给了就用 `SpaceSpec`, 不给就走现有手工 JSON 路径(完全不变)。
- 这样**现有实验可继续复跑**(回归基线不动), 同时新模型可走自动路径。一步步迁移, 不是大爆炸重写。

### 3.4 ★硬件扫描 bridge (hw_capability → 搜索器; 现在就建骨架, 即便部分暂未消费)

manifest 的 `hw_capability` (由 `hardware_scan.py` 产, 形式化 = $\mathcal{H}=\langle\mathcal{I},\mathbf{a},\mathcal{Q},\mathcal{T}\rangle$) 也要有 bridge —— 不能让搜索器继续手写硬件常量。映射:

| `hw_capability` 字段 | 形式化符号 | → 搜索器用途 | 现状 |
|---|---|---|---|
| `int8_align` / `fp16_align` | $\mathbf{a}$ | 各旋钮 round_to 默认 + 可建性兜底 | 立即消费 |
| `legal_bits` (如 [8,16]) | $\mathcal{Q}$.bits | Q 轴合法位宽集合 | 立即消费 |
| `legal_granularity_w` / `per_channel_act` | $\mathcal{Q}$.gran | Q 轴合法粒度 | 立即消费 |
| `alignment_enforcement` (hard/soft/auto) | — | hard=违反即剔除; soft=允许但记惩罚; auto=按核回退 | 立即消费(hard 路径) |
| `dla_whitelist` / IP 枚举 $\mathcal{I}$ | $\mathcal{I},\mathcal{T}$ | D 轴设备候选 + 路由段 | **建骨架, 暂可只走 GPU**(无 DLA 时退化) |
| 未来: 多目标 IP / batch regime | $\mathcal{I}$ 扩展 | 吞吐/能耗轴、设备并行 | **预留接口, 暂不消费** |

**设计要求**: `HwCapability` 数据类把上述字段全部解析进来(缺失走 §3.1b 默认), 即使搜索器当前只用到对齐/位宽/粒度三项 —— **`alignment_enforcement` 的 soft/auto 分支、DLA 路由、IP 枚举先留好字段与接口**, 后续启用硬件轴搜索时零改 bridge。这样硬件能力是"一次解析、按需消费", 而非每加一个硬件特性就回搜索器手写常量。

---

## 4. bridge 接通后, 搜索"有什么不同" (回答用户)

| 层面 | 之前(手工) | 之后(bridge) |
|---|---|---|
| **流程** | 人读图→手写宽度/round_to/子类→JSON→搜索 | run_scan→manifest→bridge→搜索; **人出环, 新模型零手工** |
| **空间形状** | seed grid 是人预剪的 ~10^4 (可能过剪/樱桃挑) | 由 manifest consolidation **自动枚举** (round_to 步长 × [floor,cur_width]); 形状可能与手挑网格不同 → **须重跑确认结论稳健** |
| **合法性约束** | `in_per_g%4`、groups 写死, coupled 忽略 | per-block grouped_conv_g / align_int8(含 lcm) / coupled_buckets **数据驱动** |
| **多架构** | 每模型一个 QLookup 子类 | **一条代码路径**吃任意架构 |
| **核心搜索算法** | 三臂/NSGA/block-LUT | **完全不变** |

> ⇒ bridge 的学术意义: 把"在一个**人工策展**空间上验证了搜索器"升级为"**端到端自动**协同搜索 (导入网络→自动建空间→搜索)"。这也让 §0.1 的"耦合按架构测量、可分离=测量结果"真正落地 —— 因为搜索器现在**自动知道**每个架构的 groups/IC_BN, 才能对任意新网络算出耦合分数。

---

## 5. 验证计划 (bridge 接通后必做)

1. **回归**: 不给 `--manifest` 时, 搜索结果与现有基线**逐位一致** (证明没破坏旧路径)。
2. **等价性**: 给 Pyramid manifest 时, bridge 自动枚举的空间应**覆盖**现有手挑 seed grid 的关键宽度; 三臂消融结论 (A-joint>A-serial, Wilcoxon p) 在**自动空间**上**仍成立** (更强证据)。
3. **跨架构**: 给 CoDriving manifest, **不写任何子类**, 搜索器应自动得出 in_per_g=width(groups=1)、int8 恒可建、耦合分数≈1(可分离) —— 验证"测量而非规则"。
4. **新架构 smoke**: 给 V2X-ViT manifest, 跑通空间构建 (即使 LUT/AP 待测), 证明 bridge 对未见架构不报错。

---

## 6. 不在本 bridge 范围 (诚实划界)

- **测量值生成**: latency LUT 的真测 (stage1 S5b `latency_profile.py`) 与 AP 预测器训练 —— 是另一条数据流, bridge 只消费其产物, 不生成。
- **耦合分数预测器 + 自适应内环预算**: 是 bridge 之后的下一步 (用三臂消融标定的连续耦合分数驱动分流), 见 §0.1。bridge 是它的前置(搜索器得先能读结构, 才能算分数)。
- **consolidation 规则本身的泛化**: 当前 `consolidate_search_groups` 的 per-stage 绑定规则 (backbone/bev_encoder→stage, neck/heads→whole) 是否对任意架构最优, 属 stage1 扫描的增强, 不在 bridge。

---

## 7. 落地次序建议

1. 写 `framework/stage1_bridge.py` (SpaceSpec + manifest 解析), 纯函数, 带单测。
2. 搜索器加 `--manifest` 可选注入 + SpaceSpec 路径, 保留 fallback。
3. 删 `run_pqs_codriving.py` 的 QLookupCoDriving 子类, 验证 manifest 路径等价。
4. 跑 §5 的 4 项验证。
5. (下一阶段) 耦合分数预测器 + 自适应预算。

---

## 8. 实现勘误 (2026-06-22, bridge 已构造并 4/4 验收后回写)

设计 §3.1 原写"可建性 key 在 `round_to`(`w % round_to == 0`, round_to 已含 `lcm(int8_align, groups)`)"。**实现时两处订正**(都靠跨模型实测交叉核验抓到):

1. **可建性须与剪枝粒度 `round_to` 分离 —— 单独字段 `int8_buildable_align`**。若把可建对齐 overload 进 round_to(分组旋钮 32→128), 则 128-宽旋钮 `floor=min(128,128)=128` → `max_rate` 被零化 → 旋钮失去可剪性。故 `round_to` 仍是剪枝粒度(=`lcm(int8_align, groups)`, 分组实际 = 32), 而 int8 可建性走 stage1 新增字段 `int8_buildable_align`。一个宽度可"合法可剪(round_to)却不可建 int8(int8_buildable_align)" —— 这正是耦合陷阱的载体。

2. **可建对齐 = `pack_factor × groups`, 不是 `lcm(int8_align, pack_factor×groups)`**。`int8_align=32` 是 **TRT 张量 tiling 对齐**(进 round_to), **不进 TVM dp4a/WMMA 可建性**(TVM 路径无 32-floor, 只需 per-group 打包 `in_per_g % pack_factor == 0`)。判据: CoDriving(g=1)实测 `codriving_int8_verify.json` Cin=48/16 **全 build** → 4×1=4(全可建); 若叠 `lcm(32,4)=32` 会假判 48/16 不可建, 与实测矛盾。Pyramid(g=32) `4×32=128 = lcm(32,128)` 两公式同值, 故先前被 masked、未暴露。

**落地**: `graph_scan.py` 算 `int8_buildable_align = hw.int8_pack_factor * g`(剪枝组 + consolidated 旋钮 lcm 聚合); `hardware_scan.py` + `rtx4090.yaml` 加 `int8_pack_factor: 4`; `stage1_bridge.py` `KnobSpec.buildable_int8(w) = (w % int8_buildable_align == 0)`; 搜索器 `build_from_manifest()` + `key_scale`(cur_width=key_scale×num_filters) + QLookup `bridge_int8_align` 原生谓词。**§5 验收 4/4 PASS**(回归/等价 p=4.88e-4 不变, CoDriving 无子类 SERIAL, V2X-ViT smoke 通)。

> 普适性铁律未变: 仍零 per-model 特判; CoDriving 的 groups=1 / 无 DLA 行为从默认涌现。订正只是把"可建对齐"从错误的 lcm 公式改成实测验证的 `pack_factor×groups`, 且与剪枝粒度解耦。
