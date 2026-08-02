# Lane C Orin Stage6 Five-Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Orin 上完成 31 号文档第 11.3、14.3 节中 Pyramid 与 CoDriving 各五个可执行代表配置的 AP、multiscale-backbone latency 和 energy 测量，并把最新证据回填 34 号文档第 7 节。

**Architecture:** 从 H800 只读取回 SHA 锁定的 checkpoint、config、ONNX 和 INT8 calibration，不复制 H800 engine/timing cache；在 Orin TensorRT 8.5.2.2 上逐配置本地构建。`original/default` 使用 native PyTorch/CUDA 同边界测量，其余配置使用 Orin 本地 TensorRT；所有 AP 都在 Orin full_1789 路径执行。

**Tech Stack:** Python、PyTorch、ONNX、TensorRT 8.5.2.2、CUDA event、tegrastats、HEAL/OpenCOOD、DAIR-V2X。

## Global Constraints

- 五配置必须逐字来自 31 号文档第 11.3、14.3 节，不执行无合法配置的 `tune -> compress`。
- Pyramid 配置为 `(64,128,256,fp32)` native、`(24,128,64,fp16)`、`(64,128,256,fp32)` TRT、`(32,48,128,fp16)`、`(16,32,64,int8)`。
- CoDriving 配置为 `(64,128,256,fp32)` native、`(32,64,96,fp16)`、`(64,128,256,fp32)` TRT、`(32,80,128,fp16)`、`(16,32,64,int8)`。
- H800 engine 和 timing cache 不可复制；TRT engine 必须在配置的 Orin 目标（`$LANE_C_ORIN_HOST`）本地 fresh build。
- latency 主合同为 batch=2、warmup=20、iters=300、repeat=5、CUDA event、multiscale-backbone compute、no data transfer。
- AP 主合同为 full_1789，只替换对应 multiscale backbone/resnet，0 silent fallback。
- energy 必须由同一 latency scope/window 的 Orin tegrastats 命名 rail 得到；rail 不可访问时写 `missing_tegrastats_power_rails`，不得估算或复用历史数值。
- H800 源指标与 Orin 实测列必须分离，不输出跨硬件 speedup。
- 全部源文件、engine、build log、inspector、AP report 和结果表必须记录 SHA。
- 当前工作树已有用户修改，不重置、不覆盖无关文件、不提交。

---

### Task 1: 冻结十行实验清单与源资产

**Files:**
- Create: `results/lane_c_orin_stage6_five_config_20260724/contracts/five_config_manifest.json`
- Create: `results/lane_c_orin_stage6_five_config_20260724/source/SHA256SUMS.txt`
- Modify: `multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/34_7_14_冷启动文档_LaneC_Orin快速真测与映射表_v1.md`

**Interfaces:**
- Consumes: 31 号文档第 11.3、14.3 节和 H800 source evidence JSON。
- Produces: 十行 immutable manifest，供所有 build/AP/latency 任务读取。

- [ ] **Step 1:** 从 H800 evidence bundle 提取十行 H800 源指标、checkpoint/ONNX/calibration 路径与 SHA。
- [ ] **Step 2:** 只下载 checkpoint、config、ONNX、calibration 和 source evidence，不下载 engine/cache。
- [ ] **Step 3:** 本地逐文件校验 SHA，并生成 manifest；任一不一致时将对应行标为 `source_sha_mismatch`。
- [ ] **Step 4:** 检查 Orin 目标目录磁盘、TensorRT/PyTorch 环境和现有进程，写 preflight。

### Task 2: 建立通用 Orin 五配置测量入口

**Files:**
- Modify: `tools/orin_deploy/lane_c_backbone_parity_runner.py`
- Modify: `scripts/stage3_trt_multiscale_ap_bridge_v3.py`
- Modify: `scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py`
- Create: `tools/orin_deploy/lane_c_stage6_five_config.py`
- Create: `framework/tests/test_lane_c_stage6_five_config.py`

**Interfaces:**
- Consumes: Task 1 manifest。
- Produces: source gate、Orin fresh build、native/TRT latency、power 状态、AP command 和结果标准化函数。

- [ ] **Step 1:** 先写失败测试，覆盖十行配置、native/TRT 路径分离、输入/output channel 合同、禁止 H800 engine、缺 rail 降级和结果字段。
- [ ] **Step 2:** 运行测试确认 RED。
- [ ] **Step 3:** 实现最小通用入口；保持现有 Lane C runner 向后兼容。
- [ ] **Step 4:** 运行单元/集成测试确认 GREEN，并执行 `py_compile`。

### Task 3: 执行 Pyramid 五配置

**Files:**
- Create: `results/lane_c_orin_stage6_five_config_20260724/pyramid/<arm>/`

**Interfaces:**
- Consumes: Pyramid 五行源资产、Orin DAIR 数据和 Task 2 runner。
- Produces: 五行 build/numerical/latency/power/AP 证据。

- [ ] **Step 1:** 对 native original 执行同边界 20/300/5 CUDA-event latency 和 full_1789 native AP。
- [ ] **Step 2:** 对 compression-only、schedule-only、compress-then-tune、joint 分别在 Orin fresh build；INT8 使用该配置自己的 calibration。
- [ ] **Step 3:** 每个 TRT engine 先做 reference 数值 gate，再执行 20/300/5 latency/power。
- [ ] **Step 4:** 每个 TRT 配置执行 full_1789 AP bridge，记录 processed/failed/fallback 和三层输出比较。
- [ ] **Step 5:** 逐配置生成 terminal receipt 和 SHA 清单。

### Task 4: 执行 CoDriving 五配置

**Files:**
- Create: `results/lane_c_orin_stage6_five_config_20260724/codriving/<arm>/`

**Interfaces:**
- Consumes: CoDriving 五行源资产、Orin DAIR 数据和 Task 2 runner。
- Produces: 五行 build/numerical/latency/power/AP 证据。

- [ ] **Step 1:** 对 native original 执行 resnet 同边界 20/300/5 CUDA-event latency 和 full_1789 native AP。
- [ ] **Step 2:** 对其余四行分别在 Orin fresh build，严格绑定各自 output channels 和空间 shape。
- [ ] **Step 3:** 每个 TRT engine 执行 reference 数值 gate 与 20/300/5 latency/power。
- [ ] **Step 4:** 每个 TRT 配置执行 full_1789 AP bridge，记录 1789/failed/fallback 和全部输出比较。
- [ ] **Step 5:** 逐配置生成 terminal receipt 和 SHA 清单。

### Task 5: 汇总、回填与审阅

**Files:**
- Create: `results/lane_c_orin_stage6_five_config_20260724/orin_stage6_five_config.csv`
- Create: `results/lane_c_orin_stage6_five_config_20260724/final_summary.json`
- Create: `results/lane_c_orin_stage6_five_config_20260724/summary.md`
- Modify: `multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/34_7_14_冷启动文档_LaneC_Orin快速真测与映射表_v1.md`

**Interfaces:**
- Consumes: Tasks 3、4 的十行 terminal receipt。
- Produces: 34 号文档第 7 节最终十行 Orin 表和审计结论。

- [ ] **Step 1:** fail-closed 汇总十行 AP/latency/energy/status，禁止用 H800 值补 Orin 空缺。
- [ ] **Step 2:** 回填第 7 节并追加 source、AP、latency、energy 口径说明和 failure ledger。
- [ ] **Step 3:** 运行全套相关测试、JSON 校验、SHA 校验和 `git diff --check`。
- [ ] **Step 4:** 独立审阅表格、证据绑定、时延边界和功耗状态；修复全部 CRITICAL/HIGH/MEDIUM 问题。
