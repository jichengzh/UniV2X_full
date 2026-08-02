# Stage1 Model Scanner V2X 关联索引

> 可移植执行约定：下文保留的 `/home/jichengzhi/V2X` 是历史实验记录；在新 checkout 中以 `$V2X_ROOT` 替换，并先按 [`REPRODUCIBILITY.zh-CN.md`](REPRODUCIBILITY.zh-CN.md) 设置环境变量。

更新时间: 2026-06-24

索引关键字: `stage1-model-scanner-aaai`, `Stage1 Model Scanner`, `stage1_index`, `framework/stage1`, `stage1_model_predict`, `AUTO_REGISTRY`, `REGISTRY`

## 1. 结论

`/home/jichengzhi/V2X/github/stage1-model-scanner-aaai` 是一个抽出的 Stage1 开源运行库，核心功能是 dense-core graph scan、硬件 capability 过滤、S2/S2.5/S3/S4 证据读取、coupling predictor 和模型三分类。

在当前 V2X 工作区中，它对应的主工作副本不是 `github/` 目录本身，而是 V2X 根目录下这些路径:

| 类型 | V2X 位置 | 用途 |
|---|---|---|
| GitHub 项目源 | `github/stage1-model-scanner-aaai/` | 外部/可迁移版本，对照源 |
| Stage1 核心代码 | `framework/stage1/` | scanner、adapter、auto trace、predictor、classifier |
| 共享 schema/bridge | `framework/capability_schema.py`, `framework/stage1_bridge.py` | 硬件 YAML 校验、manifest 到搜索空间转换 |
| CLI 脚本 | `scripts/stage1_*.py` | 分类、coupling 预测 |
| Phase2 证据脚本 | `scripts/phase2/stage1_*.py` | S2/S2.5/S3/S4 证据生成与报告 |
| Trace 原型工具 | `tools/configurable/depgraph_*.py` | Pyramid/V2X-ViT DepGraph wrapper |
| 硬件输入 | `configs/hardware/*.yaml` | `run_scan --hw` 输入 |
| Stage1 manifest | `framework/partitions/*_partition.yaml` | scan 输出，也是后续搜索入口 |
| Stage1 证据与报告 | `results/stage1_model_predict/` | predictor/classifier 证据、输出、报告 |
| Stage1 测试 | `framework/tests/test_stage1_*.py`, `framework/tests/test_*predictor*.py` | 更新后的回归测试 |
| 设计/论文关联 | `multi_agent/methods/design/stage1*`, `multi_agent/methods/design/stage1-model-predict/` | 历史设计、实现计划、验证说明 |

## 2. 项目功能分层

| 层 | 核心文件 | 说明 |
|---|---|---|
| 硬件能力 | `framework/capability_schema.py`, `framework/stage1/hardware_scan.py`, `configs/hardware/*.yaml` | 读取并规范化硬件 capability，供 scan 过滤非法结构 |
| 模型接入 | `framework/stage1/adapters.py`, `framework/stage1/auto_trace.py` | 手写 `TraceAdapter` 和自动 `AutoTraceAdapter` 注册模型 |
| 图扫描 | `framework/stage1/graph_scan.py`, `framework/stage1/latency_profile.py`, `framework/stage1/run_scan.py` | 生成 `framework/partitions/<model>_partition.yaml` |
| 搜索桥接 | `framework/stage1_bridge.py` | 将 Stage1 manifest 转成搜索器可消费的 `SpaceSpec` |
| 预测器 | `framework/stage1/coupling_predictor.py`, `framework/stage1/calibrated_predictor.py` | 根据 manifest 和 evidence 生成 coupling 风险预测 |
| 模型分类 | `framework/stage1/model_classifier.py`, `scripts/stage1_classify_models.py` | 输出 `CO_ACCELERATION_REQUIRED` / `SEPARABLE_ACCELERATION` / `SCAN_FAILED` |
| 证据闭环 | `scripts/phase2/stage1_*.py`, `results/stage1_model_predict/` | S2/S2.5/S3/S4 evidence 的生成和读取 |

## 3. GitHub 项目文件到 V2X 的对应关系

状态含义:

- `identical`: GitHub 子项目与 V2X 根目录同路径文件内容一致。
- `different`: 两边有同路径文件，但存在差异，更新时需要人工 review。
- `missing`: V2X 根目录没有同路径文件。
- `different-semantic`: 同名但不是同一语义的文件，不能直接覆盖。

| 状态 | GitHub 项目路径 | V2X 对应位置 | 更新建议 |
|---|---|---|---|
| missing | `docs/add-new-model-adapter.zh-CN.md` | `docs/add-new-model-adapter.zh-CN.md` | 可考虑复制到 V2X `docs/` 或保持只在 GitHub 子项目中维护 |
| identical | `framework/capability_schema.py` | `framework/capability_schema.py` | 可直接同步 |
| identical | `framework/__init__.py` | `framework/__init__.py` | 可直接同步 |
| different | `framework/stage1/adapters.py` | `framework/stage1/adapters.py` | V2X 使用本机绝对路径；GitHub 版本使用环境变量，合并时保留本地路径策略或迁移到环境变量 |
| different | `framework/stage1/auto_trace.py` | `framework/stage1/auto_trace.py` | V2X 含本机绝对路径和动态 BEV shape 推断；GitHub 版本更可迁移 |
| identical | `framework/stage1_bridge.py` | `framework/stage1_bridge.py` | 可直接同步 |
| identical | `framework/stage1/calibrated_predictor.py` | `framework/stage1/calibrated_predictor.py` | 可直接同步 |
| identical | `framework/stage1/coupling_predictor.py` | `framework/stage1/coupling_predictor.py` | 可直接同步 |
| identical | `framework/stage1/graph_scan.py` | `framework/stage1/graph_scan.py` | 可直接同步 |
| identical | `framework/stage1/hardware_scan.py` | `framework/stage1/hardware_scan.py` | 可直接同步 |
| identical | `framework/stage1/__init__.py` | `framework/stage1/__init__.py` | 可直接同步 |
| different | `framework/stage1/latency_profile.py` | `framework/stage1/latency_profile.py` | 仅 `_RESULTS` 根路径差异；优先改为相对/环境变量形式 |
| identical | `framework/stage1/model_classifier.py` | `framework/stage1/model_classifier.py` | 可直接同步 |
| different | `framework/stage1/run_scan.py` | `framework/stage1/run_scan.py` | 重要差异: GitHub 版本支持 `REGISTRY + AUTO_REGISTRY`，V2X 当前只走 `REGISTRY` |
| identical | `framework/stage1/standard_conv_census.py` | `framework/stage1/standard_conv_census.py` | 可直接同步 |
| different | `framework/stage1/standard_conv_probe_plan.py` | `framework/stage1/standard_conv_probe_plan.py` | 仅命令示例中的 `PYTHONPATH` 差异 |
| different-semantic | `README.md` | `README.md` | GitHub README 是 Stage1 Scanner；V2X README 是 UniV2X，总项目 README 不应覆盖 |
| missing | `README.zh-CN.md` | `README.zh-CN.md` | 若需要中文入口，可复制为 Stage1 专用文档，不建议作为 V2X 根 README |
| different | `scripts/autoscan_reproduce_check.py` | `scripts/autoscan_reproduce_check.py` | 主要是本机 `_REPO` 和命令示例差异 |
| different | `scripts/phase2/stage1_s2_5_s3_evidence_report.py` | `scripts/phase2/stage1_s2_5_s3_evidence_report.py` | GitHub 用 `HEAL_ROOT` 环境变量；V2X 用本机绝对路径 |
| identical | `scripts/phase2/stage1_s2_anchor_runner.py` | `scripts/phase2/stage1_s2_anchor_runner.py` | 可直接同步 |
| identical | `scripts/phase2/stage1_s2_probe_completion_report.py` | `scripts/phase2/stage1_s2_probe_completion_report.py` | 可直接同步 |
| identical | `scripts/phase2/stage1_s4_three_arm_validation.py` | `scripts/phase2/stage1_s4_three_arm_validation.py` | 可直接同步 |
| identical | `scripts/stage1_classify_models.py` | `scripts/stage1_classify_models.py` | 可直接同步 |
| identical | `scripts/stage1_predict_coupling.py` | `scripts/stage1_predict_coupling.py` | 可直接同步 |
| identical | `scripts/stage1_predict_coupling_v1.py` | `scripts/stage1_predict_coupling_v1.py` | 可直接同步 |
| different | `tools/configurable/depgraph_pyramid.py` | `tools/configurable/depgraph_pyramid.py` | V2X 用本机 HEAL/checkpoint 绝对路径；GitHub 用环境变量 |
| different | `tools/configurable/depgraph_v2xvit.py` | `tools/configurable/depgraph_v2xvit.py` | V2X 用本机 HEAL/checkpoint 绝对路径；GitHub 用环境变量 |

## 4. 当前差异摘要

本次扫描共检查 GitHub 项目 28 个文件:

| 类别 | 数量 | 说明 |
|---|---:|---|
| 完全一致 | 16 | 可作为同步基线 |
| 内容不同 | 9 | 大多是本机绝对路径、`PYTHONPATH` 示例、环境变量策略差异 |
| 语义不同 | 1 | `README.md` 同名但不是同一文档 |
| V2X 缺失 | 2 | `docs/add-new-model-adapter.zh-CN.md`, `README.zh-CN.md` |

需要重点人工 review 的差异:

1. `framework/stage1/run_scan.py`: GitHub 版本把 `REGISTRY` 和 `AUTO_REGISTRY` 合并到 `ALL_MODELS`，V2X 当前版本只通过 `get_adapter()` 使用手写 `REGISTRY`。如果要让 V2X 的 CLI 支持 auto-trace 新模型，这里需要合并 GitHub 逻辑。
2. `framework/stage1/auto_trace.py`: GitHub 版本偏可迁移，路径由 `HEAL_ROOT`、`HEAL_CKPT_ROOT`、`V2XVERSE_ROOT`、`V2XVERSE_CKPT_ROOT` 驱动；V2X 版本写死本机路径，同时有动态 `_infer_heter_bev_shape(...)`。合并时应保留动态 shape 推断，同时把路径改成环境变量优先、本机默认兜底。
3. `framework/stage1/adapters.py`, `tools/configurable/depgraph_*.py`, `scripts/phase2/stage1_s2_5_s3_evidence_report.py`: 差异集中在 HEAL/V2Xverse/checkpoint 路径。不要盲目覆盖，否则会破坏本机复现实验。
4. `README.md`: V2X 根 README 是 UniV2X 项目说明，GitHub README 是 Stage1 Scanner 说明。若要纳入 V2X，应另存为 `docs/stage1_model_scanner_readme.md` 或保留在 `github/stage1-model-scanner-aaai/README.md`。

## 5. 关联产物和证据位置

| 产物 | 位置 | 生成/消费方 |
|---|---|---|
| 硬件 capability | `configs/hardware/rtx4090.yaml`, `configs/hardware/orin_agx.yaml`, `configs/hardware/schema.yaml` | `framework/stage1/hardware_scan.py`, `framework/capability_schema.py` |
| Stage1 partition manifest | `framework/partitions/*_partition.yaml` | `framework/stage1/run_scan.py` 生成；predictor/classifier/bridge 消费 |
| scan 结果说明 | `framework/partitions/README_scan_results_v1.md` | Stage1 graph scan 结果索引 |
| coupling predictor v0/v1 | `results/stage1_model_predict/stage1_coupling_predictions_v*.json`, `.md` | `scripts/stage1_predict_coupling*.py` |
| calibrated predictor | `results/stage1_model_predict/calibrated_predictor_report_v1.*` | `framework/stage1/calibrated_predictor.py` |
| model classifier | `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.*` | `scripts/stage1_classify_models.py` |
| S2 anchor scan | `results/stage1_model_predict/s2_anchor_scan/` | `scripts/phase2/stage1_s2_anchor_runner.py` |
| S2 probe completion | `results/stage1_model_predict/s2_probe_results/` | `scripts/phase2/stage1_s2_probe_completion_report.py` |
| S2.5/S3 evidence | `results/stage1_model_predict/s2_5_coverage_gates/`, `results/stage1_model_predict/s3_quant_sensitivity/` | `scripts/phase2/stage1_s2_5_s3_evidence_report.py` |
| S4 validation | `results/stage1_model_predict/s4_three_arm_validation/` | `scripts/phase2/stage1_s4_three_arm_validation.py` |

## 6. 关联测试

更新 Stage1 Scanner 相关文件后，优先跑这些测试:

```bash
cd /home/jichengzhi/V2X
PYTHONPATH=/home/jichengzhi/V2X pytest -q \
  framework/tests/test_stage1_autoscan_extensions.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

如果改动 `tools/configurable/depgraph_*.py` 或真实 scan 流程，再补充运行:

```bash
cd /home/jichengzhi/V2X
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1.run_scan \
  --model all \
  --hw configs/hardware/rtx4090.yaml \
  --device cpu \
  --profile-latency off
```

注意: 上面的 `run_scan --model all` 在 V2X 当前实现中只覆盖 `REGISTRY`。如果合并了 GitHub 版 `AUTO_REGISTRY` 支持，测试范围会扩大。

## 7. 下次更新/索引流程

从 V2X 根目录执行，先只比对，不覆盖:

```bash
cd /home/jichengzhi/V2X
while IFS= read -r f; do
  rel=${f#github/stage1-model-scanner-aaai/}
  target="/home/jichengzhi/V2X/$rel"
  if [ -f "$target" ]; then
    if cmp -s "$f" "$target"; then
      status=identical
    else
      status=different
    fi
  else
    status=missing
  fi
  printf '%s\t%s\t%s\n' "$status" "$rel" "$target"
done < <(find github/stage1-model-scanner-aaai -type f | sort)
```

查看所有差异行数:

```bash
cd /home/jichengzhi/V2X
while IFS= read -r f; do
  rel=${f#github/stage1-model-scanner-aaai/}
  target="/home/jichengzhi/V2X/$rel"
  if [ -f "$target" ] && ! cmp -s "$f" "$target"; then
    git diff --no-index --numstat -- "$f" "$target" || true
  fi
done < <(find github/stage1-model-scanner-aaai -type f | sort)
```

查看单文件差异:

```bash
cd /home/jichengzhi/V2X
git diff --no-index -- \
  github/stage1-model-scanner-aaai/framework/stage1/run_scan.py \
  framework/stage1/run_scan.py
```

建议更新顺序:

1. 先同步 `identical` 或低风险文档。
2. 对 `different` 文件逐个 `git diff --no-index`，确认差异是否属于本机路径、实验环境、或真实功能变更。
3. 路径差异优先收敛为“环境变量优先 + 本机默认兜底”，避免同一逻辑在 GitHub/V2X 间长期漂移。
4. 修改后运行第 6 节测试。
5. 若更新了 manifest 或 evidence，重新生成 `results/stage1_model_predict/*` 并在本文件第 5 节补充新产物位置。

## 8. 不建议索引/同步的目录

这些目录体积大或是实验产物，索引时只记录目录级说明，不应逐文件跟踪:

| 目录 | 原因 |
|---|---|
| `models/`, `ckpts/`, `checkpoints/`, `calibration/` | 模型、engine、校准缓存，体积大且频繁变化 |
| `results/m4_*`, `output/`, `logs/`, `work_dirs/`, `test/` | 实验运行产物，适合按实验主题建摘要，不适合作为 Stage1 源码索引 |
| `.claude/`, `.serena/`, `.codegraph/`, `.pytest_cache/`, `__pycache__/` | 工具缓存或本地状态 |
| `tvm_venv310/`, `tvm_offline_cp310/` | 本地环境，不属于项目源码映射 |

## 9. 快速定位命令

```bash
cd /home/jichengzhi/V2X
rg -n "stage1_model_predict|AUTO_REGISTRY|REGISTRY|run_scan|stage1_partition" \
  framework scripts tools docs multi_agent
```

```bash
cd /home/jichengzhi/V2X
find framework/stage1 scripts/phase2 tools/configurable framework/tests \
  -type f \
  \( -name '*stage1*' -o -name 'depgraph_*' -o -name '*predictor*' \) \
  | sort
```
