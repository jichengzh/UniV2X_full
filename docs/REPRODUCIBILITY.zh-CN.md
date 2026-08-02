# 软硬件协同实验复现索引

本仓库保留完整的实验代码、配置、测试和设计决策；模型 checkpoint、ONNX/TensorRT engine、校准缓存、运行日志和结果目录均不提交。它们由下列脚本在具备数据集、预训练权重和目标硬件的环境中重新生成。

## 1. 复现边界

| 内容 | 位置 | 是否随 Git 提供 |
| --- | --- | --- |
| UniV2X 基线安装、数据准备、训练和评测 | `docs/INSTALL.md`、`docs/DATA_PREP.md`、`docs/TRAIN_EVAL.md` | 是 |
| 模型扫描与硬件能力过滤 | `framework/stage1/`、`configs/hardware/` | 是 |
| LUT、测量合约和联合 P×Q 搜索 | `framework/stage2/`、`scripts/stage2_*.py` | 是 |
| 搜索策略、代价模型选择、闭环验证 | `framework/stage3/` 至 `framework/stage7/`、`scripts/stage[3-7]_*.py` | 是 |
| 单元/回归测试 | `framework/tests/` | 是 |
| 权重、模型导出、engine、校准缓存、结果、日志 | `models/`、`checkpoints/`、`calibration/`、`results/`、`logs/` | 否，按脚本生成 |

## 2. 设计理由与入口

- Stage 1 先分析模型的稠密路径和硬件能力，避免将不适合联合优化的模型直接送入搜索；完整映射见 [`stage1_model_scanner_v2x_index.md`](stage1_model_scanner_v2x_index.md)。
- Stage 2 用统一 `config_id` 对齐 latency、AP 与 energy 证据，并区分真实测量与估计值；数据合约和产物规则见 [`stage2-evidence-registry.zh-CN.md`](stage2-evidence-registry.zh-CN.md)。
- Stage 3--7 将候选生成、成本模型、真实硬件回流、独立验证和消融拆开，保证最终结论能回溯至配置、脚本与测试，而不是依赖本地缓存。
- 具体设计记录保存在 `docs/superpowers/plans/` 和 `docs/superpowers/specs/`；它们说明每个阶段的前提、决策和验证标准。

## 3. 环境约定

克隆后在仓库根目录执行，并按本机环境设置路径；不要把密码、token 或内部服务器地址写入脚本或配置。

```bash
export V2X_ROOT="$PWD"
export V2X_PYTHON="$(command -v python)"
export HEAL_ROOT="/path/to/HEAL"
export V2X_RESULTS_DIR="$V2X_ROOT/results"
export V2X_FORMAL_H800_HOSTNAME="your-h800-hostname"  # 仅正式 H800 测量需要
export PYTHONPATH="$V2X_ROOT:${PYTHONPATH:-}"
```

先按 [`INSTALL.md`](INSTALL.md) 建立 Python、PyTorch、CUDA/TVM（以及需要 TensorRT 时的 TensorRT）环境，再按 [`DATA_PREP.md`](DATA_PREP.md) 准备数据和合法获取的预训练权重。权重路径、GPU 编号和远端执行位置均应通过命令参数或环境变量传入。H800 的 TVM 路径通过 `V2X_TVM_PYTHON`、`V2X_TVM_SITE`、`V2X_TVM_RUN_ROOT` 配置；原生 INT8 路线通过 `V2X_NATIVE_INT8_ROUTE` 和 `V2X_NATIVE_INT8_MODEL_ROOT` 配置。

## 4. 最小本地验证

以下测试不要求运行大规模训练或生成 engine，可用于确认代码、配置和核心合约能够协同工作：

```bash
cd "$V2X_ROOT"
"$V2X_PYTHON" -m pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage2_lut_productization.py \
  framework/tests/test_stage2_integration_contract.py
```

若新增或修改了模型适配器，先使用对应硬件 YAML 执行 `framework.stage1.run_scan`；再由 Stage 2 的 LUT job-plan/worker 生成真实测量行；最后再启动联合搜索或 Stage 3--7 的闭环验证。不要把历史 `results/` 直接作为新的真实测量结论。

## 5. 可复现的产物策略

所有生成产物写入 `V2X_RESULTS_DIR`（默认 `results/`）或命令指定的输出目录。提交代码时只提交：

- 产生结果的 Python/Shell 脚本；
- 固定搜索空间、硬件能力和实验协议的 YAML/JSON；
- 设计原因、输入前提和运行顺序的 Markdown；
- 能验证核心行为的测试。

不要提交 checkpoint、权重副本、ONNX/TensorRT engine、timing cache、校准缓存、原始日志、PID 文件或可由上述步骤重新生成的结果。`.gitignore` 已为这些内容提供默认保护。
