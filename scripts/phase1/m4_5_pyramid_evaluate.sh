#!/bin/bash
# M4.5: Pyramid Fusion 评估脚本骨架
# 流程: 等下载完成 → 合并分片 → 解压 → 修 config → 跑 inference → 收 AP
#
# 前置条件:
#   - /home/jichengzhi/heal_research/dataset/OPV2V_Hetero/ 含 OPV2V-H-LiDAR-part?? 41 分片 (~39 GB)
#   - /home/jichengzhi/heal_research/checkpoints/ 含 HEAL_OPV2V.zip
#   - UniV2X_2.0 conda env 装好 spconv-cu118 + open3d + cython + tensorboardX + h5py
#
# 输出:
#   - /home/jichengzhi/heal_research/dataset/OPV2V_Hetero/test/ (解压后的 test split)
#   - /home/jichengzhi/heal_research/checkpoints/HEAL_OPV2V/ (解压后的 model_dir)
#   - /home/jichengzhi/UniV2X/results/m4_5_pyramid_baseline_eval.txt (AP30/AP50/AP70 报告)

set -e
HEAL_RESEARCH=/home/jichengzhi/heal_research
HEAL_REPO=$HEAL_RESEARCH/HEAL
DATASET_DIR=$HEAL_RESEARCH/dataset/OPV2V_Hetero
CKPT_DIR=$HEAL_RESEARCH/checkpoints
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
RESULT_DIR=/home/jichengzhi/UniV2X/results
LOG=$RESULT_DIR/m4_5_pyramid_eval.log
mkdir -p "$RESULT_DIR"

echo "=== M4.5 Pyramid Fusion evaluation ===" | tee "$LOG"
date | tee -a "$LOG"

# ----- Step 1: verify downloads -----
echo -e "\n[Step 1] 下载完成检查" | tee -a "$LOG"
if ! ls "$DATASET_DIR"/OPV2V-H-LiDAR-partbn 2>/dev/null; then
  echo "  ❌ OPV2V-H 未下完 (last part missing)" | tee -a "$LOG"
  exit 1
fi
echo "  ✅ 41 个 LiDAR 分片齐" | tee -a "$LOG"

if ! ls "$CKPT_DIR"/HEAL_OPV2V.zip 2>/dev/null; then
  echo "  ❌ HEAL_OPV2V.zip 未下完" | tee -a "$LOG"
  exit 1
fi
echo "  ✅ HEAL_OPV2V.zip 在位" | tee -a "$LOG"

# ----- Step 2: merge OPV2V-H LiDAR parts + extract -----
echo -e "\n[Step 2] 合并 + 解压 OPV2V-H 数据 (39 GB → ~50 GB 解压后)" | tee -a "$LOG"
cd "$DATASET_DIR"
if [ ! -f "OPV2V_Hetero.tar.gz" ]; then
  echo "  cat OPV2V-H-LiDAR-part?? > OPV2V_Hetero.tar.gz ..." | tee -a "$LOG"
  cat OPV2V-H-LiDAR-part?? > OPV2V_Hetero.tar.gz
  echo "  ✅ merged: $(du -sh OPV2V_Hetero.tar.gz)" | tee -a "$LOG"
fi
if [ ! -d "OPV2V_Hetero/test" ] && [ ! -d "test" ]; then
  echo "  tar xzf OPV2V_Hetero.tar.gz ..." | tee -a "$LOG"
  tar xzf OPV2V_Hetero.tar.gz
fi
echo "  ✅ 解压后结构:" | tee -a "$LOG"
ls "$DATASET_DIR" | tee -a "$LOG"

# ----- Step 3: HEAL ckpt 已解压 (stage1/Pyramid_m1_base 是 LiDAR-only baseline) -----
echo -e "\n[Step 3] 验证 HEAL ckpt 解压结构" | tee -a "$LOG"
PYRAMID_CKPT_DIR=$(find "$CKPT_DIR/stage1" -maxdepth 2 -type d -name "Pyramid_m1_base*" 2>/dev/null | head -1)
if [ -z "$PYRAMID_CKPT_DIR" ]; then
  echo "  ❌ stage1/Pyramid_m1_base* 未找到, 请先 unzip HEAL_OPV2V.zip" | tee -a "$LOG"
  exit 1
fi
echo "  ✅ Pyramid baseline ckpt: $PYRAMID_CKPT_DIR" | tee -a "$LOG"
ls "$PYRAMID_CKPT_DIR" | head -5 | tee -a "$LOG"

# ----- Step 4: 修 config.yaml 路径指向本地 OPV2V-H -----
echo -e "\n[Step 4] 修 config.yaml test_dir/validate_dir" | tee -a "$LOG"
CONFIG="$PYRAMID_CKPT_DIR/config.yaml"
if [ -f "$CONFIG" ]; then
  cp "$CONFIG" "$CONFIG.bak"
  # 找出 test/validate split 真实路径
  TEST_DIR=$(find "$DATASET_DIR" -maxdepth 4 -type d -iname "test" 2>/dev/null | head -1)
  VALIDATE_DIR=$(find "$DATASET_DIR" -maxdepth 4 -type d -iname "validate" 2>/dev/null | head -1)
  ROOT_DIR=$(find "$DATASET_DIR" -maxdepth 4 -type d -iname "train" 2>/dev/null | head -1)
  echo "  test_dir: $TEST_DIR" | tee -a "$LOG"
  echo "  validate_dir: $VALIDATE_DIR" | tee -a "$LOG"
  if [ -n "$TEST_DIR" ]; then
    $PY -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
c['root_dir'] = '$ROOT_DIR'
c['validate_dir'] = '$VALIDATE_DIR'
c['test_dir'] = '$TEST_DIR'
with open('$CONFIG', 'w') as f:
    yaml.safe_dump(c, f, default_flow_style=False)
print('  config.yaml 已更新')
"
  fi
fi

# ----- Step 5: 跑 HEAL inference (AP30/50/70) -----
echo -e "\n[Step 5] 跑 HEAL inference (Pyramid LiDAR-only baseline)" | tee -a "$LOG"
cd "$HEAL_REPO"
$PY opencood/tools/inference.py \
  --model_dir "$PYRAMID_CKPT_DIR" \
  --fusion_method intermediate 2>&1 | tee -a "$LOG"

echo -e "\n=== M4.5 baseline evaluation 完成 ===" | tee -a "$LOG"
date | tee -a "$LOG"
echo "  AP results: 见 $LOG 末尾的 'The Average Precision' 行"
