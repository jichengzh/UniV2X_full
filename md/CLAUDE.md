# UniV2X — Migration to UniAD 2.0 Environment (UniV2X_2.0)

This document describes the full process of adapting UniV2X to run inside the UniAD 2.0 (`UniV2X_2.0`) conda environment. It covers environment differences, required code modifications, data preparation, model weight downloads, and inference validation.

---

## 1. Environment Differences: UniV2X Original vs UniAD 2.0

| Component | UniV2X (original) | UniAD 2.0 (target) |
|---|---|---|
| Python | 3.8 | 3.9 |
| PyTorch | 1.9.1+cu111 | 2.0.1+cu118 |
| CUDA | 11.1 | 11.8 |
| mmcv-full | 1.4.0 | 1.6.1 |
| mmdet | 2.14.0 | 2.26.0 |
| mmsegmentation | 0.14.1 | 0.29.1 |
| mmdet3d | v0.17.1 (source install) | 1.0.0rc6 (pip) |
| argoverse-api | required | not required |
| einops | 0.4.1 | 0.8.1 |
| numpy | 1.21.5 | 1.22.4 |
| casadi | 3.5.5 | 3.6.7 |
| opencv-python | not pinned | 4.8.0.76 |
| Distributed launch | `python -m torch.distributed.launch` | `torchrun` |

### Key architectural differences between UniV2X and UniAD codebases

- **UniV2X** wraps all agents into a `MultiAgent` container: `cfg.model_ego_agent` (vehicle) + `cfg.model_other_agent_inf` (infrastructure). The outer wrapper is `MultiAgent(MVXTwoStageDetector)`.
- **UniAD** uses a single `cfg.model` entry with no multi-agent wrapper.
- UniV2X `tools/test.py` and `tools/train.py` manually construct `MultiAgent`; UniAD tools use a single `build_model(cfg.model)` call.
- UniV2X `apis/test.py` accesses attributes via `model.module.model_ego_agent.*`; UniAD accesses via `model.module.*`.
- UniV2X `apis/train.py` checks `cfg.model_ego_agent.type`; UniAD checks `cfg.model.type`.
- UniAD 2.0 `apis/train.py` contains a `train_model` function in addition to `custom_train_model`; UniV2X is missing this.
- UniV2X `apis/mmdet_train.py` extends `custom_train_detector` with an `eval_model` parameter (for saving infrastructure agent queries during training).
- Launch scripts: UniV2X uses `python -m torch.distributed.launch`; UniAD 2.0 uses `torchrun`.
- UniAD 2.0 dense_heads includes `bevformer_head.py` and `bevformer.py` detector; UniV2X does not.
- UniAD 2.0 datasets includes `nuscenes_bev_dataset.py` and `nuscenes_eval.py`; UniV2X does not.

---

## 2. Environment Setup

```bash
conda create -n UniV2X_2.0 python=3.9 -y
conda activate UniV2X_2.0

# PyTorch 2.0.1 + CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# mmcv-full 1.6.1 (must match torch 2.0 / cu118)
pip install -v mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# mmdet / mmseg / mmdet3d
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6

# Project requirements
cd /home/jichengzhi/UniV2X
pip install -r requirements.txt

# Additional UniAD 2.0 requirements not in UniV2X requirements.txt
pip install opencv-python==4.8.0.76 einops==0.8.1 numpy==1.22.4 \
    casadi==3.6.7 ipython==8.12.3 pandas==1.2.2 \
    torchmetrics==0.6.2 networkx==2.5 "motmetrics<=1.1.3"
```

> Note: argoverse-api is NOT needed in the UniAD 2.0 environment.

---

## 3. Code Modifications Required

All modifications adapt UniV2X to the UniAD 2.0 environment while preserving the multi-agent architecture. Changes are grouped by file.

### 3.1 Launch scripts — replace deprecated distributed launcher

**Files:** `tools/univ2x_dist_train.sh`, `tools/univ2x_dist_eval.sh`

Replace:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
```
With:
```bash
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
```

### 3.2 `tools/test.py` — default launcher

Change the `--launcher` default from `'none'` to `'pytorch'`:
```python
# line ~92
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                    default='pytorch', ...)
```

### 3.3 `projects/mmdet3d_plugin/univ2x/apis/test.py` — model attribute access

The `MultiAgent` wrapper exposes the ego-vehicle model as `model.module.model_ego_agent`. All attribute checks must go through this path (already correct in UniV2X). Verify that no line accidentally uses the flat `model.module.*` pattern from UniAD:

```python
# Correct for UniV2X (lines 59-60, 74-75):
eval_occ = hasattr(model.module.model_ego_agent, 'with_occ_head') \
            and model.module.model_ego_agent.with_occ_head
eval_planning = hasattr(model.module.model_ego_agent, 'with_planning_head') \
                and model.module.model_ego_agent.with_planning_head
```

No change needed if this is already in place.

### 3.4 `projects/mmdet3d_plugin/univ2x/apis/train.py` — add `train_model` wrapper

UniAD 2.0 exposes both `custom_train_model` and `train_model`. To avoid import errors from any downstream code that imports `train_model`, add an alias at the bottom of `apis/train.py`:

```python
# Add after the existing custom_train_model definition:
train_model = custom_train_model
```

### 3.5 `projects/mmdet3d_plugin/univ2x/apis/train.py` — `cfg.model_ego_agent` check

The type-check currently reads `cfg.model_ego_agent.type`. With mmdet 2.26+ this still works; no change needed. However if `EncoderDecoder3D` support is ever added, the assertion block must be changed to call `train_segmentor`.

### 3.6 `projects/mmdet3d_plugin/univ2x/detectors/multi_agent.py` — `MVXTwoStageDetector` import

In mmdet3d 1.0.0rc6 the path is the same:
```python
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
```
No change needed. Verify after install.

### 3.7 `projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py` and `univ2x_e2e.py`

In mmdet3d 1.0.0rc6, `mmdet3d.core` is retained as a compatibility shim. The following imports should still work:
```python
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
```
Verify after install. If `ImportError` is raised, replace with:
```python
from mmdet3d.structures import bbox3d2result           # mmdet3d >= 1.0
from mmdet3d.models.task_modules.coders import build_bbox_coder
```

### 3.8 `projects/mmdet3d_plugin/univ2x/dense_heads/track_head_plugin/tracker.py`

Same mmdet3d.core compatibility concern:
```python
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (...)
```
If this fails:
```python
from mmdet3d.models.task_modules.assigners.iou_calculators import (...)
```

### 3.9 `requirements.txt` — pin versions to UniAD 2.0

Update `requirements.txt` to align with UniAD 2.0:
```
google-cloud-bigquery
opencv-python==4.8.0.76
einops==0.8.1
numpy==1.22.4
casadi==3.6.7
pytorch-lightning==1.2.5
ipython==8.12.3
yapf==0.40.1
motmetrics<=1.1.3
pandas==1.2.2
torchmetrics==0.6.2
networkx==2.5
rich==13.9.4
```

---

## 4. Full API Adaptation Checklist

Run through this checklist after environment setup and before evaluation:

- [ ] `torch.distributed.launch` replaced with `torchrun` in all shell scripts
- [ ] `apis/train.py`: `train_model` alias added
- [ ] `tools/test.py`: default launcher set to `'pytorch'`
- [ ] Verify `from mmdet3d.core import bbox3d2result` works in UniV2X_2.0 env
- [ ] Verify `from mmdet3d.core.bbox.coders import build_bbox_coder` works
- [ ] Verify `from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import ...` works
- [ ] Verify `from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector` works
- [ ] Verify `from mmcv.runner import auto_fp16` works (mmcv 1.6.1)
- [ ] Verify `from mmcv.parallel import DataContainer as DC` works (mmcv 1.6.1)
- [ ] Run a quick import test:
  ```bash
  cd /home/jichengzhi/UniV2X
  PYTHONPATH=. python -c "
  from projects.mmdet3d_plugin.univ2x.detectors.univ2x_e2e import UniV2X
  from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
  from projects.mmdet3d_plugin.univ2x.apis.test import custom_multi_gpu_test
  from projects.mmdet3d_plugin.univ2x.apis.train import custom_train_model
  print('All imports OK')
  "
  ```

---

## 5. Data Directory Setup

### 5.1 Create directory structure

```bash
cd /home/jichengzhi/UniV2X
mkdir -p datasets ckpts data/infos data/split_datas
```

### 5.2 Link V2X-Seq-SPD raw dataset

```bash
# Source is at /data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD/
ln -s /data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD datasets/V2X-Seq-SPD
```

Verify the link:
```bash
ls datasets/V2X-Seq-SPD/
# Expected: cooperative/  infrastructure-side/  maps/  vehicle-side/
```

### 5.3 Generate V2X-Seq-SPD-New (processed dataset)

```bash
# Full dataset (slow, ~all sequences)
python tools/spd_data_converter/gen_example_data.py \
    --input datasets/V2X-Seq-SPD \
    --output datasets/V2X-Seq-SPD-New \
    --sequences all \
    --update-label \
    --freq 2

# For fast validation use a subset:
# --sequences '0010 0016 0018 0022 0023 0025 0029 0030 0032 0033 0034 0035 0014 0015 0017 0020 0021'
```

### 5.4 Convert to UniV2X format

```bash
bash tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New vehicle-side
bash tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New infrastructure-side
bash tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New cooperative
```

This produces:
```
data/infos/V2X-Seq-SPD-New/
├── vehicle-side/
│   ├── spd_infos_temporal_train.pkl
│   └── spd_infos_temporal_val.pkl
├── infrastructure-side/
│   ├── spd_infos_temporal_train.pkl
│   └── spd_infos_temporal_val.pkl
└── cooperative/
    ├── spd_infos_temporal_train.pkl
    └── spd_infos_temporal_val.pkl
data/split_datas/cooperative-split-data-spd.json
```

### 5.5 Final directory structure

```
UniV2X/
├── projects/
├── tools/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth        # BEVFormer pretrained
│   ├── univ2x_sub_inf_stg1.pth
│   ├── univ2x_sub_inf_stg2.pth
│   ├── univ2x_sub_veh_stg1.pth
│   ├── univ2x_sub_veh_stg2.pth
│   ├── univ2x_coop_e2e_stg1.pth
│   └── univ2x_coop_e2e_stg2.pth
├── datasets/
│   ├── V2X-Seq-SPD -> /data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD/  (symlink)
│   └── V2X-Seq-SPD-New/
│       ├── vehicle-side/
│       ├── infrastructure-side/
│       └── cooperative/
└── data/
    ├── infos/
    │   └── V2X-Seq-SPD-New/
    │       ├── vehicle-side/
    │       ├── infrastructure-side/
    │       └── cooperative/
    └── split_datas/
        └── cooperative-split-data-spd.json
```

---

## 6. Model Weight Downloads

All weights are hosted on Google Drive. Use `gdown` or download manually.

```bash
pip install gdown
cd /home/jichengzhi/UniV2X/ckpts
```

### BEVFormer pretrained backbone

```bash
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
```

### Infrastructure sub-system

```bash
# Stage 1 (MD5: f14ef0d540156cc9318399661fc08d5e)
gdown "https://drive.google.com/uc?id=1XJvMDmdasO-eHnLLQQgctU1x7FmPGOJR" \
    -O univ2x_sub_inf_stg1.pth

# Stage 2 (MD5: 7337567c6012f8b9fc326b66235d7c9b)
gdown "https://drive.google.com/uc?id=1ubZySia8smrlPbgTxVhAe3PyFIpoliYK" \
    -O univ2x_sub_inf_stg2.pth
```

### Vehicle sub-system

```bash
# Stage 1 (MD5: 7ee07fc34dfac28070e640b16aebf26c)
gdown "https://drive.google.com/uc?id=1tEpnqKwTFgnz40oAr4lvPQvfSdU3b2s2" \
    -O univ2x_sub_veh_stg1.pth

# Stage 2 (MD5: 2843db6bfabf4572ef621a486f5097e1)
gdown "https://drive.google.com/uc?id=1kaU0_Vf_DpiLNh0r4h2ciKmQaAkkytWe" \
    -O univ2x_sub_veh_stg2.pth
```

### Cooperative models

```bash
# Coop Perception stage 1 (MD5: 66a8e1eace582bdaadf1fd0293dd9a5c)
gdown "https://drive.google.com/uc?id=1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a" \
    -O univ2x_coop_e2e_stg1.pth

# Coop Planning stage 2 (MD5: 8a08c5826059af32264025054b38f16e)
gdown "https://drive.google.com/uc?id=1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN" \
    -O univ2x_coop_e2e_stg2.pth

# Coop Planning old-mode (MD5: for univ2x_coop_e2e_stg2_old_mode_inference_only.pth)
gdown "https://drive.google.com/uc?id=1Zu5pYkEms9q9n2ucMU6CYTWx3FfMVCpr" \
    -O univ2x_coop_e2e_stg2_old_mode_inference_only.pth
```

### Verify checksums

```bash
cd /home/jichengzhi/UniV2X/ckpts
md5sum -c <<'EOF'
f14ef0d540156cc9318399661fc08d5e  univ2x_sub_inf_stg1.pth
7337567c6012f8b9fc326b66235d7c9b  univ2x_sub_inf_stg2.pth
7ee07fc34dfac28070e640b16aebf26c  univ2x_sub_veh_stg1.pth
2843db6bfabf4572ef621a486f5097e1  univ2x_sub_veh_stg2.pth
66a8e1eace582bdaadf1fd0293dd9a5c  univ2x_coop_e2e_stg1.pth
8a08c5826059af32264025054b38f16e  univ2x_coop_e2e_stg2.pth
EOF
```

---

## 7. Inference & Evaluation

All evaluation commands use the `UniV2X_2.0` conda environment.

```bash
conda activate UniV2X_2.0
cd /home/jichengzhi/UniV2X
```

### 7.1 Cooperative Perception (Stage 1)

```bash
GPU_IDs=0,1,2,3
GPU_NUM=4
CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh \
    ./projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    ./ckpts/univ2x_coop_e2e_stg1.pth \
    ${GPU_NUM}
```

**Reference performance (must match within ±0.005 / ±0.5% absolute):**

| Metric | Reference |
|---|---|
| Tracking AMOTA | 0.300 |
| Mapping IoU-lane | 16.2% |

### 7.2 Cooperative Planning (Stage 2) — Current mode

```bash
CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh \
    ./projects/configs_e2e_univ2x/univ2x_coop_e2e.py \
    ./ckpts/univ2x_coop_e2e_stg2.pth \
    ${GPU_NUM}
```

**Reference performance (must match within ±0.005 / ±0.5% absolute):**

| Metric | Reference |
|---|---|
| Tracking AMOTA | 0.239 |
| Mapping IoU-lane | 17.8% |
| Occupancy IoU-n | 22.6% |
| Planning Col. | 0.54% |

### 7.3 Cooperative Planning (Stage 2) — Old-mode inference (higher occ/planning)

```bash
CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh \
    ./projects/configs_e2e_univ2x/univ2x_coop_e2e_old_mode_inference_only.py \
    ./ckpts/univ2x_coop_e2e_stg2_old_mode_inference_only.pth \
    ${GPU_NUM}
```

**Reference performance (must match within ±0.005 / ±0.5% absolute):**

| Metric | Reference |
|---|---|
| Tracking AMOTA | 0.239 |
| Mapping IoU-lane | 17.8% |
| Occupancy IoU-n | 25.2% |
| Planning Col. | 0.34% |

### 7.4 Single-agent evaluation (optional validation)

Infrastructure:
```bash
CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh \
    ./projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py \
    ./ckpts/univ2x_sub_inf_stg1.pth \
    ${GPU_NUM}
```

Vehicle:
```bash
CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh \
    ./projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e_track.py \
    ./ckpts/univ2x_sub_veh_stg1.pth \
    ${GPU_NUM}
```

### 7.5 Visualization

After evaluation, results are saved to `./output/results.pkl`. Run:
```bash
./tools/univ2x_vis_results.sh
```

---

## 8. Troubleshooting

### `ImportError: cannot import name 'bbox3d2result' from 'mmdet3d.core'`

mmdet3d 1.0.0rc6 moved some APIs. Fix all occurrences:
```python
# Old
from mmdet3d.core import bbox3d2result
# New (if needed)
try:
    from mmdet3d.core import bbox3d2result
except ImportError:
    from mmdet3d.structures import bbox3d2result
```

### `RuntimeError: Expected all tensors to be on the same device`

Likely a torch 2.x AMP / fp16 issue. Ensure `auto_fp16` from `mmcv.runner` is imported correctly and that `torch.cuda.amp` is not double-wrapping operations.

### `AssertionError` in `torchrun` / `NCCL`

Make sure `MASTER_PORT` is not occupied:
```bash
MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/univ2x_dist_eval.sh ...
```

### `FileNotFoundError: datasets/V2X-Seq-SPD-New`

Run the data preparation steps in Section 5 before evaluation.

### Metric deviation > tolerance

1. Confirm checkpoint MD5 checksums match (Section 6).
2. Confirm the correct config is paired with the correct checkpoint (old-mode vs current-mode).
3. Confirm all sequences were included in `gen_example_data.py` (`--sequences all`).
4. Check that the symlink `datasets/V2X-Seq-SPD` correctly resolves to the raw dataset root containing `cooperative/`, `vehicle-side/`, `infrastructure-side/`, `maps/`.
