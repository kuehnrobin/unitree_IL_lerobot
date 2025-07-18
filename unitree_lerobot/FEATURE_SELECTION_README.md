# Feature Selection and Ablation Studies

This guide explains how to use the feature selection system to rapidly test different combinations of sensors and state features without recreating datasets.

## Overview

The feature selection system allows you to:
- **Select specific cameras** (e.g., only use high-level camera)
- **Choose state features** (positions, velocities, torques, pressures)
- **Filter joint groups** (arms, hands, camera joints)
- **Run systematic ablation studies** to understand feature importance

## Quick Start

### Basic Usage

Train with only joint positions (no velocities/torques):
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id your_dataset \
  --feature_selection.use_joint_velocities=false \
  --feature_selection.use_joint_torques=false \
  --steps 15000
```

Train with only one camera:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id your_dataset \
  --feature_selection.cameras='["cam_high"]' \
  --steps 15000
```

Train without camera movement data:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id your_dataset \
  --feature_selection.exclude_joint_groups='["camera"]' \
  --steps 15000
```

## Feature Selection Options

### Camera Selection

```bash
# Use only specific cameras
--feature_selection.cameras='["cam_high", "cam_low"]'

# Exclude specific cameras  
--feature_selection.exclude_cameras='["cam_stereo_left", "cam_stereo_right"]'
```

### State Feature Selection

```bash
# Joint data types
--feature_selection.use_joint_positions=true    # Default: true
--feature_selection.use_joint_velocities=true   # Default: true  
--feature_selection.use_joint_torques=false     # Default: false
--feature_selection.use_pressure_sensors=true   # Default: true

# Joint groups
--feature_selection.joint_groups='["left_arm", "right_arm"]'  # Only arms
--feature_selection.exclude_joint_groups='["camera"]'         # No camera joints
```

### Advanced Options

```bash
# Manual state dimension selection (advanced users)
--feature_selection.custom_state_indices='[0,1,2,10,11,12]'
```

## Common Ablation Studies

### 1. Camera Ablation

Test which cameras are most important:

```bash
# Baseline (all cameras)
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name baseline_all_cams

# Single camera tests
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name single_cam_high \
  --feature_selection.cameras='["cam_high"]'

python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name single_cam_wrist \
  --feature_selection.cameras='["cam_left_wrist"]'

# Stereo vs monocular
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name stereo_only \
  --feature_selection.cameras='["cam_stereo_left", "cam_stereo_right"]'
```

### 2. State Feature Ablation

Test which state features matter:

```bash
# Only positions
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name positions_only \
  --feature_selection.use_joint_velocities=false \
  --feature_selection.use_joint_torques=false \
  --feature_selection.use_pressure_sensors=false

# Positions + velocities
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name pos_vel \
  --feature_selection.use_joint_torques=false \
  --feature_selection.use_pressure_sensors=false

# No pressure sensors
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name no_pressure \
  --feature_selection.use_pressure_sensors=false
```

### 3. Body Part Ablation

Test which body parts are needed:

```bash
# Arms only (no hands)
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name arms_only \
  --feature_selection.joint_groups='["left_arm", "right_arm"]'

# Single arm
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name left_arm_only \
  --feature_selection.joint_groups='["left_arm", "left_hand"]'

# No camera movement
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name no_camera_movement \
  --feature_selection.exclude_joint_groups='["camera"]'
```

## Automated Ablation Studies

Use the ablation study script for systematic experiments:

### Camera Ablation Study
```bash
python scripts/run_ablation_study.py \
  --study_type cameras \
  --dataset_repo your_dataset \
  --wandb_project camera_ablation \
  --steps 10000
```

### State Feature Ablation Study  
```bash
python scripts/run_ablation_study.py \
  --study_type state_features \
  --dataset_repo your_dataset \
  --wandb_project state_ablation \
  --steps 10000
```

### Custom Ablation Study
```bash
python scripts/run_ablation_study.py \
  --study_type custom \
  --config_file examples/custom_ablation.yaml \
  --dataset_repo your_dataset \
  --wandb_project custom_ablation \
  --steps 10000
```

## Feature Mapping

Your dataset features are mapped as follows:

### Cameras (based on your 6-camera setup)
- `cam_left_high` → `color_0` (left head cam )
- `cam_right_high` → `color_1` (right head cam)  
- `cam_left_wrist` → `color_2` (left wrist camera)
- `cam_right_wrist` → `color_3` (right wrist camera)
- `cam_left_active` → `color_4` (stereo active head left)
- `cam_right_active` → `color_5` (stereo active head right)

### Joint Groups
- `left_arm` → 7 joints (shoulder to wrist)
- `right_arm` → 7 joints (shoulder to wrist)
- `left_hand` → Hand/gripper joints
- `right_hand` → Hand/gripper joints  
- `camera` → 2 joints (camera pan/tilt servos)

### State Features
- `*_pos` → Joint positions (qpos)
- `*_vel` → Joint velocities (qvel)
- `*_effort` → Joint torques
- `*_pressure_*` → Pressure sensor readings

## Tips for Effective Ablation Studies

1. **Start with baselines**: Always run a full-feature baseline first
2. **Use consistent training settings**: Keep batch size, learning rate, etc. the same
3. **Run multiple seeds**: Use different random seeds for robust results
4. **Monitor convergence**: Some minimal setups may need more training steps
5. **Compare final performance**: Look at both convergence speed and final performance
6. **Use WandB**: Enable logging to easily compare results

## Example: Finding Minimal Sensor Setup

```bash
# 1. Baseline (all features)
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name baseline

# 2. Test minimal cameras
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name minimal_cams \
  --feature_selection.cameras='["cam_high"]'

# 3. Test minimal state features  
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name minimal_state \
  --feature_selection.cameras='["cam_high"]' \
  --feature_selection.use_joint_velocities=false \
  --feature_selection.use_pressure_sensors=false

# 4. Test without camera movement
python lerobot/scripts/train.py --dataset.repo_id your_dataset --job_name minimal_no_cam_move \
  --feature_selection.cameras='["cam_high"]' \
  --feature_selection.use_joint_velocities=false \
  --feature_selection.use_pressure_sensors=false \
  --feature_selection.exclude_joint_groups='["camera"]'
```

This approach lets you find the minimal sensor configuration that still achieves good performance!
