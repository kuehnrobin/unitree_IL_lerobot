# Reproducing Paper Experiments

This directory contains scripts and configurations to reproduce the experiments from our RAL paper.

## Overview

Our paper evaluates **15 sensor configurations** across **2 manipulation tasks** using the **Unified Ablation Framework**. All policies were trained on identical demonstration sequences using runtime sensor masking.

## Quick Start

### Prerequisites

1. Install the UAF_lerobot framework (see main README)
2. Download or prepare datasets:
   - Sort Cans task dataset
   - Grasp Cubes task dataset

### Running the Ablation Study

The core of our methodology is the ablation study script that trains multiple sensor configurations from a single master dataset:

```bash
cd experiments/scripts

# Run complete ablation study (15 policies)
python run_ablation_study.py \
  --config_file ../configs/paper_ablation.yaml \
  --dataset_repo your_username/master_dataset \
  --wandb_project ral_reproduction \
  --steps 50000 \
  --eval_freq 10000 \
  --save_freq 10000
```

### Individual Policy Training

To train a specific configuration from the paper:

```bash
# Minimal Active Vision (R-A) - Best overall performer
python unitree_lerobot/UAF_lerobot/src/lerobot/scripts/train.py \
  --dataset.repo_id your_username/master_dataset \
  --policy.type=act \
  --feature_selection.cameras='["cam_head_active"]' \
  --steps 50000 \
  --wandb.enable=true \
  --wandb.project=ral_reproduction

# Complex Multi-Sensor (R-WA-P) - Best for structured tasks
python unitree_lerobot/UAF_lerobot/src/lerobot/scripts/train.py \
  --dataset.repo_id your_username/master_dataset \
  --policy.type=act \
  --feature_selection.cameras='["cam_wrist_left", "cam_wrist_right", "cam_head_active"]' \
  --feature_selection.use_pressure_sensors=true \
  --steps 50000 \
  --wandb.enable=true \
  --wandb.project=ral_reproduction
```

## Policy Naming Convention

As described in Section III.C of the paper:
- **R** = ResNet18 backbone
- **A** = Active stereo camera
- **S** = Static wide-angle camera
- **W** = Wrist cameras
- **P** = Pressure sensors
- **V** = Joint velocities
- **T** = Joint torques

Examples:
- `R-A`: ResNet18 + Active camera only
- `R-WA-P`: ResNet18 + Wrist cameras + Active camera + Pressure sensors
- `R-S_LWA-PV_AT_A`: Complex configuration with left wrist active, multiple modalities

## Configuration Files

### `paper_ablation.yaml`
Complete ablation study configuration matching the 15 policies evaluated in the paper.

**Sensor Configurations Tested:**

1. **Camera Variations:**
   - R-A (Active only)
   - R-S (Static wide-angle)
   - R-WA (Wrist + Active)
   - R-SW (Static + Wrist)

2. **Tactile Integration:**
   - R-A-P (Active + Pressure)
   - R-WA-P (Wrist + Active + Pressure)

3. **Proprioception:**
   - R-WA-PV_AT_A (Full proprioceptive + tactile)

## Cluster Training (Optional)

For training on SLURM clusters:

```bash
# Edit SLURM configuration
vim experiments/scripts/run_ablation_slurm.sh

# Submit job
sbatch experiments/scripts/run_ablation_slurm.sh
```

**Note**: Remove cluster-specific paths (e.g., LUIS cluster references) and adapt to your infrastructure.

## Evaluation

After training, evaluate policies on the real robot:

```bash
# Evaluate on Sort Cans task
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py \
  --policy.path=outputs/train/R-A/checkpoints/last/pretrained_model/ \
  --repo_id=your_username/can_sorting_dataset \
  --arm_speed 10.0

# Evaluate on Grasp Cubes task (20 trials for spatial generalization)
for i in {1..20}; do
  python unitree_lerobot/eval_robot/eval_g1/eval_g1.py \
    --policy.path=outputs/train/R-A/checkpoints/last/pretrained_model/ \
    --repo_id=your_username/cube_grasping_dataset \
    --arm_speed 10.0
done
```

## Expected Results

### Task 1: Sort Cans
- **R-A**: ~94.4% success, ~3.57 min execution
- **R-WA-P**: ~97.6% success, ~3.17 min execution (best)
- **R-A-P**: ~67.3% success (demonstrates pressure sensor interference)

### Task 2: Grasp Cubes
- **R-A**: ~87.5% success, ~0.38 min execution (best)
- **R-S**: ~10% success (hovering behavior)
- **R-WA-P**: ~68.1% success

Refer to Table I in the paper for complete results.

## Troubleshooting

**Issue**: Hovering behavior in cube task
**Cause**: Combining active and static cameras creates feature interference
**Solution**: Use only active camera (R-A) for spatial generalization tasks

**Issue**: Low success with pressure sensors
**Cause**: Distribution shift in data-limited regimes
**Solution**: Either collect more data or add wrist cameras for visual disambiguation

## Citation

```bibtex
@article{kuehn2026more,
  title={More Is Not Always Better: Active Stereo Camera Setup Outperforms Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation Task},
  author={K{\"u}hn, Robin and Bank, Dennis and Schappler, Moritz and Seel, Thomas},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```
