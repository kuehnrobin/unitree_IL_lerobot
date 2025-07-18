# Ablation Study Script Updates

## Changes Made

### 1. Added CLI Parameters
The `run_ablation_study.py` script now supports these additional CLI parameters:
- `--eval_freq`: Evaluation frequency (default: 10000)
- `--save_freq`: Save checkpoint frequency (default: 10000)  
- `--log_freq`: Logging frequency (default: 1000)
- `--batch_size`: Training batch size (default: 12)

### 2. Policy Type Configuration
- Added `--policy.type=act` to all generated training commands
- This ensures the correct policy type is used for all experiments

### 3. Proper Experiment Naming
- Uses experiment names from `custom_ablation.yaml` as `job_name` parameter
- This creates output directories like: `outputs/train/{date}/{time}_{experiment_name}/`
- Added `--wandb.run_name={name}` for better WandB tracking
- Each saved policy will be easily identifiable by its experiment name

## Usage Example

```bash
python run_ablation_study.py \
    --config_file custom_ablation.yaml \
    --dataset_repo kuehnrobin/g1_cubes_box_s_61 \
    --wandb_project my_ablation_study \
    --steps 50000 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 1000 \
    --batch_size 12
```

## Output Structure
After running the ablation study, you'll find saved policies in:
```
outputs/train/
├── 2025-07-18/
│   ├── 14-30-15_baseline_all_features/
│   ├── 14-45-22_no_joint_velocities_and_torques/
│   ├── 15-02-18_no_active_cam/
│   └── ...
```

Each directory name includes the experiment name from your YAML config, making it easy to identify which policy corresponds to which configuration.

## Generated Commands
The script now generates commands like:
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=kuehnrobin/g1_cubes_box_s_61 \
    --policy.type=act \
    --steps=50000 \
    --eval_freq=10000 \
    --save_freq=10000 \
    --log_freq=1000 \
    --batch_size=12 \
    --job_name=baseline_all_features \
    --wandb.project=my_ablation_study \
    --wandb.run_name=baseline_all_features \
    --wandb.enable=true \
    --feature_selection.use_joint_velocities=false \
    ...
```
