#!/bin/bash

# Example usage of the updated ablation study script

# Run ablation study with custom training parameters
python run_ablation_study.py \
    --config_file custom_ablation.yaml \
    --dataset_repo kuehnrobin/g1_cubes_box_s_61 \
    --wandb_project my_ablation_study \
    --steps 50000 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 1000 \
    --batch_size 12

# The above command will:
# 1. Add --policy.type=act to each training command
# 2. Use the experiment names from custom_ablation.yaml as job names
# 3. Save each policy in outputs/train/{date}/{time}_{experiment_name}/
# 4. Configure training parameters via CLI
# 5. Set appropriate WandB run names for tracking

echo "Ablation study completed!"
echo "Check outputs/train/ directory for saved policies"
echo "Each subdirectory will be named with the experiment name from the YAML config"
