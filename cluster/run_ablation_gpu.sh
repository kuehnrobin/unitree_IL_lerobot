#!/bin/bash
#SBATCH --job-name=act_parameter_study
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robin.kuhn@stud.uni-hannover.de

# Set up environment
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $CUDA_VISIBLE_DEVICES"

# Navigate to your project directory
cd $BIGWORK/unitree_IL_lerobot

# Activate conda environment
conda activate $SOFTWARE/humanoid/IL_env

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the ablation study
echo "Starting ablation study..."
python unitree_lerobot/scripts/run_ablation_study.py \
    --config_file ablation_config.yaml \
    --dataset_repo kuehnrobin/g1_cubes_box_no_hover \
    --wandb_project "dinov2_ablation_luis" \
    --steps 50000 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 1000 \
    --batch_size 8

echo "Job completed at: $(date)"

