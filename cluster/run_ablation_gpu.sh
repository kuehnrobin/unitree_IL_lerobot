#!/bin/bash -l
#SBATCH --job-name=act_ablation_study
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robin.kuehn@stud.uni-hannover.de

# Set up environment
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $CUDA_VISIBLE_DEVICES"

# Navigate to your project directory
cd $BIGWORK/unitree_IL_lerobot

# Load necessary modules 
module load Miniforge3

# Activate your conda/virtual environment
conda activate $SOFTWARE/humanoid/IL_env

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set up local model and dataset paths for cluster
export TORCH_HOME=$BIGWORK/torch_models
export HF_HOME=$BIGWORK/huggingface_cache
export TRANSFORMERS_CACHE=$BIGWORK/huggingface_cache
export HF_DATASETS_CACHE=$BIGWORK/datasets_cache

# Disable internet access for offline training
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HUB_OFFLINE=1

# W&B local logging directory
export WANDB_DIR=$BIGWORK/wandb_logs

# Local dataset path
export LOCAL_DATASET_PATH=$BIGWORK/LargeFiles/g1_cubes_s_fixed

# Run the ablation study
echo "Starting ablation study..."
echo "Using local dataset: $LOCAL_DATASET_PATH"


srun python unitree_lerobot/scripts/run_ablation_study.py \
        --config_file unitree_lerobot/examples/cluster_run_config.yaml \
        --dataset_repo "$LOCAL_DATASET_PATH" \
        --wandb_project "act_ablation_luis" \
        --steps 100000 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --log_freq 1000 \
        --batch_size 12

echo "Job completed at: $(date)"

