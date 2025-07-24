#!/bin/bash
#SBATCH --job-name=dinov2_ablation_study
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
#SBATCH --mail-user=your.email@uni-hannover.de

# Set up environment
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $CUDA_VISIBLE_DEVICES"

# Navigate to your project directory
cd /home/robin/humanoid/humanoid_ws/src/unitree_IL_lerobot

# Load necessary modules (adjust based on your cluster's module system)
# module load Python/3.9.6-GCCcore-11.2.0
# module load CUDA/11.7.0
# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# Activate your conda/virtual environment
# Example for conda:
# source /home/robin/miniconda3/etc/profile.d/conda.sh
# conda activate lerobot_env

# Example for venv:
# source /path/to/your/venv/bin/activate

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set up local model and dataset paths for cluster
export TORCH_HOME=$BIGWORK/torch_models
export HF_HOME=$BIGWORK/huggingface_cache
export TRANSFORMERS_CACHE=$BIGWORK/huggingface_cache
export HF_DATASETS_CACHE=$BIGWORK/datasets_cache

# Create directories if they don't exist
mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Local dataset path
export LOCAL_DATASET_PATH=$BIGWORK/LargeFiles/g1_cubes_s_fixed

# Run the ablation study
echo "Starting ablation study..."
echo "Using local dataset: $LOCAL_DATASET_PATH"
echo "Using torch cache: $TORCH_HOME"

python unitree_lerobot/scripts/run_ablation_study.py \
    --config_file unitree_lerobot/examples/cluster_run_config.yaml \
    --dataset_repo "$LOCAL_DATASET_PATH" \
    --wandb_project "dinov2_ablation_luis" \
    --steps 50000 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 1000 \
    --batch_size 8

echo "Job completed at: $(date)"

# Optional: Clean up temporary files
# rm -f ablation_config.yaml
