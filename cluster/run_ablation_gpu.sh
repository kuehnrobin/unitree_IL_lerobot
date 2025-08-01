#!/bin/bash -l
#SBATCH --job-name=act_ablation_study
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100m40:1
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
export TRANSFORMERS_CACHE=$BIGWORK/huggingface_cache

# Configure cache directories to avoid filling home directory
export PIP_CACHE_DIR=$SOFTWARE/humanoid/.cache/pip
export TORCH_HOME=$SOFTWARE/humanoid/.cache/torch
export HF_HOME=$BIGWORK/LargeFiles/huggingface
export HF_DATASETS_CACHE=$BIGWORK/LargeFiles/huggingface/datasets
export HF_HUB_CACHE=$BIGWORK/LargeFiles/huggingface/hub
export HF_LEROBOT_HOME=$BIGWORK/LageFiles/huggingface/lerobot



# Configure for offline W&B logging (cluster branch - always offline)
export WANDB_MODE=offline
export WANDB_DIR=$BIGWORK/wandb_logs

# Ensure offline operation
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HUB_OFFLINE=1

# Local dataset path
export LOCAL_DATASET_PATH=$BIGWORK/LargeFiles/g1_cubes_s_fixed

# Set output directory for training results
export OUTPUTS_DIR=$BIGWORK/outputs


# Run the ablation study
echo "=== Starting Cluster Training ==="
echo "W&B logs will be saved to: $WANDB_DIR"
echo "Training outputs will be saved to: $OUTPUTS_DIR"
echo "Running in OFFLINE mode - no internet required"
echo ""

# Verify local paths exist
if [[ ! -d "$LOCAL_DATASET_PATH" ]]; then
    echo "ERROR: Dataset path does not exist: $LOCAL_DATASET_PATH"
    echo "Please ensure the dataset is copied to the cluster first."
    exit 1
fi

if [[ ! -d "$TORCH_HOME" ]]; then
    echo "ERROR: Torch models path does not exist: $TORCH_HOME"
    echo "Please run the download_dinov2_models.py script first."
    exit 1
fi

echo "✓ Found dataset: $LOCAL_DATASET_PATH"
echo "✓ Found torch models: $TORCH_HOME"
echo ""

# Run the ablation study (cluster branch - simplified)
srun python unitree_lerobot/scripts/run_ablation_study.py \
        --config_file unitree_lerobot/examples/cluster_run_config.yaml \
        --dataset_repo "$LOCAL_DATASET_PATH" \
        --wandb_project "act_ablation_luis" \
        --steps 100000 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --log_freq 1000 \
        --batch_size 12

echo ""
echo "Job completed at: $(date)"
echo "Check results in: $OUTPUTS_DIR"
echo "Check W&B logs in: $WANDB_DIR"

