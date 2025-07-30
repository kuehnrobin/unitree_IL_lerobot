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

# Set output directory for training results
export OUTPUTS_DIR=$BIGWORK/outputs


# Run the ablation study
echo "Starting ablation study..."
echo "Using local dataset: $LOCAL_DATASET_PATH"
echo "Using torch cache: $TORCH_HOME"
echo "W&B logs directory: $WANDB_DIR"
echo "Running in OFFLINE mode - W&B logs saved locally"
echo ""
echo "Environment variables:"
echo "WANDB_MODE=$WANDB_MODE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
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
echo "Training completed!"
echo "Training outputs saved to: $OUTPUTS_DIR"
echo "W&B logs saved to: $WANDB_DIR"
echo ""
echo "To view W&B logs after training:"
echo "1. Copy logs to local machine: scp -r username@luis:$WANDB_DIR ."
echo "2. Install wandb locally: pip install wandb"
echo "3. Sync offline logs: wandb sync wandb_logs/"
echo "4. View in browser: wandb server"

echo "Job completed at: $(date)"

