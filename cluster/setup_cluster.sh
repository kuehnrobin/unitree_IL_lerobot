#!/bin/bash
"""
Cluster Setup Script

This script sets up the environment for running DINOv2 training on the LUIS cluster.
It creates necessary directories and helps transfer models and datasets.

Usage:
  1. First, download models on a machine with internet access:
     python cluster/download_dinov2_models.py --output_dir ./torch_models
     
  2. Transfer to cluster:
     scp -r torch_models username@luis.uni-hannover.de:$BIGWORK/
     scp -r your_local_dataset username@luis.uni-hannover.de:$BIGWORK/LargeFiles/g1_cubes_s_fixed
     
  3. Run this setup script on the cluster:
     bash cluster/setup_cluster.sh
"""

echo "=== LUIS Cluster Setup for DINOv2 Training ==="

# Check if running on cluster
if [[ -z "$BIGWORK" ]]; then
    echo "Warning: \$BIGWORK environment variable not found."
    echo "Are you running this on the LUIS cluster?"
    echo "Setting BIGWORK to current directory for testing..."
    export BIGWORK="$(pwd)/bigwork_test"
fi

echo "Using BIGWORK: $BIGWORK"

# Create necessary directories
echo "Creating directories..."
mkdir -p $BIGWORK/torch_models
mkdir -p $BIGWORK/huggingface_cache
mkdir -p $BIGWORK/datasets_cache
mkdir -p $BIGWORK/LargeFiles

echo "✓ Created cache directories"

# Check for DINOv2 models
echo ""
echo "Checking for DINOv2 models..."
MODEL_DIR="$BIGWORK/torch_models"
DINOV2_MODELS=("dinov2_vits14" "dinov2_vits14_reg" "dinov2_vitb14" "dinov2_vitb14_reg" "dinov2_vitl14" "dinov2_vitl14_reg")
RESNET_MODELS=("resnet18" "resnet34" "resnet50")

missing_dinov2=()
for model in "${DINOV2_MODELS[@]}"; do
    if [[ -f "$MODEL_DIR/${model}.pth" ]]; then
        echo "✓ Found: ${model}.pth"
    else
        echo "✗ Missing: ${model}.pth"
        missing_dinov2+=("$model")
    fi
done

echo ""
echo "Checking for ResNet models..."
missing_resnet=()
for model in "${RESNET_MODELS[@]}"; do
    if [[ -f "$MODEL_DIR/${model}_pretrained.pth" ]]; then
        echo "✓ Found: ${model}_pretrained.pth"
    else
        echo "✗ Missing: ${model}_pretrained.pth"
        missing_resnet+=("$model")
    fi
done

if [[ ${#missing_dinov2[@]} -gt 0 ]] || [[ ${#missing_resnet[@]} -gt 0 ]]; then
    echo ""
    echo "Missing models detected. Please download them first:"
    echo "1. On a machine with internet access, run:"
    echo "   python cluster/download_dinov2_models.py --output_dir ./torch_models"
    echo "2. Transfer to cluster:"
    echo "   scp -r torch_models username@luis.uni-hannover.de:$BIGWORK/"
else
    echo "✓ All models found!"
fi

# Check for dataset
echo ""
echo "Checking for local dataset..."
DATASET_DIR="$BIGWORK/LargeFiles/g1_cubes_s_fixed"
if [[ -d "$DATASET_DIR" ]]; then
    echo "✓ Found local dataset: $DATASET_DIR"
    
    # Check dataset structure
    if [[ -f "$DATASET_DIR/data.json" ]] || [[ -d "$DATASET_DIR/data" ]]; then
        echo "✓ Dataset structure looks correct"
    else
        echo "⚠ Dataset found but structure unclear. Expected data.json or data/ directory"
    fi
else
    echo "✗ Local dataset not found: $DATASET_DIR"
    echo "Please transfer your dataset:"
    echo "  scp -r your_local_dataset username@luis.uni-hannover.de:$BIGWORK/LargeFiles/g1_cubes_s_fixed"
fi

# Display environment variables
echo ""
echo "Environment setup:"
echo "export TORCH_HOME=$BIGWORK/torch_models"
echo "export HF_HOME=$BIGWORK/huggingface_cache"
echo "export TRANSFORMERS_CACHE=$BIGWORK/huggingface_cache"
echo "export HF_DATASETS_CACHE=$BIGWORK/datasets_cache"
echo "export LOCAL_DATASET_PATH=$BIGWORK/LargeFiles/g1_cubes_s_fixed"

# Create module load script
MODULE_SCRIPT="$HOME/load_modules.sh"
cat > "$MODULE_SCRIPT" << 'EOF'
#!/bin/bash
# Load necessary modules for DINOv2 training
# Adjust these based on your cluster's available modules

# Example module loads (uncomment and adjust as needed):
# module load Python/3.9.6-GCCcore-11.2.0
# module load CUDA/11.7.0  
# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
# module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

# Or if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lerobot_env

echo "Loaded modules for DINOv2 training"
EOF

chmod +x "$MODULE_SCRIPT"
echo ""
echo "✓ Created module loading script: $MODULE_SCRIPT"
echo "  Edit this file to match your cluster's module system"

# Summary
echo ""
echo "=== Setup Summary ==="
echo "1. Cache directories: ✓ Created"
echo "2. DINOv2 models: ${#missing_dinov2[@]} missing"
echo "3. ResNet models: ${#missing_resnet[@]} missing"
echo "4. Local dataset: $(if [[ -d "$DATASET_DIR" ]]; then echo "✓ Found"; else echo "✗ Missing"; fi)"
echo "5. Module script: ✓ Created"
echo ""
echo "Next steps:"
echo "1. Edit $MODULE_SCRIPT to load correct modules"
echo "2. Transfer missing models/datasets if needed"
echo "3. Submit job: sbatch cluster/run_ablation_gpu.sh"
