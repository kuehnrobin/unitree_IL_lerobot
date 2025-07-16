#!/bin/bash
# Comprehensive Ablation Study Runner
# This script runs systematic experiments across features and ACT parameters

DATASET_REPO="kuehnrobin/g1_cubes_box"
BASE_WANDB_PROJECT="comprehensive_ablation"
STEPS=15000
EVAL_FREQ=5000
LOG_FREQ=500

echo "🚀 Starting comprehensive ablation study for $DATASET_REPO"
echo "📊 Results will be logged to WandB projects with prefix: $BASE_WANDB_PROJECT"
echo "🕐 Each experiment will run for $STEPS steps"
echo ""

# Function to run a study and handle errors
run_study() {
    local study_type=$1
    local description=$2
    local wandb_suffix=$3
    
    echo "🔄 Starting $description..."
    python scripts/run_ablation_study.py \
        --study_type "$study_type" \
        --dataset_repo "$DATASET_REPO" \
        --wandb_project "${BASE_WANDB_PROJECT}_${wandb_suffix}" \
        --steps $STEPS
    
    if [ $? -eq 0 ]; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
        read -p "Continue with next study? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏹️ Stopping ablation study"
            exit 1
        fi
    fi
    echo ""
}

# Function to run custom study
run_custom_study() {
    local config_file=$1
    local description=$2
    local wandb_suffix=$3
    
    echo "🔄 Starting $description..."
    python scripts/run_ablation_study.py \
        --study_type custom \
        --config_file "$config_file" \
        --dataset_repo "$DATASET_REPO" \
        --wandb_project "${BASE_WANDB_PROJECT}_${wandb_suffix}" \
        --steps $STEPS
    
    if [ $? -eq 0 ]; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
        read -p "Continue with next study? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏹️ Stopping ablation study"
            exit 1
        fi
    fi
    echo ""
}

echo "📋 Study Plan:"
echo "1. Camera ablation (12 experiments, ~3-4 hours)"
echo "2. State feature ablation (9 experiments, ~2-3 hours)"
echo "3. ACT architecture ablation (8 experiments, ~2-3 hours)"
echo "4. ACT hyperparameters (10 experiments, ~3-4 hours)"
echo "5. Combined optimization (8 experiments, ~2-3 hours)"
echo "6. ACT architecture comprehensive (custom, ~6-8 hours)"
echo "7. Combined optimization comprehensive (custom, ~6-8 hours)"
echo ""
echo "Total estimated time: 24-30 hours"
echo ""

read -p "Continue with full study? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏹️ Ablation study cancelled"
    exit 0
fi

echo "⏰ $(date): Starting comprehensive ablation study"

# 1. Camera ablation (quick baseline understanding)
run_study "cameras" "Camera Ablation Study" "cameras"

# 2. State features ablation 
run_study "state_features" "State Feature Ablation Study" "state_features"

# 3. ACT architecture ablation (built-in configurations)
run_study "act_architecture" "ACT Architecture Ablation Study" "act_arch"

# 4. ACT hyperparameters
run_study "act_hyperparameters" "ACT Hyperparameter Ablation Study" "act_hyper"

# 5. Combined optimization (built-in combinations)
run_study "combined" "Combined Feature + ACT Ablation Study" "combined"

# 6. Comprehensive ACT architecture study (custom YAML)
run_custom_study "examples/act_architecture_study.yaml" "Comprehensive ACT Architecture Study" "act_comprehensive"

# 7. Comprehensive combined optimization (custom YAML)
run_custom_study "examples/combined_optimization_study.yaml" "Comprehensive Combined Optimization Study" "combined_comprehensive"

echo "🎉 All ablation studies completed!"
echo "⏰ $(date): Study finished"
echo ""
echo "📊 Results Summary:"
echo "Check the following WandB projects for detailed results:"
echo "  🎥 Camera ablation: ${BASE_WANDB_PROJECT}_cameras"
echo "  🔧 State features: ${BASE_WANDB_PROJECT}_state_features"
echo "  🏗️ ACT architecture: ${BASE_WANDB_PROJECT}_act_arch"
echo "  ⚙️ ACT hyperparameters: ${BASE_WANDB_PROJECT}_act_hyper"
echo "  🔄 Combined optimization: ${BASE_WANDB_PROJECT}_combined"
echo "  📈 ACT comprehensive: ${BASE_WANDB_PROJECT}_act_comprehensive"
echo "  🎯 Combined comprehensive: ${BASE_WANDB_PROJECT}_combined_comprehensive"
echo ""
echo "🔍 Next steps:"
echo "1. Review training curves in WandB to identify best configurations"
echo "2. Run real robot evaluations on promising configurations using eval_g1.py"
echo "3. Consider running focused studies on top-performing parameter combinations"
echo ""
echo "💡 Pro tip: Look for configurations that achieve good performance with:"
echo "   - Fast convergence (fewer training steps needed)"
echo "   - Stable training (smooth loss curves)"
echo "   - Good final performance (low validation loss)"
echo "   - Efficient inference (smaller models, shorter chunks)"
