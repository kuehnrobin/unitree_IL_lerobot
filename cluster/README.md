# Cluster Scripts for DINOv2 Training

Scripts for running DINOv2 vs ResNet18 ablation studies on the LUIS cluster with offline model support.

## 📁 Files

- **`download_dinov2_models.py`** - Downloads models for offline use (run with internet)
- **`setup_cluster.sh`** - Verifies cluster environment and models
- **`run_ablation_gpu.sh`** - SLURM job script for the ablation study

## 🚀 Quick Start

### 1. Download Models (Local Machine with Internet)
```bash
cd unitree_IL_lerobot/cluster
python download_dinov2_models.py --output_dir ./torch_models
```

### 2. Transfer to Cluster
```bash
# Transfer models
scp -r torch_models username@luis.uni-hannover.de:$BIGWORK/

# Transfer dataset  
scp -r your_dataset username@luis.uni-hannover.de:$BIGWORK/LargeFiles/g1_cubes_s_fixed

# Transfer project
scp -r unitree_IL_lerobot username@luis.uni-hannover.de:~/humanoid/humanoid_ws/src/
```

### 3. Setup on Cluster
```bash
cd ~/humanoid/humanoid_ws/src/unitree_IL_lerobot
bash cluster/setup_cluster.sh
```

### 4. Configure Environment
Edit the module loading script:
```bash
vim ~/load_modules.sh
```
Add your cluster's modules (Python, CUDA, PyTorch, etc.)

### 5. Submit Job
```bash
# Copy to home directory (cluster requirement)
cp cluster/run_ablation_gpu.sh ~/run_ablation_gpu.sh

# Edit email address
vim ~/run_ablation_gpu.sh  # Change your.email@uni-hannover.de

# Submit
sbatch ~/run_ablation_gpu.sh
```

## 📊 What It Does

Runs two experiments comparing:
- **ResNet18 Baseline** - Standard ACT with ImageNet pretrained backbone  
- **DINOv2 Small** - ACT with DINOv2-ViT-S/14 backbone

Results logged to WandB project: `dinov2_ablation_luis`

## 🔧 Key Features

- **Offline Model Loading**: No internet needed on compute nodes
- **Automatic Image Resizing**: Handles 480×640 images for DINOv2's 14×14 patches
- **Local Dataset Support**: Uses `$BIGWORK/LargeFiles/g1_cubes_s_fixed`
- **Robust Fallbacks**: Multiple model loading strategies

## 🐛 Quick Troubleshooting

**Missing models**: Run `python $BIGWORK/torch_models/verify_models.py`  
**Job status**: `squeue -u $USER`  
**Job logs**: Check `dinov2_ablation_study_<JOB_ID>.out`  
**Dataset issues**: Verify `$BIGWORK/LargeFiles/g1_cubes_s_fixed/` exists

## 📈 Monitor Progress

- **SLURM**: `squeue -u $USER`
- **WandB**: Check `dinov2_ablation_luis` project  
- **Logs**: `tail -f dinov2_ablation_study_<JOB_ID>.out`
