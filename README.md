# More Is Not Always Better: Active Stereo Camera Setup Outperforms Multi-Sensor Setup in ACT Imitation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-RAL%202026-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)]()

**Robin Kühn¹, Dennis Bank¹, Moritz Schappler¹, Thomas Seel¹**

¹Institute of Mechatronic Systems, Leibniz University Hannover, Germany

> **TL;DR**: We demonstrate that a minimal active stereo camera setup outperforms complex multi-sensor arrays for humanoid manipulation with Action Chunking Transformers (ACT), achieving 87.5% success in spatial generalization tasks. Our **Unified Ablation Framework** enables rigorous sensor comparison by training all policies on identical demonstration sequences.

---

## 📰 News

- **[Jan 2026]** Repository released with reproduction code
- **[Jan 2026]** Paper submitted to IEEE RA-L

## 🎬 Demo Videos

<table>
<tr>
<td width="50%" align="center">
<h3>Grasp Cubes (Spatial Generalization)</h3>
<img src="media/demos/cubes_demo.gif" alt="Cubes Demo" width="100%"/>
<br/>
<a href="https://seafile.projekt.uni-hannover.de/f/d40a7d127f5a4941a352/">📹 Full Video (HD)</a>
<br/>
<em>R-A policy achieving 87.5% success on randomized cube positions</em>
</td>
<td width="50%" align="center">
<h3>Sort Cans (Structured Task)</h3>
<img src="media/demos/cans_demo.gif" alt="Cans Demo" width="100%"/>
<br/>
<a href="https://seafile.projekt.uni-hannover.de/f/fc056a5182c34eaa8c84/">📹 Full Video (HD)</a>
<br/>
<em>R-A policy achieving 94.4% success in can sorting</em>
</td>
</tr>
</table>

## 🔬 Abstract

While Action Chunking with Transformers (ACT) enables rapid task acquisition for humanoid robots, there is no consensus on optimal sensor configuration. We benchmark **15 sensor combinations** on the Unitree G1, evaluating visual, proprioceptive, and tactile modalities across two manipulation tasks.

**Key Finding**: Strategic sensor selection outperforms complex configurations for small datasets. A minimal active stereo camera ($R-A$) achieved 87.5% success in spatial generalization, while adding pressure sensors to this setup reduced performance from 94% to 67% due to introduced noise.

## 🏆 Main Contributions

1. **Unified Ablation Framework**: Open-source toolchain using runtime sensor masking on a master dataset, eliminating human demonstration variance
2. **Tactile Integration Study**: First systematic evaluation of finger-tip pressure sensors with ACT-based policies
3. **Design Guidelines**: Empirical evidence that active vision often suffices, enabling cost-effective system design

## 📊 Key Results

<p align="center">
<img src="media/results/pareto_can_sorting.png" alt="Can Sorting Results" width="48%"/>
<img src="media/results/pareto_cube_in_box.png" alt="Cube Grasping Results" width="48%"/>
</p>
<p align="center"><em>Execution time vs. success rate for different sensor configurations. Active vision (R-A) achieves near-optimal performance with minimal hardware complexity.</em></p>

### Task 1: Sort Cans (Structured Environment)

| Configuration | Success Rate | Execution Time | Hardware Complexity |
|--------------|--------------|----------------|---------------------|
| **R-A** (Ours) | 94.4% | 3.57 min | ⭐ Minimal |
| R-WA-P | **97.6%** | **3.17 min** | ⚠️ High |
| R-A-P | 67.3% ❌ | - | Medium |

*Adding pressure sensors without visual support (R-A-P) caused 27% performance drop*

### Task 2: Grasp Cubes (Spatial Generalization)

| Configuration | Success Rate | Execution Time | Generalization |
|--------------|--------------|----------------|----------------|
| **R-A** (Ours) | **87.5%** | **0.38 min** | ✅ Excellent |
| R-WA-P | 68.1% | 0.34 min | ⚠️ Moderate |
| R-S | 10.0% ❌ | - | ❌ Failed |

*Static cameras exhibited "hovering behavior" due to feature interference*

**Legend**: R=ResNet18, A=Active Camera, S=Static Camera, W=Wrist Cameras, P=Pressure Sensors

## 🚀 Quick Start

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/kuehnrobin/UAF_unitree_g1.git
cd UAF_unitree_g1

# Create environment
conda create -y -n uaf_lerobot python=3.10
conda activate uaf_lerobot

# Install dependencies
cd unitree_lerobot/UAF_lerobot && pip install -e .
cd ../.. && pip install -e .
```

### Train a Policy

```bash
# Minimal Active Vision (R-A) - Recommended baseline
python unitree_lerobot/UAF_lerobot/src/lerobot/scripts/train.py \
  --dataset.repo_id=your_username/master_dataset \
  --policy.type=act \
  --feature_selection.cameras='["cam_head_active"]' \
  --steps=50000

# Run complete ablation study (15 configurations)
python experiments/scripts/run_ablation_study.py \
  --config_file=experiments/configs/paper_ablation.yaml \
  --dataset_repo=your_username/master_dataset \
  --wandb_project=ral_reproduction
```

### Evaluate on Real Robot

```bash
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py \
  --policy.path=outputs/train/R-A/checkpoints/last/pretrained_model/ \
  --repo_id=your_username/master_dataset
```

## 📁 Repository Structure

```
UAF_unitree_g1/
├── analysis/                    # Scripts to reproduce paper figures
│   ├── plot_wandb/             # Pareto plots (Fig. 3 & 4) and analysis
│   └── README.md               # Guide to regenerating figures
├── experiments/                 # Reproducibility configs
│   ├── configs/                # Paper ablation study YAML
│   ├── scripts/                # Training and evaluation scripts
│   └── README.md               # Detailed reproduction guide
├── unitree_lerobot/
│   ├── UAF_lerobot/            # Fork of LeRobot with ablation framework
│   ├── utils/                  # Data conversion (JSON → LeRobot)
│   ├── eval_robot/             # Real robot evaluation
│   ├── FEATURE_SELECTION_README.md  # Unified Ablation Framework docs
│   └── AUGMENTATION_README.md  # Data augmentation guide
├── docs/                        # Additional documentation
├── test/                        # Unit tests
└── CITATION.bib                # BibTeX for citation
```

## 🎯 Reproducing Paper Results

### Step 1: Prepare Master Dataset

Collect demonstrations with **all** sensor modalities:
- Active stereo camera
- Static wide-angle cameras
- Wrist cameras
- Joint positions, velocities, torques
- Fingertip pressure sensors

```bash
# Convert your teleoperation data
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir=/path/to/json_dataset \
  --repo-id=your_username/master_dataset \
  --robot_type=Unitree_G1_Dex3
```

### Step 2: Run Ablation Study

```bash
python experiments/scripts/run_ablation_study.py \
  --config_file=experiments/configs/paper_ablation.yaml \
  --dataset_repo=your_username/master_dataset \
  --steps=50000 \
  --eval_freq=10000
```

This trains all 15 sensor configurations on **identical demonstration sequences** using runtime sensor masking.

### Step 3: Generate Paper Figures

```bash
# Pareto plots (Fig. 3 & 4)
python analysis/plot_wandb/pareto_plots.py \
  --data=analysis/plot_wandb/can_policies.csv \
  --output=figures/pareto_can_sorting.pdf

python analysis/plot_wandb/pareto_plots.py \
  --data=analysis/plot_wandb/cubes_policies.csv \
  --output=figures/pareto_cube_in_box.pdf
```

See [analysis/README.md](analysis/README.md) for detailed instructions.

## 🛠️ Unified Ablation Framework

Our key methodological contribution: train multiple sensor configurations from a single master dataset.

<p align="center">
<img src="media/methods/ablation_framework.png" alt="Unified Ablation Framework" width="70%"/>
</p>
<p align="center"><em>Runtime sensor masking enables training 15 policies on identical demonstrations, eliminating human variance.</em></p>

**Traditional Approach** ❌:
- Collect Dataset A (cameras only)
- Collect Dataset B (cameras + pressure)
- Human variance confounds results

**UAF Approach** ✅:
- Collect once (all sensors)
- Runtime masking selects features
- All policies train on identical demos

**Usage:**

```python
# Define feature selection
from lerobot.configs.train import FeatureSelectionConfig

config = FeatureSelectionConfig(
    cameras=["cam_head_active"],  # Only active camera
    use_joint_velocities=False,    # Disable velocities
    use_pressure_sensors=False     # Disable pressure
)

# Apply during training
python lerobot/scripts/train.py \
  --feature_selection.cameras='["cam_head_active"]' \
  --feature_selection.use_pressure_sensors=false
```

See [unitree_lerobot/FEATURE_SELECTION_README.md](unitree_lerobot/FEATURE_SELECTION_README.md) for complete API.

## 📦 Datasets

### Training Datasets

Available on HuggingFace:

- **Can Sorting**: `kuehnrobin/g1_sort_cans_master` (TBD)
- **Cube Grasping**: `kuehnrobin/g1_grasp_cubes_master` (TBD)

### Benchmark Dataset (OpenTelevision)

We compare against the OpenTelevision baseline using their published can sorting task.
& Teleoperation Setup

<p align="center">
<img src="media/methods/teleop_setup.png" alt="Teleoperation Setup" width="65%"/>
</p>
<p align="center"><em>VR-based teleoperation system enabling active perception data collection. Operator's head movements are synchronized with the robot's camera system.</em></p>

**Required Hardware**:## 🤖 Hardware Requirements

- **Robot**: Unitree G1 humanoid with Dex3-1 hands
- **Cameras**:
  - Active: OAK-D stereo camera (mounted on pan-tilt head)
  - Static: Wide-angle RGB cameras (head-mounted)
  - Wrist: Optional close-up cameras
- **Sensors**: Optional fingertip pressure sensors (12 per hand)
- **Teleoperation**: Meta Quest 3 VR headset

## 📈 Comparison to State-of-the-Art

| Method | Backbone | Success (Can Task) | Hardware | Dataset Size |
|--------|----------|-------------------|----------|--------------|
| OpenTelevision (ResNet18) | ResNet18 | 83% pick, 50% place | Active Camera | 50 episodes |
| **R-A (Ours)** | ResNet18 | **94.4%** overall | Active Camera | 80 episodes |
| **R-WA-P (Ours)** | ResNet18 | **97.6%** overall | Active + Wrist + Pressure | 80 episodes |

*Note: Direct comparison is approximate due to different evaluation protocols*

## 💡 Design Guidelines

Based on our findings, we recommend:

1. **Start with Active Vision**: Single active stereo camera ($R-A$) as baseline
2. **Avoid Co-Located Redundancy**: Don't combine active + static cameras on same link
3. **Add Tactile Carefully**: Pressure sensors require supporting visual context (wrist cameras)
4. **Prioritize Data Quality**: 80 high-quality episodes > 200 noisy episodes

## 🔬 Limitations

- Results specific to ACT architecture (not tested with Diffusion Policy)
- Data-limited regime (<100 episodes); benefits may differ at scale
- Tabletop manipulation only; dynamic tasks may require different sensors
- VR teleoperation latency (0.5-1.0s) may introduce artifacts

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{kuehn2026more,
  title={More Is Not Always Better: Active Stereo Camera Setup Outperforms 
         Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation Task},
  author={K{\"u}hn, Robin and Bank, Dennis and Schappler, Moritz and Seel, Thomas},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```

## 🙏 Acknowledgments

This work builds upon:
- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face robotics library
- [OpenTelevision](https://github.com/OpenTeleVision/TeleVision) - Active perception framework
- [Unitree SDK](https://github.com/unitreerobotics/unitree_sdk2_python) - Robot communication

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](unitree_lerobot/UAF_lerobot/CONTRIBUTING.md) for guidelines.

## 📧 Contact

- Robin Kühn: robin.kuehn@imes.uni-hannover.de
- Institute Website: https://www.imes.uni-hannover.de

---

**Related Repositories:**
- [UAF_lerobot](https://github.com/kuehnrobin/UAF_lerobot) - Our LeRobot fork with ablation framework
- [Unitree Datasets](https://huggingface.co/unitreerobotics) - Official Unitree datasets
- [AVP Teleoperate](https://github.com/unitreerobotics/avp_teleoperate) - VR teleoperation system
