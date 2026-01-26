# Analysis Scripts for Paper Figures

This directory contains scripts to reproduce the figures from our RAL paper:
**"More Is Not Always Better: Active Stereo Camera Setup Outperforms Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation Task"**

## Paper Figures

### Figure 3: Pareto Plot - Sort Cans Task
**Script**: `pareto_plots.py`
**Data**: `can_policies.csv`
**Description**: Execution time vs success rate for different sensor configurations on the structured can sorting task.

```bash
python analysis/pareto_plots.py --task cans --data analysis/can_policies.csv --output figures/pareto_can_sorting.pdf
```

### Figure 4: Pareto Plot - Grasp Cubes Task
**Script**: `pareto_plots.py`
**Data**: `cubes_policies.csv`
**Description**: Execution time vs success rate showing spatial generalization performance.

```bash
python analysis/pareto_plots.py --task cubes --data analysis/cubes_policies.csv --output figures/pareto_cube_in_box.pdf
```

## Analysis Scripts Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `pareto_plots.py` | Generate Pareto frontier plots | Figures 3 & 4 |
| `plot_training_losses.py` | Training curve analysis | Loss plots |
| `analyse_can_policies.py` | Statistical analysis of can task | Metrics & statistics |
| `analyse_cube_policies.py` | Statistical analysis of cube task | Metrics & statistics |
| `analyse_can_policies_lighting.py` | Lighting condition analysis | Robustness metrics |

## Data Files

- `can_policies.csv` - Results from can sorting experiments (Table I, Task 1)
- `cubes_policies.csv` - Results from cube grasping experiments (Table I, Task 2)
- `*_training.csv` - Training loss curves for different policies
- `lighting_test*.csv` - Lighting variation experiment data

## Requirements

```bash
pip install matplotlib numpy pandas seaborn scipy
```

## Citation

If you use these analysis scripts, please cite our paper:

```bibtex
@article{kuehn2026more,
  title={More Is Not Always Better: Active Stereo Camera Setup Outperforms Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation Task},
  author={K{\"u}hn, Robin and Bank, Dennis and Schappler, Moritz and Seel, Thomas},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```
