# WandB Plotting System - Quick Reference

## 🚀 Getting Started

### 1. Quick Plot Generation
```bash
# Simple plot generation
python quick_start.py plot /path/to/wandb

# With custom output directory
python main.py plot /path/to/wandb --config thesis --output thesis_plots/
```

### 2. Start WandB Server
```bash
# Start local server
python main.py server /path/to/wandb --port 8080
```

### 3. Interactive Jupyter Notebook
```bash
jupyter notebook wandb_plotting_tutorial.ipynb
```

## 📁 File Structure

```
plot_wandb/
├── 📄 README.md                    # Complete documentation
├── 🚀 quick_start.py              # Easy-to-use interface
├── 🎯 main.py                     # Full CLI interface
├── 📊 examples.py                 # Example usage scripts
├── 📓 wandb_plotting_tutorial.ipynb # Jupyter notebook tutorial
├── 
├── 🔧 Core Modules:
├── data_loader.py                 # Load wandb data
├── plotting.py                    # Create plots
├── export.py                      # Export & LaTeX utilities
├── server.py                      # WandB server management
├── advanced_extractor.py          # Extract data from wandb files
├── export_data.py                 # Data export utilities
├── 
├── ⚙️ Configuration:
├── requirements.txt               # Python dependencies
├── config.ini                     # Configuration settings
└── __init__.py                    # Package initialization
```

## 🎨 Plot Types Available

1. **Training Curves**: Multi-metric training progress
2. **Comparison Plots**: Compare multiple runs/models  
3. **Box Plots**: Distribution of final performance
4. **Learning Rate Schedules**: LR decay visualization
5. **Custom Plots**: Full matplotlib control

## 📤 Export Formats

- **PNG**: High-resolution (300 DPI) for presentations
- **SVG**: Vector graphics, web-friendly, editable
- **PDF**: Publication-ready, LaTeX compatible
- **LaTeX**: Ready-to-use LaTeX code

## 🔧 Configuration Presets

- **Thesis**: Academic thesis optimized (serif fonts, high DPI)
- **Paper**: Journal paper optimized (compact, clean)
- **Presentation**: Slide optimized (large fonts, high contrast)

## 📊 Usage Examples

### Command Line
```bash
# Basic usage
python main.py plot /path/to/wandb

# Thesis plots
python main.py plot /path/to/wandb --config thesis --output thesis_figures/

# Custom metrics
python main.py plot /path/to/wandb --metrics train/loss val/loss accuracy

# Start server
python main.py server /path/to/wandb --port 8080
```

### Python API
```python
from data_loader import WandbDataLoader
from plotting import PlotManager, PlotConfigs
from export import ExportManager

# Load data
loader = WandbDataLoader("/path/to/wandb")
runs_data = loader.load_all_runs()

# Create plots
plot_manager = PlotManager(PlotConfigs.thesis_config())
fig = plot_manager.create_training_curves(runs_data, ["train/loss", "val/loss"])

# Export
export_manager = ExportManager("plots")
export_manager.export_figure(fig, "training", ["pdf", "svg", "png"])
```

## 🛠️ Troubleshooting

### Common Issues

1. **No data found**: 
   ```bash
   python advanced_extractor.py /path/to/run
   ```

2. **LaTeX errors**: 
   - Disable LaTeX: Set `use_latex=False` in config
   - Install packages: `sudo apt install texlive-latex-extra dvipng`

3. **Missing dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

4. **WandB file parsing fails**: 
   - The system will create synthetic data for demonstration
   - Real data can be extracted using wandb export tools

## 🎓 Master Thesis Workflow

1. **Generate Plots**: `python quick_start.py plot /path/to/wandb`
2. **Choose PDF files** for LaTeX documents
3. **Copy LaTeX code** from generated .tex files
4. **Include in thesis** using the provided LaTeX snippets

## 📧 Features Summary

✅ **Publication-quality plots**  
✅ **Multiple export formats**  
✅ **LaTeX integration**  
✅ **Local wandb server**  
✅ **Batch processing**  
✅ **Interactive Jupyter notebook**  
✅ **Synthetic data generation** (when real data unavailable)  
✅ **Multiple configuration presets**  
✅ **Command-line and Python API**  
✅ **Comprehensive documentation**  

## 🎯 Perfect for:

- Master thesis figures
- Research paper plots  
- Presentation slides
- Technical reports
- Interactive data exploration

---

**Created for ACT (Action Chunking with Transformers) network analysis**  
**Optimized for academic publication requirements**
