# WandB Plotting System for Master Thesis

A comprehensive plotting system for creating publication-quality plots from local wandb logs. This system is specifically designed for academic use, particularly for master thesis work with ACT (Action Chunking with Transformers) networks.

## Features

- 📊 **Publication-Quality Plots**: Create professional plots optimized for academic papers and thesis
- 🎨 **Configurable Styling**: Multiple predefined configurations (paper, presentation, thesis)
- 📁 **Multiple Export Formats**: Export to PNG, SVG, PDF for different use cases
- 📄 **LaTeX Integration**: Generate ready-to-use LaTeX code for figure inclusion
- 🖥️ **Local WandB Server**: Start a local wandb server to browse logs in the web interface
- 🔄 **Batch Processing**: Process multiple experiments and runs simultaneously
- 📈 **Advanced Visualizations**: Training curves, comparisons, distributions, learning rate schedules

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For LaTeX rendering, install a LaTeX distribution:
```bash
# Ubuntu/Debian - Full installation
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Ubuntu/Debian - Minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra dvipng

# macOS
brew install --cask mactex

# Windows
# Install MiKTeX from https://miktex.org/
```

## Quick Start

### 1. Command Line Interface

The easiest way to get started is using the command line interface:

```bash
# Create thesis-quality plots from your wandb logs
python main.py plot /path/to/wandb --config thesis --output thesis_plots/

# Start a local wandb server to browse logs
python main.py server /path/to/wandb --port 8080

# List available runs
python main.py list /path/to/wandb
```

### 2. Python API

For more control, use the Python API:

```python
from data_loader import WandbDataLoader
from plotting import PlotManager, PlotConfigs
from export import ExportManager

# Load your data
loader = WandbDataLoader("/path/to/wandb")
runs_data = loader.load_all_runs()

# Create plots
plot_manager = PlotManager(PlotConfigs.thesis_config())
fig = plot_manager.create_training_curves(
    runs_data, 
    metrics=["train/loss", "val/loss", "train/accuracy"],
    smooth=True
)

# Export for thesis
export_manager = ExportManager("plots")
export_manager.export_figure(fig, "training_curves", ["pdf", "svg", "png"])

# Generate LaTeX code
latex_code = export_manager.generate_latex_figure(
    "training_curves.pdf",
    "Training curves showing model performance over training steps.",
    "training_curves"
)
```

### 3. Run Examples

See the provided examples:

```bash
python examples.py
```

## Usage Guide

### Directory Structure

Your wandb directory should look like this:
```
wandb/
├── resnet/
│   └── offline-run-20250806_123142-r4inlmse/
│       ├── run-r4inlmse.wandb
│       └── files/
└── dinov2/
    └── offline-run-20250807_145623-abc123de/
        ├── run-abc123de.wandb
        └── files/
```

### Plot Configurations

Three predefined configurations are available:

1. **Thesis Configuration** (`PlotConfigs.thesis_config()`):
   - Optimized for academic thesis
   - Serif fonts, LaTeX support
   - High DPI, multiple formats

2. **Paper Configuration** (`PlotConfigs.paper_config()`):
   - Optimized for journal papers
   - Compact size, clean styling
   - PDF/SVG export

3. **Presentation Configuration** (`PlotConfigs.presentation_config()`):
   - Optimized for presentations
   - Large fonts, high contrast colors
   - PNG export for slides

### Available Plot Types

1. **Training Curves**: Multi-metric training progress plots
2. **Comparison Plots**: Compare multiple runs/models
3. **Box Plots**: Distribution of final performance
4. **Learning Rate Schedules**: LR decay visualization

### Export Formats

- **PNG**: High-resolution raster (300 DPI) for presentations
- **SVG**: Vector graphics, web-friendly, editable text
- **PDF**: Vector graphics, publication-ready, LaTeX compatible

### LaTeX Integration

The system automatically generates LaTeX code for figure inclusion:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{training_curves.pdf}
    \caption{Training curves showing model performance over training steps.}
    \label{fig:training_curves}
\end{figure}
```

## Advanced Usage

### Custom Plot Configuration

Create your own plot configuration:

```python
from plotting import PlotConfig

custom_config = PlotConfig(
    figsize=(12, 8),
    style="seaborn-v0_8-whitegrid",
    colors=["#2E86AB", "#A23B72", "#F18F01"],
    font_family="serif",
    font_size=14,
    use_latex=True,
    export_formats=["pdf", "svg"]
)
```

### Batch Processing Multiple Experiments

```python
from data_loader import load_multiple_experiments

# Load all experiments
experiments = load_multiple_experiments("/path/to/wandb", ["resnet", "dinov2"])

# Create comparison across experiments
for exp_name, runs_data in experiments.items():
    # Process each experiment
    pass
```

### WandB Server Management

```python
from server import WandbServerManager

# Start server with context manager (auto-cleanup)
with WandbServerManager("/path/to/wandb", port=8080) as server:
    # Server runs while in this block
    print(f"Server running at {server.server_url}")
    # Do your analysis
    pass
# Server automatically stopped
```

## File Structure

```
plot_wandb/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point
├── data_loader.py           # WandB data loading utilities
├── plotting.py              # Plotting and visualization
├── server.py                # WandB server management
├── export.py                # Export and LaTeX utilities
├── examples.py              # Example usage scripts
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Common Use Cases

### 1. Master Thesis Plots

```bash
# Create all thesis plots with LaTeX code
python main.py plot /path/to/wandb --config thesis --output thesis_figures/
```

### 2. Compare Different Models

```bash
# Compare specific metrics across runs
python main.py plot /path/to/wandb --metrics train/loss val/loss accuracy --config paper
```

### 3. Browse Logs Interactively

```bash
# Start server and browse in web interface
python main.py server /path/to/wandb --port 8080
```

## Tips for Thesis Writing

1. **Use PDF exports** for LaTeX documents - they're vector graphics and scale perfectly
2. **Enable LaTeX rendering** for consistent math notation
3. **Use the thesis configuration** for optimal academic styling
4. **Include generated LaTeX code** directly in your thesis
5. **Export SVG** if you need to edit plots in vector graphics software

## Troubleshooting

### Common Issues

1. **"wandb not found"**: Install wandb with `pip install wandb`
2. **LaTeX errors**: Disable LaTeX in config or install LaTeX distribution
3. **No data found**: Make sure wandb directory contains valid .wandb files
4. **Port already in use**: The server will automatically find an available port

### Data Extraction

If you have issues loading data from .wandb files, you can export data manually:

```bash
# Export data from wandb files
wandb export --dir /path/to/run --format csv --output data.csv
```

## Contributing

This is a self-contained plotting system. To extend functionality:

1. Add new plot types to `plotting.py`
2. Add new export formats to `export.py`
3. Add new data sources to `data_loader.py`

## License

This project is licensed under the Apache 2.0 License - see the main project LICENSE file for details.

## Acknowledgments

- Built for ACT (Action Chunking with Transformers) network training analysis
- Designed for academic publication requirements
- Optimized for master thesis workflow
