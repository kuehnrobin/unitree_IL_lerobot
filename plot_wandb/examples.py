#!/usr/bin/env python3
"""
Example script showing how to create custom plots for your thesis.

This script demonstrates various ways to use the plotting system to create
publication-quality plots from your wandb logs.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import WandbDataLoader, load_multiple_experiments
from plotting import PlotManager, PlotConfig, PlotConfigs
from export import ExportManager
import matplotlib.pyplot as plt


def example_basic_plots():
    """Example: Create basic training plots."""
    print("=== Basic Training Plots Example ===")
    
    # Path to your wandb directory
    wandb_dir = "../wandb"  # Adjust this path
    
    # Initialize components
    config = PlotConfigs.thesis_config()
    plot_manager = PlotManager(config)
    export_manager = ExportManager("example_plots")
    
    # Load data
    loader = WandbDataLoader(wandb_dir)
    runs_data = loader.load_all_runs()
    
    if not runs_data:
        print("No data found. Using sample data for demonstration.")
        # Create sample data for demonstration
        import pandas as pd
        import numpy as np
        
        steps = np.arange(100)
        sample_data = {
            "sample_run": pd.DataFrame({
                "step": steps,
                "train/loss": 2.0 * np.exp(-steps / 30) + 0.1 + 0.05 * np.random.randn(100),
                "val/loss": 2.2 * np.exp(-steps / 35) + 0.15 + 0.08 * np.random.randn(100),
                "train/accuracy": 1 - np.exp(-steps / 25) * 0.8 + 0.02 * np.random.randn(100),
            })
        }
        runs_data = sample_data
    
    # Create training curves
    metrics = ["train/loss", "val/loss", "train/accuracy"]
    
    if len(runs_data) > 1:
        # Multiple runs comparison
        fig = plot_manager.create_training_curves(
            runs_data, metrics,
            title="ACT Training Comparison",
            smooth=True
        )
    else:
        # Single run
        fig = plot_manager.create_training_curves(
            list(runs_data.values())[0], metrics,
            title="ACT Training Progress",
            smooth=True
        )
    
    # Export the plot
    export_manager.export_figure(fig, "act_training_curves", ["png", "svg", "pdf"])
    
    # Generate LaTeX code
    latex_code = export_manager.generate_latex_figure(
        "act_training_curves.pdf",
        "Training curves for the ACT network showing loss and accuracy over training steps.",
        "act_training"
    )
    export_manager.save_latex_code(latex_code, "act_training_figure")
    
    print("Basic plots created successfully!")


def example_custom_styling():
    """Example: Create plots with custom styling."""
    print("\n=== Custom Styling Example ===")
    
    # Create custom configuration
    custom_config = PlotConfig(
        figsize=(12, 8),
        style="seaborn-v0_8-whitegrid",
        colors=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#5D4E75"],
        font_family="serif",
        font_size=14,
        use_latex=False,  # Set to True if you have LaTeX installed
        export_formats=["png", "pdf", "svg"]
    )
    
    plot_manager = PlotManager(custom_config)
    export_manager = ExportManager("custom_plots")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    steps = np.arange(150)
    
    # Simulate different model configurations
    models_data = {
        "ResNet18": pd.DataFrame({
            "step": steps,
            "train/loss": 1.8 * np.exp(-steps / 25) + 0.08 + 0.03 * np.random.randn(150),
            "val/loss": 2.0 * np.exp(-steps / 30) + 0.12 + 0.05 * np.random.randn(150),
        }),
        "DINOv2": pd.DataFrame({
            "step": steps,
            "train/loss": 1.5 * np.exp(-steps / 35) + 0.05 + 0.02 * np.random.randn(150),
            "val/loss": 1.7 * np.exp(-steps / 40) + 0.08 + 0.04 * np.random.randn(150),
        }),
        "ResNet50": pd.DataFrame({
            "step": steps,
            "train/loss": 2.2 * np.exp(-steps / 20) + 0.12 + 0.04 * np.random.randn(150),
            "val/loss": 2.5 * np.exp(-steps / 25) + 0.18 + 0.06 * np.random.randn(150),
        })
    }
    
    # Create comparison plot
    fig = plot_manager.create_comparison_plot(
        models_data, "val/loss",
        title="Validation Loss Comparison: Different Vision Backbones",
        final_values=True
    )
    
    export_manager.export_figure(fig, "backbone_comparison", ["png", "pdf"])
    
    # Create box plot
    fig2 = plot_manager.create_box_plot(
        models_data, "val/loss",
        title="Final Performance Distribution",
        last_n=20
    )
    
    export_manager.export_figure(fig2, "performance_distribution", ["png", "pdf"])
    
    print("Custom styled plots created successfully!")


def example_thesis_package():
    """Example: Create a complete thesis plot package."""
    print("\n=== Thesis Package Example ===")
    
    config = PlotConfigs.thesis_config()
    plot_manager = PlotManager(config)
    export_manager = ExportManager("thesis_plots")
    
    # Create sample experimental data
    import pandas as pd
    import numpy as np
    
    steps = np.arange(200)
    
    # Simulate ablation study results
    experiments = {
        "Baseline": {
            "train/loss": 2.5 * np.exp(-steps / 40) + 0.15 + 0.05 * np.random.randn(200),
            "val/loss": 2.8 * np.exp(-steps / 45) + 0.20 + 0.07 * np.random.randn(200),
            "train/accuracy": 1 - 1.2 * np.exp(-steps / 35) + 0.03 * np.random.randn(200),
        },
        "With Data Aug": {
            "train/loss": 2.2 * np.exp(-steps / 35) + 0.12 + 0.04 * np.random.randn(200),
            "val/loss": 2.3 * np.exp(-steps / 40) + 0.15 + 0.05 * np.random.randn(200),
            "train/accuracy": 1 - 1.0 * np.exp(-steps / 30) + 0.02 * np.random.randn(200),
        },
        "Full Pipeline": {
            "train/loss": 1.8 * np.exp(-steps / 30) + 0.08 + 0.03 * np.random.randn(200),
            "val/loss": 1.9 * np.exp(-steps / 35) + 0.10 + 0.04 * np.random.randn(200),
            "train/accuracy": 1 - 0.8 * np.exp(-steps / 25) + 0.015 * np.random.randn(200),
        }
    }
    
    # Convert to DataFrames
    experiments_data = {}
    for name, data in experiments.items():
        df = pd.DataFrame(data)
        df["step"] = steps
        experiments_data[name] = df
    
    # Create multiple figures
    figures = []
    names = []
    captions = []
    labels = []
    
    # 1. Training curves comparison
    fig1 = plot_manager.create_training_curves(
        experiments_data, ["train/loss", "val/loss"],
        title="Training and Validation Loss Comparison",
        smooth=True
    )
    figures.append(fig1)
    names.append("loss_comparison")
    captions.append("Comparison of training and validation loss across different configurations")
    labels.append("loss_comp")
    
    # 2. Accuracy comparison
    fig2 = plot_manager.create_comparison_plot(
        experiments_data, "train/accuracy",
        title="Training Accuracy Comparison"
    )
    figures.append(fig2)
    names.append("accuracy_comparison")
    captions.append("Training accuracy progression for different model configurations")
    labels.append("acc_comp")
    
    # 3. Final performance box plot
    fig3 = plot_manager.create_box_plot(
        experiments_data, "val/loss",
        title="Final Validation Loss Distribution"
    )
    figures.append(fig3)
    names.append("final_performance")
    captions.append("Distribution of final validation loss values across configurations")
    labels.append("final_perf")
    
    # Create complete thesis package
    exported_files = export_manager.create_thesis_plot_package(
        figures, names, captions, labels,
        main_title="ACT Network Ablation Study Results"
    )
    
    print("Thesis package created successfully!")
    print("Files created:")
    for format_type, files in exported_files.items():
        print(f"  {format_type.upper()}: {len(files)} files")


def example_wandb_server():
    """Example: Start a wandb server."""
    print("\n=== WandB Server Example ===")
    
    from server import WandbServerManager
    
    wandb_dir = "../wandb"  # Adjust this path
    
    # Create server manager
    server = WandbServerManager(wandb_dir, port=8080)
    
    print("Starting wandb server...")
    print("Note: This will start a server and open your browser.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        success = server.start_server(open_browser=True)
        if success:
            print("Server started successfully!")
            print(f"Access dashboard at: {server.server_url}")
            
            # Keep server running for demonstration
            import time
            time.sleep(5)  # Run for 5 seconds for demo
            
        server.stop_server()
        print("Server stopped.")
        
    except Exception as e:
        print(f"Server example failed: {e}")


def main():
    """Run all examples."""
    print("WandB Plotting Examples for Thesis")
    print("=" * 50)
    
    # Create output directory
    Path("example_plots").mkdir(exist_ok=True)
    Path("custom_plots").mkdir(exist_ok=True)
    Path("thesis_plots").mkdir(exist_ok=True)
    
    try:
        # Run examples
        example_basic_plots()
        example_custom_styling()
        example_thesis_package()
        
        # Uncomment to test server functionality
        # example_wandb_server()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated plot directories for outputs.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
