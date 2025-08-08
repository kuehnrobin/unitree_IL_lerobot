#!/usr/bin/env python3
"""
Main CLI script for wandb plotting utilities.

This script provides a command-line interface for creating publication-quality plots
from local wandb logs and managing a local wandb server.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import WandbDataLoader, load_multiple_experiments
from plotting import PlotManager, PlotConfig, PlotConfigs
from server import WandbServerManager, quick_start_server
from export import ExportManager


def create_training_plots(wandb_dir: str, 
                         output_dir: str = "plots",
                         config_type: str = "thesis",
                         metrics: Optional[List[str]] = None,
                         smooth: bool = True) -> None:
    """
    Create training plots from wandb logs.
    
    Args:
        wandb_dir: Path to wandb directory
        output_dir: Output directory for plots
        config_type: Plot configuration type ('paper', 'presentation', 'thesis')
        metrics: List of metrics to plot. If None, uses default metrics.
        smooth: Whether to apply smoothing to curves
    """
    # Set up configuration
    if config_type == "paper":
        config = PlotConfigs.paper_config()
    elif config_type == "presentation":
        config = PlotConfigs.presentation_config()
    else:
        config = PlotConfigs.thesis_config()
    
    # Initialize managers
    plot_manager = PlotManager(config)
    export_manager = ExportManager(output_dir)
    
    # Load data
    print(f"Loading data from {wandb_dir}...")
    loader = WandbDataLoader(wandb_dir)
    runs_data = loader.load_all_runs()
    
    if not runs_data:
        print("No data found. Make sure the wandb directory contains valid runs.")
        return
    
    print(f"Found {len(runs_data)} runs")
    
    # Default metrics if none specified
    if metrics is None:
        metrics = [
            "train/loss",
            "val/loss", 
            "train/accuracy",
            "val/accuracy"
        ]
    
    # Filter metrics that exist in the data
    available_metrics = set()
    for df in runs_data.values():
        available_metrics.update(df.columns)
    
    metrics = [m for m in metrics if m in available_metrics]
    print(f"Available metrics: {list(available_metrics)}")
    print(f"Plotting metrics: {metrics}")
    
    if not metrics:
        print("No valid metrics found to plot.")
        return
    
    # Create plots
    figures = []
    figure_names = []
    captions = []
    labels = []
    
    # Training curves
    if len(runs_data) > 1:
        # Multiple runs comparison
        fig = plot_manager.create_training_curves(
            runs_data, metrics, 
            title="Training Curves Comparison",
            smooth=smooth
        )
        figures.append(fig)
        figure_names.append("training_curves_comparison")
        captions.append("Comparison of training curves across different model configurations")
        labels.append("training_comparison")
        
        # Individual metric comparisons
        for metric in metrics:
            fig = plot_manager.create_comparison_plot(
                runs_data, metric,
                title=f"{metric.replace('/', ' ').title()} Comparison"
            )
            figures.append(fig)
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            figure_names.append(f"{safe_metric_name}_comparison")
            captions.append(f"Comparison of {metric} across different configurations")
            labels.append(f"{safe_metric_name}_comp")
        
        # Box plot for final performance
        for metric in metrics:
            fig = plot_manager.create_box_plot(
                runs_data, metric,
                title=f"Final {metric.replace('/', ' ').title()} Distribution"
            )
            figures.append(fig)
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            figure_names.append(f"{safe_metric_name}_boxplot")
            captions.append(f"Distribution of final {metric} values")
            labels.append(f"{safe_metric_name}_dist")
    
    else:
        # Single run
        run_data = list(runs_data.values())[0]
        fig = plot_manager.create_training_curves(
            run_data, metrics,
            title="Training Curves",
            smooth=smooth
        )
        figures.append(fig)
        figure_names.append("training_curves")
        captions.append("Training curves showing model performance over time")
        labels.append("training_curves")
    
    # Learning rate schedule if available
    lr_metrics = [col for col in available_metrics if 'learning_rate' in col.lower()]
    if lr_metrics:
        fig = plot_manager.create_learning_rate_schedule(
            runs_data if len(runs_data) > 1 else list(runs_data.values())[0],
            lr_column=lr_metrics[0]
        )
        figures.append(fig)
        figure_names.append("learning_rate_schedule")
        captions.append("Learning rate schedule during training")
        labels.append("lr_schedule")
    
    # Export all plots
    print(f"Exporting {len(figures)} plots...")
    exported_files = export_manager.create_thesis_plot_package(
        figures, figure_names, captions, labels,
        main_title="ACT Network Training Results"
    )
    
    print(f"\nExported plots to {output_dir}/")
    print("Files created:")
    for format_type, files in exported_files.items():
        print(f"  {format_type.upper()}: {len(files)} files")


def start_server_command(wandb_dir: str, port: int = 8080, no_browser: bool = False) -> None:
    """
    Start a local wandb server.
    
    Args:
        wandb_dir: Path to wandb directory
        port: Port to run server on
        no_browser: Don't open browser automatically
    """
    print(f"Starting wandb server for {wandb_dir} on port {port}")
    
    server = quick_start_server(wandb_dir, port, not no_browser)
    
    try:
        print("Server is running. Press Ctrl+C to stop.")
        print(f"Access the dashboard at: http://localhost:{port}")
        
        # Keep the server running
        import time
        while True:
            time.sleep(1)
            if server.process.poll() is not None:
                print("Server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop_server()


def list_runs_command(wandb_dir: str) -> None:
    """
    List all available runs in the wandb directory.
    
    Args:
        wandb_dir: Path to wandb directory
    """
    loader = WandbDataLoader(wandb_dir)
    runs = loader.discover_runs()
    
    if not runs:
        print("No wandb runs found.")
        return
    
    print(f"Found {len(runs)} wandb runs:\n")
    
    for run in runs:
        print(f"Run ID: {run['run_id']}")
        print(f"  Name: {run['name']}")
        print(f"  Timestamp: {run['timestamp']}")
        print(f"  Directory: {run['run_dir']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WandB plotting utilities for thesis/publication plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create thesis plots from wandb logs
  python main.py plot /path/to/wandb --config thesis --output plots/

  # Start wandb server
  python main.py server /path/to/wandb --port 8080

  # List available runs
  python main.py list /path/to/wandb

  # Create presentation plots with custom metrics
  python main.py plot /path/to/wandb --config presentation --metrics train/loss val/loss accuracy
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Create plots from wandb logs")
    plot_parser.add_argument("wandb_dir", help="Path to wandb directory")
    plot_parser.add_argument("--output", "-o", default="plots", 
                           help="Output directory for plots (default: plots)")
    plot_parser.add_argument("--config", "-c", choices=["paper", "presentation", "thesis"],
                           default="thesis", help="Plot configuration type")
    plot_parser.add_argument("--metrics", "-m", nargs="+", 
                           help="Metrics to plot (default: common training metrics)")
    plot_parser.add_argument("--no-smooth", action="store_true",
                           help="Disable curve smoothing")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start local wandb server")
    server_parser.add_argument("wandb_dir", help="Path to wandb directory")
    server_parser.add_argument("--port", "-p", type=int, default=8080,
                             help="Port to run server on (default: 8080)")
    server_parser.add_argument("--no-browser", action="store_true",
                             help="Don't open browser automatically")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available wandb runs")
    list_parser.add_argument("wandb_dir", help="Path to wandb directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate wandb directory
    wandb_path = Path(args.wandb_dir)
    if not wandb_path.exists():
        print(f"Error: Directory {args.wandb_dir} does not exist")
        return
    
    # Execute command
    try:
        if args.command == "plot":
            create_training_plots(
                args.wandb_dir,
                args.output,
                args.config,
                args.metrics,
                not args.no_smooth
            )
        elif args.command == "server":
            start_server_command(args.wandb_dir, args.port, args.no_browser)
        elif args.command == "list":
            list_runs_command(args.wandb_dir)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
