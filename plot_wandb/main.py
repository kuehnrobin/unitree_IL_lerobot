#!/usr/bin/env python3
"""
Main CLI script for wandb plotting utilities.

This script provides a command-line interface for creating publication-quality plots
from local wandb logs and managing a local wandb server.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import WandbDataLoader, load_multiple_experiments
from plotting import PlotManager, PlotConfig, PlotConfigs
from server import WandbServerManager, quick_start_server
from export import ExportManager


def create_training_plots(wandb_dirs: Union[str, List[str]], 
                         output_dir: str = "plots",
                         config_type: str = "thesis",
                         metrics: Optional[List[str]] = None,
                         smooth: bool = True,
                         run_names: Optional[List[str]] = None) -> None:
    """
    Create training plots from wandb logs.
    
    Args:
        wandb_dirs: Path(s) to wandb directory/directories
        output_dir: Output directory for plots
        config_type: Plot configuration type ('paper', 'presentation', 'thesis')
        metrics: List of metrics to plot. If None, uses default metrics.
        smooth: Whether to apply smoothing to curves
        run_names: Optional custom names for the runs
    """
    # Ensure wandb_dirs is a list
    if isinstance(wandb_dirs, str):
        wandb_dirs = [wandb_dirs]
    
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
    
    # Load data from all directories
    print(f"Loading data from {len(wandb_dirs)} directories...")
    all_runs_data = {}
    
    for i, wandb_dir in enumerate(wandb_dirs):
        print(f"Loading from {wandb_dir}...")
        loader = WandbDataLoader(wandb_dir)
        runs_data = loader.load_all_runs()
        
        if runs_data:
            # Add custom names if provided
            for run_id, df in runs_data.items():
                if run_names and i < len(run_names):
                    combined_name = f"{run_names[i]}_{run_id}"
                else:
                    # Use directory name as prefix
                    dir_name = Path(wandb_dir).name
                    combined_name = f"{dir_name}_{run_id}"
                all_runs_data[combined_name] = df
            print(f"Found {len(runs_data)} runs in {wandb_dir}")
        else:
            print(f"No data found in {wandb_dir}")
    
    if not all_runs_data:
        print("No data found in any directory. Make sure the wandb directories contain valid runs.")
        return
    
    print(f"Total: {len(all_runs_data)} runs loaded")
    
    # Default metrics if none specified (focusing on L1 loss)
    if metrics is None:
        metrics = [
            "train/l1_loss",
            "val/l1_loss",
            "train/loss", 
            "val/loss",
            "train/accuracy",
            "val/accuracy"
        ]
    
    # Filter metrics that exist in the data
    available_metrics = set()
    for df in all_runs_data.values():
        available_metrics.update(df.columns)
    
    metrics = [m for m in metrics if m in available_metrics]
    print(f"Available metrics: {sorted(list(available_metrics))}")
    print(f"Plotting metrics: {metrics}")
    
    if not metrics:
        print("No valid metrics found to plot.")
        return
    
    # Create plots
    figures = []
    figure_names = []
    captions = []
    labels = []
    
    # Training curves comparison (always create this for multiple runs)
    fig = plot_manager.create_training_curves(
        all_runs_data, metrics, 
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
            all_runs_data, metric,
            title=f"{metric.replace('/', ' ').title()} Comparison"
        )
        figures.append(fig)
        safe_metric_name = metric.replace('/', '_').replace(' ', '_')
        figure_names.append(f"{safe_metric_name}_comparison")
        captions.append(f"Comparison of {metric} across different configurations")
        labels.append(f"{safe_metric_name}_comp")
    
    # Box plot for final performance (key metrics only)
    key_metrics = [m for m in metrics if any(key in m.lower() for key in ['l1_loss', 'loss', 'accuracy'])]
    for metric in key_metrics:
        fig = plot_manager.create_box_plot(
            all_runs_data, metric,
            title=f"Final {metric.replace('/', ' ').title()} Distribution"
        )
        figures.append(fig)
        safe_metric_name = metric.replace('/', '_').replace(' ', '_')
        figure_names.append(f"{safe_metric_name}_boxplot")
        captions.append(f"Distribution of final {metric} values")
        labels.append(f"{safe_metric_name}_dist")
    
    # Special focus on L1 loss if available
    l1_metrics = [m for m in available_metrics if 'l1' in m.lower() and 'loss' in m.lower()]
    if l1_metrics:
        print(f"Creating special L1 loss plots for: {l1_metrics}")
        for l1_metric in l1_metrics:
            # Create dedicated L1 loss comparison
            fig = plot_manager.create_comparison_plot(
                all_runs_data, l1_metric,
                title="L1 Loss Comparison Across Runs",
                final_values=True
            )
            figures.append(fig)
            figure_names.append("l1_loss_detailed_comparison")
            captions.append("Detailed comparison of L1 loss across different model configurations showing convergence behavior")
            labels.append("l1_loss_detailed")
    
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
  # Create thesis plots from single wandb log directory
  python main.py plot /path/to/wandb --config thesis --output plots/

  # Compare multiple runs from different directories
  python main.py plot /path/to/wandb1 /path/to/wandb2 --config thesis --output comparison_plots/

  # Custom run names and specific metrics
  python main.py plot /path/to/run1 /path/to/run2 --run-names "ResNet" "DINOv2" --metrics train/l1_loss val/l1_loss

  # Start wandb server
  python main.py server /path/to/wandb --port 8080

  # List available runs
  python main.py list /path/to/wandb
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Create plots from wandb logs")
    plot_parser.add_argument("wandb_dirs", nargs="+", 
                           help="Path(s) to wandb directory/directories")
    plot_parser.add_argument("--output", "-o", default="plots", 
                           help="Output directory for plots (default: plots)")
    plot_parser.add_argument("--config", "-c", choices=["paper", "presentation", "thesis"],
                           default="thesis", help="Plot configuration type")
    plot_parser.add_argument("--metrics", "-m", nargs="+", 
                           help="Metrics to plot (default: common training metrics)")
    plot_parser.add_argument("--no-smooth", action="store_true",
                           help="Disable curve smoothing")
    plot_parser.add_argument("--run-names", "-n", nargs="+",
                           help="Custom names for the runs (optional)")
    
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
    
    # Validate wandb directories
    for wandb_dir in (args.wandb_dirs if args.command == "plot" else [args.wandb_dir]):
        wandb_path = Path(wandb_dir)
        if not wandb_path.exists():
            print(f"Error: Directory {wandb_dir} does not exist")
            return
    
    # Execute command
    try:
        if args.command == "plot":
            create_training_plots(
                args.wandb_dirs,
                args.output,
                args.config,
                args.metrics,
                not args.no_smooth,
                args.run_names
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
