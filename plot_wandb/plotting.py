"""
Plotting utilities for creating publication-quality plots from wandb data.

This module provides a comprehensive plotting system with customizable styles,
multiple plot types, and export capabilities for academic publications.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import seaborn as sns


@dataclass
class PlotConfig:
    """Configuration class for plot styling and behavior."""
    
    # Figure settings
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 300
    
    # Style settings
    style: str = "seaborn-v0_8-whitegrid"  # Updated seaborn style name
    palette: str = "deep"
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    
    # Colors
    colors: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Grid and spines
    grid: bool = True
    grid_alpha: float = 0.3
    spine_width: float = 1.0
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg", "pdf"])
    export_dpi: int = 300
    bbox_inches: str = "tight"
    
    # Academic publication settings
    use_latex: bool = False
    latex_preamble: List[str] = field(default_factory=lambda: [
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}"
    ])


class PlotManager:
    """
    Main plotting manager for creating publication-quality plots.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the plot manager.
        
        Args:
            config: Plot configuration. If None, uses default config.
        """
        self.config = config or PlotConfig()
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """Setup matplotlib with the specified configuration."""
        # Set style
        try:
            plt.style.use(self.config.style)
        except OSError:
            # Fallback to default style if requested style is not available
            plt.style.use("default")
            print(f"Style '{self.config.style}' not available, using default")
        
        # Set up LaTeX if requested
        if self.config.use_latex:
            plt.rcParams['text.usetex'] = True
            plt.rcParams['text.latex.preamble'] = self.config.latex_preamble
            
        # Font settings
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['axes.titlesize'] = self.config.title_size
        plt.rcParams['axes.labelsize'] = self.config.label_size
        plt.rcParams['xtick.labelsize'] = self.config.tick_size
        plt.rcParams['ytick.labelsize'] = self.config.tick_size
        plt.rcParams['legend.fontsize'] = self.config.legend_size
        
        # Set DPI
        plt.rcParams['figure.dpi'] = self.config.dpi
        
        # Set color palette
        sns.set_palette(self.config.palette)
        
    def create_training_curves(self, 
                             data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                             metrics: List[str],
                             title: str = "Training Curves",
                             x_axis: str = "step",
                             smooth: bool = True,
                             smoothing_window: int = 10) -> plt.Figure:
        """
        Create training curve plots.
        
        Args:
            data: DataFrame or dict of DataFrames with training data
            metrics: List of metric names to plot
            title: Plot title
            x_axis: Column name for x-axis (e.g., 'step', 'epoch')
            smooth: Whether to apply smoothing
            smoothing_window: Window size for smoothing
            
        Returns:
            matplotlib Figure object
        """
        if isinstance(data, dict):
            return self._create_multi_run_curves(data, metrics, title, x_axis, smooth, smoothing_window)
        else:
            return self._create_single_run_curves(data, metrics, title, x_axis, smooth, smoothing_window)
    
    def _create_single_run_curves(self, 
                                data: pd.DataFrame,
                                metrics: List[str],
                                title: str,
                                x_axis: str,
                                smooth: bool,
                                smoothing_window: int) -> plt.Figure:
        """Create training curves for a single run."""
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.config.figsize[0] * n_cols, 
                                                         self.config.figsize[1] * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]
            
            if metric not in data.columns:
                ax.text(0.5, 0.5, f"Metric '{metric}'\nnot found", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            x = data[x_axis]
            y = data[metric]
            
            if smooth and len(y) > smoothing_window:
                y_smooth = self._apply_smoothing(y, smoothing_window)
                ax.plot(x, y_smooth, label=f"{metric} (smoothed)", 
                       color=self.config.colors[i % len(self.config.colors)], linewidth=2)
                ax.plot(x, y, alpha=0.3, 
                       color=self.config.colors[i % len(self.config.colors)], linewidth=1)
            else:
                ax.plot(x, y, label=metric, 
                       color=self.config.colors[i % len(self.config.colors)], linewidth=2)
            
            ax.set_xlabel(x_axis.capitalize())
            ax.set_ylabel(metric)
            ax.grid(self.config.grid, alpha=self.config.grid_alpha)
            ax.legend()
            
        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        return fig
    
    def _create_multi_run_curves(self, 
                               data: Dict[str, pd.DataFrame],
                               metrics: List[str],
                               title: str,
                               x_axis: str,
                               smooth: bool,
                               smoothing_window: int) -> plt.Figure:
        """Create training curves comparing multiple runs."""
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.config.figsize[0] * n_cols, 
                                                         self.config.figsize[1] * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten() if n_metrics > 1 else [axes]
            
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]
            
            for j, (run_name, df) in enumerate(data.items()):
                if metric not in df.columns:
                    continue
                    
                x = df[x_axis]
                y = df[metric]
                color = self.config.colors[j % len(self.config.colors)]
                
                if smooth and len(y) > smoothing_window:
                    y_smooth = self._apply_smoothing(y, smoothing_window)
                    ax.plot(x, y_smooth, label=f"{run_name}", color=color, linewidth=2)
                    ax.plot(x, y, alpha=0.2, color=color, linewidth=1)
                else:
                    ax.plot(x, y, label=f"{run_name}", color=color, linewidth=2)
            
            ax.set_xlabel(x_axis.capitalize())
            ax.set_ylabel(metric)
            ax.grid(self.config.grid, alpha=self.config.grid_alpha)
            ax.legend()
            
        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, 
                             data: Dict[str, pd.DataFrame],
                             metric: str,
                             title: str = "Model Comparison",
                             x_axis: str = "step",
                             final_values: bool = True) -> plt.Figure:
        """
        Create a comparison plot for multiple runs.
        
        Args:
            data: Dictionary of DataFrames for different runs
            metric: Metric to compare
            title: Plot title
            x_axis: Column name for x-axis
            final_values: Whether to show final values in legend
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for i, (run_name, df) in enumerate(data.items()):
            if metric not in df.columns:
                continue
                
            x = df[x_axis]
            y = df[metric]
            color = self.config.colors[i % len(self.config.colors)]
            
            label = run_name
            if final_values and len(y) > 0:
                final_val = y.iloc[-1]
                label += f" (final: {final_val:.4f})"
                
            ax.plot(x, y, label=label, color=color, linewidth=2)
        
        ax.set_xlabel(x_axis.capitalize())
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_box_plot(self, 
                       data: Dict[str, pd.DataFrame],
                       metric: str,
                       title: str = "Performance Distribution",
                       last_n: int = 10) -> plt.Figure:
        """
        Create box plot showing distribution of final performance.
        
        Args:
            data: Dictionary of DataFrames for different runs
            metric: Metric to plot
            title: Plot title
            last_n: Number of final values to use for distribution
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        plot_data = []
        labels = []
        
        for run_name, df in data.items():
            if metric not in df.columns:
                continue
                
            # Get last N values
            final_values = df[metric].tail(last_n).values
            plot_data.append(final_values)
            labels.append(run_name)
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(self.config.colors[i % len(self.config.colors)])
                patch.set_alpha(0.7)
        
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(self.config.grid, alpha=self.config.grid_alpha, axis='y')
        
        plt.tight_layout()
        return fig
    
    def _apply_smoothing(self, y: pd.Series, window: int) -> pd.Series:
        """Apply smoothing to a time series."""
        return y.rolling(window=window, center=True).mean().fillna(y)
    
    def save_figure(self, fig: plt.Figure, filename: str, output_dir: str = "plots"):
        """
        Save figure in multiple formats.
        
        Args:
            fig: matplotlib Figure to save
            filename: Base filename (without extension)
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for fmt in self.config.export_formats:
            filepath = output_path / f"{filename}.{fmt}"
            fig.savefig(filepath, 
                       format=fmt,
                       dpi=self.config.export_dpi,
                       bbox_inches=self.config.bbox_inches)
            print(f"Saved plot to {filepath}")
    
    def create_learning_rate_schedule(self, 
                                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                    lr_column: str = "train/learning_rate",
                                    title: str = "Learning Rate Schedule") -> plt.Figure:
        """
        Create learning rate schedule plot.
        
        Args:
            data: DataFrame or dict of DataFrames with learning rate data
            lr_column: Column name for learning rate
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        if isinstance(data, dict):
            for i, (run_name, df) in enumerate(data.items()):
                if lr_column not in df.columns:
                    continue
                ax.plot(df.index, df[lr_column], 
                       label=run_name, 
                       color=self.config.colors[i % len(self.config.colors)])
        else:
            if lr_column in data.columns:
                ax.plot(data.index, data[lr_column], 
                       color=self.config.colors[0])
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)
        
        if isinstance(data, dict):
            ax.legend()
            
        plt.tight_layout()
        return fig


# Predefined plot configurations for different use cases
class PlotConfigs:
    """Predefined plot configurations for different scenarios."""
    
    @staticmethod
    def paper_config() -> PlotConfig:
        """Configuration optimized for academic papers."""
        return PlotConfig(
            figsize=(8, 5),
            font_family="serif",
            font_size=11,
            title_size=12,
            use_latex=True,
            export_formats=["pdf", "svg"],
            style="classic"
        )
    
    @staticmethod
    def presentation_config() -> PlotConfig:
        """Configuration optimized for presentations."""
        return PlotConfig(
            figsize=(12, 7),
            font_size=14,
            title_size=16,
            label_size=14,
            export_formats=["png", "svg"],
            colors=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        )
    
    @staticmethod
    def thesis_config() -> PlotConfig:
        """Configuration optimized for master thesis."""
        return PlotConfig(
            figsize=(10, 6),
            font_family="serif",
            font_size=12,
            title_size=14,
            use_latex=True,
            export_formats=["pdf", "svg", "png"],
            export_dpi=300,
            style="seaborn-v0_8-whitegrid"
        )
