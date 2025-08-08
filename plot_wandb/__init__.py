"""
WandB Plotting Package for Master Thesis

This package provides tools for creating publication-quality plots from local wandb logs.
It includes both a matplotlib-based plotting system and utilities to start a local wandb server.
"""

__version__ = "0.1.0"
__author__ = "Robin Kuehn"

from .data_loader import WandbDataLoader
from .plotting import PlotManager, PlotConfig
from .server import WandbServerManager
from .export import ExportManager

__all__ = [
    "WandbDataLoader",
    "PlotManager", 
    "PlotConfig",
    "WandbServerManager",
    "ExportManager"
]
