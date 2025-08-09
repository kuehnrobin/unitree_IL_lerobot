"""
Data loading utilities for wandb logs.

This module provides functionality to extract and parse data from wandb binary files
and convert them into pandas DataFrames for analysis and plotting.
"""

import os
import json
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Some features may not work.")


class WandbDataLoader:
    """
    Load and parse wandb log data from local files.
    
    This class can extract metrics from wandb binary files and convert them
    to pandas DataFrames for analysis and plotting.
    """
    
    def __init__(self, wandb_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            wandb_dir: Path to the wandb directory containing run logs
        """
        self.wandb_dir = Path(wandb_dir)
        self.runs_data = {}
        
    def discover_runs(self) -> List[Dict[str, Any]]:
        """
        Discover all wandb runs in the directory.
        
        Returns:
            List of dictionaries containing run information
        """
        runs = []
        
        for run_dir in self.wandb_dir.rglob("offline-run-*"):
            if run_dir.is_dir():
                run_info = self._extract_run_info(run_dir)
                if run_info:
                    runs.append(run_info)
                    
        return runs
    
    def _extract_run_info(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Extract basic information about a run.
        
        Args:
            run_dir: Path to the run directory
            
        Returns:
            Dictionary with run information or None if extraction fails
        """
        try:
            # Look for the wandb file
            wandb_files = list(run_dir.glob("*.wandb"))
            if not wandb_files:
                return None
                
            wandb_file = wandb_files[0]
            run_id = wandb_file.stem.replace("run-", "")
            
            # Extract timestamp from directory name
            timestamp = run_dir.name.split("-")[-1]
            
            return {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "wandb_file": str(wandb_file),
                "timestamp": timestamp,
                "name": run_dir.parent.name  # e.g., 'resnet'
            }
        except Exception as e:
            warnings.warn(f"Failed to extract run info from {run_dir}: {e}")
            return None
    
    def load_run_data(self, run_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load data from a specific run.
        
        Args:
            run_info: Run information dictionary from discover_runs()
            
        Returns:
            DataFrame with the run's metrics or None if loading fails
        """
        # First try the alternative method (which includes advanced extraction)
        return self._load_run_data_alternative(run_info)
    
    def _load_from_wandb_file(self, wandb_file: str) -> Optional[pd.DataFrame]:
        """
        Load data directly from wandb binary file.
        
        Args:
            wandb_file: Path to the .wandb file
            
        Returns:
            DataFrame with metrics or None if loading fails
        """
        try:
            # This is a simplified approach - wandb files are complex binary formats
            # For production use, you might want to use wandb's internal APIs
            # or export data using wandb export command
            
            # Try to read as a simple binary file and extract JSON-like data
            with open(wandb_file, 'rb') as f:
                content = f.read()
                
            # This is a placeholder - actual wandb file parsing is complex
            # You might need to use wandb export or the wandb.restore() function
            print(f"Wandb file size: {len(content)} bytes")
            print("Note: Direct wandb file parsing is complex. Consider using 'wandb export' command.")
            
            return None
            
        except Exception as e:
            warnings.warn(f"Failed to parse wandb file {wandb_file}: {e}")
            return None
    
    def _load_run_data_alternative(self, run_dir: str) -> pd.DataFrame:
        """
        Alternative data loading method using wandb sync + API.
        This is the most reliable method for offline runs.
        """
        import subprocess
        import tempfile
        import shutil
        
        run_path = Path(run_dir)
        
        # Method 1: Try wandb sync to make offline runs accessible
        try:
            print(f"Attempting to sync offline run: {run_dir}")
            
            # Create a temporary project for syncing
            temp_project = f"temp-analysis-{int(pd.Timestamp.now().timestamp())}"
            
            # Run wandb sync on the directory
            sync_result = subprocess.run([
                'wandb', 'sync', 
                '--project', temp_project,
                '--no-include-online',  # Only offline runs
                '--mark-synced',        # Mark as synced
                str(run_path)
            ], capture_output=True, text=True, timeout=60)
            
            if sync_result.returncode == 0:
                print("Sync successful! Attempting to read via API...")
                
                # Now try to access via API
                try:
                    import wandb
                    api = wandb.Api()
                    
                    # Get runs from the temporary project
                    runs = api.runs(f"/{temp_project}")
                    
                    if runs:
                        # Get the most recent run (should be our synced run)
                        run = runs[0]
                        print(f"Found synced run: {run.id}")
                        
                        # Get the history
                        history_df = run.history()
                        
                        if not history_df.empty:
                            print(f"Successfully loaded data with shape: {history_df.shape}")
                            print(f"Columns: {list(history_df.columns)}")
                            
                            # Save the real data for future use
                            output_file = run_path / "extracted_real_data.csv"
                            history_df.to_csv(output_file, index=False)
                            print(f"Saved real data to: {output_file}")
                            
                            return history_df
                        
                except Exception as e:
                    print(f"API access after sync failed: {e}")
            else:
                print(f"Sync failed: {sync_result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Sync timed out")
        except Exception as e:
            print(f"Sync process failed: {e}")
        
        # Method 2: Try to read previously extracted real data
        try:
            extracted_file = run_path / "extracted_real_data.csv"
            if extracted_file.exists():
                print(f"Loading previously extracted real data from {extracted_file}")
                df = pd.read_csv(extracted_file)
                if not df.empty:
                    return df
        except Exception as e:
            print(f"Failed to read extracted data: {e}")
        
        # Method 3: Try different file types in order of preference
        data_files = [
            list(run_path.glob("*.csv")),
            list(run_path.glob("*.json")),
            list(run_path.glob("*.jsonl")),
            list(run_path.glob("*.log"))
        ]
        
        for file_list in data_files:
            if file_list:
                try:
                    file_path = file_list[0]
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            return df
                    elif file_path.suffix == '.json':
                        df = pd.read_json(file_path)
                        if not df.empty:
                            return df
                    elif file_path.suffix == '.jsonl':
                        df = pd.read_json(file_path, lines=True)
                        if not df.empty:
                            return df
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue
        
        # Method 4: If all else fails, use the advanced extractor
        print(f"Using advanced extractor for {run_dir}")
        from .advanced_extractor import extract_wandb_data
        return extract_wandb_data(run_dir)
    
    def _create_sample_data(self, run_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Create sample data for demonstration purposes.
        
        Args:
            run_info: Run information dictionary
            
        Returns:
            DataFrame with sample training metrics
        """
        import numpy as np
        
        # Create sample training data
        epochs = 100
        steps = np.arange(epochs)
        
        # Simulate realistic training curves
        train_loss = 2.0 * np.exp(-steps / 30) + 0.1 + 0.05 * np.random.randn(epochs)
        val_loss = 2.2 * np.exp(-steps / 35) + 0.15 + 0.08 * np.random.randn(epochs)
        accuracy = 1 - np.exp(-steps / 25) * 0.8 + 0.02 * np.random.randn(epochs)
        learning_rate = 5e-5 * np.exp(-steps / 50)
        
        data = {
            'step': steps,
            'epoch': steps,
            'train/loss': train_loss,
            'val/loss': val_loss, 
            'train/accuracy': accuracy,
            'val/accuracy': accuracy + 0.02 * np.random.randn(epochs),
            'train/learning_rate': learning_rate,
            'run_id': run_info["run_id"],
            'run_name': run_info["name"]
        }
        
        df = pd.DataFrame(data)
        print(f"Created sample data for run {run_info['run_id']} (this is demo data)")
        return df
    
    def load_all_runs(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all discovered runs.
        
        Returns:
            Dictionary mapping run_id to DataFrame
        """
        runs = self.discover_runs()
        all_data = {}
        
        for run_info in runs:
            data = self.load_run_data(run_info)
            if data is not None:
                all_data[run_info["run_id"]] = data
                
        return all_data
    
    def export_run_data(self, run_info: Dict[str, Any], output_file: str):
        """
        Export run data using wandb export command.
        
        Args:
            run_info: Run information dictionary
            output_file: Output file path for exported data
        """
        run_dir = Path(run_info["run_dir"])
        
        # Use wandb export command
        import subprocess
        
        try:
            cmd = f"wandb export --dir {run_dir} --format csv --output {output_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully exported data to {output_file}")
            else:
                print(f"Export failed: {result.stderr}")
                
        except Exception as e:
            print(f"Failed to run export command: {e}")


def load_multiple_experiments(wandb_base_dir: Union[str, Path], 
                            experiment_names: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load data from multiple experiments.
    
    Args:
        wandb_base_dir: Base wandb directory containing multiple experiment folders
        experiment_names: List of experiment names to load. If None, load all.
        
    Returns:
        Nested dictionary: {experiment_name: {run_id: DataFrame}}
    """
    base_dir = Path(wandb_base_dir)
    all_experiments = {}
    
    # Discover experiment directories
    if experiment_names is None:
        experiment_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    else:
        experiment_dirs = [base_dir / name for name in experiment_names if (base_dir / name).exists()]
    
    for exp_dir in experiment_dirs:
        loader = WandbDataLoader(exp_dir)
        exp_data = loader.load_all_runs()
        if exp_data:
            all_experiments[exp_dir.name] = exp_data
            
    return all_experiments
