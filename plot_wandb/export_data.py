#!/usr/bin/env python3
"""
WandB data export utility.

This script helps export data from wandb binary files to CSV format
that can be easily loaded by the plotting system.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def export_wandb_run(run_dir: str, output_file: str = None) -> bool:
    """
    Export a single wandb run to CSV format.
    
    Args:
        run_dir: Path to wandb run directory
        output_file: Output CSV file path. If None, uses default naming.
        
    Returns:
        True if export successful, False otherwise
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        return False
    
    # Find the wandb file
    wandb_files = list(run_path.glob("*.wandb"))
    if not wandb_files:
        print(f"Error: No .wandb files found in {run_dir}")
        return False
    
    # Set output file if not provided
    if output_file is None:
        output_file = run_path / "exported_data.csv"
    
    try:
        # Use wandb export command
        cmd = [
            "wandb", "export", 
            str(run_path),
            "--format", "csv",
            "--output", str(output_file)
        ]
        
        print(f"Exporting {run_dir} to {output_file}...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_path))
        
        if result.returncode == 0:
            print(f"Successfully exported to {output_file}")
            return True
        else:
            print(f"Export failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: wandb command not found. Please install wandb: pip install wandb")
        return False
    except Exception as e:
        print(f"Error during export: {e}")
        return False


def export_all_runs(wandb_dir: str) -> dict:
    """
    Export all wandb runs in a directory.
    
    Args:
        wandb_dir: Base wandb directory
        
    Returns:
        Dictionary with export results
    """
    wandb_path = Path(wandb_dir)
    results = {"success": [], "failed": []}
    
    # Find all wandb files
    for wandb_file in wandb_path.rglob("*.wandb"):
        run_dir = wandb_file.parent
        output_file = run_dir / "exported_data.csv"
        
        if export_wandb_run(str(run_dir), str(output_file)):
            results["success"].append(str(run_dir))
        else:
            results["failed"].append(str(run_dir))
    
    return results


def try_manual_export(run_dir: str) -> bool:
    """
    Try to manually extract some basic information from wandb files.
    This is a fallback method when wandb export fails.
    """
    run_path = Path(run_dir)
    
    # Look for any logs or summary files
    log_files = []
    for pattern in ["*.log", "*.json", "*.jsonl", "wandb-summary.json", "wandb-metadata.json"]:
        log_files.extend(run_path.rglob(pattern))
    
    if log_files:
        print(f"Found potential data files in {run_dir}:")
        for log_file in log_files:
            print(f"  - {log_file}")
        return True
    
    return False


def main():
    """Main CLI for export utility."""
    parser = argparse.ArgumentParser(description="Export wandb data to CSV format")
    parser.add_argument("wandb_dir", help="Path to wandb directory or specific run")
    parser.add_argument("--output", "-o", help="Output file (for single run export)")
    parser.add_argument("--all", "-a", action="store_true", 
                       help="Export all runs in directory")
    
    args = parser.parse_args()
    
    wandb_path = Path(args.wandb_dir)
    
    if not wandb_path.exists():
        print(f"Error: {args.wandb_dir} does not exist")
        return 1
    
    if args.all or not any(wandb_path.glob("*.wandb")):
        # Export all runs in directory
        print(f"Exporting all runs in {args.wandb_dir}...")
        results = export_all_runs(args.wandb_dir)
        
        print(f"\nExport Summary:")
        print(f"Successful: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results["failed"]:
            print("\nFailed exports:")
            for failed_run in results["failed"]:
                print(f"  - {failed_run}")
                # Try manual export as fallback
                if try_manual_export(failed_run):
                    print(f"    ^ Found some data files that might be usable")
        
        return 0 if not results["failed"] else 1
        
    else:
        # Export single run
        success = export_wandb_run(args.wandb_dir, args.output)
        
        if not success:
            print("\nTrying to find alternative data sources...")
            try_manual_export(args.wandb_dir)
        
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
