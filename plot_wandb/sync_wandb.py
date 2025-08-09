#!/usr/bin/env python3
"""
Wandb Sync Script

This script synchronizes local wandb runs to a wandb server using `wandb sync`.
It can sync all runs from the wandb directory to a specified project.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import os


def find_wandb_runs(wandb_base_dir: str) -> List[Path]:
    """
    Find all wandb run directories in the base wandb directory.
    
    Args:
        wandb_base_dir: Base wandb directory path
        
    Returns:
        List of paths to wandb run directories
    """
    base_path = Path(wandb_base_dir)
    
    if not base_path.exists():
        print(f"Error: Wandb directory {wandb_base_dir} does not exist")
        return []
    
    # Find all run directories (they typically contain .wandb files)
    run_dirs = []
    
    # Look for directories containing .wandb files
    for item in base_path.rglob("*.wandb"):
        run_dir = item.parent
        if run_dir not in run_dirs:
            run_dirs.append(run_dir)
    
    # Also look for typical wandb run directory patterns
    for pattern in ["offline-run-*", "run-*"]:
        for item in base_path.rglob(pattern):
            if item.is_dir() and item not in run_dirs:
                # Check if it has wandb files
                if list(item.glob("*.wandb")):
                    run_dirs.append(item)
    
    return sorted(run_dirs)


def get_run_info(run_dir: Path) -> dict:
    """
    Extract basic information about a wandb run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with run information
    """
    info = {
        'path': str(run_dir),
        'name': run_dir.name,
        'wandb_files': len(list(run_dir.glob("*.wandb"))),
        'size_mb': sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / 1024 / 1024
    }
    
    # Try to find run metadata
    meta_files = list(run_dir.glob("wandb-metadata.json"))
    if meta_files:
        try:
            with open(meta_files[0], 'r') as f:
                metadata = json.load(f)
                info['metadata'] = metadata
        except:
            pass
    
    return info


def sync_run(run_dir: Path, project: str, entity: Optional[str] = None, 
             dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Sync a single wandb run to the server.
    
    Args:
        run_dir: Path to run directory
        project: Project name
        entity: Entity name (optional)
        dry_run: If True, only show what would be synced
        verbose: Show detailed output
        
    Returns:
        True if sync was successful
    """
    if dry_run:
        print(f"[DRY RUN] Would sync: {run_dir} -> project: {project}")
        return True
    
    # Build sync command
    cmd = ['wandb', 'sync']
    
    # Add project
    cmd.extend(['--project', project])
    
    # Add entity if specified
    if entity:
        cmd.extend(['--entity', entity])
    
    # Add options for offline runs
    cmd.extend([
        '--no-include-online',  # Only sync offline runs
        '--mark-synced',        # Mark runs as synced after upload
        '--include-offline'     # Include offline runs
    ])
    
    # Add the run directory
    cmd.append(str(run_dir))
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the sync command
        result = subprocess.run(
            cmd, 
            capture_output=not verbose, 
            text=True, 
            timeout=300  # 5 minute timeout per run
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully synced: {run_dir.name}")
            return True
        else:
            print(f"✗ Failed to sync {run_dir.name}")
            if not verbose and result.stderr:
                print(f"  Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout syncing {run_dir.name}")
        return False
    except Exception as e:
        print(f"✗ Error syncing {run_dir.name}: {e}")
        return False


def main():
    """Main script function."""
    parser = argparse.ArgumentParser(
        description="Sync local wandb runs to wandb server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all runs to project 'my-training'
  python sync_wandb.py ../wandb --project my-training
  
  # Sync with specific entity
  python sync_wandb.py ../wandb --project my-training --entity my-team
  
  # Dry run to see what would be synced
  python sync_wandb.py ../wandb --project my-training --dry-run
  
  # Sync specific runs only
  python sync_wandb.py ../wandb/resnet/offline-run-* --project resnet-experiments
  
  # Verbose output
  python sync_wandb.py ../wandb --project my-training --verbose
        """
    )
    
    parser.add_argument(
        'wandb_paths',
        nargs='+',
        help='Path(s) to wandb directory or specific run directories'
    )
    
    parser.add_argument(
        '--project', '-p',
        required=True,
        help='Project name for wandb (required)'
    )
    
    parser.add_argument(
        '--entity', '-e',
        help='Entity name for wandb (optional)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be synced without actually syncing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List all found runs and exit'
    )
    
    parser.add_argument(
        '--filter',
        help='Filter runs by name pattern (e.g., "resnet*" or "*20250806*")'
    )
    
    args = parser.parse_args()
    
    # Collect all run directories
    all_runs = []
    
    for wandb_path in args.wandb_paths:
        path = Path(wandb_path)
        
        if not path.exists():
            print(f"Warning: Path {wandb_path} does not exist")
            continue
        
        if path.is_file() and path.suffix == '.wandb':
            # Single wandb file - use its directory
            all_runs.append(path.parent)
        elif path.is_dir():
            if list(path.glob("*.wandb")):
                # Directory with wandb files - use directly
                all_runs.append(path)
            else:
                # Directory that might contain run subdirectories
                found_runs = find_wandb_runs(str(path))
                all_runs.extend(found_runs)
        else:
            print(f"Warning: {wandb_path} is not a valid wandb directory or file")
    
    # Remove duplicates and sort
    all_runs = sorted(list(set(all_runs)))
    
    # Apply filter if specified
    if args.filter:
        import fnmatch
        filtered_runs = []
        for run in all_runs:
            if fnmatch.fnmatch(run.name, args.filter):
                filtered_runs.append(run)
        all_runs = filtered_runs
    
    if not all_runs:
        print("No wandb runs found in the specified paths")
        return
    
    print(f"Found {len(all_runs)} wandb run(s)")
    
    # List runs if requested
    if args.list_runs:
        print("\nFound runs:")
        for i, run_dir in enumerate(all_runs, 1):
            info = get_run_info(run_dir)
            print(f"{i:2d}. {run_dir.name}")
            print(f"     Path: {run_dir}")
            print(f"     Files: {info['wandb_files']} .wandb files")
            print(f"     Size: {info['size_mb']:.1f} MB")
        return
    
    # Show what will be synced
    print(f"\nSyncing to project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual syncing will occur]")
    
    print(f"\nRuns to sync:")
    for i, run_dir in enumerate(all_runs, 1):
        info = get_run_info(run_dir)
        print(f"{i:2d}. {run_dir.name} ({info['size_mb']:.1f} MB)")
    
    # Confirm if not dry run
    if not args.dry_run:
        response = input(f"\nProceed with syncing {len(all_runs)} run(s)? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled")
            return
    
    # Sync all runs
    print(f"\nStarting sync...")
    successful = 0
    failed = 0
    
    for i, run_dir in enumerate(all_runs, 1):
        print(f"\n[{i}/{len(all_runs)}] Syncing {run_dir.name}...")
        
        if sync_run(run_dir, args.project, args.entity, args.dry_run, args.verbose):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Sync Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(all_runs)}")
    
    if not args.dry_run:
        print(f"\nAll runs synced to project: {args.project}")
        if args.entity:
            print(f"You can view them at: https://wandb.ai/{args.entity}/{args.project}")
        else:
            print(f"You can view them in your wandb dashboard under project: {args.project}")


if __name__ == "__main__":
    main()
