#!/usr/bin/env python

"""
Example usage script for dataset augmentation.

This script demonstrates how to use the argument_data.py script with
different configurations for various use cases.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Error:", e.stderr)
        return False


def example_basic_augmentation():
    """Example 1: Basic augmentation with default settings."""
    cmd = [
        "python", "unitree_lerobot/argument_data.py",
        "--input_dataset_path", "/media/robin/DATA/stack_cube_left",
        "--output_dataset_path", "/media/robin/DATA/basic_augmented_stack_cube_left"
    ]
    
    return run_command(cmd, "Basic Dataset Augmentation")


def example_episode_weighting():
    """Example 2: Episode weighting with optimal and recovery episodes."""
    cmd = [
        "python", "unitree_lerobot/argument_data.py",
        "--input_dataset_path", "/media/robin/DATA/stack_cube_left", 
        "--output_dataset_path", "/media/robin/DATA/weighted_augmented_stack_cube_left",
        "--optimal_episodes", "0,2,4,6",  # Episodes 0, 2, 4, 6 are optimal
        "--recovery_episodes", "1,3,5",   # Episodes 1, 3, 5 are recovery
        "--optimal_weight", "3.0",        # Weight optimal episodes 3x
        "--augmentation_probability", "0.8"  # Higher augmentation probability
    ]
    
    return run_command(cmd, "Episode Weighting Example")


def example_heavy_augmentation():
    """Example 3: Heavy augmentation for small datasets."""
    cmd = [
        "python", "unitree_lerobot/argument_data.py",
        "--input_dataset_path", "/media/robin/DATA/stack_cube_left",
        "--output_dataset_path", "/media/robin/DATA/heavy_augmented_stack_cube_left",
        "--optimal_episodes", "0,1,2",     # First few episodes are optimal
        "--optimal_weight", "5.0",         # Heavy weighting
        "--augmentation_probability", "1.0", # Augment all episodes
        "--joint_noise_std", "0.04",       # Higher joint noise
        "--brightness_range", "0.6", "1.4", # Wider brightness range
        "--enable_lighting_augmentation",  # Explicit enable lighting
        "--enable_joint_noise"             # Explicit enable joint noise
    ]
    
    return run_command(cmd, "Heavy Augmentation for Small Datasets")


def example_conservative_augmentation():
    """Example 4: Conservative augmentation for large datasets."""
    cmd = [
        "python", "unitree_lerobot/argument_data.py",
        "--input_dataset_path", "/media/robin/DATA/stack_cube_left",
        "--output_dataset_path", "/media/robin/DATA/conservative_augmented_stack_cube_left",
        "--optimal_weight", "1.5",         # Light weighting
        "--augmentation_probability", "0.3", # Lower augmentation probability
        "--joint_noise_std", "0.01",       # Lower joint noise
        "--brightness_range", "0.8", "1.2", # Narrower brightness range
        "--contrast_range", "0.9", "1.1"   # Narrower contrast range
    ]
    
    return run_command(cmd, "Conservative Augmentation for Large Datasets")


def example_dry_run():
    """Example 5: Dry run to test configuration without processing."""
    print("\n" + "="*60)
    print("Dry Run Example - Configuration Validation")
    print("="*60)
    
    # This would be a dry-run mode if implemented
    print("Note: Dry run mode is not implemented in the current script.")
    print("To validate your configuration, you can:")
    print("1. Run with a small test dataset first")
    print("2. Check the episode count in your input dataset:")
    
    cmd = ["ls", "-1d", "/media/robin/DATA/stack_cube_left/episode_*"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        episodes = result.stdout.strip().split('\n') if result.stdout.strip() else []
        print(f"   Found {len(episodes)} episodes in input dataset")
        if episodes:
            print("   Episodes:", [Path(ep).name for ep in episodes[:5]], "..." if len(episodes) > 5 else "")
    except Exception as e:
        print(f"   Error checking episodes: {e}")


def main():
    """Main function to run examples."""
    print("Dataset Augmentation Examples")
    print("=" * 60)
    
    examples = [
        ("1", "Basic Augmentation", example_basic_augmentation),
        ("2", "Episode Weighting", example_episode_weighting),
        ("3", "Heavy Augmentation", example_heavy_augmentation),
        ("4", "Conservative Augmentation", example_conservative_augmentation),
        ("5", "Dry Run / Validation", example_dry_run)
    ]
    
    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        for num, desc, func in examples:
            if num == example_num:
                print(f"Running Example {num}: {desc}")
                success = func()
                sys.exit(0 if success else 1)
        
        print(f"Error: Example {example_num} not found")
        sys.exit(1)
    
    else:
        # Interactive mode
        print("\nAvailable Examples:")
        for num, desc, _ in examples:
            print(f"  {num}. {desc}")
        
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]} <example_number>")
        print(f"  Example: python {sys.argv[0]} 1")
        print(f"\nOr edit this script to customize the examples for your specific use case.")


if __name__ == "__main__":
    main()
