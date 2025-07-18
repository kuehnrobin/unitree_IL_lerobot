#!/usr/bin/env python
"""
Feature Ablation Study Script

This script helps you run systematic ablation studies to understand
which features in your dataset are most important for your policy.

Usage:
python run_ablation_study.py --config_file custom_ablation.yaml --dataset_repo your_dataset

The script requires a custom YAML configuration file that defines the experiments to run.
"""

import argparse
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """Manages ablation study experiments."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        
    def run_experiment(self, name: str, feature_overrides: Dict[str, Any], steps: int = 10000):
        """Run a single ablation experiment."""
        
        # Build command
        cmd = [
            "python", "lerobot/scripts/train.py",
            f"--dataset.repo_id={self.base_config['dataset_repo']}",
            f"--steps={steps}",
            f"--job_name={name}",
            f"--wandb.project={self.base_config.get('wandb_project', 'ablation_study')}",
            "--wandb.enable=true"
        ]
        
        # Add base config overrides
        for key, value in self.base_config.get('base_overrides', {}).items():
            cmd.append(f"--{key}={value}")
            
        # Add feature selection overrides
        for key, value in feature_overrides.items():
            # Convert boolean values to strings for command line
            if isinstance(value, bool):
                value = str(value).lower()
            cmd.append(f"--feature_selection.{key}={value}")
            
        logger.info(f"Running experiment: {name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run the experiment
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Experiment {name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {name} failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False





def load_custom_config(config_file: str) -> List[Dict[str, Any]]:
    """Load custom ablation configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('experiments', [])


def main():
    parser = argparse.ArgumentParser(description='Run feature ablation studies')
    parser.add_argument('--config_file', required=True, help='Custom config file for ablation study (YAML)')
    parser.add_argument('--dataset_repo', required=True, help='Dataset repository ID')
    parser.add_argument('--wandb_project', default='ablation_study', help='WandB project name')
    parser.add_argument('--steps', type=int, default=10000, help='Training steps per experiment')
    parser.add_argument('--base_config', help='Base training configuration overrides (YAML)')
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = {
        'dataset_repo': args.dataset_repo,
        'wandb_project': args.wandb_project,
        'base_overrides': {}
    }
    
    if args.base_config:
        with open(args.base_config, 'r') as f:
            base_config['base_overrides'] = yaml.safe_load(f)
    
    # Load custom experiment configurations
    experiments = load_custom_config(args.config_file)
    
    # Run ablation study
    study = AblationStudy(base_config)
    
    results = {}
    for exp in experiments:
        name = exp['name']
        config = exp['config']
        
        success = study.run_experiment(name, config, args.steps)
        results[name] = success
        
    # Print summary
    logger.info("\n=== Ablation Study Results ===")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{name}: {status}")
        
    successful_runs = sum(results.values())
    total_runs = len(results)
    logger.info(f"\nCompleted {successful_runs}/{total_runs} experiments successfully")


if __name__ == "__main__":
    main()
