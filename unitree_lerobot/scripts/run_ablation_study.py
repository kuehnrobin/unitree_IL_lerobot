#!/usr/bin/env python
"""
Feature Ablation Study Script

This script helps you run systematic ablation studies to understand
which features in your dataset are most important for your policy.

Usage examples:

1. Test different camera configurations:
python run_ablation_study.py --study_type cameras --dataset_repo your_dataset

2. Test different state features:
python run_ablation_study.py --study_type state_features --dataset_repo your_dataset

3. Custom ablation study:
python run_ablation_study.py --config_file custom_ablation.yaml --dataset_repo your_dataset
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


def get_camera_ablation_configs() -> List[Dict[str, Any]]:
    """Get configurations for camera ablation study."""
    return [
        {
            "name": "all_cameras_baseline",
            "config": {}  # Use all cameras
        },
        {
            "name": "single_camera_high",
            "config": {"cameras": '["cam_high"]'}
        },
        {
            "name": "single_camera_low", 
            "config": {"cameras": '["cam_low"]'}
        },
        {
            "name": "stereo_cameras_only",
            "config": {"cameras": '["cam_stereo_left", "cam_stereo_right"]'}
        },
        {
            "name": "wrist_cameras_only",
            "config": {"cameras": '["cam_left_wrist", "cam_right_wrist"]'}
        },
        {
            "name": "no_stereo_cameras",
            "config": {"exclude_cameras": '["cam_stereo_left", "cam_stereo_right"]'}
        },
        {
            "name": "minimal_cameras",
            "config": {"cameras": '["cam_high", "cam_left_wrist"]'}
        }
    ]


def get_state_feature_ablation_configs() -> List[Dict[str, Any]]:
    """Get configurations for state feature ablation study."""
    return [
        {
            "name": "all_features_baseline",
            "config": {}  # Use all features
        },
        {
            "name": "positions_only",
            "config": {
                "use_joint_velocities": "false",
                "use_joint_torques": "false", 
                "use_pressure_sensors": "false"
            }
        },
        {
            "name": "positions_and_velocities",
            "config": {
                "use_joint_torques": "false",
                "use_pressure_sensors": "false"
            }
        },
        {
            "name": "no_pressure_sensors",
            "config": {"use_pressure_sensors": "false"}
        },
        {
            "name": "no_camera_joints",
            "config": {"exclude_joint_groups": '["camera"]'}
        },
        {
            "name": "arms_only",
            "config": {"joint_groups": '["left_arm", "right_arm"]'}
        },
        {
            "name": "hands_only", 
            "config": {"joint_groups": '["left_hand", "right_hand"]'}
        },
        {
            "name": "left_side_only",
            "config": {"joint_groups": '["left_arm", "left_hand"]'}
        }
    ]


def load_custom_config(config_file: str) -> List[Dict[str, Any]]:
    """Load custom ablation configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('experiments', [])


def main():
    parser = argparse.ArgumentParser(description='Run feature ablation studies')
    parser.add_argument('--study_type', choices=['cameras', 'state_features', 'custom'], 
                       help='Type of ablation study to run')
    parser.add_argument('--config_file', help='Custom config file for ablation study')
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
    
    # Get experiment configurations
    if args.study_type == 'cameras':
        experiments = get_camera_ablation_configs()
    elif args.study_type == 'state_features':
        experiments = get_state_feature_ablation_configs()
    elif args.study_type == 'custom':
        if not args.config_file:
            raise ValueError("--config_file required for custom study type")
        experiments = load_custom_config(args.config_file)
    else:
        raise ValueError("Must specify --study_type or --config_file")
    
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
