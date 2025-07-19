#!/usr/bin/env python
"""
Feature Ablation Study Script

This script helps you run systematic ablation studies to understand
which features in your dataset are most important for your policy, and
how different ACT policy architectures affect performance.

Usage:
python run_ablation_study.py --config_file custom_ablation.yaml --dataset_repo your_dataset \
    --steps 50000 --eval_freq 10000 --save_freq 10000 --log_freq 1000 --batch_size 12

The script requires a custom YAML configuration file that defines the experiments to run.
Each experiment will be saved with a name matching the experiment name in the YAML config,
making it easy to identify which policy corresponds to which configuration.

You can configure both feature selection parameters AND ACT policy parameters in the YAML:

Feature Selection Parameters:
  - cameras, exclude_cameras: Control which cameras to use
  - use_joint_velocities, use_joint_torques: Include joint dynamics
  - use_pressure_sensors: Include pressure sensor data
  - joint_groups, exclude_joint_groups: Filter joint groups

ACT Policy Parameters:
  - vision_backbone: "resnet18", "resnet34", "dinov2_vits14", etc.
  - n_decoder_layers, n_encoder_layers: Transformer architecture
  - dim_model, n_heads, dim_feedforward: Model dimensions
  - chunk_size, n_action_steps: Action chunking configuration
  - use_vae, latent_dim, kl_weight: VAE settings
  - temporal_ensemble_coeff: Temporal ensembling
  - dropout, optimizer_lr: Training parameters

See ACT_CONFIG_GUIDE.md for detailed parameter explanations and examples.

Optional CLI parameters:
  --eval_freq: How often to run evaluation (default: 10000)
  --save_freq: How often to save checkpoints (default: 10000)  
  --log_freq: How often to log metrics (default: 1000)
  --batch_size: Training batch size (default: 12)
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
        
    def run_experiment(self, name: str, feature_overrides: Dict[str, Any], 
                      steps: int = 10000, eval_freq: int = 10000, save_freq: int = 10000, 
                      log_freq: int = 1000, batch_size: int = 12):
        """Run a single ablation experiment."""
        
        # Determine the correct path to train.py based on current working directory
        import os
        cwd = os.getcwd()
        if 'unitree_lerobot/scripts' in cwd:
            # Running from scripts directory
            train_script_path = "../lerobot/lerobot/scripts/train.py"
        else:
            # Running from root directory
            train_script_path = "./unitree_lerobot/lerobot/lerobot/scripts/train.py"
        
        # Build command with policy type and training parameters
        cmd = [
            "python", train_script_path,
            f"--dataset.repo_id={self.base_config['dataset_repo']}",
            "--policy.type=act",  # Add policy type
            f"--steps={steps}",
            f"--eval_freq={eval_freq}",
            f"--save_freq={save_freq}",
            f"--log_freq={log_freq}",
            f"--batch_size={batch_size}",
            f"--job_name={name}",  # This will be used in the output directory naming
            f"--wandb.project={self.base_config.get('wandb_project', 'ablation_study')}",
            f"--wandb.notes={name}",  # Use notes to identify the experiment
            "--wandb.enable=true"
        ]
        
        # Add base config overrides
        for key, value in self.base_config.get('base_overrides', {}).items():
            cmd.append(f"--{key}={value}")
            
        # Add feature selection overrides
        policy_params = []
        feature_params = []
        
        for key, value in feature_overrides.items():
            # Check if this is a policy parameter
            if key in ['vision_backbone', 'n_decoder_layers', 'n_encoder_layers', 'dim_model', 
                      'n_heads', 'dim_feedforward', 'chunk_size', 'n_action_steps', 'use_vae', 
                      'latent_dim', 'n_vae_encoder_layers', 'temporal_ensemble_coeff', 'dropout', 
                      'kl_weight', 'optimizer_lr', 'optimizer_weight_decay', 'optimizer_lr_backbone',
                      'pretrained_backbone_weights', 'pre_norm', 'feedforward_activation',
                      'replace_final_stride_with_dilation']:
                # This is a policy parameter
                if isinstance(value, bool):
                    value = str(value).lower()
                elif value is None:
                    value = "null"
                cmd.append(f"--policy.{key}={value}")
                policy_params.append(f"{key}={value}")
            else:
                # This is a feature selection parameter
                if isinstance(value, bool):
                    value = str(value).lower()
                cmd.append(f"--feature_selection.{key}={value}")
                feature_params.append(f"{key}={value}")
        
        # Log configuration details
        if policy_params:
            logger.info(f"Policy parameters: {', '.join(policy_params)}")
        if feature_params:
            logger.info(f"Feature parameters: {', '.join(feature_params)}")
            
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
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=10000, help='Save checkpoint frequency')
    parser.add_argument('--log_freq', type=int, default=1000, help='Logging frequency')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
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
        
        success = study.run_experiment(
            name, 
            config, 
            steps=args.steps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            batch_size=args.batch_size
        )
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
