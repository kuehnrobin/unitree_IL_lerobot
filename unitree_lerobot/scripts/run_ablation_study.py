#!/usr/bin/env python
"""
Comprehensive Ablation Study Script

This script runs systematic experiments to understand:
1. Feature importance (cameras, state features, joint groups)
2. ACT parameter sensitivity (architecture, training parameters)
3. Combined feature + architecture optimization

Usage examples:

1. Test different camera configurations:
python run_ablation_study.py --study_type cameras --dataset_repo your_dataset

2. Test different state features:
python run_ablation_study.py --study_type state_features --dataset_repo your_dataset

3. Test ACT architecture parameters:
python run_ablation_study.py --study_type act_architecture --dataset_repo your_dataset

4. Test ACT hyperparameters:
python run_ablation_study.py --study_type act_hyperparameters --dataset_repo your_dataset

5. Combined optimization:
python run_ablation_study.py --study_type combined --dataset_repo your_dataset

6. Custom ablation study:
python run_ablation_study.py --study_type custom --config_file custom_ablation.yaml --dataset_repo your_dataset
"""

import argparse
import json
import os
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """Runs systematic ablation studies for LeRobot training."""
    
    def __init__(
        self,
        dataset_repo: str,
        wandb_project: str,
        base_steps: int = 15000,
        base_config_overrides: Optional[Dict] = None
    ):
        self.dataset_repo = dataset_repo
        self.wandb_project = wandb_project
        self.base_steps = base_steps
        self.base_config_overrides = base_config_overrides or {}
        
        # Base training command template
        self.base_cmd = [
            "python", "lerobot/scripts/train.py",
            f"--dataset.repo_id={dataset_repo}",
            f"--wandb.project={wandb_project}",
            f"--steps={base_steps}",
            "--wandb.enable=true",
            "--policy.type=act"
        ]
        
        # Results tracking
        self.completed_runs = []
        self.failed_runs = []

    def run_camera_ablation(self) -> List[Dict]:
        """Test different camera configurations."""
        logger.info("🎥 Starting camera ablation study...")
        
        experiments = [
            # Baseline
            {
                "name": "baseline_all_cameras",
                "description": "All 6 cameras (baseline)",
                "args": []
            },
            
            # Single camera tests
            {
                "name": "single_cam_high",
                "description": "Only high overview camera",
                "args": ["--feature_selection.cameras=['cam_high']"]
            },
            {
                "name": "single_cam_low", 
                "description": "Only low overview camera",
                "args": ["--feature_selection.cameras=['cam_low']"]
            },
            {
                "name": "single_cam_left_wrist",
                "description": "Only left wrist camera", 
                "args": ["--feature_selection.cameras=['cam_left_wrist']"]
            },
            {
                "name": "single_cam_right_wrist",
                "description": "Only right wrist camera",
                "args": ["--feature_selection.cameras=['cam_right_wrist']"]
            },
            
            # Stereo cameras
            {
                "name": "stereo_head_only",
                "description": "Only stereo head cameras (color_4, color_5)",
                "args": ["--feature_selection.cameras=['cam_stereo_left','cam_stereo_right']"]
            },
            {
                "name": "stereo_left_only",
                "description": "Only left stereo camera",
                "args": ["--feature_selection.cameras=['cam_stereo_left']"]
            },
            
            # Functional groupings
            {
                "name": "overview_cameras",
                "description": "Overview cameras only",
                "args": ["--feature_selection.cameras=['cam_high','cam_low']"]
            },
            {
                "name": "wrist_cameras", 
                "description": "Wrist cameras only",
                "args": ["--feature_selection.cameras=['cam_left_wrist','cam_right_wrist']"]
            },
            {
                "name": "no_stereo",
                "description": "All except stereo cameras",
                "args": ["--feature_selection.exclude_cameras=['cam_stereo_left','cam_stereo_right']"]
            },
            
            # Practical combinations
            {
                "name": "high_plus_wrists",
                "description": "High camera + both wrist cameras",
                "args": ["--feature_selection.cameras=['cam_high','cam_left_wrist','cam_right_wrist']"]
            },
            {
                "name": "minimal_2cam",
                "description": "Minimal 2-camera setup",
                "args": ["--feature_selection.cameras=['cam_high','cam_left_wrist']"]
            }
        ]
        
        return self._run_experiments(experiments, "camera_ablation")

    def run_state_feature_ablation(self) -> List[Dict]:
        """Test different state feature combinations."""
        logger.info("🔧 Starting state feature ablation study...")
        
        experiments = [
            # Baseline
            {
                "name": "baseline_all_features",
                "description": "All state features (baseline)", 
                "args": []
            },
            
            # Single feature types
            {
                "name": "positions_only",
                "description": "Joint positions only",
                "args": [
                    "--feature_selection.use_joint_velocities=false",
                    "--feature_selection.use_joint_torques=false", 
                    "--feature_selection.use_pressure_sensors=false"
                ]
            },
            {
                "name": "pos_vel_only",
                "description": "Positions + velocities only",
                "args": [
                    "--feature_selection.use_joint_torques=false",
                    "--feature_selection.use_pressure_sensors=false"
                ]
            },
            {
                "name": "no_pressure",
                "description": "No pressure sensors",
                "args": ["--feature_selection.use_pressure_sensors=false"]
            },
            {
                "name": "no_torques",
                "description": "No torque feedback",
                "args": ["--feature_selection.use_joint_torques=false"]
            },
            {
                "name": "no_velocities",
                "description": "No velocity feedback", 
                "args": ["--feature_selection.use_joint_velocities=false"]
            },
            
            # Body part filtering
            {
                "name": "arms_only",
                "description": "Arms only (no hands/camera)",
                "args": ["--feature_selection.joint_groups=['left_arm','right_arm']"]
            },
            {
                "name": "no_camera_movement",
                "description": "No camera joint movement",
                "args": ["--feature_selection.exclude_joint_groups=['camera']"]
            },
            {
                "name": "left_side_only",
                "description": "Left arm + hand only",
                "args": ["--feature_selection.joint_groups=['left_arm','left_hand']"]
            },
        ]
        
        return self._run_experiments(experiments, "state_feature_ablation")

    def run_act_architecture_ablation(self) -> List[Dict]:
        """Test different ACT architecture configurations."""
        logger.info("🏗️ Starting ACT architecture ablation study...")
        
        experiments = [
            # Baseline configurations
            {
                "name": "act_default",
                "description": "Default ACT config",
                "args": []
            },
            {
                "name": "act_small_model", 
                "description": "Small transformer model",
                "args": [
                    "--policy.dim_model=256",
                    "--policy.n_heads=4",
                    "--policy.dim_feedforward=1024"
                ]
            },
            {
                "name": "act_large_model",
                "description": "Large transformer model",
                "args": [
                    "--policy.dim_model=768",
                    "--policy.n_heads=12",
                    "--policy.dim_feedforward=3072"
                ]
            },
            {
                "name": "act_short_horizon",
                "description": "Short horizon chunks",
                "args": [
                    "--policy.chunk_size=30",
                    "--policy.n_action_steps=6"
                ]
            },
            {
                "name": "act_long_horizon",
                "description": "Long horizon chunks",
                "args": [
                    "--policy.chunk_size=120",
                    "--policy.n_action_steps=24"
                ]
            },
            {
                "name": "act_no_vae",
                "description": "Without VAE",
                "args": ["--policy.use_vae=false"]
            },
            {
                "name": "act_large_vae",
                "description": "Large VAE latent dimension",
                "args": [
                    "--policy.latent_dim=64",
                    "--policy.kl_weight=15.0"
                ]
            },
            {
                "name": "act_resnet34",
                "description": "ResNet34 vision backbone",
                "args": ["--policy.vision_backbone=resnet34"]
            }
        ]
        
        return self._run_experiments(experiments, "act_architecture_ablation")

    def run_act_hyperparameter_ablation(self) -> List[Dict]:
        """Test different ACT hyperparameter settings."""
        logger.info("⚙️ Starting ACT hyperparameter ablation study...")
        
        experiments = [
            # Chunk size variations
            {
                "name": "chunk_15",
                "description": "Very short chunks (15 steps)",
                "args": ["--policy.chunk_size=15", "--policy.n_action_steps=3"]
            },
            {
                "name": "chunk_30",
                "description": "Short chunks (30 steps)",
                "args": ["--policy.chunk_size=30", "--policy.n_action_steps=6"]
            },
            {
                "name": "chunk_90",
                "description": "Medium-long chunks (90 steps)", 
                "args": ["--policy.chunk_size=90", "--policy.n_action_steps=18"]
            },
            {
                "name": "chunk_180",
                "description": "Very long chunks (180 steps)",
                "args": ["--policy.chunk_size=180", "--policy.n_action_steps=36"]
            },
            
            # Learning rate variations
            {
                "name": "lr_high",
                "description": "High learning rate",
                "args": ["--optimizer.lr=5e-5"]
            },
            {
                "name": "lr_low",
                "description": "Low learning rate",
                "args": ["--optimizer.lr=5e-6"]
            },
            
            # Temporal ensembling variations
            {
                "name": "light_temporal_ensemble",
                "description": "Light temporal ensembling",
                "args": [
                    "--policy.temporal_ensemble_coeff=0.001",
                    "--policy.n_action_steps=1"
                ]
            },
            {
                "name": "strong_temporal_ensemble",
                "description": "Strong temporal ensembling",
                "args": [
                    "--policy.temporal_ensemble_coeff=0.01",
                    "--policy.n_action_steps=1"
                ]
            },
            
            # Dropout variations
            {
                "name": "high_dropout",
                "description": "High dropout for regularization",
                "args": ["--policy.dropout=0.2"]
            },
            {
                "name": "no_dropout",
                "description": "No dropout",
                "args": ["--policy.dropout=0.0"]
            }
        ]
        
        return self._run_experiments(experiments, "act_hyperparameter_ablation")

    def run_combined_ablation(self) -> List[Dict]:
        """Test combinations of promising features + ACT configs."""
        logger.info("🔄 Starting combined feature + ACT ablation study...")
        
        experiments = [
            # Minimal setups for fast iteration
            {
                "name": "minimal_fast",
                "description": "Single camera + short chunks for rapid prototyping",
                "args": [
                    "--feature_selection.cameras=['cam_high']",
                    "--feature_selection.use_joint_velocities=false",
                    "--feature_selection.use_pressure_sensors=false",
                    "--policy.chunk_size=15",
                    "--policy.n_action_steps=3",
                    "--policy.dim_model=256"
                ]
            },
            {
                "name": "minimal_robust",
                "description": "Single camera + robust ACT config",
                "args": [
                    "--feature_selection.cameras=['cam_high']",
                    "--feature_selection.use_joint_torques=false",
                    "--policy.chunk_size=30",
                    "--policy.n_action_steps=1",
                    "--policy.temporal_ensemble_coeff=0.005",
                    "--policy.dropout=0.15"
                ]
            },
            
            # Wrist camera specialization
            {
                "name": "wrist_cams_short_horizon",
                "description": "Wrist cameras for short-horizon manipulation",
                "args": [
                    "--feature_selection.cameras=['cam_left_wrist','cam_right_wrist']",
                    "--policy.chunk_size=30",
                    "--policy.n_action_steps=6",
                    "--policy.temporal_ensemble_coeff=0.01"
                ]
            },
            {
                "name": "wrist_cams_medium_horizon",
                "description": "Wrist cameras for medium-horizon tasks",
                "args": [
                    "--feature_selection.cameras=['cam_left_wrist','cam_right_wrist']",
                    "--policy.chunk_size=60",
                    "--policy.n_action_steps=12",
                    "--policy.latent_dim=32"
                ]
            },
            
            # Stereo vision experiments
            {
                "name": "stereo_3d_awareness",
                "description": "Stereo cameras with larger model for 3D understanding",
                "args": [
                    "--feature_selection.cameras=['cam_stereo_left','cam_stereo_right']",
                    "--policy.dim_model=768",
                    "--policy.n_heads=12",
                    "--policy.chunk_size=60"
                ]
            },
            
            # Multi-camera with optimized ACT
            {
                "name": "multi_cam_optimized",
                "description": "Multiple cameras with optimized ACT parameters",
                "args": [
                    "--feature_selection.cameras=['cam_high','cam_left_wrist','cam_right_wrist']",
                    "--policy.chunk_size=60",
                    "--policy.n_action_steps=10",
                    "--policy.dim_model=512",
                    "--policy.temporal_ensemble_coeff=0.005"
                ]
            },
            
            # Specialized configurations
            {
                "name": "manipulation_focused",
                "description": "Optimized for fine manipulation tasks",
                "args": [
                    "--feature_selection.cameras=['cam_left_wrist','cam_right_wrist','cam_stereo_left']",
                    "--feature_selection.exclude_joint_groups=['camera']",
                    "--policy.chunk_size=45",
                    "--policy.n_action_steps=1",
                    "--policy.temporal_ensemble_coeff=0.01",
                    "--policy.dim_model=512"
                ]
            },
            {
                "name": "navigation_focused",
                "description": "Optimized for navigation tasks",
                "args": [
                    "--feature_selection.cameras=['cam_high','cam_low']",
                    "--feature_selection.joint_groups=['left_arm','right_arm','camera']",
                    "--policy.chunk_size=90",
                    "--policy.n_action_steps=18",
                    "--policy.dim_model=384"
                ]
            }
        ]
        
        return self._run_experiments(experiments, "combined_ablation")

    def run_custom_ablation(self, config_file: str) -> List[Dict]:
        """Run custom ablation study from YAML config."""
        logger.info(f"📝 Starting custom ablation study from {config_file}...")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        experiments = []
        for exp_config in config.get('experiments', []):
            args = []
            
            # Convert config to command line arguments
            for key, value in exp_config.get('params', {}).items():
                if isinstance(value, bool):
                    args.append(f"--{key}={str(value).lower()}")
                elif isinstance(value, list):
                    # Handle list arguments (like camera lists)
                    list_str = str(value).replace("'", '"')
                    args.append(f"--{key}={list_str}")
                else:
                    args.append(f"--{key}={value}")
            
            experiments.append({
                "name": exp_config.get('name'),
                "description": exp_config.get('description', ''),
                "args": args
            })
        
        return self._run_experiments(experiments, "custom_ablation")

    def _run_experiments(self, experiments: List[Dict], study_name: str) -> List[Dict]:
        """Execute a list of experiments."""
        results = []
        
        logger.info(f"Running {len(experiments)} experiments for {study_name}...")
        
        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}] Running: {exp['name']}")
            logger.info(f"Description: {exp['description']}")
            
            # Build command
            cmd = self.base_cmd.copy()
            cmd.append(f"--job_name={study_name}_{exp['name']}")
            cmd.extend(exp['args'])
            
            # Add any base config overrides
            for key, value in self.base_config_overrides.items():
                cmd.append(f"--{key}={value}")
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run experiment
            start_time = time.time()
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd="lerobot")
                success = True
                error_msg = None
                logger.info(f"✅ Experiment {exp['name']} completed successfully")
            except subprocess.CalledProcessError as e:
                success = False
                error_msg = e.stderr
                logger.error(f"❌ Experiment {exp['name']} failed: {error_msg}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results
            exp_result = {
                "name": exp['name'],
                "description": exp['description'],
                "command": ' '.join(cmd),
                "success": success,
                "duration_seconds": duration,
                "error_message": error_msg
            }
            results.append(exp_result)
            
            if success:
                self.completed_runs.append(f"{study_name}/{exp['name']}")
            else:
                self.failed_runs.append(f"{study_name}/{exp['name']}")
            
            # Brief pause between experiments
            time.sleep(5)
        
        # Save results summary
        self._save_results_summary(results, study_name)
        return results

    def _save_results_summary(self, results: List[Dict], study_name: str):
        """Save experiment results to JSON file."""
        summary = {
            "study_name": study_name,
            "dataset_repo": self.dataset_repo,
            "wandb_project": self.wandb_project,
            "total_experiments": len(results),
            "successful_experiments": sum(1 for r in results if r['success']),
            "total_duration_hours": sum(r['duration_seconds'] for r in results) / 3600,
            "experiments": results
        }
        
        output_file = f"ablation_results_{study_name}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n📊 Results saved to: {output_file}")
        logger.info(f"Success rate: {summary['successful_experiments']}/{summary['total_experiments']}")
        logger.info(f"Total time: {summary['total_duration_hours']:.1f} hours")


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
    parser = argparse.ArgumentParser(description="Run comprehensive ablation studies")
    
    parser.add_argument("--study_type", required=True, 
                       choices=["cameras", "state_features", "act_architecture", 
                               "act_hyperparameters", "combined", "custom"],
                       help="Type of ablation study to run")
    
    parser.add_argument("--dataset_repo", required=True,
                       help="Dataset repository ID")
    
    parser.add_argument("--wandb_project", default="ablation_study",
                       help="Weights & Biases project name")
    
    parser.add_argument("--steps", type=int, default=15000,
                       help="Number of training steps per experiment")
    
    parser.add_argument("--config_file", 
                       help="YAML config file for custom ablation studies")
    
    parser.add_argument("--base_config", 
                       help="JSON string with base configuration overrides")
    
    args = parser.parse_args()
    
    # Parse base config overrides
    base_config_overrides = {}
    if args.base_config:
        base_config_overrides = json.loads(args.base_config)
    
    # Initialize runner
    runner = AblationStudyRunner(
        dataset_repo=args.dataset_repo,
        wandb_project=args.wandb_project,
        base_steps=args.steps,
        base_config_overrides=base_config_overrides
    )
    
    # Run selected study
    try:
        if args.study_type == "cameras":
            results = runner.run_camera_ablation()
        elif args.study_type == "state_features":
            results = runner.run_state_feature_ablation()
        elif args.study_type == "act_architecture":
            results = runner.run_act_architecture_ablation()
        elif args.study_type == "act_hyperparameters":
            results = runner.run_act_hyperparameter_ablation()
        elif args.study_type == "combined":
            results = runner.run_combined_ablation()
        elif args.study_type == "custom":
            if not args.config_file:
                raise ValueError("--config_file required for custom ablation studies")
            results = runner.run_custom_ablation(args.config_file)
        
        # Print final summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        logger.info(f"\n🎯 Ablation study completed!")
        logger.info(f"✅ {successful}/{total} experiments successful")
        logger.info(f"❌ {total - successful} experiments failed")
        
        if runner.completed_runs:
            logger.info(f"\n📈 Successful experiments:")
            for run in runner.completed_runs:
                logger.info(f"  - {run}")
                
        if runner.failed_runs:
            logger.info(f"\n💥 Failed experiments:")
            for run in runner.failed_runs:
                logger.info(f"  - {run}")
                
        logger.info(f"\n📊 Check WandB project '{args.wandb_project}' for detailed results")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Ablation study interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Ablation study failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
