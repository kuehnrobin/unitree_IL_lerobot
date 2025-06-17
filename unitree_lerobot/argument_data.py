#!/usr/bin/env python

"""
Data Augmentation Script for Teleoperation Datasets

This script provides comprehensive data augmentation capabilities for teleoperation datasets
to improve imitation learning performance. It automatically reads episode quality from each
episode's data.json file and applies appropriate weighting and augmentation.

Key Features:
1. Automatic episode quality detection from data.json files ('optimal', 'suboptimal', 'recovery')
2. Episode weighting based on quality (optimal episodes get additional copies)
3. Image augmentation with lighting variations and noise
4. Joint coordinate noise injection for robustness
5. Proper episode naming and organization

Author: Created for Unitree IL Lerobot project
Date: 2025-06-17
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration class for data augmentation parameters."""
    
    # Input/Output paths
    input_dataset_path: str
    output_dataset_path: str
    
    # Episode weighting parameters
    optimal_weight: float = 2.0  # Multiplication factor for optimal episodes
    
    # Image augmentation parameters
    enable_lighting_augmentation: bool = True
    brightness_range: Tuple[float, float] = (0.7, 1.3)  # Range for brightness adjustment
    contrast_range: Tuple[float, float] = (0.8, 1.2)    # Range for contrast adjustment
    saturation_range: Tuple[float, float] = (0.8, 1.2)  # Range for saturation adjustment
    hue_shift_range: int = 15  # Max hue shift in degrees
    
    # Noise parameters
    enable_joint_noise: bool = True
    joint_noise_std: float = 0.02  # Standard deviation for joint noise (in radians)
    max_joint_noise: float = 0.05  # Maximum noise magnitude (in radians)
    
    # Processing parameters
    augmentation_probability: float = 0.5  # Probability of applying augmentation to each episode
    preserve_original: bool = True  # Keep original episodes alongside augmented ones
    seed: Optional[int] = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.optimal_weight <= 0:
            raise ValueError("optimal_weight must be positive")
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be between 0 and 1")
        if self.joint_noise_std < 0:
            raise ValueError("joint_noise_std must be non-negative")


class ImageAugmentor:
    """Handles image augmentation operations including lighting variations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def generate_timestep_augmentation_params(self) -> Dict:
        """
        Generate consistent augmentation parameters for all cameras in a timestep.
        
        Returns:
            Dictionary with augmentation parameters to apply to all cameras
        """
        params = {
            'brightness_factor': self.rng.uniform(*self.config.brightness_range),
            'contrast_factor': self.rng.uniform(*self.config.contrast_range),
            'saturation_factor': self.rng.uniform(*self.config.saturation_range),
            'apply_blur': self.rng.random() < 0.2,  # 20% chance
            'blur_radius': self.rng.uniform(0.3, 0.8) if self.rng.random() < 0.2 else 0.5,
            'apply_noise': self.rng.random() < 0.3,  # 30% chance
            'noise_std': self.rng.uniform(2, 5) if self.rng.random() < 0.3 else 3,
        }
        return params
    
    def augment_image_with_params(self, image_path: str, output_path: str, params: Dict) -> None:
        """
        Apply lighting augmentation to a single image using provided parameters.
        
        Args:
            image_path: Path to input image
            output_path: Path to save augmented image
            params: Augmentation parameters from generate_timestep_augmentation_params()
        """
        try:
            # Load image using PIL for better color manipulation
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply brightness adjustment
                img = ImageEnhance.Brightness(img).enhance(params['brightness_factor'])
                
                # Apply contrast adjustment
                img = ImageEnhance.Contrast(img).enhance(params['contrast_factor'])
                
                # Apply saturation adjustment
                img = ImageEnhance.Color(img).enhance(params['saturation_factor'])
                
                # Add subtle blur if specified
                if params['apply_blur']:
                    img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_radius']))
                
                # Add slight noise if specified
                if params['apply_noise']:
                    img_array = np.array(img)
                    noise = self.rng.normal(0, params['noise_std'], img_array.shape).astype(np.int16)
                    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                
                # Save augmented image
                img.save(output_path, 'JPEG', quality=95)
                
        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")
            # Fallback: copy original image
            shutil.copy2(image_path, output_path)
    
    def augment_image(self, image_path: str, output_path: str) -> None:
        """
        Apply lighting augmentation to a single image (for backward compatibility).
        
        Args:
            image_path: Path to input image
            output_path: Path to save augmented image
        """
        params = self.generate_timestep_augmentation_params()
        self.augment_image_with_params(image_path, output_path, params)
    
    def should_augment(self) -> bool:
        """Determine if augmentation should be applied based on probability."""
        return self.rng.random() < self.config.augmentation_probability


class JointNoiseInjector:
    """Handles joint coordinate noise injection for robustness."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def add_joint_noise(self, joint_positions: List[float]) -> List[float]:
        """
        Add Gaussian noise to joint positions.
        
        Args:
            joint_positions: List of joint positions in radians
            
        Returns:
            List of joint positions with added noise
        """
        if not self.config.enable_joint_noise:
            return joint_positions
        
        noisy_positions = []
        for pos in joint_positions:
            # Generate noise with clipping to prevent extreme values
            noise = self.rng.normal(0, self.config.joint_noise_std)
            noise = np.clip(noise, -self.config.max_joint_noise, self.config.max_joint_noise)
            noisy_positions.append(pos + noise)
        
        return noisy_positions


class DatasetAugmentor:
    """Main class for dataset augmentation operations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.image_augmentor = ImageAugmentor(config)
        self.joint_noise_injector = JointNoiseInjector(config)
        self.episode_counter = 0
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
    
    def get_episode_directories(self) -> List[Path]:
        """Get all episode directories from input dataset."""
        input_path = Path(self.config.input_dataset_path)
        episode_dirs = []
        
        for item in input_path.iterdir():
            if item.is_dir() and item.name.startswith('episode_'):
                episode_dirs.append(item)
        
        # Sort by episode number
        episode_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        return episode_dirs
    
    def get_episode_number(self, episode_dir: Path) -> int:
        """Extract episode number from directory name."""
        return int(episode_dir.name.split('_')[1])
    
    def get_episode_quality(self, episode_dir: Path) -> str:
        """
        Extract quality value from episode's data.json file.
        
        Args:
            episode_dir: Path to episode directory
            
        Returns:
            Quality string ('optimal', 'suboptimal', 'recovery') or 'unknown' if not found
        """
        data_json_path = episode_dir / "data.json"
        
        if not data_json_path.exists():
            logger.warning(f"data.json not found in {episode_dir.name}, assuming 'unknown' quality")
            return "unknown"
        
        try:
            with open(data_json_path, 'r') as f:
                data = json.load(f)
            
            quality = data.get('quality', 'unknown')
            if quality not in ['optimal', 'suboptimal', 'recovery']:
                logger.warning(f"Invalid quality '{quality}' in {episode_dir.name}, assuming 'unknown'")
                return "unknown"
            
            return quality
            
        except Exception as e:
            logger.error(f"Error reading data.json from {episode_dir.name}: {e}")
            return "unknown"
    
    def create_output_episode_name(self) -> str:
        """Generate new episode directory name with proper numbering."""
        episode_name = f"episode_{self.episode_counter:04d}"
        self.episode_counter += 1
        return episode_name
    
    def copy_episode_structure(self, src_episode: Path, dst_episode: Path, augment: bool = False) -> None:
        """
        Copy episode structure and apply augmentations if specified.
        
        Args:
            src_episode: Source episode directory
            dst_episode: Destination episode directory  
            augment: Whether to apply augmentations
        """
        # Create destination directory
        dst_episode.mkdir(parents=True, exist_ok=True)
        
        # Copy and process data.json
        src_data_file = src_episode / "data.json"
        dst_data_file = dst_episode / "data.json"
        
        if src_data_file.exists():
            self.process_data_json(src_data_file, dst_data_file, src_episode, dst_episode, augment)
        
        # Copy other directories (audios, depths) without modification
        for subdir in ["audios", "depths"]:
            src_subdir = src_episode / subdir
            dst_subdir = dst_episode / subdir
            if src_subdir.exists():
                shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
    
    def process_data_json(
        self, 
        src_file: Path, 
        dst_file: Path, 
        src_episode: Path, 
        dst_episode: Path, 
        augment: bool
    ) -> None:
        """
        Process the data.json file and apply augmentations.
        
        Args:
            src_file: Source data.json file
            dst_file: Destination data.json file
            src_episode: Source episode directory
            dst_episode: Destination episode directory
            augment: Whether to apply augmentations
        """
        try:
            with open(src_file, 'r') as f:
                data = json.load(f)
            
            # Create colors directory in destination
            dst_colors_dir = dst_episode / "colors"
            dst_colors_dir.mkdir(exist_ok=True)
            
            # Process each timestep
            for timestep_data in tqdm(data["data"], desc=f"Processing {src_episode.name}", leave=False):
                # Process color images
                if "colors" in timestep_data and timestep_data["colors"]:
                    self.process_timestep_images(
                        timestep_data["colors"], 
                        src_episode, 
                        dst_episode, 
                        augment
                    )
                
                # Process joint states with noise injection
                if augment and "states" in timestep_data and timestep_data["states"]:
                    self.process_timestep_states(timestep_data["states"])
            
            # Save processed data.json
            with open(dst_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error processing data.json for {src_episode.name}: {e}")
            # Fallback: copy original file
            shutil.copy2(src_file, dst_file)
    
    def process_timestep_images(
        self, 
        colors_data: Dict, 
        src_episode: Path, 
        dst_episode: Path, 
        augment: bool
    ) -> None:
        """Process color images for a single timestep with consistent augmentation."""
        # Check if colors_data is None or empty
        if not colors_data:
            return
        
        # Generate augmentation parameters once per timestep for all cameras
        augmentation_params = None
        if augment and self.config.enable_lighting_augmentation:
            augmentation_params = self.image_augmentor.generate_timestep_augmentation_params()
        
        for camera_key, relative_path in colors_data.items():
            # Skip if relative_path is None or empty
            if not relative_path:
                continue
                
            src_image_path = src_episode / relative_path
            dst_image_path = dst_episode / relative_path
            
            # Ensure destination directory exists
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if source image doesn't exist
            if not src_image_path.exists():
                logger.warning(f"Source image not found: {src_image_path}")
                continue
            
            if augment and self.config.enable_lighting_augmentation and augmentation_params:
                # Apply consistent augmentation to all cameras in this timestep
                self.image_augmentor.augment_image_with_params(
                    str(src_image_path), 
                    str(dst_image_path), 
                    augmentation_params
                )
            else:
                # Copy original image
                shutil.copy2(src_image_path, dst_image_path)
    
    def process_timestep_states(self, states_data: Dict) -> None:
        """Process joint states and add noise if enabled."""
        if not states_data:
            return
            
        for component, state_info in states_data.items():
            if not state_info or not isinstance(state_info, dict):
                continue
                
            if "qpos" in state_info and isinstance(state_info["qpos"], list) and state_info["qpos"]:
                # Add noise to joint positions
                original_qpos = state_info["qpos"]
                noisy_qpos = self.joint_noise_injector.add_joint_noise(original_qpos)
                state_info["qpos"] = noisy_qpos
    
    def process_single_episode(self, episode_dir: Path, episode_num: int) -> None:
        """Process a single episode with appropriate weighting and augmentation."""
        output_path = Path(self.config.output_dataset_path)
        
        # Get quality from the episode's data.json file
        episode_quality = self.get_episode_quality(episode_dir)
        
        # Determine episode type based on quality
        is_optimal = (episode_quality == 'optimal')
        is_recovery = (episode_quality == 'recovery')
        is_suboptimal = (episode_quality == 'suboptimal')
        
        # Log episode quality
        logger.info(f"Episode {episode_num} quality: {episode_quality}")
        
        # Always copy original episode if preserve_original is True
        if self.config.preserve_original:
            dst_episode_name = self.create_output_episode_name()
            dst_episode_path = output_path / dst_episode_name
            self.copy_episode_structure(episode_dir, dst_episode_path, augment=False)
            logger.info(f"Copied original episode {episode_num} -> {dst_episode_name} (quality: {episode_quality})")
        
        # Create additional copies for optimal episodes (weighting)
        if is_optimal:
            num_additional_copies = int(self.config.optimal_weight) - 1
            for i in range(num_additional_copies):
                dst_episode_name = self.create_output_episode_name()
                dst_episode_path = output_path / dst_episode_name
                # Apply augmentation to additional copies
                self.copy_episode_structure(episode_dir, dst_episode_path, augment=True)
                logger.info(f"Created optimal episode copy {episode_num} -> {dst_episode_name} (copy {i+1})")
        
        # Create augmented version for all episodes
        if self.image_augmentor.should_augment():
            dst_episode_name = self.create_output_episode_name()
            dst_episode_path = output_path / dst_episode_name
            self.copy_episode_structure(episode_dir, dst_episode_path, augment=True)
            logger.info(f"Created augmented episode {episode_num} -> {dst_episode_name} (quality: {episode_quality})")
    
    def augment_dataset(self) -> None:
        """Main method to perform dataset augmentation."""
        logger.info("Starting dataset augmentation...")
        logger.info(f"Input dataset: {self.config.input_dataset_path}")
        logger.info(f"Output dataset: {self.config.output_dataset_path}")
        
        # Create output directory
        output_path = Path(self.config.output_dataset_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all episode directories
        episode_dirs = self.get_episode_directories()
        logger.info(f"Found {len(episode_dirs)} episodes to process")
        logger.info(f"Optimal episodes will be weighted {self.config.optimal_weight}x based on quality in data.json")
        
        # Process each episode
        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            episode_num = self.get_episode_number(episode_dir)
            self.process_single_episode(episode_dir, episode_num)
        
        logger.info(f"Dataset augmentation completed!")
        logger.info(f"Total episodes created: {self.episode_counter}")
        logger.info(f"Output saved to: {output_path}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Augment teleoperation datasets for improved imitation learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default="/media/robin/DATA/stack_cube_left",
        help="Path to input dataset directory"
    )
    
    parser.add_argument(
        "--output_dataset_path", 
        type=str,
        default="/media/robin/DATA/argumented_stack_cube_left",
        help="Path to output augmented dataset directory"
    )
    
    # Episode weighting arguments
    parser.add_argument(
        "--optimal_weight",
        type=float,
        default=2.0,
        help="Multiplication factor for optimal episodes (quality will be read from data.json)"
    )
    
    # Image augmentation arguments
    parser.add_argument(
        "--enable_lighting_augmentation",
        action="store_true",
        help="Enable lighting and color augmentation (default: True)"
    )
    
    parser.add_argument(
        "--disable_lighting_augmentation",
        action="store_true",
        help="Disable lighting and color augmentation"
    )
    
    parser.add_argument(
        "--brightness_range",
        type=float,
        nargs=2,
        default=[0.7, 1.3],
        help="Min and max brightness factors"
    )
    
    parser.add_argument(
        "--contrast_range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        help="Min and max contrast factors"
    )
    
    # Joint noise arguments
    parser.add_argument(
        "--enable_joint_noise",
        action="store_true", 
        help="Enable joint coordinate noise injection (default: True)"
    )
    
    parser.add_argument(
        "--disable_joint_noise",
        action="store_true",
        help="Disable joint coordinate noise injection"
    )
    
    parser.add_argument(
        "--joint_noise_std",
        type=float,
        default=0.02,
        help="Standard deviation for joint noise (radians)"
    )
    
    parser.add_argument(
        "--max_joint_noise", 
        type=float,
        default=0.05,
        help="Maximum joint noise magnitude (radians)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--augmentation_probability",
        type=float,
        default=0.5,
        help="Probability of applying augmentation to each episode"
    )
    
    parser.add_argument(
        "--preserve_original",
        action="store_true",
        help="Keep original episodes alongside augmented ones (default: True)"
    )
    
    parser.add_argument(
        "--skip_original",
        action="store_true", 
        help="Skip copying original episodes, only create augmented versions"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle boolean flags with defaults
    enable_lighting = not args.disable_lighting_augmentation if hasattr(args, 'disable_lighting_augmentation') else True
    if hasattr(args, 'enable_lighting_augmentation') and args.enable_lighting_augmentation:
        enable_lighting = True
    
    enable_joint_noise = not args.disable_joint_noise if hasattr(args, 'disable_joint_noise') else True
    if hasattr(args, 'enable_joint_noise') and args.enable_joint_noise:
        enable_joint_noise = True
    
    preserve_original = not args.skip_original if hasattr(args, 'skip_original') else True
    if hasattr(args, 'preserve_original') and args.preserve_original:
        preserve_original = True
    
    # Create configuration (episode quality will be read from each episode's data.json)
    config = AugmentationConfig(
        input_dataset_path=args.input_dataset_path,
        output_dataset_path=args.output_dataset_path,
        optimal_weight=args.optimal_weight,
        enable_lighting_augmentation=enable_lighting,
        brightness_range=tuple(args.brightness_range),
        contrast_range=tuple(args.contrast_range),
        enable_joint_noise=enable_joint_noise,
        joint_noise_std=args.joint_noise_std,
        max_joint_noise=args.max_joint_noise,
        augmentation_probability=args.augmentation_probability,
        preserve_original=preserve_original,
        seed=args.seed
    )
    
    # Validate paths
    input_path = Path(config.input_dataset_path)
    if not input_path.exists():
        logger.error(f"Input dataset path does not exist: {input_path}")
        sys.exit(1)
    
    # Create augmentor and run
    try:
        augmentor = DatasetAugmentor(config)
        augmentor.augment_dataset()  
        logger.info("Data augmentation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
