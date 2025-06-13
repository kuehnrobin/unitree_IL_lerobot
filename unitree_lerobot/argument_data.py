#!/usr/bin/env python

"""
Data Augmentation Script for Teleoperation Datasets

This script provides comprehensive data augmentation capabilities for teleoperation datasets
to improve imitation learning performance. It includes episode weighting, lighting augmentation,
and noise injection to create more robust training datasets.

Key Features:
1. Episode weighting based on quality (optimal vs recovery episodes)
2. Image augmentation with lighting variations and noise
3. Joint coordinate noise injection for robustness
4. Proper episode naming and organization

Author: Created for Unitree IL Lerobot project
Date: 2025-06-12
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
    optimal_episodes: Optional[List[int]] = None
    recovery_episodes: Optional[List[int]] = None
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
    
    def augment_image(self, image_path: str, output_path: str) -> None:
        """
        Apply lighting augmentation to a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save augmented image
        """
        try:
            # Load image using PIL for better color manipulation
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply random brightness adjustment
                brightness_factor = self.rng.uniform(*self.config.brightness_range)
                img = ImageEnhance.Brightness(img).enhance(brightness_factor)
                
                # Apply random contrast adjustment
                contrast_factor = self.rng.uniform(*self.config.contrast_range)
                img = ImageEnhance.Contrast(img).enhance(contrast_factor)
                
                # Apply random saturation adjustment
                saturation_factor = self.rng.uniform(*self.config.saturation_range)
                img = ImageEnhance.Color(img).enhance(saturation_factor)
                
                # Add subtle blur occasionally for realism
                if self.rng.random() < 0.2:  # 20% chance
                    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                # Add slight noise
                if self.rng.random() < 0.3:  # 30% chance
                    img_array = np.array(img)
                    noise = self.rng.normal(0, 3, img_array.shape).astype(np.int16)
                    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                
                # Save augmented image
                img.save(output_path, 'JPEG', quality=95)
                
        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")
            # Fallback: copy original image
            shutil.copy2(image_path, output_path)
    
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
                if "colors" in timestep_data:
                    self.process_timestep_images(
                        timestep_data["colors"], 
                        src_episode, 
                        dst_episode, 
                        augment
                    )
                
                # Process joint states with noise injection
                if augment and "states" in timestep_data:
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
        """Process color images for a single timestep."""
        for camera_key, relative_path in colors_data.items():
            src_image_path = src_episode / relative_path
            dst_image_path = dst_episode / relative_path
            
            # Ensure destination directory exists
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            if augment and self.config.enable_lighting_augmentation:
                # Apply augmentation
                self.image_augmentor.augment_image(str(src_image_path), str(dst_image_path))
            else:
                # Copy original image
                shutil.copy2(src_image_path, dst_image_path)
    
    def process_timestep_states(self, states_data: Dict) -> None:
        """Process joint states and add noise if enabled."""
        for component, state_info in states_data.items():
            if "qpos" in state_info and isinstance(state_info["qpos"], list):
                # Add noise to joint positions
                original_qpos = state_info["qpos"]
                noisy_qpos = self.joint_noise_injector.add_joint_noise(original_qpos)
                state_info["qpos"] = noisy_qpos
    
    def process_single_episode(self, episode_dir: Path, episode_num: int) -> None:
        """Process a single episode with appropriate weighting and augmentation."""
        output_path = Path(self.config.output_dataset_path)
        
        # Determine if this is an optimal episode
        is_optimal = (self.config.optimal_episodes and 
                     episode_num in self.config.optimal_episodes)
        is_recovery = (self.config.recovery_episodes and 
                      episode_num in self.config.recovery_episodes)
        
        # Always copy original episode if preserve_original is True
        if self.config.preserve_original:
            dst_episode_name = self.create_output_episode_name()
            dst_episode_path = output_path / dst_episode_name
            self.copy_episode_structure(episode_dir, dst_episode_path, augment=False)
            logger.info(f"Copied original episode {episode_num} -> {dst_episode_name}")
        
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
            logger.info(f"Created augmented episode {episode_num} -> {dst_episode_name}")
    
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
        
        # Log episode categorization
        if self.config.optimal_episodes:
            logger.info(f"Optimal episodes (weight {self.config.optimal_weight}x): {self.config.optimal_episodes}")
        if self.config.recovery_episodes:
            logger.info(f"Recovery episodes: {self.config.recovery_episodes}")
        
        # Process each episode
        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            episode_num = self.get_episode_number(episode_dir)
            self.process_single_episode(episode_dir, episode_num)
        
        logger.info(f"Dataset augmentation completed!")
        logger.info(f"Total episodes created: {self.episode_counter}")
        logger.info(f"Output saved to: {output_path}")


def parse_episode_list(episode_str: str) -> List[int]:
    """Parse comma-separated episode numbers."""
    if not episode_str:
        return []
    
    episodes = []
    for part in episode_str.split(','):
        part = part.strip()
        if '-' in part:
            # Handle ranges like "1-5"
            start, end = map(int, part.split('-'))
            episodes.extend(range(start, end + 1))
        else:
            episodes.append(int(part))
    
    return sorted(list(set(episodes)))  # Remove duplicates and sort


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
        "--optimal_episodes",
        type=str,
        default="",
        help="Comma-separated list of optimal episode numbers (e.g., '1,2,5-8')"
    )
    
    parser.add_argument(
        "--recovery_episodes",
        type=str, 
        default="",
        help="Comma-separated list of recovery episode numbers (e.g., '3,4,9-12')"
    )
    
    parser.add_argument(
        "--optimal_weight",
        type=float,
        default=2.0,
        help="Multiplication factor for optimal episodes"
    )
    
    # Image augmentation arguments
    parser.add_argument(
        "--enable_lighting_augmentation",
        action="store_true",
        default=True,
        help="Enable lighting and color augmentation"
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
        default=True,
        help="Enable joint coordinate noise injection"
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
        default=True,
        help="Keep original episodes alongside augmented ones"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse episode lists
    optimal_episodes = parse_episode_list(args.optimal_episodes) if args.optimal_episodes else None
    recovery_episodes = parse_episode_list(args.recovery_episodes) if args.recovery_episodes else None
    
    # Create configuration
    config = AugmentationConfig(
        input_dataset_path=args.input_dataset_path,
        output_dataset_path=args.output_dataset_path,
        optimal_episodes=optimal_episodes,
        recovery_episodes=recovery_episodes,
        optimal_weight=args.optimal_weight,
        enable_lighting_augmentation=args.enable_lighting_augmentation,
        brightness_range=tuple(args.brightness_range),
        contrast_range=tuple(args.contrast_range),
        enable_joint_noise=args.enable_joint_noise,
        joint_noise_std=args.joint_noise_std,
        max_joint_noise=args.max_joint_noise,
        augmentation_probability=args.augmentation_probability,
        preserve_original=args.preserve_original,
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
