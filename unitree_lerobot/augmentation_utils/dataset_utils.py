"""
Dataset processing utilities for teleoperation data augmentation.

This module provides high-level utilities for processing and manipulating
teleoperation datasets, including episode management, data validation,
and batch processing operations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    High-level dataset processing utilities for teleoperation data.
    
    This class provides methods for dataset validation, episode management,
    and batch processing operations commonly needed for dataset augmentation.
    """
    
    @staticmethod
    def validate_episode_structure(episode_path: Path) -> bool:
        """
        Validate that an episode directory has the expected structure.
        
        Args:
            episode_path: Path to episode directory
            
        Returns:
            True if episode structure is valid, False otherwise
        """
        try:
            # Check if data.json exists
            data_json_path = episode_path / "data.json"
            if not data_json_path.exists():
                logger.warning(f"Missing data.json in {episode_path}")
                return False
            
            # Check if colors directory exists
            colors_dir = episode_path / "colors"
            if not colors_dir.exists():
                logger.warning(f"Missing colors directory in {episode_path}")
                return False
            
            # Try to load and validate data.json structure
            with open(data_json_path, 'r') as f:
                data = json.load(f)
            
            # Check required top-level keys
            required_keys = ["info", "data"]
            for key in required_keys:
                if key not in data:
                    logger.warning(f"Missing required key '{key}' in {episode_path}/data.json")
                    return False
            
            # Check if data contains timesteps
            if not isinstance(data["data"], list) or len(data["data"]) == 0:
                logger.warning(f"Empty or invalid data array in {episode_path}/data.json")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating episode {episode_path}: {e}")
            return False
    
    @staticmethod
    def get_episode_info(episode_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract information about an episode.
        
        Args:
            episode_path: Path to episode directory
            
        Returns:
            Dictionary with episode information or None if error
        """
        try:
            data_json_path = episode_path / "data.json"
            with open(data_json_path, 'r') as f:
                data = json.load(f)
            
            info = {
                'episode_path': str(episode_path),
                'episode_name': episode_path.name,
                'num_timesteps': len(data.get("data", [])),
                'info': data.get("info", {}),
                'text': data.get("text", {}),
                'has_colors': (episode_path / "colors").exists(),
                'has_depths': (episode_path / "depths").exists(),
                'has_audios': (episode_path / "audios").exists()
            }
            
            # Get joint information
            if data.get("data") and len(data["data"]) > 0:
                first_timestep = data["data"][0]
                if "states" in first_timestep:
                    info['joint_components'] = list(first_timestep["states"].keys())
                    info['total_joints'] = sum(
                        len(component_data.get("qpos", [])) 
                        for component_data in first_timestep["states"].values()
                    )
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting episode info for {episode_path}: {e}")
            return None
    
    @staticmethod
    def count_images_in_episode(episode_path: Path) -> int:
        """
        Count the total number of images in an episode.
        
        Args:
            episode_path: Path to episode directory
            
        Returns:
            Total number of images
        """
        try:
            colors_dir = episode_path / "colors"
            if not colors_dir.exists():
                return 0
            
            image_count = 0
            for file_path in colors_dir.iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_count += 1
            
            return image_count
            
        except Exception as e:
            logger.error(f"Error counting images in {episode_path}: {e}")
            return 0
    
    @staticmethod
    def copy_episode_metadata(src_episode: Path, dst_episode: Path) -> None:
        """
        Copy non-data files from source to destination episode.
        
        Args:
            src_episode: Source episode directory
            dst_episode: Destination episode directory
        """
        try:
            # Create destination directory
            dst_episode.mkdir(parents=True, exist_ok=True)
            
            # Copy audios directory if it exists
            src_audios = src_episode / "audios"
            if src_audios.exists():
                dst_audios = dst_episode / "audios"
                shutil.copytree(src_audios, dst_audios, dirs_exist_ok=True)
            
            # Copy depths directory if it exists
            src_depths = src_episode / "depths"
            if src_depths.exists():
                dst_depths = dst_episode / "depths"
                shutil.copytree(src_depths, dst_depths, dirs_exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error copying episode metadata from {src_episode} to {dst_episode}: {e}")
    
    @staticmethod
    def get_dataset_statistics(dataset_path: Path) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about a dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_episodes': 0,
            'valid_episodes': 0,
            'total_timesteps': 0,
            'total_images': 0,
            'episode_lengths': [],
            'joint_components': set(),
            'has_depths': 0,
            'has_audios': 0
        }
        
        try:
            episode_dirs = []
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name.startswith('episode_'):
                    episode_dirs.append(item)
            
            stats['total_episodes'] = len(episode_dirs)
            
            for episode_dir in episode_dirs:
                if DatasetProcessor.validate_episode_structure(episode_dir):
                    stats['valid_episodes'] += 1
                    
                    episode_info = DatasetProcessor.get_episode_info(episode_dir)
                    if episode_info:
                        stats['total_timesteps'] += episode_info['num_timesteps']
                        stats['episode_lengths'].append(episode_info['num_timesteps'])
                        stats['total_images'] += DatasetProcessor.count_images_in_episode(episode_dir)
                        
                        if episode_info.get('joint_components'):
                            stats['joint_components'].update(episode_info['joint_components'])
                        
                        if episode_info['has_depths']:
                            stats['has_depths'] += 1
                        if episode_info['has_audios']:  
                            stats['has_audios'] += 1
            
            # Convert set to list for JSON serialization
            stats['joint_components'] = list(stats['joint_components'])
            
            # Calculate additional statistics
            if stats['episode_lengths']:
                stats['avg_episode_length'] = sum(stats['episode_lengths']) / len(stats['episode_lengths'])
                stats['min_episode_length'] = min(stats['episode_lengths'])
                stats['max_episode_length'] = max(stats['episode_lengths'])
            
        except Exception as e:
            logger.error(f"Error generating dataset statistics: {e}")
        
        return stats
    
    @staticmethod
    def create_dataset_summary(dataset_path: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the dataset and optionally save to file.
        
        Args:
            dataset_path: Path to dataset directory
            output_file: Optional path to save summary JSON file
            
        Returns:
            Dictionary with dataset summary
        """
        summary = {
            'dataset_path': str(dataset_path),
            'statistics': DatasetProcessor.get_dataset_statistics(dataset_path),
            'episodes': []
        }
        
        try:
            episode_dirs = []
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name.startswith('episode_'):
                    episode_dirs.append(item)
            
            # Sort episodes by number
            episode_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
            
            for episode_dir in episode_dirs:
                episode_info = DatasetProcessor.get_episode_info(episode_dir)
                if episode_info:
                    summary['episodes'].append(episode_info)
            
            # Save summary to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Dataset summary saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating dataset summary: {e}")
        
        return summary
    
    @staticmethod
    def verify_augmented_dataset(
        original_path: Path, 
        augmented_path: Path,
        expected_multiplier: float = 2.0
    ) -> bool:
        """
        Verify that augmented dataset has expected properties.
        
        Args:
            original_path: Path to original dataset
            augmented_path: Path to augmented dataset
            expected_multiplier: Expected size multiplier
            
        Returns:
            True if verification passes, False otherwise
        """
        try:
            orig_stats = DatasetProcessor.get_dataset_statistics(original_path)
            aug_stats = DatasetProcessor.get_dataset_statistics(augmented_path)
            
            # Check episode count
            expected_episodes = int(orig_stats['total_episodes'] * expected_multiplier)
            if aug_stats['total_episodes'] < expected_episodes * 0.9:  # Allow 10% tolerance
                logger.warning(
                    f"Augmented dataset has fewer episodes than expected: "
                    f"{aug_stats['total_episodes']} < {expected_episodes}"
                )
                return False
            
            # Check that all episodes are valid
            if aug_stats['valid_episodes'] != aug_stats['total_episodes']:
                logger.warning(
                    f"Some episodes in augmented dataset are invalid: "
                    f"{aug_stats['valid_episodes']} valid out of {aug_stats['total_episodes']}"
                )
                return False
            
            logger.info("Augmented dataset verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying augmented dataset: {e}")
            return False
