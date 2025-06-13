#!/usr/bin/env python3

"""
Simple script to create a combined dataset from two individual LeRobot datasets
and push it to Hugging Face Hub.

This script uses a simpler approach that mimics the convert_unitree_json_to_lerobot.py pattern.
"""

import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import tqdm
import torch

def create_combined_dataset():
    # Configuration
    dataset1_repo_id = "kuehnrobin/g1_pour_can_left_hand"
    dataset2_repo_id = "kuehnrobin/g1_stack_cube_left"
    combined_repo_id = "kuehnrobin/g1_combined_task"
    
    # Task names for each dataset
    task1_name = "pour_can"
    task2_name = "stack_cube"
    
    print("=== Loading Individual Datasets ===")
    
    # Load the two individual datasets
    dataset1 = LeRobotDataset(repo_id=dataset1_repo_id)
    dataset2 = LeRobotDataset(repo_id=dataset2_repo_id)
    
    print(f"Dataset 1 ({task1_name}):")
    print(f"  Episodes: {dataset1.num_episodes}")
    print(f"  Frames: {dataset1.num_frames}")
    
    print(f"\nDataset 2 ({task2_name}):")
    print(f"  Episodes: {dataset2.num_episodes}")
    print(f"  Frames: {dataset2.num_frames}")
    
    # Remove the combined dataset directory if it exists
    combined_path = HF_LEROBOT_HOME / combined_repo_id
    if combined_path.exists():
        print(f"\nRemoving existing combined dataset at {combined_path}")
        shutil.rmtree(combined_path)
    
    print(f"\n=== Creating Combined Dataset ===")
    
    # Create features based on dataset1 but exclude problematic features
    features = {}
    for key, value in dataset1.features.items():
        # Only include core features that we want in the combined dataset
        if key in ["observation.state", "action", "observation.images.cam_left_high", 
                  "observation.images.cam_right_high", "observation.images.cam_left_wrist", 
                  "observation.images.cam_right_wrist", "timestamp"]:
            features[key] = value
    
    print(f"Selected features for combined dataset: {list(features.keys())}")
    
    # Don't add task to features - it will be added at frame level like in the original datasets
    
    # Create a new empty dataset
    combined_dataset = LeRobotDataset.create(
        repo_id=combined_repo_id,
        fps=dataset1.fps,
        features=features,
        robot_type=dataset1.meta.info.get("robot_type", "unknown"),
        use_videos=len(dataset1.meta.video_keys) > 0
    )
    
    print(f"Created empty combined dataset with {len(combined_dataset.features)} features")
    
    # Function to copy data from a source dataset to the combined dataset
    def copy_dataset_data(source_dataset, task_name):
        print(f"\nCopying data from {source_dataset.repo_id} with task '{task_name}'...")
        
        for episode_idx in tqdm.tqdm(range(source_dataset.num_episodes), desc=f"Processing {task_name} episodes"):
            # Get episode boundaries
            from_idx = source_dataset.episode_data_index["from"][episode_idx].item()
            to_idx = source_dataset.episode_data_index["to"][episode_idx].item()
            
            # Copy all frames in this episode
            for frame_idx in range(from_idx, to_idx):
                frame_data = source_dataset[frame_idx]
                
                # Create new frame with only the features we want
                # Exclude the auto-added index features that cause validation errors
                excluded_keys = {"frame_index", "episode_index", "index", "task_index"}
                
                new_frame = {}
                for key, value in frame_data.items():
                    # Skip excluded keys and keys not in target dataset
                    if key in excluded_keys:
                        continue
                    if key not in combined_dataset.features:
                        continue
                        
                    # Fix shape mismatch for timestamp
                    if key == "timestamp" and value.shape == torch.Size([]):
                        # Convert scalar to shape (1,) as expected by feature definition
                        value = value.unsqueeze(0)
                    
                    new_frame[key] = value
                
                # Add task at frame level (not as a feature)
                new_frame["task"] = task_name
                
                # Add frame to combined dataset
                combined_dataset.add_frame(new_frame)
            
            # Save the episode
            combined_dataset.save_episode()
    
    # Copy data from both datasets
    copy_dataset_data(dataset1, task1_name)
    copy_dataset_data(dataset2, task2_name)
    
    print(f"\n=== Combined Dataset Summary ===")
    print(f"Total episodes: {combined_dataset.num_episodes}")
    print(f"Total frames: {combined_dataset.num_frames}")
    
    # Verify task distribution
    print(f"\n=== Task Distribution ===")
    task_counts = {}
    for frame_idx in tqdm.tqdm(range(min(1000, len(combined_dataset))), desc="Sampling frames to check tasks"):
        frame = combined_dataset[frame_idx]
        task = frame["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in task_counts.items():
        print(f"Task '{task}': {count} samples (sampled)")
    
    print(f"\n=== Pushing to Hugging Face Hub ===")
    print(f"Uploading to: https://huggingface.co/datasets/{combined_repo_id}")
    
    try:
        # Push to hub (following the pattern from convert_unitree_json_to_lerobot.py)
        combined_dataset.push_to_hub(upload_large_folder=True)
        print("✅ Successfully pushed to Hugging Face Hub!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{combined_repo_id}")
        
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        print("You may need to:")
        print("1. Log in to Hugging Face: huggingface-cli login")
        print("2. Make sure you have write access to the repository")
        print("3. Check your internet connection")
        
        return combined_dataset
    
    return combined_dataset

if __name__ == "__main__":
    combined_dataset = create_combined_dataset()
    print("\n🎉 Dataset creation completed!")
