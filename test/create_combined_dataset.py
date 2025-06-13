#!/usr/bin/env python3

"""
Script to create a combined dataset from two individual LeRobot datasets
and push it to Hugging Face Hub.

This script:
1. Loads two existing datasets
2. Creates a new combined dataset 
3. Copies all data with proper task labels
4. Pushes the result to HuggingFace Hub
"""

import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import tqdm

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
    print(f"  Features: {list(dataset1.features.keys())}")
    
    print(f"\nDataset 2 ({task2_name}):")
    print(f"  Episodes: {dataset2.num_episodes}")
    print(f"  Frames: {dataset2.num_frames}")
    print(f"  Features: {list(dataset2.features.keys())}")
    
    # Check that datasets have compatible features
    common_features = set(dataset1.features.keys()) & set(dataset2.features.keys())
    print(f"\nCommon features: {common_features}")
    
    if len(common_features) == 0:
        raise ValueError("Datasets have no common features!")
    
    # Remove the combined dataset directory if it exists
    combined_path = HF_LEROBOT_HOME / combined_repo_id
    if combined_path.exists():
        print(f"\nRemoving existing combined dataset at {combined_path}")
        shutil.rmtree(combined_path)
    
    print(f"\n=== Creating Combined Dataset ===")
    
    # Create a new empty dataset with the same features as dataset1
    # We'll use dataset1's metadata as the base
    combined_dataset = LeRobotDataset.create(
        repo_id=combined_repo_id,
        fps=dataset1.fps,
        features=dataset1.features,
        robot_type=dataset1.meta.info.get("robot_type", "unknown"),
        use_videos=len(dataset1.meta.video_keys) > 0
    )
    
    print(f"Created empty combined dataset with {len(combined_dataset.features)} features")
    
    # Function to copy data from a source dataset to the combined dataset
    def copy_dataset_data(source_dataset, task_name, start_episode_idx=0):
        print(f"\nCopying data from {source_dataset.repo_id} with task '{task_name}'...")
        
        episode_offset = start_episode_idx
        
        for episode_idx in tqdm.tqdm(range(source_dataset.num_episodes), desc=f"Processing {task_name} episodes"):
            # Get episode boundaries
            from_idx = source_dataset.episode_data_index["from"][episode_idx].item()
            to_idx = source_dataset.episode_data_index["to"][episode_idx].item()
            
            # Copy all frames in this episode
            for frame_idx in range(from_idx, to_idx):
                frame_data = source_dataset[frame_idx]
                
                # Create new frame with updated task
                new_frame = {}
                for key, value in frame_data.items():
                    if key == "task":
                        new_frame["task"] = task_name  # Override with our task name
                    elif key in combined_dataset.features:  # Only copy features that exist in combined dataset
                        new_frame[key] = value
                
                # Add frame to combined dataset
                combined_dataset.add_frame(new_frame)
            
            # Save the episode
            combined_dataset.save_episode()
        
        return episode_offset + source_dataset.num_episodes
    
    # Copy data from both datasets
    next_episode_idx = copy_dataset_data(dataset1, task1_name, 0)
    copy_dataset_data(dataset2, task2_name, next_episode_idx)
    
    print(f"\n=== Combined Dataset Summary ===")
    print(f"Total episodes: {combined_dataset.num_episodes}")
    print(f"Total frames: {combined_dataset.num_frames}")
    print(f"Tasks: {combined_dataset.meta.tasks}")
    
    # Verify task distribution
    print(f"\n=== Task Distribution ===")
    task_counts = {}
    for frame_idx in range(len(combined_dataset)):
        frame = combined_dataset[frame_idx]
        task = frame["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in task_counts.items():
        print(f"Task '{task}': {count} samples")
    
    print(f"\n=== Pushing to Hugging Face Hub ===")
    print(f"Uploading to: https://huggingface.co/datasets/{combined_repo_id}")
    
    try:
        # Push to hub with appropriate tags and metadata
        combined_dataset.push_to_hub(
            tags=["robotics", "LeRobot", "unitree", "g1", "combined", "multi-task"],
            license="apache-2.0",
            upload_large_folder=True,
            description=f"Combined dataset from {dataset1_repo_id} and {dataset2_repo_id} with tasks '{task1_name}' and '{task2_name}'"
        )
        print("✅ Successfully pushed to Hugging Face Hub!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{combined_repo_id}")
        
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        print("You may need to:")
        print("1. Log in to Hugging Face: huggingface-cli login")
        print("2. Make sure you have write access to the repository")
        print("3. Check your internet connection")
        
        # Save locally for now
        #print(f"\n📁 Dataset saved locally at: {combined_dataset.root}")
        return combined_dataset
    
    return combined_dataset

if __name__ == "__main__":
    combined_dataset = create_combined_dataset()
    print("\n🎉 Dataset creation completed!")
