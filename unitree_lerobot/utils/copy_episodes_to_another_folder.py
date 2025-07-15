#!/usr/bin/env python3
"""
Script to copy episodes from one folder to another with automatic renaming.
Finds the highest episode number in the destination folder and starts numbering from there.
"""

import os
import shutil
import argparse
import re
from pathlib import Path


def find_highest_episode_number(folder_path):
    """Find the highest episode number in a folder."""
    if not os.path.exists(folder_path):
        return -1
    
    highest_num = -1
    episode_pattern = re.compile(r'episode_(\d+)')
    
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):
            match = episode_pattern.match(item)
            if match:
                episode_num = int(match.group(1))
                highest_num = max(highest_num, episode_num)
    
    return highest_num


def get_sorted_episodes(folder_path):
    """Get all episode folders sorted by episode number."""
    if not os.path.exists(folder_path):
        print(f"Source folder {folder_path} does not exist!")
        return []
    
    episodes = []
    episode_pattern = re.compile(r'episode_(\d+)')
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            match = episode_pattern.match(item)
            if match:
                episode_num = int(match.group(1))
                episodes.append((episode_num, item, item_path))
    
    # Sort by episode number
    episodes.sort(key=lambda x: x[0])
    return episodes


def copy_episodes(source_folder, dest_folder):
    """Copy episodes from source to destination with proper renaming."""
    # Expand user paths
    source_folder = os.path.expanduser(source_folder)
    dest_folder = os.path.expanduser(dest_folder)
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Find highest episode number in destination
    highest_dest = find_highest_episode_number(dest_folder)
    print(f"Highest episode number in destination: {highest_dest}")
    
    # Get sorted episodes from source
    source_episodes = get_sorted_episodes(source_folder)
    if not source_episodes:
        print(f"No episodes found in source folder: {source_folder}")
        return
    
    print(f"Found {len(source_episodes)} episodes in source folder")
    
    # Start numbering from highest_dest + 1
    start_num = highest_dest + 1
    
    # Copy and rename episodes
    for i, (orig_num, orig_name, orig_path) in enumerate(source_episodes):
        new_num = start_num + i
        new_name = f"episode_{new_num:04d}"
        new_path = os.path.join(dest_folder, new_name)
        
        print(f"Copying {orig_name} -> {new_name}")
        
        try:
            shutil.copytree(orig_path, new_path)
            print(f"  ✅ Successfully copied to {new_path}")
        except Exception as e:
            print(f"  ❌ Error copying {orig_name}: {e}")
    
    print(f"\n🎉 Finished copying {len(source_episodes)} episodes!")
    print(f"Episodes renamed from episode_{start_num:04d} to episode_{start_num + len(source_episodes) - 1:04d}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy episodes from one folder to another with automatic renaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python copy_episodes_to_another_folder.py \\
    --source ~/humanoid/avp_teleop/teleop/utils/data/cubes_box_s_39 \\
    --dest ~/humanoid/avp_teleop/teleop/utils/data/cubes_box_single

  python copy_episodes_to_another_folder.py \\
    --source /path/to/source/episodes \\
    --dest /path/to/destination/episodes
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source folder containing episodes to copy"
    )
    
    parser.add_argument(
        "--dest", "-d", 
        required=True,
        help="Destination folder where episodes will be copied"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be copied")
        source_episodes = get_sorted_episodes(args.source)
        highest_dest = find_highest_episode_number(args.dest)
        start_num = highest_dest + 1
        
        print(f"Source: {args.source}")
        print(f"Destination: {args.dest}")
        print(f"Highest episode in destination: {highest_dest}")
        print(f"Would copy {len(source_episodes)} episodes:")
        
        for i, (orig_num, orig_name, _) in enumerate(source_episodes):
            new_num = start_num + i
            new_name = f"episode_{new_num:04d}"
            print(f"  {orig_name} -> {new_name}")
    else:
        copy_episodes(args.source, args.dest)


if __name__ == "__main__":
    main()
