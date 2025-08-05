"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
"""
import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.
        
        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = 'data.json'
        
        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key


    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []
        
        for task_path in glob.glob(os.path.join(self.data_dirs, '*')):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path, '*'))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)
        
        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))


    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)


    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, 'r', encoding='utf-8') as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached


    def _extract_ordered_state_data(self, episode_data: Dict) -> np.ndarray:
        """
        Extract state data in the correct order to create a properly structured state vector.
        
        Args:
            episode_data: Dictionary containing episode data
            
        Returns:
            Numpy array with shape (num_frames, state_dim) containing properly ordered state data
        """
        result = []
        sample_data = episode_data['data'][0]  # Check first frame for debugging
        
        for sample_data in episode_data['data']:
            state_vector = []
            
            # Extract in the correct order to match our 108D structure:
            # 0-6: left_arm qpos (7D)
            if 'left_arm' in sample_data['states'] and sample_data['states']['left_arm']:
                left_arm_qpos = sample_data['states']['left_arm'].get('qpos', [])
                state_vector.extend(left_arm_qpos)
            
            # 7-13: left_arm qvel (7D)
            if 'left_arm' in sample_data['states'] and sample_data['states']['left_arm']:
                left_arm_qvel = sample_data['states']['left_arm'].get('qvel', [])
                state_vector.extend(left_arm_qvel)
            
            # 14-20: left_arm torque (7D)
            if 'left_arm' in sample_data['states'] and sample_data['states']['left_arm']:
                left_arm_torque = sample_data['states']['left_arm'].get('torque', [])
                state_vector.extend(left_arm_torque)
            
            # 21-27: right_arm qpos (7D)
            if 'right_arm' in sample_data['states'] and sample_data['states']['right_arm']:
                right_arm_qpos = sample_data['states']['right_arm'].get('qpos', [])
                state_vector.extend(right_arm_qpos)
            
            # 28-34: right_arm qvel (7D)
            if 'right_arm' in sample_data['states'] and sample_data['states']['right_arm']:
                right_arm_qvel = sample_data['states']['right_arm'].get('qvel', [])
                state_vector.extend(right_arm_qvel)
            
            # 35-41: right_arm torque (7D)
            if 'right_arm' in sample_data['states'] and sample_data['states']['right_arm']:
                right_arm_torque = sample_data['states']['right_arm'].get('torque', [])
                state_vector.extend(right_arm_torque)
            
            # 42-48: left_hand qpos (7D)
            if 'left_hand' in sample_data['states'] and sample_data['states']['left_hand']:
                left_hand_qpos = sample_data['states']['left_hand'].get('qpos', [])
                state_vector.extend(left_hand_qpos)
            
            # 49-55: left_hand qvel (7D)
            if 'left_hand' in sample_data['states'] and sample_data['states']['left_hand']:
                left_hand_qvel = sample_data['states']['left_hand'].get('qvel', [])
                state_vector.extend(left_hand_qvel)
            
            # 56-62: left_hand torque (7D)
            if 'left_hand' in sample_data['states'] and sample_data['states']['left_hand']:
                left_hand_torque = sample_data['states']['left_hand'].get('torque', [])
                state_vector.extend(left_hand_torque)
            
            # 63-74: left_hand pressures (12D)
            if 'left_hand' in sample_data['states'] and sample_data['states']['left_hand']:
                left_hand_pressures = sample_data['states']['left_hand'].get('pressures', [])
                state_vector.extend(left_hand_pressures)
            
            # 75-81: right_hand qpos (7D)
            if 'right_hand' in sample_data['states'] and sample_data['states']['right_hand']:
                right_hand_qpos = sample_data['states']['right_hand'].get('qpos', [])
                state_vector.extend(right_hand_qpos)
            
            # 82-88: right_hand qvel (7D)
            if 'right_hand' in sample_data['states'] and sample_data['states']['right_hand']:
                right_hand_qvel = sample_data['states']['right_hand'].get('qvel', [])
                state_vector.extend(right_hand_qvel)
            
            # 89-95: right_hand torque (7D)
            if 'right_hand' in sample_data['states'] and sample_data['states']['right_hand']:
                right_hand_torque = sample_data['states']['right_hand'].get('torque', [])
                state_vector.extend(right_hand_torque)
            
            # 96-107: right_hand pressures (12D)
            if 'right_hand' in sample_data['states'] and sample_data['states']['right_hand']:
                right_hand_pressures = sample_data['states']['right_hand'].get('pressures', [])
                state_vector.extend(right_hand_pressures)
            
            # 108-109: camera qpos (2D) - if present
            if 'camera' in sample_data['states'] and sample_data['states']['camera']:
                camera_qpos = sample_data['states']['camera'].get('qpos', [])
                state_vector.extend(camera_qpos)
            
            result.append(np.array(state_vector, dtype=np.float32))
        
        result_array = np.array(result)
        
        # Debug print for first episode to verify structure
        if len(result) > 0:
            print(f"==> State extraction debug:")
            print(f"    Episode frames: {len(result)}")
            print(f"    State dimension: {result_array.shape[1] if len(result_array.shape) > 1 else len(result_array[0])}")
            
            # Print breakdown of first frame
            if len(result) > 0:
                first_frame = result[0]
                idx = 0
                sample = episode_data['data'][0]
                print(f"    State breakdown (first frame):")
                
                if 'left_arm' in sample['states'] and sample['states']['left_arm']:
                    qpos_len = len(sample['states']['left_arm'].get('qpos', []))
                    qvel_len = len(sample['states']['left_arm'].get('qvel', []))
                    torque_len = len(sample['states']['left_arm'].get('torque', []))
                    print(f"      left_arm: qpos[{idx}:{idx+qpos_len}], qvel[{idx+qpos_len}:{idx+qpos_len+qvel_len}], torque[{idx+qpos_len+qvel_len}:{idx+qpos_len+qvel_len+torque_len}]")
                    idx += qpos_len + qvel_len + torque_len
                
                if 'right_arm' in sample['states'] and sample['states']['right_arm']:
                    qpos_len = len(sample['states']['right_arm'].get('qpos', []))
                    qvel_len = len(sample['states']['right_arm'].get('qvel', []))
                    torque_len = len(sample['states']['right_arm'].get('torque', []))
                    print(f"      right_arm: qpos[{idx}:{idx+qpos_len}], qvel[{idx+qpos_len}:{idx+qpos_len+qvel_len}], torque[{idx+qpos_len+qvel_len}:{idx+qpos_len+qvel_len+torque_len}]")
                    idx += qpos_len + qvel_len + torque_len
                
                if 'left_hand' in sample['states'] and sample['states']['left_hand']:
                    qpos_len = len(sample['states']['left_hand'].get('qpos', []))
                    qvel_len = len(sample['states']['left_hand'].get('qvel', []))
                    torque_len = len(sample['states']['left_hand'].get('torque', []))
                    pressure_len = len(sample['states']['left_hand'].get('pressures', []))
                    print(f"      left_hand: qpos[{idx}:{idx+qpos_len}], qvel[{idx+qpos_len}:{idx+qpos_len+qvel_len}], torque[{idx+qpos_len+qvel_len}:{idx+qpos_len+qvel_len+torque_len}], pressures[{idx+qpos_len+qvel_len+torque_len}:{idx+qpos_len+qvel_len+torque_len+pressure_len}]")
                    idx += qpos_len + qvel_len + torque_len + pressure_len
                
                if 'right_hand' in sample['states'] and sample['states']['right_hand']:
                    qpos_len = len(sample['states']['right_hand'].get('qpos', []))
                    qvel_len = len(sample['states']['right_hand'].get('qvel', []))
                    torque_len = len(sample['states']['right_hand'].get('torque', []))
                    pressure_len = len(sample['states']['right_hand'].get('pressures', []))
                    print(f"      right_hand: qpos[{idx}:{idx+qpos_len}], qvel[{idx+qpos_len}:{idx+qpos_len+qvel_len}], torque[{idx+qpos_len+qvel_len}:{idx+qpos_len+qvel_len+torque_len}], pressures[{idx+qpos_len+qvel_len+torque_len}:{idx+qpos_len+qvel_len+torque_len+pressure_len}]")
                    idx += qpos_len + qvel_len + torque_len + pressure_len
                
                if 'camera' in sample['states'] and sample['states']['camera']:
                    camera_len = len(sample['states']['camera'].get('qpos', []))
                    print(f"      camera: qpos[{idx}:{idx+camera_len}]")
                    idx += camera_len
                
                print(f"    Total features extracted: {idx}")
        
        return result_array

    def _extract_ordered_action_data(self, episode_data: Dict) -> np.ndarray:
        """
        Extract action data in the correct order.
        
        Args:
            episode_data: Dictionary containing episode data
            
        Returns:
            Numpy array with shape (num_frames, action_dim) containing properly ordered action data
        """
        result = []
        
        for sample_data in episode_data['data']:
            action_vector = []
            
            # Extract actions in order: left_arm, right_arm, left_hand, right_hand, camera
            # 0-6: left_arm qpos (7D)
            if 'left_arm' in sample_data['actions'] and sample_data['actions']['left_arm']:
                left_arm_qpos = sample_data['actions']['left_arm'].get('qpos', [])
                action_vector.extend(left_arm_qpos)
            
            # 7-13: right_arm qpos (7D)
            if 'right_arm' in sample_data['actions'] and sample_data['actions']['right_arm']:
                right_arm_qpos = sample_data['actions']['right_arm'].get('qpos', [])
                action_vector.extend(right_arm_qpos)
            
            # 14-20: left_hand qpos (7D)
            if 'left_hand' in sample_data['actions'] and sample_data['actions']['left_hand']:
                left_hand_qpos = sample_data['actions']['left_hand'].get('qpos', [])
                action_vector.extend(left_hand_qpos)
            
            # 21-27: right_hand qpos (7D)
            if 'right_hand' in sample_data['actions'] and sample_data['actions']['right_hand']:
                right_hand_qpos = sample_data['actions']['right_hand'].get('qpos', [])
                action_vector.extend(right_hand_qpos)
            
            # 28-29: camera qpos (2D) - if present
            if 'camera' in sample_data['actions'] and sample_data['actions']['camera']:
                camera_qpos = sample_data['actions']['camera'].get('qpos', [])
                action_vector.extend(camera_qpos)
            
            result.append(np.array(action_vector, dtype=np.float32))
        
        result_array = np.array(result)
        
        # Debug print for first episode to verify action structure
        if len(result) > 0:
            print(f"==> Action extraction debug:")
            print(f"    Episode frames: {len(result)}")
            print(f"    Action dimension: {result_array.shape[1] if len(result_array.shape) > 1 else len(result_array[0])}")
            
            # Print breakdown of first frame
            if len(result) > 0:
                sample = episode_data['data'][0]
                idx = 0
                print(f"    Action breakdown (first frame):")
                
                if 'left_arm' in sample['actions'] and sample['actions']['left_arm']:
                    qpos_len = len(sample['actions']['left_arm'].get('qpos', []))
                    print(f"      left_arm: qpos[{idx}:{idx+qpos_len}]")
                    idx += qpos_len
                
                if 'right_arm' in sample['actions'] and sample['actions']['right_arm']:
                    qpos_len = len(sample['actions']['right_arm'].get('qpos', []))
                    print(f"      right_arm: qpos[{idx}:{idx+qpos_len}]")
                    idx += qpos_len
                
                if 'left_hand' in sample['actions'] and sample['actions']['left_hand']:
                    qpos_len = len(sample['actions']['left_hand'].get('qpos', []))
                    print(f"      left_hand: qpos[{idx}:{idx+qpos_len}]")
                    idx += qpos_len
                
                if 'right_hand' in sample['actions'] and sample['actions']['right_hand']:
                    qpos_len = len(sample['actions']['right_hand'].get('qpos', []))
                    print(f"      right_hand: qpos[{idx}:{idx+qpos_len}]")
                    idx += qpos_len
                
                if 'camera' in sample['actions'] and sample['actions']['camera']:
                    camera_len = len(sample['actions']['camera'].get('qpos', []))
                    print(f"      camera: qpos[{idx}:{idx+camera_len}]")
                    idx += camera_len
                
                print(f"    Total action features extracted: {idx}")
        
        return result_array

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]['colors'].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data['data']:
                relative_path = sample_data['colors'].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_item(self, index: Optional[int] = None,) -> Dict:
        """Get a training sample from the dataset."""
            
        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Extract properly ordered state and action data
        state = self._extract_ordered_state_data(episode_data)
        action = self._extract_ordered_action_data(episode_data)
        
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else action.shape[0]
        
        # Determine what features are available
        sample_states = episode_data['data'][0]['states']
        has_velocity = any(
            part_data and 'qvel' in part_data and part_data['qvel']
            for part_data in [
                sample_states.get('left_arm'),
                sample_states.get('right_arm'),
                sample_states.get('left_hand'),
                sample_states.get('right_hand')
            ]
            if part_data
        )
        
        has_pressure = any(
            part_data and 'pressures' in part_data and part_data['pressures']
            for part_data in [
                sample_states.get('left_hand'),
                sample_states.get('right_hand')
            ]
            if part_data
        )
        
        has_torque = any(
            part_data and 'torque' in part_data and part_data['torque']
            for part_data in [
                sample_states.get('left_arm'),
                sample_states.get('right_arm'),
                sample_states.get('left_hand'),
                sample_states.get('right_hand')
            ]
            if part_data
        )
        
        has_active_camera = any(
            part_data and 'qpos' in part_data and part_data['camera']
            for part_data in [
                sample_states.get('camera'),
            ]
            if part_data
        )

        # Load task description
        task = episode_data.get('text', {}).get('goal', "")
        
        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            'camera_names': list(cameras.keys()),
            'cam_height': cam_height,
            'cam_width': cam_width,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'has_velocity': has_velocity,
            'has_pressure': has_pressure,
            'has_torque': has_torque,
            'has_active_camera': has_active_camera 
        }
        
        print(f"==> Episode {index} configuration:")
        print(f"    State shape: {state.shape}")
        print(f"    Action shape: {action.shape}")
        print(f"    Features detected: velocity={has_velocity}, pressure={has_pressure}, torque={has_torque}")
        
        return {'episode_index': index,
                'episode_length': episode_length,
                'state': state,
                'action': action,
                'cameras': cameras,
                'task': task,
                'data_cfg': data_cfg}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_pressure: bool = False,
    has_torque: bool = False,
    has_active_camera: bool = False,
    action_dim: int = None,
    state_dim: int = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    # Create feature names for enriched state based on the actual order used in extraction
    state_names = []
    
    # For G1_Dex3, create feature names that match the extraction order in _extract_ordered_state_data
    if robot_type == "Unitree_G1_Dex3":
        # Left arm: qpos (7D) + qvel (7D) + torque (7D) = 21D
        for i in range(7):
            state_names.append(f"left_arm_qpos_{i}")
        if has_velocity:
            for i in range(7):
                state_names.append(f"left_arm_qvel_{i}")
        if has_torque:
            for i in range(7):
                state_names.append(f"left_arm_torque_{i}")
        
        # Right arm: qpos (7D) + qvel (7D) + torque (7D) = 21D
        for i in range(7):
            state_names.append(f"right_arm_qpos_{i}")
        if has_velocity:
            for i in range(7):
                state_names.append(f"right_arm_qvel_{i}")
        if has_torque:
            for i in range(7):
                state_names.append(f"right_arm_torque_{i}")
        
        # Left hand: qpos (7D) + qvel (7D) + torque (7D) + pressures (12D) = 33D
        for i in range(7):
            state_names.append(f"left_hand_qpos_{i}")
        if has_velocity:
            for i in range(7):
                state_names.append(f"left_hand_qvel_{i}")
        if has_torque:
            for i in range(7):
                state_names.append(f"left_hand_torque_{i}")
        if has_pressure:
            for i in range(12):
                state_names.append(f"left_hand_pressure_{i}")
        
        # Right hand: qpos (7D) + qvel (7D) + torque (7D) + pressures (12D) = 33D
        for i in range(7):
            state_names.append(f"right_hand_qpos_{i}")
        if has_velocity:
            for i in range(7):
                state_names.append(f"right_hand_qvel_{i}")
        if has_torque:
            for i in range(7):
                state_names.append(f"right_hand_torque_{i}")
        if has_pressure:
            for i in range(12):
                state_names.append(f"right_hand_pressure_{i}")
        
        # Camera: qpos (2D) - always include for G1_Dex3 (110D total)
        if has_active_camera:
            for i in range(2):
                state_names.append(f"camera_qpos_{i}")
    else:
        # For other robot types, use original logic
        # Add position names (qpos)
        state_names.extend([f"{motor}_pos" for motor in motors])
        
        # Add velocity names (qvel) if available
        if has_velocity:
            state_names.extend([f"{motor}_vel" for motor in motors])
        
        # Add pressure names if available
        if has_pressure:
            # Add pressure sensors for hands (12 per hand for dex3 hands)
            state_names.extend([
                f"left_hand_pressure_{i}" for i in range(12)
            ])
            state_names.extend([
                f"right_hand_pressure_{i}" for i in range(12)
            ])
    
    # Calculate state dimension from the actual feature names
    if state_dim is None:
        state_dim = len(state_names)

    # Create action feature names that match the extraction order in _extract_ordered_action_data
    action_names = []
    if robot_type == "Unitree_G1_Dex3":
        # Action order: left_arm (7D) + right_arm (7D) + left_hand (7D) + right_hand (7D) + camera (2D)
        for i in range(7):
            action_names.append(f"left_arm_qpos_{i}")
        for i in range(7):
            action_names.append(f"right_arm_qpos_{i}")
        for i in range(7):
            action_names.append(f"left_hand_qpos_{i}")
        for i in range(7):
            action_names.append(f"right_hand_qpos_{i}")
        # Always include camera for G1_Dex3 (30D total)
        if has_active_camera:
            for i in range(2):
                action_names.append(f"camera_qpos_{i}")
        action_dim = len(action_names)
    else:
        # For other robot types, use motor names
        action_names = motors
        action_dim = len(motors)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }

    if has_effort:
        if robot_type == "Unitree_G1_Dex3":
            effort_names = []
            for i in range(7):
                effort_names.append(f"left_arm_torque_{i}")
            for i in range(7):
                effort_names.append(f"right_arm_torque_{i}")
            for i in range(7):
                effort_names.append(f"left_hand_torque_{i}")
            for i in range(7):
                effort_names.append(f"right_hand_torque_{i}")
            features["observation.effort"] = {
                "dtype": "float32",
                "shape": (len(effort_names),),
                "names": effort_names,
            }
        else:
            features["observation.effort"] = {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [f"{motor}_effort" for motor in motors],
            }


    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
) -> LeRobotDataset:

    json_dataset = JsonDataset(raw_dir, robot_type)
    
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]  # Already contains all sensor data combined
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,        # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    # Create a temporary dataset instance to detect available features
    temp_json_dataset = JsonDataset(raw_dir, robot_type)
    temp_episode = temp_json_dataset.get_item(0)
    data_cfg = temp_episode["data_cfg"]
    
    # Detect available features
    has_velocity = data_cfg.get('has_velocity', False)
    has_pressure = data_cfg.get('has_pressure', False)
    has_torque = data_cfg.get('has_torque', False)
    state_dim = data_cfg.get('state_dim', None)
    
    print(f"Detected features - Velocity: {has_velocity}, Pressure: {has_pressure}, Torque: {has_torque}")
    print(f"State dimension: {state_dim}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=has_velocity,
        has_pressure=has_pressure,
        has_torque=has_torque,
        state_dim=state_dim,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder = True)


def local_push_to_hub(
        repo_id: str,
        root_path: Path,):

    dataset = LeRobotDataset(repo_id = repo_id, root = root_path)
    dataset.push_to_hub(upload_large_folder = True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
