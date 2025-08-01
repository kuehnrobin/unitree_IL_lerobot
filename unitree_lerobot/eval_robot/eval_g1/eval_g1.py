''''
Copyright Robin Kühn 2025
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import time
import torch
import logging
import threading
import numpy as np
import signal
import sys

import tty
import termios

import json
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from multiprocessing import shared_memory, Array, Lock
from pathlib import Path

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import predict_action

from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from unitree_lerobot.eval_robot.eval_g1.robot_control.active_head_cam import ActiveCameraController
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig
from unitree_lerobot.eval_robot.eval_g1.utils import EpisodeWriter

# Apply feature filtering to get the correct metadata dimensions
from unitree_lerobot.lerobot.lerobot.common.datasets.feature_filter import create_filtered_dataset_wrapper
from unitree_lerobot.lerobot.lerobot.configs.train import FeatureSelectionConfig

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutdown requested via signal, will exit gracefully...")
    logging.info("Shutdown requested via signal, will exit gracefully...")
    # Force exit if called multiple times
    if signal_handler.call_count > 0:
        print("Force exit requested")
        sys.exit(1)
    signal_handler.call_count += 1

# Initialize call counter
signal_handler.call_count = 0

# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    cfg: EvalRealConfig,
    policy_config_info: dict,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    # Check which cameras are needed based on policy config - uniform handling
    policy_cameras = policy_config_info.get('cameras', [])
  
    # Check which camera types are required by the policy uniformly
    required_camera_types = policy_config_info.get('camera_types', {})
    
    # Extract individual flags for backward compatibility
    use_head_cameras = required_camera_types.get('head', False)
    use_wrist_cameras = required_camera_types.get('wrist', False)
    use_active_camera = required_camera_types.get('active', False)
    
    logging.info(f"Camera requirements based on policy config:")
    logging.info(f"  - Policy cameras: {policy_cameras}")
    logging.info(f"  - Required camera types: {required_camera_types}")
    logging.info(f"  - Head cameras: {use_head_cameras}")
    logging.info(f"  - Wrist cameras: {use_wrist_cameras}")
    logging.info(f"  - Active cameras: {use_active_camera}")

    # Initialize shared memory based on what cameras are needed (compatible with ImageClient)
    head_cam_img_shm = None
    head_cam_img_array = None
    tv_img_shape = None
    
    if use_head_cameras:
        # Head camera setup - use standard recording resolution like in teleop script
        tv_img_shape = (480, 1280, 3)  # Standard head camera recording resolution
        head_cam_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
        head_cam_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=head_cam_img_shm.buf)
        logging.info(f"Initialized head camera shared memory with shape {tv_img_shape}")

    # Initialize active camera shared memory if needed
    active_cam_img_shm = None
    active_cam_img_array = None
    active_cam_img_shape = None
    
    if use_active_camera:
        # Active camera setup - use standard recording resolution like in teleop script  
        active_cam_img_shape = (480, 1280, 3)  # Standard active camera recording resolution
        active_cam_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(active_cam_img_shape) * np.uint8().itemsize)
        active_cam_img_array = np.ndarray(active_cam_img_shape, dtype=np.uint8, buffer=active_cam_img_shm.buf)
        logging.info(f"Initialized active camera shared memory with shape {active_cam_img_shape}")

    # Initialize wrist camera shared memory if needed  
    wrist_img_shm = None
    wrist_img_array = None
    wrist_img_shape = None
    
    if use_wrist_cameras:
        # Wrist camera setup - use standard recording resolution like in teleop script
        wrist_img_shape = (480, 1280, 3)  # Standard wrist camera recording resolution (2 cameras side by side)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        logging.info(f"Initialized wrist camera shared memory with shape {wrist_img_shape}")
    
    # Set legacy flags for ImageClient compatibility
    BINOCULAR = use_head_cameras and head_cam_img_shm is not None
    WRIST = use_wrist_cameras and wrist_img_shm is not None
    
    # Initialize ImageClient with dynamic parameters based on available cameras
    img_client_params = {}
    
    # Add parameters for each camera type that's available
    if head_cam_img_shm is not None:
        img_client_params.update({
            'head_cam_img_shape': tv_img_shape,
            'head_cam_img_shm_name': head_cam_img_shm.name
        })
    
    if wrist_img_shm is not None:
        img_client_params.update({
            'wrist_img_shape': wrist_img_shape,
            'wrist_img_shm_name': wrist_img_shm.name
        })
    
    if active_cam_img_shm is not None:
        img_client_params.update({
            'active_cam_img_shape': active_cam_img_shape,
            'active_cam_img_shm_name': active_cam_img_shm.name,
            'use_active_camera': True
        })
    else:
        img_client_params['use_active_camera'] = False
    
    # Check if we have any cameras to initialize
    if not img_client_params:
        raise ValueError("No cameras are required by the policy - cannot initialize ImageClient")
    
    img_client = ImageClient(**img_client_params)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    robot_config = {
        'arm_type': 'g1',
        'hand_type': "dex3",
    }

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]
    to_idx = dataset.episode_data_index["to"][0].item()

    # arm
    arm_ctrl = G1_29_ArmController(networkInterface=cfg.cyclonedds_uri)
    
    # Apply speed control settings
    if cfg.arm_speed is not None:
        arm_ctrl.arm_velocity_limit = cfg.arm_speed
        logging.info(f"Setting custom arm velocity limit: {cfg.arm_speed}")
    
    init_left_arm_pose = step['observation.state'][:14].cpu().numpy()

    # hand
    if robot_config['hand_type'] == "dex3":
        left_hand_array = Array('d', 7, lock = True)          # [input]
        right_hand_array = Array('d', 7, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=cfg.cyclonedds_uri, force=cfg.force)
        # For 80D state (qpos+qvel+pressure), extract hand poses from qpos portion (first 28 dims)
        # TODO Was macht das hier?
        init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
        init_right_hand_pose = step['observation.state'][21:28].cpu().numpy()

    elif robot_config['hand_type'] == "gripper":
        left_hand_array = Array('d', 1, lock=True)             # [input]
        right_hand_array = Array('d', 1, lock=True)            # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, networkInterface=cfg.cyclonedds_uri)
        init_left_hand_pose = step['observation.state'][14].cpu().numpy()
        init_right_hand_pose = step['observation.state'][15].cpu().numpy()
        hand_ctrl = None  # No pressure data available for gripper
    else:
        hand_ctrl = None

    # active camera controller
    camera_controller = None
    init_camera_pose = None
    
    
    if use_active_camera:
        try:
            logging.info("Initializing active camera controller...")
            camera_controller = ActiveCameraController(
                port=cfg.camera_port,
                safe_mode=cfg.camera_safe_mode,
                max_movement_deg=cfg.camera_max_movement,
                logger=logging.getLogger('ActiveCameraController')
            )
            init_camera_pose = np.array([195.0 * np.pi / 180, 90.0 * np.pi / 180])
            logging.info(f"Initial camera positions from dataset: Pitch={np.rad2deg(init_camera_pose[0]):.2f}°, Yaw={np.rad2deg(init_camera_pose[1]):.2f}°")
        except Exception as e:
            logging.error(f"Failed to initialize active camera: {e}")
            camera_controller = None
    else:
        logging.info("Policy doesn't use active camera - camera controller disabled")

    #===============init robot=====================
    # Initialize recorder only if recording is enabled
    if cfg.record:
        recorder = EpisodeWriter(task_dir=cfg.task_dir, frequency=cfg.frequency, rerun_log=True)
        logging.info(f"Episode recorder initialized with task_dir={cfg.task_dir}")
        logging.info("Recording will use terminal controls - press 's' to start recording episodes")
        
    if cfg.pressure and hand_ctrl and robot_config['hand_type'] == "dex3":
        # TODO force in hand_torque umbenennen. Aktuell irreführend.
        hand_torque_mode = "enabled" if cfg.force else "disabled"
        logging.info(f"Pressure sensor data collection enabled via hand controller (torque mode: {hand_torque_mode})")
    else:
        logging.info("Pressure sensor data collection disabled")
    
    print("Please press 's' to start the subsequent program (no Enter needed):")
    
    # Non-blocking input for initial start
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            char = sys.stdin.read(1)
            if char == '\x03':  # Ctrl+C
                print("\nExiting...")
                sys.exit(0)
            elif char.lower() == 's':
                print("\nStarting program...")
                break
            elif char == '\r' or char == '\n':
                continue
            else:
                print(f"\nPress 's' to start (pressed: {repr(char)})")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    # Now continue with the program
    if True:

        # Apply gradual speed increase if not disabled
        if not cfg.no_gradual_speed:
            arm_ctrl.speed_gradual_max()
            logging.info("Gradual speed increase enabled")

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print("init robot pose")
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        left_hand_array[:] = init_left_hand_pose
        right_hand_array[:] = init_right_hand_pose
        
        # Initialize camera to initial position if available
        if camera_controller and camera_controller.connected and init_camera_pose is not None:
            try:
                camera_controller._set_target_positions(init_camera_pose[0], init_camera_pose[1])
                logging.info(f"Camera initialized to: Pitch={np.rad2deg(init_camera_pose[0]):.2f}°, Yaw={np.rad2deg(init_camera_pose[1]):.2f}°")
            except Exception as e:
                logging.error(f"Error initializing camera position: {e}")

        print("wait robot to pose")
        time.sleep(1)

        frequency = 50.0
        frame_counter = 0
        last_instruction_time = time.time()
        
        # Add debugging variables
        camera_selection_logged = False  # Track if camera selection has been logged once
        last_state_log_time = time.time()  # For periodic state vector logging

        # Add terminal input thread for recording controls
        if cfg.record:
            # Use Array for thread-safe recording state
            recording_state = Array('i', [0])  # 0 = not recording, 1 = recording
            
            def terminal_input_handler():
                """Clean terminal input handler for recording controls"""
                import sys
                import tty
                import termios
                import fcntl
                import os
                
                global shutdown_requested
                nonlocal recording_state
                
                # Print initial instructions
                print("\n" + "="*50)
                print("🎬 RECORDING CONTROLS ACTIVE")
                print("="*50)
                print("s - Start recording episode")
                print("r - Abort current recording") 
                print("q - Save recording as optimal")
                print("w - Save recording as suboptimal")
                print("e - Save recording as recovery")
                print("x - Exit program")
                print("Ctrl+C - Exit program")
                print("="*50)
                print("Ready for commands...")
                
                # Save terminal settings
                old_settings = None
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())
                    
                    # Make stdin non-blocking
                    fd = sys.stdin.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    
                    while not shutdown_requested:
                        try:
                            char = sys.stdin.read(1)
                            
                            if char == '\x03':  # Ctrl+C
                                shutdown_requested = True
                                print("\n⚠️  Shutdown requested...")
                                break
                            elif char == 'x':  # Exit with 'x' key
                                shutdown_requested = True
                                print("\n⚠️  Exit requested via 'x' key...")
                                break
                            elif char == 's' and recording_state[0] == 0:
                                if recorder.create_episode():
                                    recording_state[0] = 1
                                    print("\n🔴 Recording started!")
                                else:
                                    print("\n❌ Failed to start recording")
                            elif char == 'r' and recording_state[0] == 1:
                                recorder.abort_episode()
                                recording_state[0] = 0
                                print("\n⏹️  Recording aborted!")
                            elif recording_state[0] == 1 and char in ['q', 'w', 'e']:
                                quality_map = {'q': 'optimal', 'w': 'suboptimal', 'e': 'recovery'}
                                quality = quality_map[char]
                                recorder.save_episode(quality=quality)
                                recording_state[0] = 0
                                print(f"\n✅ Recording saved as {quality.upper()}!")
                            elif char and ord(char) >= 32:  # Only show message for printable characters
                                print(f"\n⚠️  Unknown key '{char}' - use s/r/q/w/e/x")
                        except IOError:
                            # No input available, continue
                            pass
                        except Exception as e:
                            print(f"\n⚠️  Input error: {e}")
                            logging.error(f"Terminal input error: {e}")
                        
                        time.sleep(0.1)  # Small delay to prevent busy waiting
                    
                    # Restore blocking mode
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags)
                        
                except Exception as e:
                    logging.error(f"Terminal input handler error: {e}")
                finally:
                    # Restore terminal settings
                    if old_settings:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        except:
                            pass
            
            # Start terminal input handler in daemon thread
            terminal_thread = threading.Thread(target=terminal_input_handler, daemon=True)
            terminal_thread.start()
            logging.info("Terminal recording controls started")
        else:
            recording_state = None

        try:
            logging.info("Starting main evaluation loop")
            while not shutdown_requested:

                # Build observation dict using individual camera variables
                observation = {}
                
                # Get images from available shared memory arrays
                current_head_image = head_cam_img_array.copy() if head_cam_img_array is not None else None
                current_wrist_image = wrist_img_array.copy() if wrist_img_array is not None else None
                current_active_image = active_cam_img_array.copy() if active_cam_img_array is not None else None
                
                # Build available images mapping
                available_images = {}
                
                # Process head cameras
                if current_head_image is not None and BINOCULAR:
                    available_images['cam_left_high'] = current_head_image[:, :tv_img_shape[1] // 2]
                    available_images['cam_right_high'] = current_head_image[:, tv_img_shape[1] // 2:]
                elif current_head_image is not None:
                    available_images['cam_left_high'] = current_head_image
                
                # Process wrist cameras  
                if current_wrist_image is not None:
                    available_images['cam_left_wrist'] = current_wrist_image[:, :wrist_img_shape[1] // 2]
                    available_images['cam_right_wrist'] = current_wrist_image[:, wrist_img_shape[1] // 2:]
                
                # Process active cameras
                if current_active_image is not None:
                    available_images['cam_left_active'] = current_active_image[:, :active_cam_img_shape[1] // 2]
                    available_images['cam_right_active'] = current_active_image[:, active_cam_img_shape[1] // 2:]
                
                # Apply training-based camera filtering (matching training pipeline)
                filtered_cameras = []
                all_available_cameras = list(available_images.keys())
                
                if cfg.feature_selection.cameras is not None:
                    # Use only specified cameras (include list from training config)
                    filtered_cameras = [cam for cam in cfg.feature_selection.cameras if cam in all_available_cameras]
                    if len(filtered_cameras) != len(cfg.feature_selection.cameras):
                        missing = set(cfg.feature_selection.cameras) - set(all_available_cameras)
                        logging.warning(f"Training-specified cameras not available: {missing}")
                elif cfg.feature_selection.exclude_cameras is not None:
                    # Use all cameras except excluded ones (from training config)
                    filtered_cameras = [cam for cam in all_available_cameras if cam not in cfg.feature_selection.exclude_cameras]
                    logging.info(f"Excluding cameras per training config: {cfg.feature_selection.exclude_cameras}")
                else:
                    # Use all available cameras
                    filtered_cameras = all_available_cameras
                
                # Cross-check with policy expectations
                policy_cameras = policy_config_info.get('cameras', [])
                if policy_cameras:
                    # Ensure our filtered cameras match what the policy expects
                    expected_but_missing = set(policy_cameras) - set(filtered_cameras)
                    filtered_but_unexpected = set(filtered_cameras) - set(policy_cameras)
                    
                    if expected_but_missing:
                        logging.warning(f"Policy expects cameras that are filtered out: {expected_but_missing}")
                    if filtered_but_unexpected:
                        logging.warning(f"Training config includes cameras that policy doesn't expect: {filtered_but_unexpected}")
                        
                    # Use intersection to be safe
                    final_cameras = [cam for cam in filtered_cameras if cam in policy_cameras]
                else:
                    # No policy camera info, trust training config filtering
                    final_cameras = filtered_cameras
                
                # Log camera selection only once
                if not camera_selection_logged:
                    # logging.info(f"🔍 CAMERA SELECTION: available={all_available_cameras}, CLI-filtered={filtered_cameras}, final={final_cameras}")
                    camera_selection_logged = True
                
                # Add filtered cameras to observation
                for camera_name in final_cameras:
                    if camera_name in available_images and available_images[camera_name] is not None:
                        observation[f"observation.images.{camera_name}"] = torch.from_numpy(available_images[camera_name])
                        # if frame_counter % 300 == 0:  # Log every 10 seconds
                        #     logging.debug(f"Added camera: {camera_name}")
                    else:
                        logging.warning(f"Camera {camera_name} selected but not available!")
            
                # Log which cameras are being used
                if len(observation) == 0:
                    logging.error("No cameras added to observation!")
                    logging.error(f"Available images: {list(available_images.keys())}")
                    logging.error(f"Training camera settings: cameras={cfg.feature_selection.cameras}, exclude_cameras={cfg.feature_selection.exclude_cameras}")
                    raise RuntimeError("No valid camera observations available!")
                else:
                    if frame_counter % 600 == 0:  # Log every 20 seconds instead of every frame
                        logging.info(f"Using cameras: {list(observation.keys())}")

                # Get camera positions if active camera is enabled by policy
                if use_active_camera and camera_controller and camera_controller.connected:
                    servo_states = camera_controller.get_servo_states()
                    current_camera_q = np.array([servo_states['current_pitch'], servo_states['current_yaw']])  # 2D
                elif use_active_camera:
                    # Use default positions if camera not available but expected by policy
                    current_camera_q = np.array([195.0 * np.pi / 180, 90.0 * np.pi / 180])  # Default safe positions
                else:
                    current_camera_q = None

                # Build observation state using training feature selection (loaded from train_config.json)
                # --------------------------------------
                # Get state_data
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()  # 14D
                if cfg.feature_selection.use_joint_velocities:
                    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()  # 14D velocity
                if cfg.feature_selection.use_joint_torques:
                    current_lr_arm_tau = arm_ctrl.get_current_dual_arm_tau()  # 14D torques
                if cfg.feature_selection.use_pressure_sensors and hand_ctrl:
                    try:
                        pressure_data = hand_ctrl.get_pressure_data() # 24 D
                        left_hand_pressure = np.array(pressure_data['left_pressure'])  # 12D
                        right_hand_pressure = np.array(pressure_data['right_pressure'])  # 12D
                    except Exception as e:
                        logging.error(f"Failed to get pressure data: {e}")
                        left_hand_pressure = np.zeros(12)
                        right_hand_pressure = np.zeros(12)

                state_components = []
                feature_log = []
                # --------------
                # 1. left_arm_qpos 0-7
                state_components.append(current_lr_arm_q[0:7])  # 7D
                feature_log.append(f"arm_positions: {current_lr_arm_q.shape[0]}D")
                # 2. left_arm_qvel 7-14 if enabled in training config
                if cfg.feature_selection.use_joint_velocities:
                    state_components.append(current_lr_arm_dq[0:7])  # +7D arm velocities
                    feature_log.append(f"arm_velocities: {current_lr_arm_dq.shape[0]}D")

                # 3. left_arm_torques 14-21 if enabled in training config
                if cfg.feature_selection.use_joint_torques:
                    state_components.append(current_lr_arm_tau[0:7])  # +7D arm torques
                    feature_log.append(f"arm_torques: {current_lr_arm_tau.shape[0]}D")
                
                # 4. right_arm_qpos 21-28
                state_components.append(current_lr_arm_q[7:14])  # 7D

                # 5. right_arm_qvel 28-35 if enabled in training config
                if cfg.feature_selection.use_joint_velocities:
                    state_components.append(current_lr_arm_dq[7:14])
                
                # 6. right_arm_torques 35-42 if enabled in training config
                if cfg.feature_selection.use_joint_torques:
                    state_components.append(current_lr_arm_tau[7:14])
                
                # 7. left_hand_qpos 42-49
                if robot_config['hand_type'] == "dex3":
                    state_components.append(np.array(dual_hand_state_array[0:7]))  # 7D
                elif robot_config['hand_type'] == "gripper":
                    state_components.append(np.array([dual_gripper_state_array[1]]))  # 1D
                
                # 8. left_hand_pressure 49-61 if enabled in training config
                if cfg.feature_selection.use_pressure_sensors and robot_config['hand_type'] == "dex3":
                    state_components.append(np.array(left_hand_pressure))  # 12D
                
                # 9. right_hand_qpos 61-68
                if robot_config['hand_type'] == "dex3":
                    state_components.append(np.array(dual_hand_state_array[7:14]))
                elif robot_config['hand_type'] == "gripper":
                    state_components.append(np.array([dual_gripper_state_array[0]]))    
                # 10. right_hand_pressure 68-80 if enabled in training config
                if cfg.feature_selection.use_pressure_sensors and robot_config['hand_type'] == "dex3":
                    state_components.append(np.array(right_hand_pressure))  # 12D
                
                # 11. camera positions 80-82 if active camera is used in policy
                if current_camera_q is not None:
                    state_components.append(current_camera_q)  # 2D
                    feature_log.append(f"camera_positions: {current_camera_q.shape[0]}D")
                
                
                # Concatenate all state components
                observation_state = np.concatenate(state_components)
                # Print observation_state every 30 seconds
                if time.time() - last_state_log_time > 10:
                    logging.info(f"Current left_hand_state: {observation_state[14:21]}")
                    logging.info(f"Current right_hand_state: {observation_state[33:40]}")
                    last_state_log_time = time.time()
                #------------------
                                
                # Verify state dimension matches policy expectation
                expected_state_dim = policy_config_info.get('state_dim')
                actual_state_dim = observation_state.shape[0]
                
                if expected_state_dim and actual_state_dim != expected_state_dim:
                    logging.error(f"State dimension mismatch! Expected: {expected_state_dim}, Got: {actual_state_dim}")
                    logging.error(f"State components: {[comp.shape for comp in state_components]}")
                    logging.error(f"Training feature selection settings:")
                    logging.error(f"  - cameras: {cfg.feature_selection.cameras}")
                    logging.error(f"  - exclude_cameras: {cfg.feature_selection.exclude_cameras}")
                    logging.error(f"  - use_joint_positions: {cfg.feature_selection.use_joint_positions}")
                    logging.error(f"  - use_joint_velocities: {cfg.feature_selection.use_joint_velocities}")
                    logging.error(f"  - use_joint_torques: {cfg.feature_selection.use_joint_torques}")
                    logging.error(f"  - use_pressure_sensors: {cfg.feature_selection.use_pressure_sensors}")
                    logging.error(f"These settings were automatically loaded from train_config.json!")
                    logging.error(f"Check the train_config.json file in the policy directory for the exact configuration.")
                    raise RuntimeError(f"State dimension mismatch: expected {expected_state_dim}, got {actual_state_dim}")
                                
                observation["observation.state"] = torch.from_numpy(observation_state).float()

                observation = {
                    key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
                }

                action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                action = action.cpu().numpy()
                # Print action every 30 seconds
                if time.time() - last_state_log_time > 10:
                    logging.info(f"Current left_hand_action: {action[14:21]}")
                    logging.info(f"Current right_hand_action: {action[21:28]}")
                                
                if cfg.record and time.time() - last_instruction_time > 30:
                    print("\n📝 RECORDING CONTROLS:")
                    print("   s = Start | r = Abort | q = Save optimal | w = Save suboptimal | e = Save recovery | x = Exit")
                    last_instruction_time = time.time()

                # Record data if recording is active
                if cfg.record and recording_state and recording_state[0] == 1:
                    # Prepare data for recording using individual camera variables
                    colors = {}
                    depths = {}
                    
                    color_index = 0
                    
                    # Handle head cameras
                    if current_head_image is not None:
                        if BINOCULAR:
                            colors[f"color_{color_index}"] = current_head_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{color_index+1}"] = current_head_image[:, tv_img_shape[1]//2:]
                            color_index += 2
                        else:
                            colors[f"color_{color_index}"] = current_head_image
                            color_index += 1
                    
                    # Handle wrist cameras
                    if current_wrist_image is not None:
                        colors[f"color_{color_index}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                        colors[f"color_{color_index+1}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        color_index += 2
                    
                    # Handle active cameras
                    if current_active_image is not None:
                        colors[f"color_{color_index}"] = current_active_image[:, :active_cam_img_shape[1]//2]
                        colors[f"color_{color_index+1}"] = current_active_image[:, active_cam_img_shape[1]//2:]
                        color_index += 2
                    
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = action[:7]
                    right_arm_action = action[7:14]

                    # hand action and camera action (split based on hand type and camera availability)
                    if use_active_camera:
                        if robot_config['hand_type'] == "dex3":
                            left_hand_action = action[14:21]
                            right_hand_action = action[21:28]
                            camera_action = action[28:30]  # Last 2 values
                        elif robot_config['hand_type'] == "gripper":
                            left_hand_action = [action[14]]
                            right_hand_action = [action[15]]
                            camera_action = action[16:18]  # Last 2 values
                    else:
                        # No camera, use original split
                        if robot_config['hand_type'] == "dex3":
                            left_hand_action = action[14:21]
                            right_hand_action = action[21:28]
                        elif robot_config['hand_type'] == "gripper":
                            left_hand_action = [action[14]]
                            right_hand_action = [action[15]]
                        camera_action = current_camera_q  # Use current positions if no camera control
                    
                    # Get pressure sensor data if available from hand controller
                    if cfg.pressure and hand_ctrl and robot_config['hand_type'] == "dex3":
                        pressure_data = hand_ctrl.get_pressure_data()
                    else:
                        pressure_data = {
                            'left_pressure': [], 'left_temp': [], 'right_pressure': [], 'right_temp': []
                        }
                    
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    
                            "qvel":   [],  #TODO: Add velocity data                        
                            "torque": [],       
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],  #TODO: Add velocity data                        
                            "torque": [],                         
                        },                        
                        "left_hand": {                                                                    
                            "qpos":   dual_hand_state_array[0:7],           
                            "qvel":   [],                           
                            "torque": [], 
                            "pressures": pressure_data['left_pressure'],
                            #"temperatures": pressure_data['left_temp'],  TODO: Testen, denke aber nicht dass das einen großen einfluss hat           
                        }, 
                        "right_hand": {                                                                    
                            "qpos":    dual_hand_state_array[7:14],       
                            "qvel":   [],                           
                            "torque": [],
                            "pressures": pressure_data['right_pressure'],
                            #"temperatures": pressure_data['right_temp'],  
                        },
                        "camera": {
                            "qpos": current_camera_q.tolist(),
                            "qvel": [],
                            "torque": []
                        },
                        "body": None, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_hand": {                                   
                            "qpos":   left_hand_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_hand": {                                   
                            "qpos":   right_hand_action.tolist(),       
                            "qvel":   [],       
                            "torque": [], 
                        },
                        "camera": {
                            "qpos": camera_action.tolist() if isinstance(camera_action, np.ndarray) else camera_action,
                            "qvel": [],
                            "torque": []
                        },
                        "body": None, 
                    }
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)
                
                # exec action
                # Split actions dynamically based on policy configuration
                expected_action_dim = policy_config_info.get('action_dim', 30)
                actual_action_dim = action.shape[0]
                
                if actual_action_dim != expected_action_dim:
                    logging.warning(f"Action dimension mismatch! Expected: {expected_action_dim}, Got: {actual_action_dim}")
                
                # Always start with arm actions (first 14)
                arm_action = action[:14]
                
                if robot_config['hand_type'] == "dex3":
                    hand_action = action[14:28]  # Next 14 for dex3 hands
                    action_cursor = 28
                elif robot_config['hand_type'] == "gripper":
                    hand_action = action[14:16]  # Next 2 for grippers 
                    action_cursor = 16
                else:
                    hand_action = None
                    action_cursor = 14
                
                # Check if camera actions are included based on policy config
                if use_active_camera and camera_controller and camera_controller.connected:
                    if action_cursor + 2 <= actual_action_dim:
                        camera_action = action[action_cursor:action_cursor+2]  # Last 2 values are camera pitch/yaw
                        
                        # Execute camera action using active camera controller
                        try:
                            target_pitch, target_yaw = camera_action
                            camera_controller._set_target_positions(target_pitch, target_yaw)
                            # if frame_counter % 150 == 0:  # Log every 5 seconds
                            #     logging.info(f"Camera targets: Pitch={np.rad2deg(target_pitch):.1f}°, Yaw={np.rad2deg(target_yaw):.1f}°")
                        except Exception as e:
                            logging.error(f"Error controlling camera: {e}")
                    else:
                        logging.warning(f"Action dimension too small for camera control. Expected at least {action_cursor+2}, got {actual_action_dim}")
                elif use_active_camera:
                    logging.debug("Active camera expected by policy but not connected")
                else:
                    logging.debug("No active camera actions in policy")
                
                # Execute arm and hand actions
                arm_ctrl.ctrl_dual_arm(arm_action, np.zeros(14))
                if robot_config['hand_type'] == "dex3":
                    left_hand_array[:] = action[14:21] #hand_action[:7]
                    right_hand_array[:] = action[21:28] #hand_action[7:]
                elif robot_config['hand_type'] == "gripper":
                    left_hand_array[:] = hand_action[0]
                    right_hand_array[:] = hand_action[1]
            
                frame_counter += 1
            
            # Log performance info periodically (reduced frequency for performance)
            if frame_counter % 600 == 0:  # Every ~12 seconds at 50fps instead of every 6 seconds
                logging.info(f"Evaluation running - frame {frame_counter}, arm velocity limit: {arm_ctrl.arm_velocity_limit:.2f}")
        
            time.sleep(1/frequency)

        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt, exiting program...")
        except Exception as e:
            logging.exception(f"Error in main loop: {e}")
        finally:
            # Clean up
            if cfg.record:
                if 'recorder' in locals():
                    recorder.close()
                logging.info("Recording resources cleaned up")
            
            # Cleanup camera controller
            if camera_controller:
                try:
                    camera_controller.disconnect()
                    logging.info("Camera controller disconnected")
                except Exception as e:
                    logging.error(f"Error disconnecting camera controller: {e}")
                        
            arm_ctrl.ctrl_dual_arm_go_home()
            logging.info("Arms returned to home position")
            
            # Clean up shared memory resources individually
            try:
                if head_cam_img_shm is not None:
                    head_cam_img_shm.unlink()
                    head_cam_img_shm.close()
                    logging.info("Head camera shared memory cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up head camera shared memory: {e}")
                
            try:
                if wrist_img_shm is not None:
                    wrist_img_shm.unlink()
                    wrist_img_shm.close()
                    logging.info("Wrist camera shared memory cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up wrist camera shared memory: {e}")
                
            try:
                if active_cam_img_shm is not None:
                    active_cam_img_shm.unlink()
                    active_cam_img_shm.close()
                    logging.info("Active camera shared memory cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up active camera shared memory: {e}")
                
            logging.info("Resources cleaned up, exiting program")


def extract_config_from_policy(policy):
    """Extract configuration information from the policy's config.json"""
    config_info = {
        'cameras': [],
        'state_dim': None,
        'action_dim': None,
        'camera_types': {
            'head': False,
            'wrist': False, 
            'active': False
        }
    }
    
    # Define camera type mappings for uniform detection
    camera_type_mappings = {
        'head': ['cam_left_high', 'cam_right_high'],
        'wrist': ['cam_left_wrist', 'cam_right_wrist'],
        'active': ['cam_left_active', 'cam_right_active']
    }
    
    # Extract camera information from input_features
    if hasattr(policy, 'config') and hasattr(policy.config, 'input_features'):
        input_features = policy.config.input_features
        
        # Extract cameras from input features
        for feature_name in input_features.keys():
            if 'observation.images.' in feature_name:
                camera_name = feature_name.replace('observation.images.', '')
                config_info['cameras'].append(camera_name)
        
        # Determine which camera types are used uniformly
        for camera_type, camera_names in camera_type_mappings.items():
            config_info['camera_types'][camera_type] = any(cam in config_info['cameras'] for cam in camera_names)
        
        # Extract state and action dimensions
        if 'observation.state' in input_features:
            state_feature = input_features['observation.state']
            if hasattr(state_feature, 'shape'):
                config_info['state_dim'] = state_feature.shape[0]
            elif isinstance(state_feature, dict) and 'shape' in state_feature:
                config_info['state_dim'] = state_feature['shape'][0]
            
    if hasattr(policy, 'config') and hasattr(policy.config, 'output_features'):
        output_features = policy.config.output_features
        if 'action' in output_features:
            action_feature = output_features['action']
            if hasattr(action_feature, 'shape'):
                config_info['action_dim'] = action_feature.shape[0]
            elif isinstance(action_feature, dict) and 'shape' in action_feature:
                config_info['action_dim'] = action_feature['shape'][0]
    
    # Log detected configuration with state breakdown analysis
    logging.info(f"Detected cameras: {config_info['cameras']}")
    logging.info(f"Camera types in use: {config_info['camera_types']}")
    logging.info(f"Expected state dimension: {config_info['state_dim']}")
    logging.info(f"Expected action dimension: {config_info['action_dim']}")
    
    # Analyze expected state composition
    if config_info['state_dim']:
        expected_dim = config_info['state_dim']
        logging.info(f"State dimension analysis for {expected_dim}D:")
        
        # Base components
        arm_dim = 14
        hand_dim = 14  # Assuming dex3, will be adjusted if gripper
        camera_dim = 2 if config_info['camera_types']['active'] else 0
        base_dim = arm_dim + hand_dim + camera_dim
        
        logging.info(f"  - Base (arm + hand + camera): {base_dim}D ({arm_dim} + {hand_dim} + {camera_dim})")
        
        remaining = expected_dim - base_dim
        if remaining == 28:  # velocities for dex3
            logging.info(f"  - Velocities (arm + hand): {remaining}D")
        elif remaining == 16:  # velocities for gripper
            logging.info(f"  - Velocities (arm + gripper): {remaining}D")  
        elif remaining == 14:  # arm velocities only
            logging.info(f"  - Velocities (arm only): {remaining}D")
        elif remaining == 24:  # pressure sensors
            logging.info(f"  - Pressure sensors: {remaining}D")
        elif remaining == 52:  # velocities + pressure for dex3
            logging.info(f"  - Velocities (28D) + Pressure (24D): {remaining}D")
        else:
            logging.info(f"  - Additional features: {remaining}D")
    
    return config_info

def load_training_feature_config(policy_path: str):
    """
    Load the feature selection configuration from the training config file.
    
    Args:
        policy_path: Path to the policy directory
        
    Returns:
        Dict containing the feature_selection configuration used during training
    """
    # Construct path to train_config.json
    train_config_path = Path(policy_path) / "train_config.json"
    
    if not train_config_path.exists():
        logging.warning(f"Training config not found at {train_config_path}")
        logging.warning("Using default feature selection settings")
        return {
            'feature_selection': {
                'cameras': None,
                'exclude_cameras': None,
                'use_joint_positions': True,
                'use_joint_velocities': True,
                'use_joint_torques': False,
                'use_pressure_sensors': True,
                'joint_groups': None,
                'exclude_joint_groups': None,
                'custom_state_indices': None
            },
            'policy': {}
        }
    
    try:
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
        
        feature_config = train_config.get('feature_selection', {})
        policy_config = train_config.get('policy', {})
        
        logging.info(f"Loaded training feature configuration from {train_config_path}")
        logging.info(f"Training feature selection: {feature_config}")
        
        # Extract ACT-specific policy parameters that may differ from defaults
        policy_overrides = {}
        act_specific_params = [
            'chunk_size', 'n_action_steps', 'vision_backbone', 'pretrained_backbone_weights',
            'n_decoder_layers', 'n_encoder_layers', 'dim_model', 'n_heads', 'dim_feedforward',
            'temporal_ensemble_coeff', 'use_vae', 'latent_dim', 'n_vae_encoder_layers',
            'kl_weight', 'dropout', 'feedforward_activation', 'pre_norm',
            'replace_final_stride_with_dilation', 'optimizer_lr', 'optimizer_weight_decay',
            'optimizer_lr_backbone'
        ]
        
        for param in act_specific_params:
            if param in policy_config:
                policy_overrides[param] = policy_config[param]
                logging.info(f"Found policy override: {param} = {policy_config[param]}")
        
        return {
            'feature_selection': feature_config,
            'policy': policy_overrides
        }
        
    except Exception as e:
        logging.error(f"Failed to load training config from {train_config_path}: {e}")
        logging.warning("Using default feature selection settings")
        return {
            'feature_selection': {
                'cameras': None,
                'exclude_cameras': None,
                'use_joint_positions': True,
                'use_joint_velocities': True,
                'use_joint_torques': False,
                'use_pressure_sensors': True,
                'joint_groups': None,
                'exclude_joint_groups': None,
                'custom_state_indices': None
            },
            'policy': {}
        }


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load training feature configuration from train_config.json BEFORE creating policy
    logging.info("Loading training feature configuration from train_config.json...")
    training_config = load_training_feature_config(cfg.policy.pretrained_path)
    training_feature_config = training_config['feature_selection']
    training_policy_config = training_config['policy']
    
    logging.info(f"Training feature configuration: {training_feature_config}")
    logging.info(f"Training policy configuration: {training_policy_config}")
    
    # Override CLI feature selection with training configuration BEFORE creating policy
    cfg.feature_selection.cameras = training_feature_config.get('cameras')
    cfg.feature_selection.exclude_cameras = training_feature_config.get('exclude_cameras')
    cfg.feature_selection.use_joint_positions = training_feature_config.get('use_joint_positions', True)
    cfg.feature_selection.use_joint_velocities = training_feature_config.get('use_joint_velocities', True)
    cfg.feature_selection.use_joint_torques = training_feature_config.get('use_joint_torques', False)
    cfg.feature_selection.use_pressure_sensors = training_feature_config.get('use_pressure_sensors', True)
    cfg.feature_selection.joint_groups = training_feature_config.get('joint_groups')
    cfg.feature_selection.exclude_joint_groups = training_feature_config.get('exclude_joint_groups')
    cfg.feature_selection.custom_state_indices = training_feature_config.get('custom_state_indices')
    
    # Apply training policy configuration to override defaults
    for param, value in training_policy_config.items():
        if hasattr(cfg.policy, param):
            setattr(cfg.policy, param, value)
            logging.info(f"Applied policy override: {param} = {value}")
        else:
            logging.warning(f"Policy parameter {param} not found in config, skipping")
    
    # Verify critical parameters are properly set
    logging.info("=== POLICY CONFIGURATION VERIFICATION ===")
    logging.info(f"✓ Vision backbone: {cfg.policy.vision_backbone}")
    logging.info(f"✓ Chunk size: {cfg.policy.chunk_size}")
    logging.info(f"✓ N action steps: {cfg.policy.n_action_steps}")
    logging.info(f"✓ N decoder layers: {cfg.policy.n_decoder_layers}")
    logging.info(f"✓ Temporal ensemble coeff: {cfg.policy.temporal_ensemble_coeff}")
    if cfg.policy.temporal_ensemble_coeff is not None:
        logging.info("🔄 Temporal ensembling ENABLED")
    else:
        logging.info("⚠️  Temporal ensembling DISABLED")
    logging.info("==========================================")
    
    logging.info("Feature selection configuration synchronized with training:")
    logging.info(f"  - cameras: {cfg.feature_selection.cameras}")
    logging.info(f"  - exclude_cameras: {cfg.feature_selection.exclude_cameras}")
    logging.info(f"  - use_joint_positions: {cfg.feature_selection.use_joint_positions}")
    logging.info(f"  - use_joint_velocities: {cfg.feature_selection.use_joint_velocities}")
    logging.info(f"  - use_joint_torques: {cfg.feature_selection.use_joint_torques}")
    logging.info(f"  - use_pressure_sensors: {cfg.feature_selection.use_pressure_sensors}")

    logging.info("Making policy with training configuration...")

    # Load the original dataset
    original_dataset = LeRobotDataset(repo_id = cfg.repo_id)
    
    
    
    # Create FeatureSelectionConfig from the loaded training config
    feature_selection_config = FeatureSelectionConfig(
        cameras=training_feature_config.get('cameras'),
        exclude_cameras=training_feature_config.get('exclude_cameras'),
        use_joint_positions=training_feature_config.get('use_joint_positions', True),
        use_joint_velocities=training_feature_config.get('use_joint_velocities', True),
        use_joint_torques=training_feature_config.get('use_joint_torques', False),
        use_pressure_sensors=training_feature_config.get('use_pressure_sensors', True),
        joint_groups=training_feature_config.get('joint_groups'),
        exclude_joint_groups=training_feature_config.get('exclude_joint_groups'),
        custom_state_indices=training_feature_config.get('custom_state_indices')
    )
    
    # Create filtered dataset with the same feature selection used during training
    filtered_dataset = create_filtered_dataset_wrapper(original_dataset, feature_selection_config)
    
    logging.info(f"Original dataset state shape: {original_dataset.meta.features['observation.state']['shape']}")
    logging.info(f"Filtered dataset state shape: {filtered_dataset.meta.features['observation.state']['shape']}")
    
    # Use the filtered dataset metadata for policy creation
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=filtered_dataset.meta
    )
    
    # Post-policy creation verification
    logging.info("=== POST-POLICY CREATION VERIFICATION ===")
    logging.info(f"✓ Policy type: {type(policy).__name__}")
    logging.info(f"✓ Policy config type: {type(policy.config).__name__}")
    logging.info(f"✓ Vision backbone in use: {policy.config.vision_backbone}")
    logging.info(f"✓ Temporal ensemble coeff: {policy.config.temporal_ensemble_coeff}")
    
    # Check if temporal ensembler is properly initialized
    if hasattr(policy, 'temporal_ensembler'):
        if policy.temporal_ensembler is not None:
            logging.info("🔄 Temporal ensembler SUCCESSFULLY initialized")
        else:
            logging.info("⚠️  Temporal ensembler is None")
    else:
        logging.info("⚠️  No temporal_ensembler attribute found")
    
    # Check if DINOv2 backbone is properly loaded
    if hasattr(policy, 'act') and hasattr(policy.act, 'backbone'):
        backbone = policy.act.backbone
        logging.info(f"✓ Backbone type: {type(backbone).__name__}")
        if hasattr(backbone, 'dinov2_model'):
            logging.info("🖼️  DINOv2 backbone SUCCESSFULLY loaded")
            logging.info(f"   - Feature dim: {backbone.feature_dim}")
            logging.info(f"   - Patch size: {backbone.patch_size}")
        else:
            logging.info("📷 ResNet backbone detected")
    
    logging.info("===========================================")
    
    # Extract configuration from policy
    policy_config_info = extract_config_from_policy(policy)
    logging.info(f"Policy configuration: {policy_config_info}")
    
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, filtered_dataset, cfg, policy_config_info)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
