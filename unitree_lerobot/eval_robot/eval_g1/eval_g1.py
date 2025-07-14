''''
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
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from multiprocessing import shared_memory, Array, Lock

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from unitree_lerobot.eval_robot.eval_g1.robot_control.active_head_cam import ActiveCameraController
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig
from unitree_lerobot.eval_robot.eval_g1.utils import EpisodeWriter

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


# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "images" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    cfg: EvalRealConfig,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    # image
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [6],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [8, 10],
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocularPlease press 's' to start the subsequent program (no Enter needed):
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

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
        # Enable force mode to get pressure sensor data when pressure is enabled
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=cfg.cyclonedds_uri, force=cfg.force)
        # For 80D state (qpos+qvel+pressure), extract hand poses from qpos portion (first 28 dims)
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
    if cfg.active_camera:
        try:
            logging.info("Initializing active camera controller...")
            camera_controller = ActiveCameraController(
                port=cfg.camera_port,
                safe_mode=cfg.camera_safe_mode,
                max_movement_deg=cfg.camera_max_movement,
                logger=logging.getLogger('ActiveCameraController')
            )
            
            # Connect to servos
            logging.info("Connecting to camera servos...")
            if not camera_controller.connect():
                logging.error("Failed to connect to camera servos")
                camera_controller = None
            else:
                logging.info("Camera controller connected successfully")
                # Extract initial camera positions from dataset if available
                state_dim = step['observation.state'].shape[0]
                logging.info(f"Dataset state dimension: {state_dim}")
                if state_dim >= 30:  # Check if we have camera data (28 robot joints + 2 camera)
                    # Camera positions should be after arm and hand positions
                    if robot_config['hand_type'] == "dex3":
                        # Layout: arm(14) + hand(14) + camera(2) + velocities + pressure
                        init_camera_pose = step['observation.state'][28:30].cpu().numpy()  # positions 28-29 are camera
                    elif robot_config['hand_type'] == "gripper": 
                        # Layout: arm(14) + gripper(2) + camera(2) + velocities + pressure  
                        init_camera_pose = step['observation.state'][16:18].cpu().numpy()  # positions 16-17 are camera
                    else:
                        init_camera_pose = camera_controller.start_positions
                    logging.info(f"Initial camera positions from dataset: Pitch={np.rad2deg(init_camera_pose[0]):.2f}°, Yaw={np.rad2deg(init_camera_pose[1]):.2f}°")
                else:
                    init_camera_pose = camera_controller.start_positions
                    logging.info(f"Using default camera positions: Pitch={np.rad2deg(init_camera_pose[0]):.2f}°, Yaw={np.rad2deg(init_camera_pose[1]):.2f}°")
        except Exception as e:
            logging.error(f"Failed to initialize active camera: {e}")
            camera_controller = None
    else:
        logging.info("Active camera disabled")

    #===============init robot=====================
    # Initialize recorder only if recording is enabled
    if cfg.record:
        recorder = EpisodeWriter(task_dir=cfg.task_dir, frequency=cfg.frequency, rerun_log=True)
        logging.info(f"Episode recorder initialized with task_dir={cfg.task_dir}")
        logging.info("Recording will use terminal controls - press 's' to start recording episodes")
        
    if cfg.pressure and hand_ctrl and robot_config['hand_type'] == "dex3":
        force_mode = "enabled" if cfg.force else "disabled"
        logging.info(f"Pressure sensor data collection enabled via hand controller (force mode: {force_mode})")
    else:
        logging.info("Pressure sensor data collection disabled")
    
    print("Please press 's' to start the subsequent program (no Enter needed):")
    
    # Non-blocking input for initial start
    import sys
    import tty
    import termios
    
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
        if camera_controller and camera_controller.connected:
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

                # Get images
                current_tv_image = tv_img_array.copy()
                current_wrist_image = wrist_img_array.copy() if WRIST else None

                # Assign image data
                left_top_camera = current_tv_image[:, :tv_img_shape[1] // 2] if BINOCULAR else current_tv_image
                right_top_camera = current_tv_image[:, tv_img_shape[1] // 2:] if BINOCULAR else None
                left_wrist_camera, right_wrist_camera = (
                    (current_wrist_image[:, :wrist_img_shape[1] // 2], current_wrist_image[:, wrist_img_shape[1] // 2:])
                    if WRIST else (None, None)
                )

                observation = {
                    "observation.images.cam_left_high": torch.from_numpy(left_top_camera),
                    "observation.images.cam_right_high": torch.from_numpy(right_top_camera) if BINOCULAR else None,
                    "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_camera) if WRIST else None,
                    "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_camera) if WRIST else None,
                }

                # Get camera positions if active camera is enabled
                if camera_controller and camera_controller.connected:
                    servo_states = camera_controller.get_servo_states()
                    current_camera_q = np.array([servo_states['current_pitch'], servo_states['current_yaw']])  # 2D
                    current_camera_dq = np.zeros(2)  # Camera velocities not available, use zeros for now
                else:
                    # Use default positions if camera not available
                    current_camera_q = np.array([195.0 * np.pi / 180, 90.0 * np.pi / 180])  # Default safe positions
                    current_camera_dq = np.zeros(2)

                # Build observation state: depends on whether camera is included in training data
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()  # 14D
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()  # 14D velocity
                
                # Get hand state data
                if robot_config['hand_type'] == "dex3":
                    with dual_hand_data_lock:
                        left_hand_state = dual_hand_state_array[:7]   # 7D position
                        right_hand_state = dual_hand_state_array[-7:] # 7D position
                    
                    # For velocity, hands provide velocity if available in shared array
                    # For now using zeros for hand velocities since not available in simple mode
                    left_hand_vel = np.zeros(7)
                    right_hand_vel = np.zeros(7)
                    
                    # Get pressure sensor data if available
                    if cfg.pressure and hand_ctrl:
                        pressure_data = hand_ctrl.get_pressure_data()
                        left_pressure = np.array(pressure_data['left_pressure'])  # 12D
                        right_pressure = np.array(pressure_data['right_pressure']) # 12D
                    else:
                        left_pressure = np.zeros(12)
                        right_pressure = np.zeros(12)
                    
                    # Construct state with camera: [arm_q(14) + hand_q(14) + camera_q(2)] + [arm_dq(14) + hand_dq(14) + camera_dq(2)] + [pressure(24)]
                    observation_state = np.concatenate([
                        current_lr_arm_q,      # 14D arm positions
                        left_hand_state,       # 7D left hand positions  
                        right_hand_state,      # 7D right hand positions
                        current_camera_q,      # 2D camera positions (pitch, yaw)
                        current_lr_arm_dq,     # 14D arm velocities
                        left_hand_vel,         # 7D left hand velocities
                        right_hand_vel,        # 7D right hand velocities  
                        current_camera_dq,     # 2D camera velocities
                        left_pressure,         # 12D left hand pressure
                        right_pressure         # 12D right hand pressure
                    ])
                    
                elif robot_config['hand_type'] == "gripper":
                    with dual_gripper_data_lock:
                        left_hand_state = [dual_gripper_state_array[1]]
                        right_hand_state = [dual_gripper_state_array[0]]
                    
                    # For grippers, simpler state structure - extend with zeros to match training data dimensions
                    left_hand_vel = [0.0]
                    right_hand_vel = [0.0]
                    # Pad with zeros to match expected dimensions
                    padding = np.zeros(24)  # Adjust based on actual trained model expectations
                    
                    observation_state = np.concatenate([
                        current_lr_arm_q,      # 14D
                        left_hand_state,       # 1D
                        right_hand_state,      # 1D
                        current_camera_q,      # 2D camera positions
                        current_lr_arm_dq,     # 14D
                        left_hand_vel,         # 1D
                        right_hand_vel,        # 1D
                        current_camera_dq,     # 2D camera velocities  
                        padding                # Padding to match training dimensions
                    ])
                
                observation["observation.state"] = torch.from_numpy(observation_state).float()

                observation = {
                    key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
                }

                action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                action = action.cpu().numpy()
                
                # Show periodic instructions for terminal controls (every 60 seconds instead of 30)
                if cfg.record and time.time() - last_instruction_time > 60:
                    print("\n📝 RECORDING CONTROLS:")
                    print("   s = Start | r = Abort | q = Save optimal | w = Save suboptimal | e = Save recovery | x = Exit")
                    last_instruction_time = time.time()

                # Record data if recording is active
                if cfg.record and recording_state and recording_state[0] == 1:
                    # Prepare data for recording
                    colors = {}
                    depths = {}
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                        if WRIST:
                            colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                        if WRIST:
                            colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = action[:7]
                    right_arm_action = action[7:14]
                    
                    # hand action and camera action (split based on hand type and camera availability)
                    if camera_controller and camera_controller.connected:
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
                            "qpos":   left_hand_state,           
                            "qvel":   [],                           
                            "torque": [], 
                            "pressures": pressure_data['left_pressure'],
                            #"temperatures": pressure_data['left_temp'],  TODO: Testen, denke aber nicht dass das einen großen einfluss hat           
                        }, 
                        "right_hand": {                                                                    
                            "qpos":   right_hand_state,       
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
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_hand": {                                   
                            "qpos":   right_hand_action,       
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
                # Split actions: camera actions are at the end
                if camera_controller and camera_controller.connected:
                    if robot_config['hand_type'] == "dex3":
                        # Actions: [arm(14) + hand(14) + camera(2)] = 30D total
                        arm_action = action[:14]
                        hand_action = action[14:28]
                        camera_action = action[28:30]  # Last 2 values are camera pitch/yaw
                    elif robot_config['hand_type'] == "gripper":
                        # Actions: [arm(14) + gripper(2) + camera(2)] = 18D total
                        arm_action = action[:14]
                        hand_action = action[14:16]
                        camera_action = action[16:18]  # Last 2 values are camera pitch/yaw
                    
                    # Execute camera action using active camera controller
                    try:
                        target_pitch, target_yaw = camera_action
                        # Use threaded control method to set target positions
                        camera_controller._set_target_positions(target_pitch, target_yaw)
                        if frame_counter % 150 == 0:  # Log every 5 seconds
                            logging.info(f"Camera targets: Pitch={np.rad2deg(target_pitch):.1f}°, Yaw={np.rad2deg(target_yaw):.1f}°")
                    except Exception as e:
                        logging.error(f"Error controlling camera: {e}")
                else:
                    # No active camera, use original action dimensions
                    arm_action = action[:14]
                    if robot_config['hand_type'] == "dex3":
                        hand_action = action[14:28]
                    elif robot_config['hand_type'] == "gripper":
                        hand_action = action[14:16]
                
                # Execute arm and hand actions
                arm_ctrl.ctrl_dual_arm(arm_action, np.zeros(14))
                if robot_config['hand_type'] == "dex3":
                    left_hand_array[:] = hand_action[:7]
                    right_hand_array[:] = hand_action[7:]
                elif robot_config['hand_type'] == "gripper":
                    left_hand_array[:] = hand_action[0]
                    right_hand_array[:] = hand_action[1]
            
                frame_counter += 1
            
            # Log performance info periodically
            if frame_counter % 300 == 0:  # Every ~6 seconds at 50fps
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
            
            tv_img_shm.unlink()
            tv_img_shm.close()
            if WRIST:
                wrist_img_shm.unlink()
                wrist_img_shm.close()
            logging.info("Resources cleaned up, exiting program")


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id = cfg.repo_id)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, dataset, cfg)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
