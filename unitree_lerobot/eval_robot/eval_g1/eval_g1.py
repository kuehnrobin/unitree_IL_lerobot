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
import cv2
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
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig
from unitree_lerobot.eval_robot.eval_g1.utils.episode_writer import EpisodeWriter

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logging.info("Shutdown requested via signal, will exit gracefully...")

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
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
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
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=cfg.cyclonedds_uri)
        init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
        init_right_hand_pose = step['observation.state'][21:].cpu().numpy()

    elif robot_config['hand_type'] == "gripper":
        left_hand_array = Array('d', 1, lock=True)             # [input]
        right_hand_array = Array('d', 1, lock=True)            # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, networkInterface=cfg.cyclonedds_uri)
        init_left_hand_pose = step['observation.state'][14].cpu().numpy()
        init_right_hand_pose = step['observation.state'][15].cpu().numpy()
    else:
        pass

    #===============init robot=====================
    if cfg.record:
        recorder = EpisodeWriter(task_dir=cfg.task_dir, frequency=cfg.frequency, rerun_log=True)
        recording = False
        logging.info(f"Episode recorder initialized with task_dir={cfg.task_dir}")
        
        # Try to initialize OpenCV window for recording interface
        try:
            # Set OpenCV backend explicitly to avoid Qt issues
            cv2.namedWindow("record image", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("record image", 100, 100)  # Position window
            
            # Create a simple placeholder image to show initially
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Recording Interface", (50, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(placeholder, "Waiting for camera feed...", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(placeholder, "Click window & press keys", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(placeholder, "OR use terminal controls", (20, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("record image", placeholder)
            cv2.waitKey(10)
            
            logging.info("Recording interface window created")
            opencv_available = True
        except Exception as e:
            logging.warning(f"Could not create OpenCV window: {e}")
            logging.info("Recording will use terminal input only")
            opencv_available = False
    
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        # Apply gradual speed increase if not disabled
        if not cfg.no_gradual_speed:
            arm_ctrl.speed_gradual_max()
            logging.info("Gradual speed increase enabled")

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print("init robot pose")
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        left_hand_array[:] = init_left_hand_pose
        right_hand_array[:] = init_right_hand_pose

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
                """Non-blocking terminal input handler using select on stdin"""
                import sys
                import select
                import tty
                import termios
                
                global shutdown_requested
                nonlocal recording_state
                
                # Print initial instructions
                print("\n=== RECORDING CONTROLS ===")
                print("s - Start recording episode")
                print("r - Abort current recording") 
                print("q - Save recording as optimal")
                print("w - Save recording as suboptimal")
                print("e - Save recording as recovery")
                print("Ctrl+C - Exit program")
                print("============================")
                print("Ready for single-key commands (no Enter needed)...")
                
                # Save terminal settings
                old_settings = None
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())
                    
                    while not shutdown_requested:
                        # Non-blocking check for input
                        if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                            char = sys.stdin.read(1)
                            
                            if char == '\x03':  # Ctrl+C
                                shutdown_requested = True
                                print("\nShutdown requested via Ctrl+C")
                                break
                            elif char == 's' and recording_state[0] == 0:
                                if recorder.create_episode():
                                    recording_state[0] = 1
                                    logging.info("Started recording episode via terminal")
                                    print("\n✓ Recording started!")
                                else:
                                    print("\n✗ Failed to start recording")
                            elif char == 'r' and recording_state[0] == 1:
                                recorder.abort_episode()
                                recording_state[0] = 0
                                logging.info("Aborted recording episode via terminal")
                                print("\n✓ Recording aborted!")
                            elif recording_state[0] == 1 and char in ['q', 'w', 'e']:
                                quality_map = {'q': 'optimal', 'w': 'suboptimal', 'e': 'recovery'}
                                quality = quality_map[char]
                                recorder.save_episode(quality=quality)
                                recording_state[0] = 0
                                logging.info(f"Saved recording episode with quality: {quality} via terminal")
                                print(f"\n✓ Recording saved as {quality}!")
                        
                        time.sleep(0.01)  # Small delay to prevent busy waiting
                        
                except Exception as e:
                    logging.error(f"Terminal input handler error: {e}")
                finally:
                    # Restore terminal settings
                    if old_settings:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    print("\nTerminal input handler stopped")
            
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

                # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                # dex hand or gripper
                if robot_config['hand_type'] == "dex3":
                    with dual_hand_data_lock:
                        left_hand_state = dual_hand_state_array[:7]
                        right_hand_state = dual_hand_state_array[-7:]
                elif robot_config['hand_type'] == "gripper":
                    with dual_gripper_data_lock:
                        left_hand_state = [dual_gripper_state_array[1]]
                        right_hand_state = [dual_gripper_state_array[0]]
                
                observation["observation.state"] = torch.from_numpy(np.concatenate((current_lr_arm_q, left_hand_state, right_hand_state), axis=0)).float()

                observation = {
                    key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
                }

                action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                action = action.cpu().numpy()
                
                # Handle recording display and keyboard input
                if cfg.record and opencv_available:
                    current_recording = recording_state[0] == 1
                    # Prepare display image
                    tv_resized_image = cv2.resize(current_tv_image, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                    
                    # Add recording status overlay
                    status_text = ""
                    help_text = ""
                    if current_recording:
                        status_text = "RECORDING"
                        help_text = "OpenCV: [r]=abort | [q]=save optimal | [w]=save suboptimal | [e]=save recovery"
                    else:
                        status_text = "READY TO RECORD"
                        help_text = "OpenCV: [s]=start recording | Terminal: use terminal commands"
                    
                    # Add status text overlay
                    cv2.putText(tv_resized_image, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if current_recording else (0, 255, 0), 2)
                    
                    # Add help text overlay
                    cv2.putText(tv_resized_image, help_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Show the window
                    cv2.imshow("record image", tv_resized_image)
                    
                    # Only check for OpenCV keys if window is available
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Handle key presses for recording control (OpenCV window)
                    if key == ord('s') and not current_recording:
                        # Start recording
                        if recorder.create_episode():
                            recording_state[0] = 1
                            logging.info("Started recording episode via OpenCV")
                        else:
                            logging.warning("Failed to create recording episode")
                    elif key == ord('r') and current_recording:
                        # Abort recording
                        recorder.abort_episode()
                        recording_state[0] = 0
                        logging.info("Aborted recording episode via OpenCV")
                    elif current_recording and key in [ord('q'), ord('w'), ord('e')]:
                        # Save with quality labels
                        quality_map = {ord('q'): 'optimal', ord('w'): 'suboptimal', ord('e'): 'recovery'}
                        quality = quality_map[key]
                        recorder.save_episode(quality=quality)
                        recording_state[0] = 0
                        logging.info(f"Saved recording episode with quality: {quality} via OpenCV")
                
                # Show periodic instructions for terminal controls
                if cfg.record and time.time() - last_instruction_time > 30:  # Every 30 seconds
                    logging.info("Recording controls: Use terminal interface or click OpenCV window and press keys")
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
                    
                    # hand action (use actual actions for recording)
                    if robot_config['hand_type'] == "dex3":
                        left_hand_action = action[14:21]
                        right_hand_action = action[21:]
                    elif robot_config['hand_type'] == "gripper":
                        left_hand_action = [action[14]]
                        right_hand_action = [action[15]]
                    
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_hand": {                                                                    
                            "qpos":   left_hand_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_hand": {                                                                    
                            "qpos":   right_hand_state,       
                            "qvel":   [],                           
                            "torque": [],  
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
                        "body": None, 
                    }
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)
                
                # exec action
                arm_ctrl.ctrl_dual_arm(action[:14], np.zeros(14))
                if robot_config['hand_type'] == "dex3":
                    left_hand_array[:] = action[14:21]
                    right_hand_array[:] = action[21:]
                elif robot_config['hand_type'] == "gripper":
                    left_hand_array[:] = action[14]
                    right_hand_array[:] = action[15]
            
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
                cv2.destroyAllWindows()
                logging.info("Recording resources cleaned up")
            
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
