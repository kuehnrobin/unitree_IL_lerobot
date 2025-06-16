#!/usr/bin/env python3
"""
Pressure sensor data collection utility for G1 robot hands
Based on test_pressure_sensors.py from avp_teleoperate
"""
import numpy as np
import time
import threading
import logging

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, PressSensorState_
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: unitree_sdk2py not available, pressure sensor will be disabled")

# Topic names for pressure sensor data
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

class PressureSensorCollector:
    """
    Collects pressure sensor data from Unitree DEX3 hands
    """
    
    def __init__(self, networkInterface='enxa0cec8616f27', enabled=True):
        """
        Initialize pressure sensor collector
        
        Args:
            networkInterface: Network interface for CycloneDX
            enabled: Whether to enable pressure sensor collection
        """
        self.enabled = enabled and SDK_AVAILABLE
        self.logger = logging.getLogger('pressure_sensor')
        
        if not self.enabled:
            self.logger.warning("Pressure sensor collection disabled")
            return
            
        self.logger.info("Initializing Pressure Sensor Collector...")
        
        # Initialize DDS
        ChannelFactoryInitialize(0, networkInterface)
        
        # Initialize subscribers for hand states (which contain pressure sensor data)
        self.left_hand_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.left_hand_subscriber.Init()
        
        self.right_hand_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.right_hand_subscriber.Init()
        
        # Data storage - 12 pressure values and 12 temperature values per hand
        self.left_pressure_data = np.zeros(12)
        self.left_temperature_data = np.zeros(12)
        self.right_pressure_data = np.zeros(12)
        self.right_temperature_data = np.zeros(12)
        
        self.data_received = False
        self.running = False
        
        # Thread for background data collection
        self.subscriber_thread = None
        
        self.logger.info("Pressure Sensor Collector initialized successfully!")
        
    def _subscribe_pressure_data(self):
        """
        Background thread to continuously read pressure sensor data
        """
        self.logger.info("Starting pressure sensor data subscription...")
        
        while self.running:
            try:
                # Read left hand state
                left_msg = self.left_hand_subscriber.Read()
                if left_msg is not None and len(left_msg.press_sensor_state) > 0:
                    press_sensor = left_msg.press_sensor_state[0]  # Get first pressure sensor state
                    self.left_pressure_data = np.array(press_sensor.pressure[:12])
                    self.left_temperature_data = np.array(press_sensor.temperature[:12])
                    self.data_received = True
                
                # Read right hand state
                right_msg = self.right_hand_subscriber.Read()
                if right_msg is not None and len(right_msg.press_sensor_state) > 0:
                    press_sensor = right_msg.press_sensor_state[0]  # Get first pressure sensor state
                    self.right_pressure_data = np.array(press_sensor.pressure[:12])
                    self.right_temperature_data = np.array(press_sensor.temperature[:12])
                    self.data_received = True
                    
            except Exception as e:
                self.logger.error(f"Error reading pressure data: {e}")
                
            time.sleep(0.001)  # 1ms sleep to avoid busy waiting
            
    def start(self):
        """Start the pressure sensor collection"""
        if not self.enabled:
            return True
            
        self.running = True
        self.subscriber_thread = threading.Thread(target=self._subscribe_pressure_data)
        self.subscriber_thread.daemon = True
        self.subscriber_thread.start()
        
        # Wait for initial data with timeout
        self.logger.info("Waiting for pressure sensor data...")
        timeout = 5.0  # 5 second timeout
        start_time = time.time()
        
        while not self.data_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.data_received:
            self.logger.warning("No pressure sensor data received within timeout!")
            return False
            
        self.logger.info("Pressure sensor data received successfully!")
        return True
        
    def stop(self):
        """Stop the pressure sensor collection"""
        if not self.enabled:
            return
            
        self.running = False
        if self.subscriber_thread and self.subscriber_thread.is_alive():
            self.subscriber_thread.join(timeout=1.0)
            
    def get_pressure_data(self):
        """
        Get current pressure data
        
        Returns:
            dict: Pressure data in format compatible with recording system
                 {'left_pressure': list, 'left_temp': list, 'right_pressure': list, 'right_temp': list}
        """
        if not self.enabled or not self.data_received:
            # Return empty/zero data if pressure sensing is disabled or no data available
            return {
                'left_pressure': [0.0] * 12,
                'left_temp': [0.0] * 12, 
                'right_pressure': [0.0] * 12,
                'right_temp': [0.0] * 12
            }
            
        return {
            'left_pressure': self.left_pressure_data.tolist(),
            'left_temp': self.left_temperature_data.tolist(),
            'right_pressure': self.right_pressure_data.tolist(),
            'right_temp': self.right_temperature_data.tolist()
        }
        
    def get_summary_stats(self):
        """
        Get summary statistics of current pressure data
        
        Returns:
            dict: Summary statistics
        """
        if not self.enabled or not self.data_received:
            return {
                'left_max_pressure': 0.0,
                'left_avg_pressure': 0.0,
                'right_max_pressure': 0.0,
                'right_avg_pressure': 0.0
            }
            
        return {
            'left_max_pressure': float(np.max(self.left_pressure_data)),
            'left_avg_pressure': float(np.mean(self.left_pressure_data)),
            'right_max_pressure': float(np.max(self.right_pressure_data)),
            'right_avg_pressure': float(np.mean(self.right_pressure_data))
        }
