"""
Noise injection utilities for joint coordinates and sensor data.

This module provides sophisticated noise injection techniques for robot
joint coordinates and other sensor data to improve policy robustness.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class NoiseInjector:
    """
    Advanced noise injection class for robotics sensor data augmentation.
    
    This class provides methods for adding realistic noise to joint positions,
    velocities, and other sensor readings to improve the robustness of
    imitation learning policies.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the noise injector.
        
        Args:
            seed: Random seed for reproducible noise generation
        """
        self.rng = np.random.RandomState(seed)
    
    def add_gaussian_noise(
        self, 
        values: List[float],
        noise_std: float = 0.02,
        max_noise: float = 0.05
    ) -> List[float]:
        """
        Add Gaussian noise to a list of values with clipping.
        
        Args:
            values: List of input values
            noise_std: Standard deviation of Gaussian noise
            max_noise: Maximum allowed noise magnitude
            
        Returns:
            List of values with added noise
        """
        noisy_values = []
        for val in values:
            noise = self.rng.normal(0, noise_std)
            noise = np.clip(noise, -max_noise, max_noise)
            noisy_values.append(val + noise)
        
        return noisy_values
    
    def add_joint_position_noise(
        self, 
        joint_positions: List[float],
        position_noise_params: Optional[Dict] = None
    ) -> List[float]:
        """
        Add realistic noise to joint positions.
        
        Args:
            joint_positions: List of joint positions in radians
            position_noise_params: Parameters for position noise
            
        Returns:
            Joint positions with added noise
        """
        if position_noise_params is None:
            position_noise_params = {
                'std': 0.02,
                'max_noise': 0.05,
                'bias_drift_std': 0.001  # Small drift to simulate sensor bias
            }
        
        noisy_positions = []
        for pos in joint_positions:
            # Add Gaussian noise
            noise = self.rng.normal(0, position_noise_params['std'])
            noise = np.clip(noise, -position_noise_params['max_noise'], 
                          position_noise_params['max_noise'])
            
            # Add small bias drift
            bias_drift = self.rng.normal(0, position_noise_params['bias_drift_std'])
            
            noisy_positions.append(pos + noise + bias_drift)
        
        return noisy_positions
    
    def add_joint_velocity_noise(
        self, 
        joint_velocities: List[float],
        velocity_noise_params: Optional[Dict] = None
    ) -> List[float]:
        """
        Add realistic noise to joint velocities.
        
        Args:
            joint_velocities: List of joint velocities in rad/s
            velocity_noise_params: Parameters for velocity noise
            
        Returns:
            Joint velocities with added noise
        """
        if velocity_noise_params is None:
            velocity_noise_params = {
                'std': 0.1,
                'max_noise': 0.2,
                'quantization_noise': 0.01  # Simulate encoder quantization
            }
        
        noisy_velocities = []
        for vel in joint_velocities:
            # Add Gaussian noise
            noise = self.rng.normal(0, velocity_noise_params['std'])
            noise = np.clip(noise, -velocity_noise_params['max_noise'], 
                          velocity_noise_params['max_noise'])
            
            # Add quantization noise
            quant_noise = self.rng.uniform(-velocity_noise_params['quantization_noise'],
                                         velocity_noise_params['quantization_noise'])
            
            noisy_velocities.append(vel + noise + quant_noise)
        
        return noisy_velocities
    
    def add_torque_noise(
        self, 
        torques: List[float],
        torque_noise_params: Optional[Dict] = None
    ) -> List[float]:
        """
        Add realistic noise to joint torques.
        
        Args:
            torques: List of joint torques
            torque_noise_params: Parameters for torque noise
            
        Returns:
            Joint torques with added noise
        """
        if torque_noise_params is None:
            torque_noise_params = {
                'std': 0.05,
                'max_noise': 0.1,
                'friction_variation': 0.02  # Simulate friction variations
            }
        
        noisy_torques = []
        for torque in torques:
            # Add Gaussian noise
            noise = self.rng.normal(0, torque_noise_params['std'])
            noise = np.clip(noise, -torque_noise_params['max_noise'],
                          torque_noise_params['max_noise'])
            
            # Add friction variation
            friction_var = self.rng.normal(0, torque_noise_params['friction_variation'])
            
            noisy_torques.append(torque + noise + friction_var)
        
        return noisy_torques
    
    def add_temporal_correlation_noise(
        self, 
        values: List[float],
        correlation_factor: float = 0.1,
        base_noise_std: float = 0.02
    ) -> List[float]:
        """
        Add temporally correlated noise to simulate sensor drift.
        
        Args:
            values: List of input values
            correlation_factor: Correlation factor between consecutive noise samples
            base_noise_std: Base standard deviation for noise
            
        Returns:
            Values with temporally correlated noise
        """
        noisy_values = []
        prev_noise = 0.0
        
        for val in values:
            # Generate correlated noise
            current_noise = (correlation_factor * prev_noise + 
                           np.sqrt(1 - correlation_factor**2) * 
                           self.rng.normal(0, base_noise_std))
            
            noisy_values.append(val + current_noise)
            prev_noise = current_noise
        
        return noisy_values
    
    def add_state_dependent_noise(
        self, 
        joint_states: Dict[str, Dict[str, List[float]]],
        noise_config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Add state-dependent noise to complete joint state information.
        
        Args:
            joint_states: Dictionary containing joint state data
            noise_config: Configuration for different types of noise
            
        Returns:
            Joint states with added noise
        """
        if noise_config is None:
            noise_config = {
                'position_noise': {'std': 0.02, 'max_noise': 0.05},
                'velocity_noise': {'std': 0.1, 'max_noise': 0.2},
                'torque_noise': {'std': 0.05, 'max_noise': 0.1},
                'enable_correlation': True,
                'correlation_factor': 0.1
            }
        
        noisy_states = {}
        
        for component_name, component_data in joint_states.items():
            noisy_states[component_name] = {}
            
            for data_type, data_values in component_data.items():
                if not isinstance(data_values, list) or len(data_values) == 0:
                    # Keep non-list data or empty lists unchanged
                    noisy_states[component_name][data_type] = data_values
                    continue
                
                # Apply appropriate noise based on data type
                if data_type == 'qpos':
                    if noise_config.get('enable_correlation', False):
                        noisy_values = self.add_temporal_correlation_noise(
                            data_values,
                            correlation_factor=noise_config.get('correlation_factor', 0.1),
                            base_noise_std=noise_config['position_noise']['std']
                        )
                    else:
                        noisy_values = self.add_joint_position_noise(
                            data_values, 
                            noise_config.get('position_noise')
                        )
                
                elif data_type == 'qvel':
                    noisy_values = self.add_joint_velocity_noise(
                        data_values,
                        noise_config.get('velocity_noise')
                    )
                
                elif data_type == 'torque':
                    noisy_values = self.add_torque_noise(
                        data_values,
                        noise_config.get('torque_noise')
                    )
                
                else:
                    # For unknown data types, apply basic Gaussian noise
                    noisy_values = self.add_gaussian_noise(
                        data_values,
                        noise_std=0.01,
                        max_noise=0.02
                    )
                
                noisy_states[component_name][data_type] = noisy_values
        
        return noisy_states
    
    def simulate_sensor_dropout(
        self, 
        values: List[float],
        dropout_probability: float = 0.001,
        dropout_duration: int = 1
    ) -> List[float]:
        """
        Simulate sensor dropout events.
        
        Args:
            values: List of input values
            dropout_probability: Probability of dropout per timestep
            dropout_duration: Duration of dropout in timesteps
            
        Returns:
            Values with simulated dropout events
        """
        if len(values) == 0:
            return values
        
        result = values.copy()
        i = 0
        
        while i < len(result):
            if self.rng.random() < dropout_probability:
                # Simulate dropout by holding previous value
                dropout_end = min(i + dropout_duration, len(result))
                hold_value = result[i-1] if i > 0 else result[i]
                
                for j in range(i, dropout_end):
                    result[j] = hold_value
                
                i = dropout_end
            else:
                i += 1
        
        return result
