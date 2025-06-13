"""
Advanced image transformation utilities for dataset augmentation.

This module provides sophisticated image augmentation techniques specifically
designed for robotics datasets, including lighting variations, noise injection,
and color space manipulations that maintain visual realism.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional
import cv2


class ImageTransforms:
    """
    Advanced image transformation class for robotics dataset augmentation.
    
    This class provides methods for applying realistic image transformations
    that simulate various environmental conditions and sensor variations
    commonly encountered in robotic applications.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the image transforms.
        
        Args:
            seed: Random seed for reproducible transformations
        """
        self.rng = np.random.RandomState(seed)
    
    def adjust_lighting(
        self, 
        image: Image.Image,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Image.Image:
        """
        Apply realistic lighting adjustments to simulate different environmental conditions.
        
        Args:
            image: Input PIL Image
            brightness_range: Min and max brightness adjustment factors
            contrast_range: Min and max contrast adjustment factors  
            saturation_range: Min and max saturation adjustment factors
            
        Returns:
            Transformed PIL Image
        """
        # Apply brightness adjustment
        brightness_factor = self.rng.uniform(*brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Apply contrast adjustment
        contrast_factor = self.rng.uniform(*contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Apply saturation adjustment
        saturation_factor = self.rng.uniform(*saturation_range)
        image = ImageEnhance.Color(image).enhance(saturation_factor)
        
        return image
    
    def add_camera_noise(
        self, 
        image: Image.Image,
        noise_std: float = 3.0,
        salt_pepper_prob: float = 0.001
    ) -> Image.Image:
        """
        Add realistic camera sensor noise to the image.
        
        Args:
            image: Input PIL Image
            noise_std: Standard deviation for Gaussian noise
            salt_pepper_prob: Probability for salt and pepper noise
            
        Returns:
            Image with added noise
        """
        img_array = np.array(image)
        
        # Add Gaussian noise
        if noise_std > 0:
            gaussian_noise = self.rng.normal(0, noise_std, img_array.shape)
            img_array = img_array.astype(np.float32) + gaussian_noise
        
        # Add salt and pepper noise occasionally
        if salt_pepper_prob > 0 and self.rng.random() < 0.1:
            mask = self.rng.random(img_array.shape[:2]) < salt_pepper_prob
            img_array[mask] = self.rng.choice([0, 255], size=np.sum(mask))
        
        # Clip values and convert back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def simulate_motion_blur(
        self, 
        image: Image.Image,
        blur_probability: float = 0.15,
        max_blur_radius: float = 1.0
    ) -> Image.Image:
        """
        Simulate motion blur that might occur during robot movement.
        
        Args:
            image: Input PIL Image
            blur_probability: Probability of applying blur
            max_blur_radius: Maximum blur radius
            
        Returns:
            Image with potential motion blur
        """
        if self.rng.random() < blur_probability:
            blur_radius = self.rng.uniform(0.2, max_blur_radius)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image
    
    def adjust_color_temperature(
        self, 
        image: Image.Image,
        temperature_shift: int = 30
    ) -> Image.Image:
        """
        Apply color temperature adjustments to simulate different lighting conditions.
        
        Args:
            image: Input PIL Image
            temperature_shift: Maximum temperature shift in Kelvin (approximate)
            
        Returns:
            Image with adjusted color temperature
        """
        # Convert to numpy array for processing
        img_array = np.array(image).astype(np.float32)
        
        # Generate random temperature shift
        temp_shift = self.rng.uniform(-temperature_shift, temperature_shift)
        
        # Apply temperature adjustment (simplified model)
        if temp_shift > 0:  # Warmer (more red/yellow)
            img_array[:, :, 0] *= (1 + temp_shift / 1000)  # Red channel
            img_array[:, :, 1] *= (1 + temp_shift / 2000)  # Green channel
        else:  # Cooler (more blue)
            img_array[:, :, 2] *= (1 - temp_shift / 1000)  # Blue channel
        
        # Clip and convert back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def apply_random_shadow(
        self, 
        image: Image.Image,
        shadow_probability: float = 0.2,
        shadow_intensity: float = 0.3
    ) -> Image.Image:
        """
        Apply random shadow effects to simulate varying lighting conditions.
        
        Args:
            image: Input PIL Image
            shadow_probability: Probability of applying shadow
            shadow_intensity: Intensity of shadow effect (0-1)
            
        Returns:
            Image with potential shadow effects
        """
        if self.rng.random() < shadow_probability:
            img_array = np.array(image).astype(np.float32)
            height, width = img_array.shape[:2]
            
            # Create random shadow pattern
            shadow_mask = np.ones((height, width))
            
            # Random shadow direction and size
            shadow_type = self.rng.choice(['linear', 'circular'])
            
            if shadow_type == 'linear':
                # Linear shadow (like from a edge/wall)
                start_pos = self.rng.randint(0, width)
                for i in range(height):
                    shadow_width = int(self.rng.uniform(0.2, 0.6) * width)
                    start_x = max(0, start_pos - shadow_width // 2)
                    end_x = min(width, start_pos + shadow_width // 2)
                    shadow_mask[i, start_x:end_x] *= (1 - shadow_intensity)
            
            elif shadow_type == 'circular':
                # Circular shadow (like from overhead object)
                center_x = self.rng.randint(width // 4, 3 * width // 4)
                center_y = self.rng.randint(height // 4, 3 * height // 4)
                radius = self.rng.randint(min(width, height) // 8, min(width, height) // 4)
                
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                shadow_mask[mask] *= (1 - shadow_intensity)
            
            # Apply shadow
            img_array *= shadow_mask[:, :, np.newaxis]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        return image
    
    def generate_timestep_augmentation_params(self, config: dict = None) -> dict:
        """
        Generate consistent augmentation parameters for all cameras in a timestep.
        
        Args:
            config: Configuration dictionary for augmentation parameters
            
        Returns:
            Dictionary with augmentation parameters to apply to all cameras
        """
        if config is None:
            config = {}
        
        params = {
            # Lighting parameters (consistent across all cameras)
            'brightness_factor': self.rng.uniform(*config.get('brightness_range', (0.7, 1.3))),
            'contrast_factor': self.rng.uniform(*config.get('contrast_range', (0.8, 1.2))),
            'saturation_factor': self.rng.uniform(*config.get('saturation_range', (0.8, 1.2))),
            
            # Color temperature adjustment
            'apply_temperature_adjust': self.rng.random() < config.get('temperature_prob', 0.4),
            'temperature_shift': self.rng.uniform(-30, 30) if self.rng.random() < 0.4 else 0,
            
            # Motion blur (consistent across cameras)
            'apply_blur': self.rng.random() < config.get('blur_prob', 0.15),
            'blur_radius': self.rng.uniform(0.2, config.get('max_blur_radius', 1.0)),
            
            # Shadow effects (consistent lighting conditions)
            'apply_shadow': self.rng.random() < config.get('shadow_prob', 0.2),
            'shadow_intensity': self.rng.uniform(0.1, config.get('shadow_intensity', 0.3)),
            'shadow_type': self.rng.choice(['linear', 'circular']),
            'shadow_params': self._generate_shadow_params(),
            
            # Noise parameters (camera-specific but similar characteristics)
            'apply_noise': self.rng.random() < 0.3,
            'noise_std': self.rng.uniform(2, config.get('noise_std', 3.0)),
            'salt_pepper_prob': config.get('salt_pepper_prob', 0.001)
        }
        
        return params
    
    def _generate_shadow_params(self) -> dict:
        """Generate shadow parameters for consistent application across cameras."""
        return {
            'linear_start_ratio': self.rng.uniform(0.2, 0.8),  # Position as ratio of width
            'linear_width_ratio': self.rng.uniform(0.2, 0.6),  # Shadow width as ratio
            'circular_center_x_ratio': self.rng.uniform(0.25, 0.75),
            'circular_center_y_ratio': self.rng.uniform(0.25, 0.75),
            'circular_radius_ratio': self.rng.uniform(0.125, 0.25)  # Radius as ratio of min(width, height)
        }
    
    def apply_timestep_augmentation(
        self, 
        image: Image.Image,
        augmentation_params: dict
    ) -> Image.Image:
        """
        Apply consistent augmentation to an image using pre-generated parameters.
        
        Args:
            image: Input PIL Image
            augmentation_params: Parameters from generate_timestep_augmentation_params()
            
        Returns:
            Augmented PIL Image
        """
        # Apply lighting adjustments
        image = ImageEnhance.Brightness(image).enhance(augmentation_params['brightness_factor'])
        image = ImageEnhance.Contrast(image).enhance(augmentation_params['contrast_factor'])
        image = ImageEnhance.Color(image).enhance(augmentation_params['saturation_factor'])
        
        # Apply color temperature adjustment
        if augmentation_params['apply_temperature_adjust']:
            image = self._apply_temperature_shift(image, augmentation_params['temperature_shift'])
        
        # Apply motion blur
        if augmentation_params['apply_blur']:
            image = image.filter(ImageFilter.GaussianBlur(radius=augmentation_params['blur_radius']))
        
        # Apply shadow effects
        if augmentation_params['apply_shadow']:
            image = self._apply_shadow_with_params(
                image, 
                augmentation_params['shadow_type'],
                augmentation_params['shadow_intensity'],
                augmentation_params['shadow_params']
            )
        
        # Add camera noise
        if augmentation_params['apply_noise']:
            image = self._add_noise_with_params(
                image,
                augmentation_params['noise_std'],
                augmentation_params['salt_pepper_prob']
            )
        
        return image
    
    def _apply_temperature_shift(self, image: Image.Image, temp_shift: float) -> Image.Image:
        """Apply color temperature shift using pre-calculated parameters."""
        img_array = np.array(image).astype(np.float32)
        
        if temp_shift > 0:  # Warmer (more red/yellow)
            img_array[:, :, 0] *= (1 + temp_shift / 1000)  # Red channel
            img_array[:, :, 1] *= (1 + temp_shift / 2000)  # Green channel
        else:  # Cooler (more blue)
            img_array[:, :, 2] *= (1 - temp_shift / 1000)  # Blue channel
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_shadow_with_params(
        self, 
        image: Image.Image, 
        shadow_type: str,
        shadow_intensity: float,
        shadow_params: dict
    ) -> Image.Image:
        """Apply shadow effects using pre-calculated parameters."""
        img_array = np.array(image).astype(np.float32)
        height, width = img_array.shape[:2]
        shadow_mask = np.ones((height, width))
        
        if shadow_type == 'linear':
            start_pos = int(shadow_params['linear_start_ratio'] * width)
            shadow_width = int(shadow_params['linear_width_ratio'] * width)
            start_x = max(0, start_pos - shadow_width // 2)
            end_x = min(width, start_pos + shadow_width // 2)
            shadow_mask[:, start_x:end_x] *= (1 - shadow_intensity)
        
        elif shadow_type == 'circular':
            center_x = int(shadow_params['circular_center_x_ratio'] * width)
            center_y = int(shadow_params['circular_center_y_ratio'] * height)
            radius = int(shadow_params['circular_radius_ratio'] * min(width, height))
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            shadow_mask[mask] *= (1 - shadow_intensity)
        
        img_array *= shadow_mask[:, :, np.newaxis]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _add_noise_with_params(
        self, 
        image: Image.Image, 
        noise_std: float,
        salt_pepper_prob: float
    ) -> Image.Image:
        """Add noise using pre-calculated parameters."""
        img_array = np.array(image)
        
        # Add Gaussian noise
        if noise_std > 0:
            gaussian_noise = self.rng.normal(0, noise_std, img_array.shape)
            img_array = img_array.astype(np.float32) + gaussian_noise
        
        # Add salt and pepper noise occasionally
        if salt_pepper_prob > 0 and self.rng.random() < 0.1:
            mask = self.rng.random(img_array.shape[:2]) < salt_pepper_prob
            img_array[mask] = self.rng.choice([0, 255], size=np.sum(mask))
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
