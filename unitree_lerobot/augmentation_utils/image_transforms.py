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
    
    def augment_image_comprehensive(
        self, 
        image: Image.Image,
        config: dict = None
    ) -> Image.Image:
        """
        Apply comprehensive image augmentation pipeline.
        
        Args:
            image: Input PIL Image
            config: Configuration dictionary for augmentation parameters
            
        Returns:
            Fully augmented image
        """
        if config is None:
            config = {}
        
        # Apply lighting adjustments
        image = self.adjust_lighting(
            image,
            brightness_range=config.get('brightness_range', (0.7, 1.3)),
            contrast_range=config.get('contrast_range', (0.8, 1.2)),
            saturation_range=config.get('saturation_range', (0.8, 1.2))
        )
        
        # Apply color temperature adjustment
        if self.rng.random() < config.get('temperature_prob', 0.4):
            image = self.adjust_color_temperature(
                image, 
                temperature_shift=config.get('temperature_shift', 30)
            )
        
        # Apply motion blur
        image = self.simulate_motion_blur(
            image,
            blur_probability=config.get('blur_prob', 0.15),
            max_blur_radius=config.get('max_blur_radius', 1.0)
        )
        
        # Apply shadow effects
        image = self.apply_random_shadow(
            image,
            shadow_probability=config.get('shadow_prob', 0.2),
            shadow_intensity=config.get('shadow_intensity', 0.3)
        )
        
        # Add camera noise
        image = self.add_camera_noise(
            image,
            noise_std=config.get('noise_std', 3.0),
            salt_pepper_prob=config.get('salt_pepper_prob', 0.001)
        )
        
        return image
