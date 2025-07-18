#!/usr/bin/env python3
"""
Test script to validate DINOv2 integration with ACT.
"""

import torch
import sys
import os

# Add the path to access the lerobot modules
sys.path.append('/home/robin/humanoid/humanoid_ws/src/unitree_IL_lerobot/unitree_lerobot')

def test_dinov2_backbone_creation():
    """Test creating DINOv2 backbones."""
    print("Testing DINOv2 backbone creation...")
    
    try:
        from lerobot.common.policies.act.modeling_act import create_dinov2_backbone, DINOv2Wrapper
        
        # Test different DINOv2 models
        models_to_test = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vits14_reg"]
        
        for model_name in models_to_test:
            print(f"Testing {model_name}...")
            
            # Create backbone
            backbone = create_dinov2_backbone(model_name)
            print(f"  ✓ Created {model_name} with feature_dim: {backbone.feature_dim}")
            
            # Create wrapper
            wrapper = DINOv2Wrapper(backbone)
            print(f"  ✓ Created wrapper")
            
            # Test forward pass
            test_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
            with torch.no_grad():
                output = wrapper(test_input)
                feature_map = output["feature_map"]
                print(f"  ✓ Forward pass successful. Output shape: {feature_map.shape}")
                
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_act_config_dinov2():
    """Test ACT configuration with DINOv2."""
    print("\nTesting ACT configuration with DINOv2...")
    
    try:
        from lerobot.common.policies.act.configuration_act import (
            DINOv2ACTConfig, 
            DINOv2RegisterACTConfig,
            DINOv2RegisterLargeACTConfig
        )
        
        # Test small DINOv2 config
        config_small = DINOv2ACTConfig()
        print(f"  ✓ Created DINOv2ACTConfig: {config_small.vision_backbone}")
        
        # Test register variants
        config_reg_small = DINOv2RegisterACTConfig()
        print(f"  ✓ Created DINOv2RegisterACTConfig: {config_reg_small.vision_backbone}")
        
        config_reg_large = DINOv2RegisterLargeACTConfig()
        print(f"  ✓ Created DINOv2RegisterLargeACTConfig: {config_reg_large.vision_backbone}")
        
        # Test validation
        config_small.__post_init__()
        config_reg_small.__post_init__()
        config_reg_large.__post_init__()
        print("  ✓ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_act_with_dinov2():
    """Test full ACT model with DINOv2."""
    print("\nTesting ACT model with DINOv2...")
    
    try:
        from lerobot.common.policies.act.configuration_act import DINOv2RegisterACTConfig
        from lerobot.common.policies.act.modeling_act import ACT
        from lerobot.configs.types import FeatureType, PolicyFeature
        
        # Create configuration with register variant
        config = DINOv2RegisterACTConfig()
        
        # Set up input and output features properly
        config.input_features = {
            "observation.images": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 224, 224)
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(14,)
            )
        }
        
        config.output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(14,)
            )
        }
        
        print(f"  ✓ Created config with vision_backbone: {config.vision_backbone}")
        
        # Create ACT model
        model = ACT(config)
        print("  ✓ Created ACT model with DINOv2 register backbone")
        
        # Create dummy batch for testing
        batch = {
            "observation.images": [torch.randn(2, 3, 224, 224)],  # List of camera views
            "observation.state": torch.randn(2, 14),
            "action": torch.randn(2, config.chunk_size, 14),
            "action_is_pad": torch.zeros(2, config.chunk_size, dtype=torch.bool),  # Required for training
        }
        
        # Test forward pass
        with torch.no_grad():
            output = model(batch)
            actions, (mu, log_sigma) = output
            print(f"  ✓ Forward pass successful. Actions shape: {actions.shape}")
            print(f"  ✓ Register variant working correctly!")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("DINOv2 Integration Test for ACT")
    print("=" * 40)
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_dinov2_backbone_creation():
        success_count += 1
        
    if test_act_config_dinov2():
        success_count += 1
        
    if test_act_with_dinov2():
        success_count += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {success_count}/{total_tests} successful")
    
    if success_count == total_tests:
        print("✅ All tests passed! DINOv2 integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
