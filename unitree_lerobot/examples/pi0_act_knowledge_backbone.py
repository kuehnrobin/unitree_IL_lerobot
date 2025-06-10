#!/usr/bin/env python3
"""
Example usage of ACT with Pi0 Knowledge Backbone

This script demonstrates how to use the Pi0 Vision-Language-Action model 
as a knowledge backbone for the ACT policy, including:
- Basic configuration and usage
- Knowledge extraction and validation
- Training setup with proper parameter groups
- Performance evaluation and debugging
"""

from lerobot.common.policies.act.modeling_act import ACTPolicy, Pi0KnowledgeBackbone
from lerobot.common.policies.act.configuration_act import Pi0ACTConfig, ACTConfigUnitreeG1, ACTConfig
from lerobot.common.policies.act.pi0_knowledge_utils import (
    validate_pi0_integration,
    debug_pi0_batch_preparation,
    create_manipulation_prompts,
    compute_knowledge_similarity
)
import torch
import numpy as np
from typing import Dict, List, Optional
import warnings

def example_pi0_act_usage():
    """Example of using ACT with Pi0 knowledge backbone."""
    
    # Method 1: Use the specialized Pi0ACTConfig
    config = Pi0ACTConfig()
    
    # Method 2: Use UnitreeG1 config with Pi0 enabled (already configured)
    # config = ACTConfigUnitreeG1()
    
    # Create policy with pi0 knowledge backbone
    policy = ACTPolicy(config)
    
    print(f"Pi0 knowledge backbone enabled: {policy.model.pi0_knowledge_backbone is not None}")
    
    # Example batch structure for inference
    batch = {
        "observation.images": [torch.randn(1, 3, 224, 224)],  # Camera images
        "observation.state": torch.randn(1, 10),               # Robot state
        "task": ["pick up the red cup"],                       # Task description for pi0
    }
    
    # Forward pass - pi0 will provide manipulation knowledge
    with torch.no_grad():
        action = policy.select_action(batch)
        print(f"Generated action shape: {action.shape}")
    
    # Training setup with proper parameter groups
    optimizer_params = policy.get_optim_params()
    print(f"Number of parameter groups: {len(optimizer_params)}")
    for i, group in enumerate(optimizer_params):
        param_count = sum(p.numel() for p in group["params"])
        lr = group.get("lr", config.optimizer_lr)
        print(f"  Group {i}: {param_count:,} parameters, lr={lr}")

def example_configuration_options():
    """Show different configuration options for pi0 integration."""
    
    # Basic configuration with pi0
    basic_config = ACTConfigUnitreeG1()
    basic_config.use_pi0_knowledge = True
    basic_config.pi0_model_path = "lerobot/pi0"
    
    # Advanced configuration
    advanced_config = Pi0ACTConfig()
    advanced_config.pi0_knowledge_weight = 0.8  # Higher weight for pi0 knowledge
    advanced_config.pi0_num_knowledge_queries = 8  # More knowledge queries
    
    # Custom configuration for specific tasks
    custom_config = Pi0ACTConfig()
    custom_config.chunk_size = 200
    custom_config.n_action_steps = 50
    custom_config.dim_model = 1536  # Larger model
    custom_config.pi0_feature_dim = 1536
    
    configs = {
        "basic": basic_config,
        "advanced": advanced_config, 
        "custom": custom_config
    }
    
    for name, cfg in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print(f"  Use Pi0: {cfg.use_pi0_knowledge}")
        print(f"  Pi0 Model: {cfg.pi0_model_path}")
        print(f"  Knowledge Weight: {cfg.pi0_knowledge_weight}")
        print(f"  Feature Dim: {cfg.pi0_feature_dim}")
        print(f"  Model Dim: {cfg.dim_model}")

if __name__ == "__main__":
    print("=== Pi0 ACT Knowledge Backbone Example ===\n")
    
    print("1. Basic Usage:")
    example_pi0_act_usage()
    
    print("\n2. Configuration Options:")
    example_configuration_options()
    
    print("\n=== Key Benefits ===")
    benefits = [
        "🧠 Rich manipulation knowledge from pi0 VLA model",
        "🤖 Better grasping affordance understanding", 
        "🎯 Faster learning on new manipulation tasks",
        "🔧 Pre-learned inverse kinematics knowledge",
        "📊 Stable training with frozen pi0 backbone",
        "⚡ Efficient knowledge transfer without full VLA training"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
