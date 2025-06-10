#!/usr/bin/env python3
"""
Comprehensive Example: ACT with Pi0 Knowledge Backbone

This script demonstrates how to use the Pi0 Vision-Language-Action model 
as a knowledge backbone for the ACT policy, including:
- Basic configuration and usage
- Knowledge extraction and validation
- Training setup with proper parameter groups
- Performance evaluation and debugging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lerobot'))

try:
    from lerobot.common.policies.act.modeling_act import ACTPolicy, Pi0KnowledgeBackbone
    from lerobot.common.policies.act.configuration_act import Pi0ACTConfig, ACTConfigUnitreeG1, ACTConfig
    from lerobot.common.policies.act.pi0_knowledge_utils import (
        validate_pi0_integration,
        debug_pi0_batch_preparation,
        create_manipulation_prompts,
        compute_knowledge_similarity,
        prepare_pi0_batch_from_act
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

import torch
import numpy as np
from typing import Dict, List, Optional
import warnings

def create_sample_batch(batch_size: int = 1, task_description: str = "pick up the red cup") -> Dict:
    """Create a sample batch for testing pi0 integration."""
    return {
        "observation.images": [
            torch.randn(batch_size, 3, 224, 224),  # Main camera
            torch.randn(batch_size, 3, 224, 224),  # Secondary camera
        ],
        "observation.state": torch.randn(batch_size, 16),  # Robot joint states
        "observation.environment_state": torch.randn(batch_size, 8),  # Environment state
        "task": [task_description] * batch_size,
        "action": torch.randn(batch_size, 20, 14),  # For training mode
        "action_is_pad": torch.zeros(batch_size, 20, dtype=torch.bool),
    }

def example_pi0_act_usage():
    """Example of using ACT with Pi0 knowledge backbone."""
    print("=== Basic Pi0 ACT Usage ===")
    
    # Method 1: Use the specialized Pi0ACTConfig
    config = Pi0ACTConfig()
    print(f"Using Pi0ACTConfig with model path: {config.pi0_model_path}")
    
    # Method 2: Use UnitreeG1 config with Pi0 enabled (already configured)
    # config = ACTConfigUnitreeG1()
    
    try:
        # Create policy with pi0 knowledge backbone
        # Note: This may fail if pi0 model is not available
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            policy = ACTPolicy(config)
        
        pi0_available = policy.model.pi0_knowledge_backbone is not None
        if pi0_available:
            pi0_working = policy.model.pi0_knowledge_backbone.pi0_available
        else:
            pi0_working = False
            
        print(f"Pi0 knowledge backbone created: {pi0_available}")
        print(f"Pi0 model loaded successfully: {pi0_working}")
        
        # Example batch structure for inference
        batch = create_sample_batch(batch_size=2, task_description="grasp the blue object")
        
        # Forward pass - pi0 will provide manipulation knowledge if available
        with torch.no_grad():
            if pi0_working:
                action = policy.select_action(batch)
                print(f"Generated action shape: {action.shape}")
                
                # Test knowledge extraction directly
                knowledge = policy.model.pi0_knowledge_backbone(batch)
                print(f"Extracted knowledge shape: {knowledge.shape}")
                print(f"Knowledge statistics: mean={knowledge.mean():.4f}, std={knowledge.std():.4f}")
            else:
                print("Pi0 not available - would use fallback zero knowledge")
        
        # Training setup with proper parameter groups
        optimizer_params = policy.get_optim_params()
        print(f"Number of parameter groups: {len(optimizer_params)}")
        for i, group in enumerate(optimizer_params):
            param_count = sum(p.numel() for p in group["params"])
            lr = group.get("lr", config.optimizer_lr)
            print(f"  Group {i}: {param_count:,} parameters, lr={lr}")
            
    except Exception as e:
        print(f"Note: Pi0 model not available ({e})")
        print("This is expected if pi0 is not installed or model not downloaded")
        print("The integration will work with fallback zero knowledge")

def example_knowledge_validation():
    """Demonstrate knowledge extraction validation."""
    print("\n=== Knowledge Validation ===")
    
    config = Pi0ACTConfig()
    config.pi0_model_path = None  # Force fallback mode for testing
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            policy = ACTPolicy(config)
        
        sample_batch = create_sample_batch(batch_size=1)
        
        # Validate pi0 integration
        validation_results = validate_pi0_integration(
            policy.model.pi0_knowledge_backbone, 
            sample_batch
        )
        
        print("Validation Results:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")
            
        # Test batch preparation debugging
        pi0_batch = prepare_pi0_batch_from_act(sample_batch)
        
        debug_info = debug_pi0_batch_preparation(sample_batch, pi0_batch)
        print("\nBatch Preparation Debug:")
        for key, value in debug_info.items():
            if key != "warnings" or value:  # Only show warnings if there are any
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Validation error: {e}")

def example_task_specific_prompts():
    """Demonstrate task-specific manipulation prompts."""
    print("\n=== Task-Specific Knowledge Prompts ===")
    
    task_types = ["grasping", "pouring", "insertion", None]
    
    for task_type in task_types:
        prompts = create_manipulation_prompts(task_type)
        task_name = task_type if task_type else "general"
        print(f"\n{task_name.upper()} Task Prompts:")
        for i, prompt in enumerate(prompts[:3], 1):  # Show first 3
            print(f"  {i}. {prompt}")

def example_knowledge_similarity():
    """Demonstrate knowledge similarity computation."""
    print("\n=== Knowledge Similarity Analysis ===")
    
    # Simulate extracted knowledge features
    batch_size = 3
    feature_dim = 512
    
    # Simulate knowledge from different scenarios
    grasping_knowledge = torch.randn(batch_size, feature_dim)
    pouring_knowledge = torch.randn(batch_size, feature_dim)
    
    # Add some correlation to pouring knowledge
    pouring_knowledge = 0.7 * grasping_knowledge + 0.3 * pouring_knowledge
    
    # Compute similarities
    similarity_scores = compute_knowledge_similarity(grasping_knowledge, pouring_knowledge)
    
    print(f"Knowledge similarity scores: {similarity_scores}")
    print(f"Average similarity: {similarity_scores.mean():.4f}")
    print(f"Similarity interpretation:")
    print(f"  > 0.8: Very similar manipulation strategies")
    print(f"  0.5-0.8: Moderately related strategies") 
    print(f"  < 0.5: Different manipulation approaches")

def example_configuration_options():
    """Show different configuration options for pi0 integration."""
    print("\n=== Configuration Options ===")
    
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
    
    # Lightweight configuration for faster inference
    lightweight_config = ACTConfig()
    lightweight_config.use_pi0_knowledge = True
    lightweight_config.pi0_model_path = "lerobot/pi0_lite"
    lightweight_config.pi0_knowledge_weight = 0.5
    lightweight_config.dim_model = 256
    lightweight_config.pi0_feature_dim = 512
    
    configs = {
        "basic": basic_config,
        "advanced": advanced_config, 
        "custom": custom_config,
        "lightweight": lightweight_config
    }
    
    for name, cfg in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print(f"  Use Pi0: {cfg.use_pi0_knowledge}")
        print(f"  Pi0 Model: {getattr(cfg, 'pi0_model_path', 'Not set')}")
        print(f"  Knowledge Weight: {getattr(cfg, 'pi0_knowledge_weight', 'Default')}")
        print(f"  Feature Dim: {getattr(cfg, 'pi0_feature_dim', 'Default')}")
        print(f"  Model Dim: {cfg.dim_model}")
        print(f"  Chunk Size: {cfg.chunk_size}")

def example_training_considerations():
    """Show training considerations for pi0 integration."""
    print("\n=== Training Considerations ===")
    
    config = Pi0ACTConfig()
    
    print("Parameter Management:")
    print(f"  Main learning rate: {config.optimizer_lr}")
    print(f"  Backbone learning rate: {config.optimizer_lr_backbone}")
    print(f"  Pi0 projection rate: {config.optimizer_lr * 0.5} (auto-computed)")
    
    print("\nMemory Considerations:")
    print("  ✓ Pi0 parameters are frozen (no gradients)")
    print("  ✓ Only projection layers require training")
    print("  ✓ Knowledge extraction uses torch.no_grad()")
    
    print("\nPerformance Tips:")
    print("  • Start with pi0_knowledge_weight=0.5-0.7")
    print("  • Use larger batch sizes for stable knowledge extraction")
    print("  • Monitor knowledge statistics during training")
    print("  • Consider temporal ensembling for smoother actions")
    
    print("\nTroubleshooting:")
    print("  • If knowledge is all zeros, check pi0 model loading")
    print("  • If training is unstable, reduce knowledge weight")
    print("  • If actions are jerky, enable temporal ensembling")

if __name__ == "__main__":
    print("=== Pi0 ACT Knowledge Backbone Comprehensive Example ===\n")
    
    # Run all examples
    example_pi0_act_usage()
    example_knowledge_validation()
    example_task_specific_prompts()
    example_knowledge_similarity() 
    example_configuration_options()
    example_training_considerations()
    
    print("\n=== Key Benefits of Pi0 Integration ===")
    benefits = [
        "🧠 Rich manipulation knowledge from pi0 VLA model",
        "🤖 Better grasping affordance understanding", 
        "🎯 Faster learning on new manipulation tasks",
        "🔧 Pre-learned inverse kinematics knowledge",
        "📊 Stable training with frozen pi0 backbone",
        "⚡ Efficient knowledge transfer without full VLA training",
        "🎛️ Flexible integration with existing ACT workflows",
        "🔍 Built-in validation and debugging tools"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
        
    print("\n=== Next Steps ===")
    print("1. Install pi0 dependencies if not already installed")
    print("2. Download a pi0 model checkpoint")
    print("3. Configure pi0_model_path in your config")
    print("4. Test knowledge extraction with your data")
    print("5. Tune pi0_knowledge_weight for your task")
    print("6. Monitor training metrics and knowledge statistics")
