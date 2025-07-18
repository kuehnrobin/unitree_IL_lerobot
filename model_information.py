import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Replace with your actual checkpoint path
model_path = "unitree_lerobot/lerobot/outputs/train/2025-07-04/cubes_box_act_100/checkpoints/last/pretrained_model/"

try:
    policy = ACTPolicy.from_pretrained(model_path)
    
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"🤖 Model: {model_path}")
    print(f"📊 Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"🎯 Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"💾 Model size (approx): {total*4/1e6:.1f} MB")
    
except Exception as e:
    print(f"Error loading model: {e}")