#!/usr/bin/env python3
"""
Model Download Script for Cluster

This script downloads DINOv2 and ResNet models locally so they can be used on cluster nodes
without internet access. Run this on a machine with internet access, then transfer
the models to your cluster's $BIGWORK directory.

Usage:
    python download_dinov2_models.py --output_dir $BIGWORK/torch_models

The script will download all DINOv2 variants and ResNet models and save them as .pth files.
"""

import argparse
import os
import torch
import torchvision
import sys
from pathlib import Path


def download_dinov2_model(model_name: str, output_dir: str):
    """Download a specific DINOv2 model and save it locally."""
    
    print(f"Downloading DINOv2 {model_name}...")
    
    try:
        # Load model from torch hub
        model = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=True)
        model.eval()
        
        # Save model to local file
        output_path = os.path.join(output_dir, f"{model_name}.pth")
        torch.save(model, output_path)
        
        print(f"✓ Saved {model_name} to {output_path}")
        
        # Also save the state dict separately (alternative loading method)
        state_dict_path = os.path.join(output_dir, f"{model_name}_state_dict.pth")
        torch.save(model.state_dict(), state_dict_path)
        
        print(f"✓ Saved {model_name} state dict to {state_dict_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False


def download_resnet_model(model_name: str, output_dir: str):
    """Download a specific ResNet model with pretrained weights and save it locally."""
    
    print(f"Downloading ResNet {model_name}...")
    
    try:
        # Load model with pretrained weights
        model = getattr(torchvision.models, model_name)(weights='IMAGENET1K_V1')
        model.eval()
        
        # Save the pretrained state dict
        state_dict_path = os.path.join(output_dir, f"{model_name}_pretrained.pth")
        torch.save(model.state_dict(), state_dict_path)
        
        print(f"✓ Saved {model_name} pretrained weights to {state_dict_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download models for offline cluster use')
    parser.add_argument('--output_dir', required=True, 
                       help='Directory to save models (e.g., $BIGWORK/torch_models)')
    parser.add_argument('--dinov2_models', nargs='+', 
                       default=['dinov2_vits14', 'dinov2_vits14_reg', 
                               'dinov2_vitb14', 'dinov2_vitb14_reg',
                               'dinov2_vitl14', 'dinov2_vitl14_reg'],
                       help='DINOv2 models to download')
    parser.add_argument('--resnet_models', nargs='+',
                       default=['resnet18', 'resnet34', 'resnet50'],
                       help='ResNet models to download')
    parser.add_argument('--skip_dinov2', action='store_true',
                       help='Skip downloading DINOv2 models')
    parser.add_argument('--skip_resnet', action='store_true',
                       help='Skip downloading ResNet models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {output_dir}")
    print("-" * 50)
    
    # Download DINOv2 models
    dinov2_success = 0
    if not args.skip_dinov2:
        print("Downloading DINOv2 models...")
        for model_name in args.dinov2_models:
            if download_dinov2_model(model_name, str(output_dir)):
                dinov2_success += 1
            print()
    
    # Download ResNet models
    resnet_success = 0
    if not args.skip_resnet:
        print("Downloading ResNet models...")
        for model_name in args.resnet_models:
            if download_resnet_model(model_name, str(output_dir)):
                resnet_success += 1
            print()
    
    print("-" * 50)
    print(f"Downloaded {dinov2_success}/{len(args.dinov2_models) if not args.skip_dinov2 else 0} DINOv2 models successfully")
    print(f"Downloaded {resnet_success}/{len(args.resnet_models) if not args.skip_resnet else 0} ResNet models successfully")
    
    # Create a verification script
    verify_script = output_dir / "verify_models.py"
    with open(verify_script, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
'''Verify downloaded models'''
import torch
import torchvision
import os

dinov2_models = {args.dinov2_models if not args.skip_dinov2 else []}
resnet_models = {args.resnet_models if not args.skip_resnet else []}
model_dir = "{output_dir}"

print("Verifying downloaded models...")

# Check DINOv2 models
for model_name in dinov2_models:
    model_path = os.path.join(model_dir, f"{{model_name}}.pth")
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path, map_location='cpu')
            print(f"✓ DINOv2 {{model_name}}: OK")
        except Exception as e:
            print(f"✗ DINOv2 {{model_name}}: Error - {{e}}")
    else:
        print(f"✗ DINOv2 {{model_name}}: File not found")

# Check ResNet models  
for model_name in resnet_models:
    model_path = os.path.join(model_dir, f"{{model_name}}_pretrained.pth")
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"✓ ResNet {{model_name}}: OK")
        except Exception as e:
            print(f"✗ ResNet {{model_name}}: Error - {{e}}")
    else:
        print(f"✗ ResNet {{model_name}}: File not found")
""")
    
    print(f"Created verification script: {verify_script}")
    print("\nTo verify models on cluster, run:")
    print(f"python {verify_script}")
    
    print("\nTo transfer to cluster:")
    print(f"scp -r {output_dir} username@cluster:$BIGWORK/")


if __name__ == "__main__":
    main()
