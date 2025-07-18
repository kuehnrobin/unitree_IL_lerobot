#!/usr/bin/env python
"""
Test script to verify the ablation study configuration loading.
This helps debug issues without actually running the full training.
"""

import yaml
import sys
from pathlib import Path

def test_ablation_config(config_file):
    """Test loading and parsing of the ablation configuration."""
    print(f"Testing ablation config: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        experiments = config.get('experiments', [])
        print(f"Found {len(experiments)} experiments:")
        
        for i, exp in enumerate(experiments):
            name = exp.get('name', f'experiment_{i}')
            config_dict = exp.get('config', {})
            
            print(f"\n  Experiment {i+1}: {name}")
            if not config_dict:
                print("    - Uses all default features")
            else:
                for key, value in config_dict.items():
                    # Convert boolean values to strings for command line
                    if isinstance(value, bool):
                        value_str = str(value).lower()
                    else:
                        value_str = str(value)
                    print(f"    - --feature_selection.{key}={value_str}")
        
        print(f"\n✓ Configuration is valid!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ablation_config.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    if not Path(config_file).exists():
        print(f"Config file does not exist: {config_file}")
        sys.exit(1)
    
    success = test_ablation_config(config_file)
    sys.exit(0 if success else 1)
