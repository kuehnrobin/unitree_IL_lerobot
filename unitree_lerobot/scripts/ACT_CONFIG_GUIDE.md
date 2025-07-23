# ACT Policy Configuration Guide for Ablation Studies

This guide explains all the ACT (Action Chunking Transformer) policy parameters you can configure in your `custom_ablation.yaml` file for comprehensive ablation studies.

## Dataset Parameters

You can specify different datasets for different experiments:

```yaml
config:
  dataset_repo: "kuehnrobin/g1_cubes_box_s_61"  # Override the default dataset
  batch_size: 8  # Override the default batch size
```

This allows you to:
- Test the same model configuration on different datasets
- Compare dataset difficulty/characteristics
- Find dataset-specific optimal configurations
- Adjust batch size for memory constraints (e.g., smaller batches for DINOv2)

## Feature Selection Parameters

These parameters control which input features are used:

```yaml
config:
  # Camera configuration
  cameras: '["cam_left_active", "cam_right_active"]'  # Specify which cameras to use
  exclude_cameras: '["cam_left_head", "cam_right_head"]'  # Exclude specific cameras
  
  # Joint state features
  use_joint_positions: true    # Include joint positions (default: true)
  use_joint_velocities: false  # Include joint velocities
  use_joint_torques: false     # Include joint torques
  
  # Sensor features
  use_pressure_sensors: false  # Include pressure sensor data
  
  # Joint group filtering
  joint_groups: '["arm", "hand"]'           # Include only specific joint groups
  exclude_joint_groups: '["camera"]'       # Exclude specific joint groups
  
  # Custom state selection
  custom_state_indices: '[0, 1, 2, 3, 4, 5]'  # Manually specify state indices
```

## Vision Backbone Parameters

Configure the visual encoder backbone:

```yaml
config:
  # Vision backbone options
  vision_backbone: "resnet18"  # Options: "resnet18", "resnet34", "resnet50", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
  pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"  # Use pretrained weights or null
  replace_final_stride_with_dilation: false  # Replace final stride with dilation
```

### Popular Vision Backbone Combinations:
- **ResNet18**: `vision_backbone: "resnet18"`, `pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"`, `batch_size: 12`
- **ResNet34**: `vision_backbone: "resnet34"`, `pretrained_backbone_weights: "ResNet34_Weights.IMAGENET1K_V1"`, `batch_size: 10`
- **DINOv2 Small**: `vision_backbone: "dinov2_vits14"`, `pretrained_backbone_weights: null`, `batch_size: 6`
- **DINOv2 Base**: `vision_backbone: "dinov2_vitb14"`, `pretrained_backbone_weights: null`, `batch_size: 4`

## Batch Size and Memory Considerations

Configure batch size for different model architectures:

```yaml
config:
  batch_size: 8  # Adjust based on model size and available GPU memory
```

### Recommended Batch Sizes:
- **ResNet18/34 + Small Transformer**: `batch_size: 12-16`
- **ResNet18/34 + Large Transformer**: `batch_size: 6-8`
- **DINOv2 Small + Small Transformer**: `batch_size: 6-8`
- **DINOv2 Small + Large Transformer**: `batch_size: 4-6`
- **DINOv2 Base/Large**: `batch_size: 2-4`

### Memory-Optimized Configurations:
```yaml
# For limited GPU memory
- name: "memory_efficient"
  config:
    vision_backbone: "resnet18"
    dim_model: 256
    n_encoder_layers: 2
    batch_size: 16
    
# For high-end GPU with lots of memory
- name: "memory_intensive"
  config:
    vision_backbone: "dinov2_vitb14"
    dim_model: 768
    n_encoder_layers: 6
    batch_size: 4
```

## Transformer Architecture Parameters

Configure the main transformer architecture:

```yaml
config:
  # Model dimensions
  dim_model: 512           # Main hidden dimension (typical: 256, 512, 768, 1024)
  n_heads: 8              # Number of attention heads (typical: 4, 8, 12, 16)
  dim_feedforward: 3200   # Feed-forward layer dimension (typically 4x dim_model)
  
  # Transformer layers
  n_encoder_layers: 4     # Number of encoder layers (typical: 2-8)
  n_decoder_layers: 1     # Number of decoder layers (typical: 1-4)
  
  # Other transformer settings
  pre_norm: false                    # Use pre-normalization
  feedforward_activation: "relu"     # Activation function ("relu", "gelu")
  dropout: 0.1                      # Dropout rate (typical: 0.0-0.3)
```

### Architecture Size Presets:
- **Small**: `dim_model: 256`, `n_heads: 4`, `dim_feedforward: 1024`, `n_encoder_layers: 2`
- **Medium**: `dim_model: 512`, `n_heads: 8`, `dim_feedforward: 2048`, `n_encoder_layers: 4`
- **Large**: `dim_model: 768`, `n_heads: 12`, `dim_feedforward: 3072`, `n_encoder_layers: 6`

## Action Chunking Parameters

Configure how actions are predicted and executed:

```yaml
config:
  chunk_size: 100        # Number of actions predicted at once (typical: 50-200)
  n_action_steps: 100    # Number of actions executed per inference (≤ chunk_size)
  n_obs_steps: 1         # Number of observation steps (usually 1)
```

### Common Chunking Strategies:
- **Short horizon**: `chunk_size: 50`, `n_action_steps: 25`
- **Medium horizon**: `chunk_size: 100`, `n_action_steps: 50`
- **Long horizon**: `chunk_size: 200`, `n_action_steps: 100`

## VAE Parameters

Configure the Variational Autoencoder component:

```yaml
config:
  use_vae: true              # Enable/disable VAE
  latent_dim: 32            # VAE latent dimension (typical: 16-64)
  n_vae_encoder_layers: 4   # Number of VAE encoder layers (typical: 2-6)
  kl_weight: 10.0           # KL divergence loss weight (typical: 1.0-20.0)
```

## Temporal Ensembling

Configure temporal ensembling for smoother actions:

```yaml
config:
  temporal_ensemble_coeff: 0.01  # Ensembling coefficient (null to disable, typical: 0.001-0.1)
  # Note: Requires n_action_steps: 1 when enabled
```

## Optimization Parameters

Configure learning rates and training dynamics:

```yaml
config:
  optimizer_lr: 1e-5              # Main learning rate (typical: 1e-6 to 1e-4)
  optimizer_lr_backbone: 1e-5     # Backbone learning rate (typically same or lower)
  optimizer_weight_decay: 1e-4    # Weight decay (typical: 1e-5 to 1e-3)
```

## Normalization Parameters

Configure how different modalities are normalized:

```yaml
config:
  normalization_mapping:
    VISUAL: "MEAN_STD"    # Options: "MEAN_STD", "MIN_MAX"
    STATE: "MEAN_STD"     # Options: "MEAN_STD", "MIN_MAX"  
    ACTION: "MEAN_STD"    # Options: "MEAN_STD", "MIN_MAX"
```

## Complete Example Configurations

### Minimal Configuration (Active Cameras Only)
```yaml
- name: "minimal_setup"
  config:
    cameras: '["cam_left_active", "cam_right_active"]'
    use_joint_velocities: false
    use_joint_torques: false
    use_pressure_sensors: false
    dim_model: 256
    n_heads: 4
    n_encoder_layers: 2
    n_decoder_layers: 1
```

### Robust Configuration (All Features, Large Model)
```yaml
- name: "robust_setup"
  config:
    vision_backbone: "resnet34"
    pretrained_backbone_weights: "ResNet34_Weights.IMAGENET1K_V1"
    dim_model: 768
    n_heads: 12
    dim_feedforward: 3072
    n_encoder_layers: 6
    n_decoder_layers: 2
    dropout: 0.15
    latent_dim: 48
    kl_weight: 15.0
    chunk_size: 150
    n_action_steps: 75
```

### DINOv2 Vision Configuration
```yaml
- name: "dinov2_vision"
  config:
    vision_backbone: "dinov2_vits14"
    pretrained_backbone_weights: null
    dim_model: 384  # Match DINOv2 output dimension
    temporal_ensemble_coeff: 0.005
    n_action_steps: 1  # Required for temporal ensembling
```

## Dataset Ablation Examples

### Cross-Dataset Comparison
```yaml
- name: "baseline_dataset_a"
  config:
    dataset_repo: "kuehnrobin/dataset_a"
    
- name: "baseline_dataset_b"  
  config:
    dataset_repo: "kuehnrobin/dataset_b"
```

### Dataset-Specific Optimization
```yaml
- name: "simple_task_config"
  config:
    dataset_repo: "kuehnrobin/simple_task"
    dim_model: 256
    n_encoder_layers: 2
    chunk_size: 50
    
- name: "complex_task_config"
  config:
    dataset_repo: "kuehnrobin/complex_task"
    vision_backbone: "dinov2_vits14"
    dim_model: 768
    n_encoder_layers: 6
    chunk_size: 150
```

### Combined Dataset + Architecture Ablation
```yaml
- name: "dataset_a_optimized"
  config:
    dataset_repo: "kuehnrobin/dataset_a"
    cameras: '["cam_left_active", "cam_right_active"]'
    vision_backbone: "resnet34"
    chunk_size: 75
    
- name: "dataset_b_optimized"
  config:
    dataset_repo: "kuehnrobin/dataset_b"
    vision_backbone: "dinov2_vits14"
    pretrained_backbone_weights: null
    dim_model: 384
    temporal_ensemble_coeff: 0.01
    n_action_steps: 1
```

## Tips for Effective Ablations

1. **Start Simple**: Begin with minimal configurations and gradually add complexity
2. **Dataset Selection**: Use `dataset_repo` to test configurations across different tasks
3. **Vision Backbone**: DINOv2 often works better but is slower; ResNet is faster
3. **Architecture Size**: Larger models need more data and compute but may perform better
4. **Chunking**: Longer horizons are better for complex tasks but harder to train
5. **VAE**: Helps with action diversity but adds complexity
6. **Temporal Ensembling**: Great for smooth execution but requires n_action_steps=1

## Running the Ablation Study

```bash
python unitree_lerobot/scripts/run_ablation_study.py \
    --config_file unitree_lerobot/examples/custom_ablation.yaml \
    --dataset_repo your_dataset \
    --wandb_project your_project \
    --steps 50000 \
    --eval_freq 10000 \
    --save_freq 10000 \
    --log_freq 1000 \
    --batch_size 12
```
