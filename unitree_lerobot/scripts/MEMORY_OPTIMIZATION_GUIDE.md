# DINOv2 Memory Optimization Guide

This guide provides strategies for running DINOv2-based ACT policies with limited GPU memory.

## Memory Optimization Strategies

### 1. **Batch Size Reduction**
- **Minimum**: Start with `batch_size: 1` for DINOv2
- **ResNet comparison**: ResNet can handle `batch_size: 8-16`
- **Trade-off**: Smaller batch sizes may lead to noisier gradients

### 2. **Camera Reduction**
- **Strategy**: Use `cameras: ["cam_left_active", "cam_right_active"]` instead of all 6 cameras
- **Memory impact**: ~60-70% memory reduction
- **Performance impact**: May reduce spatial awareness but often sufficient for many tasks

### 3. **Architecture Downsizing**
```yaml
# Memory-efficient architecture
dim_model: 256          # Instead of 512
n_heads: 4              # Instead of 8  
n_encoder_layers: 2     # Instead of 4
n_decoder_layers: 1     # Instead of 2
latent_dim: 16          # Instead of 32
n_vae_encoder_layers: 2 # Instead of 4
```

### 4. **Mixed Precision Training**
```yaml
use_amp: true  # Enable Automatic Mixed Precision
```
- **Memory savings**: ~40-50% reduction
- **Speed improvement**: ~1.5-2x faster training
- **Precision impact**: Minimal for most tasks

### 5. **Sequence Length Reduction**
```yaml
chunk_size: 50  # Instead of 60
```
- **Memory impact**: Linear reduction with sequence length
- **Performance impact**: May affect long-horizon planning

## Complete Memory-Optimized Configuration

```yaml
- name: "dinov2_memory_optimized"
  config:
    vision_backbone: "dinov2_vits14"
    pretrained_backbone_weights: null
    batch_size: 1
    use_amp: true
    
    # Camera configuration
    cameras: '["cam_left_active", "cam_right_active"]'
    
    # Reduced architecture
    chunk_size: 50
    dim_model: 256
    n_heads: 4
    n_encoder_layers: 2
    n_decoder_layers: 1
    latent_dim: 16
    n_vae_encoder_layers: 2
    
    # Optimized learning rates for DINOv2
    optimizer_lr: 5e-6
    optimizer_lr_backbone: 1e-6
    
    # Standard VAE settings
    use_vae: true
    kl_weight: 10.0
    dropout: 0.1
```

## Memory Requirements by Configuration

| Configuration | Estimated GPU Memory | Batch Size |
|--------------|---------------------|------------|
| ResNet18 (all cameras) | ~8GB | 8 |
| ResNet18 (2 cameras) | ~4GB | 16 |
| DINOv2 (all cameras) | ~22GB+ | OOM |
| DINOv2 (2 cameras, full arch) | ~16GB | 1-2 |
| DINOv2 (2 cameras, reduced) | ~8GB | 1 |
| DINOv2 (2 cameras, reduced + AMP) | ~4-6GB | 1-2 |

## Troubleshooting OOM Errors

### If you still get OOM with the optimized config:

1. **Reduce to single camera**: `cameras: '["cam_left_active"]'`
2. **Further reduce architecture**:
   ```yaml
   dim_model: 128
   n_heads: 2
   n_encoder_layers: 1
   n_decoder_layers: 1
   latent_dim: 8
   ```
3. **Consider gradient accumulation** (if supported)
4. **Use CPU offloading** (if available in your framework)

### Alternative Approaches:

1. **Use smaller DINOv2 variant**: Consider `dinov2_vits14` vs `dinov2_vitb14` or `dinov2_vitl14`
2. **Feature extraction mode**: Freeze DINOv2 and only train the transformer layers
3. **Distributed training**: Use multiple GPUs if available

## Performance vs Memory Trade-offs

| Strategy | Memory Savings | Performance Impact |
|----------|---------------|-------------------|
| Batch size 1 | High | Moderate (noisy gradients) |
| 2 cameras only | Very High | Low-Moderate |
| Reduced architecture | High | Moderate |
| Mixed precision | Moderate | Very Low |
| Shorter sequences | Moderate | Low-Moderate |

## Recommended Starting Point

For most users with limited GPU memory (8-12GB), start with:

```yaml
vision_backbone: "dinov2_vits14"
batch_size: 1
use_amp: true
cameras: '["cam_left_active", "cam_right_active"]'
dim_model: 256
n_heads: 4
n_encoder_layers: 2
n_decoder_layers: 1
```

If this still fails, progressively reduce the architecture further or consider using ResNet18/34 as a baseline comparison.
