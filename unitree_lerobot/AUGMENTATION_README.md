# Dataset Augmentation for Teleoperation Data

This module provides comprehensive data augmentation capabilities for teleoperation datasets to improve imitation learning performance. The augmentation script includes episode weighting, lighting variations, noise injection, and other robustness techniques.

## Overview

The `argument_data.py` script augments teleoperation datasets with the following features:

1. **Episode Weighting**: Multiply optimal episodes to increase their representation in the dataset while preserving recovery episodes for robustness.

2. **Image Augmentation**: Apply realistic lighting variations, color adjustments, motion blur, shadows, and noise to improve visual robustness.

3. **Joint Noise Injection**: Add controlled noise to joint coordinates to improve policy robustness without disrupting the task objective.

4. **Proper Episode Management**: Automatically handles episode numbering and directory structure creation.

## Installation

### Prerequisites

Ensure you have the main lerobot requirements installed, then install additional dependencies:

```bash
pip install -r augmentation_requirements.txt
```

The additional requirements include:
- `Pillow` (PIL) for advanced image processing
- `opencv-python` for computer vision operations
- `tqdm` for progress bars
- `numpy` for numerical operations

## Usage

### Basic Usage

```bash
python unitree_lerobot/argument_data.py \
    --input_dataset_path /media/robin/DATA/stack_cube_left \
    --output_dataset_path /media/robin/DATA/argumented_stack_cube_left
```

### Advanced Usage with Episode Weighting

```bash
python unitree_lerobot/argument_data.py \
    --input_dataset_path /media/robin/DATA/stack_cube_left \
    --output_dataset_path /media/robin/DATA/argumented_stack_cube_left \
    --optimal_episodes "1,3,5-8,12" \
    --recovery_episodes "2,4,9-11" \
    --optimal_weight 3.0 \
    --joint_noise_std 0.03 \
    --augmentation_probability 0.7
```

### Configuration Options

#### Path Configuration
- `--input_dataset_path`: Path to input dataset directory (default: `/media/robin/DATA/stack_cube_left`)
- `--output_dataset_path`: Path to output augmented dataset directory (default: `/media/robin/DATA/argumented_stack_cube_left`)

#### Episode Weighting
- `--optimal_episodes`: Comma-separated list of optimal episode numbers (e.g., `"1,2,5-8"`)
- `--recovery_episodes`: Comma-separated list of recovery episode numbers 
- `--optimal_weight`: Multiplication factor for optimal episodes (default: 2.0)

#### Image Augmentation
- `--enable_lighting_augmentation`: Enable lighting and color augmentation (default: True)
- `--brightness_range`: Min and max brightness factors (default: [0.7, 1.3])
- `--contrast_range`: Min and max contrast factors (default: [0.8, 1.2])

#### Joint Noise Injection
- `--enable_joint_noise`: Enable joint coordinate noise injection (default: True)
- `--joint_noise_std`: Standard deviation for joint noise in radians (default: 0.02)
- `--max_joint_noise`: Maximum joint noise magnitude in radians (default: 0.05)

#### Processing Parameters
- `--augmentation_probability`: Probability of applying augmentation to each episode (default: 0.5)
- `--preserve_original`: Keep original episodes alongside augmented ones (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

## Episode List Format

Episode numbers can be specified in several formats:
- Individual episodes: `"1,3,5"`
- Ranges: `"1-5"` (includes episodes 1,2,3,4,5)
- Mixed: `"1,3,5-8,12"` (includes episodes 1,3,5,6,7,8,12)

## Augmentation Process

### 1. Episode Classification and Weighting

Episodes are classified into categories:
- **Optimal Episodes**: High-quality demonstrations that should be weighted more heavily
- **Recovery Episodes**: Demonstrations that include mistake recovery, important for robustness
- **Regular Episodes**: Standard demonstrations

### 2. Augmentation Pipeline

For each episode, the script:

1. **Preserves Original**: Copies the original episode if `--preserve_original` is True
2. **Creates Weighted Copies**: For optimal episodes, creates additional copies based on `--optimal_weight`
3. **Applies Augmentation**: Creates augmented versions with:
   - Lighting variations (brightness, contrast, saturation)
   - Color temperature adjustments
   - Motion blur simulation
   - Shadow effects
   - Camera noise
   - Joint coordinate noise

### 3. Output Structure

The output dataset maintains the same structure as the input:
```
/media/robin/DATA/argumented_stack_cube_left/
├── episode_0000/
│   ├── data.json
│   ├── colors/
│   ├── audios/
│   └── depths/
├── episode_0001/
├── episode_0002/
...
```

Episodes are automatically renumbered starting from `episode_0000`.

## Technical Details

### Image Augmentation

The image augmentation pipeline applies:
- **Brightness**: Random adjustment between specified range
- **Contrast**: Random contrast enhancement
- **Saturation**: Color saturation variations
- **Color Temperature**: Simulates different lighting conditions
- **Motion Blur**: Simulates camera/robot movement
- **Shadows**: Random shadow patterns
- **Noise**: Gaussian and salt-and-pepper noise

### Joint Noise Injection

Joint noise is carefully controlled to maintain task feasibility:
- **Gaussian Noise**: Added to joint positions with configurable standard deviation
- **Clipping**: Noise is clipped to prevent extreme values
- **Bias Simulation**: Small drift added to simulate sensor bias
- **Temporal Correlation**: Optional correlated noise across time steps

### Memory and Performance

The script is designed for efficiency:
- **Streaming Processing**: Processes one episode at a time to minimize memory usage
- **Progress Tracking**: Uses tqdm for progress visualization
- **Error Handling**: Robust error handling with fallback to original data
- **Validation**: Automatic validation of episode structure

## Examples

### Example 1: Basic Augmentation
```bash
python unitree_lerobot/argument_data.py \
    --input_dataset_path /path/to/original/dataset \
    --output_dataset_path /path/to/augmented/dataset
```

### Example 2: Heavy Augmentation for Small Dataset
```bash
python unitree_lerobot/argument_data.py \
    --input_dataset_path /path/to/original/dataset \
    --output_dataset_path /path/to/augmented/dataset \
    --optimal_episodes "1,3,5,7,9" \
    --optimal_weight 5.0 \
    --augmentation_probability 1.0 \
    --joint_noise_std 0.05
```

### Example 3: Conservative Augmentation
```bash
python unitree_lerobot/argument_data.py \
    --input_dataset_path /path/to/original/dataset \
    --output_dataset_path /path/to/augmented/dataset \
    --optimal_weight 1.5 \
    --augmentation_probability 0.3 \
    --joint_noise_std 0.01
```

## Monitoring and Validation

The script provides comprehensive logging:
- Progress bars for each episode
- Summary statistics
- Error reporting
- Final dataset statistics

After augmentation, verify your dataset:
```bash
# Check the augmented dataset structure
ls -la /media/robin/DATA/argumented_stack_cube_left/

# Check episode count
ls -1d /media/robin/DATA/argumented_stack_cube_left/episode_* | wc -l
```

## Integration with Training

Use the augmented dataset with your training script:
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=/media/robin/DATA/argumented_stack_cube_left \
    --policy.type=act \
    --output_dir=outputs/train/augmented_policy
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `--augmentation_probability` or process in smaller batches
2. **Disk Space**: Monitor output directory size, especially with high `--optimal_weight`
3. **Invalid Episodes**: Check input dataset structure and permissions
4. **Performance**: Use SSD storage for better I/O performance

### Logging

Enable debug logging for detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Custom Augmentation Utilities

The `augmentation_utils/` module provides advanced capabilities:
- `ImageTransforms`: Sophisticated image augmentation
- `NoiseInjector`: Advanced noise injection for joint data
- `DatasetProcessor`: Dataset validation and processing utilities

These can be used for custom augmentation pipelines or integration with other tools.

## Best Practices

1. **Start Conservative**: Begin with low augmentation settings and increase gradually
2. **Validate Results**: Always check augmented episodes before training
3. **Monitor Performance**: Track training performance with augmented vs. original data
4. **Preserve Originals**: Keep original episodes in the augmented dataset
5. **Use Appropriate Weighting**: Don't over-weight optimal episodes (2-3x is usually sufficient)
6. **Balance Augmentation**: Mix augmented and original episodes for best results
