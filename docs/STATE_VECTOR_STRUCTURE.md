# State Vector Structure Documentation

## Issue Identified
The current dataset has incorrectly mapped features from the raw JSON data. The logging output shows that features are mixed up - for example, "left_arm qvel" contains right arm position data instead of left arm velocities.

## Root Cause
The dataset conversion script (`convert_unitree_json_to_lerobot.py`) is concatenating features in the wrong order or with incorrect mapping when creating the LeRobot dataset from the raw JSON.

## Raw JSON Structure (Correct)
From the raw JSON data, the correct structure should be:

### States (from `states` field):
- **left_arm**: qpos (7D), qvel (7D), torque (7D)
- **right_arm**: qpos (7D), qvel (7D), torque (7D) 
- **left_hand**: qpos (7D), qvel (7D), torque (7D), pressures (12D)
- **right_hand**: qpos (7D), qvel (7D), torque (7D), pressures (12D)
- **camera**: qpos (2D)

### Actions (from `actions` field):
- **left_arm**: qpos (7D)
- **right_arm**: qpos (7D)
- **left_hand**: qpos (7D)  
- **right_hand**: qpos (7D)
- **camera**: qpos (2D)

## Proposed 108D State Vector Structure
When including ALL features from raw JSON:

```
0-6:     left_arm qpos (7D)
7-13:    left_arm qvel (7D)  
14-20:   left_arm torque (7D)
21-27:   right_arm qpos (7D)
28-34:   right_arm qvel (7D)
35-41:   right_arm torque (7D)
42-48:   left_hand qpos (7D)
49-55:   left_hand qvel (7D)
56-62:   left_hand torque (7D)
63-74:   left_hand pressures (12D)
75-81:   right_hand qpos (7D)
82-88:   right_hand qvel (7D)
89-95:   right_hand torque (7D)
96-107:  right_hand pressures (12D)
108-109: camera qpos (2D) [total = 110D if included]
```

## Action Vector Structure (30D)
```
0-6:   left_arm qpos (7D)
7-13:  right_arm qpos (7D)
14-20: left_hand qpos (7D)
21-27: right_hand qpos (7D)
28-29: camera qpos (2D)
```

## Current Issues to Fix

### 1. Dataset Creation Script
The `convert_unitree_json_to_lerobot.py` needs to be updated to:
- Correctly concatenate features in the right order
- Preserve all velocity and torque data
- Ensure feature names match the actual data positions

### 2. Feature Filter Updates
✅ **COMPLETED**: Updated `feature_filter.py` to:
- Handle both 108D (new) and 82D (legacy) structures  
- Correctly map feature indices for filtering
- Support hand velocities and torques in the new structure
- Log state breakdowns for debugging

### 3. Training Debug Output  
✅ **COMPLETED**: Updated `train.py` to:
- Log 108D state breakdowns correctly
- Maintain backward compatibility with 82D legacy structure
- Show proper feature mapping in debug output

## Next Steps

1. **Fix dataset conversion**: Update `convert_unitree_json_to_lerobot.py` to correctly map features from raw JSON
2. **Regenerate datasets**: Create new datasets with correct feature mapping
3. **Test filtering**: Verify that feature filtering works correctly with the new structure
4. **Validate training**: Ensure that hand movements work properly with correct feature mapping

## Benefits of Full Feature Dataset

1. **Complete data**: Includes all available sensor information (velocities, torques, pressures)
2. **Flexible filtering**: Can filter out unwanted features at training time without recreating datasets
3. **Better hand control**: Hand velocities and torques should improve dexterous manipulation
4. **Research flexibility**: Enables ablation studies on different sensor modalities
