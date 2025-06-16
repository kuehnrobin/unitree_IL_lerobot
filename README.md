**Read this in other languages: [中文](./docs/README_zh.md).**

|Unitree Robotics  repositories        | link |
|---------------------|------|
| Unitree Datasets   | [unitree datasets](https://huggingface.co/unitreerobotics) |
| AVP Teleoperate    | [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) |


# 0. 📖 Introduction

This repository is used for `lerobot training validation`(Supports LeRobot datasets version 2.0 and above.) and `unitree data conversion`.

`❗Tips： If you have any questions, ideas or suggestions that you want to realize, please feel free to raise them at any time. We will do our best to solve and implement them.`

| Directory          | Description                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | The code in the `lerobot repository` for training;  its corresponding commit version number is `725b446a`.|
| utils         | `unitree data processing tool `   |
| eval_robot    | `unitree real machine inference verification of the model`     |


# 1. 📦 Environment Setup

## 1.1 🦾 LeRobot Environment Setup

The purpose of this project is to use the [LeRobot](https://github.com/huggingface/lerobot) open-source framework to train and test data collected from Unitree robots. Therefore, it is necessary to install the LeRobot-related dependencies first. The installation steps are as follows, and you can also refer to the official [LeRobot](https://github.com/huggingface/lerobot) installation guide:

```bash
# Clone the source code
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git

# If already downloaded:
git submodule update --init --recursive

# Create a conda environment
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# Install LeRobot
cd unitree_lerobot/lerobot && pip install -e .

# Install unitree_lerobot
cd ../../ && pip install -e .
```

## 1.2 🕹️ unitree_sdk2_python

For `DDS communication` on Unitree robots, some dependencies need to be installed. Follow the installation steps below:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 2. ⚙️ Data Collection and Conversion

## 2.1 🖼️ Load Datasets
If you want to directly load the dataset we have already recorded,
Load the [`unitreerobotics/G1_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_ToastedBread_Dataset) dataset from Hugging Face. The default download location is `~/.cache/huggingface/lerobot/unitreerobotics`. If you want to load data from a local source, please change the `root` parameter.

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1
dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ToastedBread_Dataset")

from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
```

`visualization`

```bash
cd unitree_lerobot/lerobot

python lerobot/scripts/visualize_dataset.py \
    --repo-id unitreerobotics/G1_ToastedBread_Dataset \
    --episode-index 0
```

## 2.2 🔨 Data Collection

If you want to record your own dataset. The open-source teleoperation project [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) can be used to collect data using the Unitree G1 humanoid robot. For more details, please refer to the [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) project.

## 2.3 🛠️ Data Conversion

The data collected using [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) is stored in JSON format. Assuming the collected data is stored in the `$HOME/datasets/g1_grabcube_double_hand`, the format is as follows

    g1_grabcube_double_hand/        # Task name
    │
    ├── episode_0001                # First trajectory
    │    ├──audios/                 # Audio information
    │    ├──colors/                 # Image information
    │    ├──depths/                 # Depth image information
    │    └──data.json               # State and action information
    ├── episode_0002
    ├── episode_...
    ├── episode_xxx

### 2.3.1 🔀 Sort and Rename

When generating datasets for LeRobot, it is recommended to ensure that the data naming convention, starting from `episode_0`, is sequential and continuous. You can use the following script to `sort and rename` the data accordingly.


```bash
python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir $HOME/datasets/g1_grabcube_double_hand
```

#### 2.3.2 🔄 Conversion

Convert `Unitree JSON` Dataset to `LeRobot` Format. You can define your own `robot_type` based on [ROBOT_CONFIGS](https://github.com/unitreerobotics/unitree_IL_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L154).
```bash
# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
```


# 3. 🚀 Training

[For training, please refer to the official LeRobot training example and parameters for further guidance.](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)


- `Train Act Policy`

```bash
cd unitree_lerobot/lerobot

python lerobot/scripts/train.py \
  --dataset.repo_id kuehnrobin/g1_pour_can_left_hand \
  --policy.type=act
  --policy.path outputs/train/pouring_unitree_2025-05-09/10-08-45_act/checkpoints/last/pretrained_model/ \
  --optimizer.lr 1e-5 \
  --steps 50000 \
  --wandb.enable True \
  --wandb.project pour_can


Fine Tune:

python lerobot/scripts/train.py \
  --dataset.repo_id kuehnrobin/g1_pour_can_left_hand \
  --policy.path=outputs/train/pouring_unitree_2025-05-09/10-08-45_act/checkpoints/last/pretrained_model/ \
  --wandb.enable True \
  --wandb.project pour_can

    
First try of fine tuning Unitree Pouring with my custom Dataset g1_can_pour_left_hand on 26.05.2025



Given your specific scenario (right hand + water bottle → left hand + can), here are my(Claude Sonnet 4) suggested parameters:

```bash
cd unitree_lerobot/lerobot

python lerobot/scripts/train.py \
  --dataset.repo_id kuehnrobin/g1_pour_can_left_hand \
  --policy.path=outputs/train/pouring_unitree_2025-05-09/10-08-45_act/checkpoints/last/pretrained_model/ \
  --optimizer.lr 5e-6 \
  --optimizer.weight_decay 1e-4 \
  --steps 15000 \
  --eval_freq 1000 \
  --save_freq 1000 \
  --log_freq 100 \
  --batch_size 16 \
  --wandb.enable true \
  --wandb.project pour_can \
  --job_name left_hand_can_adaptation
```

#### Phase 1: Conservative Adaptation

  python lerobot/scripts/train.py \
    --dataset.repo_id kuehnrobin/g1_pour_can_left_hand \
    --policy.path=outputs/train/pouring_unitree_2025-05-09/10-08-45_act/checkpoints/last/pretrained_model/ \
    --optimizer.lr 1e-5 \
    --optimizer.weight_decay 5e-5 \
    --steps 8000 \
    --eval_freq 500 \
    --save_freq 1000 \
    --log_freq 50 \
    --batch_size 8 \
    --wandb.enable true \
    --wandb.project pour_can \
    --job_name conservative_adaptation

#### Phase 2: If Phase 1 works, increase learning rate

  python lerobot/scripts/train.py \
    --dataset.repo_id kuehnrobot/g1_pour_can_left_hand \
    --policy.path=outputs/from_phase1/checkpoints/last/pretrained_model/ \
    --optimizer.lr 3e-5 \
    --optimizer.weight_decay 1e-4 \
    --steps 10000 \
    --eval_freq 500 \
    --save_freq 1000 \
    --batch_size 12 \
    --wandb.enable true \
    --wandb.project pour_can \
    --job_name aggressive_adaptation
## 3. Parameter Rationale

**Learning Rate (`5e-6`)**: 
- Much lower than training from scratch (typically 1e-4 to 1e-5)
- Prevents catastrophic forgetting of the pouring skills
- Allows gradual adaptation to left hand + can

**Steps (`15000`)**:
- Fewer than full training since you're fine-tuning
- Should be enough to adapt to the new hand/object combination
- Monitor loss curves to adjust if needed

**Batch Size (`16`)**:
- Smaller batch size can help with stability during fine-tuning
- Adjust based on your GPU memory

**Evaluation Frequency (`1000`)**:
- More frequent evaluation to monitor adaptation progress
- Important to catch overfitting early

## 4. Additional Considerations

**Data Augmentation**: Consider if your dataset has enough diversity in:
- Grasping poses for the left hand
- Can orientations and positions
- Pouring trajectories

**Monitoring**: Watch for:
- Loss plateauing (might need longer training)
- Evaluation performance degrading (overfitting)
- Gradual improvement in success rate

**Alternative Approach**: If the above doesn't work well, you might also try:

```bash
# Even more conservative fine-tuning
python lerobot/scripts/train.py \
  --dataset.repo_id your_username/g1_pour_can_left_hand \
  --policy.path=outputs/train/pouring_unitree_2025-05-09/10-08-45_act/checkpoints/last/pretrained_model/ \
  --optimizer.lr 1e-6 \
  --steps 25000 \
  --wandb.enable True \
  --wandb.project pour_can_domain_adaptation
```

The key is starting conservative and increasing learning rate/steps if the model isn't adapting fast enough.

```

- `Train Diffusion Policy`

```bash
cd unitree_lerobot/lerobot

python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=diffusion \
  --use_wandb=True

```

- `Train Pi0 Policy`

```bash
cd unitree_lerobot/lerobot

python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=pi0 \
  --use_wandb=True

```

# 4. 🤖 Real-World Testing

To test your trained model on a real robot, you can use the eval_g1.py script located in the eval_robot/eval_g1 folder. Here’s how to run it:

[To open the image_server, follow these steps](https://github.com/unitreerobotics/avp_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-server)

```bash
# --policy.path Path to the trained model checkpoint
# --repo_id     Dataset repository ID (Why use it? The first frame state of the dataset is loaded as the initial state)
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/pour_can_2025-05-25/19-48-48_act/checkpoints/last/pretrained_model/     --repo_id=kuehnrobin/g1_pour_can_left_hand \
    --arm_speed 10.0 \
    --no_gradual_speed=true \
    --cyclonedds_uri enxa0cec8616f27


python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-05-27/13-53-10_pour_can_mark_3/checkpoints/015000/pretrained_model/ \
    --repo_id=kuehnrobin/g1_pour_can_left_hand \
    --arm_speed 10.0 \
    --no_gradual_speed=true \
    --cyclonedds_uri enxa0cec8616f27 \
    --record true


# If you want to evaluate the model's performance on the dataset, use the command below for testing
python unitree_lerobot/eval_robot/eval_g1/eval_g1_dataset.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_ToastedBread_Dataset
```
 python unitree_lerobot/eval_robot/eval_g1/eval_g1_dataset.py --policy.path=unitree_lerobot/lerobot/outputs/train/2025-05-26/14-57-19_left_hand_can_adaptation/checkpoints/last/pretrained_model/ --repo_id=kuehnrobin/pour_can_left_hand


# 5. 🤔 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Why use `LeRobot v2.0`?** | [Explanation](https://github.com/huggingface/lerobot/pull/461) |
| **401 Client Error: Unauthorized** (`huggingface_hub.errors.HfHubHTTPError`) | Run `huggingface-cli login` to authenticate. |
| **FFmpeg-related errors:**  <br> Q1: `Unknown encoder 'libsvtav1'` <br> Q2: `FileNotFoundError: No such file or directory: 'ffmpeg'` <br> Q3: `RuntimeError: Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed.` | Install FFmpeg: <br> `conda install -c conda-forge ffmpeg` |
| **Access to model `google/paligemma-3b-pt-224` is restricted.** | Run `huggingface-cli login` and request access if needed. |

# 6. 📤 Sharing Your Trained Policies

After training your robot policy, you can share it with the community by uploading it to the Hugging Face Hub. This allows others to use your trained models and helps advance the field of robot learning.

## 6.1 🚀 Push Policy to Hugging Face Hub

Once you have trained a policy with our training script (lerobot/scripts/train.py), use this script to push it
to the hub.

Example:

```bash
python lerobot/scripts/push_pretrained.py \
    --pretrained_path=outputs/train/act_aloha_sim_transfer_cube_human/checkpoints/last/pretrained_model \
    --repo_id=lerobot/act_aloha_sim_transfer_cube_human
```
### Using Uploaded Policies

Once uploaded, others can use your policy:

```python
# Loading from Hub
from lerobot.common.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("your_username/g1_pour_can_act_policy")

# Or in evaluation script
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py \
    --policy.path=your_username/g1_pour_can_act_policy \
    --repo_id=original_training_dataset \
    --arm_speed 10.0
```

## 6.2 📝 Best Practices

1. **Descriptive Names**: Use clear, descriptive repository names
2. **Documentation**: Add a good README to your model repository describing the task, training data, and performance
3. **Versioning**: Use branches or separate repositories for different versions
4. **Testing**: Test your uploaded model before sharing publicly
5. **Licensing**: Consider adding appropriate licenses to your model repositories

# 7. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
