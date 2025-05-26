
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py --raw-dir ~/humanoid/humanoid_ws/src/unitree_IL_lerobot/data/ --repo-id kuehnrobin/g1_pour_can_left_hand --robot_type Unitree_G1_Dex3 --task "pour_can" --push_to_hub



## 2. Training an ACT Policy on a Unitree Dataset
LeRobot uses a unified training script that supports different policy architectures (ACT, Diffusion, Pi0, etc.) via configuration flags. To train an Action Chunking Transformer (ACT) policy on a provided Unitree dataset, run the training script with policy.type=act. For example, to train on the Toasted Bread task dataset:

    cd unitree_IL_lerobot/lerobot
    python lerobot/scripts/train.py \
        --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
        --policy.type=act \
        --device=cuda 

This will launch training for an ACT imitation policy using the specified dataset
github.com
. By default, outputs (checkpoints, logs) are saved under lerobot/outputs/train/<DATE>/<TIME>_act/. You can monitor training logs or use tools like Weights & Biases (enable via --wandb.enable=true) for visualization. Key points for training ACT:
Default hyperparameters: The LeRobot framework has default configs for ACT (e.g. transformer layers, learning rate, chunk size of actions) tuned for ALOHA and similar tasks. You can override these via command-line or config files if needed. For instance, you can set --train.max_epochs=... or --train.total_steps=... to control training length, or adjust policy.* parameters (see lerobot/configs/policy/act*.yaml for reference). In many cases, the defaults will work out-of-the-box for the provided datasets.
Compute requirements: ACT training involves transformers and can be heavy. Ensure you have a GPU (--device=cuda) for faster training. If training is slow or memory-intensive, consider lowering the batch size (--train.batch_size) or sequence length, though the defaults are usually manageable with a modern GPU.
Resuming or checkpointing: The training script periodically saves checkpoints (e.g. every N steps). If your training interrupts or you want to resume, you can restart using the latest checkpoint by specifying --policy.path=<path_to_checkpoint_folder> when re-running train.py. For example: --policy.path=lerobot/outputs/train/2025-05-01/12-00-00_act/checkpoints/last/pretrained_model. This will load the saved model and continue training instead of starting from scratch
huggingface.co
huggingface.co
.

## 3. Fine-Tuning on Your Own Data (Domain Adaptation)
Once you have a base policy (e.g. an ACT model trained on Unitree’s dataset), you can fine-tune it with your own demonstrations to perform domain adaptation. There are two common approaches:
(a) Continue training with new data (sequential fine-tuning): Load the pre-trained model and train further on your dataset. This adapts the policy to your environment/data distribution. For example, to fine-tune the ACT model on your custom dataset:

    python lerobot/scripts/train.py \
        --dataset.repo_id=<your_hf_username>/my_task_dataset \
        --dataset.root ~/datasets  --local-files-only true \ 
        --policy.type=act \
        --policy.path=path/to/pretrained_model_checkpoint \
        --device=cuda

Here, policy.path points to the pre-trained model directory (either a local checkpoint folder or a Hub model ID). This initializes the policy with the learned weights
. LeRobot will then train further using my_task_dataset. (If you uploaded your dataset to Hugging Face, you can omit --dataset.root and --local-files-only and just use the Hub repo ID.)

Fine-tuning is especially useful for domain adaptation – e.g., if your camera viewpoint or lighting differs from the original data, the model will adjust to these new visuals while retaining the core behavior learned from the larger Unitree dataset.

### Tips for successful fine-tuning:
Match observation spaces: Ensure your demonstration data provides the same observation modalities (camera images, robot states) as the original. If your camera resolution or intrinsics differ (e.g. 1080p vs 720p), update the robot_type during conversion accordingly so the model input shape matches. LeRobot configs (in ROBOT_CONFIGS) define image resolution and camera params for each robot type – using the correct one (or adding a new config) avoids dimension mismatches
github.com
. If needed, you can also resize images or adjust a config setting (e.g., a custom robot_type).
Hyperparameters: When fine-tuning, you might want to use a smaller learning rate and fewer training steps, to avoid destroying pre-trained features too quickly. Monitor validation loss if possible (LeRobot might allow holding out some trajectories or using the original dataset for eval) to ensure you don’t overfit to the new data.
Evaluation: After fine-tuning, test the policy in your scenario. LeRobot’s eval_robot tools can run the policy on the real robot or in simulation. For example, Unitree’s repository provides an eval_g1.py script for running a policy on the G1 robot with a given initial state
github.com
. Use your fine-tuned model’s checkpoint with such scripts to verify performance.
