from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1

#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ToastedBread_Dataset")
#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_DualArmGrasping_Dataset")
#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_Pouring_Dataset")
##
#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ObjectPlacement_Dataset")
#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_BlockStacking_Dataset")
#dataset = LeRobotDataset(repo_id="unitreerobotics/G1_CameraPackaging_Dataset")
dataset = LeRobotDataset(repo_id="kuehnrobin/g1_pour_can_left_hand")


from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]