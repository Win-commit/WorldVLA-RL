import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
from tqdm import tqdm  
import tensorflow as tf
from Helper import *
import cv2

sub_dataname_choices = ['libero_spatial_no_noops', "libero_goal_no_noops", "libero_object_no_noops", "libero_10_no_noops"]
sub_dataname = sub_dataname_choices[3]
dataset_dirs = f"/liujinxin/zhy/dataset_libero/modified_libero_rlds/{sub_dataname}/1.0.0"
builder = tfds.builder_from_directory(dataset_dirs)

ds_all_dict = builder.as_dataset(split="train")


base_output_dir = "/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all"
os.makedirs(base_output_dir, exist_ok=True)

count = 0

#===================================================
# STAGE 1 Get the original third-person image of libero, the robotic arm image, the state, the action, and save them
#====================================================

# Process the dataset and save with tqdm progress bar for episodes
for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
    # tensor to string
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    # concat last two elements of file_path
    name = file_path.split("/")[-2] + "__" + file_path.split("/")[-1].split(".")[0]+ "__" + str(count)
    
    episode_dir = os.path.join(base_output_dir, name)
    os.makedirs(episode_dir, exist_ok=True)

    # Create a subdirectory for images
    image_dir = os.path.join(episode_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    gripper_image_dir = os.path.join(episode_dir, 'gripper_images')
    os.makedirs(gripper_image_dir, exist_ok=True)

    # Create a subdirectory for actions
    action_dir = os.path.join(episode_dir, 'actions')
    os.makedirs(action_dir, exist_ok=True)

    # Create a subdirectory for states
    state_dir = os.path.join(episode_dir, 'states')
    os.makedirs(state_dir, exist_ok=True)
    
    # Create a subdirectory for rewards and returnToGo
    reward_dir = os.path.join(episode_dir, 'reward')
    returnToGo_dir = os.path.join(episode_dir, 'returnToGo')
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(returnToGo_dir, exist_ok=True)


    # Prepare to store images, text, states, actions
    images = []
    images_np = []
    gripper_images = []
    gripper_images_np = []
    languages = []
    actions = []
    states = []

    for i, step in tqdm(enumerate(episode["steps"]), desc=f"Processing episode {name}", total=len(episode["steps"]), unit="step"):
        observation = step["observation"]
        action = step["action"].numpy()
        state = observation["state"].numpy()
        image_np = observation["image"].numpy()
        image = Image.fromarray(image_np)
        gripper_image_np = observation["wrist_image"].numpy()
        gripper_image = Image.fromarray(gripper_image_np)

        language = step["language_instruction"].numpy().decode()

        images.append(image)
        images_np.append(image_np)
        gripper_images.append(gripper_image)
        gripper_images_np.append(gripper_image_np)
        languages.append(language)
        actions.append(action)
        states.append(state)
    
    #Init the eposite reward calculator
    rewardCalculator = RewardCalculator(np.stack(states,axis=0),np.stack(actions,axis=0),
                        np.stack(images_np, axis=0),np.stack(gripper_images_np, axis=0))
    rewards, _, _ = rewardCalculator.get_keypoints_reward()
    # print(rewards.shape)

    # Save 
    for i in range(len(images)):
        image = images[i]
        image.save(os.path.join(image_dir, f"{i}.jpg"))
        gripper_image = gripper_images[i]
        gripper_image.save(os.path.join(gripper_image_dir, f"{i}.jpg"))
        action = actions[i]
        np.save(os.path.join(action_dir, f"{i}.npy"), action)
        state = states[i]
        np.save(os.path.join(state_dir, f"{i}.npy"), state)
        reward = rewards[i]
        np.save(os.path.join(reward_dir, f"{i}.npy"), reward)
        returnToGo = np.sum(rewards[i:], axis=0)
        np.save(os.path.join(returnToGo_dir, f"{i}.npy"), returnToGo)
        if i == 0:
            with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
                f.write(languages[i])
    
    count += 1



