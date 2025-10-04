import sys
sys.path.append("/liujinxin/zhy/ICLR2026")
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.univla_rl_unified import Emu3UnifiedRewardModel
import torch
import pickle
from collections import deque
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3RewardConfig
import torch.nn.functional as F

class HistoryManager:
    """管理图像和动作的历史记录"""
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.vision_queue = deque(maxlen=self.window_size) # save gripper and main view image in a window
        self.state_queue = deque(maxlen=self.window_size) # save state in a window
        self.reward_queue = deque(maxlen=self.window_size) # save reward in a window
        self.action_queue = deque(maxlen=self.window_size) # save action in a window
    
    def add_image(self, image_inputs):
        """添加图像到历史队列"""
        self.vision_queue.append(image_inputs)
    
    def add_state(self, state):
        """添加状态到历史队列"""
        self.state_queue.append(state)
    
    def add_reward(self, reward):
        """添加奖励到历史队列"""
        self.reward_queue.append(reward)

    def get_history(self):
        """获取历史图像"""
        return {
            "vision": list(self.vision_queue),
            "state": list(self.state_queue),
            "reward": list(self.reward_queue),
            "action": list(self.action_queue)   
        }
    
    def add_action(self, action):
        """添加动作到历史队列"""
        self.action_queue.append(action)
    
    
    def reset(self):
        """重置历史队列"""
        self.vision_queue.clear()
        self.action_queue.clear()
        self.state_queue.clear()
        self.reward_queue.clear()


VISUAL_TOKEN_PATTERN = "<|visual token {token_id:0>6d}|>"

def to_videostr(tokenizer, video_tokens: torch.Tensor) -> str:
    frame_str_list: List[str] = []
    for frame in video_tokens:
        frame_token_str = [
            VISUAL_TOKEN_PATTERN.format(token_id=token_id)
            for token_id in frame.flatten()
        ]
        frame_str = "".join(frame_token_str)
        frame_str_list.append(frame_str)
    videostr = tokenizer.eof_token.join(frame_str_list)
    return videostr

def format_video_prompt(tokenizer, video_tokens: torch.Tensor) -> str:
    frames, h, w = video_tokens.shape
    videostr = to_videostr(tokenizer, video_tokens)
    video_prompt = (
        tokenizer.boi_token +
        f"{frames}*{h}*{w}" +
        tokenizer.img_token +
        videostr +
        tokenizer.eof_token +
        tokenizer.eoi_token
    )
    return video_prompt


def visualize_rewards_comparison(pred_rewards, real_rewards, reward_names, save_dir="reward_visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    
    num_timesteps = len(pred_rewards)
    num_components = pred_rewards[0].shape[1] if num_timesteps > 0 else 0
    
    for component_idx in range(num_components):
        plt.figure(figsize=(15, 8))
        
        # 收集所有数据点
        all_pred_x = []
        all_pred_y = []
        all_real_x = []
        all_real_y = []
        
        # 计算真实全局时间步
        global_step = 0
        
        for t in range(num_timesteps):
            pred_seq_length = pred_rewards[t].shape[0]
            real_seq_length = real_rewards[t].shape[0]
            
            # 提取值
            pred_values = pred_rewards[t][:, component_idx]
            real_values = real_rewards[t][:, component_idx]
            
            # 创建真实的连续时间轴
            for i in range(pred_seq_length):
                all_pred_x.append(global_step + i)
                all_pred_y.append(pred_values[i])
            
            for i in range(real_seq_length):
                all_real_x.append(global_step + i)
                all_real_y.append(real_values[i])
            
            # 更新全局步数
            global_step += max(pred_seq_length, real_seq_length)
        
        # 一次性绘制所有连续数据
        plt.plot(all_pred_x, all_pred_y, 'r-', alpha=0.7, linewidth=1.5)
        plt.plot(all_real_x, all_real_y, 'b-', alpha=0.7, linewidth=1.5)
        
        plt.title(f'Reward Component: {reward_names[component_idx]}')
        plt.xlabel('Continuous Timestep')
        plt.ylabel('Reward Value')
        plt.legend(['Prediction', 'Ground Truth'])
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f'reward_component_{component_idx}_{reward_names[component_idx].replace(" ", "_")}.png'))
        plt.close()


if __name__ == "__main__":
    env_model_path = "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk_lamda0.005/checkpoint-4600"
    data_path = "/liujinxin/zhy/ICLR2026/datasets/libero/data/meta/libero_all_norm.pkl"
    history_manager = HistoryManager(window_size=1)
    save_dir = "reward_visualizations-4600_v4"
    action_frames = 10
    reward_group_size = 10
    tokenizer = Emu3Tokenizer.from_pretrained(
            env_model_path,
            padding_side="right",
            use_fast=False,
        )
    config = Emu3RewardConfig.from_pretrained(env_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_model = Emu3UnifiedRewardModel.from_pretrained(
                env_model_path,
                tokenizer=tokenizer,
                parallel_mode=True,
                parallel_reward_groups=10,
                reward_group_size=reward_group_size,
                attn_implementation = "eager",
                torch_dtype=torch.bfloat16,
            ).to(device)
    
    for param in env_model.parameters():
            param.requires_grad = False
    env_model.eval()

    

    
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for item in data[1:2]:
        history_manager.reset()
        task_description = item["text"]
        image = item["image"]
        gripper_image = item["gripper_image"]
        robot_state = item["state"]
        reward = item["reward"]
        returnToGo = item["returnToGo"]
        
        image = [np.load(p) for p in image]
        image_tensor = [torch.from_numpy(frame) for frame in image]
        image_tensor = torch.stack(image_tensor, dim=1).squeeze(0)

        gripper_image = [np.load(p) for p in gripper_image]
        gripper_image_tensor = [torch.from_numpy(frame) for frame in gripper_image]
        gripper_image_tensor = torch.stack(gripper_image_tensor, dim=1).squeeze(0)

        robot_state_tensor = [torch.from_numpy(frame) for frame in robot_state]
        robot_state_tensor = torch.stack(robot_state_tensor, dim=0)

        reward_tensor = [torch.from_numpy(r) for r in reward]
        reward_tensor = torch.stack(reward_tensor, dim=0)

        rtg_tensor = [torch.from_numpy(r) for r in returnToGo]
        rtg_tensor = torch.stack(rtg_tensor, dim=0)


        imge_tokens = image_tensor[0::action_frames, ...]
        gripper_image_tokens = gripper_image_tensor[0::action_frames, ...]
        states = robot_state_tensor[0::action_frames, ...]
        text_prompt = tokenizer.bos_token + task_description
        text_tokenized = tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
        text_ids_list = [text_tokenized[0]]

        # Prepare to collect predicted and ground truth rewards
        all_predicted_rewards = []
        all_real_rewards = []
        Loss = []
        for i in range(len(imge_tokens)):
            main_prompt = format_video_prompt(tokenizer, imge_tokens[i:i+1])

            gripper_prompt = format_video_prompt(tokenizer, gripper_image_tokens[i:i+1])
            merged_prompt = main_prompt + gripper_prompt
            img_tokenized = tokenizer(merged_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
            img_tokenized = img_tokenized.unsqueeze(0)
            states_i = states[i].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            sampling_reward_results = env_model.sample_rewards(text_tokenized, img_tokenized, states_i, history_manager.get_history())
            
            #predict reward and rtg
            reward_pred = sampling_reward_results["selected_values"][0,0,:reward_group_size]
            rtg_pred = sampling_reward_results["selected_values"][0,0,reward_group_size:]
            # Collect predicted rewards - preserve the full [10,14] structure
            reward_pred_np = reward_pred.detach().cpu().float().numpy()[0]  # Shape: [10,14]
            all_predicted_rewards.append(reward_pred_np)
            
            # Collect real rewards for the next reward_group_size steps
            start_idx = i * action_frames
            end_idx = start_idx + reward_group_size
            
            # Handle edge case where we might run out of data
            if end_idx > len(reward_tensor):
                # Get available data
                available_length = len(reward_tensor) - start_idx
                real_reward_segment = reward_tensor.numpy()[start_idx:len(reward_tensor)]
                
                # Instead of padding real data, truncate predicted rewards to match available data
                reward_pred_np = reward_pred_np[:available_length]
                
                print(f"Timestep {i}: Truncated prediction from {reward_group_size} to {available_length} steps")
            else:
                real_reward_segment = reward_tensor.numpy()[start_idx:end_idx]
            loss = F.mse_loss(torch.from_numpy(reward_pred_np).to(device=device, dtype=torch.bfloat16), torch.from_numpy(real_reward_segment).to(device=device, dtype=torch.bfloat16))
            print(loss)
            Loss.append(loss.item())
            
            all_real_rewards.append(real_reward_segment)
            
            history_manager.add_image(img_tokenized)
            history_manager.add_state(states_i)
            history_manager.add_reward(sampling_reward_results["critical_segments"] if sampling_reward_results is not None else None)
        
        # Print information about the collected data
        print(f"Number of collected timesteps: {len(all_predicted_rewards)}")
        print(f"Loss: {np.mean(Loss)}")
        for i, (pred, real) in enumerate(zip(all_predicted_rewards, all_real_rewards)):
            print(f"Timestep {i} - Prediction shape: {pred.shape}, Real shape: {real.shape}")
        
        # Define reward component names for visualization
        reward_names = [
            # Image goal tracking
            "MSE Value", 
            "SSIM Value",
            "ORB Feature Similarity",
            "Gripper MSE Value",
            "Gripper SSIM Value",
            "Gripper ORB Feature Similarity",
            
            # Property goal tracking
            "Joint Position Error",
            
            # Auxiliary rewards
            "Joint Velocity Error",
            "Joint Acceleration Error",
            "Action Velocity Error",
            "Action Acceleration Error",
            
            # Sub-goal progress
            "Sub-goal Reward",
            "Success Reward",
            
            # Total reward
            "Total Reward"
        ]
        
        # Visualize rewards comparison
        visualize_rewards_comparison(all_predicted_rewards, all_real_rewards, reward_names, save_dir=save_dir)

            





        
        








