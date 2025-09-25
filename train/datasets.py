import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from PIL import Image
import random
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoProcessor
import torch.distributed as dist


class RewardActionDataset(Dataset):
    """ 
    Return format:
      - text_ids:          LongTensor [L_text]
      - image_token_ids:   LongTensor [K, L_img]  
      - states:            FloatTensor [K, state_dim]
      - reward:            FloatTensor [K, reward_dim]
      - rtg:               FloatTensor [1, 1, reward_dim]
      - action_token_ids:  List[Tensor] 
    """

    def __init__(self, args: "DataArguments", tokenizer: Emu3Tokenizer, stage: str = "stage2"):
        """
        Args:
            tokenizer: Emu3Tokenizer
            stage: "stage1" 或 "stage2"
        """
        with open(args.data_path, 'rb') as f:
            self.scenes = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.VISUAL_TOKEN_PATTERN = args.visual_token_pattern
        self.group = args.frames             
        self.action_frames = args.action_frames  
        self.stage = stage
        
        # 统一初始化action tokenizer（stage1时可能用不到，但保持接口一致）
        self.action_tokenizer_path = getattr(args, "action_tokenizer_path", None)
        if self.action_tokenizer_path:
            self.action_tokenizer = AutoProcessor.from_pretrained(self.action_tokenizer_path, trust_remote_code=True)
        else:
            self.action_tokenizer = None
        
        # 检查是否为主进程
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    def __len__(self) -> int:
        return len(self.scenes)

    def to_videostr(self, video_tokens: torch.Tensor) -> str:
        frame_str_list: List[str] = []
        for frame in video_tokens:
            frame_token_str = [
                self.VISUAL_TOKEN_PATTERN.format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr

    def format_video_prompt(self, video_tokens: torch.Tensor) -> str:
        frames, h, w = video_tokens.shape
        videostr = self.to_videostr(video_tokens)
        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +
            self.tokenizer.img_token +
            videostr +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )
        return video_prompt

    def random_frames_to_tensor(
        self,
        img_list: List[str],
        reward_list: List[np.ndarray],
        rtg_list: List[np.ndarray],
        state_list: List[np.ndarray],
        gripper_list: Optional[List[str]] = None,
        action_list: Optional[List[Any]] = None,
        T: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if T is None:
            T = self.group * self.action_frames
            
        start_idx = random.randint(0, len(img_list) - T - 1)

        # 主相机图像
        selected_frames = [np.load(p) for p in img_list[start_idx:start_idx + T]]
        image_tensor = [torch.from_numpy(frame) for frame in selected_frames]
        image_tensor = torch.stack(image_tensor, dim=1)

        # 奖励
        selected_reward = reward_list[start_idx:start_idx + T]
        reward_tensor = [torch.from_numpy(r) for r in selected_reward]
        reward_tensor = torch.stack(reward_tensor, dim=0)

        # 状态
        selected_state = state_list[start_idx:start_idx + T]
        state_tensor = [torch.from_numpy(s) for s in selected_state]
        state_tensor = torch.stack(state_tensor, dim=0)

        # RTG
        selected_rtg = rtg_list[start_idx + 1:start_idx + T + 1]
        rtg_tensor = [torch.from_numpy(r) for r in selected_rtg]
        rtg_tensor = torch.stack(rtg_tensor, dim=0)
        
        # Gripper图像（stage2专用）
        gripper_tensor = None
        if gripper_list is not None:
            selected_gripper = [np.load(p) for p in gripper_list[start_idx:start_idx + T]]
            gripper_tensor = [torch.from_numpy(frame) for frame in selected_gripper]
            gripper_tensor = torch.stack(gripper_tensor, dim=1)

        # 动作（统一处理）
        action_tensors = None
        if action_list is not None:
            selected_actions = action_list[start_idx:start_idx + T]
            action_tensors = torch.cat([torch.tensor(action).unsqueeze(0) for action in selected_actions], dim=0)

        return (image_tensor.squeeze(0), reward_tensor, state_tensor, rtg_tensor, 
                gripper_tensor.squeeze(0) if gripper_tensor is not None else None, 
                action_tensors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene = self.scenes[idx]

        prompt: str = scene["text"]
        image_tokens_path: List[str] = scene["image"]
        rewards: List[np.ndarray] = scene["reward"]
        rtgs: List[np.ndarray] = scene["returnToGo"]
        states: List[np.ndarray] = scene["state"]
        
        gripper_paths: Optional[List[str]] = None
        actions_raw: Optional[List[Any]] = None
        
        if self.stage == "stage2":
            gripper_paths = scene["gripper_image"]
            actions_raw = scene["action"]
        else: 
            actions_raw = scene["action"]

        if len(image_tokens_path) > self.group * self.action_frames:
            frames_num = self.group * self.action_frames
        else:
            frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        
            
        image_tokens, reward_tensor, state_tensor, rtg_tensor, gripper_tokens, action_tensors = self.random_frames_to_tensor(
            image_tokens_path, rewards, rtgs, states, gripper_paths, actions_raw, frames_num
        )

        text_prompt = self.tokenizer.bos_token + prompt
        sample_text_ids = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"]

        # 以 action_frames 为步长做抽帧，得到 K 帧
        image_tokens = image_tokens[0::self.action_frames, ...]
        states_k = state_tensor[0::self.action_frames, ...]
        # rewards_k = reward_tensor[0::self.action_frames, ...]
        reward_tensor = reward_tensor.reshape(self.group, self.action_frames, -1)
        rtg_tensor = rtg_tensor.reshape(self.group, self.action_frames, -1)
        if gripper_tokens is not None:
            gripper_tokens = gripper_tokens[0::self.action_frames, ...]

        # 准备文本 ids
        text_ids = sample_text_ids.squeeze(0)

        # 统一处理action_token_ids（stage1时可能为空）
        action_ids = []
        if action_tensors is not None and self.action_tokenizer is not None:
            action_tensor_grouped = action_tensors.reshape(self.group, self.action_frames, -1)
            action_tokens = self.action_tokenizer(action_tensor_grouped)
            last_vocab_idx = self.tokenizer.pad_token_id - 1
            action_ids = [last_vocab_idx - torch.tensor(id) for id in action_tokens]
        else:
            # stage1或没有action_tokenizer时，返回空列表
            action_ids = []

        image_token_ids: List[torch.Tensor] = []
        for i in range(len(image_tokens)):
            if self.stage == "stage2" and gripper_tokens is not None:
                # stage2: 主相机 + gripper
                main_prompt = self.format_video_prompt(image_tokens[i:i+1])
                gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                merged_prompt = main_prompt + gripper_prompt
                tokenized = self.tokenizer(merged_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"]
            else:
                main_prompt = self.format_video_prompt(image_tokens[i:i+1])
                tokenized = self.tokenizer(main_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"]
            
            image_token_ids.append(tokenized)

        image_token_ids = torch.cat(image_token_ids, dim=0)  # [K, L_img]

        return {
            'text_ids': text_ids,
            'image_token_ids': image_token_ids,   # [K, L_img]
            'states': states_k,                   # [K, state_dim]
            'reward': reward_tensor,                  # [K,action_frames, reward_dim]
            'rtg': rtg_tensor,                           # [K,action_frames, reward_dim]
            'action_token_ids': action_ids,       #stage1为空
        }


def RewardAction_collate(batch: List[Dict[str, Any]]):
    """
    统一的collate函数
    """
    text_ids_list = [b['text_ids'] for b in batch]
    image_token_ids = torch.stack([b['image_token_ids'] for b in batch], dim=0)
    states = torch.stack([b['states'] for b in batch], dim=0)
    reward = torch.stack([b['reward'] for b in batch], dim=0)
    rtg = torch.stack([b['rtg'] for b in batch], dim=0)
    action_token_ids = [b['action_token_ids'] for b in batch]

    return text_ids_list, image_token_ids, states, reward, rtg, action_token_ids
