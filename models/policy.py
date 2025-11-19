from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3Config,Emu3RewardConfig
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3PreTrainedModel
from models.action_patches import (
    ActionExpertConfig,
    ExpertType,
    create_action_expert,
)
from typing import Optional,List
import torch
import os 
import torch.nn as nn

class DumbConfig():
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class OurPolicy(nn.Module):
    def __init__(self,
                dynamic_model_path:str,
                tokenizer:Emu3Tokenizer,
                parallel_reward_groups:int,
                reward_group_size:int,
                gamma:float,
                noise_factor:float,
                action_config:ActionExpertConfig,
                action_expert_path:Optional[str],
                device
                 ):
        super().__init__()
        self.device = device

        self.dynamic_model_config = Emu3RewardConfig.from_pretrained(dynamic_model_path)
        self.dynamic_model = Emu3UnifiedRewardModel.from_pretrained(
            dynamic_model_path,
            config=self.dynamic_model_config,
            tokenizer=tokenizer,
            trust_remote_code=True,
            parallel_mode=True,
            parallel_reward_groups=parallel_reward_groups,
            reward_group_size=reward_group_size,
            gamma=gamma,
            noise_factor=noise_factor,
            attn_implementation= 'eager',
            torch_dtype=torch.bfloat16
            )

        self.action_expert = create_action_expert(
            action_config,
            model_path=action_expert_path if action_expert_path and os.path.exists(action_expert_path) else None,
            map_location=device
            )
        self.action_expert.to(dtype=torch.bfloat16)


        self.config = DumbConfig(self.action_expert.config.hidden_size)

        self.to(device)

    def forward(self,
                text_ids_list: List,
                image_token_ids: torch.LongTensor,
                states: torch.Tensor,
                target_actions: torch.Tensor):
        
        with torch.no_grad():
            reward_sampling_results = self.dynamic_model.sample_rewards(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states
            )
        loss_dict = self.action_expert.compute_flow_loss(reward_sampling_results,target_actions)
        
        return loss_dict
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def frozen_dynamic_world(self):
        for param in self.dynamic_model.parameters():
            param.requires_grad = False
        self.dynamic_model.eval()
    
