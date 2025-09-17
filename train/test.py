import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import wandb
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    TrainerCallback
)
import sys
sys.path.append("/liujinxin/zhy/ICLR2026")
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3Config, Emu3RewardConfig
from datasets import RewardActionDataset, RewardAction_collate
import logging


tokenizer = Emu3Tokenizer.from_pretrained(
        "/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200",
        model_max_length=6400,
        trust_remote_code=True,
    )

# Load environment model (frozen)
env_config = Emu3RewardConfig.from_pretrained("/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200")
env_model = Emu3UnifiedRewardModel.from_pretrained(
    "/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200",
    config=env_config,
    tokenizer=tokenizer,
    trust_remote_code=True,
    parallel_mode=True,
    parallel_reward_groups=10,
    reward_group_size=5,
    p=1,
    gamma=0.9,
    noise_factor=0.4,
    detach_selected_reward_hs=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16 
)



actor_model_path = "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance_StateNorm_warmup/checkpoint-300"
actor_config = Emu3RewardConfig.from_pretrained(actor_model_path)
actor_model = Emu3UnifiedRewardModel.from_pretrained(
    actor_model_path,
    config=actor_config,
    tokenizer=tokenizer,
    trust_remote_code=True,
    parallel_mode=True,
    parallel_reward_groups=10,
    reward_group_size=5,
    p=1,
    gamma=0.9,
    noise_factor=0.4,
    detach_selected_reward_hs=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

env_model.set_mode(parallel_mode=True)
actor_model.set_mode(parallel_mode=True)
print(f"Embedding size: {env_model.get_input_embeddings().weight.shape[0]}")
print(f"Actor model embedding size: {actor_model.get_input_embeddings().weight.shape[0]}")
