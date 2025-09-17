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
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3Config, Emu3RewardConfig
from datasets import RewardActionDataset, RewardAction_collate
import deepspeed
import json
# import logging
# logging.getLogger("transformers").setLevel(logging.DEBUG)

import websockets
import asyncio
import numpy as np

os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"


@dataclass
class ModelArguments:
    env_model_path: str = field(default="/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER/checkpoint-8000", 
                              metadata={"help": "环境模型路径(使用stage1权重)"})
    actor_model_path: str = field(default="", 
                              metadata={"help": "动作生成模型路径(可选，如果为空则使用env_model_path初始化)"})
    stage: str = field(default="stage2", metadata={"help": "训练阶段: stage1 或 stage2"})
    parallel_mode: bool = field(default=True, metadata={"help": "是否启用并行奖励采样模式(stage2专用)"})
    parallel_reward_groups: int = field(default=5, metadata={"help": "并行奖励组数(stage2专用)"})
    reward_group_size: int = field(default=10, metadata={"help": "每组奖励token数量(stage2专用)"})
    p: float = field(default=0.85, metadata={"help": "stage2执行奖励采样的概率"})
    gamma: float = field(default=0.9, metadata={"help": "时间加权参数(stage2专用)"})
    noise_factor: float = field(default=0.1, metadata={"help": "噪声因子(stage2专用)"})


@dataclass
class DataArguments:
    data_path: str = field(default="")
    frames: int = field(default=2)
    action_frames: int = field(default=10)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: int = field(default=32768)
    action_tokenizer_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(HFTrainingArguments):
    # Inherit from HF TrainingArguments and add custom fields
    attn_type: str = field(default="eager")
    min_learning_rate: Optional[float] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=6400)
    exp_name: str = field(default="")

    def __post_init__(self):
        super().__post_init__()
        # Set up lr_scheduler_kwargs if min_learning_rate is provided
        if self.min_learning_rate is not None:
            if not hasattr(self, 'lr_scheduler_kwargs') or self.lr_scheduler_kwargs is None:
                self.lr_scheduler_kwargs = {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_learning_rate


class EnvModelClient:
    def __init__(self, host="localhost", port=8765):
        self.uri = f"ws://{host}:{port}"
        
    async def sample_rewards_async(self, text_ids_list, image_token_ids, states):
        # 准备数据
        data = {
            'text_ids_list': [ids.cpu().numpy().tolist() for ids in text_ids_list],
            'image_token_ids': image_token_ids.cpu().numpy().tolist(),
            'states': states.cpu().numpy().tolist()
        }
        
        async with websockets.connect(self.uri) as websocket:
            # 发送数据
            await websocket.send(json.dumps(data))
            
            # 接收结果
            response = await websocket.recv()
            results = json.loads(response)
            
            # 转换回PyTorch张量
            device = image_token_ids.device
            reward_results = {
                'reward_preds_group_mean': torch.tensor(results['reward_preds_group_mean']).to(device),
                'best_reward_group': torch.tensor(results['best_reward_group']).to(device),
                'noise_norm': results['noise_norm'],
                'reward_embedding_norm': results['reward_embedding_norm'],
                'rwd_noise_ratio': results['rwd_noise_ratio'],
                'rtg_noise_ratio': results['rtg_noise_ratio']
            }
            
            return reward_results
    
    def sample_rewards(self, text_ids_list, image_token_ids, states):
        # 同步包装
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.sample_rewards_async(text_ids_list, image_token_ids, states)
        )


class Stage2DualModelTrainer(Trainer):
    """Stage2专用的双模型训练器，一个环境模型(冻结)负责采样奖励，一个动作生成模型负责生成动作"""

    def __init__(
        self,
        env_model_client,  # 改为客户端
        **kwargs
    ):
        super().__init__(**kwargs)
        self.env_model_client = env_model_client
        self.actor_model = self.model  

        # 冻结环境模型的所有参数
        for param in self.env_model_client.model.parameters():
            param.requires_grad = False

        # 确保环境模型处于评估模式
        self.env_model_client.model.eval()
        

    def get_train_dataloader(self):
        """Override to use custom collate function"""
        from torch.utils.data import DataLoader, DistributedSampler
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=RewardAction_collate,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        # Use DistributedSampler for multi-GPU training
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=RewardAction_collate,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Unpack inputs
        text_ids_list, image_token_ids, states, reward_targets, rtg_targets, action_token_ids = inputs
        
        # 确保模型在正确的设备上
        device = model.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        
        # 将输入数据转移到正确的设备
        text_ids_list = [x.to(device, non_blocking=True) for x in text_ids_list]
        image_token_ids = image_token_ids.to(device, non_blocking=True)
        states = states.to(device, dtype=dtype, non_blocking=True)
        
        if reward_targets is not None:
            reward_targets = reward_targets.to(device, dtype=dtype, non_blocking=True)
        if rtg_targets is not None:
            rtg_targets = rtg_targets.to(device, dtype=dtype, non_blocking=True)
            
        # 使用客户端调用环境模型服务
        reward_sampling_results = self.env_model_client.sample_rewards(
            text_ids_list=text_ids_list,
            image_token_ids=image_token_ids,
            states=states
        )
        
        # 2. 使用动作生成模型(actor_model)生成动作
        if len(action_token_ids) > 0:
            action_token_ids = [x.to(device, non_blocking=True) for x in action_token_ids]
            outputs = self.actor_model.generate_actions(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states,
                action_token_ids=action_token_ids,
                reward_sampling_results=reward_sampling_results
            )
            
            loss = outputs["action_ce_loss"]
            
            # 记录训练指标
            if hasattr(self, "log"):
                log_data = {
                    "action_ce_loss": outputs["action_ce_loss"].item(),
                }
                
                # 记录噪声与奖励嵌入的比例关系
                if "noise_norm" in reward_sampling_results:
                    log_data.update({
                        "reward_embedding_norm": reward_sampling_results["reward_embedding_norm"],
                        "noise_norm": reward_sampling_results["noise_norm"],
                        "rwd_noise_ratio": reward_sampling_results["rwd_noise_ratio"],
                        "rtg_noise_ratio": reward_sampling_results["rtg_noise_ratio"]
                    })
                
                self.log(log_data)
        else:
            raise ValueError("Stage2训练需要action_token_ids")
        
        return (loss, outputs) if return_outputs else loss


class WandbLoggingCallback(TrainerCallback):
    """Custom callback for wandb logging"""
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # Log to wandb
            log_data = {
                "train/loss": logs.get("loss", 0),
                "train/learning_rate": logs.get("learning_rate", 0),
                "train/global_step": state.global_step,
            }
            
            # 添加额外指标
            for key in ["action_ce_loss", "reward_embedding_norm", "noise_norm", 
                       "rwd_noise_ratio", "rtg_noise_ratio"]:
                if key in logs:
                    log_data[f"train/{key}"] = logs[key]
                    
            wandb.log(log_data)


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project=training_args.exp_name,
            name=f"training_{training_args.run_name or 'default'}",
            config={
                "env_model_path": model_args.env_model_path,
                "actor_model_path": model_args.actor_model_path,
                "data_path": data_args.data_path,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "max_steps": training_args.max_steps,
                "frames": data_args.frames,
                "action_frames": data_args.action_frames,
                "stage": model_args.stage,
                "parallel_mode": model_args.parallel_mode,
                "parallel_reward_groups": model_args.parallel_reward_groups,
                "reward_group_size": model_args.reward_group_size,
                "p": model_args.p,
                "gamma": model_args.gamma,
                "noise_factor": model_args.noise_factor
            }
        )

    # Load tokenizer
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.env_model_path,
        model_max_length=training_args.max_position_embeddings,
        trust_remote_code=True,
    )

    # 创建环境模型客户端而不是直接加载模型
    env_model_client = EnvModelClient(host="localhost", port=8765)
    
    # 加载动作生成模型 (要训练的模型)
    actor_model_path = model_args.actor_model_path if model_args.actor_model_path else model_args.env_model_path
    actor_config = Emu3RewardConfig.from_pretrained(actor_model_path)
    actor_model = Emu3UnifiedRewardModel.from_pretrained(
        actor_model_path,
        config=actor_config,
        tokenizer=tokenizer,
        trust_remote_code=True,
        parallel_mode=model_args.parallel_mode,
        parallel_reward_groups=model_args.parallel_reward_groups,
        reward_group_size=model_args.reward_group_size,
        p=model_args.p,
        gamma=model_args.gamma,
        noise_factor=model_args.noise_factor,
        detach_selected_reward_hs=True,
        attn_implementation=training_args.attn_type,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )

    
    # print(f"Actor model embedding size: {actor_model.get_input_embeddings().weight.shape[0]}")

    print(f"Actor model loaded from {actor_model_path}")
    
    # Enable gradient checkpointing for actor model if specified
    if training_args.gradient_checkpointing:
        actor_model.gradient_checkpointing_enable()
    
    # Initialize dataset
    train_dataset = RewardActionDataset(data_args, tokenizer, stage=model_args.stage)
    
    # 准备回调
    callbacks = []
    if training_args.report_to and "wandb" in training_args.report_to:
        callbacks.append(WandbLoggingCallback())
    
    with open(training_args.deepspeed) as f:
            ds_config = json.load(f)

    # Initialize trainer with both models
    trainer = Stage2DualModelTrainer(
        env_model_client=env_model_client,  # 使用客户端
        model=actor_model,    # 动作生成模型(训练)
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Train
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final actor model
    trainer.save_model()
    trainer.save_state()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
