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
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3MoE,Emu3Model
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3Config, Emu3RewardConfig
from datasets import RewardActionDataset, RewardAction_collate
import deepspeed
import json
# import logging
# logging.getLogger("transformers").setLevel(logging.DEBUG)
import time
import websockets
import asyncio
import numpy as np
import pickle  
import logging
import random
from datetime import datetime
import pathlib

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"


@dataclass
class ModelArguments:
    actor_model_path: str = field(default="", 
                              metadata={"help": "动作生成模型路径(可选，如果为空则使用env_model_path初始化)"})
    stage: str = field(default="stage2", metadata={"help": "训练阶段: stage1 或 stage2"})
    parallel_mode: bool = field(default=True, metadata={"help": "是否启用并行奖励采样模式(stage2专用)"})
    parallel_reward_groups: int = field(default=5, metadata={"help": "并行奖励组数(stage2专用)"})
    reward_group_size: int = field(default=10, metadata={"help": "每组奖励token数量(stage2专用)"})
    p: float = field(default=0.85, metadata={"help": "stage2执行奖励采样的概率"})
    gamma: float = field(default=0.9, metadata={"help": "时间加权参数(stage2专用)"})
    noise_factor: float = field(default=0.1, metadata={"help": "噪声因子(stage2专用)"})
    # 添加环境模型服务器配置
    env_servers: str = field(default="localhost:8765", 
                         metadata={"help": "环境模型服务器列表，格式为'host:port1,port2,...'"})


@dataclass
class DataArguments:
    data_path: str = field(default="")
    frames: int = field(default=2)
    action_frames: int = field(default=10)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: int = field(default=32768)
    action_tokenizer_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.15)


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
    def __init__(self, servers=None, rank=0, max_retries=5):
        self.servers = servers or [("localhost", 8765)]
        self.rank = rank
        self.primary_server_idx = rank % len(self.servers)
        self.max_retries = max_retries
        
        # 请求计数
        self.total_requests = 0
        self.successful_requests = 0
        
        self.loop = asyncio.new_event_loop()
        
        self.connection = None
        self.server_idx = self.primary_server_idx
        
        # 服务器状态追踪
        self.server_status = {i: {"available": True, "last_failure": 0, "failure_count": 0} 
                             for i in range(len(self.servers))}
        
        self.connection = self.loop.run_until_complete(self._initialize_connection())
        
        host, port = self.servers[self.server_idx]
        logger.info(f"[Rank {self.rank}] 初始化客户端，已连接到 {host}:{port}")
    
    async def _initialize_connection(self):
        """初始化WebSocket连接"""
        for attempt in range(self.max_retries):
            server_idx = self.server_idx
            host, port = self.servers[server_idx]
            uri = f"ws://{host}:{port}"
            
            try:
                # logger.info(f"[Rank {self.rank}] 连接到 {uri}...")
                # 建立连接
                connection = await websockets.connect(
                    uri,
                    max_size=1024*1024*500,
                    compression=None,
                    close_timeout=300
                )
                logger.info(f"[Rank {self.rank}] 成功连接到 {uri}")
                return connection
            except Exception as e:
                logger.error(f"[Rank {self.rank}] 连接失败: {str(e)}")
                self.server_idx = (self.server_idx + 1) % len(self.servers)
                await asyncio.sleep(1)
        
        raise RuntimeError(f"无法连接到任何服务器")
    
    async def _ensure_connection(self):
        """确保连接可用，如果不可用则重新连接"""
        if self.connection is None:
            logger.info(f"[Rank {self.rank}] 连接不可用，重新建立...")
            self.connection = await self._initialize_connection()
    
    async def sample_rewards_async(self, text_ids_list, image_token_ids, states):
        """异步获取奖励采样结果"""
        request_id = f"{self.rank}-{self.total_requests}"
        self.total_requests += 1
        
        # 准备数据
        data = {
            'text_ids_list': [ids.cpu().numpy() for ids in text_ids_list],
            'image_token_ids': image_token_ids.cpu().numpy(),
            'states': states.cpu().to(torch.float32).numpy()
        }
        binary_data = pickle.dumps(data)
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                await self._ensure_connection()
                host, port = self.servers[self.server_idx]
                uri = f"ws://{host}:{port}"
                
                # 发送请求
                # logger.info(f"[Rank {self.rank}] 请求 #{request_id} 尝试 {attempt+1}: 发送数据到 {uri}")
                await self.connection.send(binary_data)
                
                binary_response = await self.connection.recv()
                
                if isinstance(binary_response, str):
                    results = json.loads(binary_response)
                    # logger.info(f"[Rank {self.rank}] 请求 #{request_id}: 收到JSON响应")
                else:
                    results = pickle.loads(binary_response)
                    # logger.info(f"[Rank {self.rank}] 请求 #{request_id}: 收到二进制响应 ({len(binary_response)/(1024*1024):.2f}MB)")
                
                device = image_token_ids.device
                dtype = torch.bfloat16
                   
                
                # 构建完整结果
                reward_results = {
                    'reward_preds_group_mean': torch.from_numpy(results['reward_preds_group_mean']).to(device=device, dtype=dtype),
                    'best_reward_group': torch.from_numpy(results['best_reward_group']).to(device=device, dtype=torch.long),
                    'critical_segments': torch.from_numpy(results['critical_segments']).to(device=device, dtype=dtype),
                    'context_lengths': results['context_lengths'],
                    'noise_norm': results['noise_norm'],
                    'reward_embedding_norm': results['reward_embedding_norm'],
                    'rwd_noise_ratio': results['rwd_noise_ratio'],
                    'rtg_noise_ratio': results['rtg_noise_ratio']
                }
                
                end_time = time.time()
                self.successful_requests += 1
                
                # logger.info(f"[Rank {self.rank}] 请求 #{request_id} 成功完成，用时 {end_time-start_time:.2f}s，"
                #           f"使用服务器 {uri} ({self.successful_requests}/{self.total_requests} 成功)")
                
                return reward_results
                
            except Exception as e:
                logger.error(f"[Rank {self.rank}] 请求 #{request_id} 失败: {type(e).__name__}: {str(e)[:200]}...")
                
                # 关闭失败的连接
                if self.connection:
                    try:
                        await self.connection.close()
                    except:
                        pass
                self.connection = None
                
                # 标记服务器失败并切换
                self.server_status[self.server_idx]["last_failure"] = time.time()
                self.server_status[self.server_idx]["failure_count"] += 1
                
                # 切换服务器并重试
                if attempt < self.max_retries - 1:
                    old_server_idx = self.server_idx
                    self.server_idx = (self.server_idx + 1) % len(self.servers)
                    old_host, old_port = self.servers[old_server_idx]
                    new_host, new_port = self.servers[self.server_idx]
                    
                    logger.info(f"[Rank {self.rank}] 切换服务器从 {old_host}:{old_port} 到 {new_host}:{new_port}，"
                              f"将在 {(attempt+1)}s 后重试")
                    
                    # 添加重试延迟
                    await asyncio.sleep((attempt+1) + random.random())
                else:
                    logger.critical(f"[Rank {self.rank}] 请求 #{request_id} 在 {self.max_retries} 次尝试后失败")
                    raise RuntimeError(f"采样奖励失败，已尝试 {self.max_retries} 次")
    
    def sample_rewards(self, text_ids_list, image_token_ids, states):
        return self.loop.run_until_complete(
            self.sample_rewards_async(text_ids_list, image_token_ids, states)
        )
    
    def __del__(self):
        """析构函数 - 关闭事件循环"""
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            # 关闭所有连接
            if hasattr(self, 'connection') and self.connection and not self.connection.closed:
                try:
                    self.loop.run_until_complete(self.connection.close())
                except:
                    pass
            
            # 关闭事件循环
            self.loop.close()

class Stage2DualModelTrainer(Trainer):

    def __init__(
        self,
        env_model_client,  
        **kwargs
    ):
        super().__init__(**kwargs)
        self.env_model_client = env_model_client
        self.actor_model = self.model  
        

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
        text_ids_list, image_token_ids, states, reward_targets, rtg_targets, action_token_ids = inputs
        
        device = model.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        
        text_ids_list = [x.to(device, non_blocking=True) for x in text_ids_list]
        image_token_ids = image_token_ids.to(device, non_blocking=True)
        states = states.to(device, dtype=dtype, non_blocking=True)
        
        if reward_targets is not None:
            reward_targets = reward_targets.to(device, dtype=dtype, non_blocking=True)
        if rtg_targets is not None:
            rtg_targets = rtg_targets.to(device, dtype=dtype, non_blocking=True)
            
        reward_sampling_results = self.env_model_client.sample_rewards(
            text_ids_list=text_ids_list,
            image_token_ids=image_token_ids,
            states=states
        )
        
        if len(action_token_ids) > 0:
            if isinstance(action_token_ids[0], list):
                action_token_ids = [[tensor.to(device, non_blocking=True) if isinstance(tensor, torch.Tensor) else 
                                    torch.tensor(tensor, device=device) for tensor in action_group] 
                                    for action_group in action_token_ids]
            else:
                action_token_ids = [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else
                                    torch.tensor(x, device=device) for x in action_token_ids]
            outputs = self.actor_model.generate_actions(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states,
                action_token_ids=action_token_ids,
                reward_sampling_results=reward_sampling_results
            )
            
            loss = outputs["action_ce_loss"]
            
            # 记录训练指标
            if hasattr(self, "log") and return_outputs:
                log_data = {
                    "action_ce_loss": outputs["action_ce_loss"].item(),
                }
                
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
            log_data = {
                "train/loss": logs.get("loss", 0),
                "train/learning_rate": logs.get("learning_rate", 0),
                "train/global_step": state.global_step,
            }
            
            for key in ["action_ce_loss", "reward_embedding_norm", "noise_norm", 
                       "rwd_noise_ratio", "rtg_noise_ratio"]:
                if key in logs:
                    log_data[f"train/{key}"] = logs[key]
                    
            wandb.log(log_data)


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir,exist_ok=True)
        
    # Initialize wandb
    if training_args.report_to and "wandb" in training_args.report_to:
        if training_args.resume_from_checkpoint:
            run_id = (pathlib.Path(training_args.output_dir).resolve() / "wandb_id.txt").read_text().strip()
            wandb.init(
                project=training_args.exp_name,
                id=run_id, 
                resume="must"
            )
        else:
            wandb.init(
                project=training_args.exp_name,
                name=f"training_{training_args.run_name or 'default'}",
                config={
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
                    "noise_factor": model_args.noise_factor,
                    "env_servers": model_args.env_servers
                }
            )
            (pathlib.Path(training_args.output_dir).resolve() / "wandb_id.txt").write_text(wandb.run.id)


    # Load tokenizer
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.actor_model_path,
        model_max_length=training_args.max_position_embeddings,
        trust_remote_code=True,
    )

    servers = []
    ip = model_args.env_servers.split(":")[0]
    ports = model_args.env_servers.split(":")[1].split(",")
    for port in ports:
        servers.append((ip, int(port)))
    
    print(f"Environment Server List: {servers}")

    
    actor_model_path = model_args.actor_model_path
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

    actor_model.model = Emu3Model.from_pretrained(
        "/liujinxin/zhy/UniVLA/ckpts/UniVLA/UNIVLA_LIBERO_VIDEO_BS192-8K_original",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )


    rank = getattr(training_args, 'local_rank', 0)
    env_model_client = EnvModelClient(servers=servers, rank=rank)
    


    print(f"Actor model loaded from {actor_model_path}")
    
    # Enable gradient checkpointing for actor model if specified
    if training_args.gradient_checkpointing:
        actor_model.gradient_checkpointing_enable()
    
    # Initialize dataset
    train_dataset = RewardActionDataset(data_args, tokenizer, stage=model_args.stage)
    
    # 准备回调
    # callbacks = []
    # if training_args.report_to and "wandb" in training_args.report_to:
    #     callbacks.append(WandbLoggingCallback())
    

    # Initialize trainer with both models
    trainer = Stage2DualModelTrainer(
        env_model_client=env_model_client,  
        model=actor_model,   
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
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
