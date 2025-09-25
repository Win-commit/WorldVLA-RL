import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
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
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3Config,Emu3RewardConfig
from datasets import RewardActionDataset, RewardAction_collate

os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"


@dataclass
class ModelArguments:
    model_path: str = field(default="/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER/checkpoint-8000")
    stage: str = field(default="stage1", metadata={"help": "训练阶段: stage1 或 stage2"})
    parallel_mode: bool = field(default=True, metadata={"help": "是否启用并行奖励采样模式(stage2专用)"})
    parallel_reward_groups: int = field(default=5, metadata={"help": "并行奖励组数(stage2专用)"})
    reward_group_size: int = field(default=10, metadata={"help": "每组奖励数(不含state value)"})
    p: float = field(default=0.85, metadata={"help": "stage2执行奖励采样的概率"})
    gamma: float = field(default=0.9, metadata={"help": "时间加权参数(stage2专用)"})
    noise_factor: float = field(default=0.4, metadata={"help": "噪声因子"})

@dataclass
class DataArguments:
    data_path: str = field(default="")
    frames: int = field(default=2)
    action_frames: int = field(default=10)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: int = field(default=32768)
    action_tokenizer_path: str = field(default=None)

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

class UnifiedRewardTrainer(Trainer):

    def __init__(self, stage="stage1", **kwargs):
        super().__init__(**kwargs)
        self.stage = stage

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
        """通用损失计算，支持stage1和stage2"""
        # Unpack inputs
        text_ids_list, image_token_ids, states, reward_targets, rtg_targets, action_token_ids = inputs
        
        # Move to device and convert dtypes
        device = model.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        
        text_ids_list = [x.to(device, non_blocking=True) for x in text_ids_list]
        image_token_ids = image_token_ids.to(device, non_blocking=True)
        states = states.to(device, dtype=dtype, non_blocking=True)
        reward_targets = reward_targets.to(device, dtype=dtype, non_blocking=True)
        rtg_targets = rtg_targets.to(device, dtype=dtype, non_blocking=True)
        
        # Forward pass
        if self.stage == "stage1":
            outputs = model(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states,
                reward_targets=reward_targets,
                rtg_targets=rtg_targets,
            )
            loss = outputs["loss"]
            
            # 记录stage1特有指标
            if hasattr(self, "log"):
                self.log({
                    "vision_loss": outputs["vision_loss"].item(),
                    "reward_loss": outputs["reward_loss"].item() if outputs["reward_loss"] is not None else 0,
                    "rtg_loss": outputs["rtg_loss"].item() if outputs["rtg_loss"] is not None else 0,
                    "img_ce_ema": outputs["img_ce_ema"].item(),
                    "rwd_mse_ema": outputs["rwd_mse_ema"].item(),
                    "rtg_mse_ema": outputs["rtg_mse_ema"].item(),
                    "balanced_rwd": outputs["balanced_rwd"].item(),
                    "balanced_rtg": outputs["balanced_rtg"].item(),
                    "noise_norm": outputs["noise_norm"],
                    "rwd_noise_ratio": outputs["rwd_noise_ratio"],
                    "rtg_noise_ratio": outputs["rtg_noise_ratio"]
                })
            
        else:  # stage2
            # 处理action_token_ids
            if len(action_token_ids) > 0:
                outputs = model(
                    text_ids_list=text_ids_list,
                    image_token_ids=image_token_ids,
                    states=states,
                    action_token_ids=action_token_ids,
                )
                loss = outputs["action_ce_loss"]
                
                # 记录stage2特有指标
                if hasattr(self, "log"):
                    log_data = {
                        "action_ce_loss": outputs["action_ce_loss"].item(),
                    }
                    
                    # 记录噪声与奖励嵌入的比例关系
                    if "noise_norm" in outputs:
                        log_data.update({
                            "reward_embedding_norm": outputs["reward_embedding_norm"],
                            "noise_norm": outputs["noise_norm"],
                            "rwd_noise_ratio": outputs["rwd_noise_ratio"],
                            "rtg_noise_ratio": outputs["rtg_noise_ratio"]
                        })
                    
                    self.log(log_data)
            else:
                raise ValueError("     ")
        
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
            for key in ["vision_loss", "reward_loss", "rtg_loss", "img_ce_ema", "rwd_mse_ema", "rtg_mse_ema",
                        "reward_embedding_norm", "noise_norm", "rwd_noise_ratio", "rtg_noise_ratio", "balanced_rwd", "balanced_rtg"]:
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
                "model_path": model_args.model_path,
                "data_path": data_args.data_path,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "max_steps": training_args.max_steps,
                "frames": data_args.frames,
                "action_frames": data_args.action_frames,
                "stage": model_args.stage,
                "parallel_mode": model_args.parallel_mode,
            }
        )

    # Load tokenizer and config
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.max_position_embeddings,
        trust_remote_code=True,
    )

    config = Emu3RewardConfig.from_pretrained(model_args.model_path)

    # Load model
    model = Emu3UnifiedRewardModel.from_pretrained(
        model_args.model_path,
        config=config,
        tokenizer=tokenizer,
        trust_remote_code=True,
        parallel_mode=model_args.parallel_mode,
        parallel_reward_groups=model_args.parallel_reward_groups,
        reward_group_size=model_args.reward_group_size,
        p=model_args.p,
        gamma=model_args.gamma,
        noise_factor=model_args.noise_factor,
        detach_selected_reward_hs=True,
        attn_implementation= training_args.attn_type,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )

    # 设置模型训练模式
    model.set_mode(parallel_mode=model_args.parallel_mode)

    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize dataset
    train_dataset = RewardActionDataset(data_args, tokenizer, stage=model_args.stage)


    # 准备回调
    callbacks = []
    if training_args.report_to == "wandb":
        callbacks.append(WandbLoggingCallback())

    # Initialize trainer
    trainer = UnifiedRewardTrainer(
        stage=model_args.stage,
        model=model,
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

    # Save final model
    trainer.save_model()
    trainer.save_state()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()