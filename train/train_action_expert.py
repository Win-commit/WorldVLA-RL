import os
import torch
from typing import Optional
from dataclasses import dataclass, field
import wandb
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    TrainerCallback
)
import logging
import pathlib
from train.datasets import RewardActionDataset, RewardAction_collate
# Import our action expert modules
from models.action_patches import (
    ActionExpertConfig,
    ExpertType,
    create_action_expert,
)

# Import existing modules for dynamic model
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Wandb setup
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"


@dataclass
class ModelArguments:
    #=====================Dynamic model args===========================
    dynamic_model_path: str = field(
        default="/liujinxin/zhy/ICLR2026/logs/discard/after_VAE/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000"
    )
    max_position_embeddings: Optional[int] = field(default=6400)
    stage: str = field(default="stage2", metadata={"help": "训练阶段: stage1 或 stage2"})
    parallel_mode: bool = field(default=True, metadata={"help": "是否启用并行奖励采样模式(stage2专用)"})
    parallel_reward_groups: int = field(default=10, metadata={"help": "并行奖励组数(stage2专用)"})
    reward_group_size: int = field(default=10, metadata={"help": "每组奖励数(不含state value)"})
    gamma: float = field(default=0.9, metadata={"help": "时间加权参数(stage2专用)"})
    noise_factor: float = field(default=0.4, metadata={"help": "噪声因子"})

    #=====================Action expert args===========================
    action_expert_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to action expert checkpoint for resuming training"}
    )
    freeze_dynamic_model: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the dynamic model"}
    )
    action_expert_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to action expert config JSON file"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to the data"""
    data_path: str = field(default="")
    frames: int = field(default=1)
    action_frames: int = field(default=10)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: int = field(default=32768)
    action_tokenizer_path: str = field(default="/liujinxin/zhy/ICLR2026/pretrain/fast")
    null_prompt_prob: float = field(default=0)



@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments for action expert"""
    min_learning_rate: Optional[float] = field(default=None)
    exp_name: str = field(default="")

    def __post_init__(self):
        super().__post_init__()
        # Set up lr_scheduler_kwargs if min_learning_rate is provided
        if self.min_learning_rate is not None:
            if not hasattr(self, 'lr_scheduler_kwargs') or self.lr_scheduler_kwargs is None:
                self.lr_scheduler_kwargs = {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_learning_rate



class ActionExpertTrainer(Trainer):
    """
    Custom trainer for Action Expert using Flow Matching
    """

    def __init__(
        self,
        dynamic_model: Optional[Emu3UnifiedRewardModel] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dynamic_model = dynamic_model

        self.model.train()

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
        """
        Compute flow matching loss

        Args:
            model: Action expert model
            inputs: Batch dictionary with visual_features, reward_features, target_actions, etc.
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor and optionally model outputs
        """
        # Extract inputs
        text_ids_list, image_token_ids, states, _, _, _, target_actions = inputs
        device = model.device
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        
        text_ids_list = [x.to(device, non_blocking=True) for x in text_ids_list]
        image_token_ids = image_token_ids.to(device, non_blocking=True)
        states = states.to(device, dtype=dtype, non_blocking=True)
        #[B,K(K=1),seq_len,action_dim] -> [B,action_frames,action_dim]
        target_actions = target_actions.to(device, dtype=dtype, non_blocking=True).squeeze(1)

        reward_sampling_results = self.dynamic_model.sample_rewards(
            text_ids_list=text_ids_list,
            image_token_ids=image_token_ids,
            states=states
        )

        loss_dict = model.compute_flow_loss(
                reward_sampling_results = reward_sampling_results,
                target_actions=target_actions
            )
       

        loss = loss_dict['loss']

        # Log additional metrics
        if hasattr(self, 'log') and self.state.is_world_process_zero:
            log_dict = {
                "train/flow_loss": loss_dict['flow_loss'].item(),
                "train/predicted_flow_norm": loss_dict['predicted_flow_norm'].item()
            }

            self.log(log_dict)

        return (loss, loss_dict) if return_outputs else loss



class WandbLoggingCallback(TrainerCallback):
    """Custom callback for wandb logging"""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # Log to wandb
            wandb.log(logs, step=state.global_step)


def load_dynamic_model(model_args, tokenizer: Emu3Tokenizer) -> Emu3UnifiedRewardModel:
    """Load and freeze the dynamic model"""
    logger.info(f"Loading frozen dynamic model from {model_args.dynamic_model_path}")

    from models.Emu3.emu3.mllm.configuration_emu3 import Emu3RewardConfig
    config = Emu3RewardConfig.from_pretrained(model_args.dynamic_model_path)

    model = Emu3UnifiedRewardModel.from_pretrained(
        model_args.dynamic_model_path,
        config=config,
        tokenizer=tokenizer,
        trust_remote_code=True,
        parallel_mode=model_args.parallel_mode,
        parallel_reward_groups=model_args.parallel_reward_groups,
        reward_group_size=model_args.reward_group_size,
        gamma=model_args.gamma,
        noise_factor=model_args.noise_factor,
        attn_implementation= 'eager',
        torch_dtype=torch.bfloat16
    )

    if model_args.freeze_dynamic_model:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def main():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

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
                project=training_args.exp_name or "action_expert_training",
                name=f"action_expert_{training_args.run_name or 'default'}",
                config={
                    "model_path": model_args.dynamic_model_path,
                    "learning_rate": training_args.learning_rate,
                    "batch_size": training_args.per_device_train_batch_size,
                    "data_path": data_args.data_path,
                    "freeze_dynamic_model": model_args.freeze_dynamic_model,
                }
            )
            (pathlib.Path(training_args.output_dir).resolve() / "wandb_id.txt").write_text(wandb.run.id)

    # Create action expert config
    assert model_args.action_expert_config is not None, "Please provide a valid action expert config path"
    action_config = ActionExpertConfig.from_json(model_args.action_expert_config)

    logger.info(f"Action expert config: {action_config.to_dict()}")

    # Create action expert model
    action_expert = create_action_expert(
        action_config,
        model_path=model_args.action_expert_path if model_args.action_expert_path and os.path.exists(model_args.action_expert_path) else None,
        map_location=training_args.device
    )
    action_expert.to(training_args.device, dtype=torch.bfloat16)

    logger.info(f"Created action expert with {sum(p.numel() for p in action_expert.parameters())} parameters")

    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.dynamic_model_path,
        model_max_length=model_args.max_position_embeddings or 6400,
        trust_remote_code=True,
    )


    dynamic_model = load_dynamic_model(model_args, tokenizer)

    # Create dataset
    train_dataset = RewardActionDataset(data_args, tokenizer, stage=model_args.stage)

    # Create trainer
    trainer = ActionExpertTrainer(
        model=action_expert,
        dynamic_model=dynamic_model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[WandbLoggingCallback()] if training_args.report_to == "wandb" else None,
    )

    # Train
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    trainer.save_model()
    trainer.save_state()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()