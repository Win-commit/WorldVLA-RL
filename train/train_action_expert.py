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
import logging
import pathlib
import pickle

# Import our action expert modules
from models.action_patches import (
    ActionExpertConfig,
    ExpertType,
    create_action_expert,
    ActionCollator
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
    """Arguments pertaining to the action expert model"""
    dynamic_model_path: str = field(
        default="/liujinxin/zhy/ICLR2026/logs/discard/after_VAE/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000",
        metadata={"help": "Path to the frozen dynamic model (Stage 1 checkpoint)"}
    )
    expert_type: str = field(
        default="feature_based",
        metadata={"help": "Type of action expert: feature_based or cross_attention"}
    )
    action_expert_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to action expert checkpoint for resuming training"}
    )
    freeze_dynamic_model: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the dynamic model"}
    )
    # Action expert model parameters
    action_dim: int = field(default=7, metadata={"help": "Action dimension"})
    visual_dim: int = field(default=4096, metadata={"help": "Visual feature dimension"})
    hidden_dim: int = field(default=2048, metadata={"help": "Action expert hidden dimension"})
    num_layers: int = field(default=6, metadata={"help": "Number of DiT layers"})
    num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    time_horizon: int = field(default=10, metadata={"help": "Action sequence length"})
    use_reward_conditioning: bool = field(
        default=True,
        metadata={"help": "Whether to use reward conditioning"}
    )
    reward_dim: int = field(default=14, metadata={"help": "Reward feature dimension"})


@dataclass
class DataArguments:
    """Arguments pertaining to the data"""
    data_path: str = field(default="", metadata={"help": "Path to training data"})
    max_seq_length: int = field(default=100, metadata={"help": "Maximum sequence length"})
    frames: int = field(default=2, metadata={"help": "Number of frames"})
    action_frames: int = field(default=10, metadata={"help": "Number of action frames"})
    null_prompt_prob: float = field(default=0.15, metadata={"help": "Probability of null prompt"})
    # Data loading parameters
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments for action expert"""
    # Inherit from HF TrainingArguments and add custom fields
    attn_type: str = field(default="eager")
    min_learning_rate: Optional[float] = field(default=None)
    exp_name: str = field(default="")

    # Action expert specific parameters
    gradient_clip_val: float = field(default=1.0, metadata={"help": "Gradient clipping value"})
    ema_decay: float = field(default=0.9999, metadata={"help": "EMA decay rate"})
    use_ema: bool = field(default=True, metadata={"help": "Whether to use EMA"})

    # Flow matching parameters
    sigma_min: float = field(default=1e-4, metadata={"help": "Minimum sigma for flow matching"})
    sigma_max: float = field(default=1e-2, metadata={"help": "Maximum sigma for flow matching"})
    num_sampling_steps: int = field(default=20, metadata={"help": "Number of sampling steps"})

    # Validation parameters
    eval_during_training: bool = field(default=True, metadata={"help": "Whether to evaluate during training"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation frequency"})

    def __post_init__(self):
        super().__post_init__()
        # Set up lr_scheduler_kwargs if min_learning_rate is provided
        if self.min_learning_rate is not None:
            if not hasattr(self, 'lr_scheduler_kwargs') or self.lr_scheduler_kwargs is None:
                self.lr_scheduler_kwargs = {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_learning_rate


class ActionExpertDataset(torch.utils.data.Dataset):
    """
    Dataset for action expert training
    Expected format: Each sample contains visual_features, reward_features, and target_actions
    """

    def __init__(
        self,
        data_path: str,
        action_dim: int = 7,
        max_seq_length: int = 100,
        frames: int = 2,
        action_frames: int = 10,
        null_prompt_prob: float = 0.15,
        cache_dir: Optional[str] = None,
    ):
        self.data_path = data_path
        self.action_dim = action_dim
        self.max_seq_length = max_seq_length
        self.frames = frames
        self.action_frames = action_frames
        self.null_prompt_prob = null_prompt_prob
        self.cache_dir = cache_dir

        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            self.scenes = pickle.load(f)

        logger.info(f"Loaded {len(self.scenes)} scenes")

        # Check if this is main process for logging
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset

        Returns:
            dict: {
                'visual_features': torch.Tensor,  # [visual_dim]
                'reward_features': torch.Tensor,   # [reward_dim] or None
                'target_actions': torch.Tensor,   # [seq_len, action_dim]
                'seq_length': int,                 # actual sequence length
                'dynamic_hidden_states': torch.Tensor  # [dyn_seq_len, dyn_hidden_dim] (for cross-attention)
            }
        """
        scene_data = self.scenes[idx]

        # Extract features from scene data
        # This is a placeholder - you need to adapt this to your actual data format
        visual_features = scene_data.get('visual_features', torch.randn(4096))
        reward_features = scene_data.get('reward_features', torch.randn(14))
        target_actions = scene_data.get('actions', torch.randn(self.action_frames, self.action_dim))

        # Ensure tensors have correct shapes
        if visual_features.dim() == 0:
            visual_features = visual_features.unsqueeze(0)
        if reward_features.dim() == 0:
            reward_features = reward_features.unsqueeze(0)

        seq_length = min(target_actions.shape[0], self.action_frames)

        # Dynamic hidden states (for cross-attention expert)
        dynamic_hidden_states = scene_data.get('dynamic_hidden_states',
                                            torch.randn(100, 4096))  # [seq_len, hidden_dim]

        return {
            'visual_features': visual_features,
            'reward_features': reward_features,
            'target_actions': target_actions[:seq_length],
            'seq_length': seq_length,
            'dynamic_hidden_states': dynamic_hidden_states,
        }


class ActionExpertTrainer(Trainer):
    """
    Custom trainer for Action Expert using Flow Matching
    """

    def __init__(
        self,
        dynamic_model: Optional[Emu3UnifiedRewardModel] = None,
        expert_type: str = "feature_based",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dynamic_model = dynamic_model
        self.expert_type = expert_type

        # Set model to train mode
        self.model.train()

    def get_train_dataloader(self):
        """Override to use custom collate function"""
        from torch.utils.data import DataLoader, DistributedSampler

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Create custom collator
        collator = ActionCollator(
            action_dim=self.model.action_dim,
            max_seq_length=self.train_dataset.max_seq_length
        )

        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        # Use DistributedSampler for multi-GPU training
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Override evaluation dataloader"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            return None

        # Create custom collator
        collator = ActionCollator(
            action_dim=self.model.action_dim,
            max_seq_length=eval_dataset.max_seq_length
        )

        from torch.utils.data import DataLoader, DistributedSampler

        # Use DistributedSampler for evaluation
        sampler = DistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
        )

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=sampler,
            collate_fn=collator,
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
        target_actions = inputs['target_actions']  # [B, seq_len, action_dim]
        seq_lengths = inputs.get('seq_lengths')  # [B]
        visual_features = inputs['visual_features']  # [B, visual_dim]
        reward_features = inputs.get('reward_features')  # [B, reward_dim] or None

        # Prepare loss computation inputs based on expert type
        if self.expert_type == "feature_based":
            loss_dict = model.compute_flow_loss(
                visual_features=visual_features,
                reward_features=reward_features,
                target_actions=target_actions,
                seq_lengths=seq_lengths
            )
        elif self.expert_type == "cross_attention":
            dynamic_hidden_states = inputs['dynamic_hidden_states']  # [B, dyn_seq_len, dyn_hidden_dim]
            loss_dict = model.compute_flow_loss(
                dynamic_hidden_states=dynamic_hidden_states,
                visual_features=visual_features,
                reward_features=reward_features,
                target_actions=target_actions,
                seq_lengths=seq_lengths
            )
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")

        loss = loss_dict['loss']

        # Log additional metrics
        if hasattr(self, 'log') and self.state.is_world_process_zero:
            log_dict = {
                "train/flow_loss": loss_dict['flow_loss'].item(),
                "train/flow_magnitude": loss_dict['flow_magnitude'].item(),
                "train/sigma_mean": loss_dict['sigma_mean'].item(),
                "train/timestep_mean": loss_dict['timestep_mean'].item(),
            }

            if 'action_error' in loss_dict:
                log_dict["train/action_error"] = loss_dict['action_error'].item()

            if 'attention_entropy' in loss_dict:
                log_dict["train/attention_entropy"] = loss_dict['attention_entropy'].item()

            self.log(log_dict)

        return (loss, loss_dict) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and return metrics
        """
        # Run standard evaluation
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Sample some actions for visualization
        if hasattr(self.model, 'sample_actions') and self.state.is_world_process_zero:
            try:
                # Get a sample batch from eval dataset
                eval_dataset = eval_dataset or self.eval_dataset
                if eval_dataset and len(eval_dataset) > 0:
                    sample_inputs = eval_dataset[0]

                    # Add batch dimension
                    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                            for k, v in sample_inputs.items()}

                    # Sample actions
                    with torch.no_grad():
                        if self.expert_type == "feature_based":
                            sampled_actions = self.model.sample_actions(
                                visual_features=batch['visual_features'],
                                reward_features=batch['reward_features'],
                                num_steps=20,
                                temperature=1.0
                            )
                        else:  # cross_attention
                            sampled_actions = self.model.sample_actions(
                                dynamic_hidden_states=batch['dynamic_hidden_states'],
                                visual_features=batch['visual_features'],
                                reward_features=batch['reward_features'],
                                num_steps=20,
                                temperature=1.0
                            )

                    # Log sampled action stats
                    result.update({
                        f"{metric_key_prefix}/sampled_action_mean": sampled_actions.mean().item(),
                        f"{metric_key_prefix}/sampled_action_std": sampled_actions.std().item(),
                        f"{metric_key_prefix}/sampled_action_max": sampled_actions.max().item(),
                        f"{metric_key_prefix}/sampled_action_min": sampled_actions.min().item(),
                    })

            except Exception as e:
                logger.warning(f"Failed to sample actions during evaluation: {e}")

        return result


class WandbLoggingCallback(TrainerCallback):
    """Custom callback for wandb logging"""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # Log to wandb
            wandb.log(logs, step=state.global_step)


def load_frozen_dynamic_model(model_path: str, tokenizer: Emu3Tokenizer) -> Emu3UnifiedRewardModel:
    """Load and freeze the dynamic model"""
    logger.info(f"Loading frozen dynamic model from {model_path}")

    # Load config and model
    from models.Emu3.emu3.mllm.configuration_emu3 import Emu3RewardConfig
    config = Emu3RewardConfig.from_pretrained(model_path)

    model = Emu3UnifiedRewardModel.from_pretrained(
        model_path,
        config=config,
        tokenizer=tokenizer,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    logger.info(f"Loaded frozen dynamic model with {sum(p.numel() for p in model.parameters())} parameters")

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
        wandb.init(
            project=training_args.exp_name or "action_expert_training",
            name=f"action_expert_{training_args.run_name or 'default'}",
            config={
                "model_path": model_args.dynamic_model_path,
                "expert_type": model_args.expert_type,
                "action_dim": model_args.action_dim,
                "hidden_dim": model_args.hidden_dim,
                "num_layers": model_args.num_layers,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "data_path": data_args.data_path,
                "freeze_dynamic_model": model_args.freeze_dynamic_model,
                "use_reward_conditioning": model_args.use_reward_conditioning,
                "ema_decay": training_args.ema_decay,
                "use_ema": training_args.use_ema,
            }
        )

    # Create action expert config
    action_config = ActionExpertConfig(
        expert_type=ExpertType(model_args.expert_type),
        action_dim=model_args.action_dim,
        visual_dim=model_args.visual_dim,
        hidden_dim=model_args.hidden_dim,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        time_horizon=model_args.time_horizon,
        use_reward_conditioning=model_args.use_reward_conditioning,
        reward_dim=model_args.reward_dim,
        sigma_min=training_args.sigma_min,
        sigma_max=training_args.sigma_max,
        learning_rate=training_args.learning_rate,
        device=training_args.device,
    )

    logger.info(f"Action expert config: {action_config.to_dict()}")

    # Create action expert model
    action_expert = create_action_expert(action_config)
    action_expert.to(training_args.device)

    logger.info(f"Created action expert with {sum(p.numel() for p in action_expert.parameters())} parameters")

    # Load tokenizer (needed for dynamic model)
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.dynamic_model_path,
        model_max_length=training_args.max_position_embeddings or 6400,
        trust_remote_code=True,
    )

    # Load frozen dynamic model if needed
    dynamic_model = None
    if model_args.freeze_dynamic_model:
        dynamic_model = load_frozen_dynamic_model(model_args.dynamic_model_path, tokenizer)

    # Create dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    train_dataset = ActionExpertDataset(
        data_path=data_args.data_path,
        action_dim=model_args.action_dim,
        max_seq_length=data_args.max_seq_length,
        frames=data_args.frames,
        action_frames=data_args.action_frames,
        null_prompt_prob=data_args.null_prompt_prob,
        cache_dir=data_args.cache_dir,
    )

    eval_dataset = None
    if training_args.do_eval:
        # For now, use the same dataset for evaluation (you might want a separate eval set)
        eval_dataset = train_dataset

    # Create trainer
    trainer = ActionExpertTrainer(
        model=action_expert,
        dynamic_model=dynamic_model,
        expert_type=model_args.expert_type,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLoggingCallback()] if training_args.report_to == "wandb" else None,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model()
    trainer.save_state()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()