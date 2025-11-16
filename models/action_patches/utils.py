from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import torch
from enum import Enum


class ActionType(Enum):
    """Type of action generation"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


class ExpertType(Enum):
    """Type of action expert"""
    FEATURE_BASED = "feature_based"
    CROSS_ATTENTION = "cross_attention"


@dataclass
class ActionExpertConfig:
    """Configuration for Action Expert models"""

    # Basic model parameters
    action_dim: int = 7
    visual_dim: int = 4096
    hidden_dim: int = 2048
    num_layers: int = 6
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    expert_type: ExpertType = ExpertType.FEATURE_BASED

    # Sequence parameters
    time_horizon: int = 10

    # Cross-attention specific (only used by CrossAttentionActionExpert)
    cross_attention_dim: Optional[int] = None
    use_rotary_emb: bool = True
    rope_dim: Optional[int] = None
    base_seq_length: int = 100
    max_context_length: int = 1000
    use_flash_attention: bool = False

    # Reward conditioning
    use_reward_conditioning: bool = True
    reward_dim: int = 14

    # Flow matching parameters
    sigma_min: float = 1e-4
    sigma_max: float = 1e-2

    # Feature-based specific (only used by FeatureActionExpert)
    feature_projection_dim: Optional[int] = None

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    scheduler_type: str = "cosine"
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 1
    ema_decay: float = 0.9999
    use_ema: bool = True

    # Sampling parameters
    num_sampling_steps: int = 20
    solver: str = "euler"  # "euler" or "heun"
    temperature: float = 1.0

    # Checkpointing and logging
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 1000

    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True

    # Stage2 specific settings
    freeze_dynamic_model: bool = True
    dynamic_model_path: Optional[str] = None
    use_reward_sampling: bool = True
    reward_sampling_prob: float = 0.85

    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.cross_attention_dim is None:
            self.cross_attention_dim = self.visual_dim

        if self.feature_projection_dim is None:
            self.feature_projection_dim = self.hidden_dim

        # Validate parameters
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert 0 <= self.sigma_min < self.sigma_max, "sigma_min must be less than sigma_max"
        assert self.accumulation_steps >= 1, "accumulation_steps must be at least 1"

        if self.solver not in ["euler", "heun"]:
            raise ValueError(f"solver must be 'euler' or 'heun', got {self.solver}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ActionExpertConfig":
        """Create config from dictionary"""
        # Handle enum conversion
        if "expert_type" in config_dict:
            config_dict["expert_type"] = ExpertType(config_dict["expert_type"])

        return cls(**config_dict)

    @classmethod
    def get_preset_configs(cls) -> Dict[str, "ActionExpertConfig"]:
        """Get preset configurations"""
        presets = {}

        # Lightweight feature-based expert
        presets["lightweight_feature"] = cls(
            expert_type=ExpertType.FEATURE_BASED,
            hidden_dim=1024,
            num_layers=4,
            num_heads=8,
            mlp_ratio=2.0,
            drop_rate=0.1,
        )

        # Heavy cross-attention expert
        presets["heavy_cross_attention"] = cls(
            expert_type=ExpertType.CROSS_ATTENTION,
            hidden_dim=3072,
            num_layers=16,
            num_heads=24,
            mlp_ratio=4.0,
            drop_rate=0.05,
            use_rotary_emb=True,
        )

        # Balanced expert (default)
        presets["balanced"] = cls()

        # Debug configuration
        presets["debug"] = cls(
            expert_type=ExpertType.FEATURE_BASED,
            hidden_dim=512,
            num_layers=2,
            num_heads=8,
            time_horizon=5,
            num_sampling_steps=10,
            checkpoint_dir="./debug_checkpoints",
        )

        return presets


class ActionCollator:
    """Collator for action expert training data"""

    def __init__(
        self,
        action_dim: int,
        max_seq_length: int = 100,
        pad_actions: bool = True,
        pad_value: float = 0.0,
    ):
        self.action_dim = action_dim
        self.max_seq_length = max_seq_length
        self.pad_actions = pad_actions
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples

        Args:
            batch: List of dictionaries containing samples

        Returns:
            Batched tensors
        """
        batch_size = len(batch)

        # Extract and pad actions
        actions = []
        seq_lengths = []

        for sample in batch:
            action = sample["actions"]  # [seq_len, action_dim]
            seq_len = action.shape[0]
            seq_lengths.append(seq_len)

            if self.pad_actions and seq_len < self.max_seq_length:
                pad_len = self.max_seq_length - seq_len
                padding = torch.full((pad_len, self.action_dim), self.pad_value, dtype=action.dtype)
                action = torch.cat([action, padding], dim=0)
            elif not self.pad_actions:
                # Keep original length, will handle later
                pass

            actions.append(action)

        # Pad to max length in batch if not using fixed padding
        max_len_in_batch = max(seq_lengths)
        if not self.pad_actions:
            padded_actions = []
            for action in actions:
                if action.shape[0] < max_len_in_batch:
                    pad_len = max_len_in_batch - action.shape[0]
                    padding = torch.full((pad_len, self.action_dim), self.pad_value, dtype=action.dtype)
                    action = torch.cat([action, padding], dim=0)
                padded_actions.append(action)
            actions = padded_actions

        # Stack actions
        actions = torch.stack(actions, dim=0)  # [B, seq_len, action_dim]
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)  # [B]

        batch_dict = {
            "target_actions": actions,
            "seq_lengths": seq_lengths,
        }

        # Add other fields from batch
        for key in batch[0].keys():
            if key != "actions":
                if isinstance(batch[0][key], torch.Tensor):
                    batch_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)
                else:
                    batch_dict[key] = [sample[key] for sample in batch]

        return batch_dict


def create_action_expert(config: ActionExpertConfig) -> torch.nn.Module:
    """
    Create action expert from configuration

    Args:
        config: Action expert configuration

    Returns:
        Action expert model
    """
    if config.expert_type == ExpertType.FEATURE_BASED:
        from .feature_expert import FeatureActionExpert
        return FeatureActionExpert(
            action_dim=config.action_dim,
            visual_dim=config.visual_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            time_horizon=config.time_horizon,
            feature_projection_dim=config.feature_projection_dim,
            use_reward_conditioning=config.use_reward_conditioning,
            reward_dim=config.reward_dim,
        )

    elif config.expert_type == ExpertType.CROSS_ATTENTION:
        from .attention_expert import CrossAttentionActionExpert
        return CrossAttentionActionExpert(
            action_dim=config.action_dim,
            visual_dim=config.visual_dim,
            hidden_dim=config.hidden_dim,
            cross_attention_dim=config.cross_attention_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            time_horizon=config.time_horizon,
            use_rotary_emb=config.use_rotary_emb,
            rope_dim=config.rope_dim,
            base_seq_length=config.base_seq_length,
            use_reward_conditioning=config.use_reward_conditioning,
            reward_dim=config.reward_dim,
        )
    else:
        raise ValueError(f"Unknown expert type: {config.expert_type}")


# def create_trainer_from_config(
#     config: ActionExpertConfig,
#     checkpoint_path: Optional[str] = None,
# ) -> "FlowMatchingTrainer":
#     """
#     Create trainer from configuration

#     Args:
#         config: Action expert configuration
#         checkpoint_path: Optional checkpoint to load

#     Returns:
#         Configured trainer
#     """
#     # Create model
#     model = create_action_expert(config)
#     model.to(config.device)

#     # Freeze if specified (for Stage2 training)
#     if config.freeze_dynamic_model:
#         model.freeze_except_action_expert()

#     # Create trainer
#     from .flow_matching import create_flow_matching_trainer
#     trainer = create_flow_matching_trainer(
#         action_expert=model,
#         learning_rate=config.learning_rate,
#         weight_decay=config.weight_decay,
#         beta1=config.beta1,
#         beta2=config.beta2,
#         scheduler_type=config.scheduler_type,
#         gradient_clip_val=config.gradient_clip_val,
#         accumulation_steps=config.accumulation_steps,
#         ema_decay=config.ema_decay,
#         use_ema=config.use_ema,
#         device=config.device,
#     )

#     # Load checkpoint if provided
#     if checkpoint_path is not None:
#         trainer.load_checkpoint(checkpoint_path)

#     return trainer


def save_config(config: ActionExpertConfig, filepath: str):
    """Save configuration to file"""
    import json
    config_dict = config.to_dict()
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(filepath: str) -> ActionExpertConfig:
    """Load configuration from file"""
    import json
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return ActionExpertConfig.from_dict(config_dict)