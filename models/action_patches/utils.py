from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import torch
from enum import Enum
import json
import os
import logging

logger = logging.getLogger(__name__)



class ExpertType(Enum):
    FEATURE_BASED = "feature_based"
    CROSS_ATTENTION = "cross_attention"




@dataclass
class ActionExpertConfig:
    """Configuration for Action Expert models"""

    # Basic model parameters
    action_dim: int = 7
    dynamic_dim: int = 4096
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

    # Reward conditioning
    use_reward_conditioning: bool = True

    # Rotary parameters
    use_rotary_emb: bool = True
    rope_dim: Optional[int] = None
    base_seq_length: int = 100

    # Feature-based specific (only used by FeatureActionExpert)
    feature_projection_dim: Optional[int] = None


    def __post_init__(self):
        if self.cross_attention_dim is None:
            self.cross_attention_dim = self.dynamic_dim

        if self.feature_projection_dim is None:
            self.feature_projection_dim = self.hidden_dim

        assert self.action_dim > 0, "action_dim must be positive"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

    @property
    def hidden_size(self) -> int:
        """Alias for hidden_dim to match HuggingFace model convention"""
        return self.hidden_dim


    def to_dict(self) -> Dict[str, Any]:
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
        if "expert_type" in config_dict:
            config_dict["expert_type"] = ExpertType(config_dict["expert_type"])

        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "ActionExpertConfig":
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)




def create_action_expert(config: ActionExpertConfig, model_path: Optional[str] = None, map_location: str = "cpu") -> torch.nn.Module:
    """
    Create action expert from configuration

    Args:
        config: Action expert configuration
        model_path: Optional path to load model from checkpoint
        map_location: Device to map tensors to

    Returns:
        Action expert model
    """
    if config.expert_type == ExpertType.FEATURE_BASED:
        from .feature_expert import FeatureActionExpert
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading FeatureActionExpert from {model_path}")
            return FeatureActionExpert.from_pretrained(model_path, map_location=map_location, **config.to_dict())
        else:
            return FeatureActionExpert(
                action_dim=config.action_dim,
                dynamic_dim=config.dynamic_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop_rate=config.drop_rate,
                time_horizon=config.time_horizon,
                feature_projection_dim=config.feature_projection_dim,
                use_reward_conditioning=config.use_reward_conditioning,
                use_rotary_emb=config.use_rotary_emb,
                rope_dim=config.rope_dim,
                base_seq_length=config.base_seq_length,
            )

    elif config.expert_type == ExpertType.CROSS_ATTENTION:
        from .attention_expert import CrossAttentionActionExpert
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading CrossAttentionActionExpert from {model_path}")
            return CrossAttentionActionExpert.from_pretrained(model_path, map_location=map_location, **config.to_dict())
        else:
            return CrossAttentionActionExpert(
                action_dim=config.action_dim,
                dynamic_dim=config.dynamic_dim,
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
            )
    else:
        raise ValueError(f"Unknown expert type: {config.expert_type}")



def save_config(config: ActionExpertConfig, filepath: str):
    """Save configuration to file"""
    config_dict = config.to_dict()
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(filepath: str) -> ActionExpertConfig:
    """Load configuration from file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return ActionExpertConfig.from_dict(config_dict)