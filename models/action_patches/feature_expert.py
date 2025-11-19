import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from .dit_blocks import DiTBlock, ActionRotaryPosEmbed
import logging
import glob
from .utils import ActionExpertConfig, ExpertType
import json
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.normalization import AdaLayerNormSingle

from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import argparse
logger = logging.getLogger(__name__)


class FeatureActionExpert(nn.Module):
    """
    Feature-based Action Expert using Flow Matching with DiT blocks

    This approach extracts features from the dynamic model and uses them as conditioning
    for the flow matching process to generate actions.
    """

    def __init__(
        self,
        action_dim: int = 7,
        dynamic_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 1.0,
        drop_rate: float = 0.1,
        time_horizon: int = 10,
        # Feature projection parameters
        feature_projection_dim: Optional[int] = None,
        use_reward_conditioning: bool = True,
        # RoPE parameters
        use_rotary_emb: bool = True,
        rope_dim: Optional[int] = None,
        base_seq_length: int = 10,
        # Flow matching weighting scheme parameters
        weighting_scheme: str = "none",
        logit_mean: Optional[float] = None,
        logit_std: Optional[float] = None,
        mode_scale: Optional[float] = None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.use_reward_conditioning = use_reward_conditioning
        self.use_rotary_emb = use_rotary_emb
        self.hidden_dim = hidden_dim

        self.config = ActionExpertConfig(
            action_dim=action_dim,
            dynamic_dim=dynamic_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            time_horizon=time_horizon,
            use_reward_conditioning=use_reward_conditioning,
            use_rotary_emb=use_rotary_emb,
            feature_projection_dim=feature_projection_dim,
            expert_type=ExpertType.FEATURE_BASED,
        )


        if feature_projection_dim is None:
            feature_projection_dim = hidden_dim

        # Feature projection from dynamic model
        input_dim = dynamic_dim
        if use_reward_conditioning:
            input_dim += dynamic_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, feature_projection_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop_rate),
            nn.Linear(feature_projection_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop_rate),
        )

        if use_reward_conditioning:
            self.reward_aggregator = nn.Sequential(
                nn.Linear(dynamic_dim, dynamic_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dynamic_dim, dynamic_dim)
            )

        # Time embeddings for flow matching
        self.action_time_embed = AdaLayerNormSingle(self.hidden_dim, use_additional_conditions=False)

        # Action embedding
        self.action_proj_in = nn.Linear(action_dim, hidden_dim)

        # Rotary position embeddings
        if use_rotary_emb:
            rope_dim = rope_dim or hidden_dim
            self.action_rope = ActionRotaryPosEmbed(
                dim=rope_dim,
                base_seq_length=base_seq_length,
                theta=10000.0,
            )

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attention_type="self"
            )
            for _ in range(num_layers)
        ])


        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.hidden_dim) / self.hidden_dim **0.5)
        self.action_proj_out = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

        #=============SD3 STYLE WEIGHTING SCHEME================
        self.flow_matching_args = argparse.Namespace()
        self.flow_matching_args.weighting_scheme = weighting_scheme
        self.flow_matching_args.logit_mean = logit_mean
        self.flow_matching_args.logit_std = logit_std
        self.flow_matching_args.mode_scale = mode_scale
        self.flow_matching_args.scheduler = FlowMatchEulerDiscreteScheduler()


    def _init_weights(self):
        """Initialize weights using xavier uniform for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extract_features(
        self,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract and project features from dynamic model

        Args:
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim] or [B, S, visual_dim]
                              Can be either pre-aggregated [B, H] or time sequence [B, S, H]
                              For time sequences, uses attention weighting over time dimension

        Returns:
            Projected features [B, hidden_dim]
        """
        features = visual_features

        if self.use_reward_conditioning and reward_features is not None:
            if reward_features.dim() == 3:
                reward_agg = self.reward_aggregator(reward_features)  # [B, reward_group_size, H]
                reward_group_size = reward_agg.size(1)
                decay_factor = 0.9

                # Weights: [decay_factor^1, decay_factor^2, ..., decay_factor^reward_group_size]
                indices = torch.arange(1, reward_group_size + 1, device=reward_agg.device, dtype=reward_agg.dtype)
                attention_weights = (decay_factor ** indices).unsqueeze(0)  # [1, reward_group_size]
                reward_features_agg = (reward_agg * attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
            else:
                reward_features_agg = reward_features

            # Concatenate visual and aggregated reward features
            features = torch.cat([visual_features, reward_features_agg], dim=-1)

        # Project to hidden dimension
        projected_features = self.feature_projector(features)

        return projected_features

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = actions.size(0), actions.size(1)

        conditioning_features = self.extract_features(visual_features, reward_features)
        conditioning_features = conditioning_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Time embeddings
        action_temb, action_embedded_timestep = self.action_time_embed(
            timesteps.flatten(),
            batch_size=batch_size,
            hidden_dtype=conditioning_features.dtype,
            )


        # Action embeddings
        action_emb = self.action_proj_in(actions)  # [B, seq_len, hidden_dim]

        # Combine action embeddings with conditioning
        conditioned_action_emb = action_emb + conditioning_features

        # Generate rotary embeddings for action sequence
        rotary_emb = None
        if self.use_rotary_emb:
            rotary_emb = self.action_rope(actions, seq_len)

        # Apply DiT blocks
        hidden_states = conditioned_action_emb
        for dit_block in self.dit_blocks:
            hidden_states = dit_block(
                x=hidden_states,
                temb=action_temb,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb
            )
        hidden_states = self.final_norm(hidden_states)
        action_embedded_timestep = action_embedded_timestep.view(batch_size, seq_len, -1)
        action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
        action_shift, action_scale = action_scale_shift_values[:,:,0], action_scale_shift_values[:,:,1]
        hidden_states = hidden_states * (1 + action_scale) + action_shift

        flow_prediction = self.action_proj_out(hidden_states)

        return flow_prediction

    def compute_flow_loss(
        self,
        reward_sampling_results: Dict,
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, action_dim = target_actions.shape

        visual_features = reward_sampling_results['last_hidden_states'][torch.arange(batch_size), reward_sampling_results["last_image_token_pos"].squeeze(-1)]  # [B, visual_dim]
        visual_features = visual_features.to(dtype=target_actions.dtype)

        reward_features = reward_sampling_results["critical_segments"].squeeze(1)[:,:-1,:]  #[B, S, visual_dim] remove rtg represntation
        reward_features = reward_features.to(dtype=target_actions.dtype)

        # Sample random timesteps
        action_weights = compute_density_for_timestep_sampling(
            weighting_scheme=self.flow_matching_args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.flow_matching_args.logit_mean,
            logit_std=self.flow_matching_args.logit_std,
            mode_scale=self.flow_matching_args.mode_scale,
        ) #[0,1]
        action_indices = (action_weights * self.flow_matching_args.scheduler.config.num_train_timesteps).long() #[0,1000]
        scheduler_sigmas = self.flow_matching_args.scheduler.sigmas.clone().to(device=target_actions.device, dtype=target_actions.dtype)
        action_sigmas = scheduler_sigmas[action_indices] #[0,1]   
        action_timesteps = (action_sigmas * 1000.0).long() # [B]


        action_timesteps = action_timesteps.unsqueeze(-1).repeat(1, seq_len)  # [B, seq_len]
        action_ss = action_sigmas.reshape(-1, 1, 1).repeat(1, 1, action_dim) #[B,1,action_dim]
        

        # Sample noise (Gaussian)
        noise = torch.randn_like(target_actions,dtype=target_actions.dtype, device=target_actions.device)

        # x(t) = (1-t) * x_1 + t * noise
        noisy_actions = (1.0 - action_ss) * target_actions + action_ss * noise

        # v = dx/dt = noise - actions
        target_flow = noise - target_actions

        predicted_flow = self.forward(
            actions=noisy_actions,
            timesteps=action_timesteps,
            visual_features=visual_features,
            reward_features=reward_features,
        )
        

        action_weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.flow_matching_args.weighting_scheme, 
            sigmas=action_sigmas
        ).reshape(-1, 1, 1).repeat(1, 1, action_dim)
        flow_loss = action_weights.float() * (predicted_flow.float() - target_flow.float()).pow(2)
        
        flow_loss = flow_loss.mean()

        with torch.no_grad():
            flow_magnitude = predicted_flow.norm(dim=-1).mean()


        return {
            'loss': flow_loss,
            'flow_loss': flow_loss,
            'target_flow_norm': target_flow.norm(dim=-1).mean(),
            'predicted_flow_norm': flow_magnitude,
        }

    # @torch.no_grad()
    # def sample_actions(
    #     self,
    #     visual_features: torch.Tensor,
    #     reward_features: Optional[torch.Tensor] = None,
    #     num_steps: int = 20,
    #     solver: str = "euler",
    #     stochastic: bool = False,
    #     temperature: float = 1.0,
    # ) -> torch.Tensor:
    #     """
    #     Sample actions using Flow Matching ODE solver

    #     Args:
    #         visual_features: Visual features from dynamic model [B, visual_dim]
    #         reward_features: Reward features from dynamic model [B, reward_dim] or [B, S+1, visual_dim]
    #         num_steps: Number of discretization steps
    #         solver: ODE solver type ('euler' or 'heun')
    #         stochastic: Whether to use stochastic sampling (adds noise back)
    #         temperature: Temperature for sampling

    #     Returns:
    #         Sampled actions [B, seq_len, action_dim]
    #     """
    #     batch_size = visual_features.size(0)
    #     device = visual_features.device

    #     # Initialize with pure noise (sigma=1.0)
    #     actions = torch.randn(batch_size, self.time_horizon, self.action_dim, device=device)

    #     # Discretize sigma from 1.0 -> 0.0
    #     sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    #     for i in range(num_steps):
    #         current_sigma = sigmas[i].expand(batch_size)
    #         next_sigma = sigmas[i + 1].expand(batch_size)
    #         sigma_diff = next_sigma - current_sigma  # negative value

    #         # Predict flow at current sigma
    #         flow = self.forward(
    #             actions=actions,
    #             timesteps=current_sigma,
    #             visual_features=visual_features,
    #             reward_features=reward_features,
    #         )

    #         if stochastic:
    #             # Genie-style stochastic sampling
    #             # 1. Predict clean data x0
    #             x0 = actions - current_sigma.view(-1, 1, 1) * flow
    #             # 2. Re-add noise for next step
    #             noise = torch.randn_like(actions)
    #             actions = (1.0 - next_sigma.view(-1, 1, 1)) * x0 + next_sigma.view(-1, 1, 1) * noise
    #         else:
    #             # Deterministic Euler step
    #             # dx/dσ = flow, so x_{i+1} = x_i + (σ_{i+1} - σ_i) * flow
    #             actions = actions + sigma_diff.view(-1, 1, 1) * flow * temperature

    #     return actions


    def get_trainable_parameters(self):
        """Get parameters that should be trained (excluding frozen components)"""
        return [p for _, p in self.named_parameters() if p.requires_grad]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        map_location: str = "cpu",
        **kwargs
    ) -> "FeatureActionExpert":
        if os.path.isdir(model_path):

            checkpoint_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
            checkpoint_files.extend(glob.glob(os.path.join(model_path, "model.safetensors")))
            if checkpoint_files:
                state_dict_path = checkpoint_files[0]
            else:
                state_dict_path = model_path
        else:
            state_dict_path = model_path

        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")


        try:
            state_dict = torch.load(state_dict_path, map_location=map_location, weights_only=False)
        except Exception:
            state_dict = torch.load(state_dict_path, map_location=map_location, weights_only=True)

        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        else:
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Merge with kwargs (kwargs takes precedence)
            for key, value in kwargs.items():
                if key in config_dict:
                    config_dict[key] = value
            # Create instance with config
            instance = cls(**config_dict)
        else:
            # Use default config
            instance = cls(**kwargs)

        instance.config = ActionExpertConfig(
            action_dim=instance.action_dim,
            dynamic_dim=getattr(instance, 'dynamic_dim', 4096),
            hidden_dim=instance.hidden_dim,
            num_layers=len(instance.dit_blocks) if hasattr(instance, 'dit_blocks') else 6,
            num_heads=instance.dit_blocks[0].num_heads if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 16,
            mlp_ratio=instance.dit_blocks[0].mlp_ratio if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 4.0,
            drop_rate=instance.dit_blocks[0].drop_rate if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 0.1,
            time_horizon=instance.time_horizon,
            use_reward_conditioning=instance.use_reward_conditioning,
            use_rotary_emb=instance.use_rotary_emb,
            expert_type=ExpertType.FEATURE_BASED,
        )

        # Load weights
        missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logger.info(f"Successfully loaded model from {model_path}")
        return instance

    def save_pretrained(self, save_directory: str, **kwargs):
 
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save config
        config_dict = {
            "action_dim": self.action_dim,
            "dynamic_dim": self.dynamic_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": len(self.dit_blocks),
            "num_heads": self.dit_blocks[0].num_heads if self.dit_blocks else 16,
            "mlp_ratio": self.dit_blocks[0].mlp_ratio if self.dit_blocks else 4.0,
            "drop_rate": self.dit_blocks[0].drop_rate if self.dit_blocks else 0.1,
            "time_horizon": self.time_horizon,
            "use_reward_conditioning": self.use_reward_conditioning,
            "use_rotary_emb": self.use_rotary_emb,
            "feature_projection_dim": getattr(self, 'feature_projector', None) and
                                   self.feature_projector[0].in_features if hasattr(self, 'feature_projector') else None,
        }

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


