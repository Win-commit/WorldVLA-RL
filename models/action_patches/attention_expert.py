import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .dit_blocks import DiTBlock, ActionRotaryPosEmbed
import os
import logging
from .utils import ActionExpertConfig, ExpertType
logger = logging.getLogger(__name__)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.normalization import AdaLayerNormSingle
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import argparse

class CrossAttentionActionExpert(nn.Module):
    """
    Cross-Attention Action Expert using Flow Matching with DiT blocks

    This approach uses cross-attention to directly attend to the dynamic model's hidden states,
    allowing for more flexible information extraction compared to feature-based approach.
    """

    def __init__(
        self,
        action_dim: int = 7,
        dynamic_dim: int = 4096,
        hidden_dim: int = 2048,
        cross_attention_dim: int = 4096,  #matches dynamic model's hidden dim
        num_layers: int = 6,  # More layers for cross-attention
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        time_horizon: int = 10,
        # Cross-attention specific parameters
        use_rotary_emb: bool = True,
        rope_dim: Optional[int] = None,
        base_seq_length: int = 100,
        # Reward conditioning
        use_reward_conditioning: bool = True,
        # Flow matching weighting scheme parameters
        weighting_scheme: str = "none",
        logit_mean: Optional[float] = None,
        logit_std: Optional[float] = None,
        mode_scale: Optional[float] = None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.visual_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.use_reward_conditioning = use_reward_conditioning
        self.use_rotary_emb = use_rotary_emb


        self.config = ActionExpertConfig(
            action_dim=action_dim,
            dynamic_dim=dynamic_dim,
            hidden_dim=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            time_horizon=time_horizon,
            use_reward_conditioning=use_reward_conditioning,
            use_rotary_emb=use_rotary_emb,
            expert_type=ExpertType.CROSS_ATTENTION,
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

        # Reward conditioning (if enabled) ???????????????????????????????????????????????????????????????????????????????????
        conditioning_dim = dynamic_dim
        if use_reward_conditioning:
            conditioning_dim += dynamic_dim
            self.reward_proj = nn.Linear(dynamic_dim, dynamic_dim)  # Project to reward_dim, not hidden_dim

        # Visual conditioning projection
        self.visual_proj = nn.Linear(dynamic_dim, hidden_dim)

        # Combined conditioning for cross-attention
        # Note: conditioning_proj input dimension should match conditioning_dim (visual_dim + reward_dim)
        self.conditioning_proj = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop_rate),
        )

        # DiT blocks with cross-attention
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attention_type="both",  # Both self and cross attention
                cross_attention_dim=cross_attention_dim
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.hidden_dim) / self.hidden_dim **0.5)

        self.action_proj_out = nn.Linear(hidden_dim, action_dim)


        # Context projection for dynamic model features
        self.context_proj = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )


        # Initialize weights
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
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, std=0.02)

    def extract_dynamic_context(
        self,
        dynamic_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract context from dynamic model's hidden states

        Args:
            dynamic_hidden_states: Hidden states from dynamic model [B, seq_len, hidden_dim]
            visual_features: Visual features [B, visual_dim]
            reward_features: Reward features [B, reward_dim]
            attention_positions: Positions to attend to in dynamic model

        Returns:
            Projected context [B, seq_len, cross_attention_dim]
        """

        # Project dynamic hidden states
        context = self.context_proj(dynamic_hidden_states)


        return context

    def extract_conditioning_features(
        self,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract conditioning features for cross-attention

        Args:
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim]

        Returns:
            Conditioning features [B, hidden_dim]
        """
        conditioning = [visual_features]

        if self.use_reward_conditioning and reward_features is not None:
            conditioning.append(self.reward_proj(reward_features))

        conditioning = torch.cat(conditioning, dim=-1)
        projected_conditioning = self.conditioning_proj(conditioning)

        return projected_conditioning

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        dynamic_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None,
        dynamic_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention action expert

        Args:
            actions: Noisy actions [B, seq_len, action_dim]
            timesteps: Timesteps for flow matching [B,]
            dynamic_hidden_states: Hidden states from dynamic model [B, dyn_seq_len, dyn_hidden_dim]
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim]
            dynamic_context: Pre-computed context from dynamic model [B, ctx_len, cross_attention_dim]
            attention_positions: Positions to attend to in dynamic model

        Returns:
            Predicted flow [B, seq_len, action_dim]
        """
        batch_size, seq_len = actions.size(0), actions.size(1)

        # Extract context from dynamic model
        if dynamic_context is None:
            dynamic_context = self.extract_dynamic_context(
                dynamic_hidden_states=dynamic_hidden_states,
                visual_features=visual_features,
                reward_features=reward_features
            )

        # Extract conditioning features
        conditioning_features = self.extract_conditioning_features(
            visual_features=visual_features,
            reward_features=reward_features
        )

        # Time embeddings
        action_temb, action_embedded_timestep = self.action_time_embed(
            timesteps.flatten(),
            batch_size=batch_size,
            hidden_dtype=conditioning_features.dtype,
        )

        # Action embeddings with learnable tokens
        action_emb = self.action_proj_in(actions)  # [B, seq_len, hidden_dim]

        # Add conditioning
        conditioning_broadcast = conditioning_features.unsqueeze(1).expand(-1, seq_len, -1)
        action_emb = action_emb + conditioning_broadcast

        # Generate rotary embeddings for action sequence
        rotary_emb = None
        if self.use_rotary_emb:
            # Use hidden_dim (which matches action_emb dimension) for RoPE
            rotary_emb = self.action_rope(action_emb, seq_len)

        # Apply DiT blocks with cross-attention
        hidden_states = action_emb
        for dit_block in self.dit_blocks:
            
            hidden_states = dit_block(
                x=hidden_states,
                temb=action_temb,
                encoder_hidden_states=dynamic_context,
                rotary_emb=rotary_emb
            )

        # Final norm and projection
        hidden_states = self.final_norm(hidden_states)

        action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
        action_shift, action_scale = action_scale_shift_values[:,:,0], action_scale_shift_values[:,:,1]
        hidden_states = hidden_states * (1 + action_scale) + action_shift

        flow_prediction = self.action_proj_out(hidden_states)

        return flow_prediction

    def compute_flow_loss(
        self,
        dynamic_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor],
        target_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss

        Args:
            dynamic_hidden_states: Hidden states from dynamic model [B, dyn_seq_len, dyn_hidden_dim]
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim]
            target_actions: Target actions [B, seq_len, action_dim]
            seq_lengths: Actual sequence lengths [B,]
            attention_positions: Positions to attend to in dynamic model

        Returns:
            Dictionary with loss and metrics
        """
        batch_size, seq_len, action_dim = target_actions.shape

        # Ensure all features have the same dtype as target_actions
        dynamic_hidden_states = dynamic_hidden_states.to(dtype=target_actions.dtype)
        visual_features = visual_features.to(dtype=target_actions.dtype)
        if reward_features is not None:
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
        noise = torch.randn_like(target_actions, dtype=target_actions.dtype)


        # x(t) = (1-t) * x_data + t * noise
        noisy_actions = (1.0 - action_ss) * target_actions + action_ss * noise

        # Compute target flow (velocity field)
        # v = dx/dt = noise - actions
        target_flow = noise - target_actions

        # Predict flow
        predicted_flow = self.forward(
            actions=noisy_actions,
            timesteps=action_timesteps,
            dynamic_hidden_states=dynamic_hidden_states,
            visual_features=visual_features,
            reward_features=reward_features
        )

        action_weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.flow_matching_args.weighting_scheme, 
            sigmas=action_sigmas
        ).reshape(-1, 1, 1).repeat(1, 1, action_dim)
        flow_loss = action_weights.float() * (predicted_flow.float() - target_flow.float()).pow(2)
        
        flow_loss = flow_loss.mean()

        # Additional metrics
        with torch.no_grad():
            # Flow magnitude
            flow_magnitude = predicted_flow.norm(dim=-1).mean()

        # Ensure loss is float32 for stable backward pass
        loss = flow_loss.float()

        return {
            'loss': loss,
            'flow_loss': flow_loss,
            'target_flow_norm': target_flow.norm(dim=-1).mean(),
            'predicted_flow_norm': flow_magnitude
        }


    # @torch.no_grad()
    # def sample_actions(
    #     self,
    #     dynamic_hidden_states: torch.Tensor,
    #     visual_features: torch.Tensor,
    #     reward_features: Optional[torch.Tensor] = None,
    #     attention_positions: Optional[List[int]] = None,
    #     num_steps: int = 20,
    #     solver: str = "euler",
    #     stochastic: bool = False,
    #     temperature: float = 1.0,
    # ) -> torch.Tensor:
    #     """
    #     Sample actions using Flow Matching ODE solver

    #     Args:
    #         dynamic_hidden_states: Hidden states from dynamic model [B, dyn_seq_len, dyn_hidden_dim]
    #         visual_features: Visual features from dynamic model [B, visual_dim]
    #         reward_features: Reward features from dynamic model [B, reward_dim]
    #         attention_positions: Positions to attend to in dynamic model
    #         num_steps: Number of discretization steps
    #         solver: ODE solver type ('euler' or 'heun')
    #         stochastic: Whether to use stochastic sampling (adds noise back)
    #         temperature: Temperature for sampling

    #     Returns:
    #         Sampled actions [B, seq_len, action_dim]
    #     """
    #     batch_size = visual_features.size(0)
    #     device = visual_features.device

    #     # Pre-compute dynamic context for efficiency
    #     dynamic_context = self.extract_dynamic_context(
    #         dynamic_hidden_states=dynamic_hidden_states,
    #         visual_features=visual_features,
    #         reward_features=reward_features,
    #         attention_positions=attention_positions
    #     )

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
    #             dynamic_hidden_states=dynamic_hidden_states,
    #             visual_features=visual_features,
    #             reward_features=reward_features,
    #             dynamic_context=dynamic_context,
    #             attention_positions=attention_positions
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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs (dict, optional): Additional keyword arguments passed to the
                checkpoint function. (default: None)
        """
        # For CrossAttentionActionExpert, we don't need to do anything special for gradient checkpointing
        # PyTorch's gradient checkpointing works automatically with any nn.Module
        pass

    def get_trainable_parameters(self):
        """Get parameters that should be trained (excluding frozen components)"""
        return [p for n, p in self.named_parameters() if p.requires_grad]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        map_location: str = "cpu",
        **kwargs
    ) -> "CrossAttentionActionExpert":
        """Load model from saved checkpoint

        Args:
            model_path: Path to model checkpoint or directory containing checkpoint
            map_location: Device to map tensors to
            **kwargs: Additional arguments to override config

        Returns:
            Loaded CrossAttentionActionExpert model
        """
        # Check if path is directory or file
        if os.path.isdir(model_path):
            # Look for pytorch_model.bin or model.safetensors
            import glob
            checkpoint_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
            checkpoint_files.extend(glob.glob(os.path.join(model_path, "model.safetensors")))
            if checkpoint_files:
                state_dict_path = checkpoint_files[0]
            else:
                # Try loading all files in directory
                state_dict_path = model_path
        else:
            state_dict_path = model_path

        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # Load state dict
        try:
            state_dict = torch.load(state_dict_path, map_location=map_location, weights_only=False)
        except Exception:
            # Fallback to weights_only=True
            state_dict = torch.load(state_dict_path, map_location=map_location, weights_only=True)

        # Try to load config from config.json if it exists
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
            dynamic_dim=instance.visual_dim,
            hidden_dim=instance.hidden_dim,
            cross_attention_dim=instance.cross_attention_dim,
            num_layers=len(instance.dit_blocks) if hasattr(instance, 'dit_blocks') else 6,
            num_heads=instance.dit_blocks[0].num_heads if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 16,
            mlp_ratio=instance.dit_blocks[0].mlp_ratio if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 4.0,
            drop_rate=instance.dit_blocks[0].drop_rate if hasattr(instance, 'dit_blocks') and instance.dit_blocks else 0.1,
            time_horizon=instance.time_horizon,
            use_reward_conditioning=instance.use_reward_conditioning,
            use_rotary_emb=instance.use_rotary_emb,
            expert_type=ExpertType.CROSS_ATTENTION,
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
        """Save model and configuration to a directory

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save config
        config_dict = {
            "action_dim": self.action_dim,
            "dynamic_dim": self.visual_dim,
            "hidden_dim": self.hidden_dim,
            "cross_attention_dim": self.cross_attention_dim,
            "num_layers": len(self.dit_blocks),
            "num_heads": self.dit_blocks[0].num_heads if self.dit_blocks else 16,
            "mlp_ratio": self.dit_blocks[0].mlp_ratio if self.dit_blocks else 4.0,
            "drop_rate": self.dit_blocks[0].drop_rate if self.dit_blocks else 0.1,
            "time_horizon": self.time_horizon,
            "use_reward_conditioning": self.use_reward_conditioning,
            "use_rotary_emb": self.use_rotary_emb,
        }

        config_path = os.path.join(save_directory, "config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {save_directory}")


