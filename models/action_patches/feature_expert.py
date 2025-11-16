import os
import sys
# Add the parent directory to path for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dit_blocks import DiTBlock, SinusoidalPositionEmbeddings, ActionRotaryPosEmbed


class FeatureActionExpert(nn.Module):
    """
    Feature-based Action Expert using Flow Matching with DiT blocks

    This approach extracts features from the dynamic model and uses them as conditioning
    for the flow matching process to generate actions.
    500M
    """

    def __init__(
        self,
        action_dim: int = 7,
        dynamic_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        time_horizon: int = 10,
        # Feature projection parameters
        feature_projection_dim: Optional[int] = None,
        use_reward_conditioning: bool = True,
        # RoPE parameters
        use_rotary_emb: bool = True,
        rope_dim: Optional[int] = None,
        base_seq_length: int = 10,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.visual_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.use_reward_conditioning = use_reward_conditioning
        self.use_rotary_emb = use_rotary_emb

        # Feature projection dimensions
        if feature_projection_dim is None:
            feature_projection_dim = hidden_dim

        # Time embeddings for flow matching
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

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

        # Reward aggregator: adaptive aggregation of reward tokens [B, S+1, H] -> [B, H]
        if use_reward_conditioning:
            self.reward_aggregator = nn.Sequential(
                nn.Linear(dynamic_dim, dynamic_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dynamic_dim, dynamic_dim)
            )
            # Attention weighting for time steps
            self.reward_attention = nn.Sequential(
                nn.Linear(dynamic_dim, 256),
                nn.GELU(approximate="tanh"),
                nn.Linear(256, 1),
            )

        # Action embedding
        self.action_proj_in = nn.Linear(action_dim, hidden_dim)
        self.action_proj_out = nn.Linear(hidden_dim, action_dim)

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

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)


        # Initialize weights
        self._init_weights()

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
            reward_features: Reward features from dynamic model [B, reward_dim] or [B, S+1, visual_dim]
                              Can be either pre-aggregated [B, H] or time sequence [B, S+1, H]
                              For time sequences, uses attention weighting over time dimension

        Returns:
            Projected features [B, hidden_dim]
        """
        features = visual_features

        if self.use_reward_conditioning and reward_features is not None:
            # Check if reward_features is a time sequence [B, S, H] or aggregated [B, H]
            if reward_features.dim() == 3:
                # Time sequence: apply adaptive aggregation [B, S, H] -> [B, H]
                reward_agg = self.reward_aggregator(reward_features)  # [B, S, H]
                attention_scores = self.reward_attention(reward_features)  # [B, S, 1]
                attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1, dtype=reward_agg.dtype)  # [B, S]
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
        """
        Forward pass for action expert (predicts flow)

        Args:
            actions: Noisy actions [B, seq_len, action_dim]
            timesteps: Timesteps for flow matching [B,]
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim] or [B, S+1, visual_dim]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Predicted flow [B, seq_len, action_dim]
        """
        _, seq_len = actions.size(0), actions.size(1)

        # Extract features from dynamic model
        conditioning_features = self.extract_features(visual_features, reward_features)
        conditioning_features = conditioning_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Time embeddings
        time_emb = self.time_embed(timesteps)
        time_emb = self.time_mlp(time_emb)  # [B, hidden_dim]

        # Action embeddings
        action_emb = self.action_proj_in(actions)  # [B, seq_len, hidden_dim]

        # Combine action embeddings with conditioning
        conditioned_action_emb = action_emb + conditioning_features

        # Generate rotary embeddings for action sequence
        rotary_emb = None
        if self.use_rotary_emb:
            rotary_emb = self.action_rope(conditioned_action_emb, seq_len)

        # Apply DiT blocks
        hidden_states = conditioned_action_emb
        for dit_block in self.dit_blocks:
            hidden_states = dit_block(
                x=hidden_states,
                timestep=time_emb,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb
            )

        # Final norm and projection
        hidden_states = self.final_norm(hidden_states)
        flow_prediction = self.action_proj_out(hidden_states)

        return flow_prediction

    def compute_flow_loss(
        self,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor],
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss

        Args:
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim] or [B, S+1, visual_dim]
            target_actions: Target actions [B, seq_len, action_dim]
            seq_lengths: Actual sequence lengths [B,]

        Returns:
            Dictionary with loss and metrics
        """
        batch_size, seq_len, action_dim = target_actions.shape

        # Sample random timesteps
        timesteps = torch.rand(batch_size, device=target_actions.device)

        # Sample noise (Gaussian)
        noise = torch.randn_like(target_actions)

        sigma = timesteps  # sigma ∈ [0, 1]

        # x(t) = (1-t) * x_1 + t * noise
        noisy_actions = (1.0 - sigma.view(-1, 1, 1)) * target_actions + sigma.view(-1, 1, 1) * noise

        # Compute target flow (velocity field)
        # v = dx/dt = noise - actions
        target_flow = noise - target_actions

        # Predict flow
        predicted_flow = self.forward(
            actions=noisy_actions,
            timesteps=timesteps,
            visual_features=visual_features,
            reward_features=reward_features,
        )

        # Compute MSE loss with SD3-style weighting
        flow_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')

        # Apply SD3-style loss weighting
        # Weighting scheme: uniform timesteps, weight by 1/sigma^2 for numerical stability
        loss_weights = 1.0 / (sigma.view(-1, 1, 1) ** 2 + 1e-8)
        flow_loss = flow_loss * loss_weights

        flow_loss = flow_loss.mean()

        # Additional metrics
        with torch.no_grad():
            # Flow magnitude
            flow_magnitude = predicted_flow.norm(dim=-1).mean()

        return {
            'loss': flow_loss,
            'flow_loss': flow_loss,
            'target_flow_norm': target_flow.norm(dim=-1).mean(),
            'predicted_flow_norm': flow_magnitude,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        solver: str = "euler",
        stochastic: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample actions using Flow Matching ODE solver

        Args:
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim] or [B, S+1, visual_dim]
            num_steps: Number of discretization steps
            solver: ODE solver type ('euler' or 'heun')
            stochastic: Whether to use stochastic sampling (adds noise back)
            temperature: Temperature for sampling

        Returns:
            Sampled actions [B, seq_len, action_dim]
        """
        batch_size = visual_features.size(0)
        device = visual_features.device

        # Initialize with pure noise (sigma=1.0)
        actions = torch.randn(batch_size, self.time_horizon, self.action_dim, device=device)

        # Discretize sigma from 1.0 -> 0.0
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            current_sigma = sigmas[i].expand(batch_size)
            next_sigma = sigmas[i + 1].expand(batch_size)
            sigma_diff = next_sigma - current_sigma  # negative value

            # Predict flow at current sigma
            flow = self.forward(
                actions=actions,
                timesteps=current_sigma,
                visual_features=visual_features,
                reward_features=reward_features,
            )

            if stochastic:
                # Genie-style stochastic sampling
                # 1. Predict clean data x0
                x0 = actions - current_sigma.view(-1, 1, 1) * flow
                # 2. Re-add noise for next step
                noise = torch.randn_like(actions)
                actions = (1.0 - next_sigma.view(-1, 1, 1)) * x0 + next_sigma.view(-1, 1, 1) * noise
            else:
                # Deterministic Euler step
                # dx/dσ = flow, so x_{i+1} = x_i + (σ_{i+1} - σ_i) * flow
                actions = actions + sigma_diff.view(-1, 1, 1) * flow * temperature

        return actions

    def get_trainable_parameters(self):
        """Get parameters that should be trained (excluding frozen components)"""
        return [p for n, p in self.named_parameters() if p.requires_grad]




if __name__ == "__main__":
    # Test the feature-based action expert
    '''
    visual feature 以及language feature怎么从dynamic model拿出来
    '''
    batch_size = 4
    seq_len = 10
    action_dim = 7
    visual_dim = 4096
    reward_dim = 4096
    reward_seq_len = 11  # S+1 where S=10

    # Initialize model
    model = FeatureActionExpert(
        action_dim=action_dim,
        visual_dim=visual_dim,
        hidden_dim=2048,
        num_layers=6,
        num_heads=16,
        use_reward_conditioning=True
    )

    # Test data
    visual_features = torch.randn(batch_size, visual_dim)
    reward_features = torch.randn(batch_size, reward_dim)
    reward_features_3d = torch.randn(batch_size, reward_seq_len, visual_dim)  # 3D version
    target_actions = torch.randn(batch_size, seq_len, action_dim)

    # Test forward pass
    model.train()

    # Compute loss with 2D reward features
    loss_dict_2d = model.compute_flow_loss(
        visual_features=visual_features,
        reward_features=reward_features,
        target_actions=target_actions
    )

    print("Feature-based Action Expert Test (2D reward features):")
    print(f"Loss: {loss_dict_2d['loss'].item():.6f}")

    # Compute loss with 3D reward features (new feature)
    loss_dict_3d = model.compute_flow_loss(
        visual_features=visual_features,
        reward_features=reward_features_3d,
        target_actions=target_actions
    )

    print("Feature-based Action Expert Test (3D reward features):")
    print(f"Loss: {loss_dict_3d['loss'].item():.6f}")

    # Test sampling (deterministic)
    sampled_actions = model.sample_actions(
        visual_features=visual_features,
        reward_features=reward_features,
        num_steps=10,
        stochastic=False
    )

    print(f"Sampled actions shape: {sampled_actions.shape}")
    print(f"Deterministic sampling mean: {sampled_actions.mean().item():.6f}")

    # Test sampling (stochastic)
    sampled_actions_stoch = model.sample_actions(
        visual_features=visual_features,
        reward_features=reward_features,
        num_steps=10,
        stochastic=True
    )

    print(f"Stochastic sampling mean: {sampled_actions_stoch.mean().item():.6f}")
    print(f"2D Flow loss mean: {loss_dict_2d['flow_loss'].item():.6f}")
    print(f"3D Flow loss mean: {loss_dict_3d['flow_loss'].item():.6f}")
    print(f"Target flow norm: {loss_dict_2d['target_flow_norm'].item():.6f}")
    print(f"Predicted flow norm: {loss_dict_2d['predicted_flow_norm'].item():.6f}")
    print("Test passed!")