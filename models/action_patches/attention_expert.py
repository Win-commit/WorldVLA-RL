import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .dit_blocks import DiTBlock, SinusoidalPositionEmbeddings, ActionRotaryPosEmbed


class CrossAttentionActionExpert(nn.Module):
    """
    Cross-Attention Action Expert using Flow Matching with DiT blocks

    This approach uses cross-attention to directly attend to the dynamic model's hidden states,
    allowing for more flexible information extraction compared to feature-based approach.
    """

    def __init__(
        self,
        action_dim: int = 7,
        visual_dim: int = 4096,
        hidden_dim: int = 2048,
        cross_attention_dim: int = 4096,  #matches dynamic model's hidden dim
        num_layers: int = 12,  # More layers for cross-attention
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
        reward_dim: int = 14,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.use_reward_conditioning = use_reward_conditioning
        self.use_rotary_emb = use_rotary_emb

        # Time embeddings for flow matching
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Action embedding
        self.action_proj_in = nn.Linear(action_dim, hidden_dim)
        self.action_proj_out = nn.Linear(hidden_dim, action_dim)

        # Learnable action tokens (like learnable queries)
        self.learnable_action_tokens = nn.Parameter(torch.randn(1, time_horizon, hidden_dim))

        # Rotary position embeddings
        if use_rotary_emb:
            rope_dim = rope_dim or hidden_dim
            self.action_rope = ActionRotaryPosEmbed(
                dim=rope_dim,
                base_seq_length=base_seq_length,
                theta=10000.0,
            )

        # Reward conditioning (if enabled)
        conditioning_dim = visual_dim
        if use_reward_conditioning:
            conditioning_dim += reward_dim
            self.reward_proj = nn.Linear(reward_dim, reward_dim)  # Project to reward_dim, not hidden_dim

        # Visual conditioning projection
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

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

        # Context projection for dynamic model features
        self.context_proj = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )


        # Initialize weights
        self._init_weights()

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
        dynamic_context: Optional[torch.Tensor] = None,
        attention_positions: Optional[List[int]] = None,
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
                reward_features=reward_features,
                attention_positions=attention_positions
            )

        # Extract conditioning features
        conditioning_features = self.extract_conditioning_features(
            visual_features=visual_features,
            reward_features=reward_features
        )

        # Time embeddings
        time_emb = self.time_embed(timesteps)
        time_emb = self.time_mlp(time_emb)  # [B, hidden_dim]

        # Action embeddings with learnable tokens
        action_emb = self.action_proj_in(actions)  # [B, seq_len, hidden_dim]

        # Add learnable action tokens
        learnable_tokens = self.learnable_action_tokens.expand(batch_size, -1, -1)
        if seq_len == self.time_horizon:
            action_emb = action_emb + learnable_tokens
        else:
            action_emb = action_emb + learnable_tokens[:, :seq_len, :]

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
                timestep=time_emb,
                encoder_hidden_states=dynamic_context,
                rotary_emb=rotary_emb
            )

        # Final norm and projection
        hidden_states = self.final_norm(hidden_states)
        flow_prediction = self.action_proj_out(hidden_states)

        return flow_prediction

    def compute_flow_loss(
        self,
        dynamic_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor],
        target_actions: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        attention_positions: Optional[List[int]] = None,
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

        # Sample random timesteps
        timesteps = torch.rand(batch_size, device=target_actions.device)

        # Sample noise (Gaussian)
        noise = torch.randn_like(target_actions)

        sigma = timesteps  # sigma ∈ [0, 1]

        # x(t) = (1-t) * x_data + t * noise
        noisy_actions = (1.0 - sigma.view(-1, 1, 1)) * target_actions + sigma.view(-1, 1, 1) * noise

        # Compute target flow (velocity field)
        # v = dx/dt = noise - actions
        target_flow = noise - target_actions

        # Predict flow
        predicted_flow = self.forward(
            actions=noisy_actions,
            timesteps=timesteps,
            dynamic_hidden_states=dynamic_hidden_states,
            visual_features=visual_features,
            reward_features=reward_features,
            attention_positions=attention_positions
        )

        # Compute MSE loss with SD3-style weighting
        flow_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')

        # Apply SD3-style loss weighting
        # Weighting scheme: uniform timesteps, weight by 1/sigma^2 for numerical stability
        loss_weights = 1.0 / (sigma.view(-1, 1, 1) ** 2 + 1e-8)
        flow_loss = flow_loss * loss_weights

        # Apply sequence length masking if provided
        if seq_lengths is not None:
            mask = torch.arange(seq_len, device=target_actions.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(flow_loss)
            flow_loss = flow_loss * mask
            flow_loss = flow_loss.sum() / mask.sum()
        else:
            flow_loss = flow_loss.mean()

        # Additional metrics
        with torch.no_grad():
            # Flow magnitude
            flow_magnitude = predicted_flow.norm(dim=-1).mean()

            # Cross-attention analysis (for debugging)
            if hasattr(self, '_analyze_attention'):
                attention_weights = self._analyze_attention(
                    hidden_states=hidden_states,
                    context=dynamic_context
                )
                attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            else:
                attention_entropy = torch.tensor(0.0, device=target_actions.device)

            # Action prediction quality
            action_error = torch.tensor(0.0, device=target_actions.device)

        return {
            'loss': flow_loss,
            'flow_loss': flow_loss,
            'flow_magnitude': flow_magnitude,
            'attention_entropy': attention_entropy,
            'action_error': action_error,
            'sigma_mean': sigma.mean(),
            'timestep_mean': timesteps.mean(),
            'target_flow_norm': target_flow.norm(dim=-1).mean(),
            'predicted_flow_norm': predicted_flow.norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        dynamic_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        reward_features: Optional[torch.Tensor] = None,
        attention_positions: Optional[List[int]] = None,
        num_steps: int = 20,
        solver: str = "euler",
        stochastic: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample actions using Flow Matching ODE solver

        Args:
            dynamic_hidden_states: Hidden states from dynamic model [B, dyn_seq_len, dyn_hidden_dim]
            visual_features: Visual features from dynamic model [B, visual_dim]
            reward_features: Reward features from dynamic model [B, reward_dim]
            attention_positions: Positions to attend to in dynamic model
            num_steps: Number of discretization steps
            solver: ODE solver type ('euler' or 'heun')
            stochastic: Whether to use stochastic sampling (adds noise back)
            temperature: Temperature for sampling

        Returns:
            Sampled actions [B, seq_len, action_dim]
        """
        batch_size = visual_features.size(0)
        device = visual_features.device

        # Pre-compute dynamic context for efficiency
        dynamic_context = self.extract_dynamic_context(
            dynamic_hidden_states=dynamic_hidden_states,
            visual_features=visual_features,
            reward_features=reward_features,
            attention_positions=attention_positions
        )

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
                dynamic_hidden_states=dynamic_hidden_states,
                visual_features=visual_features,
                reward_features=reward_features,
                dynamic_context=dynamic_context,
                attention_positions=attention_positions
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
    # Test the cross-attention action expert
    batch_size = 4
    seq_len = 10
    action_dim = 7
    visual_dim = 4096
    reward_dim = 14
    dyn_seq_len = 100
    dyn_hidden_dim = 4096

    # Initialize model
    model = CrossAttentionActionExpert(
        action_dim=action_dim,
        visual_dim=visual_dim,
        hidden_dim=2048,
        cross_attention_dim=dyn_hidden_dim,
        num_layers=12,
        num_heads=16,
        use_reward_conditioning=True,
        reward_dim=reward_dim
    )

    # Test data
    visual_features = torch.randn(batch_size, visual_dim)
    reward_features = torch.randn(batch_size, reward_dim)
    dynamic_hidden_states = torch.randn(batch_size, dyn_seq_len, dyn_hidden_dim)
    target_actions = torch.randn(batch_size, seq_len, action_dim)

    # Test forward pass
    model.train()

    # Compute loss
    loss_dict = model.compute_flow_loss(
        dynamic_hidden_states=dynamic_hidden_states,
        visual_features=visual_features,
        reward_features=reward_features,
        target_actions=target_actions
    )

    print("Cross-Attention Action Expert Test:")
    print(f"Loss: {loss_dict['loss'].item():.6f}")
    print(f"Flow magnitude: {loss_dict['flow_magnitude'].item():.6f}")

    # Test sampling
    sampled_actions = model.sample_actions(
        dynamic_hidden_states=dynamic_hidden_states,
        visual_features=visual_features,
        reward_features=reward_features,
        num_steps=10
    )

    print(f"Sampled actions shape: {sampled_actions.shape}")
    print("Test passed!")