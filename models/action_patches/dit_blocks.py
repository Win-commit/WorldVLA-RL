import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for flow matching"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    """
    DiT (Diffusion Transformer) Block for action expert
    Based on "Scalable Diffusion Models with Transformers"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attention_type: str = "self",  # "self" (only self-attn), "cross" (only cross-attn), or "both" (self + cross)
        cross_attention_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.head_dim = hidden_size // num_heads

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if attention_type in ["cross", "both"] and cross_attention_dim is not None:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention components (split into q, k, v, out)
        self.attn_query = nn.Linear(hidden_size, hidden_size)
        self.attn_key = nn.Linear(hidden_size, hidden_size)
        self.attn_value = nn.Linear(hidden_size, hidden_size)
        self.attn_output = nn.Linear(hidden_size, hidden_size)

        # Cross-attention (if needed)
        if attention_type in ["cross", "both"] and cross_attention_dim is not None:
            self.cross_attn_query = nn.Linear(hidden_size, hidden_size)
            self.cross_attn_key = nn.Linear(cross_attention_dim, hidden_size)
            self.cross_attn_value = nn.Linear(cross_attention_dim, hidden_size)
            self.cross_attn_output = nn.Linear(hidden_size, hidden_size)

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(drop_rate)
        )

        # Time embedding layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, seq_len, hidden_size]
            timestep: Time embedding [B, hidden_size]
            encoder_hidden_states: Cross-attention context [B, ctx_len, cross_attention_dim]
            attention_mask: Self-attention mask [B, seq_len, seq_len]
            cross_attention_mask: Cross-attention mask [B, seq_len, ctx_len]
            rotary_emb: Tuple of (cos, sin) for rotary position embedding [B, seq_len, dim] or [1, seq_len, dim]
        """

        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(timestep).chunk(6, dim=-1)

        # Self-attention (only for "self" and "both")
        if self.attention_type in ["self", "both"]:
            x_norm = self.norm1(x)
            x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

            # Prepare query/key/value for attention
            query = self.attn_query(x_norm)
            key = self.attn_key(x_norm)
            value = self.attn_value(x_norm)

            # Apply rotary position embedding if provided
            if rotary_emb is not None:
                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

            # Apply self-attention
            batch_size, seq_len, _ = query.shape
            query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            attn_output = self.attn_output(attn_output)

            x = x + gate_msa.unsqueeze(1) * attn_output

        # Cross-attention (only for "cross" and "both")
        if self.attention_type in ["cross", "both"] and encoder_hidden_states is not None:
            x_norm = self.norm_cross(x)
            cross_query = self.cross_attn_query(x_norm)
            cross_key = self.cross_attn_key(encoder_hidden_states)
            cross_value = self.cross_attn_value(encoder_hidden_states)

            batch_size, ctx_len, _ = cross_query.shape
            cross_query = cross_query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            cross_key = cross_key.view(batch_size, ctx_len, self.num_heads, -1).transpose(1, 2)
            cross_value = cross_value.view(batch_size, ctx_len, self.num_heads, -1).transpose(1, 2)

            cross_attn_output = F.scaled_dot_product_attention(
                cross_query, cross_key, cross_value, attn_mask=cross_attention_mask, dropout_p=0.0
            )
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            cross_attn_output = self.cross_attn_output(cross_attn_output)

            x = x + cross_attn_output

        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        mlp_output = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class ActionRotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding for actions
    Based on Genie-Envisioner implementation
    """

    def __init__(
        self,
        dim: int,
        base_seq_length: int = 100,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        dim = hidden_states.size(-1)  

        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)
        grid = grid / self.base_seq_length
        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, :dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, :dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        # Expand for batch dimension
        cos_freqs = cos_freqs.expand(batch_size, -1, -1)
        sin_freqs = sin_freqs.expand(batch_size, -1, -1)

        return cos_freqs, sin_freqs


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Apply rotary embeddings to query and key tensors

    Args:
        x: Query or key tensor [B, seq_len, hidden_dim]
        freqs: Tuple of (cos, sin) frequency tensors [B, seq_len, hidden_dim] or [1, seq_len, hidden_dim]

    Returns:
        Rotated tensor with same shape as input
    """
    cos, sin = freqs
    batch_size = x.shape[0]

    # Broadcast cos/sin to match batch size if needed
    if cos.shape[0] == 1 and batch_size > 1:
        cos = cos.repeat(batch_size, 1, 1)
    if sin.shape[0] == 1 and batch_size > 1:
        sin = sin.repeat(batch_size, 1, 1)

    # Ensure dtype is float32 for computation
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D//2]
    cos = cos.unflatten(2, (-1, 2))
    sin = sin.unflatten(2, (-1, 2))

    # Apply rotation: [x_real * cos - x_imag * sin, x_real * sin + x_imag * cos]
    x_rotated = torch.stack([
        x_real * cos - x_imag * sin,
        x_real * sin + x_imag * cos
    ], dim=-1).flatten(3)

    return x_rotated.type_as(x)