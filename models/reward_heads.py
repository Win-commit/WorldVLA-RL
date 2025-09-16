from torch import nn
import torch


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class RwdHead(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        Rwd_dim=14,
    ):
        super().__init__()
        self.action_dim = Rwd_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=Rwd_dim
        )

    def forward(self, rwd_hidden_states):
        #rwd_hidden_states: [1,hidden_dim]
        return self.model(rwd_hidden_states)

class RtgHead(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        Rtg_dim=14,
    ):
        super().__init__()
        self.action_dim = Rtg_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=Rtg_dim
        )

    def forward(self, rtg_hidden_states):
        #rtg_hidden_states: [1,hidden_dim]
        return self.model(rtg_hidden_states)
        