from torch import nn
import torch
import torch.nn.functional as F

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


class ValueEncoder(nn.Module):
    def __init__(self, input_dim=14, condition_dim=4096, latent_dim=14):
        super().__init__()
        current_dim = input_dim + condition_dim  

        self.fc1 = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, 1024),
            nn.ReLU()
        )
        self.model = MLPResNet(
            num_blocks=2, 
            input_dim=1024, 
            hidden_dim=512,
            output_dim=256
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        
    def forward(self, rwd_hidden_states, rwd):
        # rwd_hidden_states(c): [batch, 4096], rwd: [batch, 14]
        x_cond = torch.cat([rwd_hidden_states, rwd], dim=-1)  # [batch, 4110]
        x = self.fc1(x_cond)        # [batch, 1024]
        features = self.model(x)     # [batch, 256]
        mu = self.mu(features)       # [batch, 14]
        log_var = self.log_var(features)  # [batch, 14]
        return mu, log_var


class ValueDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=14,
        condition_dim=4096,
        output_dim=14,
    ):
        super().__init__()
        current_dim = latent_dim + condition_dim  # 4110
        
        # 渐进式上采样（与 Encoder 对称）：4110 -> 1024 -> 512 -> 256 -> 14
        self.fc1 = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, 1024),
            nn.ReLU()
        )
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=1024,
            hidden_dim=512,
            output_dim=256
        )
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, rwd_hidden_states, latent_vectors):
        x = torch.cat([rwd_hidden_states, latent_vectors], dim=-1)  # [batch, 4110]
        x = self.fc1(x)                 # [batch, 1024]
        x = self.model(x)                 # [batch, 256]
        output = self.fc_out(x)           # [batch, 14]
        return output


def vae_loss(recon_x, x, mu, log_var, recon_weight=1.0, kl_weight=1.0, beta=0.1):
    """
    条件VAE的损失函数
    
    Args:
        recon_x: 重建的输出
        x: 原始输入
        mu: 潜变量的均值
        log_var: 潜变量的对数方差
        recon_weight: 重建损失的权重
        kl_weight: KL散度的权重
    """

    recon_loss = F.smooth_l1_loss(recon_x, x, reduction='sum', beta=beta)
    
    # 2. KL散度损失 (KL Divergence)
    # KL(q(z|x) || p(z))，其中p(z)是标准正态分布
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 3. 总损失
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }

def reparameterize(mu, log_var):
    """
    Reparameterization trick to sample from a Gaussian distribution.
    
    Args:
        mu: mean of the Gaussian distribution
        log_var: log of the variance of the Gaussian distribution
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std