import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        # 将输入和条件拼接
        current_dim = input_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.mu = nn.Linear(current_dim, latent_dim)
        self.log_var = nn.Linear(current_dim, latent_dim)
        
    def forward(self, x, c):
        # 拼接输入和条件
        x_cond = torch.cat([x, c], dim=-1)
        features = self.network(x_cond)
        mu = self.mu(features)
        log_var = self.log_var(features)
        return mu, log_var

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        # 将潜变量和条件拼接
        current_dim = latent_dim + condition_dim
        
        for hidden_dim in hidden_dims[::-1]:  # 反向，与编码器对称
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, z, c):
        # 拼接潜变量和条件
        z_cond = torch.cat([z, c], dim=-1)
        reconstruction = self.network(z_cond)
        return reconstruction

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = ConditionalEncoder(input_dim, condition_dim, hidden_dims, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, hidden_dims, input_dim)
        
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z, c)
        return reconstruction, mu, log_var


def vae_loss(recon_x, x, mu, log_var, recon_weight=1.0, kl_weight=1.0):
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
    # 1. 重建损失 (Reconstruction Loss)
    # 对于实值数据使用MSE，对于二值数据使用BCE
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # 或者使用BCE:
    # recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    
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

def conditional_vae_loss(model, x, c, beta=1.0):
    """
    针对条件VAE的专用损失函数
    
    Args:
        model: 条件VAE模型
        x: 输入数据
        c: 条件信息
        beta: β-VAE参数，控制KL散度的权重
    """
    recon_x, mu, log_var = model(x, c)
    
    # 重建损失
    recon_loss = F.mse_loss(recon_x, x)
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # β-VAE损失
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }

class ConditionalLossCalculator:
    """针对不同数据类型的条件自编码器损失计算器"""
    
    @staticmethod
    def continuous_loss(recon_x, x, mu, log_var, beta=1.0):
        """连续数据的损失（如图像像素值）"""
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def binary_loss(recon_x, x, mu, log_var, beta=1.0):
        """二值数据的损失"""
        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def multimodal_loss(recon_x, x, mu, log_var, conditions, alpha=0.1, beta=1.0):
        """
        多模态数据的损失，包含条件一致性约束
        
        Args:
            alpha: 条件一致性损失的权重
        """
        # 基础重建损失
        recon_loss = F.mse_loss(recon_x, x)
        
        # KL散度
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # 条件一致性损失：确保相同条件产生相似的潜变量
        condition_consistency_loss = 0
        unique_conditions = torch.unique(conditions, dim=0)
        for cond in unique_conditions:
            cond_mask = (conditions == cond.unsqueeze(0)).all(dim=1)
            if cond_mask.sum() > 1:
                cond_mus = mu[cond_mask]
                # 最小化相同条件下的潜变量方差
                condition_consistency_loss += torch.var(cond_mus, dim=0).mean()
        
        total_loss = recon_loss + beta * kl_loss + alpha * condition_consistency_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'condition_consistency_loss': condition_consistency_loss
        }

def train_conditional_vae(model, dataloader, optimizer, device, num_epochs=100):
    """训练条件VAE的完整流程"""
    model.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        for batch_idx, (x, conditions) in enumerate(dataloader):
            x = x.to(device)
            conditions = conditions.to(device)
            
            # 前向传播
            recon_x, mu, log_var = model(x, conditions)
            
            # 计算损失
            loss_dict = conditional_vae_loss(model, x, conditions, beta=0.1)
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += loss_dict['recon_loss'].item()
            epoch_losses['kl'] += loss_dict['kl_loss'].item()
            
        # 打印epoch统计信息
        avg_total = epoch_losses['total'] / len(dataloader)
        avg_recon = epoch_losses['recon'] / len(dataloader)
        avg_kl = epoch_losses['kl'] / len(dataloader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Total Loss: {avg_total:.4f}, '
              f'Recon Loss: {avg_recon:.4f}, '
              f'KL Loss: {avg_kl:.4f}')
        
        loss_history.append({
            'total': avg_total,
            'recon': avg_recon, 
            'kl': avg_kl
        })
    
    return loss_history

# 模型参数
input_dim = 784  # MNIST图像
condition_dim = 10  # 10个类别的one-hot编码
hidden_dims = [512, 256]
latent_dim = 20

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConditionalVAE(input_dim, condition_dim, hidden_dims, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 假设我们有一个数据加载器
# dataloader返回 (images, conditions) 对
# loss_history = train_conditional_vae(model, dataloader, optimizer, device)

def generate_new_samples(model, c, num_samples=1, use_mean=False):
    """生成新样本"""
    model.eval()
    with torch.no_grad():
        if use_mean:
            # 使用零向量作为均值
            z = torch.zeros(num_samples, model.latent_dim)
        else:
            # 从标准正态分布采样
            z = torch.randn(num_samples, model.latent_dim)
        
        # 解码生成新样本
        generated = model.decoder(z, c)
        return generated

# 使用示例：生成数字"7"的新样本
digit_7_condition = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])  # 数字7的one-hot
new_samples = generate_new_samples(model, digit_7_condition, num_samples=5)

