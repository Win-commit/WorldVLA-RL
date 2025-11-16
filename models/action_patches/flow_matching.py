import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class FlowMatchingTrainer(nn.Module):
    """
    Unified trainer for Flow Matching based action experts
    Supports both feature-based and cross-attention action experts
    """

    def __init__(
        self,
        action_expert: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip_val: float = 1.0,
        accumulation_steps: int = 1,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.action_expert = action_expert
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        self.use_ema = use_ema
        self.device = device

        # EMA model
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.training_losses = []

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    def _create_ema_model(self):
        """Create EMA version of the action expert"""
        ema_model = type(self.action_expert)(
            **self.action_expert.__dict__
        )
        ema_model.load_state_dict(self.action_expert.state_dict())
        ema_model.eval()
        return ema_model

    def update_ema(self):
        """Update EMA model parameters"""
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.action_expert.parameters()):
                ema_param.data.copy_(
                    self.ema_decay * ema_param.data + (1.0 - self.ema_decay) * param.data
                )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Single training step

        Args:
            batch: Batch containing necessary inputs
            return_dict: Whether to return detailed metrics

        Returns:
            Loss tensor or dictionary with metrics
        """
        self.action_expert.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass and compute loss
        if hasattr(self.action_expert, 'compute_flow_loss'):
            loss_dict = self.action_expert.compute_flow_loss(**batch)
            loss = loss_dict['loss']
        else:
            raise ValueError("Action expert must have compute_flow_loss method")

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Update weights if accumulation step is reached
        if (self.global_step + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.action_expert.parameters(), self.gradient_clip_val)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update EMA
            self.update_ema()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Update step counter
        self.global_step += 1

        # Log metrics
        self.training_losses.append(loss.item())
        if self.global_step % 100 == 0:
            avg_loss = np.mean(self.training_losses[-100:])
            print(f"Step {self.global_step}: Loss = {avg_loss:.6f}")

        if return_dict:
            loss_dict['step'] = self.global_step
            loss_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            return loss_dict
        else:
            return loss

    def validate_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Validation step

        Args:
            batch: Batch containing necessary inputs
            return_dict: Whether to return detailed metrics

        Returns:
            Loss tensor or dictionary with metrics
        """
        model = self.ema_model if self.use_ema else self.action_expert
        model.eval()

        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass and compute loss
            if hasattr(model, 'compute_flow_loss'):
                loss_dict = model.compute_flow_loss(**batch)
                loss = loss_dict['loss']
            else:
                raise ValueError("Action expert must have compute_flow_loss method")

        if return_dict:
            return loss_dict
        else:
            return loss

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            log_interval: Logging interval

        Returns:
            Dictionary with training and validation metrics
        """
        epoch_metrics = {'train_loss': 0.0, 'val_loss': 0.0}

        # Training phase
        self.action_expert.train()
        train_losses = []

        for step, batch in enumerate(train_loader):
            loss_dict = self.train_step(batch, return_dict=True)
            train_losses.append(loss_dict['loss'].item())

            if step % log_interval == 0:
                print(f"Epoch {self.current_epoch}, Step {step}: Loss = {loss_dict['loss'].item():.6f}")

        epoch_metrics['train_loss'] = np.mean(train_losses)

        # Validation phase
        if val_loader is not None:
            self.action_expert.eval()
            val_losses = []

            for step, batch in enumerate(val_loader):
                loss_dict = self.validate_step(batch, return_dict=True)
                val_losses.append(loss_dict['loss'].item())

            epoch_metrics['val_loss'] = np.mean(val_losses)

        self.current_epoch += 1
        return epoch_metrics

    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
    ):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.action_expert.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'training_losses': self.training_losses,
        }

        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if include_scheduler and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.action_expert.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.training_losses = checkpoint.get('training_losses', [])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        print(f"Checkpoint loaded from {filepath}")

    @torch.no_grad()
    def sample_actions(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 1,
        num_steps: int = 20,
        temperature: float = 1.0,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Sample actions from the trained model

        Args:
            batch: Batch containing conditioning inputs
            num_samples: Number of action sequences to sample
            num_steps: Number of ODE solver steps
            temperature: Sampling temperature
            use_ema: Whether to use EMA model

        Returns:
            Sampled actions [num_samples, B, seq_len, action_dim]
        """
        model = self.ema_model if use_ema and self.use_ema else self.action_expert
        model.eval()

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        sampled_actions = []

        for _ in range(num_samples):
            if hasattr(model, 'sample_actions'):
                actions = model.sample_actions(
                    **batch,
                    num_steps=num_steps,
                    temperature=temperature
                )
            else:
                raise ValueError("Action expert must have sample_actions method")

            sampled_actions.append(actions)

        return torch.stack(sampled_actions, dim=0)


class FlowMatchingLoss(nn.Module):
    """
    Various flow matching loss functions
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted_flow: torch.Tensor,
        target_flow: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute flow matching loss

        Args:
            predicted_flow: Predicted flow [..., action_dim]
            target_flow: Target flow [..., action_dim]
            mask: Optional mask for padding [..., 1]
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Loss tensor
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(predicted_flow, target_flow, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(predicted_flow, target_flow, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

        # Reduction
        if reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def create_flow_matching_trainer(
    action_expert: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    scheduler_type: str = "cosine",
    gradient_clip_val: float = 1.0,
    accumulation_steps: int = 1,
    ema_decay: float = 0.9999,
    use_ema: bool = True,
    device: str = "cuda",
) -> FlowMatchingTrainer:
    """
    Helper function to create a complete flow matching trainer

    Args:
        action_expert: Action expert model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        beta1: Adam beta1
        beta2: Adam beta2
        scheduler_type: Learning rate scheduler type
        gradient_clip_val: Gradient clipping value
        accumulation_steps: Gradient accumulation steps
        ema_decay: EMA decay rate
        use_ema: Whether to use EMA
        device: Device to use

    Returns:
        Configured FlowMatchingTrainer
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        action_expert.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # Create scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=learning_rate * 0.1
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.9
        )
    elif scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000
        )

    # Create trainer
    trainer = FlowMatchingTrainer(
        action_expert=action_expert,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_val=gradient_clip_val,
        accumulation_steps=accumulation_steps,
        ema_decay=ema_decay,
        use_ema=use_ema,
        device=device,
    )

    return trainer