from torch import nn
import torch

class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim

        self.arm_encoder = nn.Linear(6, self.llm_dim, bias=True)
        self.gripper_encoder = nn.Linear(2, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.state_projector = nn.Linear(self.llm_dim * 2, self.llm_dim, bias=True)

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, T, 8)
        B, T, _ = proprio.shape
        proprio = proprio.flatten(0,1)
        arm_state_feature = self.arm_encoder(proprio[:, :6])
        gripper_state_feature = self.gripper_encoder(proprio[:, 6:])
        projected_features  = self.act_fn1(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
        projected_features = self.state_projector(projected_features)
        projected_features = projected_features.view(B, T, -1)
        return projected_features