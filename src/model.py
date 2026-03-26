import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Tuple


class SimCLR(nn.Module):
    """
    SimCLR architecture, consisting of a base encoder (ResNet50) and a Projection Head (MLP).
    """

    def __init__(self, out_dim: int = 128) -> None:
        """
        Init of the model.

        Args:
            out_dim (int): Dimension of the output vector z.
        """
        super().__init__()

        # Base Encoder: f(x)
        resnet = resnet50(weights=None)
        features_dim: int = resnet.fc.in_features

        # Removing the last classifying layer from ResNet
        resnet.fc = nn.Identity()  # type: ignore
        self.encoder = resnet

        # Projection Head: g(h) -> z
        # 2-Layers MLP (Linear -> ReLU -> Linear)
        self.projector = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, returns both the h embedding and the z projection.
        For training z is needed, for downstream tasks (e.g. linear evaluation) h is needed.

        Args:
            x (torch.Tensor): Input tensor of pictures, [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - h: encoder features [B, 2048].
                - z: Projection Head output [B, 128].
        """
        h: torch.Tensor = self.encoder(x)
        z: torch.Tensor = self.projector(h)
        return h, z
