import kornia.augmentation as K
import torch
import torch.nn as nn
from typing import Tuple


class SimCLRTransform(nn.Module):
    """
    SimCLR augmentation pipeline that generates two augmented views of the same image.
    Uses Kornia for GPU-accelerated transformations.
    """

    def __init__(self, input_size: int = 96, s: float = 1.0) -> None:
        """
        Initializes the augmentation pipeline.

        Args:
            input_size (int): Size of tje input image (default 96 for STL-10).
            s (float): Strength of color jittering (default 1.0).
        """
        super().__init__()
        kernel_size = int(0.1 * input_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.transform = K.AugmentationSequential(
            K.RandomResizedCrop(size=(input_size, input_size)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.8 * s,
                contrast=0.8 * s,
                saturation=0.8 * s,
                hue=0.2 * s,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(
                kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2.0), p=0.5
            ),
            K.Normalize(
                mean=torch.tensor([0.4467, 0.4398, 0.4066]),
                std=torch.tensor([0.2603, 0.2566, 0.2713]),
            ),
            data_keys=["input"],
            same_on_batch=False,
        )

    def forward(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the transformations to the input batch and returns two augmented views.

        Args:
            batch_x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the input batch.
        """
        view1 = self.transform(batch_x)
        view2 = self.transform(batch_x)
        return view1, view2
