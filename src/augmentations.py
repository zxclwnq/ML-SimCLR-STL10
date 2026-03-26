import torchvision.transforms.v2 as T
import torch
from typing import Tuple


class SimCLRTransform(torch.nn.Module):
    """
    Generates 2 augmented views of 1 image (as described in the SimCLR paper).
    """

    def __init__(self, input_size: int = 96, s: float = 1.0) -> None:
        """
        Init of the augmentation pipeline for SimCLR.

        Args:
            input_size (int): Size of the image (for STL-10 it is 96).
            s (float): Color distortion strength. As in the paper, it is 1.0.
        """
        super().__init__()
        # Color distortion: brightness, contrast, saturation, hue
        color_jitter = T.ColorJitter(
            brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s
        )

        # Gaussian blur ~10%
        kernel_size: int = int(0.1 * input_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.transform = T.Compose(
            [
                T.RandomResizedCrop(size=input_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=kernel_size)], p=0.5),
                # Normalization for STL-10
                T.Normalize(
                    mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies augmentation to an image tensor.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing 2 augmented versions of the image.
        """
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
