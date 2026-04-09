import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        """
        Initializes the loss function.

        Args:
            temperature (float): Temperature parameter tau, which controls the "strictness" of the distribution.
        """
        super().__init__()
        self.temperature: float = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes the Contrastive loss.

        Args:
            z_i (torch.Tensor): Projector output for the first augmentation view [B, out_dim].
            z_j (torch.Tensor): Projector output for the second augmentation view [B, out_dim].

        Returns:
            torch.Tensor: Calculated loss value.
        """
        batch_size: int = z_i.shape[0]

        # L2-normalize vectors (so the dot product equals cosine similarity)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate vectors into a single tensor [2N, out_dim]
        z = torch.cat([z_i, z_j], dim=0)

        similarity_matrix: torch.Tensor = torch.matmul(z, z.T) / self.temperature
        device: torch.device = similarity_matrix.device

        # Mask out diagonal elements (similarity of a vector with itself)
        mask: torch.Tensor = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        # Changing values on the diagonal to a very small number so their exp is ~ 0
        similarity_matrix.masked_fill_(mask, float("-inf"))

        # Generate correct labels
        # If z_i indexes are [0, 1, ..., N-1], their positive pairs in z_j indexes are [N, N+1, ..., 2N-1]
        labels: torch.Tensor = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])

        similarity_matrix = similarity_matrix.float()

        # Cross-entropy loss
        # For each of the 2N vectors, we need to "guess" the correct positive pair among the other 2N-1 vectors
        loss: torch.Tensor = self.criterion(similarity_matrix, labels)

        return loss
