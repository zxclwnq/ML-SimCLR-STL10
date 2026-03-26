import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable
import os


def get_stl10_dataloader(
    root_dir: str = "./data",
    split: str = "unlabeled",
    transform: Optional[Callable] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = True,
) -> DataLoader:
    """
    Creates and returns a DataLoader for the STL-10 dataset.

    Args:
        root_dir (str): Directory for installation and storing of files.
        split (str): Data split ('unlabeled', 'train', 'test', 'train+unlabeled').
        transform (Optional[Callable]): Augmentation function to apply.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        download (bool): True if needed to download dataset if it is not in root_dir.

    Returns:
        DataLoader: STL-10 dataset PyTorch DataLoader.
    """
    os.makedirs(root_dir, exist_ok=True)

    # Downloading STL-10 dataset
    dataset: Dataset = datasets.STL10(
        root=root_dir, split=split, transform=transform, download=download
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Essential for SimCLR, because it prevents errors in Contrastive Loss computation for the last incomplete batch
    )

    return dataloader
