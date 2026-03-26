import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
from typing import Dict, Any, Optional
from torch.amp import autocast, GradScaler

from src.utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    gpu_transform: nn.Module,
    scaler: GradScaler,
) -> float:
    """
    Trains the model for 1 epoch.

    Args:
        model (nn.Module): SimCLR model.
        dataloader (DataLoader): Data loader.
        criterion (nn.Module): NT-Xent loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Compute device (CPU/GPU).
        epoch (int): Current epoch number.
        total_epochs (int): Total epochs count (for logging).
        gpu_transform (nn.Module): GPU-based augmentation transforms.
        scaler (GradScaler): Scaler for mixed precision training.

    Returns:
        float: Mean loss in the epoch.
    """
    model.train()
    gpu_transform.train()
    total_loss: float = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

    for batch in pbar:
        images, _ = batch

        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            view1, view2 = gpu_transform(images)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16):
            # Forward pass
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss: torch.Tensor = criterion(z1, z2)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Updating stats
        total_loss += loss.item()
        current_lr: float = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

    avg_loss: float = total_loss / len(dataloader)
    return avg_loss


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    config: Dict[str, Any],
    device: torch.device,
    start_epoch: int,
    best_loss: float,
    gpu_transform: nn.Module,
    scheduler: Optional[LRScheduler] = None,
) -> None:
    """
    Main training loop.

    Args:
        model (nn.Module): SimCLR model.
        dataloader (DataLoader): Data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        config (Dict[str, Any]): Dictionary with configuration parameters.
        device (torch.device): Compute device (CPU/GPU).
        start_epoch (int): Epoch from which training starts.
        best_loss (float): Best loss for saving the best model.
        gpu_transform (nn.Module): GPU-based augmentation transforms.
        scheduler (Optional[LRScheduler]): Learning rate scheduler.
    """

    total_epochs: int = config["training"]["epochs"]
    save_dir: str = config["training"]["save_dir"]
    log_dir: str = config["training"]["log_dir"]

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Training from epoch {start_epoch + 1} to {total_epochs}")
    scaler = GradScaler("cuda")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        avg_loss: float = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=total_epochs,
            gpu_transform=gpu_transform,
            scaler=scaler,
        )

        current_lr: float = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch} is done! Avg loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/LR", current_lr, epoch)

        is_best: bool = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            print("New best loss!")

        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state: Dict[str, Any] = {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": config,
            "scheduler": scheduler.state_dict() if scheduler else None,
        }

        save_checkpoint(state=state, is_best=is_best, save_dir=save_dir)

        # Clearing cache
        gc.collect()
        torch.cuda.empty_cache()

        print("-" * 50)

    writer.close()
    print("Training successfully completed!")
