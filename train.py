import torch
import os
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch_optimizer as optim
from torchvision.transforms import v2
from typing import List, Dict, Any

from src.utils import load_config, set_seed, load_checkpoint
from src.augmentations import SimCLRTransform
from src.dataset import get_stl10_dataloader
from src.model import SimCLR
from src.loss import NTXentLoss
from src.trainer import train
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent


def get_params_groups(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """
    Separates model parameters into groups with and without weight decay.

    Args:
        model (torch.nn.Module): The model to extract parameters from.
        weight_decay (float): The weight decay value to apply.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing parameter groups.
    """
    regular_params: List[torch.nn.Parameter] = []
    exclude_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Exclude bias and BatchNorm weights (they are always 1D)
        if p.ndim <= 1 or name.endswith(".bias"):
            exclude_params.append(p)
        else:
            regular_params.append(p)

    return [
        {"params": regular_params, "weight_decay": weight_decay},
        {"params": exclude_params, "weight_decay": 0.0},
    ]


def main(config_path: str) -> None:
    """
    Main function to setup and run SimCLR training.

    Args:
        config_path (str): Path to the .yaml configuration file.
    """
    print("=== Init of SimCLR learning ===")

    config: Dict[str, Any] = load_config(config_path=config_path)
    set_seed(config["seed"])

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Transforms init
    basic_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    gpu_transform = SimCLRTransform(input_size=config["dataset"]["image_size"]).to(
        device
    )

    dataloader = get_stl10_dataloader(
        root_dir=str(BASE_PATH / config["dataset"]["root_dir"]),
        transform=basic_transform,
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )
    print(f"[*] Dataset size: {len(dataloader.dataset)}, batches: {len(dataloader)}")  # type: ignore

    model = SimCLR(out_dim=config["model"]["out_dim"])
    if torch.cuda.device_count() > 1:
        print(f"[*] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = NTXentLoss(temperature=config["training"]["temperature"]).to(device)

    param_groups = get_params_groups(model, config["training"]["weight_decay"])

    optimizer = optim.LARS(
        param_groups,
        lr=config["training"]["lr"],
        momentum=0.9,
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler init
    warmup_epochs: int = 10
    total_epochs: int = config["training"]["epochs"]

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),
        eta_min=config["training"]["min_lr"],
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    start_epoch: int = 0
    best_loss: float = float("inf")

    save_dir = BASE_PATH / config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    auto_path: str = os.path.join(save_dir, "checkpoint.pth")
    resume_path: str = config["training"]["resume_checkpoint"] or auto_path

    if os.path.exists(resume_path):
        print(f"[*] Found checkpoint. Resuming from {resume_path}")
        start_epoch, best_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
    else:
        print("[*] No checkpoints found. Start of training")

    print("=== Start ===")
    train(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        start_epoch=start_epoch,
        best_loss=best_loss,
        gpu_transform=gpu_transform,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR training on STL-10 dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to .yaml config file",
    )
    args = parser.parse_args()

    main(args.config)
