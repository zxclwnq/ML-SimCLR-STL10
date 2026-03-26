import yaml
import torch
import os
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML config as a dictionary.

    Args:
        config_path (str): The path to the .yaml config file.

    Returns:
        Dict[str, Any]: A dictionary with hyperparameters.
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """
    Fixes the seed for reproducibility.

    Args:
        seed (int): Value of the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str,
    filename: str = "checkpoint.pth",
) -> None:
    """
    Saves the current state of the model.

    Args:
        state (Dict[str, Any]): A dictionary of model states (state_dict, optimizer, etc.).
        is_best (bool): Flag, True if the current model is the best over all epochs.
        save_dir (str): The directory for the checkpoint file.
        filename (str): The name of the checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(save_dir, "model_best.pth")
        torch.save(state, best_path)

    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, float]:
    """
    Loads a checkpoint to resume training.

    Args:
        checkpoint_path (str): The path to the .pth file.
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The model's optimizer.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): The learning rate scheduler (if used).

    Returns:
        Tuple[int, float]: The epoch number to resume from and the best recorded loss.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    start_epoch: int = checkpoint["epoch"]
    best_loss: float = checkpoint.get("best_loss", float("inf"))

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"=> Checkpoint loaded (resuming from epoch {start_epoch})")

    return start_epoch, best_loss


def update_results_json(
    results_file: str, checkpoint_name: str, method_name: str, accuracy: float
) -> None:
    """
    Updates a JSON file with evaluation results.

    Args:
        results_file (str): The path to the JSON results file.
        checkpoint_name (str): The name of the checkpoint being evaluated.
        method_name (str): The name of the evaluation method ('linear' or 'knn').
        accuracy (float): The accuracy score.
    """
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    if checkpoint_name not in results:
        results[checkpoint_name] = {}

    results[checkpoint_name][method_name] = accuracy
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
