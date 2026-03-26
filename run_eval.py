import torch
import argparse
import os
import torchvision.transforms.v2 as T
from pathlib import Path
from typing import Dict, Any

from src.utils import load_config, set_seed, update_results_json
from src.dataset import get_stl10_dataloader
from src.model import SimCLR
from src.eval import evaluate_model

BASE_PATH = Path(__file__).resolve().parent


def main(config_path: str, simclr_path: str, method: str) -> None:
    """
    Main function to run the evaluation of a trained SimCLR model.

    Args:
        config_path (str): Path to the configuration file.
        simclr_path (str): Path to the saved SimCLR model checkpoint.
        method (str): The evaluation method to use ('linear' or 'knn').
    """
    config: Dict[str, Any] = load_config(config_path)
    set_seed(config["seed"])

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    simclr_model = SimCLR(out_dim=config["model"]["out_dim"])
    checkpoint = torch.load(simclr_path, map_location=device)
    simclr_model.load_state_dict(checkpoint["state_dict"])
    simclr_model = simclr_model.to(device)
    print(f"SimCLR weights loaded from {simclr_path}")

    eval_transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ]
    )

    train_loader = get_stl10_dataloader(
        root_dir=str(BASE_PATH / config["dataset"]["root_dir"]),
        split="train",
        transform=eval_transform,
        batch_size=config["linear_eval"]["batch_size"],
        num_workers=config["linear_eval"]["num_workers"],
    )

    val_loader = get_stl10_dataloader(
        root_dir=str(BASE_PATH / config["dataset"]["root_dir"]),
        split="test",
        transform=eval_transform,
        batch_size=config["linear_eval"]["batch_size"],
        num_workers=config["linear_eval"]["num_workers"],
    )

    ckpt_name: str = os.path.splitext(os.path.basename(simclr_path))[0]
    save_path: str = os.path.join(
        str(BASE_PATH / config["training"]["save_dir"]),
        f"linear_classifier_{ckpt_name}.pth",
    )

    # Call the common interface passing the config
    accuracy: float = evaluate_model(
        method=method,
        simclr_model=simclr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        save_path=save_path,
    )

    # Format the key name for JSON
    checkpoint_name: str = os.path.basename(simclr_path)
    if method == "knn":
        k_val: int = config.get("knn_eval", {}).get("k", 20)
        method_key: str = f"knn_k{k_val}"
    else:
        method_key: str = "linear_eval"

    # Save the result
    update_results_json("results.json", checkpoint_name, method_key, accuracy)
    print(
        f"Results for {method_key} ({accuracy:.2f}%) successfully saved in results.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation for SimCLR")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--simclr_path",
        type=str,
        default="checkpoints/checkpoint.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["linear", "knn"],
        required=True,
        help="Evaluation method: linear or knn",
    )
    args = parser.parse_args()

    main(args.config, args.simclr_path, args.method)
