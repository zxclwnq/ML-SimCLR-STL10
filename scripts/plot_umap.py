#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vizualization_utils import plot_umap


def main():
    parser = argparse.ArgumentParser(description="Plot UMAP projection for SimCLR.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/checkpoint.pth",
        help="Path to SimCLR checkpoint",
    )
    parser.add_argument(
        "--out", type=str, default="assets/simclr_umap.png", help="Output PNG path"
    )

    args = parser.parse_args()
    plot_umap(config_path=args.config, checkpoint_path=args.ckpt, out_path=args.out)


if __name__ == "__main__":
    main()
