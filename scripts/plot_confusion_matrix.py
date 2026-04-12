#!/usr/bin/env python3
import sys
import os
import argparse

# to import from another directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization_utils import plot_cm


def main():
    parser = argparse.ArgumentParser(
        description="Plot Confusion Matrix for Linear Evaluation."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--simclr_ckpt",
        type=str,
        default="checkpoints/checkpoint.pth",
        help="SimCLR base checkpoint",
    )
    parser.add_argument(
        "--linear_ckpt",
        type=str,
        default="checkpoints/linear_classifier_checkpoint.pth",
        help="Linear head checkpoint",
    )
    parser.add_argument(
        "--out", type=str, default="assets/confusion_matrix.png", help="Output PNG path"
    )

    args = parser.parse_args()
    plot_cm(
        config_path=args.config,
        simclr_ckpt=args.simclr_ckpt,
        linear_ckpt=args.linear_ckpt,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
