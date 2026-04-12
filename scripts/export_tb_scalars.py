#!/usr/bin/env python3
import sys
import os
import argparse

# to import from another directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization_utils import export_scalars, DEFAULT_TAGS


def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar curves to PNGs."
    )
    parser.add_argument(
        "--logdir", type=str, default="runs", help="Directory with TB event files"
    )
    parser.add_argument(
        "--event", type=str, default=None, help="Explicit path to a single event file"
    )
    parser.add_argument(
        "--outdir", type=str, default="assets", help="Output directory for PNGs"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=list(DEFAULT_TAGS),
        help="Scalar tags to export",
    )

    args = parser.parse_args()
    export_scalars(
        logdir=args.logdir, outdir=args.outdir, tags=args.tags, event_path=args.event
    )


if __name__ == "__main__":
    main()
