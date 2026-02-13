#!/usr/bin/env python3
"""Generate synthetic BridgeData V2 dataset for testing."""

import argparse
from vla_encoder.datasets.bridge_data_v2 import create_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data/bridge_v2",
        help="Directory to save synthetic data"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1000,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--frames-per-trajectory",
        type=int,
        default=50,
        help="Frames per trajectory"
    )
    args = parser.parse_args()

    create_synthetic_dataset(
        save_dir=args.save_dir,
        num_trajectories=args.num_trajectories,
        frames_per_trajectory=args.frames_per_trajectory,
    )


if __name__ == "__main__":
    main()
