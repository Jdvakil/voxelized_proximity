#!/usr/bin/env python3
"""Run complete ablation study across all encoders and data fractions.

This script trains and evaluates all three vision encoders (ResNet-18, VC-1, DINOv2)
at different data fractions (10%, 25%, 50%, 100%) to generate sample efficiency curves.
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict
import time

from vla_encoder.configs import create_encoder_config


ENCODERS = ["resnet18", "vc1", "dinov2"]
DATA_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def run_training(
    encoder: str,
    data_fraction: float,
    base_config_path: str,
    seed: int = 42
) -> Dict[str, str]:
    """Run training for a specific encoder and data fraction.

    Args:
        encoder: Encoder type
        data_fraction: Data fraction
        base_config_path: Path to base config
        seed: Random seed

    Returns:
        Dictionary with checkpoint paths
    """
    print(f"\n{'='*80}")
    print(f"Training {encoder.upper()} at {data_fraction*100:.0f}% data")
    print(f"{'='*80}\n")

    # Create config for this run
    config = create_encoder_config(
        encoder_type=encoder,
        data_fraction=data_fraction,
        seed=seed
    )

    # Save temporary config
    temp_config_path = Path(f"./tmp_{encoder}_{int(data_fraction*100)}.yaml")
    config.save(temp_config_path)

    # Run training
    cmd = [
        "python", "scripts/train.py",
        "--config", str(temp_config_path)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        success = False

    # Clean up temp config
    if temp_config_path.exists():
        temp_config_path.unlink()

    # Return checkpoint paths
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
    return {
        'success': success,
        'checkpoint_dir': str(checkpoint_dir),
        'best_model': str(checkpoint_dir / "best_model.pt"),
        'config': str(checkpoint_dir / "config.yaml"),
    }


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 100,
    save_results: str = None
) -> Dict:
    """Run evaluation on a trained model.

    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config
        num_episodes: Number of evaluation episodes
        save_results: Path to save results

    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating: {checkpoint_path}")

    cmd = [
        "python", "scripts/evaluate.py",
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--num-episodes", str(num_episodes),
        "--eval-ood",
        "--profile",
    ]

    if save_results:
        cmd.extend(["--save-results", save_results])

    try:
        subprocess.run(cmd, check=True, capture_output=False)

        # Load results if saved
        if save_results and Path(save_results).exists():
            with open(save_results, 'r') as f:
                results = json.load(f)
            return results
        else:
            return {}

    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run VLA encoder ablation study")
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=ENCODERS,
        choices=ENCODERS,
        help="Encoders to evaluate"
    )
    parser.add_argument(
        "--data-fractions",
        nargs="+",
        type=float,
        default=DATA_FRACTIONS,
        help="Data fractions to evaluate"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run evaluation"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation, only run training"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiments/ablation_results",
        help="Directory to save results"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ablation_results = {
        'encoders': {},
        'metadata': {
            'encoders': args.encoders,
            'data_fractions': args.data_fractions,
            'seed': args.seed,
            'num_eval_episodes': args.num_eval_episodes,
        }
    }

    start_time = time.time()

    # Run experiments for each encoder
    for encoder in args.encoders:
        print(f"\n{'#'*80}")
        print(f"# ENCODER: {encoder.upper()}")
        print(f"{'#'*80}\n")

        ablation_results['encoders'][encoder] = {}

        for data_fraction in args.data_fractions:
            # Training
            if not args.skip_training:
                train_results = run_training(
                    encoder=encoder,
                    data_fraction=data_fraction,
                    base_config_path=f"vla_encoder/configs/{encoder}.yaml",
                    seed=args.seed
                )

                ablation_results['encoders'][encoder][data_fraction] = {
                    'training': train_results
                }
            else:
                # Assume checkpoint exists
                config = create_encoder_config(encoder, data_fraction, args.seed)
                checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
                train_results = {
                    'checkpoint_dir': str(checkpoint_dir),
                    'best_model': str(checkpoint_dir / "best_model.pt"),
                    'config': str(checkpoint_dir / "config.yaml"),
                }

            # Evaluation
            if not args.skip_evaluation:
                eval_save_path = results_dir / f"{encoder}_data{int(data_fraction*100)}_results.json"

                eval_results = run_evaluation(
                    checkpoint_path=train_results['best_model'],
                    config_path=train_results['config'],
                    num_episodes=args.num_eval_episodes,
                    save_results=str(eval_save_path)
                )

                if data_fraction not in ablation_results['encoders'][encoder]:
                    ablation_results['encoders'][encoder][data_fraction] = {}

                ablation_results['encoders'][encoder][data_fraction]['evaluation'] = eval_results

    # Save complete ablation results
    total_time = time.time() - start_time
    ablation_results['metadata']['total_time_hours'] = total_time / 3600

    results_file = results_dir / "complete_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {results_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Success Rates by Encoder and Data Fraction")
    print(f"{'='*80}\n")
    print(f"{'Encoder':<12} ", end="")
    for frac in args.data_fractions:
        print(f"{int(frac*100):>6}% ", end="")
    print()
    print("-" * 80)

    for encoder in args.encoders:
        print(f"{encoder:<12} ", end="")
        for frac in args.data_fractions:
            if frac in ablation_results['encoders'][encoder]:
                eval_data = ablation_results['encoders'][encoder][frac].get('evaluation', {})
                standard_eval = eval_data.get('standard_eval', {})
                success_rate = standard_eval.get('success_rate', 0.0)
                print(f"{success_rate*100:>6.1f}% ", end="")
            else:
                print(f"{'N/A':>7} ", end="")
        print()

    print()


if __name__ == "__main__":
    main()
