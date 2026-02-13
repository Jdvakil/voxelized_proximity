#!/usr/bin/env python3
"""Evaluation script for trained VLA policies."""

import argparse
import torch
import json
from pathlib import Path

from vla_encoder.configs import load_config
from vla_encoder.models import create_vla_policy
from vla_encoder.envs import FrankaTabletopEnv
from vla_encoder.evaluation import VLAEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (if not in checkpoint dir)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--eval-ood",
        action="store_true",
        help="Evaluate OOD generalization"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile compute resources"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = args.config
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.yaml"

    config = load_config(config_path)
    print(f"Evaluating: {config.experiment_name}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device)

    # Create model
    model = create_vla_policy(
        vision_encoder_type=config.vision_encoder,
        vision_encoder_kwargs={
            'freeze_backbone': config.freeze_vision_encoder,
            'pretrained': False,  # Loading from checkpoint
            'image_size': config.image_size,
        },
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    # Create environment
    print("\nInitializing Isaac Gym environment...")
    env = FrankaTabletopEnv(
        num_envs=1,
        device=config.device,
        image_size=config.image_size,
    )

    # Create evaluator
    evaluator = VLAEvaluator(
        model=model,
        env=env,
        device=config.device,
        num_eval_episodes=args.num_episodes,
        success_threshold=config.success_threshold,
    )

    results = {}

    # Standard evaluation
    print(f"\nEvaluating on {args.num_episodes} episodes...")
    eval_results = evaluator.evaluate()

    print(f"\nResults:")
    print(f"  Success rate: {eval_results['success_rate']:.2%}")
    print(f"  Avg episode length: {eval_results['avg_episode_length']:.1f}")
    print(f"\nPer-task success rates:")
    for task, success_rate in eval_results['task_success_rates'].items():
        print(f"  {task}: {success_rate:.2%}")

    results['standard_eval'] = eval_results

    # OOD evaluation
    if args.eval_ood:
        print("\nEvaluating OOD generalization...")
        ood_results = evaluator.evaluate_ood(
            train_tasks=["pick place can", "pick place sponge"],
            test_tasks=["pick place bottle", "pick place block"]
        )

        print(f"\nOOD Results:")
        print(f"  In-distribution success: {ood_results['in_distribution_success']:.2%}")
        print(f"  OOD success: {ood_results['ood_success']:.2%}")
        print(f"  OOD generalization ratio: {ood_results['ood_generalization']:.2f}")

        results['ood_eval'] = ood_results

    # Compute profiling
    if args.profile:
        print("\nProfiling compute resources...")
        profile_results = evaluator.profile_compute(num_steps=100)

        print(f"\nCompute Profile:")
        print(f"  Peak VRAM: {profile_results['peak_memory_mb']:.1f} MB")
        print(f"  Avg inference time: {profile_results['avg_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {profile_results['throughput_fps']:.1f} FPS")

        results['compute_profile'] = profile_results

    # Save results
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {save_path}")

    env.close()


if __name__ == "__main__":
    main()
