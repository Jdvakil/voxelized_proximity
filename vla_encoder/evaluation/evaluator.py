"""Evaluation metrics and OOD testing for VLA policies.

Metrics:
- Normalized task success rate (threshold: 5cm, gripper correct)
- Success on OOD object sets (train on cans/sponges → test on bottles/blocks)
- Compute profiling (VRAM, wall-clock time per epoch)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import json
from tqdm import tqdm

from ..envs.franka_tabletop import FrankaTabletopEnv
from ..utils.metrics import compute_success_metrics


class VLAEvaluator:
    """Evaluator for VLA policies in Isaac Gym environments."""

    def __init__(
        self,
        model: nn.Module,
        env: FrankaTabletopEnv,
        device: str = 'cuda',
        num_eval_episodes: int = 100,
        max_episode_length: int = 200,
        success_threshold: float = 0.05,  # 5cm
    ):
        """
        Args:
            model: VLA policy model
            env: Isaac Gym environment
            device: Device for inference
            num_eval_episodes: Number of episodes to evaluate
            max_episode_length: Maximum steps per episode
            success_threshold: Distance threshold for success (meters)
        """
        self.model = model.to(device)
        self.env = env
        self.device = device
        self.num_eval_episodes = num_eval_episodes
        self.max_episode_length = max_episode_length
        self.success_threshold = success_threshold

    @torch.no_grad()
    def evaluate(
        self,
        tasks: Optional[List[str]] = None,
        return_trajectories: bool = False
    ) -> Dict[str, any]:
        """Evaluate policy on specified tasks.

        Args:
            tasks: List of tasks to evaluate (None = all)
            return_trajectories: Whether to return full trajectories

        Returns:
            Dictionary with:
                - success_rate: Overall success rate
                - task_success_rates: Per-task success rates
                - avg_episode_length: Average episode length
                - trajectories: List of trajectories (if return_trajectories=True)
        """
        self.model.eval()

        if tasks is None:
            tasks = self.env.tasks

        results = {
            'success_rate': 0.0,
            'task_success_rates': {},
            'avg_episode_length': 0.0,
            'trajectories': [] if return_trajectories else None,
        }

        # Track metrics
        all_successes = []
        all_episode_lengths = []
        task_successes = {task: [] for task in tasks}

        print(f"Evaluating on {self.num_eval_episodes} episodes...")

        for episode_idx in tqdm(range(self.num_eval_episodes)):
            # Reset environment
            task = np.random.choice(tasks)
            obs = self.env.reset()

            episode_success = False
            episode_trajectory = []

            for step in range(self.max_episode_length):
                # Get action from policy
                image = obs['images'][0].to(self.device)
                text_prompt = obs['text_prompts'][0]

                action_dict = self.model.predict_action(image, text_prompt)

                # Take first action from horizon
                action = action_dict['continuous_actions'][0].cpu().numpy()  # (6,)
                gripper = action_dict['gripper_actions'][0].cpu().item()  # scalar

                # Combine into 7-DoF action
                full_action = np.concatenate([action, [gripper]])
                full_action = torch.from_numpy(full_action).unsqueeze(0).float()

                # Step environment
                obs, reward, done, info = self.env.step(full_action)

                if return_trajectories:
                    episode_trajectory.append({
                        'image': image.cpu().numpy(),
                        'action': full_action.cpu().numpy(),
                        'reward': reward.item(),
                    })

                if done[0]:
                    episode_success = info[0]['success']
                    break

            # Record metrics
            all_successes.append(episode_success)
            all_episode_lengths.append(step + 1)
            task_successes[task].append(episode_success)

            if return_trajectories:
                results['trajectories'].append({
                    'task': task,
                    'success': episode_success,
                    'length': step + 1,
                    'trajectory': episode_trajectory,
                })

        # Compute aggregate metrics
        results['success_rate'] = np.mean(all_successes)
        results['avg_episode_length'] = np.mean(all_episode_lengths)

        for task in tasks:
            if len(task_successes[task]) > 0:
                results['task_success_rates'][task] = np.mean(task_successes[task])

        return results

    def evaluate_ood(
        self,
        train_tasks: List[str],
        test_tasks: List[str]
    ) -> Dict[str, float]:
        """Evaluate out-of-distribution performance.

        Args:
            train_tasks: Tasks model was trained on
            test_tasks: OOD tasks to evaluate

        Returns:
            Dictionary with:
                - in_distribution_success: Success on train tasks
                - ood_success: Success on test tasks
                - ood_generalization: Ratio of OOD to in-dist success
        """
        print(f"Evaluating in-distribution tasks: {train_tasks}")
        in_dist_results = self.evaluate(tasks=train_tasks)

        print(f"Evaluating OOD tasks: {test_tasks}")
        ood_results = self.evaluate(tasks=test_tasks)

        return {
            'in_distribution_success': in_dist_results['success_rate'],
            'ood_success': ood_results['success_rate'],
            'ood_generalization': ood_results['success_rate'] / max(in_dist_results['success_rate'], 1e-6),
        }

    def profile_compute(
        self,
        num_steps: int = 100
    ) -> Dict[str, any]:
        """Profile compute resources (VRAM, inference time).

        Args:
            num_steps: Number of forward passes to profile

        Returns:
            Dictionary with:
                - peak_memory_mb: Peak GPU memory usage
                - avg_inference_time_ms: Average inference time
                - throughput_fps: Inference throughput (frames/sec)
        """
        self.model.eval()

        # Warm up
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_text = ["pick place can"]

        for _ in range(10):
            _ = self.model(dummy_image, dummy_text)

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Profile
        inference_times = []
        for _ in tqdm(range(num_steps), desc="Profiling"):
            start_time = time.time()
            _ = self.model(dummy_image, dummy_text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)

        # Get memory stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        else:
            peak_memory = 0.0

        avg_time = np.mean(inference_times) * 1000  # ms
        throughput = 1.0 / np.mean(inference_times)  # fps

        return {
            'peak_memory_mb': peak_memory,
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
        }


def evaluate_sample_efficiency(
    model_checkpoints: Dict[str, str],
    data_fractions: List[float],
    env: FrankaTabletopEnv,
    num_episodes: int = 100,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """Evaluate sample efficiency across data fractions.

    Args:
        model_checkpoints: Dict mapping data_fraction -> checkpoint_path
        data_fractions: List of data fractions (e.g., [0.1, 0.25, 0.5, 1.0])
        env: Isaac Gym environment
        num_episodes: Episodes per evaluation
        device: Device for inference

    Returns:
        Dictionary mapping data_fraction -> metrics
    """
    from ..models import create_vla_policy

    results = {}

    for fraction in data_fractions:
        print(f"\n{'='*60}")
        print(f"Evaluating at {fraction*100:.0f}% data fraction")
        print(f"{'='*60}")

        if fraction not in model_checkpoints:
            print(f"  Warning: No checkpoint for fraction {fraction}")
            continue

        # Load model
        checkpoint_path = model_checkpoints[fraction]
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model (this is simplified; actual implementation needs config)
        model = create_vla_policy(
            vision_encoder_type='resnet18',  # This should come from config
            d_model=768,
            n_layers=6,
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        evaluator = VLAEvaluator(
            model=model,
            env=env,
            device=device,
            num_eval_episodes=num_episodes
        )

        metrics = evaluator.evaluate()

        results[fraction] = {
            'success_rate': metrics['success_rate'],
            'avg_episode_length': metrics['avg_episode_length'],
            'task_success_rates': metrics['task_success_rates'],
        }

        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Avg episode length: {metrics['avg_episode_length']:.1f}")

    return results


def compare_encoders(
    encoder_checkpoints: Dict[str, str],
    env: FrankaTabletopEnv,
    data_fraction: float = 0.25,
    num_episodes: int = 100,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, any]]:
    """Compare different vision encoders at a given data fraction.

    Args:
        encoder_checkpoints: Dict mapping encoder_name -> checkpoint_path
        env: Isaac Gym environment
        data_fraction: Data fraction to compare at
        num_episodes: Episodes per evaluation
        device: Device for inference
        save_path: Path to save results JSON

    Returns:
        Dictionary mapping encoder_name -> metrics
    """
    from ..models import create_vla_policy

    results = {}

    for encoder_name, checkpoint_path in encoder_checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {encoder_name} encoder")
        print(f"{'='*60}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        model = create_vla_policy(
            vision_encoder_type=encoder_name,
            d_model=768,
            n_layers=6,
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        evaluator = VLAEvaluator(
            model=model,
            env=env,
            device=device,
            num_eval_episodes=num_episodes
        )

        # In-distribution evaluation
        metrics = evaluator.evaluate()

        # Compute profiling
        profile = evaluator.profile_compute(num_steps=100)

        # OOD evaluation
        ood_metrics = evaluator.evaluate_ood(
            train_tasks=["pick place can", "pick place sponge"],
            test_tasks=["pick place bottle", "pick place block"]
        )

        results[encoder_name] = {
            'success_rate': metrics['success_rate'],
            'task_success_rates': metrics['task_success_rates'],
            'ood_metrics': ood_metrics,
            'compute_profile': profile,
            'model_params': checkpoint.get('model_config', {}),
        }

        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  OOD success: {ood_metrics['ood_success']:.2%}")
        print(f"  Peak VRAM: {profile['peak_memory_mb']:.1f} MB")

    # Save results
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return results
