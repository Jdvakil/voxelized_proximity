"""Metrics computation utilities."""

import numpy as np
from typing import Dict, List, Tuple


def compute_success_metrics(
    ee_poses: np.ndarray,
    target_poses: np.ndarray,
    gripper_states: np.ndarray,
    target_grippers: np.ndarray,
    distance_threshold: float = 0.05
) -> Dict[str, float]:
    """Compute success metrics for task completion.

    Args:
        ee_poses: End-effector poses (N, 7) - [x, y, z, qx, qy, qz, qw]
        target_poses: Target poses (N, 7)
        gripper_states: Gripper states (N,) - 0 or 1
        target_grippers: Target gripper states (N,)
        distance_threshold: Success distance threshold (meters)

    Returns:
        Dictionary with:
            - position_errors: Mean position error
            - success_rate: Fraction of successful completions
            - gripper_accuracy: Gripper state accuracy
    """
    # Position errors
    position_errors = np.linalg.norm(ee_poses[:, :3] - target_poses[:, :3], axis=1)
    mean_position_error = np.mean(position_errors)

    # Gripper accuracy
    gripper_correct = (gripper_states == target_grippers)
    gripper_accuracy = np.mean(gripper_correct)

    # Success: position within threshold AND gripper correct
    position_success = position_errors < distance_threshold
    success = position_success & gripper_correct
    success_rate = np.mean(success)

    return {
        'position_errors': mean_position_error,
        'success_rate': success_rate,
        'gripper_accuracy': gripper_accuracy,
    }


def compute_action_statistics(
    predicted_actions: np.ndarray,
    target_actions: np.ndarray
) -> Dict[str, float]:
    """Compute statistics for action predictions.

    Args:
        predicted_actions: Predicted actions (N, H, 7)
        target_actions: Target actions (N, H, 7)

    Returns:
        Dictionary with MAE, MSE, etc.
    """
    # Continuous actions (first 6 dims)
    pred_continuous = predicted_actions[..., :6]
    target_continuous = target_actions[..., :6]

    mae = np.mean(np.abs(pred_continuous - target_continuous))
    mse = np.mean((pred_continuous - target_continuous) ** 2)
    rmse = np.sqrt(mse)

    # Per-dimension errors
    per_dim_mae = np.mean(np.abs(pred_continuous - target_continuous), axis=(0, 1))

    # Gripper accuracy
    pred_gripper = (predicted_actions[..., 6] > 0.5).astype(int)
    target_gripper = (target_actions[..., 6] > 0.5).astype(int)
    gripper_acc = np.mean(pred_gripper == target_gripper)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'per_dim_mae': per_dim_mae.tolist(),
        'gripper_accuracy': gripper_acc,
    }


def analyze_sample_efficiency(
    results: Dict[float, Dict[str, float]],
    target_success_rate: float = 0.8
) -> Dict[str, any]:
    """Analyze sample efficiency results.

    Args:
        results: Dict mapping data_fraction -> metrics
        target_success_rate: Target success rate threshold

    Returns:
        Analysis dictionary
    """
    fractions = sorted(results.keys())
    success_rates = [results[f]['success_rate'] for f in fractions]

    # Find data fraction needed to reach target success rate
    data_needed = None
    for fraction, success in zip(fractions, success_rates):
        if success >= target_success_rate:
            data_needed = fraction
            break

    # Compute improvement over baseline (smallest fraction)
    baseline_success = success_rates[0]
    final_success = success_rates[-1]
    improvement = final_success - baseline_success

    return {
        'data_fractions': fractions,
        'success_rates': success_rates,
        'data_needed_for_target': data_needed,
        'target_success_rate': target_success_rate,
        'baseline_success': baseline_success,
        'final_success': final_success,
        'improvement': improvement,
    }
