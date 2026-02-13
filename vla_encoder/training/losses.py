"""Loss functions for VLA training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VLALoss(nn.Module):
    """Combined loss for VLA policy training.

    Combines:
    1. MSE loss for continuous actions (delta pose)
    2. Cross-entropy loss for discrete actions (gripper)
    """

    def __init__(
        self,
        continuous_weight: float = 1.0,
        gripper_weight: float = 0.5,
        use_huber: bool = False,
        huber_delta: float = 1.0
    ):
        """
        Args:
            continuous_weight: Weight for continuous action loss
            gripper_weight: Weight for gripper loss
            use_huber: Use Huber loss instead of MSE for continuous actions
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.continuous_weight = continuous_weight
        self.gripper_weight = gripper_weight
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        if use_huber:
            self.continuous_loss_fn = nn.SmoothL1Loss(beta=huber_delta)
        else:
            self.continuous_loss_fn = nn.MSELoss()

        self.gripper_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_continuous: torch.Tensor,
        pred_gripper_logits: torch.Tensor,
        target_continuous: torch.Tensor,
        target_gripper: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_continuous: (B, H, 6) predicted delta poses
            pred_gripper_logits: (B, H, 2) gripper logits
            target_continuous: (B, H, 6) target delta poses
            target_gripper: (B, H) target gripper states (0 or 1)

        Returns:
            Dictionary with:
                - loss: Total loss
                - continuous_loss: Continuous action loss
                - gripper_loss: Gripper action loss
        """
        # Continuous action loss
        continuous_loss = self.continuous_loss_fn(pred_continuous, target_continuous)

        # Gripper loss (reshape for cross-entropy)
        B, H, _ = pred_gripper_logits.shape
        gripper_logits_flat = pred_gripper_logits.reshape(-1, 2)  # (B*H, 2)
        gripper_targets_flat = target_gripper.reshape(-1)  # (B*H,)
        gripper_loss = self.gripper_loss_fn(gripper_logits_flat, gripper_targets_flat)

        # Combined loss
        total_loss = (
            self.continuous_weight * continuous_loss +
            self.gripper_weight * gripper_loss
        )

        return {
            'loss': total_loss,
            'continuous_loss': continuous_loss,
            'gripper_loss': gripper_loss,
        }

    def compute_accuracy(
        self,
        pred_gripper_logits: torch.Tensor,
        target_gripper: torch.Tensor
    ) -> torch.Tensor:
        """Compute gripper prediction accuracy.

        Args:
            pred_gripper_logits: (B, H, 2)
            target_gripper: (B, H)

        Returns:
            Accuracy scalar
        """
        pred_gripper = torch.argmax(pred_gripper_logits, dim=-1)  # (B, H)
        accuracy = (pred_gripper == target_gripper).float().mean()
        return accuracy
