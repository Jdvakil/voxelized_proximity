#!/usr/bin/env python3
"""Training script for VLA policy."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path

from vla_encoder.configs import load_config
from vla_encoder.models import create_vla_policy
from vla_encoder.datasets import create_dataloaders
from vla_encoder.training import VLATrainer, VLALoss


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train VLA policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Override data fraction from config"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override data fraction if specified
    if args.data_fraction is not None:
        config.data_fraction = args.data_fraction
        config.experiment_name = f"{config.experiment_name}_data{int(args.data_fraction*100)}"

    print(f"Experiment: {config.experiment_name}")
    print(f"Vision encoder: {config.vision_encoder}")
    print(f"Data fraction: {config.data_fraction*100:.0f}%")

    # Set seed
    set_seed(config.seed)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        data_subsample=config.data_fraction,
        image_size=config.image_size,
        action_horizon=config.action_horizon,
        max_trajectories=config.max_trajectories,
    )

    # Create model
    print("\nCreating model...")
    model = create_vla_policy(
        vision_encoder_type=config.vision_encoder,
        vision_encoder_kwargs={
            'freeze_backbone': config.freeze_vision_encoder,
            'pretrained': config.vision_pretrained,
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

    # Print model info
    param_counts = model.get_num_params()
    print(f"\nModel parameters:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create loss function
    loss_fn = VLALoss(
        continuous_weight=config.continuous_loss_weight,
        gripper_weight=config.gripper_loss_weight,
        use_huber=config.use_huber_loss,
    )

    # Update checkpoint and log directories
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
    log_dir = Path(config.log_dir) / config.experiment_name

    # Create trainer
    trainer = VLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config.device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        grad_clip_norm=config.grad_clip_norm,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Save config
    config.save(checkpoint_dir / "config.yaml")

    # Train
    print("\nStarting training...\n")
    history = trainer.train()

    # Plot training curves
    trainer.logger.plot_metrics(log_dir / "training_curves.png")

    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
