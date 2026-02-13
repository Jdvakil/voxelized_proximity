"""Training loop for VLA policy.

Implements training with:
- AdamW optimizer (lr=1e-4)
- Batch size 32
- 100 epochs or early stopping at validation plateau
- Gradient clipping
- Learning rate scheduling
- Checkpoint saving
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from .losses import VLALoss
from ..utils.logging import MetricsLogger


class VLATrainer:
    """Trainer for VLA policy."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[VLALoss] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        grad_clip_norm: float = 1.0,
        log_interval: int = 10,
        val_interval: int = 1,
        save_interval: int = 5,
    ):
        """
        Args:
            model: VLA policy model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer (default: AdamW with lr=1e-4)
            scheduler: LR scheduler (default: ReduceLROnPlateau)
            loss_fn: Loss function (default: VLALoss)
            device: Device for training
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            max_epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            grad_clip_norm: Gradient clipping norm
            log_interval: Log metrics every N steps
            val_interval: Validate every N epochs
            save_interval: Save checkpoint every N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = scheduler

        # Loss function
        if loss_fn is None:
            self.loss_fn = VLALoss()
        else:
            self.loss_fn = loss_fn

        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = MetricsLogger(self.log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'continuous_loss': 0.0,
            'gripper_loss': 0.0,
            'gripper_accuracy': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['images'].to(self.device)
            text_prompts = batch['text_prompts']
            target_continuous = batch['continuous_actions'].to(self.device)
            target_gripper = batch['gripper_actions'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, text_prompts)

            # Compute loss
            loss_dict = self.loss_fn(
                outputs['continuous_actions'],
                outputs['gripper_logits'],
                target_continuous,
                target_gripper
            )

            # Backward pass
            loss_dict['loss'].backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.optimizer.step()

            # Compute accuracy
            accuracy = self.loss_fn.compute_accuracy(
                outputs['gripper_logits'],
                target_gripper
            )

            # Update metrics
            batch_size = images.size(0)
            epoch_metrics['loss'] += loss_dict['loss'].item() * batch_size
            epoch_metrics['continuous_loss'] += loss_dict['continuous_loss'].item() * batch_size
            epoch_metrics['gripper_loss'] += loss_dict['gripper_loss'].item() * batch_size
            epoch_metrics['gripper_accuracy'] += accuracy.item() * batch_size

            # Log
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': loss_dict['loss'].item(),
                    'acc': accuracy.item()
                })

            self.global_step += 1

        # Average metrics
        num_samples = len(self.train_loader.dataset)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_samples

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'continuous_loss': 0.0,
            'gripper_loss': 0.0,
            'gripper_accuracy': 0.0,
        }

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            images = batch['images'].to(self.device)
            text_prompts = batch['text_prompts']
            target_continuous = batch['continuous_actions'].to(self.device)
            target_gripper = batch['gripper_actions'].to(self.device)

            # Forward pass
            outputs = self.model(images, text_prompts)

            # Compute loss
            loss_dict = self.loss_fn(
                outputs['continuous_actions'],
                outputs['gripper_logits'],
                target_continuous,
                target_gripper
            )

            # Compute accuracy
            accuracy = self.loss_fn.compute_accuracy(
                outputs['gripper_logits'],
                target_gripper
            )

            # Update metrics
            batch_size = images.size(0)
            val_metrics['loss'] += loss_dict['loss'].item() * batch_size
            val_metrics['continuous_loss'] += loss_dict['continuous_loss'].item() * batch_size
            val_metrics['gripper_loss'] += loss_dict['gripper_loss'].item() * batch_size
            val_metrics['gripper_accuracy'] += accuracy.item() * batch_size

        # Average metrics
        num_samples = len(self.val_loader.dataset)
        for key in val_metrics:
            val_metrics[key] /= num_samples

        return val_metrics

    def train(self) -> Dict[str, any]:
        """Main training loop.

        Returns:
            Training history
        """
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")

        # Model parameters
        param_counts = self.model.get_num_params()
        print(f"\nModel parameters:")
        for key, value in param_counts.items():
            print(f"  {key}: {value:,}")

        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Log training metrics
            self.logger.log_metrics('train', train_metrics, epoch)
            print(f"\nEpoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
                  f"acc={train_metrics['gripper_accuracy']:.4f}")

            # Validate
            if epoch % self.val_interval == 0:
                val_metrics = self.validate()

                # Log validation metrics
                self.logger.log_metrics('val', val_metrics, epoch)
                print(f"Epoch {epoch} - Val: loss={val_metrics['loss']:.4f}, "
                      f"acc={val_metrics['gripper_accuracy']:.4f}")

                # Learning rate scheduling
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint('best_model.pt', is_best=True)
                    print(f"  New best model! Val loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # Save periodic checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint('final_model.pt')

        return self.logger.get_history()

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model.get_num_params(),
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")

        if is_best:
            # Also save metadata
            metadata = {
                'epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'model_params': self.model.get_num_params(),
            }
            with open(self.checkpoint_dir / 'best_model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']

            print(f"  Resuming from epoch {self.current_epoch}")
