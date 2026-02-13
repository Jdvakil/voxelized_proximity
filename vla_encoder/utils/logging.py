"""Logging utilities for training metrics."""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class MetricsLogger:
    """Logger for training and validation metrics."""

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train': [],
            'val': [],
        }

    def log_metrics(
        self,
        split: str,
        metrics: Dict[str, float],
        epoch: int
    ):
        """Log metrics for a split.

        Args:
            split: 'train' or 'val'
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        entry = {'epoch': epoch, **metrics}
        self.history[split].append(entry)

        # Save to JSON
        log_file = self.log_dir / f'{split}_metrics.json'
        with open(log_file, 'w') as f:
            json.dump(self.history[split], f, indent=2)

    def get_history(self) -> Dict[str, List[Dict]]:
        """Get full training history."""
        return self.history

    def get_best_epoch(self, metric: str = 'loss', split: str = 'val') -> int:
        """Get epoch with best metric value.

        Args:
            metric: Metric name
            split: 'train' or 'val'

        Returns:
            Best epoch number
        """
        if len(self.history[split]) == 0:
            return -1

        values = [entry[metric] for entry in self.history[split]]
        best_idx = np.argmin(values)
        return self.history[split][best_idx]['epoch']

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training curves.

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Metrics')

            metrics_to_plot = [
                ('loss', 'Loss'),
                ('continuous_loss', 'Continuous Action Loss'),
                ('gripper_loss', 'Gripper Loss'),
                ('gripper_accuracy', 'Gripper Accuracy'),
            ]

            for idx, (metric, title) in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]

                if len(self.history['train']) > 0:
                    train_epochs = [e['epoch'] for e in self.history['train']]
                    train_values = [e.get(metric, 0) for e in self.history['train']]
                    ax.plot(train_epochs, train_values, label='Train', marker='o')

                if len(self.history['val']) > 0:
                    val_epochs = [e['epoch'] for e in self.history['val']]
                    val_values = [e.get(metric, 0) for e in self.history['val']]
                    ax.plot(val_epochs, val_values, label='Val', marker='s')

                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')

            plt.close()

        except ImportError:
            print("Warning: matplotlib not available, skipping plot")
