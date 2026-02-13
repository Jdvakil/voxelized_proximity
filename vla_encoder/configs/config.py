"""Experiment configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for VLA ablation experiments."""

    # Experiment metadata
    experiment_name: str = "vla_encoder_ablation"
    vision_encoder: str = "resnet18"  # resnet18, vc1, dinov2
    data_fraction: float = 1.0  # 0.1, 0.25, 0.5, 1.0
    seed: int = 42

    # Model architecture
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    action_dim: int = 7
    action_horizon: int = 8
    image_size: int = 224

    # Vision encoder settings
    freeze_vision_encoder: bool = True  # True for VC-1/DINOv2, False for ResNet
    vision_pretrained: bool = True

    # Training
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10

    # Data
    data_dir: str = "./data/bridge_v2"
    max_trajectories: int = 10000
    train_fraction: float = 0.8
    val_fraction: float = 0.1

    # Loss weights
    continuous_loss_weight: float = 1.0
    gripper_loss_weight: float = 0.5
    use_huber_loss: bool = False

    # Evaluation
    num_eval_episodes: int = 100
    max_episode_length: int = 200
    success_threshold: float = 0.05  # 5cm

    # Paths
    checkpoint_dir: str = "./experiments/checkpoints"
    log_dir: str = "./experiments/logs"

    # Device
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, path: str):
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig.load(config_path)


def save_config(config: ExperimentConfig, save_path: str):
    """Save experiment configuration.

    Args:
        config: ExperimentConfig instance
        save_path: Path to save YAML file
    """
    config.save(save_path)


def create_encoder_config(
    encoder_type: str,
    data_fraction: float = 1.0,
    seed: int = 42,
    **overrides
) -> ExperimentConfig:
    """Create configuration for specific encoder type.

    Args:
        encoder_type: 'resnet18', 'vc1', or 'dinov2'
        data_fraction: Data fraction for sample efficiency
        seed: Random seed
        **overrides: Additional config overrides

    Returns:
        ExperimentConfig instance
    """
    # Base config
    config = ExperimentConfig(
        vision_encoder=encoder_type,
        data_fraction=data_fraction,
        seed=seed
    )

    # Encoder-specific settings
    if encoder_type == "resnet18":
        config.freeze_vision_encoder = False  # ResNet-18 is trainable
        config.image_size = 224
    elif encoder_type == "vc1":
        config.freeze_vision_encoder = True  # VC-1 is frozen
        config.image_size = 336
    elif encoder_type == "dinov2":
        config.freeze_vision_encoder = True  # DINOv2 is frozen
        config.image_size = 224
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Update experiment name
    config.experiment_name = f"vla_{encoder_type}_data{int(data_fraction*100)}_seed{seed}"

    return config
