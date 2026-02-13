"""BridgeData V2 dataset loader for VLA training.

Loads 10k training trajectories for tabletop pick-place tasks:
- "pick place can"
- "pick place sponge"
- "square in circle drawer"

Supports data subsampling at 10%/25%/50% fractions for sample efficiency curves.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from PIL import Image
import torchvision.transforms as T


class BridgeDataV2Dataset(Dataset):
    """BridgeData V2 dataset for robotic manipulation.

    Data structure (expected):
        data_dir/
            trajectory_0000/
                metadata.json (task, success, ...)
                images/
                    frame_0000.png
                    frame_0001.png
                    ...
                actions.npy  # (T, 7) array of actions
            trajectory_0001/
            ...

    Each action is 7-DoF: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    """

    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        image_size: int = 224,
        action_horizon: int = 8,
        transform: Optional[callable] = None,
        max_trajectories: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing trajectory folders
            tasks: List of tasks to include (None = all tasks)
            image_size: Image resolution for resizing
            action_horizon: Number of action steps to predict
            transform: Optional image transform
            max_trajectories: Maximum number of trajectories to load
        """
        self.data_dir = Path(data_dir)
        self.tasks = tasks or [
            "pick place can",
            "pick place sponge",
            "square in circle drawer"
        ]
        self.image_size = image_size
        self.action_horizon = action_horizon

        # Default image transform
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Load dataset
        self.samples = self._load_samples(max_trajectories)

    def _load_samples(self, max_trajectories: Optional[int]) -> List[Dict]:
        """Load trajectory samples from disk.

        Returns:
            List of sample dictionaries with:
                - trajectory_dir: Path to trajectory
                - frame_idx: Frame index within trajectory
                - task: Task description string
                - actions: Action sequence (action_horizon, 7)
        """
        samples = []

        # Find all trajectory directories
        traj_dirs = sorted(self.data_dir.glob("trajectory_*"))

        if max_trajectories is not None:
            traj_dirs = traj_dirs[:max_trajectories]

        for traj_dir in traj_dirs:
            # Load metadata
            metadata_path = traj_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            task = metadata.get('task', 'unknown')

            # Filter by task
            if self.tasks and task not in self.tasks:
                continue

            # Load actions
            actions_path = traj_dir / "actions.npy"
            if not actions_path.exists():
                continue

            actions = np.load(actions_path)  # (T, 7)
            num_frames = len(actions)

            # Create samples for each valid frame
            # Each sample predicts action_horizon future actions
            for frame_idx in range(num_frames - self.action_horizon + 1):
                samples.append({
                    'trajectory_dir': traj_dir,
                    'frame_idx': frame_idx,
                    'task': task,
                    'actions': actions[frame_idx:frame_idx + self.action_horizon].copy()
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary containing:
                - image: (3, H, W) tensor
                - text_prompt: Task description string
                - actions: (action_horizon, 7) tensor
                - continuous_actions: (action_horizon, 6) tensor
                - gripper_actions: (action_horizon,) long tensor
        """
        sample = self.samples[idx]

        # Load image
        image_path = sample['trajectory_dir'] / 'images' / f"frame_{sample['frame_idx']:04d}.png"
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get actions
        actions = torch.from_numpy(sample['actions']).float()  # (H, 7)
        continuous_actions = actions[:, :6]  # (H, 6) - delta pose
        gripper_actions = (actions[:, 6] > 0.5).long()  # (H,) - gripper binary

        return {
            'image': image,
            'text_prompt': sample['task'],
            'actions': actions,
            'continuous_actions': continuous_actions,
            'gripper_actions': gripper_actions,
        }


def create_synthetic_dataset(
    save_dir: str,
    num_trajectories: int = 1000,
    frames_per_trajectory: int = 50,
    tasks: Optional[List[str]] = None
):
    """Create synthetic BridgeData V2 dataset for testing.

    Args:
        save_dir: Directory to save synthetic data
        num_trajectories: Number of trajectories to generate
        frames_per_trajectory: Frames per trajectory
        tasks: List of task names
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ["pick place can", "pick place sponge", "square in circle drawer"]

    print(f"Creating synthetic dataset with {num_trajectories} trajectories...")

    for traj_idx in range(num_trajectories):
        traj_dir = save_dir / f"trajectory_{traj_idx:04d}"
        traj_dir.mkdir(exist_ok=True)

        # Create images directory
        images_dir = traj_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Random task
        task = np.random.choice(tasks)

        # Generate synthetic images (random noise)
        for frame_idx in range(frames_per_trajectory):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(images_dir / f"frame_{frame_idx:04d}.png")

        # Generate synthetic actions (random deltas + gripper)
        actions = np.random.randn(frames_per_trajectory, 7).astype(np.float32)
        actions[:, :6] = np.clip(actions[:, :6] * 0.1, -0.5, 0.5)  # Small deltas
        actions[:, 6] = (np.random.rand(frames_per_trajectory) > 0.5).astype(np.float32)  # Gripper

        np.save(traj_dir / "actions.npy", actions)

        # Save metadata
        metadata = {
            'task': task,
            'success': bool(np.random.rand() > 0.3),  # 70% success rate
            'num_frames': frames_per_trajectory,
        }
        with open(traj_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if (traj_idx + 1) % 100 == 0:
            print(f"  Created {traj_idx + 1}/{num_trajectories} trajectories")

    print(f"Synthetic dataset created at {save_dir}")


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    data_subsample: float = 1.0,
    image_size: int = 224,
    action_horizon: int = 8,
    max_trajectories: Optional[int] = 10000,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with subsampling support.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_fraction: Fraction of data for training
        val_fraction: Fraction of data for validation
        data_subsample: Subsample fraction (0.1, 0.25, 0.5, 1.0)
        image_size: Image resolution
        action_horizon: Action prediction horizon
        max_trajectories: Max trajectories to load (e.g., 10000)
        **dataset_kwargs: Additional dataset arguments

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    dataset = BridgeDataV2Dataset(
        data_dir=data_dir,
        image_size=image_size,
        action_horizon=action_horizon,
        max_trajectories=max_trajectories,
        **dataset_kwargs
    )

    # Split into train/val/test
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = int(n * train_fraction)
    val_size = int(n * val_fraction)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Subsample training data for sample efficiency experiments
    if data_subsample < 1.0:
        subsample_size = int(len(train_indices) * data_subsample)
        train_indices = np.random.choice(
            train_indices, size=subsample_size, replace=False
        )
        print(f"Subsampled training data: {len(train_indices)} samples ({data_subsample*100:.0f}%)")

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Custom collate function to handle text prompts
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        text_prompts = [item['text_prompt'] for item in batch]
        continuous_actions = torch.stack([item['continuous_actions'] for item in batch])
        gripper_actions = torch.stack([item['gripper_actions'] for item in batch])

        return {
            'images': images,
            'text_prompts': text_prompts,
            'continuous_actions': continuous_actions,
            'gripper_actions': gripper_actions,
        }

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader
