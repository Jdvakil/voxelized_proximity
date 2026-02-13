# VLA Vision Encoder Ablation Study

A controlled ablation study isolating the impact of pretrained vision encoders on sample efficiency in vision-language-action (VLA) policies for language-conditioned tabletop manipulation.

## Overview

This project implements and compares three vision encoders as drop-in replacements in an OpenVLA-derived architecture:

1. **ResNet-18** - ImageNet-pretrained, 25M parameters, fully trainable
2. **VC-1 Large** - Egocentric video-pretrained ViT-L/336px, 307M parameters, frozen weights
3. **DINOv2 ViT-B/14** - Self-supervised semantic features, 86M parameters, frozen

### Architecture

**Policy Components:**
- **Vision Encoder:** Drop-in replacement (ResNet-18/VC-1/DINOv2)
- **Text Encoder:** Frozen T5-small (60M params)
- **Transformer:** 6-layer GPT-style decoder (50M params total)
- **Action Head:** Predicts 7-DoF actions over 8-step horizons
  - 6D continuous: Δx, Δy, Δz, Δroll, Δpitch, Δyaw
  - 1D discrete: gripper state (open/close)

**Training Setup:**
- **Dataset:** BridgeData V2 (10k trajectories)
- **Environment:** Isaac Gym Preview 4 with Franka Panda robot
- **Tasks:** "pick place can", "pick place sponge", "square in circle drawer"
- **Sample Efficiency:** Evaluate at 10%/25%/50%/100% data fractions
- **Domain Randomization:**
  - Lighting intensity: [0.7, 1.3]
  - Table friction: μ ∈ [0.3, 0.7]
  - Object masses: m ∈ [0.1, 0.5] kg

**Evaluation Metrics:**
- Normalized task success rate (threshold: 5cm position + gripper correct)
- Out-of-distribution performance (train on cans/sponges → test on bottles/blocks)
- Compute profiling (RTX 4090 peak VRAM, wall-clock time per epoch)

## Expected Results

**Sample Efficiency:**
- VC-1: ≥80% success at 25% data vs CNN's 50% threshold
- DINOv2: Strong semantic features improve generalization
- ResNet-18: Baseline CNN performance

**OOD Generalization:**
- VC-1: Maintains 75% OOD performance
- ResNet-18: Drops to ≤45% on OOD tasks

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: RTX 4090)
- Isaac Gym Preview 4 (optional for simulation)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vla_encoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. (Optional) Install Isaac Gym:
   - Download from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
   - Follow official installation instructions

5. (Optional) Install VC-1:
```bash
pip install git+https://github.com/facebookresearch/eai-vc.git
```

## Quick Start

### Generate Synthetic Data

For testing without BridgeData V2:

```bash
python scripts/generate_synthetic_data.py \
    --save-dir ./data/bridge_v2 \
    --num-trajectories 1000 \
    --frames-per-trajectory 50
```

### Train a Model

Train ResNet-18 encoder:
```bash
python scripts/train.py --config vla_encoder/configs/resnet18.yaml
```

Train with specific data fraction:
```bash
python scripts/train.py \
    --config vla_encoder/configs/vc1.yaml \
    --data-fraction 0.25
```

### Evaluate a Model

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/resnet18_*/best_model.pt \
    --num-episodes 100 \
    --eval-ood \
    --profile
```

### Run Complete Ablation Study

Train and evaluate all encoders at all data fractions:

```bash
python scripts/run_ablation.py \
    --encoders resnet18 vc1 dinov2 \
    --data-fractions 0.1 0.25 0.5 1.0 \
    --num-eval-episodes 100 \
    --results-dir ./experiments/ablation_results
```

## Project Structure

```
vla_encoder/
├── vla_encoder/
│   ├── models/
│   │   ├── vision_encoders.py    # ResNet-18, VC-1, DINOv2
│   │   └── vla_policy.py         # 6-layer GPT transformer
│   ├── datasets/
│   │   └── bridge_data_v2.py     # BridgeData V2 loader
│   ├── envs/
│   │   └── franka_tabletop.py    # Isaac Gym environment
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   └── losses.py             # Loss functions
│   ├── evaluation/
│   │   └── evaluator.py          # Evaluation metrics
│   ├── configs/
│   │   ├── resnet18.yaml         # ResNet-18 config
│   │   ├── vc1.yaml              # VC-1 config
│   │   └── dinov2.yaml           # DINOv2 config
│   └── utils/
│       ├── logging.py            # Metrics logging
│       └── metrics.py            # Metric computation
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── run_ablation.py           # Full ablation study
│   └── generate_synthetic_data.py # Synthetic data generator
├── experiments/                   # Checkpoints and logs
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

Each encoder has a YAML configuration file in `vla_encoder/configs/`:

### ResNet-18 Config
```yaml
vision_encoder: resnet18
freeze_vision_encoder: false  # Trainable
image_size: 224
# 25M params, ImageNet pretrained
```

### VC-1 Config
```yaml
vision_encoder: vc1
freeze_vision_encoder: true   # Frozen
image_size: 336
# 307M params, egocentric video pretrained
```

### DINOv2 Config
```yaml
vision_encoder: dinov2
freeze_vision_encoder: true   # Frozen
image_size: 224
# 86M params, self-supervised pretrained
```

## Training Details

**Optimizer:** AdamW
- Learning rate: 1e-4
- Weight decay: 0.01
- Batch size: 32

**Training Protocol:**
- Max epochs: 100
- Early stopping: patience=10 (validation plateau)
- Gradient clipping: norm=1.0
- LR scheduling: ReduceLROnPlateau

**Loss Function:**
- Continuous actions: MSE (or Huber)
- Gripper: Cross-entropy
- Weights: continuous=1.0, gripper=0.5

## Evaluation Metrics

### Success Criteria
- End-effector within 5cm of target pose
- Gripper state correct (open/closed)

### Metrics Tracked
1. **Task Success Rate:** Percentage of successful task completions
2. **Per-Task Success:** Individual success rates for each task
3. **OOD Performance:** Success on unseen objects
4. **Compute Profile:**
   - Peak GPU memory (MB)
   - Average inference time (ms)
   - Throughput (FPS)

## Results Format

Ablation results are saved as JSON:

```json
{
  "encoders": {
    "resnet18": {
      "0.25": {
        "evaluation": {
          "standard_eval": {
            "success_rate": 0.65,
            "task_success_rates": {...}
          },
          "ood_eval": {
            "ood_success": 0.42
          },
          "compute_profile": {
            "peak_memory_mb": 4500,
            "avg_inference_time_ms": 12.5
          }
        }
      }
    }
  }
}
```

## Model Parameters

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ResNet-18 Encoder | 25M | Yes |
| VC-1 Encoder | 307M | No (frozen) |
| DINOv2 Encoder | 86M | No (frozen) |
| T5-small Text Encoder | 60M | No (frozen) |
| GPT Transformer | 50M | Yes |
| **Total (ResNet-18)** | **~135M** | **~75M** |
| **Total (VC-1)** | **~417M** | **~50M** |
| **Total (DINOv2)** | **~196M** | **~50M** |

## Troubleshooting

### Isaac Gym Not Available
The code includes mock implementations for testing without Isaac Gym. For full evaluation:
1. Install Isaac Gym Preview 4
2. Set `PYTHONPATH` to include Isaac Gym

### VC-1 Model Loading Issues
If VC-1 is unavailable, the code falls back to a ViT-L model from `timm`:
```bash
pip install timm
```

### CUDA Out of Memory
Reduce batch size in config:
```yaml
batch_size: 16  # or smaller
```

## Citation

If you use this code, please cite:

```bibtex
@article{vla_encoder_ablation,
  title={Vision Encoder Ablation Study for Vision-Language-Action Policies},
  author={VLA Research Team},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- OpenVLA architecture inspiration
- BridgeData V2 dataset
- VC-1 pretrained models (Facebook AI)
- DINOv2 (Meta AI)
- Isaac Gym (NVIDIA)