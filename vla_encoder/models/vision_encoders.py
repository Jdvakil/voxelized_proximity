"""Vision encoder implementations for VLA ablation study.

Implements three vision encoders as drop-in replacements:
1. ResNet-18 (ImageNet-pretrained, 25M params, fully trainable)
2. VC-1 Large (egocentric video-pretrained ViT-L/336px, 307M params, frozen)
3. DINOv2 ViT-B/14 (self-supervised, 86M params, frozen)

All encoders output patch tokens that are projected to 768-dim embedding space
via a 2-layer MLP before being consumed by the policy transformer.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class VisionEncoderBase(nn.Module):
    """Base class for vision encoders with common projection head."""

    def __init__(
        self,
        output_dim: int = 768,
        num_patches: int = 256,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.freeze_backbone = freeze_backbone

    def _build_projection_head(self, input_dim: int) -> nn.Module:
        """2-layer MLP projection head to map encoder features to 768-dim."""
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning (B, num_patches, output_dim) patch tokens."""
        raise NotImplementedError


class ResNet18Encoder(VisionEncoderBase):
    """ResNet-18 encoder with ImageNet pretraining (25M params, trainable).

    Extracts spatial feature map from conv5 (7x7 grid) and flattens to 49 patches.
    Each patch is projected from 512-dim to 768-dim via 2-layer MLP.
    """

    def __init__(
        self,
        output_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__(output_dim=output_dim, num_patches=49, freeze_backbone=freeze_backbone)

        # Load pretrained ResNet-18 and remove FC layers
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ResNet-18 conv5 outputs 512 channels at 7x7 spatial resolution
        self.feature_dim = 512
        self.spatial_size = 7

        # 2-layer MLP projection head
        self.projection = self._build_projection_head(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            Patch tokens (B, 49, 768)
        """
        B = x.size(0)

        # Extract spatial features (B, 512, 7, 7)
        features = self.backbone(x)

        # Flatten spatial dimensions to patches (B, 512, 49) -> (B, 49, 512)
        patches = features.flatten(2).transpose(1, 2)

        # Project to output_dim (B, 49, 768)
        output = self.projection(patches)

        return output

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class VC1Encoder(VisionEncoderBase):
    """VC-1 Large encoder (egocentric video-pretrained ViT-L/336px, 307M params, frozen).

    Uses a frozen Vision Transformer pretrained on egocentric videos.
    Outputs 24x24=576 patch tokens at 1024-dim, projected to 768-dim.
    """

    def __init__(
        self,
        output_dim: int = 768,
        model_name: str = "vc1-large",
        image_size: int = 336,
        freeze_backbone: bool = True
    ):
        # ViT-L/14 at 336px gives 24x24 patches
        num_patches = (image_size // 14) ** 2
        super().__init__(output_dim=output_dim, num_patches=num_patches, freeze_backbone=freeze_backbone)

        self.image_size = image_size

        # Try to load VC-1 model (requires vc_models package)
        # If not available, use a ViT-L placeholder
        try:
            import vc_models
            self.backbone = vc_models.load_model(model_name)
            self.feature_dim = 1024  # ViT-L hidden dim
        except ImportError:
            print("Warning: vc_models not available, using placeholder ViT-L")
            self.backbone = self._create_vit_placeholder()
            self.feature_dim = 1024

        # Freeze backbone (VC-1 should always be frozen)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2-layer MLP projection head
        self.projection = self._build_projection_head(self.feature_dim)

    def _create_vit_placeholder(self) -> nn.Module:
        """Creates a ViT-L placeholder if vc_models unavailable."""
        try:
            import timm
            model = timm.create_model(
                'vit_large_patch14_clip_336.openai',
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            return model
        except ImportError:
            raise ImportError(
                "Neither vc_models nor timm available. Install with:\n"
                "pip install vc-models timm"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 336, 336)

        Returns:
            Patch tokens (B, 576, 768)
        """
        # Resize if needed
        if x.size(-1) != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

        with torch.set_grad_enabled(not self.freeze_backbone):
            # Extract patch tokens (B, num_patches, 1024)
            # Different models may have different output formats
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
            else:
                features = self.backbone(x)

            # Handle different output formats
            if isinstance(features, dict):
                # vc_models output format
                features = features.get('patch_tokens', features.get('x', features))
            elif features.dim() == 3 and features.size(1) > self.num_patches:
                # Has CLS token, remove it
                features = features[:, 1:, :]

        # Project to output_dim (B, num_patches, 768)
        output = self.projection(features)

        return output

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class DINOv2Encoder(VisionEncoderBase):
    """DINOv2 ViT-B/14 encoder (self-supervised, 86M params, frozen).

    Uses a frozen DINOv2 ViT-B/14 model pretrained with self-supervised learning.
    Outputs 16x16=256 patch tokens at 768-dim (already at target dimension).
    """

    def __init__(
        self,
        output_dim: int = 768,
        model_name: str = "dinov2_vitb14",
        image_size: int = 224,
        freeze_backbone: bool = True
    ):
        # ViT-B/14 at 224px gives 16x16 patches
        num_patches = (image_size // 14) ** 2
        super().__init__(output_dim=output_dim, num_patches=num_patches, freeze_backbone=freeze_backbone)

        self.image_size = image_size

        # Load DINOv2 model from torch.hub
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            self.feature_dim = 768  # ViT-B hidden dim
        except Exception as e:
            print(f"Warning: Could not load DINOv2 from hub: {e}")
            print("Using timm as fallback")
            import timm
            self.backbone = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=0,
            )
            self.feature_dim = 768

        # Freeze backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2-layer MLP projection head (DINOv2 already outputs 768-dim but we still project)
        self.projection = self._build_projection_head(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            Patch tokens (B, 256, 768)
        """
        # Resize if needed
        if x.size(-1) != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

        with torch.set_grad_enabled(not self.freeze_backbone):
            # Extract patch tokens
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
            elif hasattr(self.backbone, 'get_intermediate_layers'):
                # DINOv2 specific API
                features = self.backbone.get_intermediate_layers(x, n=1)[0]
            else:
                features = self.backbone(x)

            # Remove CLS token if present (B, 257, 768) -> (B, 256, 768)
            if features.size(1) == self.num_patches + 1:
                features = features[:, 1:, :]

        # Project to output_dim (B, 256, 768)
        output = self.projection(features)

        return output

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_vision_encoder(encoder_type: str, **kwargs) -> VisionEncoderBase:
    """Factory function to create vision encoders.

    Args:
        encoder_type: One of ['resnet18', 'vc1', 'dinov2']
        **kwargs: Additional arguments passed to encoder constructor

    Returns:
        Vision encoder instance
    """
    encoders = {
        'resnet18': ResNet18Encoder,
        'vc1': VC1Encoder,
        'dinov2': DINOv2Encoder,
    }

    if encoder_type.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Choose from {list(encoders.keys())}"
        )

    return encoders[encoder_type.lower()](**kwargs)
