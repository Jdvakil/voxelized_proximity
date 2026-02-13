"""VLA Policy: Vision-Language-Action transformer for robotic manipulation.

6-layer GPT-style decoder-only transformer (50M total params) that:
1. Consumes concatenated language tokens from frozen T5-small text encoder
2. Consumes vision patch tokens from vision encoder (projected to 768-dim)
3. Autoregressively predicts 7-DoF action chunks over 8-step horizons
   (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import T5EncoderModel, T5Tokenizer
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1), :]


class GPTBlock(nn.Module):
    """Single GPT-style transformer decoder block with causal self-attention."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, seq_len, d_model)
            attn_mask: Attention mask (seq_len, seq_len)
            is_causal: Whether to use causal masking

        Returns:
            Output tensor (B, seq_len, d_model)
        """
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x


class ActionHead(nn.Module):
    """Action prediction head for 7-DoF delta actions with 8-step horizon.

    Outputs:
        - 6D continuous: Δx, Δy, Δz, Δroll, Δpitch, Δyaw (each in range [-1, 1])
        - 1D discrete: gripper state (open=0, close=1)
        - 8 timesteps per prediction
    """

    def __init__(
        self,
        d_model: int = 768,
        action_dim: int = 7,
        action_horizon: int = 8
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Predict all actions at once: 8 timesteps × 7 DoF = 56 values
        total_output_dim = action_horizon * action_dim

        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, total_output_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Transformer output (B, seq_len, d_model)

        Returns:
            continuous_actions: (B, action_horizon, 6) - delta pose
            gripper_logits: (B, action_horizon, 2) - gripper state logits
        """
        # Take last token as action prediction token
        last_token = x[:, -1, :]  # (B, d_model)

        # Predict all actions (B, action_horizon * action_dim)
        actions = self.head(last_token)

        # Reshape to (B, action_horizon, action_dim)
        actions = actions.view(-1, self.action_horizon, self.action_dim)

        # Split into continuous (first 6 dims) and gripper (last dim)
        continuous_actions = torch.tanh(actions[..., :6])  # (B, H, 6) in [-1, 1]

        # Gripper: convert last dim to binary logits
        gripper_raw = actions[..., 6]  # (B, H)
        gripper_logits = torch.stack([-gripper_raw, gripper_raw], dim=-1)  # (B, H, 2)

        return continuous_actions, gripper_logits


class VLAPolicy(nn.Module):
    """Vision-Language-Action policy with 6-layer GPT-style transformer.

    Architecture:
        1. Frozen T5-small language encoder (60M params)
        2. Vision encoder (ResNet-18/VC-1/DINOv2)
        3. 6-layer GPT transformer (50M params total)
        4. Action prediction head (7-DoF × 8 timesteps)

    Total model size: ~50M trainable params (excluding frozen encoders)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        d_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        action_dim: int = 7,
        action_horizon: int = 8,
        max_seq_len: int = 1024,
        use_action_token: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.action_horizon = action_horizon
        self.use_action_token = use_action_token

        # Frozen T5-small language encoder
        self.text_encoder = T5EncoderModel.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Vision encoder (drop-in replacement: ResNet-18/VC-1/DINOv2)
        self.vision_encoder = vision_encoder

        # Project T5 hidden states (512) to d_model (768)
        self.text_projection = nn.Linear(512, d_model)

        # Learnable action query token
        if self.use_action_token:
            self.action_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 6-layer GPT-style transformer
        self.transformer_blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Action prediction head
        self.action_head = ActionHead(d_model, action_dim, action_horizon)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def encode_text(self, text_prompts: list) -> torch.Tensor:
        """Encode text prompts using frozen T5-small.

        Args:
            text_prompts: List of text strings

        Returns:
            Text embeddings (B, text_seq_len, d_model)
        """
        # Tokenize
        encoded = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors='pt'
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Encode with frozen T5
        with torch.no_grad():
            outputs = self.text_encoder(**encoded)
            text_features = outputs.last_hidden_state  # (B, text_len, 512)

        # Project to d_model
        text_embeddings = self.text_projection(text_features)  # (B, text_len, 768)

        return text_embeddings

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: list,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images (B, 3, H, W)
            text_prompts: List of B text strings
            return_features: If True, return intermediate features

        Returns:
            Dictionary containing:
                - continuous_actions: (B, action_horizon, 6)
                - gripper_logits: (B, action_horizon, 2)
                - features: (B, seq_len, d_model) [if return_features=True]
        """
        B = images.size(0)

        # Encode vision: (B, num_patches, d_model)
        vision_tokens = self.vision_encoder(images)

        # Encode text: (B, text_len, d_model)
        text_tokens = self.encode_text(text_prompts)

        # Concatenate text + vision tokens
        # Sequence: [text tokens | vision tokens | action token (optional)]
        tokens = torch.cat([text_tokens, vision_tokens], dim=1)  # (B, text_len + num_patches, d_model)

        # Add learnable action query token if enabled
        if self.use_action_token:
            action_token = self.action_token.expand(B, -1, -1)  # (B, 1, d_model)
            tokens = torch.cat([tokens, action_token], dim=1)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)

        # Pass through transformer blocks (no causal masking for encoder-decoder style)
        x = tokens
        for block in self.transformer_blocks:
            x = block(x, is_causal=False)

        # Final layer norm
        x = self.ln_f(x)

        # Predict actions from last token
        continuous_actions, gripper_logits = self.action_head(x)

        output = {
            'continuous_actions': continuous_actions,
            'gripper_logits': gripper_logits,
        }

        if return_features:
            output['features'] = x

        return output

    def predict_action(
        self,
        image: torch.Tensor,
        text_prompt: str,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Generate action prediction for inference.

        Args:
            image: Single image (1, 3, H, W) or (3, H, W)
            text_prompt: Single text string
            temperature: Sampling temperature for gripper

        Returns:
            Dictionary with action predictions
        """
        self.eval()
        with torch.no_grad():
            # Handle single image input
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Forward pass
            output = self.forward(image, [text_prompt])

            # Sample gripper state
            gripper_probs = F.softmax(output['gripper_logits'] / temperature, dim=-1)
            gripper_actions = torch.argmax(gripper_probs, dim=-1)  # (1, H)

            return {
                'continuous_actions': output['continuous_actions'][0],  # (H, 6)
                'gripper_actions': gripper_actions[0],  # (H,)
                'gripper_probs': gripper_probs[0],  # (H, 2)
            }

    def get_num_params(self) -> Dict[str, int]:
        """Return parameter counts for different components."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        vision_total, vision_trainable = self.vision_encoder.get_num_params()
        text_total = sum(p.numel() for p in self.text_encoder.parameters())

        transformer_params = sum(
            p.numel() for block in self.transformer_blocks for p in block.parameters()
        )

        return {
            'total': total,
            'trainable': trainable,
            'vision_encoder_total': vision_total,
            'vision_encoder_trainable': vision_trainable,
            'text_encoder': text_total,
            'transformer': transformer_params,
        }


def create_vla_policy(
    vision_encoder_type: str,
    vision_encoder_kwargs: Optional[dict] = None,
    **policy_kwargs
) -> VLAPolicy:
    """Factory function to create VLA policy with specified vision encoder.

    Args:
        vision_encoder_type: One of ['resnet18', 'vc1', 'dinov2']
        vision_encoder_kwargs: Arguments for vision encoder
        **policy_kwargs: Arguments for VLA policy

    Returns:
        VLAPolicy instance
    """
    from .vision_encoders import create_vision_encoder

    if vision_encoder_kwargs is None:
        vision_encoder_kwargs = {}

    # Create vision encoder
    vision_encoder = create_vision_encoder(vision_encoder_type, **vision_encoder_kwargs)

    # Create policy
    policy = VLAPolicy(vision_encoder=vision_encoder, **policy_kwargs)

    return policy
