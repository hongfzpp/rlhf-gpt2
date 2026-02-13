"""Transformer block: Pre-LN attention + feed-forward with residual connections.

Architecture (Pre-LN variant, used by GPT-2):
    x -> LayerNorm -> Attention -> + residual -> LayerNorm -> FFN -> + residual

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """A single Transformer block: Attention + FFN with residuals and LayerNorm.

    Attributes:
        config: Model configuration.
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None, deterministic: bool = True) -> jax.Array:
        """Apply one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.
            deterministic: If True, disable dropout.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config

        # Attention sub-layer with residual (Pre-LN)
        residual = x
        x = nn.LayerNorm()(x)
        x = CausalSelfAttention(config=cfg)(x, mask=mask, deterministic=deterministic)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        x = residual + x

        # Feed-forward sub-layer with residual
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=cfg.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=cfg.d_model)(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        x = residual + x

        return x
