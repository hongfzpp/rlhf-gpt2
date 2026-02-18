"""Transformer block: Pre-LN attention + feed-forward with residual connections.

Architecture (Pre-LN variant, used by GPT-2):
    x -> LayerNorm -> Attention -> + residual -> LayerNorm -> FFN -> + residual

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

Supports an optional KV-cache dict that is threaded through the attention
sub-layer.  See ``CausalSelfAttention`` for the cache format.
"""

from __future__ import annotations

from typing import Optional

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
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
        cache: Optional[dict] = None,
    ) -> jax.Array | tuple[jax.Array, dict]:
        """Apply one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.
            deterministic: If True, disable dropout.
            cache: Optional KV-cache dict for this block's attention layer.
                   See ``CausalSelfAttention`` for the expected format.

        Returns:
            If cache is None:
                Output tensor of shape (batch_size, seq_len, d_model).
            If cache is provided:
                Tuple of (output, updated_cache).
        """
        cfg = self.config

        # Attention sub-layer with residual (Pre-LN)
        residual = x
        x = nn.LayerNorm()(x)
        attn_result = CausalSelfAttention(config=cfg)(
            x, mask=mask, deterministic=deterministic, cache=cache,
        )

        updated_cache = None
        if cache is not None:
            x, updated_cache = attn_result
        else:
            x = attn_result

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

        if updated_cache is not None:
            return x, updated_cache
        return x
