"""Multi-head causal self-attention for GPT-2.

Implements scaled dot-product attention with a causal mask as a Flax module.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - Flax docs: https://flax.readthedocs.io/en/latest/
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.

    Attributes:
        config: Model configuration (n_heads, d_model, dropout_rate).
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None, deterministic: bool = True) -> jax.Array:
        """Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
                  or (1, 1, seq_len, seq_len). Values of 0 are masked out.
            deterministic: If True, disable dropout (use during eval).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config
        B, T, C = x.shape
        assert C == cfg.d_model, f"Input dim {C} != d_model {cfg.d_model}"
        head_dim = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        # Project input to Q, K, V
        q = nn.Dense(features=cfg.d_model, name='q_proj')(x)   # (B, T, d_model)
        k = nn.Dense(features=cfg.d_model, name='k_proj')(x)
        v = nn.Dense(features=cfg.d_model, name='v_proj')(x)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        atten_weights = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

        # Apply causal mask
        casual = jnp.tril(jnp.ones((T, T)))
        atten_weights = jnp.where(casual[None, None, :, :] == 0, -1e9, atten_weights)
        if mask is not None:
            atten_weights = jnp.where(mask == 0, -1e9, atten_weights)

        # Softmax and dropout
        atten_weights = jax.nn.softmax(atten_weights, axis = -1)
        atten_weights = nn.Dropout(rate=cfg.dropout_rate)(atten_weights, deterministic=deterministic)

        # Attend to values
        atten_output = atten_weights @ v

        # Concatenate heads and project output
        atten_output = atten_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = nn.Dense(features=cfg.d_model, name='out_proj')(atten_output)
        return output

