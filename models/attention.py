"""Multi-head causal self-attention for GPT-2.

Implements scaled dot-product attention with a causal mask as a Flax module.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

Supports an optional KV-cache for efficient autoregressive generation.
When a cache dict is provided, the module writes new K/V into pre-allocated
buffers and returns the updated cache alongside the output.

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - Flax docs: https://flax.readthedocs.io/en/latest/
"""

from __future__ import annotations

from typing import Optional

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
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
        cache: Optional[dict] = None,
    ) -> jax.Array | tuple[jax.Array, dict]:
        """Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
                  or (1, 1, seq_len, seq_len). Values of 0 are masked out.
            deterministic: If True, disable dropout (use during eval).
            cache: Optional KV-cache dict with keys:
                - 'key':   (batch_size, n_heads, max_len, head_dim) pre-allocated buffer.
                - 'value': (batch_size, n_heads, max_len, head_dim) pre-allocated buffer.
                - 'index': scalar int32, the next write position in the cache.
                When provided, new K/V are written at ``index`` and attention is
                computed against the full cache.  The updated cache is returned
                as the second element of a tuple.

        Returns:
            If cache is None:
                Output tensor of shape (batch_size, seq_len, d_model).
            If cache is provided:
                Tuple of (output, updated_cache).
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

        updated_cache = None
        if cache is not None:
            cache_index = cache['index']

            # Write new K, V into cache using dynamic_update_slice (XLA-safe)
            cache_key = jax.lax.dynamic_update_slice(
                cache['key'], k, (0, 0, cache_index, 0)
            )
            cache_value = jax.lax.dynamic_update_slice(
                cache['value'], v, (0, 0, cache_index, 0)
            )
            updated_cache = {
                'key': cache_key,
                'value': cache_value,
                'index': cache_index + T,
            }

            # Attend against the full cached K, V
            k = cache_key   # (B, H, max_len, head_dim)
            v = cache_value  # (B, H, max_len, head_dim)
            KV_len = k.shape[2]

            # Causal mask: query at absolute pos (index+j) attends to key pos k <= index+j
            q_pos = jnp.arange(T) + cache_index
            k_pos = jnp.arange(KV_len)
            causal = (k_pos[None, :] <= q_pos[:, None]).astype(jnp.float32)
        else:
            # Standard lower-triangular causal mask
            causal = jnp.tril(jnp.ones((T, T)))

        # Scaled dot-product attention
        atten_weights = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

        # Apply causal mask
        atten_weights = jnp.where(causal[None, None, :, :] == 0, -1e9, atten_weights)
        if mask is not None:
            atten_weights = jnp.where(mask == 0, -1e9, atten_weights)

        # Softmax and dropout
        atten_weights = jax.nn.softmax(atten_weights, axis=-1)
        atten_weights = nn.Dropout(rate=cfg.dropout_rate)(atten_weights, deterministic=deterministic)

        # Attend to values
        atten_output = atten_weights @ v

        # Concatenate heads and project output
        atten_output = atten_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = nn.Dense(features=cfg.d_model, name='out_proj')(atten_output)

        if updated_cache is not None:
            return output, updated_cache
        return output

