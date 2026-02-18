"""GPT-2 language model with weight-tied LM head.

Architecture:
    input_ids -> Token Embed + Pos Embed -> [TransformerBlock x N] -> LayerNorm -> LM Head -> logits

The LM head shares weights with the token embedding (weight tying).

Supports an optional KV-cache (list of per-layer cache dicts) for efficient
autoregressive generation.  Use ``init_cache`` to create the initial empty
cache and pass it through ``__call__`` to get updated caches.
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.transformer_block import TransformerBlock


class GPT2LMHeadModel(nn.Module):
    """GPT-2 Language Model with tied embedding weights.

    Attributes:
        config: Model configuration.
    """
    config: ModelConfig

    def init_cache(self, batch_size: int) -> list[dict]:
        """Create pre-allocated empty KV-cache for all layers.

        Args:
            batch_size: Batch dimension for the cache buffers.

        Returns:
            List of ``n_layers`` cache dicts, each with keys
            ``'key'``, ``'value'`` (zero buffers of shape
            ``(batch_size, n_heads, max_seq_len, head_dim)``)
            and ``'index'`` (scalar 0).
        """
        cfg = self.config
        head_dim = cfg.d_model // cfg.n_heads

        def _make_layer_cache():
            return {
                'key': jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, head_dim)),
                'value': jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, head_dim)),
                'index': jnp.array(0, dtype=jnp.int32),
            }

        return [_make_layer_cache() for _ in range(cfg.n_layers)]

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        cache: Optional[list[dict]] = None,
    ) -> jax.Array | tuple[jax.Array, list[dict]]:
        """Forward pass: input token IDs -> logits over vocabulary.

        Args:
            input_ids: Integer token IDs of shape (batch_size, seq_len).
            attention_mask: Optional padding mask, shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding.
            deterministic: If True, disable dropout.
            cache: Optional list of per-layer KV-cache dicts (from
                   ``init_cache`` or a previous call).  When provided, the
                   position embedding is offset by ``cache[0]['index']`` and
                   causal masking is delegated to the attention layers.

        Returns:
            If cache is None:
                Logits of shape (batch_size, seq_len, vocab_size).
            If cache is provided:
                Tuple of (logits, updated_cache).
        """
        cfg = self.config
        B, T = input_ids.shape

        # Token and position embeddings
        token_embed = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.d_model, name='token_embed')
        pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, cfg.max_seq_len, cfg.d_model))

        x = token_embed(input_ids)

        if cache is not None:
            # Offset position embedding by the cache write position
            cache_index = cache[0]['index']
            pos = jax.lax.dynamic_slice(pos_embed, (0, cache_index, 0), (1, T, cfg.d_model))
            x = x + pos
        else:
            x = x + pos_embed[:, :T, :]

        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        if cache is not None:
            # Causal masking is handled inside the attention cache logic
            mask = None
        else:
            # Combined attention mask (causal + padding)
            causal_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
            if attention_mask is not None:
                pad_mask = attention_mask[:, None, None, :]
                mask = causal_mask * pad_mask
            else:
                mask = causal_mask

        # Transformer blocks
        updated_caches: list[dict] = []
        for i in range(cfg.n_layers):
            layer_cache = cache[i] if cache is not None else None
            block_result = TransformerBlock(config=cfg, name=f'block_{i}')(
                x, mask=mask, deterministic=deterministic, cache=layer_cache,
            )
            if layer_cache is not None:
                x, new_layer_cache = block_result
                updated_caches.append(new_layer_cache)
            else:
                x = block_result

        # Final layer normalization
        x = nn.LayerNorm()(x)

        # Language model head (weight-tied with token embedding)
        logits = x @ token_embed.embedding.T

        if cache is not None:
            return logits, updated_caches
        return logits

    @nn.compact
    def get_hidden_states(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Get hidden states (before LM head) — used by the reward model.

        This method reuses the same architecture but returns the output of
        the final LayerNorm instead of projecting to logits.

        Args:
            input_ids: Integer token IDs of shape (batch_size, seq_len).
            attention_mask: Optional padding mask.
            deterministic: If True, disable dropout.

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config
        B, T = input_ids.shape

        # Token and position embeddings
        token_embed = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.d_model, name='token_embed')
        pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, cfg.max_seq_len, cfg.d_model))

        x = token_embed(input_ids)
        x = x + pos_embed[:, :T, :]
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        # Combined attention mask (causal + padding)
        casual_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            mask = casual_mask * pad_mask
        else:
            mask = casual_mask

        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(config=cfg, name=f'block_{i}')(x, mask=mask, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        return x
