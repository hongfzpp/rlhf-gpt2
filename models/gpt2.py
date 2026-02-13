"""GPT-2 language model with weight-tied LM head.

Architecture:
    input_ids -> Token Embed + Pos Embed -> [TransformerBlock x N] -> LayerNorm -> LM Head -> logits

The LM head shares weights with the token embedding (weight tying).
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

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass: input token IDs -> logits over vocabulary.

        Args:
            input_ids: Integer token IDs of shape (batch_size, seq_len).
            attention_mask: Optional padding mask, shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding.
            deterministic: If True, disable dropout.

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
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
        causal_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            mask = causal_mask * pad_mask
        else:
            mask = causal_mask

        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(config=cfg, name=f'block_{i}')(x, mask=mask, deterministic=deterministic)

        # Final layer normalization
        x = nn.LayerNorm()(x)

        # Language model head (weight-tied with token embedding)
        logits = x @ token_embed.embedding.T
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
        token_embed = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.d_model, name = 'token_embed')
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
