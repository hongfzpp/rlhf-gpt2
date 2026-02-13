"""Reward model architecture for RLHF.

Implements a scalar reward model by reusing the GPT-2 backbone and replacing
the language modeling head with a Dense(1) projection. The last non-padding
token's hidden state serves as the sequence representation (following InstructGPT).

Architecture:
    input_ids -> GPT-2 backbone -> last token hidden state -> Dense(1) -> scalar reward

References:
    - Ouyang et al., 2022, "Training language models to follow instructions
      with human feedback" (InstructGPT)
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.gpt2 import GPT2LMHeadModel


class RewardModel(nn.Module):
    """Reward model: GPT-2 backbone + scalar reward head.

    Attributes:
        config: Model configuration (shared with GPT-2 backbone).
    """
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute scalar rewards for input sequences.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len).
            attention_mask: Padding mask, shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding.
            deterministic: If True, disable dropout.

        Returns:
            Scalar rewards, shape (batch_size,).
        """

        backbone = GPT2LMHeadModel(config=self.config, name='backbone')
        hidden_states = backbone.get_hidden_states(input_ids, attention_mask=attention_mask, deterministic=deterministic)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(axis=-1).astype(jnp.int32) - 1
        else:
            seq_lengths = jnp.full((input_ids.shape[0],), input_ids.shape[1] - 1, dtype=jnp.int32)
        
        batch_indices = jnp.arange(input_ids.shape[0])
        last_hidden = hidden_states[batch_indices, seq_lengths, :]

        reward = nn.Dense(features=1, name='reward_head')(last_hidden)
        reward = reward.squeeze(-1)

        return reward
