"""Autoregressive text generation utilities.

Provides top-k and nucleus (top-p) sampling for generating responses
during PPO rollouts and evaluation. Uses a Python loop over timesteps
for clarity; a production implementation would use jax.lax.while_loop
for full XLA compilation.
"""

from __future__ import annotations

from typing import Optional, Callable
import jax
import jax.numpy as jnp
import functools


def top_k_sampling(
    logits: jax.Array,
    rng: jax.Array,
    k: int = 50,
    temperature: float = 1.0,
) -> jax.Array:
    """Sample from the top-k logits.

    Args:
        logits: Shape (vocab_size,) — raw logits for one position.
        rng: PRNG key for sampling.
        k: Number of top tokens to consider.
        temperature: Sampling temperature (lower = more greedy).

    Returns:
        Sampled token ID as a scalar integer array.
    """
    logits = logits / temperature

    # Get top-k values and indices
    top_k_values, top_k_indices = jax.lax.top_k(logits, k)

    # Sample from top-k distribution
    sampled_idx = jax.random.categorical(rng, top_k_values)
    return top_k_indices[sampled_idx]


def nucleus_sampling(
    logits: jax.Array,
    rng: jax.Array,
    p: float = 0.9,
    temperature: float = 1.0,
) -> jax.Array:
    """Nucleus (top-p) sampling.

    Args:
        logits: Shape (vocab_size,) — raw logits for one position.
        rng: PRNG key for sampling.
        p: Cumulative probability threshold.
        temperature: Sampling temperature.

    Returns:
        Sampled token ID as a scalar integer array.
    """
    logits = logits / temperature
    probs = jax.nn.softmax(logits)

    # Sort probabilities in descending order
    sorted_indices = jnp.argsort(-probs)
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities
    cumulative_probs = jnp.cumsum(sorted_probs)

    # Mask tokens beyond the nucleus
    mask = cumulative_probs - sorted_probs <= p  # Keep first token that exceeds p
    masked_probs = jnp.where(mask, sorted_probs, 0.0)
    masked_probs = masked_probs / masked_probs.sum()

    # Sample from the nucleus
    sampled_idx = jax.random.categorical(rng, jnp.log(masked_probs + 1e-10))
    return sorted_indices[sampled_idx]


def generate(
    apply_fn: Callable,
    params,
    input_ids: jax.Array,
    rng: jax.Array,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
) -> jax.Array:
    """Autoregressive generation using top-k sampling.

    This is a simple Python-loop implementation for clarity. In production,
    you'd use jax.lax.while_loop for full XLA compilation.

    Args:
        apply_fn: Model apply function: (params, input_ids) -> logits.
                  logits should have shape (batch_size, seq_len, vocab_size).
        params: Model parameters pytree.
        input_ids: Initial token IDs, shape (batch_size, prompt_len).
        rng: PRNG key.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Number of top tokens for top-k sampling.
        eos_token_id: If set, stop generation when EOS is produced.

    Returns:
        Generated token IDs, shape (batch_size, prompt_len + num_generated).
    """
    batch_size = input_ids.shape[0]
    generated = input_ids

    for _ in range(max_new_tokens):
        # Forward pass — get logits for the last position
        logits = apply_fn(params, generated)         # (batch, seq, vocab)
        next_logits = logits[:, -1, :]               # (batch, vocab)

        # Sample next token for each sequence in the batch
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, batch_size)

        next_tokens = jax.vmap(
            lambda lg, r: top_k_sampling(lg, r, k=top_k, temperature=temperature)
        )(next_logits, step_rngs)

        next_tokens = next_tokens[:, None]           # (batch, 1)
        generated = jnp.concatenate([generated, next_tokens], axis=1)

        # Check for EOS (simple — stops all sequences when any produces EOS)
        if eos_token_id is not None:
            if jnp.any(next_tokens == eos_token_id):
                break

    return generated
