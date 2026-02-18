"""Autoregressive text generation utilities.

Provides top-k and nucleus (top-p) sampling for generating responses
during PPO rollouts and evaluation. Uses jax.lax.while_loop for full
XLA compilation with early stopping on EOS.

``generate_with_cache`` is a KV-cache–accelerated variant of ``generate``
that avoids redundant computation by caching key/value tensors across steps.
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

    Uses jax.lax.while_loop so the entire generation loop is compiled
    into a single XLA program (no Python-level per-step overhead).

    Args:
        apply_fn: Model apply function: (params, input_ids) -> logits.
                  logits should have shape (batch_size, seq_len, vocab_size).
                  Must use causal masking so future pad positions don't
                  affect logits at earlier positions.
        params: Model parameters pytree.
        input_ids: Initial token IDs, shape (batch_size, prompt_len).
        rng: PRNG key.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Number of top tokens for top-k sampling.
        eos_token_id: If set, stop generation when all sequences produce EOS.

    Returns:
        Generated token IDs, shape (batch_size, prompt_len + max_new_tokens).
        Positions after EOS are padded with 0.
    """
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + max_new_tokens

    # Pre-allocate fixed-size output buffer (required by XLA)
    tokens = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
    tokens = tokens.at[:, :prompt_len].set(input_ids)

    # Per-sequence finished flag
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)

    # Sentinel that never matches any real token when eos_token_id is None
    eos_id = jnp.array(
        eos_token_id if eos_token_id is not None else -1, dtype=jnp.int32
    )

    # Carry: (step, tokens, rng, finished)
    init_carry = (jnp.array(0, dtype=jnp.int32), tokens, rng, finished)

    def cond_fn(carry):
        step, _, _, finished = carry
        return (step < max_new_tokens) & (~jnp.all(finished))

    def body_fn(carry):
        step, tokens, rng, finished = carry

        # Forward pass over the full buffer; causal mask ensures that
        # logits at position (cur_pos - 1) only depend on tokens 0..cur_pos-1
        logits = apply_fn(params, tokens)           # (batch, total_len, vocab)
        cur_pos = prompt_len + step                  # position to write
        next_logits = logits[:, cur_pos - 1, :]      # (batch, vocab)

        # Sample next tokens
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, batch_size)

        next_tokens = jax.vmap(
            lambda lg, r: top_k_sampling(lg, r, k=top_k, temperature=temperature)
        )(next_logits, step_rngs)                    # (batch,)

        # Pad finished sequences with 0 instead of continuing to generate
        next_tokens = jnp.where(finished, 0, next_tokens)
        tokens = tokens.at[:, cur_pos].set(next_tokens)

        # Update finished flags
        newly_finished = next_tokens == eos_id
        finished = finished | newly_finished

        return (step + 1, tokens, rng, finished)

    _, tokens, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    return tokens


def generate_with_cache(
    apply_fn: Callable,
    params,
    input_ids: jax.Array,
    rng: jax.Array,
    init_cache_fn: Callable[[int], list[dict]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
) -> jax.Array:
    """Autoregressive generation with KV-cache acceleration.

    Instead of reprocessing the full sequence at every step, this function
    caches key/value tensors and only feeds the single newest token at each
    decode step.

    Uses jax.lax.while_loop so the entire generation loop (including cache
    updates) is compiled into a single XLA program.

    Args:
        apply_fn: Cached model apply function with signature
                  ``(params, input_ids, cache) -> (logits, updated_cache)``.
                  ``input_ids`` may have seq_len > 1 (prefill) or 1 (decode).
        params: Model parameters pytree.
        input_ids: Initial token IDs, shape (batch_size, prompt_len).
        rng: PRNG key.
        init_cache_fn: Callable that takes ``batch_size`` and returns the
                       initial empty cache (e.g. ``model.init_cache``).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Number of top tokens for top-k sampling.
        eos_token_id: If set, stop generation when all sequences produce EOS.

    Returns:
        Generated token IDs, shape (batch_size, prompt_len + max_new_tokens).
        Positions after EOS are padded with 0.
    """
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + max_new_tokens

    # Pre-allocate fixed-size output buffer (required by XLA)
    tokens = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
    tokens = tokens.at[:, :prompt_len].set(input_ids)

    # Per-sequence finished flag
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)

    # Sentinel that never matches any real token when eos_token_id is None
    eos_id = jnp.array(
        eos_token_id if eos_token_id is not None else -1, dtype=jnp.int32
    )

    # --- Prefill: process the entire prompt in one forward pass -----------
    cache = init_cache_fn(batch_size)
    prefill_logits, cache = apply_fn(params, input_ids, cache)

    # Logits at the last prompt position predict the first new token
    cur_logits = prefill_logits[:, -1, :]  # (batch, vocab)

    # Carry: (step, tokens, rng, finished, cache, cur_logits)
    init_carry = (jnp.array(0, dtype=jnp.int32), tokens, rng, finished, cache, cur_logits)

    def cond_fn(carry):
        step, _, _, finished, _, _ = carry
        return (step < max_new_tokens) & (~jnp.all(finished))

    def body_fn(carry):
        step, tokens, rng, finished, cache, cur_logits = carry
        cur_pos = prompt_len + step

        # Sample next token from cur_logits
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, batch_size)

        next_tokens = jax.vmap(
            lambda lg, r: top_k_sampling(lg, r, k=top_k, temperature=temperature)
        )(cur_logits, step_rngs)  # (batch,)

        # Pad finished sequences with 0
        next_tokens = jnp.where(finished, 0, next_tokens)
        tokens = tokens.at[:, cur_pos].set(next_tokens)

        # Update finished flags
        newly_finished = next_tokens == eos_id
        finished = finished | newly_finished

        # Feed the new token to get logits for the next step
        new_token_ids = next_tokens[:, None].astype(jnp.int32)  # (batch, 1)
        next_logits, cache = apply_fn(params, new_token_ids, cache)
        cur_logits = next_logits[:, 0, :]  # (batch, vocab)

        return (step + 1, tokens, rng, finished, cache, cur_logits)

    _, tokens, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    return tokens
