"""Policy wrapper: log-probability computation and sampling utilities.

Wraps the GPT-2 model with RLHF-specific helper functions:
- Computing log-probabilities of given sequences (needed for PPO ratios, DPO)
- Computing entropy (needed for PPO entropy bonus)
- Sampling new responses given prompts
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp


def compute_log_probs(
    logits: jax.Array,
    labels: jax.Array,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute per-token log-probabilities of labels under the model.

    This is a core utility used by:
    - SFT loss (Phase 3)
    - PPO policy ratios (Phase 5)
    - DPO log-probability ratios (Phase 6)

    Args:
        logits: Model output logits, shape (batch_size, seq_len, vocab_size).
        labels: Target token IDs, shape (batch_size, seq_len).
        mask: Optional mask, shape (batch_size, seq_len). 1 for valid, 0 for ignore.

    Returns:
        Per-token log-probs, shape (batch_size, seq_len). Masked positions are 0.
    """
    # Shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_labels = labels[:, 1:]       # (B, T-1)

    # Log-softmax for numerical stability
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)  # (B, T-1, V)

    # Gather log-probs of the actual tokens
    token_log_probs = jnp.take_along_axis(
        log_probs,
        shift_labels[:, :, None],
        axis=-1,
    ).squeeze(-1)  # (B, T-1)

    # Apply mask if provided
    if mask is not None:
        shift_mask = mask[:, 1:]
        token_log_probs = token_log_probs * shift_mask

    return token_log_probs


def compute_sequence_log_probs(
    logits: jax.Array,
    labels: jax.Array,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute total sequence log-probability (sum of per-token log-probs).

    Used in DPO to compute log pi(y|x).

    Args:
        logits: Shape (batch_size, seq_len, vocab_size).
        labels: Shape (batch_size, seq_len).
        mask: Shape (batch_size, seq_len).

    Returns:
        Sequence log-probs, shape (batch_size,).
    """
    token_log_probs = compute_log_probs(logits, labels, mask)
    return token_log_probs.sum(axis=-1)


def compute_entropy(logits: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
    """Compute per-token entropy of the model's output distribution.

    Used in PPO as an entropy bonus to encourage exploration.

    Args:
        logits: Shape (batch_size, seq_len, vocab_size).
        mask: Shape (batch_size, seq_len).

    Returns:
        Mean entropy (scalar).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)  # (B, T)

    if mask is not None:
        entropy = entropy * mask
        return entropy.sum() / jnp.maximum(mask.sum(), 1.0)
    return entropy.mean()
