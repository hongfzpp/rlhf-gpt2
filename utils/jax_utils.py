"""JAX utility functions: tree helpers, RNG management, dtype casting.

Small, reusable building blocks used throughout the RLHF pipeline.
"""

from __future__ import annotations

from typing import Any, Optional
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Device / backend verification
# ---------------------------------------------------------------------------

def check_backend():
    """Print JAX backend info for verification.

    On Apple M4, this should show 'metal' or 'gpu'.
    """
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend : {backend}")
    print(f"JAX devices : {devices}")
    print(f"Device count: {len(devices)}")
    return backend


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def create_rng(seed: int = 0) -> jax.Array:
    """Create a JAX PRNG key from an integer seed.

    JAX uses an explicit PRNG system (no global state), which guarantees
    reproducibility — critical for RL experiments where you need to
    replicate exact rollout sequences.

    Args:
        seed: Integer seed.

    Returns:
        A JAX PRNGKey array.
    """
    return jax.random.PRNGKey(seed)


def split_rng(rng: jax.Array, n: int = 2):
    """Split a PRNG key into n new keys.

    Args:
        rng: Current PRNG key.
        n: Number of new keys to produce.

    Returns:
        Tuple of n new PRNG keys if n > 1, or a single key if n == 1.
    """
    keys = jax.random.split(rng, n)
    if n == 1:
        return keys[0]
    return keys


# ---------------------------------------------------------------------------
# Parameter tree utilities
# ---------------------------------------------------------------------------

def count_params(params: Any) -> int:
    """Count total number of parameters in a pytree.

    Args:
        params: A JAX pytree (e.g., Flax model parameters).

    Returns:
        Total number of scalar parameters.
    """
    return sum(p.size for p in jax.tree.leaves(params))


def tree_dtype_cast(tree: Any, dtype: jnp.dtype) -> Any:
    """Cast all arrays in a pytree to a given dtype.

    Useful for mixed-precision: cast params to float16 before forward pass,
    keep float32 master copy for optimizer.

    Args:
        tree: A JAX pytree.
        dtype: Target dtype (e.g., jnp.float16).

    Returns:
        New pytree with all arrays cast to the given dtype.
    """
    return jax.tree.map(lambda x: x.astype(dtype) if hasattr(x, 'astype') else x, tree)


def clone_params(params: Any) -> Any:
    """Create a frozen copy of model parameters (for reference models in DPO/PPO).

    In JAX's functional paradigm, parameters are just pytrees. 'Freezing' them
    simply means keeping a separate copy and never passing it to the optimizer.
    No special .eval() mode or requires_grad=False needed — this is one of JAX's
    advantages for RLHF where you constantly need reference model copies.

    Args:
        params: Model parameters pytree.

    Returns:
        A detached copy of the parameters.
    """
    return jax.tree.map(lambda x: jnp.array(x), params)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def create_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Create a causal (lower-triangular) attention mask.

    Args:
        seq_len: Sequence length.
        dtype: Output dtype.

    Returns:
        Boolean mask of shape (1, 1, seq_len, seq_len) where True means 'attend'.
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    return mask[None, None, :, :]  # (1, 1, seq_len, seq_len) for broadcasting


def combine_masks(
    causal_mask: jax.Array,
    attention_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Combine causal mask with padding attention mask.

    Args:
        causal_mask: Shape (1, 1, seq_len, seq_len).
        attention_mask: Shape (batch_size, seq_len), 1 for real tokens, 0 for padding.

    Returns:
        Combined mask of shape (batch_size, 1, seq_len, seq_len).
    """
    if attention_mask is None:
        return causal_mask
    # Expand attention_mask: (batch, seq) -> (batch, 1, 1, seq)
    pad_mask = attention_mask[:, None, None, :]
    return causal_mask * pad_mask
