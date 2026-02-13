"""Supervised Fine-Tuning (SFT) — Stage 1 of the RLHF pipeline.

SFT model can be used as:
1. Starting policy model for the later RLHF updates.
2. reference model with which we calculate the KL divergence to monitoring the policy model updates.
"""

from __future__ import annotations

from typing import Any, Tuple, Dict
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from configs.model_config import ModelConfig


# ============================================================================
# Cross-entropy loss for next-token prediction
# ============================================================================
# Actually for the one-hot targets, cross entropy loss is equivalent as 
# NLL loss.
#
# loss = -(1/N) * sum_{t where label_t != -100} log P(label_t | x_{<t})
#
# Labels are shifted by 1 relative to logits: logits[:, :-1, :] vs labels[:, 1:]
# ============================================================================

def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Compute cross-entropy loss for next-token prediction.

    Args:
        logits: Model output, shape (batch_size, seq_len, vocab_size).
        labels: Target token IDs, shape (batch_size, seq_len).
                Positions with value -100 are ignored (padding / prompt tokens).

    Returns:
        Scalar loss value (mean over non-ignored tokens).
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis = -1)

    token_log_probs = jnp.take_along_axis(log_probs, shift_labels[:, :, None], axis=-1).squeeze(-1)

    mask = shift_labels != -100
    token_log_probs = token_log_probs * mask

    loss = -token_log_probs.sum() / jnp.maximum(mask.sum(), 1.0)

    return loss


# ============================================================================
# SFT training state and JIT-compiled training step
# ============================================================================

def create_sft_train_state(
    model: GPT2LMHeadModel,
    config: ModelConfig,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize model parameters and optimizer for SFT.

    Args:
        model: GPT2LMHeadModel instance.
        config: Model config.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_steps: Linear warmup steps.
        rng: PRNG key for initialization.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Initialize model parameters
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    # Create Optax optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def sft_train_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
) -> Tuple[Any, Any, jax.Array]:
    """Perform one SFT training step.

    This function should be JIT-compiled for performance.

    Args:
        params: Model parameters pytree.
        opt_state: Optimizer state.
        optimizer: Optax optimizer (GradientTransformation).
        batch: Dictionary with:
            - 'input_ids': (batch_size, seq_len)
            - 'labels': (batch_size, seq_len), -100 for ignored positions
            - 'attention_mask': (batch_size, seq_len)
        model: GPT2LMHeadModel instance (for apply).

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value).
    """
    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'], attention_mask=batch['attention_mask'], deterministic=False)
        return cross_entropy_loss(logits, batch['labels'])

    loss, grad = jax.value_and_grad(loss_fn)(params)
    
    updates, new_opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


# ============================================================================
# SFT evaluation step
# ============================================================================

def sft_eval_step(
    params: Any,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
) -> jax.Array:
    """Compute SFT loss for evaluation (no gradient computation).

    Args:
        params: Model parameters.
        batch: Same format as sft_train_step.
        model: GPT2LMHeadModel instance.

    Returns:
        Scalar loss value.
    """
    logits = model.apply(params, batch['input_ids'], attention_mask=batch['attention_mask'], deterministic=True)
    return cross_entropy_loss(logits, batch['labels'])
