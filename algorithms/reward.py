"""Reward model training via Bradley-Terry preference learning.

The trained reward model provides the reward signal for PPO and GRPO.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from models.reward_model import RewardModel
from configs.model_config import ModelConfig


# ============================================================================
# Bradley-Terry preference loss
# ============================================================================
#
# L = -(1/N) * sum_i log_sigmoid(r_chosen_i - r_rejected_i)
# accuracy = mean(r_chosen > r_rejected)
# ============================================================================

def preference_loss(
    chosen_rewards: jax.Array,
    rejected_rewards: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the Bradley-Terry preference loss.

    Args:
        chosen_rewards: Scalar rewards for chosen responses, shape (batch_size,).
        rejected_rewards: Scalar rewards for rejected responses, shape (batch_size,).

    Returns:
        Tuple of:
            - Scalar loss value.
            - Dictionary of metrics: {'accuracy': ..., 'reward_margin': ...}
    """
    
    reward_diff = chosen_rewards - rejected_rewards
    loss = - jax.nn.log_sigmoid(reward_diff).mean()

    accuracy = (chosen_rewards > rejected_rewards).astype(jnp.float32).mean()
    reward_margin = reward_diff.mean()

    return loss, {"accuracy": accuracy, "reward_margin": reward_margin}


# ============================================================================
# Reward model training state and training step
# ============================================================================

def create_reward_train_state(
    reward_model: RewardModel,
    config: ModelConfig,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize reward model parameters and optimizer.

    Args:
        reward_model: RewardModel instance.
        config: Model config.
        learning_rate: Learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping norm.
        rng: PRNG key.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = reward_model.init(rng, dummy_input)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def reward_train_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    reward_model: RewardModel,
) -> Tuple[Any, Any, jax.Array, Dict[str, jax.Array]]:
    """Perform one reward model training step.

    Args:
        params: Reward model parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'chosen_input_ids': (batch_size, seq_len)
            - 'chosen_attention_mask': (batch_size, seq_len)
            - 'rejected_input_ids': (batch_size, seq_len)
            - 'rejected_attention_mask': (batch_size, seq_len)
        reward_model: RewardModel instance.

    Returns:
        Tuple of (new_params, new_opt_state, loss, metrics_dict).
    """
    def loss_fn(params):
        chosen_rewards = reward_model.apply(params, batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask'], deterministic=False)
        rejected_rewards = reward_model.apply(params, batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask'], deterministic=False)
        loss, metrics = preference_loss(chosen_rewards, rejected_rewards)

        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, metrics
