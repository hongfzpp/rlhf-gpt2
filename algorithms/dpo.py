"""Direct Preference Optimization (DPO) for RLHF.

DPO directly optimizes the policy on preference pairs (chosen, rejected) without
training a separate reward model. The policy itself acts as an implicit reward
model via the reparameterization: r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x)).

    L_DPO = -E[log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))]
    where log_ratio = log pi_theta(y|x) - log pi_ref(y|x)

References:
    - Rafailov et al., 2023, "Direct Preference Optimization: Your Language Model
      is Secretly a Reward Model"
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
from models import attention
import optax

from models.gpt2 import GPT2LMHeadModel
from models.policy import compute_log_probs, compute_sequence_log_probs
from configs.model_config import ModelConfig


# ============================================================================
# Sequence log-probabilities for DPO
# ============================================================================
#
# Computes log pi(y|x) = sum_{t in response} log pi(y_t | x, y_{<t}).
# Unlike the SFT loss (average over all tokens), DPO requires the SUM of
# log-probs over response tokens for comparing chosen vs. rejected sequences.
# ============================================================================

def compute_response_log_probs(
    model: GPT2LMHeadModel,
    params: Any,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    response_mask: jax.Array,
) -> jax.Array:
    """Compute log pi(response | prompt) for each sequence in the batch.

    Args:
        model: GPT2LMHeadModel instance.
        params: Model parameters.
        input_ids: Full sequence (prompt + response), shape (batch_size, seq_len).
        attention_mask: Padding mask, shape (batch_size, seq_len).
        response_mask: 1 for response tokens, 0 for prompt/padding.
                      Shape (batch_size, seq_len).

    Returns:
        Sequence log-probs, shape (batch_size,) — sum of log-probs over response tokens.
    """
    logits = model.apply(params, input_ids, attention_mask=attention_mask, deterministic=True)
    token_log_probs = compute_log_probs(logits, input_ids, response_mask)
    seq_log_probs = token_log_probs.sum(axis=-1)

    return seq_log_probs.squeeze()


# ============================================================================
# DPO loss
# ============================================================================
#
# log_ratio_w = log pi_theta(y_w | x) - log pi_ref(y_w | x)
# log_ratio_l = log pi_theta(y_l | x) - log pi_ref(y_l | x)
# L = -mean[log sigmoid(beta * (log_ratio_w - log_ratio_l))]
#
# Implicit reward: r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
# ============================================================================

def dpo_loss(
    policy_chosen_logps: jax.Array,
    policy_rejected_logps: jax.Array,
    ref_chosen_logps: jax.Array,
    ref_rejected_logps: jax.Array,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: log pi_theta(y_w | x), shape (batch_size,).
        policy_rejected_logps: log pi_theta(y_l | x), shape (batch_size,).
        ref_chosen_logps: log pi_ref(y_w | x), shape (batch_size,).
        ref_rejected_logps: log pi_ref(y_l | x), shape (batch_size,).
        beta: DPO temperature parameter.
        label_smoothing: Optional label smoothing (0 = no smoothing).

    Returns:
        Tuple of:
            - Scalar DPO loss.
            - Dict of metrics: {
                'chosen_reward': mean implicit reward for chosen,
                'rejected_reward': mean implicit reward for rejected,
                'reward_margin': mean reward difference,
                'accuracy': fraction where chosen_reward > rejected_reward,
              }
    """
    chosen_log_ratio = policy_chosen_logps - jax.lax.stop_gradient(ref_chosen_logps)
    rejected_log_ratio = policy_rejected_logps - jax.lax.stop_gradient(ref_rejected_logps)

    # DPO loss uses log-ratios (policy vs reference)
    logits = beta * (chosen_log_ratio - rejected_log_ratio)   # (batch_size,)
    if label_smoothing > 0:
        loss = -(1 - label_smoothing) * jax.nn.log_sigmoid(logits) - label_smoothing * jax.nn.log_sigmoid(-logits)
    else:
        loss = -jax.nn.log_sigmoid(logits)
    loss = loss.mean()

    # Metrics use raw policy scores for interpretability
    chosen_reward = beta * policy_chosen_logps
    rejected_reward = beta * policy_rejected_logps
    accuracy = (chosen_reward > rejected_reward).astype(jnp.float32).mean()
    reward_margin = (chosen_reward - rejected_reward).mean()

    metrics = {
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "reward_margin": reward_margin,
        "accuracy": accuracy,
    }
    return loss, metrics

# ============================================================================
# DPO training state and training step
# ============================================================================

def create_dpo_train_state(
    model: GPT2LMHeadModel,
    config: ModelConfig,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize model and optimizer for DPO training.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def dpo_train_step(
    params: Any,
    ref_params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[Any, Any, jax.Array, Dict[str, jax.Array]]:
    """Perform one DPO training step.

    Args:
        params: Current policy parameters (trainable).
        ref_params: Reference model parameters (frozen, from SFT).
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'chosen_input_ids': (batch_size, seq_len)
            - 'chosen_attention_mask': (batch_size, seq_len)
            - 'chosen_response_mask': (batch_size, seq_len) — 1 for response tokens
            - 'rejected_input_ids': (batch_size, seq_len)
            - 'rejected_attention_mask': (batch_size, seq_len)
            - 'rejected_response_mask': (batch_size, seq_len)
        model: GPT2LMHeadModel instance.
        beta: DPO temperature.
        label_smoothing: Label smoothing parameter.

    Returns:
        Tuple of (new_params, new_opt_state, loss, metrics).
    """

    def loss_fn(params):
        policy_chosen_logs = compute_response_log_probs(model, params, batch['chosen_input_ids'],batch['chosen_attention_mask'], batch['chosen_response_mask'])
        policy_rejected_logs = compute_response_log_probs(model, params, batch['rejected_input_ids'],batch['rejected_attention_mask'], batch['rejected_response_mask'])
        ref_chosen_logs = compute_response_log_probs(model, ref_params, batch['chosen_input_ids'],batch['chosen_attention_mask'], batch['chosen_response_mask'])
        ref_rejected_logs = compute_response_log_probs(model, ref_params, batch['rejected_input_ids'],batch['rejected_attention_mask'], batch['rejected_response_mask'])

        loss, metrics = dpo_loss(
            policy_chosen_logs, policy_rejected_logs,
            ref_chosen_logs, ref_rejected_logs,
            beta=beta, label_smoothing=label_smoothing
        )
        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, metrics
