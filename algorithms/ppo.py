"""Proximal Policy Optimization (PPO) for RLHF.

Implements the PPO-based RLHF pipeline:
    1. Rollout: generate responses using the current policy.
    2. Reward: score responses with the reward model.
    3. Advantage: compute advantages via GAE (Generalized Advantage Estimation).
    4. Update: optimize the policy with clipped objective + value loss + entropy bonus (optinal, 
    could also be used for monitoring polciy drifting).

References:
    - Schulman et al., 2017, "Proximal Policy Optimization Algorithms"
    - Ouyang et al., 2022, "Training language models to follow instructions
      with human feedback" (InstructGPT)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs, compute_entropy


# ============================================================================
# KL divergence between policies
# ============================================================================
#
# Per-token approximation: kl_t = log_pi_t - log_pi_ref_t
# (action-level KL used in InstructGPT)
# Ignored two facts:
#   * Only looking at the output token difference (not the full vocabulary)
#   * Simply take the mean when aggregate, not the probability of (t1, t2, .., tn)
# ============================================================================

def compute_kl_divergence(
    log_probs: jax.Array,
    ref_log_probs: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute per-token KL divergence between policy and reference.

    Uses the approximation: KL ≈ log_pi - log_pi_ref (per-token).
    This is the "action-level" KL divergence used in InstructGPT's PPO.

    Args:
        log_probs: Per-token log-probs from current policy, shape (batch_size, seq_len).
        ref_log_probs: Per-token log-probs from reference policy, shape (batch_size, seq_len).
        mask: Optional mask, shape (batch_size, seq_len). 1 for valid tokens.

    Returns:
        Mean KL divergence (scalar).
    """
    kl = log_probs - ref_log_probs
    if mask is not None:
        kl = kl * mask
        return kl.sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        return kl.mean()


# ============================================================================
# Generalized Advantage Estimation (GAE)
# ============================================================================
#
# delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)              (TD error)
# A_t = delta_t + gamma * lambda * A_{t+1}                  (recursive GAE)
#
# For language RLHF, gamma is typically 1.0. Per-token rewards combine
# a KL penalty (-kl_coeff * kl_t) with the reward model score at the
# last response token.
# ============================================================================

def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Per-token rewards, shape (batch_size, seq_len).
                 Typically: -kl_coeff * kl_t for each token, plus reward model
                 score added to the last response token.
        values: Value estimates V(s_t), shape (batch_size, seq_len).
        gamma: Discount factor (1.0 for language tasks).
        gae_lambda: GAE lambda (bias-variance tradeoff).
        mask: Optional mask for valid tokens, shape (batch_size, seq_len).

    Returns:
        Tuple of:
            - advantages: shape (batch_size, seq_len)
            - returns: shape (batch_size, seq_len) — advantages + values (for value loss)
    """
    B = rewards.shape[0]

    def _gae_step(carry, t_data):
        last_advantage, last_value = carry
        reward_t, value_t, mask_t = t_data
        delta = reward_t + gamma * last_value * mask_t - value_t
        advantage = delta + gamma * gae_lambda * last_advantage * mask_t
        return (advantage, value_t), advantage
    
    if mask is not None:
        scan_data = (rewards[:, ::-1].T, values[:, ::-1].T, mask[:, ::-1].T)
    else:
        scan_data = (rewards[:, ::-1].T, values[:, ::-1].T, jnp.ones_like(rewards.T))
    
    init_carry = (jnp.zeros(B), jnp.zeros(B))
    _, advantages_reversed = jax.lax.scan(_gae_step, init_carry, scan_data)
    advantages = advantages_reversed[::-1].T
    return advantages, advantages + values



# ============================================================================
# PPO clipped surrogate objective
# ============================================================================
#
# ratio = exp(log_pi - log_pi_old)
# surr1 = ratio * advantages
# surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
# L_clip = -min(surr1, surr2)
# ============================================================================

def ppo_policy_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the PPO clipped surrogate policy loss.

    Args:
        log_probs: Current policy log-probs, shape (batch_size, seq_len).
        old_log_probs: Old policy log-probs (from rollout), shape (batch_size, seq_len).
        advantages: GAE advantages, shape (batch_size, seq_len).
        clip_eps: Clipping parameter epsilon (default 0.2).
        mask: Optional token mask, shape (batch_size, seq_len).

    Returns:
        Tuple of:
            - Scalar policy loss.
            - Dict of metrics: {'clip_fraction': ..., 'approx_kl': ...}
    """
    
    ratio = jnp.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.minimum(surr1, surr2)

    if mask is not None:
        policy_loss = (policy_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        policy_loss = policy_loss.mean()

    clip_fraction = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32).mean()
    approx_kl = (log_probs - old_log_probs).mean()

    return policy_loss, {'clip_fraction':clip_fraction, 'approx_kl':approx_kl}


# ============================================================================
# Clipped value function loss
# ============================================================================
#
# L_vf = 0.5 * max(
#     (V - returns)^2,
#     (clip(V, V_old - eps, V_old + eps) - returns)^2
# )
# ============================================================================

def value_function_loss(
    values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    clip_eps: float = 0.2,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute the clipped value function loss.

    Args:
        values: Current value estimates, shape (batch_size, seq_len).
        old_values: Old value estimates (from rollout), shape (batch_size, seq_len).
        returns: GAE returns (advantages + values), shape (batch_size, seq_len).
        clip_eps: Value clipping epsilon.
        mask: Optional token mask.

    Returns:
        Scalar value function loss.
    """
    vf_loss1 = (values - returns) ** 2
    clipped_values = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    vf_loss2 = (clipped_values - returns) ** 2
    vf_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2)

    if mask is not None:
        vf_loss = (vf_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        vf_loss = vf_loss.mean()

    return vf_loss 


# ============================================================================
# Combined PPO update step
# ============================================================================
#
# L = L_policy + vf_coeff * L_value - entropy_coeff * entropy + kl_coeff * KL
# ============================================================================

def ppo_update_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    value_head_apply_fn,
    ref_log_probs: jax.Array,
    clip_eps: float = 0.2,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    kl_coeff: float = 0.1,
) -> Tuple[Any, Any, Dict[str, jax.Array]]:
    """Perform one PPO update step on a minibatch.

    Args:
        params: Combined policy + value head parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'input_ids': (batch_size, seq_len)
            - 'attention_mask': (batch_size, seq_len)
            - 'old_log_probs': (batch_size, seq_len) — from rollout
            - 'old_values': (batch_size, seq_len) — from rollout
            - 'advantages': (batch_size, seq_len) — from GAE
            - 'returns': (batch_size, seq_len) — advantages + values
        model: GPT2LMHeadModel for the policy.
        value_head_apply_fn: Function to compute values: (params, hidden_states) -> values.
        ref_log_probs: Reference model log-probs (frozen), shape (batch_size, seq_len).
        clip_eps: PPO clipping epsilon.
        vf_coeff: Value function loss coefficient.
        entropy_coeff: Entropy bonus coefficient.
        kl_coeff: KL penalty coefficient.

    Returns:
        Tuple of (new_params, new_opt_state, metrics_dict).
    """
    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'], attention_mask = batch['attention_mask'], deterministic=False)
        log_probs = compute_log_probs(logits, batch['input_ids'], batch['attention_mask'])

        hidden_states = model.apply(params, batch['input_ids'], attention_mask=batch['attention_mask'], determinstic=False, method=model.get_hidden_states)
        values = value_head_apply_fn(params, hidden_states)

        policy_loss, policy_metrics = ppo_policy_loss(log_probs, batch['old_log_probs'], batch['advantages'], clip_eps=clip_eps, mask=batch['attention_mask'][:, 1:])

        vf_loss = value_function_loss(values, batch['old_values'], batch['returns'], clip_eps=clip_eps, mask=batch['attention_mask'])

        entropy = compute_entropy(logits, batch['attention_mask'])

        kl = compute_kl_divergence(log_probs, ref_log_probs, batch['attention_mask'][:, 1:])

        total_loss = policy_loss + vf_coeff * vf_loss - entropy_coeff * entropy + kl_coeff * kl
        metrics = {
             "total_loss": total_loss,
             "policy_loss": policy_loss,
             "value_loss": vf_loss,
             "entropy": entropy,
             "kl": kl,
             **policy_metrics,
         }
    
        return total_loss, metrics

    (loss, metrics), grads = jax.value_and_grads(loss_fn, has_aus=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics
