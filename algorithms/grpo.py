"""Group Relative Policy Optimization (GRPO) for RLHF.

GRPO is a PPO variant that eliminates the value network by estimating advantages
from a group of responses per prompt. For each prompt:
    1. Sample G responses from the current policy.
    2. Score each response with a reward model.
    3. Normalize rewards within the group to get advantages (z-score).
    4. Update the policy with a PPO-clip objective using group-relative advantages.

References:
    - DeepSeek, 2025, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
      via Reinforcement Learning"
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Callable
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs
from utils.generation import generate


# ============================================================================
# Group sampling and scoring
# ============================================================================
#
# Shape convention:
#   prompts:   (batch_size, prompt_len)
#   responses: (batch_size, group_size, response_len)
#   rewards:   (batch_size, group_size)
# ============================================================================

def group_sample_and_score(
    generate_fn: Callable,
    reward_fn: Callable,
    prompt_ids: jax.Array,
    rng: jax.Array,
    group_size: int = 8,
    max_response_len: int = 128,
) -> Tuple[jax.Array, jax.Array]:
    """Sample a group of responses per prompt and score them.

    Args:
        generate_fn: Function (rng, prompt_ids) -> generated_ids.
                     Generates one response per prompt in the batch.
        reward_fn: Function (input_ids) -> rewards.
                   Computes scalar rewards for sequences.
        prompt_ids: Prompt token IDs, shape (batch_size, prompt_len).
        rng: PRNG key for sampling.
        group_size: Number of responses to sample per prompt (G).
        max_response_len: Maximum response length.

    Returns:
        Tuple of:
            - generated_ids: (batch_size, group_size, total_len) — all generated sequences
            - rewards: (batch_size, group_size) — reward for each response
    """
    rngs = jax.random.split(rng, group_size)

    def _sample_one(rng_g, prompt_ids):
        generated = generate_fn(rng_g, prompt_ids)
        reward = reward_fn(generated)
        return generated, reward
    
    generated_ids, rewards = jax.vmap(_sample_one, in_axes=(0, None))(rngs, prompt_ids)

    return generated_ids.transpose(1, 0, 2), rewards.T


# ============================================================================
# Group-relative advantage estimation
# ============================================================================
#
# A_{i,j} = (R_{i,j} - mean(R_i)) / (std(R_i) + eps)
#
# Responses scoring above the group mean receive positive advantages;
# those below receive negative advantages. This replaces the value network.
# ============================================================================

def group_relative_advantage(
    rewards: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute group-relative advantages.

    Args:
        rewards: Per-response rewards, shape (batch_size, group_size).
        eps: Small constant for numerical stability in std division.

    Returns:
        Advantages, shape (batch_size, group_size).
    """
    group_mean = rewards.mean(axis=-1, keepdims=True)
    group_std = rewards.std(axis=-1, keepdims=True)
    advantages = (rewards - group_mean) / (group_std + eps)

    return advantages


# ============================================================================
# GRPO objective and update step
# ============================================================================
#
# Per response j and token position t:
#   r_{j,t} = pi_theta(a_{j,t}) / pi_old(a_{j,t})
#   L_{j,t} = min(r_{j,t} * A_j, clip(r_{j,t}, 1-eps, 1+eps) * A_j)
#
# Total: L = -(1/G) sum_j L_j + kl_coeff * KL(pi || pi_ref)
# A_j is constant across all tokens in response j (sequence-level advantage).
# ============================================================================

def grpo_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    ref_log_probs: jax.Array,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.05,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the GRPO loss for a single group member.

    This function computes the loss for ONE response. The caller should
    loop over or vmap over group members and average.

    Args:
        log_probs: Current policy per-token log-probs, shape (batch_size, seq_len).
        old_log_probs: Old policy log-probs (from sampling), shape (batch_size, seq_len).
        ref_log_probs: Reference model log-probs, shape (batch_size, seq_len).
        advantages: Sequence-level advantages, shape (batch_size,).
                    Same value for all tokens in a response (group-relative).
        clip_eps: PPO clipping epsilon.
        kl_coeff: KL penalty coefficient.
        mask: Token mask, shape (batch_size, seq_len).

    Returns:
        Tuple of (scalar_loss, metrics_dict).
    """

    token_advantages = advantages[:, None] * jnp.ones_like(log_probs)
    ratio = jnp.exp(log_probs - jax.lax.stop_gradient(old_log_probs))

    surr1 = ratio * token_advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantages
    policy_loss = -jnp.minimum(surr1, surr2)

    kl = log_probs - jax.lax.stop_gradient(ref_log_probs)

    total_per_token = policy_loss + kl_coeff * kl
    if mask is not None:
        total_per_token = total_per_token * mask
        loss = total_per_token.sum() / jnp.maximum(mask.sum(), 1.0)
        kl_mean = (kl * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        loss = total_per_token.mean()
        kl_mean = kl.mean()

    clip_fraction = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
    if mask is not None:
        clip_fraction = (clip_fraction * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        clip_fraction = clip_fraction.mean()
    
    metrics = {
        "policy_loss": policy_loss.mean() if mask is None else (policy_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0),
        "kl": kl_mean,
        "clip_fraction": clip_fraction,
    }

    return loss, metrics

def grpo_update_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    ref_params: Any,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.05,
) -> Tuple[Any, Any, Dict[str, jax.Array]]:
    """Perform one GRPO update step.

    This processes all group members in a batch and averages the loss.

    Args:
        params: Current policy parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'input_ids': (batch_size * group_size, seq_len) — flattened group
            - 'attention_mask': (batch_size * group_size, seq_len)
            - 'old_log_probs': (batch_size * group_size, seq_len)
        model: GPT2LMHeadModel instance.
        ref_params: Frozen reference model parameters.
        advantages: (batch_size * group_size,) — flattened group advantages.
        clip_eps: PPO clipping epsilon.
        kl_coeff: KL penalty coefficient.

    Returns:
        Tuple of (new_params, new_opt_state, metrics).
    """
    def loss_fn(params):
        # Forward pass with current policy
        logits = model.apply(
            params, batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
        )
        log_probs = compute_log_probs(logits, batch['input_ids'], batch['attention_mask'])

        # Forward pass with reference model (frozen)
        ref_logits = model.apply(
            ref_params, batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
        )
        ref_log_probs = compute_log_probs(ref_logits, batch['input_ids'], batch['attention_mask'])
        ref_log_probs = jax.lax.stop_gradient(ref_log_probs)

        # GRPO loss
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=batch['old_log_probs'],
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            clip_eps=clip_eps,
            kl_coeff=kl_coeff,
            mask=batch['attention_mask'][:, 1:],  # shifted mask
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics
