"""Tests for GRPO: group sampling, group-relative advantages, and GRPO loss.

Run with: pytest tests/test_grpo.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.grpo import (
    group_sample_and_score,
    group_relative_advantage,
    grpo_loss,
)


# ---------------------------------------------------------------------------
# Component: Group Sampling and Scoring
# ---------------------------------------------------------------------------

class TestGroupSampleAndScore:
    """Tests for group_sample_and_score."""

    def test_output_shapes(self):
        """Generated ids and rewards should have correct shapes."""
        batch_size = 2
        group_size = 4
        prompt_len = 5
        response_len = 10

        prompt_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)

        # Mock generate: just appends zeros
        def mock_generate(rng, prompt_ids):
            B = prompt_ids.shape[0]
            response = jnp.zeros((B, response_len), dtype=jnp.int32)
            return jnp.concatenate([prompt_ids, response], axis=1)

        # Mock reward: returns random scalar per sequence
        def mock_reward(input_ids):
            return jax.random.normal(jax.random.PRNGKey(0), (input_ids.shape[0],))

        generated, rewards = group_sample_and_score(
            generate_fn=mock_generate,
            reward_fn=mock_reward,
            prompt_ids=prompt_ids,
            rng=rng,
            group_size=group_size,
        )

        assert generated.shape == (batch_size, group_size, prompt_len + response_len)
        assert rewards.shape == (batch_size, group_size)

    def test_different_rngs_give_different_results(self):
        """Different random seeds should produce different groups."""
        prompt_ids = jnp.ones((1, 3), dtype=jnp.int32)

        def mock_generate(rng, prompt_ids):
            B = prompt_ids.shape[0]
            response = jax.random.randint(rng, (B, 5), 0, 100)
            return jnp.concatenate([prompt_ids, response], axis=1)

        def mock_reward(input_ids):
            return input_ids.sum(axis=-1).astype(jnp.float32)

        gen1, _ = group_sample_and_score(
            mock_generate, mock_reward, prompt_ids, jax.random.PRNGKey(0), group_size=4,
        )
        gen2, _ = group_sample_and_score(
            mock_generate, mock_reward, prompt_ids, jax.random.PRNGKey(1), group_size=4,
        )
        assert not jnp.allclose(gen1, gen2)


# ---------------------------------------------------------------------------
# Component: Group-Relative Advantage
# ---------------------------------------------------------------------------

class TestGroupRelativeAdvantage:
    """Tests for group-relative advantage."""

    def test_output_shape(self):
        """Advantages should have same shape as rewards."""
        rewards = jax.random.normal(jax.random.PRNGKey(0), (3, 8))
        advantages = group_relative_advantage(rewards)
        assert advantages.shape == (3, 8)

    def test_zero_mean_per_group(self):
        """Advantages should have approximately zero mean per group."""
        rewards = jax.random.normal(jax.random.PRNGKey(0), (4, 16))
        advantages = group_relative_advantage(rewards)
        group_means = advantages.mean(axis=-1)
        np.testing.assert_allclose(group_means, jnp.zeros(4), atol=1e-5)

    def test_unit_variance_per_group(self):
        """Advantages should have approximately unit variance per group."""
        rewards = jax.random.normal(jax.random.PRNGKey(0), (4, 16)) * 5 + 3
        advantages = group_relative_advantage(rewards)
        group_stds = advantages.std(axis=-1)
        np.testing.assert_allclose(group_stds, jnp.ones(4), atol=0.1)

    def test_ordering_preserved(self):
        """Higher rewards should give higher advantages within each group."""
        rewards = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        advantages = group_relative_advantage(rewards)
        # Advantages should be in the same order as rewards
        assert (jnp.diff(advantages[0]) > 0).all(), "Ordering should be preserved"

    def test_constant_rewards_give_zero_advantages(self):
        """If all rewards in a group are equal, advantages should be ~0."""
        rewards = jnp.ones((2, 8)) * 5.0
        advantages = group_relative_advantage(rewards)
        # With zero std, the result depends on eps handling
        assert jnp.allclose(advantages, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Component: GRPO Loss
# ---------------------------------------------------------------------------

class TestGRPOLoss:
    """Tests for the GRPO loss."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        B, T = 2, 5
        log_probs = jax.random.normal(jax.random.PRNGKey(0), (B, T))
        old_log_probs = jax.random.normal(jax.random.PRNGKey(1), (B, T))
        ref_log_probs = jax.random.normal(jax.random.PRNGKey(2), (B, T))
        advantages = jnp.array([1.0, -0.5])

        loss, metrics = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)
        assert loss.shape == ()

    def test_zero_advantage_zero_policy_loss(self):
        """With zero advantages, policy loss should be zero."""
        B, T = 2, 5
        log_probs = jnp.zeros((B, T))
        old_log_probs = jnp.zeros((B, T))
        ref_log_probs = jnp.zeros((B, T))
        advantages = jnp.zeros(B)

        loss, metrics = grpo_loss(
            log_probs, old_log_probs, ref_log_probs, advantages, kl_coeff=0.0,
        )
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

    def test_kl_penalty(self):
        """KL penalty should increase loss when policy diverges from reference."""
        B, T = 1, 5
        old_log_probs = jnp.zeros((B, T))
        ref_log_probs = jnp.zeros((B, T))
        advantages = jnp.zeros(B)

        # Policy matches reference
        loss_same, _ = grpo_loss(
            jnp.zeros((B, T)), old_log_probs, ref_log_probs, advantages, kl_coeff=1.0,
        )
        # Policy diverges from reference
        loss_diff, _ = grpo_loss(
            jnp.ones((B, T)), old_log_probs, ref_log_probs, advantages, kl_coeff=1.0,
        )
        assert float(loss_diff) > float(loss_same)

    def test_metrics_keys(self):
        """Should return expected metric keys."""
        B, T = 2, 5
        _, metrics = grpo_loss(
            jnp.zeros((B, T)), jnp.zeros((B, T)), jnp.zeros((B, T)), jnp.zeros(B),
        )
        assert "policy_loss" in metrics
        assert "kl" in metrics
        assert "clip_fraction" in metrics

    def test_with_mask(self):
        """Should work with token mask."""
        B, T = 2, 5
        mask = jnp.ones((B, T)).at[:, 3:].set(0.0)
        loss, _ = grpo_loss(
            jnp.zeros((B, T)), jnp.zeros((B, T)), jnp.zeros((B, T)),
            jnp.zeros(B), mask=mask,
        )
        assert jnp.isfinite(loss)
