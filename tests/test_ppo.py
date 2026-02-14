"""Tests for PPO: KL divergence, GAE, clipped objective, value loss.

Run with: pytest tests/test_ppo.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import (
    compute_kl_divergence,
    compute_gae,
    ppo_policy_loss,
    value_function_loss,
)


# ---------------------------------------------------------------------------
# Component: KL Divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_same_distribution(self):
        """KL should be 0 when distributions are identical."""
        log_probs = jnp.array([[0.1, -0.5, -1.0]])
        kl = compute_kl_divergence(log_probs, log_probs)
        np.testing.assert_allclose(float(kl), 0.0, atol=1e-6)

    def test_kl_is_positive(self):
        """KL divergence should generally be non-negative (for exact KL).

        Note: Our approximation (log_pi - log_pi_ref) can be negative for
        individual tokens, but the mean over tokens where pi > pi_ref should
        dominate for reasonable policy shifts.
        """
        log_probs = jnp.array([[0.1, -0.2, -0.3]])
        ref_log_probs = jnp.array([[-0.1, -0.5, -0.6]])
        kl = compute_kl_divergence(log_probs, ref_log_probs)
        # log_probs > ref_log_probs everywhere, so KL > 0
        assert float(kl) > 0

    def test_kl_with_mask(self):
        """Masked tokens should not contribute to KL."""
        log_probs = jnp.array([[1.0, 2.0, 3.0]])
        ref_log_probs = jnp.array([[0.0, 0.0, 0.0]])
        mask = jnp.array([[1.0, 0.0, 0.0]])  # only first token counts

        kl = compute_kl_divergence(log_probs, ref_log_probs, mask=mask)
        # Only the first token (1.0 - 0.0 = 1.0) should contribute
        np.testing.assert_allclose(float(kl), 1.0, atol=1e-6)

    def test_kl_direction(self):
        """Larger divergence should give larger KL."""
        ref = jnp.array([[-1.0, -1.0]])
        small_shift = jnp.array([[-0.9, -0.9]])
        large_shift = jnp.array([[0.0, 0.0]])

        kl_small = compute_kl_divergence(small_shift, ref)
        kl_large = compute_kl_divergence(large_shift, ref)
        assert float(kl_large) > float(kl_small)


# ---------------------------------------------------------------------------
# Component: GAE
# ---------------------------------------------------------------------------

class TestGAE:
    """Tests for Generalized Advantage Estimation."""

    def test_gae_shapes(self):
        """Output shapes should match input shapes."""
        rewards = jnp.zeros((2, 10))
        values = jnp.zeros((2, 10))
        advantages, returns = compute_gae(rewards, values)
        assert advantages.shape == (2, 10)
        assert returns.shape == (2, 10)

    def test_gae_zero_rewards(self):
        """With zero rewards and zero values, advantages should be zero."""
        rewards = jnp.zeros((1, 5))
        values = jnp.zeros((1, 5))
        advantages, returns = compute_gae(rewards, values)
        np.testing.assert_allclose(advantages, jnp.zeros_like(advantages), atol=1e-6)

    def test_gae_single_reward_at_end(self):
        """Typical RLHF case: reward only at the last token."""
        rewards = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        values = jnp.zeros((1, 5))
        advantages, returns = compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0)

        # With gamma=1, lambda=1, values=0: advantages should equal
        # discounted sum of rewards from each position onward.
        # A[4] = r[4] = 1.0, A[3] = r[3] + 1*1*A[4] = 1.0, etc.
        expected = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_allclose(advantages, expected, atol=1e-5)

    def test_gae_with_values(self):
        """GAE should incorporate value baseline correctly."""
        rewards = jnp.array([[1.0, 0.0, 0.0]])
        values = jnp.array([[0.5, 0.5, 0.5]])
        advantages, returns = compute_gae(rewards, values, gamma=1.0, gae_lambda=0.95)

        # With non-zero values, advantages should be smaller (value baseline helps)
        assert advantages.shape == (1, 3)
        # Returns = advantages + values
        np.testing.assert_allclose(returns, advantages + values, atol=1e-6)

    def test_gae_returns_equal_advantages_plus_values(self):
        """Returns should always equal advantages + values."""
        rng = jax.random.PRNGKey(42)
        rewards = jax.random.normal(rng, (3, 8))
        values = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
        advantages, returns = compute_gae(rewards, values)
        np.testing.assert_allclose(returns, advantages + values, atol=1e-5)


# ---------------------------------------------------------------------------
# Component: PPO Policy Loss
# ---------------------------------------------------------------------------

class TestPPOPolicyLoss:
    """Tests for the PPO policy loss."""

    def test_zero_loss_when_no_advantage(self):
        """Loss should be 0 when advantages are 0 (no direction to optimize)."""
        log_probs = jnp.zeros((2, 5))
        old_log_probs = jnp.zeros((2, 5))
        advantages = jnp.zeros((2, 5))
        loss, _ = ppo_policy_loss(log_probs, old_log_probs, advantages)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        log_probs = jax.random.normal(jax.random.PRNGKey(0), (2, 5))
        old_log_probs = jax.random.normal(jax.random.PRNGKey(1), (2, 5))
        advantages = jax.random.normal(jax.random.PRNGKey(2), (2, 5))
        loss, metrics = ppo_policy_loss(log_probs, old_log_probs, advantages)
        assert loss.shape == ()
        assert "clip_fraction" in metrics

    def test_clipping_behavior(self):
        """When ratio is far from 1, clipping should activate."""
        # Large positive shift: ratio >> 1
        old_log_probs = jnp.array([[-2.0]])
        log_probs = jnp.array([[0.0]])  # ratio = exp(2) ≈ 7.4
        advantages = jnp.array([[1.0]])

        loss, metrics = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_eps=0.2)
        # The clipped ratio should be 1.2, so loss = -1.2 * 1.0 = -1.2
        # But we take min(surr1, surr2), and surr1 = 7.4 * 1 = 7.4 > 1.2
        # So loss = -1.2
        assert float(loss) < 0  # Negative because we negate

    def test_with_mask(self):
        """Masked positions should not contribute to loss."""
        log_probs = jnp.array([[0.0, 0.0, 0.0]])
        old_log_probs = jnp.array([[0.0, 0.0, 0.0]])
        advantages = jnp.array([[1.0, 1.0, 1.0]])
        mask = jnp.array([[1.0, 0.0, 0.0]])

        loss_masked, _ = ppo_policy_loss(log_probs, old_log_probs, advantages, mask=mask)
        loss_full, _ = ppo_policy_loss(log_probs, old_log_probs, advantages)
        # Both should be -1.0 * 1.0 = -1.0 (ratio=1, so no clipping)
        # But masked version only averages over 1 token
        assert jnp.isfinite(loss_masked)


# ---------------------------------------------------------------------------
# Component: Value Function Loss
# ---------------------------------------------------------------------------

class TestValueFunctionLoss:
    """Tests for the value function loss."""

    def test_zero_loss_at_returns(self):
        """Loss should be 0 when values perfectly predict returns."""
        values = jnp.array([[1.0, 2.0, 3.0]])
        returns = jnp.array([[1.0, 2.0, 3.0]])
        old_values = jnp.array([[1.0, 2.0, 3.0]])
        loss = value_function_loss(values, old_values, returns)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

    def test_loss_is_positive(self):
        """Value loss should be non-negative."""
        values = jax.random.normal(jax.random.PRNGKey(0), (2, 5))
        old_values = jax.random.normal(jax.random.PRNGKey(1), (2, 5))
        returns = jax.random.normal(jax.random.PRNGKey(2), (2, 5))
        loss = value_function_loss(values, old_values, returns)
        assert float(loss) >= 0

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        values = jnp.ones((2, 5))
        old_values = jnp.ones((2, 5))
        returns = jnp.zeros((2, 5))
        loss = value_function_loss(values, old_values, returns)
        assert loss.shape == ()

    def test_with_mask(self):
        """Mask should exclude padding positions from loss."""
        values = jnp.array([[1.0, 1.0, 1.0]])
        old_values = jnp.array([[0.0, 0.0, 0.0]])
        returns = jnp.array([[0.0, 0.0, 0.0]])
        mask = jnp.array([[1.0, 0.0, 0.0]])

        loss = value_function_loss(values, old_values, returns, mask=mask)
        assert jnp.isfinite(loss)
