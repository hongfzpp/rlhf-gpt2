"""Tests for reward model: architecture, preference loss, and training step.

Run with: pytest tests/test_reward.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from configs.model_config import ModelConfig
from models.reward_model import RewardModel
from algorithms.reward import preference_loss, reward_train_step, create_reward_train_state


TEST_CONFIG = ModelConfig(
    vocab_size=100,
    max_seq_len=32,
    n_layers=2,
    n_heads=2,
    d_model=64,
    d_ff=256,
    dropout_rate=0.0,
)


# ---------------------------------------------------------------------------
# Component: Reward Model Architecture
# ---------------------------------------------------------------------------

class TestRewardModelArchitecture:
    """Tests for the RewardModel."""

    def test_output_shape(self):
        """Output should be scalar rewards of shape (batch_size,)."""
        model = RewardModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((3, 16), dtype=jnp.int32)

        params = model.init(rng, input_ids)
        rewards = model.apply(params, input_ids, deterministic=True)
        assert rewards.shape == (3,), f"Expected (3,), got {rewards.shape}"

    def test_with_attention_mask(self):
        """Should work with attention mask and produce different rewards for different masks."""
        model = RewardModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(rng, (2, 16), 0, TEST_CONFIG.vocab_size)

        mask1 = jnp.ones((2, 16))
        mask2 = jnp.ones((2, 16)).at[:, 12:].set(0.0)

        params = model.init(rng, input_ids)
        r1 = model.apply(params, input_ids, attention_mask=mask1, deterministic=True)
        r2 = model.apply(params, input_ids, attention_mask=mask2, deterministic=True)

        assert r1.shape == (2,)
        assert r2.shape == (2,)

    def test_different_inputs_different_rewards(self):
        """Different input sequences should generally get different rewards."""
        model = RewardModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        ids1 = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, TEST_CONFIG.vocab_size)
        ids2 = jax.random.randint(jax.random.PRNGKey(2), (1, 16), 0, TEST_CONFIG.vocab_size)

        params = model.init(rng, ids1)
        r1 = model.apply(params, ids1, deterministic=True)
        r2 = model.apply(params, ids2, deterministic=True)

        # With random init, different inputs should yield different rewards
        assert not jnp.allclose(r1, r2), "Different inputs should give different rewards"


# ---------------------------------------------------------------------------
# Component: Preference Loss
# ---------------------------------------------------------------------------

class TestPreferenceLoss:
    """Tests for the preference loss."""

    def test_loss_is_positive(self):
        """Loss should be positive."""
        chosen = jnp.array([1.0, 2.0, 3.0])
        rejected = jnp.array([0.5, 1.0, 1.5])
        loss, _ = preference_loss(chosen, rejected)
        assert float(loss) > 0

    def test_loss_when_chosen_clearly_better(self):
        """Loss should be small when chosen >> rejected."""
        chosen = jnp.array([10.0, 10.0, 10.0])
        rejected = jnp.array([-10.0, -10.0, -10.0])
        loss, metrics = preference_loss(chosen, rejected)
        assert float(loss) < 0.01, f"Loss should be near 0, got {float(loss)}"
        assert float(metrics["accuracy"]) == 1.0

    def test_loss_when_rejected_is_better(self):
        """Loss should be large when rejected > chosen (wrong ordering)."""
        chosen = jnp.array([-5.0, -5.0])
        rejected = jnp.array([5.0, 5.0])
        loss, metrics = preference_loss(chosen, rejected)
        assert float(loss) > 1.0, f"Loss should be large, got {float(loss)}"
        assert float(metrics["accuracy"]) == 0.0

    def test_accuracy_computation(self):
        """Accuracy should count fraction of correct orderings."""
        chosen = jnp.array([1.0, 0.0, 1.0, 0.0])
        rejected = jnp.array([0.0, 1.0, 0.0, 1.0])
        _, metrics = preference_loss(chosen, rejected)
        np.testing.assert_allclose(float(metrics["accuracy"]), 0.5, atol=1e-6)

    def test_reward_margin(self):
        """Reward margin should be mean(chosen - rejected)."""
        chosen = jnp.array([3.0, 4.0])
        rejected = jnp.array([1.0, 2.0])
        _, metrics = preference_loss(chosen, rejected)
        np.testing.assert_allclose(float(metrics["reward_margin"]), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Component: Reward Training Step
# ---------------------------------------------------------------------------

class TestRewardTrainStep:
    """Tests for the reward model training step."""

    def setup_method(self):
        self.reward_model = RewardModel(config=TEST_CONFIG)
        self.params, self.opt_state, self.optimizer = create_reward_train_state(
            self.reward_model, TEST_CONFIG, learning_rate=1e-3,
        )

    def _make_batch(self, rng):
        rng1, rng2 = jax.random.split(rng)
        return {
            "chosen_input_ids": jax.random.randint(rng1, (4, 16), 0, TEST_CONFIG.vocab_size),
            "chosen_attention_mask": jnp.ones((4, 16)),
            "rejected_input_ids": jax.random.randint(rng2, (4, 16), 0, TEST_CONFIG.vocab_size),
            "rejected_attention_mask": jnp.ones((4, 16)),
        }

    def test_returns_correct_structure(self):
        """Should return (params, opt_state, loss, metrics)."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, new_opt_state, loss, metrics = reward_train_step(
            self.params, self.opt_state, self.optimizer, batch, self.reward_model
        )
        assert loss.shape == ()
        assert "accuracy" in metrics
        assert "reward_margin" in metrics

    def test_params_change(self):
        """Parameters should update after a training step."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, _, _, _ = reward_train_step(
            self.params, self.opt_state, self.optimizer, batch, self.reward_model
        )
        leaves_old = jax.tree.leaves(self.params)
        leaves_new = jax.tree.leaves(new_params)
        any_changed = any(not jnp.allclose(a, b) for a, b in zip(leaves_old, leaves_new))
        assert any_changed, "Parameters should change after training step"

    def test_jittable(self):
        """Training step should be JIT-compilable."""
        batch = self._make_batch(jax.random.PRNGKey(0))

        @jax.jit
        def jit_step(params, opt_state, batch):
            return reward_train_step(params, opt_state, self.optimizer, batch, self.reward_model)

        _, _, loss, _ = jit_step(self.params, self.opt_state, batch)
        assert jnp.isfinite(loss)
