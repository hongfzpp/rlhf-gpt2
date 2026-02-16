"""Tests for DPO: response log-probabilities, DPO loss, and training step.

Run with: pytest tests/test_dpo.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from configs.model_config import ModelConfig
from models.gpt2 import GPT2LMHeadModel
from algorithms.dpo import (
    compute_response_log_probs,
    dpo_loss,
    dpo_train_step,
    create_dpo_train_state,
)
from utils.jax_utils import clone_params


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
# Component: Response Log-Probabilities
# ---------------------------------------------------------------------------

class TestResponseLogProbs:
    """Tests for compute_response_log_probs."""

    def setup_method(self):
        self.model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, 16), dtype=jnp.int32)
        self.params = self.model.init(rng, dummy)

    def test_output_shape(self):
        """Should return one scalar per batch element."""
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (3, 16), 0, TEST_CONFIG.vocab_size)
        attention_mask = jnp.ones((3, 16))
        response_mask = jnp.ones((3, 16))

        logps = compute_response_log_probs(
            self.model, self.params, input_ids, attention_mask, response_mask
        )
        assert logps.shape == (3,), f"Expected (3,), got {logps.shape}"

    def test_log_probs_are_negative(self):
        """Log-probabilities should be negative (probabilities < 1)."""
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 16), 0, TEST_CONFIG.vocab_size)
        attention_mask = jnp.ones((2, 16))
        response_mask = jnp.ones((2, 16))

        logps = compute_response_log_probs(
            self.model, self.params, input_ids, attention_mask, response_mask
        )
        assert (logps < 0).all(), "Log-probs should be negative"

    def test_mask_reduces_magnitude(self):
        """Masking tokens should reduce the magnitude of log-probs (fewer tokens)."""
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (1, 16), 0, TEST_CONFIG.vocab_size)
        full_mask = jnp.ones((1, 16))
        partial_mask = jnp.ones((1, 16)).at[:, :8].set(0.0)  # only last 8 tokens

        logps_full = compute_response_log_probs(
            self.model, self.params, input_ids, full_mask, full_mask
        )
        logps_partial = compute_response_log_probs(
            self.model, self.params, input_ids, full_mask, partial_mask
        )
        # Fewer tokens -> less negative log-prob (closer to 0)
        assert float(logps_partial) > float(logps_full)


# ---------------------------------------------------------------------------
# Component: DPO Loss
# ---------------------------------------------------------------------------

class TestDPOLoss:
    """Tests for the DPO loss."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        loss, metrics = dpo_loss(
            policy_chosen_logps=jnp.array([-1.0, -2.0]),
            policy_rejected_logps=jnp.array([-3.0, -4.0]),
            ref_chosen_logps=jnp.array([-1.5, -2.5]),
            ref_rejected_logps=jnp.array([-3.5, -4.5]),
            beta=0.1,
        )
        assert loss.shape == ()

    def test_loss_with_clear_preference(self):
        """Loss should be low when policy clearly prefers chosen over rejected."""
        loss, metrics = dpo_loss(
            policy_chosen_logps=jnp.array([-0.1]),   # high prob for chosen
            policy_rejected_logps=jnp.array([-10.0]), # low prob for rejected
            ref_chosen_logps=jnp.array([-1.0]),
            ref_rejected_logps=jnp.array([-1.0]),
            beta=1.0,
        )
        assert float(loss) < 0.1, f"Loss should be small, got {float(loss)}"
        assert float(metrics["accuracy"]) == 1.0

    def test_loss_with_wrong_preference(self):
        """Loss should be high when policy prefers rejected over chosen."""
        loss, metrics = dpo_loss(
            policy_chosen_logps=jnp.array([-10.0]),  # low prob for chosen
            policy_rejected_logps=jnp.array([-0.1]), # high prob for rejected
            ref_chosen_logps=jnp.array([-1.0]),
            ref_rejected_logps=jnp.array([-1.0]),
            beta=1.0,
        )
        assert float(loss) > 1.0, f"Loss should be large, got {float(loss)}"
        assert float(metrics["accuracy"]) == 0.0

    def test_metrics_keys(self):
        """Should return expected metric keys."""
        _, metrics = dpo_loss(
            jnp.array([-1.0]), jnp.array([-2.0]),
            jnp.array([-1.0]), jnp.array([-2.0]),
        )
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics
        assert "accuracy" in metrics

    def test_beta_effect(self):
        """Higher beta should amplify the reward difference."""
        args = dict(
            policy_chosen_logps=jnp.array([-1.0]),
            policy_rejected_logps=jnp.array([-2.0]),
            ref_chosen_logps=jnp.array([-1.5]),
            ref_rejected_logps=jnp.array([-2.5]),
        )
        _, m_low = dpo_loss(**args, beta=0.1)
        _, m_high = dpo_loss(**args, beta=1.0)

        # Higher beta should give larger absolute reward values
        assert abs(float(m_high["reward_margin"])) > abs(float(m_low["reward_margin"]))


# ---------------------------------------------------------------------------
# Component: DPO Training Step
# ---------------------------------------------------------------------------

class TestDPOTrainStep:
    """Tests for the DPO training step."""

    def setup_method(self):
        self.model = GPT2LMHeadModel(config=TEST_CONFIG)
        self.params, self.opt_state, self.optimizer = create_dpo_train_state(
            self.model, TEST_CONFIG, learning_rate=1e-3,
        )
        self.ref_params = clone_params(self.params)

    def _make_batch(self, rng):
        rng1, rng2 = jax.random.split(rng)
        return {
            "chosen_input_ids": jax.random.randint(rng1, (2, 16), 0, TEST_CONFIG.vocab_size),
            "chosen_attention_mask": jnp.ones((2, 16)),
            "chosen_response_mask": jnp.ones((2, 16)),
            "rejected_input_ids": jax.random.randint(rng2, (2, 16), 0, TEST_CONFIG.vocab_size),
            "rejected_attention_mask": jnp.ones((2, 16)),
            "rejected_response_mask": jnp.ones((2, 16)),
        }

    def test_returns_correct_structure(self):
        """Should return (params, opt_state, loss, metrics)."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, new_opt_state, loss, metrics = dpo_train_step(
            self.params, self.ref_params, self.opt_state,
            self.optimizer, batch, self.model,
        )
        assert loss.shape == ()
        assert "accuracy" in metrics

    def test_params_change(self):
        """Parameters should change after training."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, _, _, _ = dpo_train_step(
            self.params, self.ref_params, self.opt_state,
            self.optimizer, batch, self.model,
        )
        leaves_old = jax.tree.leaves(self.params)
        leaves_new = jax.tree.leaves(new_params)
        any_changed = any(not jnp.allclose(a, b) for a, b in zip(leaves_old, leaves_new))
        assert any_changed

    def test_ref_params_unchanged(self):
        """Reference params should NOT change (frozen)."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        ref_before = jax.tree.leaves(self.ref_params)
        _ = dpo_train_step(
            self.params, self.ref_params, self.opt_state,
            self.optimizer, batch, self.model,
        )
        ref_after = jax.tree.leaves(self.ref_params)
        for a, b in zip(ref_before, ref_after):
            np.testing.assert_array_equal(a, b)

    def test_jittable(self):
        """Should be JIT-compilable."""
        batch = self._make_batch(jax.random.PRNGKey(0))

        @jax.jit
        def step(params, opt_state, batch):
            return dpo_train_step(
                params, self.ref_params, opt_state,
                self.optimizer, batch, self.model,
            )

        _, _, loss, _ = step(self.params, self.opt_state, batch)
        assert jnp.isfinite(loss)
