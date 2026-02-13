"""Tests for SFT training: cross-entropy loss, training step, and evaluation.

Run with: pytest tests/test_sft.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from configs.model_config import ModelConfig
from models.gpt2 import GPT2LMHeadModel
from algorithms.sft import cross_entropy_loss, sft_train_step, sft_eval_step, create_sft_train_state


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
# Component: Cross-Entropy Loss
# ---------------------------------------------------------------------------

class TestCrossEntropyLoss:
    """Tests for cross-entropy loss."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 100))
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 10), 0, 100)
        loss = cross_entropy_loss(logits, labels)
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"

    def test_loss_is_positive(self):
        """Cross-entropy loss should be positive."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 100))
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 10), 0, 100)
        loss = cross_entropy_loss(logits, labels)
        assert float(loss) > 0, "Cross-entropy should be positive"

    def test_loss_with_uniform_logits(self):
        """With uniform logits, loss should be approximately log(vocab_size)."""
        vocab_size = 100
        logits = jnp.zeros((1, 10, vocab_size))  # uniform distribution
        labels = jnp.ones((1, 10), dtype=jnp.int32)
        loss = cross_entropy_loss(logits, labels)
        expected = jnp.log(vocab_size)
        np.testing.assert_allclose(float(loss), float(expected), atol=0.01)

    def test_loss_ignores_padding(self):
        """Tokens with label -100 should be ignored."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 100))

        # All labels are -100 except position 5
        labels = jnp.full((1, 10), -100, dtype=jnp.int32)
        labels = labels.at[:, 5].set(42)

        loss = cross_entropy_loss(logits, labels)
        assert jnp.isfinite(loss), "Loss should be finite even with mostly masked labels"

    def test_loss_decreases_with_correct_logits(self):
        """Loss should decrease when logits favor the correct token."""
        vocab_size = 100
        labels = jnp.array([[0, 1, 2, 3, 4]])

        # Random logits
        logits_random = jax.random.normal(jax.random.PRNGKey(0), (1, 5, vocab_size))
        loss_random = cross_entropy_loss(logits_random, labels)

        # Logits that strongly favor correct tokens
        logits_correct = jnp.zeros((1, 5, vocab_size))
        for t in range(4):  # shift by 1
            logits_correct = logits_correct.at[:, t, labels[0, t + 1]].set(10.0)
        loss_correct = cross_entropy_loss(logits_correct, labels)

        assert float(loss_correct) < float(loss_random), (
            f"Correct logits should give lower loss: {float(loss_correct)} vs {float(loss_random)}"
        )


# ---------------------------------------------------------------------------
# Component: SFT Training Step
# ---------------------------------------------------------------------------

class TestSFTTrainStep:
    """Tests for the SFT training step."""

    def setup_method(self):
        """Set up model and training state for tests."""
        self.model = GPT2LMHeadModel(config=TEST_CONFIG)
        self.params, self.opt_state, self.optimizer = create_sft_train_state(
            model=self.model,
            config=TEST_CONFIG,
            learning_rate=1e-3,
        )

    def _make_batch(self, rng):
        input_ids = jax.random.randint(rng, (4, 16), 0, TEST_CONFIG.vocab_size)
        labels = input_ids.copy()
        labels = labels.at[:, :4].set(-100)  # mask first 4 tokens as "prompt"
        attention_mask = jnp.ones((4, 16))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def test_train_step_returns_correct_types(self):
        """Should return (params, opt_state, loss)."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, new_opt_state, loss = sft_train_step(
            self.params, self.opt_state, self.optimizer, batch, self.model
        )
        assert isinstance(loss, jax.Array)
        assert loss.shape == ()

    def test_train_step_changes_params(self):
        """Parameters should change after a training step."""
        batch = self._make_batch(jax.random.PRNGKey(0))
        new_params, _, _ = sft_train_step(
            self.params, self.opt_state, self.optimizer, batch, self.model
        )
        # At least some parameters should differ
        leaves_old = jax.tree.leaves(self.params)
        leaves_new = jax.tree.leaves(new_params)
        any_changed = any(
            not jnp.allclose(a, b)
            for a, b in zip(leaves_old, leaves_new)
        )
        assert any_changed, "Parameters should change after training step"

    def test_train_step_is_jittable(self):
        """Training step should be JIT-compilable."""
        batch = self._make_batch(jax.random.PRNGKey(0))

        @jax.jit
        def jit_step(params, opt_state, batch):
            return sft_train_step(params, opt_state, self.optimizer, batch, self.model)

        new_params, new_opt_state, loss = jit_step(self.params, self.opt_state, batch)
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Component: SFT Evaluation Step
# ---------------------------------------------------------------------------

class TestSFTEvalStep:
    """Tests for the SFT evaluation step."""

    def setup_method(self):
        self.model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, 16), dtype=jnp.int32)
        self.params = self.model.init(rng, dummy)

    def test_eval_returns_scalar(self):
        """Eval step should return a scalar loss."""
        batch = {
            "input_ids": jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, TEST_CONFIG.vocab_size),
            "labels": jax.random.randint(jax.random.PRNGKey(1), (4, 16), 0, TEST_CONFIG.vocab_size),
            "attention_mask": jnp.ones((4, 16)),
        }
        loss = sft_eval_step(self.params, batch, self.model)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_eval_matches_manual(self):
        """Eval loss should match manually computed loss."""
        batch = {
            "input_ids": jax.random.randint(jax.random.PRNGKey(0), (2, 16), 0, TEST_CONFIG.vocab_size),
            "labels": jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, TEST_CONFIG.vocab_size),
            "attention_mask": jnp.ones((2, 16)),
        }
        loss = sft_eval_step(self.params, batch, self.model)

        # Manual computation
        logits = self.model.apply(self.params, batch["input_ids"], deterministic=True)
        expected_loss = cross_entropy_loss(logits, batch["labels"])
        np.testing.assert_allclose(float(loss), float(expected_loss), atol=1e-5)
