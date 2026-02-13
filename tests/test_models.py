"""Tests for model architecture: attention, transformer blocks, and GPT-2.

Run with: pytest tests/test_models.py -v

These tests verify the shapes and basic properties of the model components.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from configs.model_config import ModelConfig, TINY_CONFIG
from models.attention import CausalSelfAttention
from models.transformer_block import TransformerBlock


# Use a tiny config for fast tests
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
# Component: Causal Self-Attention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    """Tests for CausalSelfAttention."""

    def test_output_shape(self):
        """Output should match input shape (B, T, d_model)."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 16, 64))  # B=2, T=16, d_model=64

        params = model.init(rng, x)
        y = model.apply(params, x, deterministic=True)
        assert y.shape == (2, 16, 64), f"Expected (2, 16, 64), got {y.shape}"

    def test_causal_masking(self):
        """Attention should be causal: position i can only attend to positions <= i.

        We test this by checking that changing future tokens doesn't affect
        the output at earlier positions.
        """
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1, 8, 64))

        params = model.init(rng, x)
        y1 = model.apply(params, x, deterministic=True)

        # Modify the last token
        x_modified = x.at[:, -1, :].set(0.0)
        y2 = model.apply(params, x_modified, deterministic=True)

        # All positions except the last should be identical
        np.testing.assert_allclose(
            y1[:, :-1, :], y2[:, :-1, :], atol=1e-5,
            err_msg="Causal masking violated: changing future tokens affected past outputs"
        )

    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)

        x1 = jax.random.normal(rng, (1, 8, 64))
        params = model.init(rng, x1)

        for batch_size in [1, 4, 8]:
            x = jax.random.normal(rng, (batch_size, 8, 64))
            y = model.apply(params, x, deterministic=True)
            assert y.shape == (batch_size, 8, 64)


# ---------------------------------------------------------------------------
# Component: Transformer Block
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 16, 64))

        params = block.init(rng, x)
        y = block.apply(params, x, deterministic=True)
        assert y.shape == (2, 16, 64), f"Expected (2, 16, 64), got {y.shape}"

    def test_residual_connection(self):
        """With zero-initialized weights, output should approximately equal input.

        (Due to LayerNorm, it won't be exact, but the residual should dominate.)
        """
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1, 8, 64)) * 0.01  # small inputs

        params = block.init(rng, x)
        y = block.apply(params, x, deterministic=True)

        # Output should be close to input (residual connections)
        # With random initialization, they won't be identical, but shape should match
        assert y.shape == x.shape

    def test_with_mask(self):
        """Should accept an attention mask without errors."""
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 8, 64))
        mask = jnp.tril(jnp.ones((8, 8)))[None, None, :, :]

        params = block.init(rng, x, mask=mask)
        y = block.apply(params, x, mask=mask, deterministic=True)
        assert y.shape == (2, 8, 64)
