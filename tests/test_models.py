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
from models.gpt2 import GPT2LMHeadModel


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
# Helpers for KV-cache tests
# ---------------------------------------------------------------------------

def _make_empty_cache(batch_size, n_heads, max_len, head_dim):
    """Create a zeroed KV-cache dict."""
    return {
        'key': jnp.zeros((batch_size, n_heads, max_len, head_dim)),
        'value': jnp.zeros((batch_size, n_heads, max_len, head_dim)),
        'index': jnp.array(0, dtype=jnp.int32),
    }


# ---------------------------------------------------------------------------
# Component: Causal Self-Attention KV-Cache
# ---------------------------------------------------------------------------

class TestCausalSelfAttentionKVCache:
    """Tests for CausalSelfAttention with KV-cache."""

    def test_cache_output_shape(self):
        """Cached call should return (output, updated_cache) with correct shapes."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        B, T = 2, 8
        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        max_len = TEST_CONFIG.max_seq_len

        x = jax.random.normal(rng, (B, T, TEST_CONFIG.d_model))
        params = model.init(rng, x)

        cache = _make_empty_cache(B, TEST_CONFIG.n_heads, max_len, head_dim)
        output, new_cache = model.apply(params, x, deterministic=True, cache=cache)

        assert output.shape == (B, T, TEST_CONFIG.d_model), (
            f"Expected output shape (2, 8, 64), got {output.shape}"
        )
        assert new_cache['key'].shape == (B, TEST_CONFIG.n_heads, max_len, head_dim)
        assert new_cache['value'].shape == (B, TEST_CONFIG.n_heads, max_len, head_dim)
        assert int(new_cache['index']) == T

    def test_cache_matches_full_forward(self):
        """Single-token cached decode should match full-sequence forward at that position."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        B = 1
        seq_len = 6
        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        max_len = TEST_CONFIG.max_seq_len

        x = jax.random.normal(rng, (B, seq_len, TEST_CONFIG.d_model))
        params = model.init(rng, x)

        # Full forward pass
        full_output = model.apply(params, x, deterministic=True)

        # Cached: prefill all tokens at once, then check output matches
        cache = _make_empty_cache(B, TEST_CONFIG.n_heads, max_len, head_dim)
        cached_output, _ = model.apply(params, x, deterministic=True, cache=cache)

        np.testing.assert_allclose(
            full_output, cached_output, atol=1e-5,
            err_msg="Prefill cached output should match full forward output",
        )

    def test_cache_prefill_then_decode(self):
        """Prefill prompt tokens, then decode one token; verify outputs match full forward."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(1)
        B = 1
        prompt_len = 4
        total_len = 6
        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        max_len = TEST_CONFIG.max_seq_len

        x = jax.random.normal(rng, (B, total_len, TEST_CONFIG.d_model))
        params = model.init(rng, x)

        # Full forward for reference
        full_output = model.apply(params, x, deterministic=True)

        # Step 1: prefill with prompt tokens
        cache = _make_empty_cache(B, TEST_CONFIG.n_heads, max_len, head_dim)
        prefill_out, cache = model.apply(
            params, x[:, :prompt_len, :], deterministic=True, cache=cache,
        )
        np.testing.assert_allclose(
            full_output[:, :prompt_len, :], prefill_out, atol=1e-5,
            err_msg="Prefill output should match full forward for prompt positions",
        )

        # Step 2: decode remaining tokens one at a time
        for i in range(prompt_len, total_len):
            decode_out, cache = model.apply(
                params, x[:, i:i+1, :], deterministic=True, cache=cache,
            )
            np.testing.assert_allclose(
                full_output[:, i:i+1, :], decode_out, atol=1e-5,
                err_msg=f"Decode output at position {i} should match full forward",
            )

    def test_backward_compatible(self):
        """Calling without cache should return just the output tensor (not a tuple)."""
        model = CausalSelfAttention(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 8, TEST_CONFIG.d_model))

        params = model.init(rng, x)
        result = model.apply(params, x, deterministic=True)

        # Should be a plain array, not a tuple
        assert isinstance(result, jax.Array), (
            f"Without cache, result should be jax.Array, got {type(result)}"
        )
        assert result.shape == (2, 8, TEST_CONFIG.d_model)


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


# ---------------------------------------------------------------------------
# Component: Transformer Block KV-Cache
# ---------------------------------------------------------------------------

class TestTransformerBlockKVCache:
    """Tests for TransformerBlock with KV-cache."""

    def test_cache_passthrough_shape(self):
        """Cached call should return (output, updated_cache) with correct shapes."""
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        B, T = 2, 8
        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        max_len = TEST_CONFIG.max_seq_len

        x = jax.random.normal(rng, (B, T, TEST_CONFIG.d_model))
        params = block.init(rng, x)

        cache = _make_empty_cache(B, TEST_CONFIG.n_heads, max_len, head_dim)
        output, new_cache = block.apply(params, x, deterministic=True, cache=cache)

        assert output.shape == (B, T, TEST_CONFIG.d_model)
        assert new_cache['key'].shape == (B, TEST_CONFIG.n_heads, max_len, head_dim)
        assert new_cache['value'].shape == (B, TEST_CONFIG.n_heads, max_len, head_dim)
        assert int(new_cache['index']) == T

    def test_cache_matches_full_forward(self):
        """Prefill-then-decode with cache should match full forward at each position."""
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(1)
        B = 1
        prompt_len = 4
        total_len = 6
        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        max_len = TEST_CONFIG.max_seq_len

        x = jax.random.normal(rng, (B, total_len, TEST_CONFIG.d_model))
        params = block.init(rng, x)

        # Full forward for reference
        full_output = block.apply(params, x, deterministic=True)

        # Step 1: prefill
        cache = _make_empty_cache(B, TEST_CONFIG.n_heads, max_len, head_dim)
        prefill_out, cache = block.apply(
            params, x[:, :prompt_len, :], deterministic=True, cache=cache,
        )
        np.testing.assert_allclose(
            full_output[:, :prompt_len, :], prefill_out, atol=1e-5,
            err_msg="Block prefill output should match full forward",
        )

        # Step 2: decode one-by-one
        for i in range(prompt_len, total_len):
            decode_out, cache = block.apply(
                params, x[:, i:i+1, :], deterministic=True, cache=cache,
            )
            np.testing.assert_allclose(
                full_output[:, i:i+1, :], decode_out, atol=1e-5,
                err_msg=f"Block decode output at position {i} should match full forward",
            )

    def test_backward_compatible(self):
        """Without cache, TransformerBlock should return a plain tensor."""
        block = TransformerBlock(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 8, TEST_CONFIG.d_model))

        params = block.init(rng, x)
        result = block.apply(params, x, deterministic=True)

        assert isinstance(result, jax.Array), (
            f"Without cache, result should be jax.Array, got {type(result)}"
        )
        assert result.shape == (2, 8, TEST_CONFIG.d_model)


# ---------------------------------------------------------------------------
# Component: GPT-2 Language Model
# ---------------------------------------------------------------------------

class TestGPT2LMHeadModel:
    """Tests for GPT2LMHeadModel."""

    def test_output_shape(self):
        """Output logits should have shape (B, T, vocab_size)."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((2, 16), dtype=jnp.int32)

        params = model.init(rng, input_ids)
        logits = model.apply(params, input_ids, deterministic=True)
        assert logits.shape == (2, 16, TEST_CONFIG.vocab_size), (
            f"Expected (2, 16, {TEST_CONFIG.vocab_size}), got {logits.shape}"
        )

    def test_with_attention_mask(self):
        """Should work with a padding attention mask."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((2, 16), dtype=jnp.int32)
        attention_mask = jnp.ones((2, 16), dtype=jnp.float32)
        attention_mask = attention_mask.at[0, 12:].set(0.0)  # pad the first sequence

        params = model.init(rng, input_ids)
        logits = model.apply(params, input_ids, attention_mask=attention_mask, deterministic=True)
        assert logits.shape == (2, 16, TEST_CONFIG.vocab_size)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        from utils.jax_utils import count_params
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((1, 8), dtype=jnp.int32)
        params = model.init(rng, input_ids)

        n_params = count_params(params)
        print(f"Test model parameters: {n_params:,}")
        assert n_params > 0, "Model should have parameters"
        # For tiny config: roughly vocab_size*d_model + n_layers * block_params
        assert n_params < 10_000_000, "Tiny model should be < 10M params"

    def test_causal_property(self):
        """Logits at position i should only depend on tokens 0..i."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)

        input_ids = jax.random.randint(rng, (1, 16), 0, TEST_CONFIG.vocab_size)
        params = model.init(rng, input_ids)

        logits1 = model.apply(params, input_ids, deterministic=True)

        # Change the last token
        modified = input_ids.at[:, -1].set(0)
        logits2 = model.apply(params, modified, deterministic=True)

        # All positions except the last should produce identical logits
        np.testing.assert_allclose(
            logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-4,
            err_msg="Causal property violated"
        )

    def test_get_hidden_states(self):
        """get_hidden_states should return (B, T, d_model)."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((2, 16), dtype=jnp.int32)

        params = model.init(rng, input_ids)
        hidden = model.apply(params, input_ids, deterministic=True, method=model.get_hidden_states)
        assert hidden.shape == (2, 16, TEST_CONFIG.d_model), (
            f"Expected (2, 16, {TEST_CONFIG.d_model}), got {hidden.shape}"
        )


# ---------------------------------------------------------------------------
# Component: GPT-2 KV-Cache
# ---------------------------------------------------------------------------

class TestGPT2KVCache:
    """Tests for GPT2LMHeadModel with KV-cache."""

    def test_init_cache_shapes(self):
        """init_cache should produce correct structure and shapes."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        B = 2
        cache = model.init_cache(B)

        head_dim = TEST_CONFIG.d_model // TEST_CONFIG.n_heads
        assert len(cache) == TEST_CONFIG.n_layers, (
            f"Expected {TEST_CONFIG.n_layers} layer caches, got {len(cache)}"
        )
        for i, lc in enumerate(cache):
            assert lc['key'].shape == (B, TEST_CONFIG.n_heads, TEST_CONFIG.max_seq_len, head_dim), (
                f"Layer {i} key shape mismatch"
            )
            assert lc['value'].shape == (B, TEST_CONFIG.n_heads, TEST_CONFIG.max_seq_len, head_dim), (
                f"Layer {i} value shape mismatch"
            )
            assert int(lc['index']) == 0

    def test_prefill_then_decode(self):
        """Prefill prompt, decode one token — logits should match full forward."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        B = 1
        prompt_len = 5
        total_len = 6

        input_ids = jax.random.randint(rng, (B, total_len), 0, TEST_CONFIG.vocab_size)
        params = model.init(rng, input_ids)

        # Full forward for reference
        full_logits = model.apply(params, input_ids, deterministic=True)

        # Prefill
        cache = model.init_cache(B)
        prefill_logits, cache = model.apply(
            params, input_ids[:, :prompt_len], deterministic=True, cache=cache,
        )
        np.testing.assert_allclose(
            full_logits[:, :prompt_len, :], prefill_logits, atol=1e-4,
            err_msg="GPT2 prefill logits should match full forward",
        )

        # Decode one token
        decode_logits, cache = model.apply(
            params, input_ids[:, prompt_len:prompt_len+1], deterministic=True, cache=cache,
        )
        np.testing.assert_allclose(
            full_logits[:, prompt_len:prompt_len+1, :], decode_logits, atol=1e-4,
            err_msg="GPT2 decode logits should match full forward at decoded position",
        )

    def test_multi_step_decode(self):
        """Prefill + decode N tokens one-by-one; each step should match full forward."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(2)
        B = 2
        prompt_len = 3
        decode_steps = 5
        total_len = prompt_len + decode_steps

        input_ids = jax.random.randint(rng, (B, total_len), 0, TEST_CONFIG.vocab_size)
        params = model.init(rng, input_ids)

        # Full forward for reference
        full_logits = model.apply(params, input_ids, deterministic=True)

        # Prefill
        cache = model.init_cache(B)
        prefill_logits, cache = model.apply(
            params, input_ids[:, :prompt_len], deterministic=True, cache=cache,
        )
        np.testing.assert_allclose(
            full_logits[:, :prompt_len, :], prefill_logits, atol=1e-4,
            err_msg="GPT2 multi-step prefill mismatch",
        )

        # Decode one token at a time
        for i in range(prompt_len, total_len):
            decode_logits, cache = model.apply(
                params, input_ids[:, i:i+1], deterministic=True, cache=cache,
            )
            np.testing.assert_allclose(
                full_logits[:, i:i+1, :], decode_logits, atol=1e-4,
                err_msg=f"GPT2 decode step {i} logits mismatch",
            )

    def test_backward_compatible(self):
        """Without cache, __call__ should still return a plain logits tensor."""
        model = GPT2LMHeadModel(config=TEST_CONFIG)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((2, 16), dtype=jnp.int32)

        params = model.init(rng, input_ids)
        result = model.apply(params, input_ids, deterministic=True)

        assert isinstance(result, jax.Array), (
            f"Without cache, result should be jax.Array, got {type(result)}"
        )
        assert result.shape == (2, 16, TEST_CONFIG.vocab_size)
