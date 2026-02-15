"""Tests for generation utilities: top-k sampling, nucleus sampling, and generate.

Run with: pytest tests/test_generation.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from utils.generation import top_k_sampling, nucleus_sampling, generate


VOCAB_SIZE = 64
RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Component: Top-K Sampling
# ---------------------------------------------------------------------------

class TestTopKSampling:
    """Tests for top_k_sampling."""

    def test_returns_scalar(self):
        """Sampled token should be a scalar."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        token = top_k_sampling(logits, RNG, k=10)
        assert token.shape == (), f"Expected scalar, got shape {token.shape}"

    def test_returns_valid_token_id(self):
        """Sampled token should be in [0, vocab_size)."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        token = top_k_sampling(logits, RNG, k=10)
        assert 0 <= int(token) < VOCAB_SIZE

    def test_samples_from_top_k_only(self):
        """Sampled token should always be among the top-k logits."""
        logits = jnp.arange(VOCAB_SIZE, dtype=jnp.float32)  # token 63 has highest logit
        k = 5
        top_k_ids = set(range(VOCAB_SIZE - k, VOCAB_SIZE))

        for i in range(50):
            rng = jax.random.PRNGKey(i)
            token = int(top_k_sampling(logits, rng, k=k))
            assert token in top_k_ids, (
                f"Token {token} not in top-{k} set {top_k_ids}"
            )

    def test_low_temperature_is_greedy(self):
        """Very low temperature should almost always pick the argmax."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        expected = int(jnp.argmax(logits))

        greedy_count = 0
        for i in range(50):
            rng = jax.random.PRNGKey(i)
            token = int(top_k_sampling(logits, rng, k=10, temperature=0.01))
            if token == expected:
                greedy_count += 1

        assert greedy_count >= 45, (
            f"With temperature=0.01, expected mostly argmax, got {greedy_count}/50"
        )

    def test_high_temperature_increases_diversity(self):
        """High temperature should spread probability across tokens."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        k = 10

        tokens_low_temp = set()
        tokens_high_temp = set()
        for i in range(100):
            rng = jax.random.PRNGKey(i)
            tokens_low_temp.add(int(top_k_sampling(logits, rng, k=k, temperature=0.1)))
            tokens_high_temp.add(int(top_k_sampling(logits, rng, k=k, temperature=5.0)))

        assert len(tokens_high_temp) >= len(tokens_low_temp), (
            f"High temp should give more diversity: {len(tokens_high_temp)} vs {len(tokens_low_temp)}"
        )

    def test_deterministic_with_same_rng(self):
        """Same RNG key should produce the same token."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        rng = jax.random.PRNGKey(7)
        t1 = top_k_sampling(logits, rng, k=10)
        t2 = top_k_sampling(logits, rng, k=10)
        assert int(t1) == int(t2)

    def test_different_rng_can_differ(self):
        """Different RNG keys should (eventually) produce different tokens."""
        logits = jnp.zeros(VOCAB_SIZE)  # uniform — high chance of different samples
        tokens = set()
        for i in range(50):
            rng = jax.random.PRNGKey(i)
            tokens.add(int(top_k_sampling(logits, rng, k=VOCAB_SIZE)))
        assert len(tokens) > 1, "Different RNG keys should produce at least some variation"


# ---------------------------------------------------------------------------
# Component: Nucleus (Top-P) Sampling
# ---------------------------------------------------------------------------

class TestNucleusSampling:
    """Tests for nucleus_sampling."""

    def test_returns_scalar(self):
        """Sampled token should be a scalar."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        token = nucleus_sampling(logits, RNG, p=0.9)
        assert token.shape == ()

    def test_returns_valid_token_id(self):
        """Sampled token should be in [0, vocab_size)."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        token = nucleus_sampling(logits, RNG, p=0.9)
        assert 0 <= int(token) < VOCAB_SIZE

    def test_small_p_concentrates_on_top(self):
        """Very small p should concentrate on the highest-probability token."""
        # Make one token dominate
        logits = jnp.zeros(VOCAB_SIZE)
        logits = logits.at[7].set(10.0)

        top_count = 0
        for i in range(50):
            rng = jax.random.PRNGKey(i)
            token = int(nucleus_sampling(logits, rng, p=0.01))
            if token == 7:
                top_count += 1

        assert top_count >= 45, (
            f"With p=0.01 and dominant token, expected mostly token 7, got {top_count}/50"
        )

    def test_p_one_allows_all_tokens(self):
        """p=1.0 should allow sampling from the full distribution."""
        logits = jnp.zeros(VOCAB_SIZE)  # uniform
        tokens = set()
        for i in range(200):
            rng = jax.random.PRNGKey(i)
            tokens.add(int(nucleus_sampling(logits, rng, p=1.0)))
        # With uniform logits and 200 samples over 64 tokens, expect good coverage
        assert len(tokens) > VOCAB_SIZE // 2, (
            f"p=1.0 with uniform logits should cover many tokens, got {len(tokens)}"
        )

    def test_deterministic_with_same_rng(self):
        """Same RNG key should produce the same token."""
        logits = jax.random.normal(RNG, (VOCAB_SIZE,))
        rng = jax.random.PRNGKey(7)
        t1 = nucleus_sampling(logits, rng, p=0.9)
        t2 = nucleus_sampling(logits, rng, p=0.9)
        assert int(t1) == int(t2)


# ---------------------------------------------------------------------------
# Helper: Dummy causal model for testing generate()
# ---------------------------------------------------------------------------

def _make_dummy_apply_fn(vocab_size: int, deterministic_token: int | None = None):
    """Create a dummy apply_fn for testing generate().

    If deterministic_token is given, the model always assigns the highest
    logit to that token (greedy decoding will always pick it).
    Otherwise, returns random-looking but deterministic logits.
    """
    def apply_fn(params, input_ids):
        batch_size, seq_len = input_ids.shape
        if deterministic_token is not None:
            logits = jnp.zeros((batch_size, seq_len, vocab_size))
            logits = logits.at[:, :, deterministic_token].set(10.0)
        else:
            # Use input_ids to produce deterministic but varied logits
            logits = jnp.sin(
                jnp.arange(vocab_size)[None, None, :]
                + input_ids[:, :, None].astype(jnp.float32)
            )
        return logits
    return apply_fn


# ---------------------------------------------------------------------------
# Component: Generate (while_loop)
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for the autoregressive generate function."""

    def test_output_shape(self):
        """Output should be (batch, prompt_len + max_new_tokens)."""
        batch_size, prompt_len, max_new_tokens = 2, 4, 8
        input_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        output = generate(apply_fn, None, input_ids, RNG, max_new_tokens=max_new_tokens)

        assert output.shape == (batch_size, prompt_len + max_new_tokens), (
            f"Expected {(batch_size, prompt_len + max_new_tokens)}, got {output.shape}"
        )

    def test_prompt_is_preserved(self):
        """The prompt tokens should appear unchanged at the start of the output."""
        input_ids = jax.random.randint(RNG, (2, 5), 1, VOCAB_SIZE)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        output = generate(apply_fn, None, input_ids, RNG, max_new_tokens=6)

        np.testing.assert_array_equal(
            np.array(output[:, :5]),
            np.array(input_ids),
            err_msg="Prompt tokens should be preserved in output",
        )

    def test_generated_tokens_are_valid(self):
        """Generated tokens should be valid token IDs in [0, vocab_size)."""
        input_ids = jnp.ones((2, 3), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        output = generate(
            apply_fn, None, input_ids, RNG,
            max_new_tokens=10, top_k=VOCAB_SIZE,
        )
        generated = output[:, 3:]  # only new tokens

        assert jnp.all(generated >= 0) and jnp.all(generated < VOCAB_SIZE), (
            "All generated tokens should be valid IDs"
        )

    def test_deterministic_with_same_rng(self):
        """Same RNG should produce identical outputs."""
        input_ids = jnp.ones((2, 3), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)
        rng = jax.random.PRNGKey(99)

        out1 = generate(apply_fn, None, input_ids, rng, max_new_tokens=8)
        out2 = generate(apply_fn, None, input_ids, rng, max_new_tokens=8)

        np.testing.assert_array_equal(np.array(out1), np.array(out2))

    def test_different_rng_gives_different_output(self):
        """Different RNG keys should (generally) produce different outputs."""
        input_ids = jnp.ones((1, 3), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        out1 = generate(apply_fn, None, input_ids, jax.random.PRNGKey(0), max_new_tokens=16)
        out2 = generate(apply_fn, None, input_ids, jax.random.PRNGKey(1), max_new_tokens=16)

        assert not jnp.array_equal(out1, out2), (
            "Different RNG keys should produce different outputs"
        )

    def test_greedy_with_low_temperature(self):
        """Low temperature + deterministic model should always pick the same token."""
        target_token = 7
        input_ids = jnp.ones((1, 3), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE, deterministic_token=target_token)

        output = generate(
            apply_fn, None, input_ids, RNG,
            max_new_tokens=5, temperature=0.01, top_k=10,
        )
        generated = output[0, 3:]

        np.testing.assert_array_equal(
            np.array(generated),
            np.full(5, target_token),
            err_msg=f"Low temp + dominant logit should always produce token {target_token}",
        )

    def test_eos_stops_generation(self):
        """When all sequences emit EOS, remaining positions should be padded with 0."""
        eos_id = 2
        # Model always outputs token 2 (the EOS) — generation should stop after step 1
        input_ids = jnp.ones((1, 3), dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE, deterministic_token=eos_id)

        output = generate(
            apply_fn, None, input_ids, RNG,
            max_new_tokens=10, temperature=0.01, top_k=10,
            eos_token_id=eos_id,
        )

        generated = output[0, 3:]  # positions after prompt

        # First generated token should be EOS
        assert int(generated[0]) == eos_id

        # All positions after the first EOS should be 0 (pad)
        np.testing.assert_array_equal(
            np.array(generated[1:]),
            np.zeros(9, dtype=np.int32),
            err_msg="Positions after EOS should be padded with 0",
        )

    def test_no_eos_generates_max_tokens(self):
        """Without eos_token_id, should generate exactly max_new_tokens."""
        input_ids = jnp.ones((2, 4), dtype=jnp.int32)
        max_new = 6
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        output = generate(
            apply_fn, None, input_ids, RNG,
            max_new_tokens=max_new, eos_token_id=None,
        )

        assert output.shape == (2, 4 + max_new)
        # All generated positions should be nonzero (no early stop padding)
        # This isn't strictly guaranteed but with vocab_size=64 and varied logits,
        # the chance of sampling token 0 for ALL positions is negligible.
        generated = output[:, 4:]
        assert jnp.any(generated != 0), "Without EOS, generation should fill all positions"

    def test_per_sequence_eos(self):
        """EOS should stop individual sequences, not the entire batch."""
        eos_id = 5

        # Custom apply_fn: sequence 0 always gets eos_id, sequence 1 gets token 10
        def apply_fn(params, input_ids):
            batch_size, seq_len = input_ids.shape
            logits = jnp.zeros((batch_size, seq_len, VOCAB_SIZE))
            # Sequence 0: dominant logit on eos_id
            logits = logits.at[0, :, eos_id].set(10.0)
            # Sequence 1: dominant logit on token 10
            logits = logits.at[1, :, 10].set(10.0)
            return logits

        input_ids = jnp.ones((2, 3), dtype=jnp.int32)
        output = generate(
            apply_fn, None, input_ids, RNG,
            max_new_tokens=5, temperature=0.01, top_k=10,
            eos_token_id=eos_id,
        )

        # Sequence 0: first generated token is EOS, rest should be 0
        assert int(output[0, 3]) == eos_id
        np.testing.assert_array_equal(
            np.array(output[0, 4:]),
            np.zeros(4, dtype=np.int32),
            err_msg="Seq 0 should be padded after EOS",
        )

        # Sequence 1: should generate token 10 for all positions
        np.testing.assert_array_equal(
            np.array(output[1, 3:]),
            np.full(5, 10, dtype=np.int32),
            err_msg="Seq 1 should keep generating (no EOS)",
        )

    def test_batch_size_one(self):
        """Should work correctly with batch_size=1."""
        input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        apply_fn = _make_dummy_apply_fn(VOCAB_SIZE)

        output = generate(apply_fn, None, input_ids, RNG, max_new_tokens=4)

        assert output.shape == (1, 7)
        np.testing.assert_array_equal(np.array(output[0, :3]), [1, 2, 3])
