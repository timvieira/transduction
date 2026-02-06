"""
Tests for StateLM KV cache behavior.

See: https://github.com/timvieira/transduction/issues/1

These tests verify that tree-structured branching (multiple children from one
parent) produces correct results. Currently passes with GPT-2 (tuple cache).
Will fail with models using DynamicCache until issue #1 is fixed.
"""
import pytest
import numpy as np

from transduction.lm import StateLM


class TestKVCacheBranching:
    """Test that tree-structured branching produces correct results."""

    @pytest.fixture
    def lm(self):
        return StateLM.initial('gpt2')

    def test_branching_produces_correct_logprobs(self, lm):
        """Two children from same parent should each get correct logprobs."""
        parent = lm << b' the'

        # Branch: create two children from the same parent
        child_a = parent << b' cat'
        child_b = parent << b' dog'

        # Evaluate both children
        logp_a = child_a.logp_next[b' is']
        logp_b = child_b.logp_next[b' is']

        # Both should be valid log probabilities
        assert np.isfinite(logp_a)
        assert np.isfinite(logp_b)

        # They should be different (different contexts)
        assert logp_a != logp_b

        # Verify against fresh computation (no shared cache)
        fresh_a = lm << b' the' << b' cat'
        fresh_b = lm << b' the' << b' dog'

        assert np.isclose(logp_a, fresh_a.logp_next[b' is'], rtol=1e-5)
        assert np.isclose(logp_b, fresh_b.logp_next[b' is'], rtol=1e-5)

    def test_deep_branching(self, lm):
        """Test branching at multiple depths."""
        root = lm << b' hello'

        # Branch at depth 1
        branch_a = root << b' world'
        branch_b = root << b' there'

        # Further branch from branch_a
        leaf_a1 = branch_a << b' how'
        leaf_a2 = branch_a << b' what'

        # Evaluate in interleaved order (stresses cache sharing)
        results = {
            'a1': leaf_a1.logp_next[b' are'],
            'b': branch_b.logp_next[b' how'],
            'a2': leaf_a2.logp_next[b' is'],
        }

        # All should be finite
        for key, val in results.items():
            assert np.isfinite(val), f"{key} logprob should be finite"

        # Verify against fresh computation
        fresh_a1 = lm << b' hello' << b' world' << b' how'
        fresh_a2 = lm << b' hello' << b' world' << b' what'
        fresh_b = lm << b' hello' << b' there'

        assert np.isclose(results['a1'], fresh_a1.logp_next[b' are'], rtol=1e-5)
        assert np.isclose(results['a2'], fresh_a2.logp_next[b' is'], rtol=1e-5)
        assert np.isclose(results['b'], fresh_b.logp_next[b' how'], rtol=1e-5)

    def test_many_children_from_one_parent(self, lm):
        """Stress test: many children branching from single parent."""
        parent = lm << b' the'

        # Create many children
        suffixes = [b' cat', b' dog', b' bird', b' fish', b' horse']
        children = [parent << s for s in suffixes]

        # Evaluate all children
        logps = [c.logp_next[b' is'] for c in children]

        # All should be finite and distinct
        assert all(np.isfinite(lp) for lp in logps)
        assert len(set(logps)) == len(logps), "All logprobs should be distinct"

        # Verify against fresh computation
        for suffix, child_logp in zip(suffixes, logps):
            fresh = lm << b' the' << suffix
            assert np.isclose(child_logp, fresh.logp_next[b' is'], rtol=1e-5)
