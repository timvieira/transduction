"""
Tests for HuggingFaceLM / TokenIDState KV cache behavior.

See: https://github.com/timvieira/transduction/issues/1

These tests verify that tree-structured branching (multiple children from one
parent) produces correct results. Currently passes with GPT-2 (tuple cache).
Will fail with models using DynamicCache until issue #1 is fixed.
"""
import pytest
import numpy as np

from transduction.lm import HuggingFaceLM, load_model_by_name


@pytest.fixture(scope='module')
def lm():
    return load_model_by_name('gpt2')


@pytest.fixture(scope='module')
def enc(lm):
    """Shorthand: maps bytes tokens to int IDs."""
    return lm._encode


class TestKVCacheBranching:
    """Test that tree-structured branching produces correct results."""

    def test_branching_produces_correct_logprobs(self, lm, enc):
        """Two children from same parent should each get correct logprobs."""
        s0 = lm.initial()
        parent = s0 >> enc[b' the']

        # Branch: create two children from the same parent
        child_a = parent >> enc[b' cat']
        child_b = parent >> enc[b' dog']

        is_id = enc[b' is']
        logp_a = child_a.logp_next[is_id]
        logp_b = child_b.logp_next[is_id]

        # Both should be valid log probabilities
        assert np.isfinite(logp_a)
        assert np.isfinite(logp_b)

        # They should be different (different contexts)
        assert logp_a != logp_b

        # Verify against fresh computation (no shared cache)
        fresh_a = lm.initial() >> enc[b' the'] >> enc[b' cat']
        fresh_b = lm.initial() >> enc[b' the'] >> enc[b' dog']

        assert np.isclose(logp_a, fresh_a.logp_next[is_id], rtol=1e-5)
        assert np.isclose(logp_b, fresh_b.logp_next[is_id], rtol=1e-5)

    def test_deep_branching(self, lm, enc):
        """Test branching at multiple depths."""
        root = lm.initial() >> enc[b' hello']

        # Branch at depth 1
        branch_a = root >> enc[b' world']
        branch_b = root >> enc[b' there']

        # Further branch from branch_a
        leaf_a1 = branch_a >> enc[b' how']
        leaf_a2 = branch_a >> enc[b' what']

        # Evaluate in interleaved order (stresses cache sharing)
        results = {
            'a1': leaf_a1.logp_next[enc[b' are']],
            'b': branch_b.logp_next[enc[b' how']],
            'a2': leaf_a2.logp_next[enc[b' is']],
        }

        # All should be finite
        for key, val in results.items():
            assert np.isfinite(val), f"{key} logprob should be finite"

        # Verify against fresh computation
        fresh_a1 = lm.initial() >> enc[b' hello'] >> enc[b' world'] >> enc[b' how']
        fresh_a2 = lm.initial() >> enc[b' hello'] >> enc[b' world'] >> enc[b' what']
        fresh_b = lm.initial() >> enc[b' hello'] >> enc[b' there']

        assert np.isclose(results['a1'], fresh_a1.logp_next[enc[b' are']], rtol=1e-5)
        assert np.isclose(results['a2'], fresh_a2.logp_next[enc[b' is']], rtol=1e-5)
        assert np.isclose(results['b'], fresh_b.logp_next[enc[b' how']], rtol=1e-5)

    def test_many_children_from_one_parent(self, lm, enc):
        """Stress test: many children branching from single parent."""
        parent = lm.initial() >> enc[b' the']

        # Create many children
        suffix_ids = [enc[b' cat'], enc[b' dog'], enc[b' bird'], enc[b' fish'], enc[b' horse']]
        children = [parent >> s for s in suffix_ids]

        # Evaluate all children
        is_id = enc[b' is']
        logps = [c.logp_next[is_id] for c in children]

        # All should be finite and distinct
        assert all(np.isfinite(lp) for lp in logps)
        assert len(set(logps)) == len(logps), "All logprobs should be distinct"

        # Verify against fresh computation
        for suffix_id, child_logp in zip(suffix_ids, logps):
            fresh = lm.initial() >> enc[b' the'] >> suffix_id
            assert np.isclose(child_logp, fresh.logp_next[is_id], rtol=1e-5)


class TestHuggingFaceLM:
    """Test HuggingFaceLM core functionality."""

    def test_eos_round_trip(self, lm):
        """eos token ID should round-trip through _encode/_decode."""
        eos_bytes = lm._decode[lm.eos]
        assert lm.eos == lm._encode[eos_bytes]

    def test_initial_logp_next(self, lm):
        """Initial state should produce valid logp_next."""
        state = lm.initial()
        logp_next = state.logp_next
        # Should have valid log probs for all token IDs
        for tok_id in range(20):
            assert np.isfinite(logp_next[tok_id])

    def test_advance(self, lm, enc):
        """Advancing should produce different logp_next than initial."""
        s0 = lm.initial()
        s1 = s0 >> enc[b' the']

        # logp_next should differ after advancing
        tok_id = enc[b' cat']
        assert s0.logp_next[tok_id] != s1.logp_next[tok_id]

    def test_multi_step_advance(self, lm, enc):
        """Multi-step advance should accumulate logp correctly."""
        tokens = [enc[b' the'], enc[b' cat'], enc[b' is']]

        state = lm.initial()
        expected_logp = 0.0
        for tid in tokens:
            expected_logp += state.logp_next[tid]
            state = state >> tid

        assert np.isclose(state.logp, expected_logp, rtol=1e-5)

    def test_token_ids(self, lm, enc):
        """token_ids() should return the int token IDs fed in."""
        the_id = enc[b' the']
        cat_id = enc[b' cat']
        state = lm.initial() >> the_id >> cat_id
        assert state.token_ids() == [the_id, cat_id]

    def test_from_name(self):
        """HuggingFaceLM.from_name should construct a working LM."""
        lm = HuggingFaceLM.from_name('gpt2')
        state = lm.initial()
        assert hasattr(state, 'logp_next')
        assert isinstance(state.logp_next.argmax(), int)

    def test_intlazyprob_contains(self, lm):
        """TokenLogProbs __contains__ should reflect valid token IDs."""
        logp_next = lm.initial().logp_next
        # Valid token ID
        assert 0 in logp_next
        # Out of range
        assert -1 not in logp_next
        assert len(lm._decode) not in logp_next

    def test_intlazyprob_top(self, lm):
        """TokenLogProbs.top(K) should return K entries sorted by logp."""
        top5 = lm.initial().logp_next.top(5)
        assert len(top5) == 5
        vals = list(top5.values())
        assert vals == sorted(vals, reverse=True)

    def test_branching(self, lm, enc):
        """Tree-structured branching should work correctly with int IDs."""
        the_id = enc[b' the']
        cat_id = enc[b' cat']
        dog_id = enc[b' dog']
        is_id = enc[b' is']

        parent = lm.initial() >> the_id
        child_a = parent >> cat_id
        child_b = parent >> dog_id

        logp_a = child_a.logp_next[is_id]
        logp_b = child_b.logp_next[is_id]

        assert np.isfinite(logp_a)
        assert np.isfinite(logp_b)
        assert logp_a != logp_b
