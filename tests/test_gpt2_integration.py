"""End-to-end tests for competitive methods (CharacterBeam, GeneralizedBeam,
FusedTransducedLM) with GPT-2 as the inner language model.

Tests cover:
- DynamicCache fix: GPT-2 no longer raises TypeError
- CharacterBeam + GPT-2: initial state has finite logp_next, advance works
- GeneralizedBeam + GPT-2: initial state has finite logp_next, advance works
- FusedTransducedLM + GPT-2: initial state has finite entries
- Batching correctness: prefetched siblings match sequential computation
"""

import pytest
import numpy as np

from transduction.util import set_memory_limit

set_memory_limit(10)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def lm():
    from transduction.lm import load_model_by_name
    return load_model_by_name('gpt2')


@pytest.fixture(scope='module')
def enc(lm):
    return lm._encode


@pytest.fixture(scope='module')
def bpe_fst(lm):
    """Small subsampled BPE FST for GPT-2 (~100 tokens)."""
    from transduction.fst import FST
    from transduction.fsa import EPSILON
    from transduction.lm.huggingface_lm import HfTokenizerVocab

    vocab = HfTokenizerVocab(lm.tokenizer)
    drop = {x.encode() for x in lm.tokenizer.all_special_tokens}

    # Use first 100 non-special tokens
    used = sorted(
        i for i in range(len(vocab.decode))
        if vocab.decode[i] is not None and vocab.decode[i] not in drop
    )[:100]

    m = FST()
    m.add_start(())
    for i in used:
        x = vocab.decode[i]
        bx = tuple(x)
        for j in range(len(bx)):
            m.add_arc(bx[:j], EPSILON, bx[j], bx[:j + 1])
        m.add_arc(bx, i, EPSILON, ())
    m.add_stop(())
    return m


@pytest.fixture(scope='module')
def cb_vocab(lm):
    """Vocab dict for CharacterBeam: token_id -> bytes."""
    from transduction.lm.huggingface_lm import HfTokenizerVocab

    vocab = HfTokenizerVocab(lm.tokenizer)
    drop = {x.encode() for x in lm.tokenizer.all_special_tokens}

    cb_v: dict = {}
    used = sorted(
        i for i in range(len(vocab.decode))
        if vocab.decode[i] is not None and vocab.decode[i] not in drop
    )[:100]
    for tid in used:
        word = vocab.decode[tid]
        if word is not None:
            cb_v[tid] = word

    eos_bytes = lm.tokenizer.eos_token.encode()
    eos_id = vocab.encode[eos_bytes]
    if eos_id not in cb_v:
        cb_v[eos_id] = eos_bytes
    return cb_v


# ---------------------------------------------------------------------------
# TestDynamicCacheFix
# ---------------------------------------------------------------------------

class TestDynamicCacheFix:

    def test_no_typeerror_on_branching(self, lm, enc):
        """DynamicCache should be converted to tuples, not raise TypeError."""
        parent = lm.initial() >> enc[b' the']
        # These should not raise
        child_a = parent >> enc[b' cat']
        child_b = parent >> enc[b' dog']
        assert np.isfinite(child_a.logp_next[enc[b' is']])
        assert np.isfinite(child_b.logp_next[enc[b' is']])

    def test_cache_is_tuple_format(self, lm, enc):
        """After forward pass, past_key_values should be tuple-of-tuples."""
        state = lm.initial() >> enc[b' the']
        kv = state.out.past_key_values
        assert isinstance(kv, tuple), f"Expected tuple, got {type(kv)}"
        assert isinstance(kv[0], tuple), f"Expected tuple layer, got {type(kv[0])}"


# ---------------------------------------------------------------------------
# TestCharacterBeamGPT2
# ---------------------------------------------------------------------------

class TestCharacterBeamGPT2:

    def test_initial_logp_next_is_finite(self, lm, cb_vocab):
        from transduction.lm.character_beam import CharacterBeam
        cb = CharacterBeam(lm, cb_vocab, K=5, eos_token=lm.eos)
        state = cb.initial()
        logp = state.logp_next
        # Should have at least some finite entries
        finite_keys = [k for k in logp.keys() if isinstance(k, int) and logp[k] > -50]
        assert len(finite_keys) > 0, "No finite entries in initial logp_next"

    def test_advance_several_bytes(self, lm, cb_vocab):
        from transduction.lm.character_beam import CharacterBeam
        cb = CharacterBeam(lm, cb_vocab, K=5, eos_token=lm.eos)
        state = cb.initial()
        # Advance through a few bytes
        for byte_val in b'The':
            logp = state.logp_next
            if byte_val in logp:
                state = state >> byte_val
            else:
                break
        # State should still have valid logp_next
        finite_keys = [k for k in state.logp_next.keys()
                       if isinstance(k, int) and state.logp_next[k] > -50]
        assert len(finite_keys) > 0

    def test_logp_accumulates(self, lm, cb_vocab):
        from transduction.lm.character_beam import CharacterBeam
        cb = CharacterBeam(lm, cb_vocab, K=5, eos_token=lm.eos)
        state = cb.initial()
        total_logp = 0.0
        for byte_val in b'Th':
            if byte_val in state.logp_next:
                total_logp += state.logp_next[byte_val]
                state = state >> byte_val
        assert np.isclose(state.logp, total_logp, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestGeneralizedBeamGPT2
# ---------------------------------------------------------------------------

class TestGeneralizedBeamGPT2:

    def test_initial_logp_next_is_finite(self, lm, bpe_fst):
        from transduction.lm.generalized_beam import GeneralizedBeam
        gb = GeneralizedBeam(lm, bpe_fst, K=5, max_beam=10, max_steps=100)
        state = gb.initial()
        logp = state.logp_next
        finite_keys = [k for k in logp.keys() if logp[k] > -50]
        assert len(finite_keys) > 0, "No finite entries in initial logp_next"

    def test_advance_works(self, lm, bpe_fst):
        from transduction.lm.generalized_beam import GeneralizedBeam
        gb = GeneralizedBeam(lm, bpe_fst, K=5, max_beam=10, max_steps=100)
        state = gb.initial()
        logp = state.logp_next
        # Pick the best non-EOS symbol to advance
        best = logp.argmax()
        if best != state.eos:
            state2 = state >> best
            logp2 = state2.logp_next
            finite_keys = [k for k in logp2.keys() if logp2[k] > -50]
            assert len(finite_keys) > 0


# ---------------------------------------------------------------------------
# TestFusedTransducedLMGPT2
# ---------------------------------------------------------------------------

class TestFusedTransducedLMGPT2:

    def test_initial_logp_next_has_finite_entries(self, lm, bpe_fst):
        from transduction.lm.fused_transduced import FusedTransducedLM
        fused = FusedTransducedLM(lm, bpe_fst, max_steps=100, max_beam=10,
                                  helper='python')
        state = fused.initial()
        logp = state.logp_next
        finite_keys = [k for k in logp.keys() if logp[k] > -50]
        assert len(finite_keys) > 0, "No finite entries in initial logp_next"


# ---------------------------------------------------------------------------
# TestBatchingCorrectness
# ---------------------------------------------------------------------------

class TestBatchingCorrectness:

    def test_prefetched_matches_sequential(self, lm, enc):
        """Prefetched siblings should produce the same logp_next as sequential."""
        parent = lm.initial() >> enc[b' the']

        # Sequential: create children one at a time (each triggers lazy eval)
        seq_a = parent >> enc[b' cat']
        seq_b = parent >> enc[b' dog']
        seq_c = parent >> enc[b' bird']

        seq_logps = {
            'a': seq_a.logp_next[enc[b' is']],
            'b': seq_b.logp_next[enc[b' is']],
            'c': seq_c.logp_next[enc[b' is']],
        }

        # Batched: create fresh children from a fresh parent, prefetch first
        parent2 = lm.initial() >> enc[b' the']
        batch_a = parent2 >> enc[b' cat']
        batch_b = parent2 >> enc[b' dog']
        batch_c = parent2 >> enc[b' bird']

        # Prefetch before accessing logp_next
        lm.prefetch([batch_a, batch_b, batch_c])

        batch_logps = {
            'a': batch_a.logp_next[enc[b' is']],
            'b': batch_b.logp_next[enc[b' is']],
            'c': batch_c.logp_next[enc[b' is']],
        }

        for key in seq_logps:
            assert np.isclose(seq_logps[key], batch_logps[key], atol=1e-4), \
                f"Mismatch for child {key}: seq={seq_logps[key]:.6f}, batch={batch_logps[key]:.6f}"

    def test_prefetch_skips_already_computed(self, lm, enc):
        """Prefetch should skip states that already have cached out."""
        parent = lm.initial() >> enc[b' the']

        child_a = parent >> enc[b' cat']
        # Force computation of child_a
        _ = child_a.logp_next

        child_b = parent >> enc[b' dog']

        calls_before = lm._calls
        lm.prefetch([child_a, child_b])
        # child_a was already computed, child_b is a singleton group (skipped).
        # So no batch call should happen.
        assert lm._calls == calls_before

    def test_prefetch_noop_for_initial_state(self, lm):
        """Prefetch on initial state (no parent) should be no-op."""
        state = lm.initial()
        calls_before = lm._calls
        lm.prefetch([state])
        assert lm._calls == calls_before

    def test_prefetch_reduces_call_count(self, lm, enc):
        """Prefetch of N siblings should use 1 call instead of N."""
        parent = lm.initial() >> enc[b' the']
        children = [parent >> enc[tok] for tok in
                     [b' cat', b' dog', b' bird', b' fish', b' horse']]

        calls_before = lm._calls
        lm.prefetch(children)
        calls_after = lm._calls

        # Should be 1 batched call (not 5 individual calls)
        assert calls_after - calls_before == 1

        # All children should now have cached results
        for child in children:
            assert 'out' in child.__dict__
            assert np.isfinite(child.logp_next[enc[b' is']])
