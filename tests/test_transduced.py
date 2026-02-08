"""Tests for TransducedLM."""

import pytest
import numpy as np
from collections import defaultdict

from transduction import examples, FST
from transduction.lm.base import LMState
from transduction.lm.base import LogpNext
from transduction.lm.ngram import ByteNgramLM, CharNgramLM, NgramState
from transduction.lm.transduced import TransducedLM, TransducedState, logsumexp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_fst(alphabet):
    """Identity/copy transducer: maps each symbol to itself."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    for x in alphabet:
        fst.add_arc(0, x, x, 0)
    return fst


class TinyState(LMState):
    """Minimal LM state for testing with a fixed token distribution."""
    def __init__(self, probs, logp=0.0):
        self._probs = probs
        self.logp = logp

    @property
    def logp_next(self):
        return LogpNext(self._probs)

    @property
    def eos(self):
        return '<EOS>'

    def __lshift__(self, token):
        lp = self._probs.get(token, -np.inf)
        return TinyState(self._probs, self.logp + lp)


class TinyLM:
    def __init__(self):
        self.eos = '<EOS>'
    def initial(self):
        probs = {'a': np.log(0.6), 'b': np.log(0.3), '<EOS>': np.log(0.1)}
        return TinyState(probs)


def brute_force_pushforward(inner_lm, fst, target, max_source_len=8):
    """Exhaustive enumeration of P_fst(target_prefix + y) for all y.

    Uses FST.relation() to enumerate (source, output) pairs, scores each
    source with the inner LM, and accumulates by output string.

    Returns dict mapping output_string → log P.
    """
    output_probs = defaultdict(lambda: -np.inf)

    def source_logp(source):
        """Compute log P_inner(source) including EOS."""
        state = inner_lm.initial()(source)
        return state.logp + state.logp_next[state.eos]

    # Group by source to deduplicate outputs (relation() may yield
    # duplicate pairs when multiple state-paths produce the same strings).
    source_outputs = defaultdict(set)
    for source, output in fst.relation(max_source_len):
        source_outputs[source].add(output)

    for source, outputs in source_outputs.items():
        lp = source_logp(source)
        if lp == -np.inf:
            continue
        for out in outputs:
            output_probs[out] = np.logaddexp(output_probs[out], lp)

    return output_probs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ngram_lm():
    """Simple bigram LM trained on 'abab' repeated."""
    return ByteNgramLM.train(b"aabb" * 20, n=2, alpha=0.1)


@pytest.fixture
def char_ngram_lm():
    """Char-level n-gram for use with char-symbol FSTs."""
    return CharNgramLM.train("aabbaabb" * 5, n=2, alpha=0.5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransducedLM:

    def test_identity_transducer(self, char_ngram_lm):
        """TransducedLM with a copy FST should reproduce the inner LM's distribution."""
        alphabet = char_ngram_lm.alphabet
        # filter out EOS from FST alphabet
        fst_alpha = [s for s in alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial()

        # Check that logp_next distribution roughly matches the inner LM
        inner_state = char_ngram_lm.initial()
        tlm_lp = state.logp_next

        for y in fst_alpha:
            inner_lp = inner_state.logp_next[y]
            got = tlm_lp[y]
            # They should be close (not exact due to normalization over finite beam)
            if inner_lp > -10:  # only check non-negligible probabilities
                assert abs(got - inner_lp) < 1.0, \
                    f"Symbol {y!r}: inner={inner_lp:.4f}, transduced={got:.4f}"

    def test_identity_after_advance(self, char_ngram_lm):
        """After advancing by a symbol, identity transducer still tracks inner LM."""
        alphabet = char_ngram_lm.alphabet
        fst_alpha = [s for s in alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial() << 'a'

        inner_state = char_ngram_lm.initial() << 'a'
        tlm_lp = state.logp_next

        for y in fst_alpha:
            inner_lp = inner_state.logp_next[y]
            got = tlm_lp[y]
            if inner_lp > -10:
                assert abs(got - inner_lp) < 1.0, \
                    f"Symbol {y!r}: inner={inner_lp:.4f}, transduced={got:.4f}"

    def test_incremental_consistency(self, char_ngram_lm):
        """logp after << y1 << y2 equals sum of logp_next[y_i] from successive states."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        state0 = tlm.initial()
        lp1 = state0.logp_next['a']

        state1 = state0 << 'a'
        lp2 = state1.logp_next['b']

        state2 = state1 << 'b'

        expected = lp1 + lp2
        assert state2.logp == pytest.approx(expected, abs=1e-10)

    def test_small_fst_nontrivial(self, char_ngram_lm):
        """TransducedLM with examples.small() produces valid distributions."""
        fst = examples.small()
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)
        state = tlm.initial()

        # The small FST maps {a,b} → {x,a,b}, so 'x' should be reachable
        lp = state.logp_next
        scores = dict(lp.items())
        assert len(scores) > 0, "Should have at least one reachable output symbol"

        # Check that probabilities are valid (sum to ≤ 1, all ≤ 0)
        for y, s in scores.items():
            assert s <= 0.0 + 1e-10, f"logp for {y!r} should be ≤ 0, got {s}"

    def test_lowercase_fst(self, char_ngram_lm):
        """TransducedLM with the lowercase FST produces a valid distribution."""
        fst = examples.lowercase()
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)
        state = tlm.initial()

        lp = state.logp_next
        scores = dict(lp.items())
        assert len(scores) > 0

    def test_brute_force_comparison(self):
        """Compare TransducedLM against brute-force enumeration for a tiny FST."""
        inner_lm = TinyLM()
        # Copy FST over {a, b}
        fst = copy_fst(['a', 'b'])

        tlm = TransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = tlm.initial()

        # Brute force: P(empty target, then next symbol y)
        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)

        # Total probability across all output strings
        Z = logsumexp(list(bf.values()))

        # P(next = 'a' | empty prefix)
        # = sum over all strings starting with 'a' / Z
        a_strings = {k: v for k, v in bf.items() if k.startswith('a')}
        b_strings = {k: v for k, v in bf.items() if k.startswith('b')}
        _eos_strings = {k: v for k, v in bf.items() if k == ''}

        bf_a = logsumexp(list(a_strings.values())) - Z if a_strings else -np.inf
        bf_b = logsumexp(list(b_strings.values())) - Z if b_strings else -np.inf

        tlm_a = state.logp_next['a']
        tlm_b = state.logp_next['b']

        # Should be approximately equal (bounded search may not be exact)
        assert abs(tlm_a - bf_a) < 0.5, f"a: bf={bf_a:.4f}, tlm={tlm_a:.4f}"
        assert abs(tlm_b - bf_b) < 0.5, f"b: bf={bf_b:.4f}, tlm={tlm_b:.4f}"

    def test_path_recovery(self, char_ngram_lm):
        """Path recovery returns the correct sequence of target symbols."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        state = tlm.initial() << 'a' << 'b' << 'a'
        assert state.path() == ['a', 'b', 'a']

    def test_logp_starts_at_zero(self, char_ngram_lm):
        """Initial state has logp = 0."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst)
        state = tlm.initial()
        assert state.logp == 0.0

    def test_repr(self, char_ngram_lm):
        """Repr doesn't crash."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst)
        state = tlm.initial()
        assert 'TransducedState' in repr(state)
        assert 'TransducedLM' in repr(tlm)

    def test_eos_copy_fst_matches_inner(self):
        """With identity/copy FST, P(EOS) from TransducedLM should match inner LM."""

        class TinyState:
            def __init__(self, probs, logp=0.0, history=()):
                self._probs = probs
                self.logp = logp
                self.history = history
                self.eos = '<EOS>'

            @property
            def logp_next(self):
                return LogpNext(self._probs)

            def __lshift__(self, token):
                lp = self._probs.get(token, -np.inf)
                return TinyState(self._probs, self.logp + lp,
                                 history=(self.history, token))

        class TinyLM:
            def __init__(self):
                self.eos = '<EOS>'
            def initial(self):
                probs = {'a': np.log(0.6), 'b': np.log(0.3), '<EOS>': np.log(0.1)}
                return TinyState(probs)

        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        tlm = TransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = tlm.initial()

        # With copy FST, the transduced EOS probability should match the
        # inner LM's EOS probability (normalized over {a, b, EOS}).
        inner_state = inner_lm.initial()
        inner_eos = inner_state.logp_next['<EOS>']
        inner_a = inner_state.logp_next['a']
        inner_b = inner_state.logp_next['b']
        inner_Z = logsumexp([inner_eos, inner_a, inner_b])
        expected_eos = inner_eos - inner_Z

        got_eos = state.logp_next['<EOS>']
        assert abs(got_eos - expected_eos) < 0.5, \
            f"EOS: expected={expected_eos:.4f}, got={got_eos:.4f}"

    def test_eos_small_fst_nontrivial(self, char_ngram_lm):
        """With examples.small(), EOS probability should be > -inf from initial state.

        The small() FST has state 0 as both initial and final, so the empty
        source string (just EOS) maps to the empty target — meaning P(EOS)
        from the initial state should be non-trivial.
        """
        fst = examples.small()
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)
        state = tlm.initial()

        eos_lp = state.logp_next[state.eos]
        assert eos_lp > -np.inf, "EOS should be reachable from initial state"
        assert eos_lp < 0.0, f"EOS log-prob should be negative, got {eos_lp}"

    def test_eos_normalization(self):
        """logp_next (including EOS) should sum to approximately 1."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        tlm = TransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = tlm.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in ['a', 'b']] + [lp[state.eos]]
        total = logsumexp(all_logps)

        assert abs(total) < 0.1, \
            f"Probabilities should sum to ~1 (log ~0), got log-sum={total:.6f}"


class TestLMState:

    def test_inheritance(self):
        """All state classes inherit from LMState."""
        assert issubclass(NgramState, LMState)
        assert issubclass(TransducedState, LMState)

    def test_ngram_greedy_decode(self, ngram_lm):
        """NgramState.greedy_decode returns a list of byte tokens."""
        state = ngram_lm.initial()
        tokens = state.greedy_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
        for t in tokens:
            assert isinstance(t, bytes)

    def test_ngram_sample_decode(self, ngram_lm):
        """NgramState.sample_decode returns a list of byte tokens."""
        state = ngram_lm.initial()
        tokens = state.sample_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
        for t in tokens:
            assert isinstance(t, bytes)

    def test_ngram_greedy_deterministic(self, ngram_lm):
        """Greedy decode is deterministic."""
        state = ngram_lm.initial()
        t1 = state.greedy_decode(max_len=10)
        t2 = state.greedy_decode(max_len=10)
        assert t1 == t2

    def test_transduced_greedy_decode(self, char_ngram_lm):
        """TransducedState.greedy_decode returns a list of string tokens."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial()
        tokens = state.greedy_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
        for t in tokens:
            assert isinstance(t, str)

    def test_transduced_sample_decode(self, char_ngram_lm):
        """TransducedState.sample_decode returns a list of string tokens."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial()
        tokens = state.sample_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10

    def test_greedy_stops_at_eos(self):
        """Greedy decode stops when EOS is the argmax."""
        # LM that always prefers EOS
        lm = ByteNgramLM.train(b"\x00" * 100, n=1, alpha=0.001)
        state = lm.initial()
        tokens = state.greedy_decode(max_len=50)
        # Should stop immediately since EOS (b'\x00') dominates
        assert len(tokens) == 0

    def test_sample_respects_max_len(self, ngram_lm):
        """sample_decode respects max_len."""
        state = ngram_lm.initial()
        tokens = state.sample_decode(max_len=3)
        assert len(tokens) <= 3

    # --- __call__ tests ---

    def test_ngram_advance(self, ngram_lm):
        """__call__ on NgramState matches sequential <<."""
        state = ngram_lm.initial()
        s1 = state << b'a' << b'b'
        s2 = state([b'a', b'b'])
        assert s1.logp == pytest.approx(s2.logp)

    def test_transduced_advance(self, char_ngram_lm):
        """__call__ on TransducedState matches sequential <<."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial()
        s1 = state << 'a' << 'b'
        s2 = state(['a', 'b'])
        assert s1.logp == pytest.approx(s2.logp)

    def test_advance_empty(self, ngram_lm):
        """__call__ with empty sequence returns same state."""
        state = ngram_lm.initial()
        s = state([])
        assert s is state

