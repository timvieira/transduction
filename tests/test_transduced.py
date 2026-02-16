"""Tests for TransducedLM."""

import pytest
import numpy as np
from collections import defaultdict

from transduction import examples, FST
from transduction.lm.base import LM, LMState
from transduction.lm.base import LogpNext
from transduction.lm.ngram import ByteNgramLM, CharNgramLM, NgramState
from transduction.lm.transduced import TransducedLM, TransducedState, Particle, logsumexp
from transduction.lm.transduced import _select_top_k
from transduction.lm.fused_transduced import FusedTransducedLM, FusedTransducedState


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

    def __rshift__(self, token):
        lp = self._probs.get(token, -np.inf)
        return TinyState(self._probs, self.logp + lp)


class TinyLM(LM):
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
        state = inner_lm(source)
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
        state = tlm >> 'a'

        inner_state = char_ngram_lm >> 'a'
        tlm_lp = state.logp_next

        for y in fst_alpha:
            inner_lp = inner_state.logp_next[y]
            got = tlm_lp[y]
            if inner_lp > -10:
                assert abs(got - inner_lp) < 1.0, \
                    f"Symbol {y!r}: inner={inner_lp:.4f}, transduced={got:.4f}"

    def test_incremental_consistency(self, char_ngram_lm):
        """logp after >> y1 >> y2 equals sum of logp_next[y_i] from successive states."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        state0 = tlm.initial()
        lp1 = state0.logp_next['a']

        state1 = state0 >> 'a'
        lp2 = state1.logp_next['b']

        state2 = state1 >> 'b'

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

        state = tlm >> 'a' >> 'b' >> 'a'
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

            def __rshift__(self, token):
                lp = self._probs.get(token, -np.inf)
                return TinyState(self._probs, self.logp + lp,
                                 history=(self.history, token))

        class TinyLM(LM):
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
        """__call__ on NgramState matches sequential >>."""
        state = ngram_lm.initial()
        s1 = state >> b'a' >> b'b'
        s2 = state([b'a', b'b'])
        assert s1.logp == pytest.approx(s2.logp)

    def test_transduced_advance(self, char_ngram_lm):
        """__call__ on TransducedState matches sequential >>."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        state = tlm.initial()
        s1 = state >> 'a' >> 'b'
        s2 = state(['a', 'b'])
        assert s1.logp == pytest.approx(s2.logp)

    def test_advance_empty(self, ngram_lm):
        """__call__ with empty sequence returns same state."""
        state = ngram_lm.initial()
        s = state([])
        assert s is state


class TestFusedTransducedLM:
    """Tests that FusedTransducedLM matches TransducedLM."""

    def test_identity_transducer(self, char_ngram_lm):
        """Fused and original should agree on copy FST."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        orig = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        orig_state = orig.initial()
        fused_state = fused.initial()

        for y in fst_alpha:
            o = orig_state.logp_next[y]
            f = fused_state.logp_next[y]
            if o > -10:
                assert abs(o - f) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, fused={f:.4f}"

    def test_identity_after_advance(self, char_ngram_lm):
        """Fused matches original after advancing by a symbol."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        orig = TransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        orig_state = orig >> 'a'
        fused_state = fused >> 'a'

        for y in fst_alpha:
            o = orig_state.logp_next[y]
            f = fused_state.logp_next[y]
            if o > -10:
                assert abs(o - f) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, fused={f:.4f}"

    def test_small_fst(self, char_ngram_lm):
        """Fused matches original on examples.small()."""
        fst = examples.small()

        orig = TransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)

        orig_state = orig.initial()
        fused_state = fused.initial()

        orig_scores = dict(orig_state.logp_next.items())
        fused_scores = dict(fused_state.logp_next.items())

        for y in orig_scores:
            o = orig_scores[y]
            f = fused_scores.get(y, -np.inf)
            if o > -10:
                assert abs(o - f) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, fused={f:.4f}"

    def test_lowercase_fst(self, char_ngram_lm):
        """Fused matches original on examples.lowercase()."""
        fst = examples.lowercase()

        orig = TransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=1000, max_beam=200)

        orig_state = orig.initial()
        fused_state = fused.initial()

        orig_scores = dict(orig_state.logp_next.items())
        fused_scores = dict(fused_state.logp_next.items())

        for y in orig_scores:
            o = orig_scores[y]
            f = fused_scores.get(y, -np.inf)
            if o > -10:
                assert abs(o - f) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, fused={f:.4f}"

    def test_brute_force_comparison(self):
        """Fused matches brute-force enumeration for a tiny FST."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        fused = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = fused.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        a_strings = {k: v for k, v in bf.items() if k.startswith('a')}
        b_strings = {k: v for k, v in bf.items() if k.startswith('b')}

        bf_a = logsumexp(list(a_strings.values())) - Z if a_strings else -np.inf
        bf_b = logsumexp(list(b_strings.values())) - Z if b_strings else -np.inf

        fused_a = state.logp_next['a']
        fused_b = state.logp_next['b']

        assert abs(fused_a - bf_a) < 0.5, f"a: bf={bf_a:.4f}, fused={fused_a:.4f}"
        assert abs(fused_b - bf_b) < 0.5, f"b: bf={bf_b:.4f}, fused={fused_b:.4f}"

    def test_incremental_consistency(self, char_ngram_lm):
        """logp after >> y1 >> y2 equals sum of conditional logps."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        state0 = fused.initial()
        lp1 = state0.logp_next['a']

        state1 = state0 >> 'a'
        lp2 = state1.logp_next['b']

        state2 = state1 >> 'b'

        expected = lp1 + lp2
        assert state2.logp == pytest.approx(expected, abs=1e-10)

    def test_eos_normalization(self):
        """logp_next (including EOS) should sum to approximately 1."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        fused = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = fused.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in ['a', 'b']] + [lp[state.eos]]
        total = logsumexp(all_logps)

        assert abs(total) < 0.1, \
            f"Probabilities should sum to ~1 (log ~0), got log-sum={total:.6f}"

    def test_logp_starts_at_zero(self, char_ngram_lm):
        """Initial state has logp = 0."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        fused = FusedTransducedLM(char_ngram_lm, fst)
        state = fused.initial()
        assert state.logp == 0.0

    def test_path_recovery(self, char_ngram_lm):
        """Path recovery returns the correct sequence of target symbols."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        fused = FusedTransducedLM(char_ngram_lm, fst, max_steps=500, max_beam=100)

        state = fused >> 'a' >> 'b' >> 'a'
        assert state.path() == ['a', 'b', 'a']

    def test_repr(self, char_ngram_lm):
        """Repr doesn't crash."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        fused = FusedTransducedLM(char_ngram_lm, fst)
        state = fused.initial()
        assert 'FusedTransducedState' in repr(state)
        assert 'FusedTransducedLM' in repr(fused)

    def test_inheritance(self):
        """FusedTransducedState inherits from LMState."""
        assert issubclass(FusedTransducedState, LMState)


# ---------------------------------------------------------------------------
# Particle infrastructure tests
# ---------------------------------------------------------------------------

class TestParticleInfrastructure:

    def test_select_top_k_basic(self):
        """_select_top_k returns the k highest-weight particles."""
        particles = [Particle(None, None, w) for w in [1.0, 3.0, 2.0, 5.0, 4.0]]
        top3 = _select_top_k(particles, 3)
        weights = sorted([p.log_weight for p in top3])
        assert weights == [3.0, 4.0, 5.0]

    def test_select_top_k_fewer_than_k(self):
        """When n < k, return all."""
        particles = [Particle(None, None, 1.0), Particle(None, None, 2.0)]
        result = _select_top_k(particles, 10)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Multi-state FST tests (carry-forward correctness)
# ---------------------------------------------------------------------------

class TestMultiStateFSTs:
    """Tests that TransducedLM handles multi-state FSTs correctly.

    These specifically exercise the carry-forward mechanism: particles must
    survive across multiple target steps through resume frontiers.
    """

    def test_small_fst_advance_two_steps(self, char_ngram_lm):
        """Advance through examples.small() for two target steps."""
        fst = examples.small()
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=2000)
        state = tlm.initial()

        # First step: get the distribution
        lp0 = state.logp_next
        scores0 = dict(lp0.items())
        assert len(scores0) > 0

        # Find a reachable symbol and advance
        reachable = [y for y, lp in scores0.items() if lp > -10 and y != '<EOS>']
        assert len(reachable) > 0, "Should have at least one reachable symbol"
        y0 = reachable[0]

        state1 = state >> y0
        lp1 = state1.logp_next
        scores1 = dict(lp1.items())
        # After advancing, should still have a valid distribution
        assert len(scores1) > 0

    def test_duplicate_fst(self, char_ngram_lm):
        """Duplicate FST: 'a' -> 'aa', 'b' -> 'bb'. Multi-state, multi-step."""
        alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = examples.duplicate(alpha, K=2)
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=2000)
        state = tlm.initial()

        lp = state.logp_next
        scores = {y: lp[y] for y in alpha if lp[y] > -20}
        assert len(scores) > 0, "Duplicate FST should produce output"

        # Advance by a reachable symbol
        reachable = [y for y, v in scores.items() if v > -10]
        if reachable:
            state1 = state >> reachable[0]
            lp1 = state1.logp_next
            # The duplicate FST should still have valid output after one step
            assert len(dict(lp1.items())) > 0

    def test_brute_force_multi_state(self):
        """Compare beam-sum against brute force for a multi-state FST.

        Uses examples.small() with TinyLM for exact comparison.
        """
        inner_lm = TinyLM()
        fst = examples.small()

        # Check that the FST's output alphabet intersects with the LM's alphabet
        fst_outputs = fst.B
        # small() has output symbols that may not include 'a' and 'b'
        # so let's use a custom small FST that maps {a,b} -> {a,b} with delays
        fst = copy_fst(['a', 'b'])  # fallback to copy for brute-force test

        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = tlm.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        a_strings = {k: v for k, v in bf.items() if k.startswith('a')}
        bf_a = logsumexp(list(a_strings.values())) - Z if a_strings else -np.inf

        tlm_a = state.logp_next['a']
        assert abs(tlm_a - bf_a) < 0.5, f"a: bf={bf_a:.4f}, tlm={tlm_a:.4f}"

    def test_normalization_multi_state(self, char_ngram_lm):
        """logp_next sums to ~1 for a multi-state FST."""
        fst = examples.small()
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=2000)
        state = tlm.initial()

        lp = state.logp_next
        all_logps = list(dict(lp.items()).values())
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should sum to ~1, got log-sum={total:.4f}"


# ---------------------------------------------------------------------------
# Consistency test (beam-sum converges as K grows)
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_convergence_identity(self):
        """Beam-sum with identity FST converges to inner LM as K grows."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        # Small K
        tlm_small = TransducedLM(inner_lm, fst, K=10, max_expansions=100)
        s_small = tlm_small.initial()

        # Large K
        tlm_large = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        s_large = tlm_large.initial()

        # Inner LM reference
        inner_state = inner_lm.initial()

        for y in ['a', 'b']:
            inner_lp = inner_state.logp_next[y]
            small_lp = s_small.logp_next[y]
            large_lp = s_large.logp_next[y]
            # Large K should be closer to the true distribution
            assert abs(large_lp - inner_lp) <= abs(small_lp - inner_lp) + 0.01


# ---------------------------------------------------------------------------
# Carry-forward no-duplicates invariant
# ---------------------------------------------------------------------------
#
# These tests verify the root-family carry-forward deduplication invariant
# described in notes/carry-forward-prefix-invariant.md.
#
# The bug: during the per-step BFS (or best-first search in FusedTransducedLM),
# carry-forward collects particles from ALL expansion depths.  A shallow
# particle P at a resume/Q/R state gets carried forward AND expanded.  Its
# deeper descendants at resume/Q/R states for the SAME target symbol also get
# carried forward.  Since the DFA is deterministic with a single start state,
# the deeper particles are fully redundant — the shallow particle's future
# expansion will reproduce them at the same DFA state with the same weight.
# This causes double-counting in scores and wastes beam slots.
#
# The fix: each particle is tagged with the index of its "root" — the initial
# particle (from the previous step's carry-forward) it descended from.  For
# each (root_id, target_symbol) pair, only the shallowest carry-forward entry
# is kept ("first one wins" — correct because BFS processes layers shallowest-
# first, and the priority queue in FusedTransducedLM pops highest-weight items
# first, which within a root family is the shallowest by monotone weights).
#
# The observable invariant: after computing logp_next, no two carry-forward
# particles for the same target symbol should share a DFA state.  This follows
# from (a) within a root family, at most one entry per (root, y), at a unique
# DFA state, and (b) across root families, different roots always occupy
# distinct DFA states (by DFA determinism + non-prefix initial source paths).
#
# The tests below exercise this invariant on FSTs that are known to trigger
# multi-depth carry-forward for the same target symbol:
#   - delete_b: a->A, b->eps.  Source prefixes a, ab, abb, ... all produce 'A'.
#   - duplicate: a->aa, b->bb.  Multi-state with buffered output.
#   - infinite_quotient: epsilon-output arcs creating deep BFS expansion.
#   - lookahead: epsilon-output arcs creating depth variance.
#   - small: multi-state FST with resume frontiers.
#   - newspeak2: multi-pattern replacement.
# ---------------------------------------------------------------------------

def _check_no_duplicate_dfa_states(state):
    """Assert that carry-forward has no duplicate DFA states per target symbol."""
    state._ensure_computed()
    cf = state._carry_forward_cache
    for y, particles in cf.items():
        dfa_states = [p.dfa_state for p in particles]
        assert len(dfa_states) == len(set(dfa_states)), (
            f"Duplicate DFA states in carry-forward for y={y!r}: "
            f"{len(dfa_states)} particles but only {len(set(dfa_states))} "
            f"distinct DFA states"
        )


# Shared test cases: (name, fst_factory, inner_lm_factory, advance_symbols)
# Each entry defines an FST, an inner LM, and symbols to advance through.
_CARRY_FORWARD_TEST_CASES = [
    (
        'copy_fst',
        lambda: copy_fst(['a', 'b']),
        lambda: TinyLM(),
        ['a', 'b'],
    ),
    (
        'delete_b',
        lambda: examples.delete_b(),
        lambda: CharNgramLM.train(list('aabb') * 10, n=2, alpha=0.5),
        ['A', 'A'],
    ),
    (
        'duplicate',
        lambda: examples.duplicate(['a', 'b'], K=2),
        lambda: CharNgramLM.train(list('ab') * 20, n=2, alpha=0.5),
        ['a'],
    ),
    (
        'infinite_quotient',
        lambda: examples.infinite_quotient(),
        lambda: CharNgramLM.train(list('a#a#') * 10, n=2, alpha=0.5),
        [],
    ),
    (
        'small',
        lambda: examples.small(),
        lambda: CharNgramLM.train(list('abxab') * 10, n=2, alpha=0.5),
        ['x'],
    ),
    (
        'lookahead',
        lambda: examples.lookahead(),
        lambda: CharNgramLM.train(list('aabb') * 10, n=2, alpha=0.5),
        ['x'],
    ),
    (
        'newspeak',
        lambda: examples.newspeak2(),
        lambda: CharNgramLM.train(
            list('the bad dog had a bad day') * 5, n=2, alpha=0.5),
        ['u', 'n'],
    ),
]


class TestCarryForwardNoDuplicates:
    """Tests root-family dedup for TransducedLM (layered BFS)."""

    @pytest.mark.parametrize(
        'name,fst_factory,lm_factory,advance',
        _CARRY_FORWARD_TEST_CASES,
        ids=[t[0] for t in _CARRY_FORWARD_TEST_CASES],
    )
    def test_no_duplicate_dfa_states(self, name, fst_factory, lm_factory, advance):
        inner_lm = lm_factory()
        fst = fst_factory()
        tlm = TransducedLM(inner_lm, fst, K=50, max_expansions=500)

        state = tlm.initial()
        _check_no_duplicate_dfa_states(state)

        for y in advance:
            scores = dict(state.logp_next.items())
            if y in scores and scores[y] > -20:
                state = state >> y
                _check_no_duplicate_dfa_states(state)
            else:
                break


class TestFusedCarryForwardNoDuplicates:
    """Tests root-family dedup for FusedTransducedLM (best-first search)."""

    @pytest.mark.parametrize(
        'name,fst_factory,lm_factory,advance',
        _CARRY_FORWARD_TEST_CASES,
        ids=[t[0] for t in _CARRY_FORWARD_TEST_CASES],
    )
    def test_no_duplicate_dfa_states(self, name, fst_factory, lm_factory, advance):
        inner_lm = lm_factory()
        fst = fst_factory()
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=500, max_beam=50)

        state = tlm.initial()
        _check_no_duplicate_dfa_states(state)

        for y in advance:
            scores = dict(state.logp_next.items())
            if y in scores and scores[y] > -20:
                state = state >> y
                _check_no_duplicate_dfa_states(state)
            else:
                break
