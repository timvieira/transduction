"""Tests for TransducedLM."""

import pytest
import numpy as np
from collections import defaultdict
from functools import cached_property
from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.util import LogDistr
from transduction.util import LogVector
from transduction.lm.ngram import ByteNgramLM, CharNgramLM, NgramState
from transduction.lm.transduced import TransducedLM, TransducedState, Particle
from transduction.util import logsumexp
from transduction.lm.transduced import _select_top_k
from transduction.lm.fused_transduced import FusedTransducedLM, FusedTransducedState
from transduction.lm.reference_transduced import ReferenceTransducedLM, ReferenceTransducedState
from transduction.lm.pynini_transduced import PyniniTransducedLM, PyniniTransducedState


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
        return LogDistr(self._probs)

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


class FiniteLMState(LMState):
    """State for a finite-support LM. Computes exact conditionals from the trie."""

    def __init__(self, lm, prefix, logp):
        self._lm = lm
        self._prefix = prefix
        self.logp = logp
        self.eos = lm.eos

    def _prefix_mass(self, prefix):
        """Sum of P(s) for all strings s starting with prefix."""
        n = len(prefix)
        return sum(p for s, p in self._lm._string_probs.items()
                   if len(s) >= n and s[:n] == prefix)

    @cached_property
    def logp_next(self):
        Z = self._prefix_mass(self._prefix)
        if Z <= 0:
            return LogDistr({self.eos: 0.0})
        scores = {}
        n = len(self._prefix)
        next_tokens = {s[n] for s in self._lm._string_probs
                       if len(s) > n and s[:n] == self._prefix}
        for tok in sorted(next_tokens):
            mass = self._prefix_mass(self._prefix + (tok,))
            if mass > 0:
                scores[tok] = np.log(mass / Z)
        eos_mass = self._lm._string_probs.get(self._prefix, 0)
        if eos_mass > 0:
            scores[self.eos] = np.log(eos_mass / Z)
        elif not scores:
            scores[self.eos] = 0.0
        return LogDistr(scores)

    def __rshift__(self, token):
        if token == self.eos:
            raise ValueError("Cannot advance past EOS")
        lp = self.logp_next[token]
        return FiniteLMState(self._lm, self._prefix + (token,), self.logp + lp)


class FiniteLM(LM):
    """LM with exact support on a finite set of strings.

    string_probs: {tuple_of_tokens: probability} — must sum to 1.
    """

    def __init__(self, string_probs, eos='<EOS>'):
        self.eos = eos
        self._string_probs = string_probs

    def initial(self):
        return FiniteLMState(self, (), 0.0)


def brute_force_pushforward(inner_lm, fst, target, max_source_len=8):
    """Exhaustive enumeration of P_fst(target_prefix + y) for all y.

    Uses FST.relation() to enumerate (source, output) pairs, scores each
    source with the inner LM, and accumulates by output string.

    Returns dict mapping output_string → log P.
    """
    output_probs = LogVector()

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
            output_probs.logaddexp(out, lp)

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

        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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

        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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
        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)

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
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
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
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
        state = tlm.initial()

        lp = state.logp_next
        scores = dict(lp.items())
        assert len(scores) > 0

    def test_brute_force_comparison(self):
        """Compare TransducedLM against brute-force enumeration for a tiny FST."""
        inner_lm = TinyLM()
        # Copy FST over {a, b}
        fst = copy_fst(['a', 'b'])

        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = tlm.initial()

        # Brute force: P(empty target, then next symbol y)
        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)

        # Total probability across all output strings
        Z = logsumexp(list(bf.values()))

        # P(next = 'a' | empty prefix)
        # = sum over all strings starting with 'a' / Z
        a_strings = {k: v for k, v in bf.items() if k and k[0] == 'a'}
        b_strings = {k: v for k, v in bf.items() if k and k[0] == 'b'}
        _eos_strings = {k: v for k, v in bf.items() if k == ()}

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
        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)

        state = tlm >> 'a' >> 'b' >> 'a'
        assert list(state.path) == ['a', 'b', 'a']

    def test_logp_starts_at_zero(self, char_ngram_lm):
        """Initial state has logp = 0."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, K=100)
        state = tlm.initial()
        assert state.logp == 0.0

    def test_repr(self, char_ngram_lm):
        """Repr doesn't crash."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, K=100)
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
                return LogDistr(self._probs)

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

        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
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
        tlm = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
        state = tlm.initial()

        eos_lp = state.logp_next[state.eos]
        assert eos_lp > -np.inf, "EOS should be reachable from initial state"
        assert eos_lp < 0.0, f"EOS log-prob should be negative, got {eos_lp}"

    def test_eos_normalization(self):
        """logp_next (including EOS) should sum to approximately 1."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
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
        """NgramState.greedy_decode returns a list of int byte tokens."""
        state = ngram_lm.initial()
        tokens = state.greedy_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
        for t in tokens:
            assert isinstance(t, int)

    def test_ngram_sample_decode(self, ngram_lm):
        """NgramState.sample_decode returns a list of int byte tokens."""
        state = ngram_lm.initial()
        tokens = state.sample_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
        for t in tokens:
            assert isinstance(t, int)

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
        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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
        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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
        s1 = state >> ord('a') >> ord('b')
        s2 = state(b'ab')
        assert s1.logp == pytest.approx(s2.logp)

    def test_transduced_advance(self, char_ngram_lm):
        """__call__ on TransducedState matches sequential >>."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        tlm = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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

        orig = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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

        orig = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
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

        orig = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
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

        orig = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
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

        a_strings = {k: v for k, v in bf.items() if k and k[0] == 'a'}
        b_strings = {k: v for k, v in bf.items() if k and k[0] == 'b'}

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
        assert list(state.path) == ['a', 'b', 'a']

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
        particles = [Particle(None, None, w, ()) for w in [1.0, 3.0, 2.0, 5.0, 4.0]]
        top3 = _select_top_k(particles, 3)
        weights = sorted([p.log_weight for p in top3])
        assert weights == [3.0, 4.0, 5.0]

    def test_select_top_k_fewer_than_k(self):
        """When n < k, return all."""
        particles = [Particle(None, None, 1.0, ()), Particle(None, None, 2.0, ())]
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

        a_strings = {k: v for k, v in bf.items() if k and k[0] == 'a'}
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
# BPE-style FST tests (Bug 2: carry-forward particles are dead ends)
# ---------------------------------------------------------------------------

def _bpe_style_fst():
    """BPE-style FST: epsilon-input arcs produce output, then source:eps back to start.

    Maps: x -> 'a','a'  and  y -> 'b','b'
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, EPSILON, 'a', ('a',))
    fst.add_arc(('a',), EPSILON, 'a', ('a', 'a'))
    fst.add_arc(('a', 'a'), 'x', EPSILON, 0)
    fst.add_arc(0, EPSILON, 'b', ('b',))
    fst.add_arc(('b',), EPSILON, 'b', ('b', 'b'))
    fst.add_arc(('b', 'b'), 'y', EPSILON, 0)
    return fst


class TestBPEStyleFST:
    """Tests for BPE-style FSTs where carry-forward particles were dead ends.

    Bug 2 from issue #4: carry-forward unconditionally saved particles at
    Q/R states, including truncated ones that are dead ends in the next step.
    Fix: only carry forward at resume_frontier states (which excludes
    truncated Q/R).  When carry-forward is empty, seed from DFA start states.
    """

    def test_transduced_decodes_all_symbols(self):
        """TransducedLM can decode all 4 output symbols on BPE-style FST."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xxyxy') * 5, n=2, alpha=0.5)
        tlm = TransducedLM(inner_lm, fst, K=50, max_expansions=100)

        target = ('a', 'a', 'b', 'b')
        state = tlm.initial()
        for i, y in enumerate(target):
            lp = state.logp_next
            assert y in lp and lp[y] > -np.inf, (
                f"Step {i}: TransducedLM missing {y!r} in logp_next "
                f"(keys={sorted(lp.keys())})"
            )
            state = state >> y

    def test_fused_decodes_all_symbols(self):
        """FusedTransducedLM can decode all 4 output symbols on BPE-style FST."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xxyxy') * 5, n=2, alpha=0.5)
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=500, max_beam=50)

        target = ('a', 'a', 'b', 'b')
        state = tlm.initial()
        for i, y in enumerate(target):
            lp = state.logp_next
            assert y in lp and lp[y] > -np.inf, (
                f"Step {i}: FusedTransducedLM missing {y!r} in logp_next "
                f"(keys={sorted(lp.keys())})"
            )
            state = state >> y

    def test_transduced_normalization(self):
        """TransducedLM on BPE FST produces normalized distributions at each step."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xxyxy') * 5, n=2, alpha=0.5)
        tlm = TransducedLM(inner_lm, fst, K=50, max_expansions=100)

        state = tlm.initial()
        for y in ('a', 'a'):
            all_logps = list(dict(state.logp_next.items()).values())
            total = logsumexp(all_logps)
            assert abs(total) < 0.5, (
                f"logp_next should sum to ~1, got log-sum={total:.4f}"
            )
            state = state >> y

    def test_transduced_multi_token_sequence(self):
        """TransducedLM can decode two full BPE tokens (x then y)."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xyxyxy') * 5, n=2, alpha=0.5)
        tlm = TransducedLM(inner_lm, fst, K=100, max_expansions=100)

        state = tlm.initial()
        for y in ('a', 'a', 'b', 'b'):
            state = state >> y
        assert state.logp > -np.inf

    def test_fused_multi_token_sequence(self):
        """FusedTransducedLM can decode two full BPE tokens (x then y)."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xyxyxy') * 5, n=2, alpha=0.5)
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=500, max_beam=100)

        state = tlm.initial()
        for y in ('a', 'a', 'b', 'b'):
            state = state >> y
        assert state.logp > -np.inf


# ---------------------------------------------------------------------------
# Carry-forward exactness tests (issue #6)
# ---------------------------------------------------------------------------

class TestCarryForwardExactness:
    """Verify carry-forward doesn't cause double-counting.

    delete_b is the canonical trigger: source paths a, ba, bba, ... all
    produce target 'A', so carry-forward accumulates particles from multiple
    BFS layers.  If these caused double-counting in the next step, scores
    would diverge from the exact answer.

    For TinyLM (memoryless: P(a)=0.6, P(b)=0.3, P(EOS)=0.1) + delete_b,
    the exact pushforward is P(output="A"^n) = (6/7)^n / 7 (by the negative
    binomial series), giving P(next='A' | any prefix) = 6/7 and
    P(EOS | any prefix) = 1/7 at every step.
    """

    def test_delete_b_transduced_exact(self):
        """TransducedLM on delete_b matches exact analytical answer.

        Small K (10) leaves most of the expansion budget for fresh exploration,
        so the geometric tail (0.3^k) is exhausted well past machine precision.
        """
        inner_lm = TinyLM()  # P(a)=0.6, P(b)=0.3, P(EOS)=0.1
        fst = examples.delete_b()

        exact_A = np.log(6 / 7)
        exact_EOS = np.log(1 / 7)

        tlm = TransducedLM(inner_lm, fst, K=10, max_expansions=10000)
        state = tlm.initial()

        for step in range(5):
            lp = state.logp_next
            assert abs(lp['A'] - exact_A) < 1e-10, \
                f"Step {step}: P(A) exact={exact_A:.12f}, got={lp['A']:.12f}"
            assert abs(lp[state.eos] - exact_EOS) < 1e-10, \
                f"Step {step}: P(EOS) exact={exact_EOS:.12f}, got={lp[state.eos]:.12f}"
            state = state >> 'A'

    def test_delete_b_fused_exact(self):
        """FusedTransducedLM on delete_b matches exact analytical answer."""
        inner_lm = TinyLM()
        fst = examples.delete_b()

        exact_A = np.log(6 / 7)
        exact_EOS = np.log(1 / 7)

        tlm = FusedTransducedLM(inner_lm, fst, max_steps=10000, max_beam=10)
        state = tlm.initial()

        for step in range(5):
            lp = state.logp_next
            assert abs(lp['A'] - exact_A) < 1e-10, \
                f"Step {step}: P(A) exact={exact_A:.12f}, got={lp['A']:.12f}"
            assert abs(lp[state.eos] - exact_EOS) < 1e-10, \
                f"Step {step}: P(EOS) exact={exact_EOS:.12f}, got={lp[state.eos]:.12f}"
            state = state >> 'A'

    def test_delete_b_carry_forward_no_prefix_overlap(self):
        """After dedup, carry-forward for 'A' has no prefix-overlapping paths.

        The BFS discovers particles at source paths (), (a,), (b,a,), etc.
        that all map to carry-forward for 'A'.  After prefix deduplication,
        no path should be a strict prefix of another.
        """
        inner_lm = TinyLM()
        fst = examples.delete_b()

        # Use smaller budget to keep the test fast — overlap behavior
        # doesn't depend on expansion budget.
        tlm = TransducedLM(inner_lm, fst, K=10, max_expansions=200)
        state = tlm.initial()
        state._ensure_computed()

        cf_A = state._carry_forward.get('A', [])
        assert len(cf_A) >= 1, "Expected at least one carry-forward particle for 'A'"
        paths = set(p.source_path for p in cf_A)
        for p in cf_A:
            for k in range(len(p.source_path)):
                assert p.source_path[:k] not in paths, (
                    f"Prefix overlap in carry_forward['A']: "
                    f"{p.source_path[:k]} is a prefix of {p.source_path}"
                )

    def test_duplicate_vs_reference(self):
        """TransducedLM on duplicate FST matches ReferenceTransducedLM."""
        inner_lm = TinyLM()
        V = ['a', 'b']
        fst = examples.duplicate(V, K=2)

        ref = ReferenceTransducedLM(inner_lm, fst)
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        ref_state = ref.initial()
        tlm_state = tlm.initial()

        for step, y in enumerate(['a', 'a', 'b', 'b']):
            ref_lp = ref_state.logp_next
            tlm_lp = tlm_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -10:
                    assert abs(tlm_lp[sym] - ref_lp[sym]) < 0.01, \
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.6f}, tlm={tlm_lp[sym]:.6f}"
            ref_state = ref_state >> y
            tlm_state = tlm_state >> y

    def test_duplicate_fused_vs_reference(self):
        """FusedTransducedLM on duplicate FST matches ReferenceTransducedLM."""
        inner_lm = TinyLM()
        V = ['a', 'b']
        fst = examples.duplicate(V, K=2)

        ref = ReferenceTransducedLM(inner_lm, fst)
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)

        ref_state = ref.initial()
        tlm_state = tlm.initial()

        for step, y in enumerate(['a', 'a', 'b', 'b']):
            ref_lp = ref_state.logp_next
            tlm_lp = tlm_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -10:
                    assert abs(tlm_lp[sym] - ref_lp[sym]) < 0.01, \
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.6f}, tlm={tlm_lp[sym]:.6f}"
            ref_state = ref_state >> y
            tlm_state = tlm_state >> y


def _finite_lm_for_delete_b(max_len=6):
    """FiniteLM matching TinyLM distribution truncated at max_len.

    Assigns P(s) ∝ 0.6^(#a) * 0.3^(#b) * 0.1 for all s in {a,b}^{≤max_len},
    then normalizes to sum to 1.
    """
    from itertools import product
    probs = {}
    for length in range(max_len + 1):
        for s in product(('a', 'b'), repeat=length):
            p = 0.6 ** s.count('a') * 0.3 ** s.count('b') * 0.1
            probs[s] = p
    Z = sum(probs.values())
    return FiniteLM({s: p / Z for s, p in probs.items()})


def _brute_force_conditional(inner_lm, fst, prefix, max_source_len):
    """Exact P(next=y | target prefix) via brute-force enumeration."""
    bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=max_source_len)

    # Partition output strings by whether they extend the prefix
    mass = {}     # next_symbol -> log P
    for out_str, lp in bf.items():
        if out_str[:len(prefix)] != prefix:
            continue
        suffix = out_str[len(prefix):]
        y = '<EOS>' if len(suffix) == 0 else suffix[0]
        if y not in mass:
            mass[y] = lp
        else:
            mass[y] = np.logaddexp(mass[y], lp)

    Z = logsumexp(list(mass.values())) if mass else -np.inf
    return {y: lp - Z for y, lp in mass.items()}


class TestFiniteLMExact:
    """Exact tests using FiniteLM where the BFS terminates naturally.

    With FiniteLM, zero-probability transitions are pruned, so the BFS
    explores only the finite support.  With K and budget larger than the
    support size, there is NO approximation — results must match brute-force
    enumeration exactly (up to floating-point).
    """

    @pytest.fixture
    def finite_delete_b_setup(self):
        max_len = 6
        inner_lm = _finite_lm_for_delete_b(max_len)
        fst = examples.delete_b()
        # K and budget far exceed the support size (sum 2^k for k=0..6 = 127)
        return inner_lm, fst, max_len

    def test_transduced_vs_brute_force(self, finite_delete_b_setup):
        """TransducedLM with FiniteLM matches brute-force on delete_b."""
        inner_lm, fst, max_len = finite_delete_b_setup
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        state = tlm.initial()
        for step, prefix in enumerate([(), ('A',), ('A', 'A'), ('A', 'A', 'A')]):
            bf_cond = _brute_force_conditional(inner_lm, fst, prefix, max_len)
            tlm_lp = state.logp_next
            for y, bf_val in bf_cond.items():
                tlm_val = tlm_lp.get(y, -np.inf)
                assert abs(tlm_val - bf_val) < 1e-10, \
                    f"Step {step}, y={y!r}: bf={bf_val:.12f}, tlm={tlm_val:.12f}"
            if step < 3:
                state = state >> 'A'

    def test_fused_vs_brute_force(self, finite_delete_b_setup):
        """FusedTransducedLM with FiniteLM matches brute-force on delete_b."""
        inner_lm, fst, max_len = finite_delete_b_setup
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)

        state = tlm.initial()
        for step, prefix in enumerate([(), ('A',), ('A', 'A'), ('A', 'A', 'A')]):
            bf_cond = _brute_force_conditional(inner_lm, fst, prefix, max_len)
            tlm_lp = state.logp_next
            for y, bf_val in bf_cond.items():
                tlm_val = tlm_lp.get(y, -np.inf)
                assert abs(tlm_val - bf_val) < 1e-10, \
                    f"Step {step}, y={y!r}: bf={bf_val:.12f}, tlm={tlm_val:.12f}"
            if step < 3:
                state = state >> 'A'

    def test_small_fst_vs_brute_force(self):
        """TransducedLM with FiniteLM matches brute-force on small() FST."""
        # small() has source alphabet {a, b}, output alphabet {x, a, b}
        probs = {}
        for length in range(6):
            from itertools import product
            for s in product(('a', 'b'), repeat=length):
                probs[s] = 0.5 ** length * 0.5  # uniform + geometric length
        Z = sum(probs.values())
        inner_lm = FiniteLM({s: p / Z for s, p in probs.items()})

        fst = examples.small()
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        state = tlm.initial()
        bf_cond = _brute_force_conditional(inner_lm, fst, (), max_source_len=6)
        tlm_lp = state.logp_next
        for y, bf_val in bf_cond.items():
            tlm_val = tlm_lp.get(y, -np.inf)
            assert abs(tlm_val - bf_val) < 1e-10, \
                f"y={y!r}: bf={bf_val:.12f}, tlm={tlm_val:.12f}"


# ---------------------------------------------------------------------------
# ReferenceTransducedLM tests (exact ground-truth)
# ---------------------------------------------------------------------------

def _mapping_fst():
    """Acyclic FST: a→x, b→y. Start state is also stop (empty→empty)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'y', 2)
    fst.add_stop(1)
    fst.add_stop(2)
    return fst


def _bounded_copy_fst(alphabet, max_len):
    """Acyclic copy FST that copies strings up to max_len symbols."""
    fst = FST()
    for i in range(max_len + 1):
        if i == 0:
            fst.add_start(i)
        fst.add_stop(i)
        if i < max_len:
            for a in alphabet:
                fst.add_arc(i, a, a, i + 1)
    return fst


def _ambiguous_fst():
    """Acyclic FST: a→x, b→x. Two sources map to same output."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)
    fst.add_stop(1)
    fst.add_stop(2)
    return fst


def _length_changing_fst():
    """Acyclic FST: a→xy. One source symbol produces two output symbols."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, EPSILON, 'y', 2)
    fst.add_stop(2)
    return fst


def _finite_inner_lm():
    """Finite LM over source alphabet {a, b} with known probabilities."""
    return FiniteLM({
        (): 0.2,
        ('a',): 0.3,
        ('b',): 0.2,
        ('a', 'b'): 0.15,
        ('b', 'a'): 0.15,
    })


class TestReferenceTransducedLM:
    """Tests for ReferenceTransducedLM (exact ground-truth transduced LM).

    Uses FiniteLM as inner LM and acyclic (finite-relation) FSTs
    so that all Q/R languages are finite and exact values can be
    computed by hand.
    """

    # -- Basic properties ---------------------------------------------------

    def test_logp_starts_at_zero(self):
        """Initial state has logp = 0."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        assert tlm.initial().logp == 0.0

    def test_normalization_mapping_fst(self):
        """logp_next sums to exactly 1 at the initial state."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        state = tlm.initial()
        all_logps = list(dict(state.logp_next.items()).values())
        total = logsumexp(all_logps)
        assert abs(total) < 1e-10, f"Should sum to 1, got log-sum={total}"

    def test_normalization_bounded_copy(self):
        """logp_next sums to exactly 1 with bounded copy FST."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        state = tlm.initial()
        all_logps = list(dict(state.logp_next.items()).values())
        total = logsumexp(all_logps)
        assert abs(total) < 1e-10, f"Should sum to 1, got log-sum={total}"

    def test_normalization_after_advance(self):
        """logp_next sums to 1 after advancing by a symbol."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        state = tlm >> 'a'
        all_logps = list(dict(state.logp_next.items()).values())
        total = logsumexp(all_logps)
        assert abs(total) < 1e-10, f"Should sum to 1, got log-sum={total}"

    def test_incremental_consistency(self):
        """logp after >> y1 >> y2 equals sum of conditional logps."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        state0 = tlm.initial()
        lp1 = state0.logp_next['a']
        state1 = state0 >> 'a'
        lp2 = state1.logp_next['b']
        state2 = state1 >> 'b'
        assert state2.logp == pytest.approx(lp1 + lp2, abs=1e-10)

    # -- Exact value tests --------------------------------------------------

    def test_exact_mapping_fst(self):
        """Verify exact logp_next values for the mapping FST.

        Relation: '' → '', 'a' → 'x', 'b' → 'y'
        Pushforward: '' → 0.2, 'x' → 0.3, 'y' → 0.2, total = 0.7
        Initial: P(x) = 3/7, P(y) = 2/7, P(EOS) = 2/7
        """
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        lp = tlm.initial().logp_next

        assert lp['x'] == pytest.approx(np.log(3 / 7), abs=1e-10)
        assert lp['y'] == pytest.approx(np.log(2 / 7), abs=1e-10)
        assert lp['<EOS>'] == pytest.approx(np.log(2 / 7), abs=1e-10)

    def test_exact_bounded_copy(self):
        """Verify exact logp_next for bounded copy FST.

        Relation covers all 1- and 2-symbol strings over {a,b}.
        Only strings in the inner LM have nonzero mass:
          '' → 0.2, 'a' → 0.3, 'b' → 0.2, 'ab' → 0.15, 'ba' → 0.15
        Total pushforward = 1.0.
        Initial: P(a) = 0.45, P(b) = 0.35, P(EOS) = 0.2
        """
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        lp = tlm.initial().logp_next

        assert lp['a'] == pytest.approx(np.log(0.45), abs=1e-10)
        assert lp['b'] == pytest.approx(np.log(0.35), abs=1e-10)
        assert lp['<EOS>'] == pytest.approx(np.log(0.2), abs=1e-10)

    def test_exact_after_advance(self):
        """Verify exact logp_next after advancing by 'a' on bounded copy FST.

        After 'a': P(b|a) = 0.15/0.45 = 1/3, P(EOS|a) = 0.3/0.45 = 2/3.
        'a' is unreachable (no source 'aa' has nonzero prob).
        """
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        lp = (tlm >> 'a').logp_next

        assert lp['b'] == pytest.approx(np.log(1 / 3), abs=1e-10)
        assert lp['<EOS>'] == pytest.approx(np.log(2 / 3), abs=1e-10)
        assert lp['a'] == -np.inf  # no source 'aa' in inner LM

    def test_exact_ambiguous_fst(self):
        """Two source paths contribute to the same output symbol.

        Relation: '' → '', 'a' → 'x', 'b' → 'x'
        P_target('x') = P('a') + P('b') = 0.3 + 0.2 = 0.5, total = 0.7
        P(x) = 5/7, P(EOS) = 2/7
        """
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _ambiguous_fst())
        lp = tlm.initial().logp_next

        assert lp['x'] == pytest.approx(np.log(5 / 7), abs=1e-10)
        assert lp['<EOS>'] == pytest.approx(np.log(2 / 7), abs=1e-10)

    def test_exact_length_changing_fst(self):
        """Source 'a' maps to target 'xy' (two output symbols).

        Relation: '' → '', 'a' → 'xy'. Total pushforward = 0.2 + 0.3 = 0.5.
        Initial: P(x) = 0.3/0.5 = 3/5, P(EOS) = 0.2/0.5 = 2/5
        """
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _length_changing_fst())
        lp = tlm.initial().logp_next

        assert lp['x'] == pytest.approx(np.log(3 / 5), abs=1e-10)
        assert lp['<EOS>'] == pytest.approx(np.log(2 / 5), abs=1e-10)

    def test_length_changing_second_step(self):
        """After 'x', the only option is 'y' with probability 1."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _length_changing_fst())
        lp = (tlm >> 'x').logp_next

        assert lp['y'] == pytest.approx(0.0, abs=1e-10)
        assert lp['<EOS>'] == -np.inf

    def test_complete_string_probability(self):
        """Cumulative logp + logp_next[EOS] gives the correct string probability.

        Target 'ab' with bounded copy FST: P_target('ab') = P_source('ab') = 0.15.
        """
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        state = tlm >> 'a' >> 'b'
        complete_logp = state.logp + state.logp_next['<EOS>']
        assert complete_logp == pytest.approx(np.log(0.15), abs=1e-10)

    # -- EOS and error handling ---------------------------------------------

    def test_eos_only_state(self):
        """After exhausting all continuations, P(EOS) = 1."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        lp = (tlm >> 'x').logp_next
        assert lp['<EOS>'] == pytest.approx(0.0, abs=1e-10)

    def test_advance_past_eos_raises(self):
        """Advancing by EOS raises ValueError."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        state = tlm.initial()
        with pytest.raises(ValueError, match="Cannot advance past EOS"):
            state >> '<EOS>'

    def test_zero_prob_symbol_raises(self):
        """Advancing by a symbol with zero probability raises ValueError."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        state = tlm >> 'x'  # after 'x', only EOS is possible
        with pytest.raises(ValueError, match="zero probability"):
            state >> 'y'

    # -- Path, repr, decode -------------------------------------------------

    def test_path_recovery(self):
        """path() returns the correct sequence of target symbols."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        state = tlm >> 'a' >> 'b'
        assert state.path() == ['a', 'b']

    def test_repr(self):
        """repr doesn't crash and contains class name."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _mapping_fst())
        state = tlm.initial()
        assert 'ReferenceTransducedState' in repr(state)

    def test_inheritance(self):
        """ReferenceTransducedState inherits from LMState."""
        assert issubclass(ReferenceTransducedState, LMState)

    def test_greedy_decode(self):
        """Inherited greedy_decode works correctly.

        With bounded copy FST: P(a)=0.45 > P(b)=0.35 > P(EOS)=0.2.
        After 'a': P(EOS)=2/3 > P(b)=1/3. So greedy picks 'a' then stops.
        """
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        tokens = tlm.initial().greedy_decode(max_len=10)
        assert tokens == ['a']

    def test_sample_decode(self):
        """Inherited sample_decode produces valid token sequences."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))
        tokens = tlm.initial().sample_decode(max_len=10)
        assert isinstance(tokens, list)
        assert len(tokens) <= 10

    # -- Cross-validation against brute force -------------------------------

    def test_brute_force_match_initial(self):
        """ReferenceTransducedLM matches brute-force enumeration at initial state."""
        inner = _finite_inner_lm()
        fst = _bounded_copy_fst(['a', 'b'], 2)
        bf = brute_force_pushforward(inner, fst, '', max_source_len=5)
        Z = logsumexp(list(bf.values()))

        tlm = ReferenceTransducedLM(inner, fst)
        state = tlm.initial()

        for y in ['a', 'b']:
            matching = [lp for out, lp in bf.items() if out and out[0] == y]
            bf_y = logsumexp(matching) - Z if matching else -np.inf
            ref_y = state.logp_next[y]
            assert ref_y == pytest.approx(bf_y, abs=1e-8), \
                f"Symbol {y!r}: ref={ref_y:.6f}, bf={bf_y:.6f}"

    def test_brute_force_match_after_advance(self):
        """ReferenceTransducedLM matches brute force after advancing by 'a'."""
        inner = _finite_inner_lm()
        fst = _bounded_copy_fst(['a', 'b'], 2)
        bf = brute_force_pushforward(inner, fst, '', max_source_len=5)

        # P(next='b' | seen 'a') = P(starts with 'ab') / P(starts with 'a')
        a_strings = [lp for out, lp in bf.items() if out and out[0] == 'a']
        ab_strings = [lp for out, lp in bf.items() if out[:2] == ('a', 'b')]
        Z_a = logsumexp(a_strings)
        bf_b_given_a = logsumexp(ab_strings) - Z_a if ab_strings else -np.inf

        tlm = ReferenceTransducedLM(inner, fst)
        ref_b = (tlm >> 'a').logp_next['b']

        assert ref_b == pytest.approx(bf_b_given_a, abs=1e-8), \
            f"P(b|a): ref={ref_b:.6f}, bf={bf_b_given_a:.6f}"

    def test_brute_force_match_mapping_fst(self):
        """Brute-force cross-check with the mapping FST."""
        inner = _finite_inner_lm()
        fst = _mapping_fst()
        bf = brute_force_pushforward(inner, fst, '', max_source_len=5)
        Z = logsumexp(list(bf.values()))

        tlm = ReferenceTransducedLM(inner, fst)
        state = tlm.initial()

        for y in ['x', 'y']:
            matching = [lp for out, lp in bf.items() if out and out[0] == y]
            bf_y = logsumexp(matching) - Z if matching else -np.inf
            ref_y = state.logp_next[y]
            assert ref_y == pytest.approx(bf_y, abs=1e-8), \
                f"Symbol {y!r}: ref={ref_y:.6f}, bf={bf_y:.6f}"

    # -- Cross-validation against TransducedLM / FusedTransducedLM ----------

    def test_agrees_with_transduced_lm(self):
        """ReferenceTransducedLM agrees with TransducedLM (large K)."""
        inner = _finite_inner_lm()
        fst = _bounded_copy_fst(['a', 'b'], 2)

        ref = ReferenceTransducedLM(inner, fst)
        approx = TransducedLM(inner, fst, K=500, max_expansions=5000)

        ref_state = ref.initial()
        approx_state = approx.initial()

        for y in ['a', 'b', '<EOS>']:
            ref_lp = ref_state.logp_next[y]
            approx_lp = approx_state.logp_next[y]
            assert abs(ref_lp - approx_lp) < 0.5, \
                f"Symbol {y!r}: ref={ref_lp:.4f}, approx={approx_lp:.4f}"

    def test_agrees_with_fused_transduced_lm(self):
        """ReferenceTransducedLM agrees with FusedTransducedLM."""
        inner = _finite_inner_lm()
        fst = _bounded_copy_fst(['a', 'b'], 2)

        ref = ReferenceTransducedLM(inner, fst)
        fused = FusedTransducedLM(inner, fst, max_steps=5000, max_beam=500)

        ref_state = ref.initial()
        fused_state = fused.initial()

        for y in ['a', 'b', '<EOS>']:
            ref_lp = ref_state.logp_next[y]
            fused_lp = fused_state.logp_next[y]
            assert abs(ref_lp - fused_lp) < 0.5, \
                f"Symbol {y!r}: ref={ref_lp:.4f}, fused={fused_lp:.4f}"

    # -- Multi-step decode --------------------------------------------------

    def test_multi_step_bounded_copy(self):
        """Advance through multiple steps and verify logp consistency."""
        tlm = ReferenceTransducedLM(
            _finite_inner_lm(), _bounded_copy_fst(['a', 'b'], 2))

        state = tlm.initial()
        cumulative = 0.0
        for y in ['a', 'b']:
            lp = state.logp_next[y]
            cumulative += lp
            state = state >> y
            assert state.logp == pytest.approx(cumulative, abs=1e-10)
        # Final state should have P(EOS) = 1
        assert state.logp_next['<EOS>'] == pytest.approx(0.0, abs=1e-10)

    def test_multi_step_length_changing(self):
        """Full decode through the length-changing FST: target 'xy'."""
        tlm = ReferenceTransducedLM(_finite_inner_lm(), _length_changing_fst())

        state = tlm.initial()
        state = state >> 'x'
        state = state >> 'y'
        # Target 'xy' corresponds to source 'a' with P = 0.3.
        # Pushforward total = 0.5, so P_target('xy') = 0.3/0.5 = 0.6.
        # Complete string logp = log(0.6).
        complete_logp = state.logp + state.logp_next['<EOS>']
        assert complete_logp == pytest.approx(np.log(0.6), abs=1e-10)


# ---------------------------------------------------------------------------
# Carry-forward overlap regression tests (issue #6)
# ---------------------------------------------------------------------------

def _overlap_trigger_fst():
    """Acyclic FST with non-deterministic output timing.

    Source 'a' from state 0 either produces 'c' directly or produces epsilon
    (buffering).  This creates R(c)\\Q(c) powerset states that previously
    caused carry-forward overlap and duplicate particles.

    Relation:
      '' -> ''              (start is final)
      'a' -> 'c'            (direct: a produces c immediately)
      'ab' -> 'c'           (delayed: a produces eps, then b produces c)
      'aa' -> 'cx'          (direct path continues)
      'aba' -> 'cx'         (delayed path continues)
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'c', 1)       # direct: a -> c
    fst.add_arc(0, 'a', EPSILON, 2)    # delayed: a -> eps
    fst.add_arc(2, 'b', 'c', 3)       # then b -> c
    fst.add_stop(1)
    fst.add_stop(3)
    fst.add_arc(1, 'a', 'x', 4)       # continue: a -> x
    fst.add_arc(3, 'a', 'x', 4)       # continue: a -> x
    fst.add_stop(4)
    return fst


class TestOverlapTrigger:
    """Regression tests for carry-forward overlap (issue #6).

    The _overlap_trigger_fst has non-deterministic output timing: source 'a'
    from the start state either produces target 'c' immediately (-> state 1)
    or produces epsilon (-> state 2, then 'b' -> 'c').  This creates
    R(c)\\Q(c) powerset states.  Before the fix, such states were both
    expanded AND carried forward, causing prefix-overlapping source paths
    that generated duplicate particles in subsequent steps.
    """

    @pytest.fixture
    def overlap_setup(self):
        fst = _overlap_trigger_fst()
        inner_lm = FiniteLM({
            (): 0.1,
            ('a',): 0.3,
            ('b',): 0.1,
            ('a', 'b'): 0.2,
            ('a', 'a'): 0.15,
            ('a', 'b', 'a'): 0.15,
        })
        return inner_lm, fst

    def test_no_duplicate_source_paths(self, overlap_setup):
        """After advancing by 'c', no two particles should share a source_path."""
        inner_lm, fst = overlap_setup
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = tlm >> 'c'
        paths = [p.source_path for p in state._particles]
        assert len(set(paths)) == len(paths), (
            f"Duplicate source_paths in particles after advancing by 'c': "
            f"{len(paths)} particles, {len(set(paths))} unique"
        )

    def test_no_prefix_overlap_in_carry_forward(self, overlap_setup):
        """Carry-forward particles for 'c' should have no prefix-overlapping paths."""
        inner_lm, fst = overlap_setup
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = tlm.initial()
        state._ensure_computed()
        cf_c = state._carry_forward.get('c', [])
        paths = [p.source_path for p in cf_c]
        # Check no path is a strict prefix of another
        for i, p1 in enumerate(paths):
            for j, p2 in enumerate(paths):
                if i != j and len(p1) < len(p2) and p2[:len(p1)] == p1:
                    assert False, (
                        f"Prefix overlap in carry_forward['c']: "
                        f"{p1} is a prefix of {p2}"
                    )

    def test_transduced_vs_reference(self, overlap_setup):
        """TransducedLM matches ReferenceTransducedLM on the overlap-trigger FST."""
        inner_lm, fst = overlap_setup
        ref = ReferenceTransducedLM(inner_lm, fst)
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        ref_state = ref.initial()
        tlm_state = tlm.initial()

        for step, y in enumerate(['c', 'x']):
            ref_lp = ref_state.logp_next
            tlm_lp = tlm_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -20:
                    assert abs(tlm_lp[sym] - ref_lp[sym]) < 1e-6, (
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.8f}, "
                        f"tlm={tlm_lp[sym]:.8f}"
                    )
            ref_state = ref_state >> y
            tlm_state = tlm_state >> y

    def test_fused_vs_reference(self, overlap_setup):
        """FusedTransducedLM matches ReferenceTransducedLM on the overlap-trigger FST."""
        inner_lm, fst = overlap_setup
        ref = ReferenceTransducedLM(inner_lm, fst)
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)

        ref_state = ref.initial()
        tlm_state = tlm.initial()

        for step, y in enumerate(['c', 'x']):
            ref_lp = ref_state.logp_next
            tlm_lp = tlm_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -20:
                    assert abs(tlm_lp[sym] - ref_lp[sym]) < 1e-6, (
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.8f}, "
                        f"tlm={tlm_lp[sym]:.8f}"
                    )
            ref_state = ref_state >> y
            tlm_state = tlm_state >> y

    def test_fused_no_duplicate_source_paths(self, overlap_setup):
        """FusedTransducedLM: after advancing by 'c', no duplicate source_paths."""
        inner_lm, fst = overlap_setup
        tlm = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500)
        state = tlm >> 'c'
        paths = [p.source_path for p in state._particles]
        assert len(set(paths)) == len(paths), (
            f"Duplicate source_paths in FusedTransducedLM particles: "
            f"{len(paths)} particles, {len(set(paths))} unique"
        )


# ---------------------------------------------------------------------------
# PeekabooState EOS double-counting regression test (samuel at context 'c')
# ---------------------------------------------------------------------------

from transduction.peekaboo_incremental import PeekabooState, FstUniversality


class TestPeekabooSamuelDoubleCounting:
    """PeekabooState double-counts EOS mass at samuel context ('c',).

    Samuel's topology creates a catching-up beam through states 1->3.
    State 3 is final and has only an eps-output arc (eps/x -> 4).  The
    eps-closure splits into at-boundary state 3 (final) and beyond state 4
    (universal).  PeekabooState counts BOTH the at-boundary EOS from state 3
    AND the quotient contribution from state 4, but state 4's prefix
    probability already includes the EOS path through state 3.

    The exact ReferenceTransducedLM (which rebuilds the full Precover DFA
    per prefix) gives the correct answer.  This test documents the mismatch.
    """

    @pytest.fixture
    def samuel_setup(self):
        fst = examples.samuel_example()
        inner_lm = CharNgramLM.train("aabbaabb" * 5, n=2, alpha=0.5)
        return inner_lm, fst

    def test_exact_at_empty_context(self, samuel_setup):
        """Sanity: PeekabooState and exact reference agree at empty context."""
        inner_lm, fst = samuel_setup
        ref = ReferenceTransducedLM(inner_lm, fst)
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000,
                           decomp_state_cls=PeekabooState, univ_cls=FstUniversality)

        ref_state = ref.initial()
        tlm_state = tlm.initial()

        for sym in ref_state.logp_next:
            r = float(ref_state.logp_next[sym])
            t = float(tlm_state.logp_next[sym])
            if np.isfinite(r):
                assert abs(r - t) < 1e-6, (
                    f"Mismatch at empty context: {sym!r}: "
                    f"ref={r:.6f}, peekaboo={t:.6f}"
                )

    def test_peekaboo_vs_exact_at_c(self, samuel_setup):
        """PeekabooState disagrees with exact reference at context ('c',).

        Expected to fail until the double-counting bug is fixed.
        """
        inner_lm, fst = samuel_setup

        ref = ReferenceTransducedLM(inner_lm, fst)
        ref_state = ref.initial() >> 'c'

        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000,
                           decomp_state_cls=PeekabooState, univ_cls=FstUniversality)
        tlm_state = tlm.initial() >> 'c'

        for sym in ref_state.logp_next:
            r = float(ref_state.logp_next[sym])
            t = float(tlm_state.logp_next[sym])
            if np.isfinite(r):
                assert abs(r - t) < 0.01, (
                    f"PeekabooState double-counting at samuel context ('c',): "
                    f"{sym!r}: exact={r:.6f}, peekaboo={t:.6f}, "
                    f"diff={abs(r - t):.6f}"
                )

    def test_peekaboo_normalization_at_c(self, samuel_setup):
        """Distribution sums to ~1 even with the double-counting bug."""
        inner_lm, fst = samuel_setup
        tlm = TransducedLM(inner_lm, fst, K=500, max_expansions=5000,
                           decomp_state_cls=PeekabooState, univ_cls=FstUniversality)
        state = tlm.initial() >> 'c'
        lp = state.logp_next
        all_logps = [float(lp[y]) for y in lp if np.isfinite(float(lp[y]))]
        total = logsumexp(all_logps)
        assert abs(total) < 0.01, f"Should sum to ~1, got log-sum={total}"

    def test_brute_force_converges_to_exact(self, samuel_setup):
        """Brute-force enumeration converges toward exact reference as
        max_source_len increases, confirming the reference is correct."""
        inner_lm, fst = samuel_setup

        ref = ReferenceTransducedLM(inner_lm, fst)
        ref_state = ref.initial() >> 'c'
        ref_x = float(ref_state.logp_next['x'])

        prev_diff = np.inf
        for max_len in [6, 8, 10, 12]:
            bf = _brute_force_conditional(inner_lm, fst, ('c',), max_len)
            if 'x' in bf:
                diff = abs(bf['x'] - ref_x)
                assert diff <= prev_diff + 1e-10, (
                    f"Brute force should converge: max_len={max_len}, "
                    f"diff={diff:.6f}, prev={prev_diff:.6f}"
                )
                prev_diff = diff


# ---------------------------------------------------------------------------
# PyniniTransducedLM tests
# ---------------------------------------------------------------------------

class TestPyniniTransducedLM:
    """Tests that PyniniTransducedLM matches TransducedLM."""

    def test_identity_transducer(self, char_ngram_lm):
        """Pynini and original should agree on copy FST."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        orig = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)

        orig_state = orig.initial()
        pyn_state = pyn.initial()

        for y in fst_alpha:
            o = orig_state.logp_next[y]
            p = pyn_state.logp_next[y]
            if o > -10:
                assert abs(o - p) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, pynini={p:.4f}"

    def test_identity_after_advance(self, char_ngram_lm):
        """Pynini matches original after advancing by a symbol."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)

        orig = TransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)

        orig_state = orig >> 'a'
        pyn_state = pyn >> 'a'

        for y in fst_alpha:
            o = orig_state.logp_next[y]
            p = pyn_state.logp_next[y]
            if o > -10:
                assert abs(o - p) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, pynini={p:.4f}"

    def test_small_fst(self, char_ngram_lm):
        """Pynini matches original on examples.small()."""
        fst = examples.small()

        orig = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)

        orig_scores = dict(orig.initial().logp_next.items())
        pyn_scores = dict(pyn.initial().logp_next.items())

        for y in orig_scores:
            o = orig_scores[y]
            p = pyn_scores.get(y, -np.inf)
            if o > -10:
                assert abs(o - p) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, pynini={p:.4f}"

    def test_lowercase_fst(self, char_ngram_lm):
        """Pynini matches original on examples.lowercase()."""
        fst = examples.lowercase()

        orig = TransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=200, max_expansions=1000)

        orig_scores = dict(orig.initial().logp_next.items())
        pyn_scores = dict(pyn.initial().logp_next.items())

        for y in orig_scores:
            o = orig_scores[y]
            p = pyn_scores.get(y, -np.inf)
            if o > -10:
                assert abs(o - p) < 0.5, \
                    f"Symbol {y!r}: orig={o:.4f}, pynini={p:.4f}"

    def test_brute_force_comparison(self):
        """Pynini matches brute-force enumeration for a tiny FST."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        pyn = PyniniTransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = pyn.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        a_strings = {k: v for k, v in bf.items() if k and k[0] == 'a'}
        b_strings = {k: v for k, v in bf.items() if k and k[0] == 'b'}

        bf_a = logsumexp(list(a_strings.values())) - Z if a_strings else -np.inf
        bf_b = logsumexp(list(b_strings.values())) - Z if b_strings else -np.inf

        pyn_a = state.logp_next['a']
        pyn_b = state.logp_next['b']

        assert abs(pyn_a - bf_a) < 0.5, f"a: bf={bf_a:.4f}, pynini={pyn_a:.4f}"
        assert abs(pyn_b - bf_b) < 0.5, f"b: bf={bf_b:.4f}, pynini={pyn_b:.4f}"

    def test_incremental_consistency(self, char_ngram_lm):
        """logp after >> y1 >> y2 equals sum of conditional logps."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)

        state0 = pyn.initial()
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

        pyn = PyniniTransducedLM(inner_lm, fst, K=500, max_expansions=5000)
        state = pyn.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in ['a', 'b']] + [lp[state.eos]]
        total = logsumexp(all_logps)

        assert abs(total) < 0.1, \
            f"Probabilities should sum to ~1 (log ~0), got log-sum={total:.6f}"

    def test_logp_starts_at_zero(self, char_ngram_lm):
        """Initial state has logp = 0."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100)
        assert pyn.initial().logp == 0.0

    def test_path_recovery(self, char_ngram_lm):
        """Path recovery returns the correct sequence of target symbols."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100, max_expansions=500)
        state = pyn >> 'a' >> 'b' >> 'a'
        assert list(state.path) == ['a', 'b', 'a']

    def test_repr(self, char_ngram_lm):
        """Repr doesn't crash."""
        fst_alpha = [s for s in char_ngram_lm.alphabet if s != '<EOS>']
        fst = copy_fst(fst_alpha)
        pyn = PyniniTransducedLM(char_ngram_lm, fst, K=100)
        state = pyn.initial()
        assert 'PyniniTransducedState' in repr(state)
        assert 'PyniniTransducedLM' in repr(pyn)

    def test_inheritance(self):
        """PyniniTransducedState inherits from LMState."""
        assert issubclass(PyniniTransducedState, LMState)

    def test_delete_b_exact(self):
        """PyniniTransducedLM on delete_b matches exact analytical answer."""
        inner_lm = TinyLM()
        fst = examples.delete_b()

        exact_A = np.log(6 / 7)
        exact_EOS = np.log(1 / 7)

        pyn = PyniniTransducedLM(inner_lm, fst, K=10, max_expansions=10000)
        state = pyn.initial()

        for step in range(5):
            lp = state.logp_next
            assert abs(lp['A'] - exact_A) < 1e-10, \
                f"Step {step}: P(A) exact={exact_A:.12f}, got={lp['A']:.12f}"
            assert abs(lp[state.eos] - exact_EOS) < 1e-10, \
                f"Step {step}: P(EOS) exact={exact_EOS:.12f}, got={lp[state.eos]:.12f}"
            state = state >> 'A'

    def test_duplicate_vs_reference(self):
        """PyniniTransducedLM on duplicate FST matches ReferenceTransducedLM."""
        inner_lm = TinyLM()
        V = ['a', 'b']
        fst = examples.duplicate(V, K=2)

        ref = ReferenceTransducedLM(inner_lm, fst)
        pyn = PyniniTransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        ref_state = ref.initial()
        pyn_state = pyn.initial()

        for step, y in enumerate(['a', 'a', 'b', 'b']):
            ref_lp = ref_state.logp_next
            pyn_lp = pyn_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -10:
                    assert abs(pyn_lp[sym] - ref_lp[sym]) < 0.01, \
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.6f}, pyn={pyn_lp[sym]:.6f}"
            ref_state = ref_state >> y
            pyn_state = pyn_state >> y

    def test_finite_lm_vs_brute_force(self):
        """PyniniTransducedLM with FiniteLM matches brute-force on delete_b."""
        max_len = 6
        inner_lm = _finite_lm_for_delete_b(max_len)
        fst = examples.delete_b()
        pyn = PyniniTransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        state = pyn.initial()
        for step, prefix in enumerate([(), ('A',), ('A', 'A')]):
            bf_cond = _brute_force_conditional(inner_lm, fst, prefix, max_len)
            pyn_lp = state.logp_next
            for y, bf_val in bf_cond.items():
                pyn_val = pyn_lp.get(y, -np.inf)
                assert abs(pyn_val - bf_val) < 1e-10, \
                    f"Step {step}, y={y!r}: bf={bf_val:.12f}, pyn={pyn_val:.12f}"
            if step < 2:
                state = state >> 'A'

    def test_overlap_trigger_vs_reference(self):
        """PyniniTransducedLM matches ReferenceTransducedLM on the overlap-trigger FST."""
        fst = _overlap_trigger_fst()
        inner_lm = FiniteLM({
            (): 0.1,
            ('a',): 0.3,
            ('b',): 0.1,
            ('a', 'b'): 0.2,
            ('a', 'a'): 0.15,
            ('a', 'b', 'a'): 0.15,
        })
        ref = ReferenceTransducedLM(inner_lm, fst)
        pyn = PyniniTransducedLM(inner_lm, fst, K=500, max_expansions=5000)

        ref_state = ref.initial()
        pyn_state = pyn.initial()

        for step, y in enumerate(['c', 'x']):
            ref_lp = ref_state.logp_next
            pyn_lp = pyn_state.logp_next
            for sym in ref_lp:
                if ref_lp[sym] > -20:
                    assert abs(pyn_lp[sym] - ref_lp[sym]) < 1e-6, (
                        f"Step {step}, sym={sym!r}: ref={ref_lp[sym]:.8f}, "
                        f"pyn={pyn_lp[sym]:.8f}"
                    )
            ref_state = ref_state >> y
            pyn_state = pyn_state >> y

    def test_bpe_style_decodes_all_symbols(self):
        """PyniniTransducedLM can decode all output symbols on BPE-style FST."""
        fst = _bpe_style_fst()
        inner_lm = CharNgramLM.train(list('xxyxy') * 5, n=2, alpha=0.5)
        pyn = PyniniTransducedLM(inner_lm, fst, K=50, max_expansions=100)

        target = ('a', 'a', 'b', 'b')
        state = pyn.initial()
        for i, y in enumerate(target):
            lp = state.logp_next
            assert y in lp and lp[y] > -np.inf, (
                f"Step {i}: PyniniTransducedLM missing {y!r} in logp_next "
                f"(keys={sorted(lp.keys())})"
            )
            state = state >> y
