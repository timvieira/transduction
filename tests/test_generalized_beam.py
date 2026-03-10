"""Tests for GeneralizedBeam."""

import pytest
import numpy as np
from collections import defaultdict
from functools import cached_property
from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.util import LogDistr, LogVector, logsumexp
from transduction.lm.generalized_beam import (
    GeneralizedBeam, GeneralizedBeamState, OutputTrie, _compute_hub_vocab,
)
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.reference_transduced import ReferenceTransducedLM
from transduction.universality import compute_ip_universal_states


# ---------------------------------------------------------------------------
# Helpers (copied from test_transduced.py)
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
    def __init__(self, lm, probs, logprefix=0.0, _history_id=0):
        self._lm = lm
        self._probs = probs
        self.logprefix = logprefix
        self._history_id = _history_id

    @property
    def logp_next(self):
        return LogDistr(self._probs)

    @property
    def eos(self):
        return None

    def __rshift__(self, token):
        lp = self._probs.get(token, -np.inf)
        return TinyState(self._lm, self._probs, self.logprefix + lp,
                         _history_id=self._lm._history_pool.intern(self._history_id, token))


class TinyLM(LM):
    def __init__(self):
        self.eos = None
    def initial(self):
        probs = {'a': np.log(0.6), 'b': np.log(0.3), None: np.log(0.1)}
        return TinyState(self, probs)


class FiniteLMState(LMState):
    """State for a finite-support LM. Computes exact conditionals from the trie."""

    def __init__(self, lm, prefix, logprefix, _history_id=0):
        self._lm = lm
        self._prefix = prefix
        self.logprefix = logprefix
        self.eos = lm.eos
        self._history_id = _history_id

    def _prefix_mass(self, prefix):
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
        return FiniteLMState(self._lm, self._prefix + (token,), self.logprefix + lp,
                             _history_id=self._lm._history_pool.intern(self._history_id, token))


class FiniteLM(LM):
    """LM with exact support on a finite set of strings."""

    def __init__(self, string_probs, eos=None):
        self.eos = eos
        self._string_probs = string_probs

    def initial(self):
        return FiniteLMState(self, (), 0.0)


def brute_force_pushforward(inner_lm, fst, target, max_source_len=8):
    """Exhaustive enumeration of P_fst(target_prefix + y) for all y."""
    output_probs = LogVector()

    def source_logp(source):
        state = inner_lm(source)
        return state.logprefix + state.logp_next[state.eos]

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
def char_ngram_lm():
    from transduction.lm.ngram import CharNgramLM
    return CharNgramLM.train("aabbaabb" * 5, n=2, alpha=0.5)


# ---------------------------------------------------------------------------
# TestGeneralizedBeamBasic
# ---------------------------------------------------------------------------

class TestGeneralizedBeamBasic:

    def test_copy_fst_matches_inner(self):
        """GeneralizedBeam with copy FST should approximate the inner LM."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)
        state = gb.initial()

        inner_state = inner_lm.initial()
        for y in ['a', 'b']:
            inner_lp = inner_state.logp_next[y]
            got = state.logp_next[y]
            if inner_lp > -10:
                assert abs(got - inner_lp) < 1.0, \
                    f"Symbol {y!r}: inner={inner_lp:.4f}, gb={got:.4f}"

    def test_normalization(self):
        """logp_next should sum to approximately 1."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)
        state = gb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in ['a', 'b']] + [lp[state.eos]]
        total = logsumexp(all_logps)

        assert abs(total) < 0.5, \
            f"Probabilities should sum to ~1 (log ~0), got log-sum={total:.6f}"

    def test_logp_starts_at_zero(self):
        """Initial state has logprefix = 0."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        gb = GeneralizedBeam(inner_lm, fst, K=100)
        state = gb.initial()
        assert state.logprefix == 0.0

    def test_logp_accumulates(self):
        """logp after >> y1 >> y2 equals sum of logp_next[y_i]."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)

        state0 = gb.initial()
        lp1 = state0.logp_next['a']

        state1 = state0 >> 'a'
        lp2 = state1.logp_next['b']

        state2 = state1 >> 'b'
        expected = lp1 + lp2
        assert state2.logprefix == pytest.approx(expected, abs=1e-10)

    def test_repr_doesnt_crash(self):
        """repr shouldn't crash."""
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        gb = GeneralizedBeam(inner_lm, fst, K=10)
        state = gb.initial()
        assert 'GeneralizedBeam' in repr(gb)
        assert 'GeneralizedBeamState' in repr(state)


# ---------------------------------------------------------------------------
# TestGeneralizedBeamVsFused
# ---------------------------------------------------------------------------

class TestGeneralizedBeamVsFused:
    """Compare GeneralizedBeam against FusedTransducedLM on various FSTs."""

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
        ("lowercase", examples.lowercase),
        ("delete_b", examples.delete_b),
        ("triplets_of_doom", examples.triplets_of_doom),
    ])
    def test_initial_logp_next(self, fst_name, fst_fn):
        """Initial logp_next should approximately match FusedTransducedLM."""
        inner_lm = TinyLM()
        fst = fst_fn()

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        gb_state = gb.initial()
        fused_state = fused.initial()

        gb_lp = gb_state.logp_next
        fused_lp = fused_state.logp_next

        # Check all symbols that appear in either distribution
        all_syms = set(gb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            gb_val = gb_lp[y]
            fused_val = fused_lp[y]
            # Only compare non-negligible probabilities
            if gb_val > -10 or fused_val > -10:
                assert abs(gb_val - fused_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: gb={gb_val:.4f}, fused={fused_val:.4f}"

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
    ])
    def test_after_one_advance(self, fst_name, fst_fn):
        """After one advance, distributions should still approximately match."""
        inner_lm = TinyLM()
        fst = fst_fn()

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        # Find a common symbol to advance on
        gb_state = gb.initial()
        fused_state = fused.initial()

        # Pick the highest-probability symbol
        sym = gb_state.logp_next.argmax()
        if sym == gb_state.eos:
            return  # Skip if EOS is most likely

        gb_state = gb_state >> sym
        fused_state = fused_state >> sym

        gb_lp = gb_state.logp_next
        fused_lp = fused_state.logp_next

        all_syms = set(gb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            gb_val = gb_lp[y]
            fused_val = fused_lp[y]
            if gb_val > -10 or fused_val > -10:
                assert abs(gb_val - fused_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: gb={gb_val:.4f}, fused={fused_val:.4f}"


# ---------------------------------------------------------------------------
# TestGeneralizedBeamVsReference
# ---------------------------------------------------------------------------

class TestGeneralizedBeamVsReference:
    """Compare against ReferenceTransducedLM (exact) for small finite FSTs."""

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
    ])
    def test_reference_match(self, fst_name, fst_fn):
        """logp_next should closely match reference for finite FSTs."""
        inner_lm = TinyLM()
        fst = fst_fn()

        gb = GeneralizedBeam(inner_lm, fst, K=200, max_beam=500, max_steps=5000)
        ref = ReferenceTransducedLM(inner_lm, fst)

        gb_state = gb.initial()
        ref_state = ref.initial()

        gb_lp = gb_state.logp_next
        ref_lp = ref_state.logp_next

        all_syms = set(gb_lp.keys()) | set(ref_lp.keys())
        for y in all_syms:
            gb_val = gb_lp[y]
            ref_val = ref_lp[y]
            if ref_val > -10:
                assert abs(gb_val - ref_val) < 0.5, \
                    f"[{fst_name}] Symbol {y!r}: gb={gb_val:.4f}, ref={ref_val:.4f}"


# ---------------------------------------------------------------------------
# TestGeneralizedBeamExact
# ---------------------------------------------------------------------------

class TestGeneralizedBeamExact:

    def test_delete_b_exact(self):
        """delete_b with TinyLM: exact analytical answer P(A)=6/7, P(EOS)=1/7."""
        inner_lm = TinyLM()
        fst = examples.delete_b()

        gb = GeneralizedBeam(inner_lm, fst, K=200, max_beam=500, max_steps=5000)
        state = gb.initial()

        # P_inner('a') = 0.6, P_inner('b') = 0.3, P_inner(EOS) = 0.1
        # Under delete_b: b is deleted, a -> A, EOS -> EOS
        # P(A) = P_inner(a) / (P_inner(a) + P_inner(EOS)) = 0.6/0.7 = 6/7
        # P(EOS) = P_inner(EOS) / (P_inner(a) + P_inner(EOS)) = 0.1/0.7 = 1/7
        # But the b's get absorbed into quotient states, so:
        # P(A | empty) should be high (close to 6/7)

        lp_A = state.logp_next['A']
        lp_eos = state.logp_next[state.eos]

        expected_A = np.log(6/7)
        expected_eos = np.log(1/7)

        assert abs(lp_A - expected_A) < 0.5, \
            f"P(A): expected={expected_A:.4f}, got={lp_A:.4f}"
        assert abs(lp_eos - expected_eos) < 0.5, \
            f"P(EOS): expected={expected_eos:.4f}, got={lp_eos:.4f}"


# ---------------------------------------------------------------------------
# TestMultiHubFSTs
# ---------------------------------------------------------------------------

class TestMultiHubFSTs:

    def test_two_hub_alternating_normalization(self):
        """two_hub_alternating should produce a normalized distribution."""
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()

        gb = GeneralizedBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = gb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_two_hub_alternating_has_hubs(self):
        """two_hub_alternating should have hub tries for both hubs."""
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()
        gb = GeneralizedBeam(inner_lm, fst, K=10)
        assert len(gb._hub_tries) == 2

    def test_hub_with_escape_normalization(self):
        """hub_with_escape should produce a normalized distribution."""
        inner_lm = TinyLM()
        fst = examples.hub_with_escape()

        gb = GeneralizedBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = gb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_hub_with_escape_has_hubs(self):
        """hub_with_escape should have deterministic hubs (root and E, not C)."""
        inner_lm = TinyLM()
        fst = examples.hub_with_escape()
        gb = GeneralizedBeam(inner_lm, fst, K=10)
        # root and E are deterministic hubs; C is non-deterministic → rejected
        assert len(gb._hub_tries) == 2
        assert 'root' in gb._hub_tries

    def test_multi_hub_chain_normalization(self):
        """multi_hub_chain should produce a normalized distribution."""
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)

        gb = GeneralizedBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = gb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_multi_hub_chain_has_hubs(self):
        """multi_hub_chain(3) should have 3 hubs."""
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)
        gb = GeneralizedBeam(inner_lm, fst, K=10)
        assert len(gb._hub_tries) == 3

    def test_partial_hub_vocab_rejected(self):
        """partial_hub's non-deterministic output should be rejected by _compute_hub_vocab."""
        fst = examples.partial_hub()
        # _compute_hub_vocab returns None for non-deterministic vocab
        assert _compute_hub_vocab(fst, 0) is None

    def test_no_hub_transducer_matches_fused(self):
        """no_hub_transducer in pure particle mode should match FusedTransducedLM."""
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        gb_lp = gb.initial().logp_next
        fused_lp = fused.initial().logp_next

        all_syms = set(gb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            gb_val = gb_lp[y]
            fused_val = fused_lp[y]
            if gb_val > -10 or fused_val > -10:
                assert abs(gb_val - fused_val) < 0.5, \
                    f"Symbol {y!r}: gb={gb_val:.4f}, fused={fused_val:.4f}"

    def test_no_hub_transducer_has_no_hubs(self):
        """no_hub_transducer should have zero hubs."""
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()
        gb = GeneralizedBeam(inner_lm, fst, K=10)
        assert len(gb._hub_tries) == 0


# ---------------------------------------------------------------------------
# TestMultiHubBruteForce
# ---------------------------------------------------------------------------

class TestMultiHubBruteForce:
    """Compare multi-hub FSTs against brute-force enumeration."""

    def test_two_hub_alternating(self):
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = gb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        # Check each first symbol
        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                gb_y = state.logp_next[y]
                assert abs(gb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, gb={gb_y:.4f}"

    def test_multi_hub_chain(self):
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = gb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                gb_y = state.logp_next[y]
                assert abs(gb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, gb={gb_y:.4f}"

    def test_no_hub_transducer(self):
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()

        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = gb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                gb_y = state.logp_next[y]
                assert abs(gb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, gb={gb_y:.4f}"


# ---------------------------------------------------------------------------
# TestOutputTrie
# ---------------------------------------------------------------------------

class TestOutputTrie:

    def test_basic_trie(self):
        """OutputTrie with simple entries computes mass correctly."""
        entries = [
            ('t1', ('x', 'y'), 'dest1'),
            ('t2', ('x', 'z'), 'dest2'),
        ]
        trie = OutputTrie(entries, inner_eos=None)
        assert not trie.is_empty

        # Root should have one child 'x'
        assert 'x' in trie.children[0]

    def test_empty_trie(self):
        """OutputTrie with no entries is empty."""
        trie = OutputTrie([], inner_eos=None)
        assert trie.is_empty


# ---------------------------------------------------------------------------
# TestHubVocabComputation
# ---------------------------------------------------------------------------

class TestHubVocabComputation:

    def test_copy_fst_hub_vocab(self):
        """Copy FST hub should have deterministic entries."""
        fst = copy_fst(['a', 'b'])
        entries = _compute_hub_vocab(fst, 0)
        assert entries is not None
        assert len(entries) == 2

    def test_partial_hub_returns_none(self):
        """Non-deterministic hub should return None."""
        fst = examples.partial_hub()
        entries = _compute_hub_vocab(fst, 0)
        assert entries is None

    def test_hub_ip_universal_check(self):
        """compute_ip_universal_states finds hubs correctly."""
        fst = copy_fst(['a', 'b'])
        ip_univ = compute_ip_universal_states(fst)
        assert 0 in ip_univ

    def test_two_hub_ip_universal(self):
        """Both states in two_hub_alternating should be IP-universal."""
        fst = examples.two_hub_alternating()
        ip_univ = compute_ip_universal_states(fst)
        assert 'h0' in ip_univ
        assert 'h1' in ip_univ
