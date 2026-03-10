"""Full test battery for CompiledBeam — mirrors test_generalized_beam.py."""

import pytest
import numpy as np
from collections import defaultdict
from functools import cached_property
from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.util import LogDistr, LogVector, logsumexp
from transduction.lm.compiled_beam import (
    CompiledBeam, CompiledBeamState, RegionAnalyzer, RegionMap,
    HubRegion, WildRegion, CorridorRegion, UniversalPlateauRegion,
    Region, Hyp, ScoredOutput,
)
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.generalized_beam import GeneralizedBeam, OutputTrie, _compute_hub_vocab
from transduction.lm.reference_transduced import ReferenceTransducedLM
from transduction.universality import compute_ip_universal_states


# ---------------------------------------------------------------------------
# Test helpers (same as test_generalized_beam.py)
# ---------------------------------------------------------------------------

def copy_fst(alphabet):
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    for x in alphabet:
        fst.add_arc(0, x, x, 0)
    return fst


class TinyState(LMState):
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
        return '<EOS>'

    def __rshift__(self, token):
        lp = self._probs.get(token, -np.inf)
        return TinyState(self._lm, self._probs, self.logprefix + lp,
                         _history_id=self._lm._history_pool.intern(self._history_id, token))


class TinyLM(LM):
    def __init__(self):
        self.eos = '<EOS>'
    def initial(self):
        probs = {'a': np.log(0.6), 'b': np.log(0.3), '<EOS>': np.log(0.1)}
        return TinyState(self, probs)


class FiniteLMState(LMState):
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
    def __init__(self, string_probs, eos='<EOS>'):
        self.eos = eos
        self._string_probs = string_probs

    def initial(self):
        return FiniteLMState(self, (), 0.0)


def brute_force_pushforward(inner_lm, fst, target, max_source_len=8):
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
# TestCompiledBeamBasic
# ---------------------------------------------------------------------------

class TestCompiledBeamBasic:

    def test_copy_fst_matches_inner(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)
        state = cb.initial()

        inner_state = inner_lm.initial()
        for y in ['a', 'b']:
            inner_lp = inner_state.logp_next[y]
            got = state.logp_next[y]
            if inner_lp > -10:
                assert abs(got - inner_lp) < 1.0, \
                    f"Symbol {y!r}: inner={inner_lp:.4f}, cb={got:.4f}"

    def test_normalization(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)
        state = cb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in ['a', 'b']] + [lp[state.eos]]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_logp_starts_at_zero(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=100)
        state = cb.initial()
        assert state.logprefix == 0.0

    def test_logp_accumulates(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=100, max_steps=5000)

        state0 = cb.initial()
        lp1 = state0.logp_next['a']
        state1 = state0 >> 'a'
        lp2 = state1.logp_next['b']
        state2 = state1 >> 'b'
        expected = lp1 + lp2
        assert state2.logprefix == pytest.approx(expected, abs=1e-10)

    def test_repr_doesnt_crash(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=10)
        state = cb.initial()
        assert 'CompiledBeam' in repr(cb)
        assert 'CompiledBeamState' in repr(state)


# ---------------------------------------------------------------------------
# TestCompiledBeamVsFused — same FSTs as test_generalized_beam
# ---------------------------------------------------------------------------

class TestCompiledBeamVsFused:

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
        ("lowercase", examples.lowercase),
        ("delete_b", examples.delete_b),
        ("triplets_of_doom", examples.triplets_of_doom),
    ])
    def test_initial_logp_next(self, fst_name, fst_fn):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        cb_lp = cb.initial().logp_next
        fused_lp = fused.initial().logp_next

        all_syms = set(cb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            cb_val = cb_lp[y]
            fused_val = fused_lp[y]
            if cb_val > -10 or fused_val > -10:
                assert abs(cb_val - fused_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: cb={cb_val:.4f}, fused={fused_val:.4f}"

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
    ])
    def test_after_one_advance(self, fst_name, fst_fn):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        cb_state = cb.initial()
        fused_state = fused.initial()

        sym = cb_state.logp_next.argmax()
        if sym == cb_state.eos:
            return

        cb_state = cb_state >> sym
        fused_state = fused_state >> sym

        cb_lp = cb_state.logp_next
        fused_lp = fused_state.logp_next

        all_syms = set(cb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            cb_val = cb_lp[y]
            fused_val = fused_lp[y]
            if cb_val > -10 or fused_val > -10:
                assert abs(cb_val - fused_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: cb={cb_val:.4f}, fused={fused_val:.4f}"


# ---------------------------------------------------------------------------
# TestCompiledBeamVsGeneralizedBeam — direct comparison
# ---------------------------------------------------------------------------

class TestCompiledBeamVsGeneralizedBeam:

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
        ("lowercase", examples.lowercase),
        ("delete_b", examples.delete_b),
        ("triplets_of_doom", examples.triplets_of_doom),
    ])
    def test_initial_logp_next(self, fst_name, fst_fn):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)

        cb_lp = cb.initial().logp_next
        gb_lp = gb.initial().logp_next

        all_syms = set(cb_lp.keys()) | set(gb_lp.keys())
        for y in all_syms:
            cb_val = cb_lp[y]
            gb_val = gb_lp[y]
            if cb_val > -10 or gb_val > -10:
                assert abs(cb_val - gb_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: cb={cb_val:.4f}, gb={gb_val:.4f}"

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
        ("lowercase", examples.lowercase),
    ])
    def test_after_two_advances(self, fst_name, fst_fn):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)

        cb_state = cb.initial()
        gb_state = gb.initial()

        for step in range(2):
            sym = cb_state.logp_next.argmax()
            if sym == cb_state.eos:
                return
            cb_state = cb_state >> sym
            gb_state = gb_state >> sym

        cb_lp = cb_state.logp_next
        gb_lp = gb_state.logp_next

        all_syms = set(cb_lp.keys()) | set(gb_lp.keys())
        for y in all_syms:
            cb_val = cb_lp[y]
            gb_val = gb_lp[y]
            if cb_val > -10 or gb_val > -10:
                assert abs(cb_val - gb_val) < 1.0, \
                    f"[{fst_name}] Symbol {y!r}: cb={cb_val:.4f}, gb={gb_val:.4f}"


# ---------------------------------------------------------------------------
# TestCompiledBeamExact — analytical ground truth
# ---------------------------------------------------------------------------

class TestCompiledBeamExact:

    def test_delete_b_exact(self):
        inner_lm = TinyLM()
        fst = examples.delete_b()

        cb = CompiledBeam(inner_lm, fst, K=200, max_beam=500, max_steps=5000)
        state = cb.initial()

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
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()
        cb = CompiledBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = cb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_two_hub_has_hub_regions(self):
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()
        cb = CompiledBeam(inner_lm, fst, K=10)
        assert len(cb._hub_regions) == 2

    def test_hub_with_escape_normalization(self):
        inner_lm = TinyLM()
        fst = examples.hub_with_escape()
        cb = CompiledBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = cb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_hub_with_escape_has_hubs(self):
        inner_lm = TinyLM()
        fst = examples.hub_with_escape()
        cb = CompiledBeam(inner_lm, fst, K=10)
        assert len(cb._hub_regions) == 2
        assert 'root' in cb._hub_regions

    def test_multi_hub_chain_normalization(self):
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)
        cb = CompiledBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = cb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

    def test_multi_hub_chain_has_hubs(self):
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)
        cb = CompiledBeam(inner_lm, fst, K=10)
        assert len(cb._hub_regions) == 3

    def test_no_hub_matches_fused(self):
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        fused = FusedTransducedLM(inner_lm, fst, max_steps=2000, max_beam=200,
                                  helper="python")

        cb_lp = cb.initial().logp_next
        fused_lp = fused.initial().logp_next

        all_syms = set(cb_lp.keys()) | set(fused_lp.keys())
        for y in all_syms:
            cb_val = cb_lp[y]
            fused_val = fused_lp[y]
            if cb_val > -10 or fused_val > -10:
                assert abs(cb_val - fused_val) < 0.5, \
                    f"Symbol {y!r}: cb={cb_val:.4f}, fused={fused_val:.4f}"

    def test_no_hub_has_no_hub_regions(self):
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()
        cb = CompiledBeam(inner_lm, fst, K=10)
        assert len(cb._hub_regions) == 0


# ---------------------------------------------------------------------------
# TestMultiHubBruteForce
# ---------------------------------------------------------------------------

class TestMultiHubBruteForce:

    def test_two_hub_alternating(self):
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = cb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                cb_y = state.logp_next[y]
                assert abs(cb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, cb={cb_y:.4f}"

    def test_multi_hub_chain(self):
        inner_lm = TinyLM()
        fst = examples.multi_hub_chain(n=3)

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = cb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                cb_y = state.logp_next[y]
                assert abs(cb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, cb={cb_y:.4f}"

    def test_no_hub_transducer(self):
        inner_lm = TinyLM()
        fst = examples.no_hub_transducer()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=5000)
        state = cb.initial()

        bf = brute_force_pushforward(inner_lm, fst, '', max_source_len=6)
        Z = logsumexp(list(bf.values()))

        for y in set(state.logp_next.keys()) - {state.eos}:
            y_strings = {k: v for k, v in bf.items() if k and k[0] == y}
            if y_strings:
                bf_y = logsumexp(list(y_strings.values())) - Z
                cb_y = state.logp_next[y]
                assert abs(cb_y - bf_y) < 0.5, \
                    f"Symbol {y!r}: bf={bf_y:.4f}, cb={cb_y:.4f}"


# ---------------------------------------------------------------------------
# TestRegionAnalyzer
# ---------------------------------------------------------------------------

class TestRegionAnalyzer:

    def test_copy_fst_single_hub(self):
        fst = copy_fst(['a', 'b'])
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert len(rmap.hub_regions) == 1
        assert rmap.wild_region is None

    def test_no_hub_has_corridors(self):
        fst = examples.no_hub_transducer()
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert len(rmap.hub_regions) == 0
        assert len(rmap.corridor_regions) > 0
        # All states covered by corridors — no wild needed
        assert rmap.wild_region is None

    def test_two_hub(self):
        fst = examples.two_hub_alternating()
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert len(rmap.hub_regions) == 2

    def test_multi_hub_chain(self):
        fst = examples.multi_hub_chain(n=4)
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert len(rmap.hub_regions) == 4

    def test_hub_with_escape_mixed(self):
        fst = examples.hub_with_escape()
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert len(rmap.hub_regions) >= 1
        # Should need wild for non-hub escape path
        # (depends on whether escape state gets hub classification)

    def test_region_for_dispatches(self):
        fst = copy_fst(['a', 'b'])
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        region = rmap.region_for(0)
        assert isinstance(region, HubRegion)

    def test_region_for_corridor(self):
        """no_hub_transducer: states are corridors, scoring uses corridor region."""
        fst = examples.no_hub_transducer()
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        region = rmap.region_for(0)
        assert isinstance(region, CorridorRegion)
        # All states covered — no wild needed
        assert rmap.wild_region is None

    def test_summary(self):
        fst = copy_fst(['a', 'b'])
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        s = rmap.summary()
        assert 'hub' in s

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("small", examples.small),
        ("lowercase", examples.lowercase),
        ("delete_b", examples.delete_b),
        ("triplets_of_doom", examples.triplets_of_doom),
        ("bpe_like", lambda: examples.bpe_like(vocab_size=20)),
    ])
    def test_analyzer_doesnt_crash(self, fst_name, fst_fn):
        fst = fst_fn()
        rmap = RegionAnalyzer(fst, TinyLM()).analyze()
        assert isinstance(rmap, RegionMap)


# ---------------------------------------------------------------------------
# TestScoredOutput
# ---------------------------------------------------------------------------

class TestScoredOutput:

    def test_merge(self):
        a = ScoredOutput()
        a.scores.logaddexp('x', -1.0)
        a.eos_score = -2.0

        b = ScoredOutput()
        b.scores.logaddexp('x', -1.5)
        b.scores.logaddexp('y', -0.5)
        b.eos_score = -3.0

        a.merge(b)
        assert 'x' in a.scores
        assert 'y' in a.scores
        assert a.eos_score > -2.0  # merged EOS should be higher

    def test_empty_merge(self):
        a = ScoredOutput()
        b = ScoredOutput()
        a.merge(b)
        assert a.eos_score == float('-inf')


# ---------------------------------------------------------------------------
# TestDuplicateSpelling
# ---------------------------------------------------------------------------

def _duplicate_spelling_fst():
    """FST where source symbols 'x' and 'y' both produce output 'ab'.

    Hub state 0 has two arcs:
        0 --x:ε--> 1 --ε:a--> 2 --ε:b--> 0
        0 --y:ε--> 3 --ε:a--> 4 --ε:b--> 0
    """
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'x', EPSILON, 1)
    fst.add_arc(1, EPSILON, 'a', 2)
    fst.add_arc(2, EPSILON, 'b', 0)
    fst.add_arc(0, 'y', EPSILON, 3)
    fst.add_arc(3, EPSILON, 'a', 4)
    fst.add_arc(4, EPSILON, 'b', 0)
    return fst


class TestDuplicateSpelling:

    def test_cb_vs_reference_duplicate_spelling(self):
        """CompiledBeam should match ReferenceTransducedLM when source
        symbols share output spellings."""
        fst = _duplicate_spelling_fst()

        string_probs = {
            ('x',): 0.3,
            ('y',): 0.1,
            ('x', 'x'): 0.2,
            ('x', 'y'): 0.1,
            ('y', 'x'): 0.15,
            ('y', 'y'): 0.15,
        }
        inner_lm = FiniteLM(string_probs, eos=None)

        cb = CompiledBeam(inner_lm, fst, K=500, max_beam=500, max_steps=10000,
                          eos=None, helper='python')
        ref = ReferenceTransducedLM(inner_lm, fst)

        cb_state = cb.initial()
        ref_state = ref.initial()
        target = ['a', 'b', 'a', 'b']

        for i, y in enumerate(target):
            cb_lp = cb_state.logp_next
            ref_lp = ref_state.logp_next

            all_syms = set(cb_lp.keys()) | set(ref_lp.keys())
            for sym in all_syms:
                cb_val = cb_lp[sym]
                ref_val = ref_lp[sym]
                if cb_val > -10 or ref_val > -10:
                    assert abs(cb_val - ref_val) < 0.1, \
                        f"Step {i}, symbol {sym!r}: cb={cb_val:.4f}, ref={ref_val:.4f}"

            cb_state = cb_state >> y
            ref_state = ref_state >> y


# ---------------------------------------------------------------------------
# TestVsReference — multi-step reference comparison across region types
# ---------------------------------------------------------------------------

class TestVsReference:
    """Compare CompiledBeam against ReferenceTransducedLM (exact ground truth)
    for multi-step decoding on finite-relation FSTs."""

    @pytest.mark.parametrize("fst_name,fst_fn,steps", [
        ("hub_with_escape", examples.hub_with_escape, ['a', 'b']),
        ("small", examples.small, ['a', 'b']),
        ("copy_ab", lambda: copy_fst(['a', 'b']), ['a', 'b', 'a']),
        ("lowercase", examples.lowercase, ['A', 'B']),
    ])
    def test_multi_step_vs_reference(self, fst_name, fst_fn, steps):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=200, max_beam=500, max_steps=5000,
                          helper='python')
        ref = ReferenceTransducedLM(inner_lm, fst)

        cb_state = cb.initial()
        ref_state = ref.initial()

        for i, y in enumerate(steps):
            cb_lp = cb_state.logp_next
            ref_lp = ref_state.logp_next

            all_syms = set(cb_lp.keys()) | set(ref_lp.keys())
            for sym in all_syms:
                cb_val = cb_lp[sym]
                ref_val = ref_lp[sym]
                if cb_val > -10 or ref_val > -10:
                    assert abs(cb_val - ref_val) < 0.5, \
                        f"[{fst_name}] Step {i}, symbol {sym!r}: " \
                        f"cb={cb_val:.4f}, ref={ref_val:.4f}"

            if y not in ref_lp or ref_lp[y] <= -50:
                break
            cb_state = cb_state >> y
            ref_state = ref_state >> y


class TestVsFused:
    """Compare CompiledBeam against FusedTransducedLM for multi-step decoding
    on FSTs with infinite quotients (where ReferenceTransducedLM is too slow)."""

    @pytest.mark.parametrize("fst_name,fst_fn,steps", [
        ("triplets_of_doom", examples.triplets_of_doom, ['a', 'b', 'a']),
        ("no_hub_transducer", examples.no_hub_transducer, ['a', 'b']),
    ])
    def test_multi_step_vs_fused(self, fst_name, fst_fn, steps):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=200, max_beam=500, max_steps=5000,
                          helper='python')
        fused = FusedTransducedLM(inner_lm, fst, max_steps=5000, max_beam=500,
                                  helper='python')

        cb_state = cb.initial()
        fused_state = fused.initial()

        for i, y in enumerate(steps):
            cb_lp = cb_state.logp_next
            fused_lp = fused_state.logp_next

            all_syms = set(cb_lp.keys()) | set(fused_lp.keys())
            for sym in all_syms:
                cb_val = cb_lp[sym]
                fused_val = fused_lp[sym]
                if cb_val > -10 or fused_val > -10:
                    assert abs(cb_val - fused_val) < 0.5, \
                        f"[{fst_name}] Step {i}, symbol {sym!r}: " \
                        f"cb={cb_val:.4f}, fused={fused_val:.4f}"

            if y not in fused_lp or fused_lp[y] <= -50:
                break
            cb_state = cb_state >> y
            fused_state = fused_state >> y
