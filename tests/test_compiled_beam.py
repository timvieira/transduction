"""Tests for CompiledBeam — region-based compiled beam search."""

import pytest
import numpy as np
from collections import defaultdict
from functools import cached_property
from transduction import examples, FST
from transduction.fst import EPSILON
from transduction.lm.base import LM, LMState
from transduction.util import LogDistr, LogVector, logsumexp
from transduction.lm.compiled_beam import CompiledBeam, RegionAnalyzer, HubRegion
from transduction.lm.fused_transduced import FusedTransducedLM
from transduction.lm.generalized_beam import GeneralizedBeam


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# TestRegionAnalyzer
# ---------------------------------------------------------------------------

class TestRegionAnalyzer:

    def test_copy_fst_single_hub(self):
        """Copy FST should produce a single hub region, no wild."""
        fst = copy_fst(['a', 'b'])
        inner = TinyLM()
        rmap = RegionAnalyzer(fst, inner).analyze()
        assert len(rmap.hub_regions) == 1
        assert rmap.wild_region is None

    def test_no_hub_has_corridors(self):
        """no_hub_transducer should produce zero hubs, corridors handle scoring."""
        fst = examples.no_hub_transducer()
        inner = TinyLM()
        rmap = RegionAnalyzer(fst, inner).analyze()
        assert len(rmap.hub_regions) == 0
        assert len(rmap.corridor_regions) > 0
        # All states covered by corridors — no wild needed
        assert rmap.wild_region is None

    def test_two_hub(self):
        """two_hub_alternating should have 2 hub regions."""
        fst = examples.two_hub_alternating()
        inner = TinyLM()
        rmap = RegionAnalyzer(fst, inner).analyze()
        assert len(rmap.hub_regions) == 2

    def test_summary_doesnt_crash(self):
        fst = copy_fst(['a', 'b'])
        inner = TinyLM()
        rmap = RegionAnalyzer(fst, inner).analyze()
        s = rmap.summary()
        assert 'hub' in s


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

    def test_repr(self):
        inner_lm = TinyLM()
        fst = copy_fst(['a', 'b'])
        cb = CompiledBeam(inner_lm, fst, K=10)
        state = cb.initial()
        assert 'CompiledBeam' in repr(cb)
        assert 'CompiledBeamState' in repr(state)


# ---------------------------------------------------------------------------
# TestCompiledBeamVsGeneralizedBeam
# ---------------------------------------------------------------------------

class TestCompiledBeamVsGeneralizedBeam:
    """CompiledBeam should match GeneralizedBeam on the same FSTs."""

    @pytest.mark.parametrize("fst_name,fst_fn", [
        ("copy_ab", lambda: copy_fst(['a', 'b'])),
        ("small", examples.small),
        ("lowercase", examples.lowercase),
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
    ])
    def test_after_one_advance(self, fst_name, fst_fn):
        inner_lm = TinyLM()
        fst = fst_fn()

        cb = CompiledBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)
        gb = GeneralizedBeam(inner_lm, fst, K=100, max_beam=200, max_steps=2000)

        cb_state = cb.initial()
        gb_state = gb.initial()

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
# TestCompiledBeamMultiHub
# ---------------------------------------------------------------------------

class TestCompiledBeamMultiHub:

    def test_two_hub_alternating_normalization(self):
        inner_lm = TinyLM()
        fst = examples.two_hub_alternating()
        cb = CompiledBeam(inner_lm, fst, K=50, max_beam=100, max_steps=1000)
        state = cb.initial()

        lp = state.logp_next
        all_logps = [lp[y] for y in lp.keys()]
        total = logsumexp(all_logps)
        assert abs(total) < 0.5, f"Should normalize, got log-sum={total:.6f}"

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
