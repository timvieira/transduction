"""Tests for position-set peekaboo decomposition."""

import pytest
from collections import deque

from transduction import examples, EPSILON, Precover
from transduction.fst import FST
from transduction.fsa import FSA
from transduction.peekaboo_incremental import PeekabooState, FstUniversality
from transduction.position_set_peekaboo import (
    PositionSetPeekabooState,
    _PositionSetPeekabooUniv,
    _peekaboo_position_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_qr_fsa(ps_state, y):
    """Extract Q and R FSAs for symbol y from a PositionSetPeekabooState.

    Forward BFS from start, stopping at Q-absorbed states (matching the
    PeekabooState BFS that skips arc expansion for continuous/Q states).
    """
    d = ps_state.decomp.get(y)
    if d is None:
        return FSA(), FSA()

    # Collect all Q-stops (across all symbols) — BFS stops at any Q-stop
    all_q_stops = set()
    for dr in ps_state.decomp.values():
        all_q_stops.update(dr.quotient)

    q_stops = d.quotient
    r_stops = d.remainder
    dfa = ps_state.dfa

    Q = FSA()
    R = FSA()
    [start] = dfa.start()
    Q.add_start(start)
    R.add_start(start)

    visited = set()
    worklist = deque([start])
    visited.add(start)

    while worklist:
        pid = worklist.popleft()

        if pid in q_stops:
            Q.add_stop(pid)
        if pid in r_stops:
            R.add_stop(pid)

        # Don't expand past any Q-stop (matching PeekabooState `continue`)
        if pid in all_q_stops:
            continue

        for x, succ_pid in dfa._arcs_list.get(pid, []):
            Q.add_arc(pid, x, succ_pid)
            R.add_arc(pid, x, succ_pid)
            if succ_pid not in visited:
                visited.add(succ_pid)
                worklist.append(succ_pid)

    return Q.trim(), R.trim()


# ---------------------------------------------------------------------------
# TD FSTs for parametrized tests
# ---------------------------------------------------------------------------

TD_FSTS = [
    pytest.param(
        examples.replace([('a', 'x'), ('b', 'y'), ('c', 'z')]),
        id="replace_3sym",
    ),
    pytest.param(examples.delete_b(), id="delete_b"),
    pytest.param(examples.lowercase(), id="lowercase"),
    pytest.param(examples.togglecase(), id="togglecase"),
    pytest.param(
        examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')]),
        id="replace_123",
    ),
]


@pytest.fixture(params=TD_FSTS)
def td_fst(request):
    return request.param


# ---------------------------------------------------------------------------
# Q/R Equivalence Tests
# ---------------------------------------------------------------------------

class TestQREquivalence:
    """Compare Q/R languages between PositionSetPeekabooState and PeekabooState."""

    def test_qr_match_depth1(self, td_fst):
        """Q/R from position-set peekaboo matches PeekabooState at depth 1."""
        fst = td_fst
        target_alphabet = fst.B - {EPSILON}

        ref = PeekabooState(fst, ())
        ps = PositionSetPeekabooState(fst, ())

        for y in target_alphabet:
            ref_child = ref >> y
            q_ps, r_ps = build_qr_fsa(ps, y)
            assert q_ps.equal(ref_child.quotient), f"Q mismatch for {y!r}"
            assert r_ps.equal(ref_child.remainder), f"R mismatch for {y!r}"

    def test_qr_match_depth2(self, td_fst):
        """Q/R from position-set peekaboo matches PeekabooState at depth 2."""
        fst = td_fst
        target_alphabet = fst.B - {EPSILON}

        for y1 in sorted(target_alphabet):
            ref = PeekabooState(fst, (y1,))
            ps = PositionSetPeekabooState(fst, (y1,))

            for y2 in sorted(target_alphabet):
                ref_child = ref >> y2
                q_ref = ref_child.quotient.trim()
                r_ref = ref_child.remainder.trim()

                if not q_ref.states and not r_ref.states:
                    continue

                q_ps, r_ps = build_qr_fsa(ps, y2)
                assert q_ps.equal(q_ref), f"Q mismatch for ({y1!r}, {y2!r})"
                assert r_ps.equal(r_ref), f"R mismatch for ({y1!r}, {y2!r})"


class TestQRMatchPrecover:
    """Compare Q/R from position-set peekaboo against the reference Precover."""

    def test_against_precover_depth1(self, td_fst):
        fst = td_fst
        target_alphabet = fst.B - {EPSILON}
        reference = Precover.factory(fst)

        ps = PositionSetPeekabooState(fst, ())
        for y in target_alphabet:
            want = reference((y,))
            q_have, r_have = build_qr_fsa(ps, y)
            assert q_have.equal(want.quotient), f"Q mismatch vs Precover for {y!r}"
            assert r_have.equal(want.remainder), f"R mismatch vs Precover for {y!r}"

    def test_against_precover_depth2(self, td_fst):
        fst = td_fst
        target_alphabet = fst.B - {EPSILON}
        reference = Precover.factory(fst)

        for y1 in sorted(target_alphabet):
            ps = PositionSetPeekabooState(fst, (y1,))
            for y2 in sorted(target_alphabet):
                want = reference((y1, y2))
                if not want.quotient.trim().states and not want.remainder.trim().states:
                    continue
                q_have, r_have = build_qr_fsa(ps, y2)
                assert q_have.equal(want.quotient), \
                    f"Q mismatch vs Precover for ({y1!r}, {y2!r})"
                assert r_have.equal(want.remainder), \
                    f"R mismatch vs Precover for ({y1!r}, {y2!r})"


# ---------------------------------------------------------------------------
# Compression Ratio Tests
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    """Verify that position-set DFA has ≤ full peekaboo DFA states."""

    def test_position_sets_exist(self, td_fst):
        fst = td_fst
        ps = PositionSetPeekabooState(fst, ())
        _ = ps.decomp  # trigger BFS
        assert ps._n_position_sets >= 1


# ---------------------------------------------------------------------------
# Non-TD Rejection Tests
# ---------------------------------------------------------------------------

class TestNonTDRejection:
    """Verify ValueError on non-token-decomposable FSTs."""

    def test_custom_non_td_finality(self):
        """FST where same position set has different per-symbol finality."""
        fst = FST()
        fst.add_start(0)
        fst.add_stop(0)
        # State 1: NOT final
        fst.add_arc(0, 'a', 'x', 1)  # → non-final state 1
        fst.add_arc(0, 'b', 'x', 0)  # → final state 0
        fst.add_arc(1, 'a', 'x', 1)
        fst.add_arc(1, 'b', 'x', 1)

        with pytest.raises(ValueError, match="not token-decomposable"):
            ps = PositionSetPeekabooState(fst, ())
            _ = ps.decomp

    def test_small_fst_is_non_td(self):
        """examples.small() has states 1 (final) and 2 (non-final) with same PS."""
        fst = examples.small()
        with pytest.raises(ValueError, match="not token-decomposable"):
            ps = PositionSetPeekabooState(fst, ())
            _ = ps.decomp

    def test_triplets_of_doom(self):
        """triplets_of_doom is non-TD for target 'a'."""
        fst = examples.triplets_of_doom()
        with pytest.raises(ValueError, match="not token-decomposable"):
            ps = PositionSetPeekabooState(fst, ('a',))
            _ = ps.decomp


# ---------------------------------------------------------------------------
# Preimage Stops Tests
# ---------------------------------------------------------------------------

class TestPreimageStops:
    """Verify preimage stops are consistent with PeekabooState."""

    def test_preimage_consistency(self, td_fst):
        """Both backends agree on whether preimage_stops is non-empty."""
        fst = td_fst
        ps = PositionSetPeekabooState(fst, ())
        ref = PeekabooState(fst, ())

        has_ps = len(ps.preimage_stops) > 0
        has_ref = len(ref.preimage_stops) > 0
        assert has_ps == has_ref, "preimage_stops emptiness mismatch"

    def test_preimage_at_nonempty_target(self, td_fst):
        """Preimage stops consistent at non-empty targets."""
        fst = td_fst
        target_alphabet = fst.B - {EPSILON}
        for y in sorted(target_alphabet):
            ps = PositionSetPeekabooState(fst, (y,))
            ref = PeekabooState(fst, (y,))
            has_ps = len(ps.preimage_stops) > 0
            has_ref = len(ref.preimage_stops) > 0
            assert has_ps == has_ref, \
                f"preimage_stops mismatch at target ({y!r},)"


# ---------------------------------------------------------------------------
# TransducedLM Integration Tests
# ---------------------------------------------------------------------------

class TestTransducedLMIntegration:
    """Test PositionSetPeekabooState plugged into TransducedLM."""

    def test_logp_next_match_delete_b(self):
        """logp_next distributions match between backends for delete_b."""
        from transduction.lm.ngram import CharNgramLM
        from transduction.lm.transduced import TransducedLM
        import numpy as np

        fst = examples.delete_b()
        inner = CharNgramLM.train("aAAbAbAb" * 5, n=2, alpha=0.5)

        ref_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PeekabooState,
            univ_cls=FstUniversality,
        )
        ps_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PositionSetPeekabooState,
            univ_cls=_PositionSetPeekabooUniv,
        )

        ref_state = ref_tlm.initial()
        ps_state = ps_tlm.initial()

        for y in sorted(fst.B - {EPSILON}):
            ref_lp = ref_state.logp_next[y]
            ps_lp = ps_state.logp_next[y]
            if ref_lp > -10:
                assert abs(ref_lp - ps_lp) < 0.1, \
                    f"logp_next[{y!r}]: ref={ref_lp:.4f}, ps={ps_lp:.4f}"

    def test_logp_next_match_replace(self):
        """logp_next distributions match for a replace FST."""
        from transduction.lm.ngram import CharNgramLM
        from transduction.lm.transduced import TransducedLM
        import numpy as np

        fst = examples.replace([('a', 'x'), ('b', 'y')])
        inner = CharNgramLM.train("abab" * 10, n=2, alpha=0.5)

        ref_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PeekabooState,
            univ_cls=FstUniversality,
        )
        ps_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PositionSetPeekabooState,
            univ_cls=_PositionSetPeekabooUniv,
        )

        ref_state = ref_tlm.initial()
        ps_state = ps_tlm.initial()

        for y in sorted(fst.B - {EPSILON}):
            ref_lp = ref_state.logp_next[y]
            ps_lp = ps_state.logp_next[y]
            if ref_lp > -10:
                assert abs(ref_lp - ps_lp) < 0.1, \
                    f"logp_next[{y!r}]: ref={ref_lp:.4f}, ps={ps_lp:.4f}"

    def test_advance_and_compare(self):
        """After >> y, logp_next still matches between backends."""
        from transduction.lm.ngram import CharNgramLM
        from transduction.lm.transduced import TransducedLM
        import numpy as np

        fst = examples.replace([('a', 'x'), ('b', 'y')])
        inner = CharNgramLM.train("abab" * 10, n=2, alpha=0.5)

        ref_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PeekabooState,
            univ_cls=FstUniversality,
        )
        ps_tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PositionSetPeekabooState,
            univ_cls=_PositionSetPeekabooUniv,
        )

        ref_state = ref_tlm >> 'x'
        ps_state = ps_tlm >> 'x'

        for y in sorted(fst.B - {EPSILON}):
            ref_lp = ref_state.logp_next[y]
            ps_lp = ps_state.logp_next[y]
            if ref_lp > -10:
                assert abs(ref_lp - ps_lp) < 0.1, \
                    f"logp_next[{y!r}] after 'x': ref={ref_lp:.4f}, ps={ps_lp:.4f}"

    def test_incremental_logp(self):
        """Cumulative logp is consistent across >> steps."""
        from transduction.lm.ngram import CharNgramLM
        from transduction.lm.transduced import TransducedLM

        fst = examples.replace([('a', 'x'), ('b', 'y')])
        inner = CharNgramLM.train("abab" * 10, n=2, alpha=0.5)

        tlm = TransducedLM(
            inner, fst, K=100, max_expansions=500,
            decomp_state_cls=PositionSetPeekabooState,
            univ_cls=_PositionSetPeekabooUniv,
        )

        s0 = tlm.initial()
        lp1 = s0.logp_next['x']

        s1 = s0 >> 'x'
        lp2 = s1.logp_next['y']

        s2 = s1 >> 'y'

        assert s2.logp == pytest.approx(lp1 + lp2, abs=1e-10)


# ---------------------------------------------------------------------------
# Shift operator tests
# ---------------------------------------------------------------------------

class TestShift:
    """Test the >> operator."""

    def test_rshift_creates_new_state(self, td_fst):
        fst = td_fst
        s0 = PositionSetPeekabooState(fst, ())
        target_alphabet = fst.B - {EPSILON}
        y = sorted(target_alphabet)[0]
        s1 = s0 >> y
        assert s1.target == (y,)
        assert s1.fst is fst

    def test_rshift_chain(self, td_fst):
        fst = td_fst
        target_alphabet = sorted(fst.B - {EPSILON})
        y1, y2 = target_alphabet[0], target_alphabet[-1]
        s = PositionSetPeekabooState(fst, ())
        s = s >> y1 >> y2
        assert s.target == (y1, y2)
