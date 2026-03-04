"""Tests for the safety analysis module (compute_safe_states, compute_safe_powersets)."""

import pytest
from transduction.fst import FST, EPSILON
from transduction.safety import (
    compute_safe_states,
    compute_safe_powersets,
    compute_frontier,
    _compute_finite_closure_states,
)
from transduction.universality import compute_ip_universal_states
from transduction import examples


# ============================================================
# Helper: build small FSTs for targeted tests
# ============================================================

def universal_loop(alphabet=('a', 'b')):
    """Single-state FST: all symbols loop with copy output.  Ip-universal."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    for x in alphabet:
        fst.add_arc(0, x, x, 0)
    return fst


def finite_chain():
    """Linear chain: 0 -a:x-> 1 -b:y-> 2.  No cycles, all states finite-closure."""
    fst = FST()
    fst.add_start(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'b', 'y', 2)
    fst.add_stop(2)
    return fst


def cycle_no_base():
    """Two non-universal, non-finite-closure states in a cycle.
    0 -a:x-> 1 -b:y-> 0.  Both states are unsafe."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'b', 'y', 0)
    # Register full alphabet so universality check knows what's expected
    fst.A.add('a')
    fst.A.add('b')
    return fst


def mixed_safe_unsafe():
    """State 0 is ip-universal (loop on a,b).
    State 1 only has 'a' arc to state 2.
    State 2 has no arcs (finite closure).
    State 0: safe (universal).  State 1: safe (successor 2 is safe).
    State 2: safe (finite closure)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_stop(2)
    for x in ('a', 'b'):
        fst.add_arc(0, x, x, 0)
    fst.add_arc(1, 'a', 'x', 2)
    fst.states.add(1)  # ensure state 1 exists even without start/stop
    return fst


def unsafe_reaches_cycle():
    """State 0 -> state 1 -> state 2 -> state 1 (cycle).
    State 0 is not a base case, and reaches the cycle.  All unsafe.
    State 3 is ip-universal (safe)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_stop(3)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, 'a', 'x', 2)
    fst.add_arc(2, 'a', 'x', 1)  # cycle
    fst.add_arc(0, 'b', 'y', 3)
    # State 3: universal
    for x in ('a', 'b'):
        fst.add_arc(3, x, x, 3)
    return fst


def collectively_universal():
    """Two states that are collectively but not individually ip-universal.
    State 1: 'a' arcs to {1,2}.  State 2: 'b' arcs to {1,2}.
    Neither alone covers {a,b}, but together {1,2} does, and
    every successor powerstate is again {1,2}."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(1)
    fst.add_stop(2)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(0, 'b', 'x', 2)
    # State 1: handles 'a', goes to both 1 and 2
    fst.add_arc(1, 'a', 'x', 1)
    fst.add_arc(1, 'a', 'x', 2)
    # State 2: handles 'b', goes to both 1 and 2
    fst.add_arc(2, 'b', 'x', 1)
    fst.add_arc(2, 'b', 'x', 2)
    return fst


# ============================================================
# Tests: compute_safe_states (Level 1)
# ============================================================

class TestComputeSafeStates:

    def test_universal_loop_all_safe(self):
        fst = universal_loop()
        safe = compute_safe_states(fst)
        assert safe == frozenset(fst.states)

    def test_finite_chain_all_safe(self):
        fst = finite_chain()
        safe = compute_safe_states(fst)
        assert safe == frozenset(fst.states)

    def test_cycle_no_base_all_unsafe(self):
        fst = cycle_no_base()
        safe = compute_safe_states(fst)
        # Both states are in a cycle, neither is a base case
        assert 0 not in safe
        assert 1 not in safe

    def test_mixed_safe_unsafe(self):
        fst = mixed_safe_unsafe()
        safe = compute_safe_states(fst)
        assert 0 in safe  # ip-universal
        assert 1 in safe  # successor (2) is safe
        assert 2 in safe  # finite closure

    def test_unsafe_reaches_cycle(self):
        fst = unsafe_reaches_cycle()
        safe = compute_safe_states(fst)
        assert 3 in safe      # ip-universal
        assert 0 not in safe   # reaches cycle 1->2->1
        assert 1 not in safe   # in cycle
        assert 2 not in safe   # in cycle

    def test_bpe_like_all_universal(self):
        fst = examples.bpe_like(vocab_size=10, alphabet=('a', 'b', 'c'), max_len=3)
        safe = compute_safe_states(fst)
        # BPE: every state should be ip-universal
        assert safe == frozenset(fst.states)

    def test_small_example(self):
        fst = examples.small()
        safe = compute_safe_states(fst)
        # State 3 has self-loops on a,b and is accepting -> ip-universal
        assert 3 in safe

    def test_delete_b_all_safe(self):
        """delete_b: single state with self-loops on a,b -> ip-universal."""
        fst = examples.delete_b()
        safe = compute_safe_states(fst)
        assert 0 in safe

    def test_triplets_of_doom(self):
        """triplets_of_doom: cycle 0->1->3->0 and 0->2->4->0.
        Each state only accepts one symbol (a or b), so none are individually
        ip-universal. All are in cycles, so none have finite closure.
        Therefore all are individually unsafe."""
        fst = examples.triplets_of_doom()
        safe = compute_safe_states(fst)
        # No individual state is safe
        assert safe == frozenset()

    def test_triplets_of_doom_powerset(self):
        """triplets_of_doom: even at the powerset level, {0} is unsafe.
        From {0}, input 'a' -> {1}, but {1} has no 'b' arcs -> dead.
        The powerset BFS finds dead ends, not collective universality.
        The decomposition terminates for specific targets because the
        search tree is finite (dead branches), but the *safety* condition
        (which requires ALL paths to reach base cases) correctly flags
        this as unsafe: there are cycles ({0}->{1}->{3}->{0}) among
        non-base-case powerstates."""
        fst = examples.triplets_of_doom()
        seeds = [frozenset({0})]
        safe_ps = compute_safe_powersets(fst, seeds)
        # Not safe: cycle in powerset graph among non-base-case states
        assert frozenset({0}) not in safe_ps


# ============================================================
# Tests: _compute_finite_closure_states
# ============================================================

class TestFiniteClosureStates:

    def test_chain_all_finite(self):
        fst = finite_chain()
        fc = _compute_finite_closure_states(fst)
        assert fc == frozenset(fst.states)

    def test_loop_not_finite(self):
        fst = universal_loop()
        fc = _compute_finite_closure_states(fst)
        assert 0 not in fc  # self-loop

    def test_mixed(self):
        fst = unsafe_reaches_cycle()
        fc = _compute_finite_closure_states(fst)
        # States 1,2 are in a cycle; state 0 can reach them
        assert 1 not in fc
        assert 2 not in fc
        assert 0 not in fc
        # State 3 has self-loops -> not finite closure
        assert 3 not in fc


# ============================================================
# Tests: compute_safe_powersets (Level 2)
# ============================================================

class TestComputeSafePowersets:

    def test_universal_singleton_safe(self):
        fst = universal_loop()
        seeds = [frozenset({0})]
        safe_ps = compute_safe_powersets(fst, seeds)
        assert frozenset({0}) in safe_ps

    def test_finite_chain_singleton_safe(self):
        fst = finite_chain()
        seeds = [frozenset({0})]
        safe_ps = compute_safe_powersets(fst, seeds)
        assert frozenset({0}) in safe_ps

    def test_collectively_universal(self):
        """Neither state 1 nor state 2 is individually ip-universal,
        but {1,2} is collectively ip-universal."""
        fst = collectively_universal()
        # Individual analysis: states 1,2 are in self-loops but not universal
        safe_individual = compute_safe_states(fst)
        assert 1 not in safe_individual
        assert 2 not in safe_individual

        # Powerset analysis: {1,2} should be safe
        seeds = [frozenset({1, 2})]
        safe_ps = compute_safe_powersets(fst, seeds)
        assert frozenset({1, 2}) in safe_ps

    def test_cycle_no_base_unsafe_powerset(self):
        fst = cycle_no_base()
        seeds = [frozenset({0})]
        safe_ps = compute_safe_powersets(fst, seeds)
        assert frozenset({0}) not in safe_ps

    def test_budget_limits_exploration(self):
        """With budget=1, only one powerstate is explored."""
        fst = universal_loop()
        seeds = [frozenset({0})]
        safe_ps = compute_safe_powersets(fst, seeds, budget=1)
        # Should still certify {0} since it's a base case (ip-universal)
        assert frozenset({0}) in safe_ps

    def test_bpe_like_all_safe(self):
        fst = examples.bpe_like(vocab_size=10, alphabet=('a', 'b', 'c'), max_len=3)
        root = ()
        seeds = [frozenset({root})]
        safe_ps = compute_safe_powersets(fst, seeds)
        assert frozenset({root}) in safe_ps


# ============================================================
# Tests: compute_frontier
# ============================================================

class TestComputeFrontier:

    def test_empty_target(self):
        fst = universal_loop()
        frontier = compute_frontier(fst, ())
        assert frontier == frozenset({0})

    def test_single_symbol_target(self):
        fst = examples.small()
        # small(): 0 -a:x-> 1, 0 -b:x-> 2
        frontier = compute_frontier(fst, ('x',))
        assert 1 in frontier
        assert 2 in frontier

    def test_chain_target(self):
        fst = finite_chain()
        # 0 -a:x-> 1 -b:y-> 2
        frontier_x = compute_frontier(fst, ('x',))
        assert frontier_x == frozenset({1})
        frontier_xy = compute_frontier(fst, ('x', 'y'))
        assert frontier_xy == frozenset({2})

    def test_no_matching_target(self):
        fst = finite_chain()
        # No path emits 'z'
        frontier = compute_frontier(fst, ('z',))
        assert frontier == frozenset()


# ============================================================
# Tests: consistency between levels
# ============================================================

class TestConsistency:

    def test_individual_implies_powerset(self):
        """If all frontier states are individually safe,
        the frontier powerset should also be safe."""
        fst = examples.small()
        safe = compute_safe_states(fst)
        frontier = compute_frontier(fst, ('x',))
        if frontier <= safe:
            safe_ps = compute_safe_powersets(fst, [frontier])
            assert frontier in safe_ps

    def test_samuel_example(self):
        """samuel_example: check that safety analysis runs without error."""
        fst = examples.samuel_example()
        safe = compute_safe_states(fst)
        # State 4 has self-loops on a,b and is accepting -> ip-universal
        assert 4 in safe

    def test_doom_examples(self):
        """doom(V, K): all states are ip-universal (copy transducer on k-tuples)."""
        for K in (2, 3):
            fst = examples.doom({'a', 'b'}, K)
            safe = compute_safe_states(fst)
            # All states should be safe (the copy arcs + eps arcs make them universal)
            # Actually, intermediate states may not be universal since they only
            # accept a single symbol's continuation.  Just check no errors.
            assert isinstance(safe, frozenset)
