"""
Tests for the lazy automaton framework.

Tests cover:
- Lazy base class methods (materialize, det, epsremove, etc.)
- EpsilonRemove (epsilon closure, caching)
- LazyDeterminize (powerset construction)
- StartAt, Renumber wrappers
"""
import pytest
from transduction.fsa import FSA, EPSILON
from transduction.lazy import (
    Lazy, EpsilonRemove, LazyDeterminize, StartAt, Renumber, LazyWrapper,
)


class TestMaterialize:
    """Test Lazy.materialize()"""

    def test_simple_fsa(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        materialized = lazy.materialize()

        assert materialized.equal(m)

    def test_with_epsilon(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'x', 3)
        m.add_arc(3, EPSILON, 2)

        lazy = LazyWrapper(m)
        assert lazy.materialize().equal(m)

    def test_max_steps(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(4)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 3)
        m.add_arc(3, 'c', 4)

        lazy = LazyWrapper(m)
        partial = lazy.materialize(max_steps=2)

        assert 1 in partial.states
        assert 2 in partial.states
        assert 3 not in partial.states or 4 not in partial.states

    def test_cycle(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(1)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 1)

        lazy = LazyWrapper(m)
        materialized = lazy.materialize()

        assert materialized.equal(m)


class TestEpsilonRemove:
    """Test EpsilonRemove class."""

    def test_simple_epsilon(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'a', 3)

        lazy = LazyWrapper(m)
        eps_removed = lazy.epsremove()
        materialized = eps_removed.materialize()

        # Should accept 'a' starting from state 1
        assert 1 in materialized.start
        assert materialized.is_final(3)

    def test_epsilon_to_final(self):
        """Epsilon path to final state makes source final."""
        m = FSA()
        m.add_start(1)
        m.add_stop(2)
        m.add_arc(1, EPSILON, 2)

        lazy = LazyWrapper(m)
        eps_removed = lazy.epsremove()

        # State 1 should be final via epsilon closure
        assert eps_removed.is_final(1)

    def test_epsilon_chain(self):
        """Multiple epsilons in sequence."""
        m = FSA()
        m.add_start(1)
        m.add_stop(4)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, EPSILON, 3)
        m.add_arc(3, 'a', 4)

        lazy = LazyWrapper(m)
        eps_removed = lazy.epsremove()
        materialized = eps_removed.materialize()

        # Should be able to reach 4 from 1 via 'a'
        assert 1 in materialized.start
        assert materialized.is_final(4)

    def test_epsilon_cycle(self):
        """Epsilon cycle should not cause infinite loop."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, EPSILON, 1)  # cycle back
        m.add_arc(2, 'a', 3)

        lazy = LazyWrapper(m)
        eps_removed = lazy.epsremove()
        materialized = eps_removed.materialize()

        assert materialized.is_final(3)

    def test_closure_caching(self):
        """Verify closure is cached."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'a', 3)

        lazy = LazyWrapper(m)
        eps_removed = EpsilonRemove(lazy)

        # Access closure twice
        closure1 = eps_removed._closure(1)
        closure2 = eps_removed._closure(1)

        # Should be same object (cached)
        assert closure1 is closure2

    def test_arcs_x_skips_epsilon(self):
        """arcs_x should skip epsilon arcs."""
        m = FSA()
        m.add_start(1)
        m.add_stop(2)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(1, 'a', 2)

        lazy = LazyWrapper(m)
        eps_removed = lazy.epsremove()

        # arcs_x for epsilon should yield nothing
        eps_results = list(eps_removed.arcs_x(1, EPSILON))
        assert eps_results == []

        # arcs_x for 'a' should work
        a_results = list(eps_removed.arcs_x(1, 'a'))
        assert len(a_results) > 0


class TestLazyDeterminize:
    """Test LazyDeterminize class."""

    def test_already_deterministic(self):
        """DFA should be unchanged by determinization."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        det = lazy.det()
        materialized = det.materialize()

        # Should recognize same language
        assert materialized.min().equal(m.min())

    def test_nondeterministic(self):
        """NFA with multiple transitions on same symbol."""
        m = FSA()
        m.add_start(1)
        m.add_stop(2)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(1, 'a', 3)

        lazy = LazyWrapper(m)
        det = lazy.det()
        materialized = det.materialize()

        # Start state should be frozenset
        starts = list(det.start())
        assert len(starts) == 1
        assert isinstance(starts[0], frozenset)

        # Should still accept 'a'
        assert materialized.min().equal(m.min())

    def test_powerset_construction(self):
        """Verify powerset states are frozensets."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(1, 'a', 3)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        det = lazy.det()

        for state in det.start():
            assert isinstance(state, frozenset)
            for _, next_state in det.arcs(state):
                assert isinstance(next_state, frozenset)

    def test_det_with_epsilon(self):
        """Determinization should handle epsilon via epsremove."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'a', 3)

        lazy = LazyWrapper(m)
        det = lazy.det()
        materialized = det.materialize()

        # Should accept 'a'
        assert materialized.min().equal(m.min())

    def test_is_final_powerset(self):
        """Powerset state is final if any member is final."""
        m = FSA()
        m.add_start(1)
        m.add_stop(2)
        m.add_arc(1, 'a', 2)
        m.add_arc(1, 'a', 3)  # 3 is not final

        lazy = LazyWrapper(m)
        det = lazy.det()

        for state in det.start():
            for _, next_state in det.arcs(state):
                # next_state = frozenset({2, 3}), should be final (2 is final)
                assert det.is_final(next_state)


class TestStartAt:
    """Test StartAt wrapper."""

    def test_start_at_different_state(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        from_2 = lazy.start_at(2)

        starts = list(from_2.start())
        assert starts == [2]
        assert from_2.is_final(3)

    def test_start_at_preserves_arcs(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, 'a', 2)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        from_2 = lazy.start_at(2)

        arcs = list(from_2.arcs(2))
        assert ('b', 3) in arcs


class TestRenumber:
    """Test Renumber wrapper."""

    def test_renumber_states(self):
        m = FSA()
        m.add_start('start')
        m.add_stop('end')
        m.add_arc('start', 'a', 'middle')
        m.add_arc('middle', 'b', 'end')

        lazy = LazyWrapper(m)
        renumbered = lazy.renumber()
        materialized = renumbered.materialize()

        # States should be integers
        for state in materialized.states:
            assert isinstance(state, int)

        # Should recognize same language
        assert materialized.min().equal(m.min())


class TestAcceptsUniversal:
    """Test Lazy.accepts_universal()"""

    def test_universal_self_loop(self):
        """State with self-loop on all symbols is universal."""
        m = FSA()
        m.add_start(1)
        m.add_stop(1)
        m.add_arc(1, 'a', 1)
        m.add_arc(1, 'b', 1)

        lazy = LazyWrapper(m)
        assert lazy.accepts_universal(1, {'a', 'b'})

    def test_not_universal_missing_arc(self):
        """State missing an arc is not universal."""
        m = FSA()
        m.add_start(1)
        m.add_stop(1)
        m.add_arc(1, 'a', 1)
        # Missing 'b' arc

        lazy = LazyWrapper(m)
        assert not lazy.accepts_universal(1, {'a', 'b'})

    def test_not_universal_non_final(self):
        """Non-final reachable state means not universal."""
        m = FSA()
        m.add_start(1)
        m.add_stop(1)
        m.add_arc(1, 'a', 1)
        m.add_arc(1, 'b', 2)  # 2 is not final

        lazy = LazyWrapper(m)
        assert not lazy.accepts_universal(1, {'a', 'b'})

    def test_universal_multiple_states(self):
        """Universal with multiple reachable states."""
        m = FSA()
        m.add_start(1)
        m.add_stop(1)
        m.add_stop(2)
        m.add_arc(1, 'a', 2)
        m.add_arc(1, 'b', 1)
        m.add_arc(2, 'a', 1)
        m.add_arc(2, 'b', 2)

        lazy = LazyWrapper(m)
        assert lazy.accepts_universal(1, {'a', 'b'})


class TestLazyWrapper:
    """Test the LazyWrapper from lazy.py"""

    def test_wrapper_materialize(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'x', 3)
        m.add_arc(3, EPSILON, 2)

        lazy = LazyWrapper(m)
        assert lazy.materialize().equal(m)

    def test_wrapper_epsremove(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'x', 3)
        m.add_arc(3, EPSILON, 2)

        lazy = LazyWrapper(m)
        E = lazy.epsremove()
        assert E.materialize().equal(m)

    def test_wrapper_materialize_max_steps(self):
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'x', 3)
        m.add_arc(3, EPSILON, 2)

        lazy = LazyWrapper(m)
        E = lazy.epsremove()
        assert E.materialize(max_steps=2).states == {1, 2}


class TestIntegration:
    """Integration tests combining multiple lazy operations."""

    def test_det_then_materialize(self):
        """NFA -> det -> materialize should give equivalent DFA."""
        # NFA accepting a*b
        m = FSA()
        m.add_start(1)
        m.add_stop(2)
        m.add_arc(1, 'a', 1)
        m.add_arc(1, 'b', 2)

        lazy = LazyWrapper(m)
        dfa = lazy.det().materialize()

        assert dfa.min().equal(m.min())

    def test_epsremove_det_materialize(self):
        """NFA with eps -> epsremove -> det -> materialize."""
        m = FSA()
        m.add_start(1)
        m.add_stop(3)
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, 'a', 2)
        m.add_arc(2, 'b', 3)

        lazy = LazyWrapper(m)
        result = lazy.epsremove().det().materialize()

        assert result.min().equal(m.min())

    def test_complex_nfa(self):
        """Complex NFA with epsilon, nondeterminism, cycles."""
        m = FSA()
        m.add_start(1)
        m.add_stop(4)
        # Epsilon transitions
        m.add_arc(1, EPSILON, 2)
        m.add_arc(2, EPSILON, 3)
        # Nondeterminism
        m.add_arc(3, 'a', 3)
        m.add_arc(3, 'a', 4)
        # Cycle
        m.add_arc(4, 'b', 3)

        lazy = LazyWrapper(m)
        dfa = lazy.det().materialize()

        # Should accept same language
        assert dfa.min().equal(m.min())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
