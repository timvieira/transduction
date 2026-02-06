"""
Tests for the lazy automaton framework.
"""
import pytest
from transduction.fsa import FSA, EPSILON
from transduction.lazy import (
    Lazy, EpsilonRemove, LazyDeterminize, StartAt, Renumber, LazyWrapper,
)


# Test machines
def make_simple():
    """Simple linear FSA: 1 -a-> 2 -b-> 3"""
    m = FSA()
    m.add_start(1)
    m.add_stop(3)
    m.add_arc(1, 'a', 2)
    m.add_arc(2, 'b', 3)
    return m


def make_epsilon():
    """FSA with epsilon transitions (original test_foo)"""
    m = FSA()
    m.add_start(1)
    m.add_stop(3)
    m.add_arc(1, EPSILON, 2)
    m.add_arc(2, 'x', 3)
    m.add_arc(3, EPSILON, 2)
    return m


def make_cycle():
    """FSA with cycle: 1 -a-> 2 -b-> 1"""
    m = FSA()
    m.add_start(1)
    m.add_stop(1)
    m.add_arc(1, 'a', 2)
    m.add_arc(2, 'b', 1)
    return m


def make_epsilon_chain():
    """Multiple epsilons in sequence"""
    m = FSA()
    m.add_start(1)
    m.add_stop(4)
    m.add_arc(1, EPSILON, 2)
    m.add_arc(2, EPSILON, 3)
    m.add_arc(3, 'a', 4)
    return m


def make_epsilon_cycle():
    """Epsilon cycle"""
    m = FSA()
    m.add_start(1)
    m.add_stop(3)
    m.add_arc(1, EPSILON, 2)
    m.add_arc(2, EPSILON, 1)
    m.add_arc(2, 'a', 3)
    return m


def make_nondeterministic():
    """NFA with multiple transitions on same symbol"""
    m = FSA()
    m.add_start(1)
    m.add_stop(2)
    m.add_stop(3)
    m.add_arc(1, 'a', 2)
    m.add_arc(1, 'a', 3)
    return m


def make_complex():
    """Complex NFA with epsilon, nondeterminism, cycles"""
    m = FSA()
    m.add_start(1)
    m.add_stop(4)
    m.add_arc(1, EPSILON, 2)
    m.add_arc(2, EPSILON, 3)
    m.add_arc(3, 'a', 3)
    m.add_arc(3, 'a', 4)
    m.add_arc(4, 'b', 3)
    return m


def make_universal():
    """Universal automaton accepting {a,b}*"""
    m = FSA()
    m.add_start(1)
    m.add_stop(1)
    m.add_arc(1, 'a', 1)
    m.add_arc(1, 'b', 1)
    return m


ALL_MACHINES = [
    pytest.param(make_simple, id='simple'),
    pytest.param(make_epsilon, id='epsilon'),
    pytest.param(make_cycle, id='cycle'),
    pytest.param(make_epsilon_chain, id='epsilon_chain'),
    pytest.param(make_epsilon_cycle, id='epsilon_cycle'),
    pytest.param(make_nondeterministic, id='nondeterministic'),
    pytest.param(make_complex, id='complex'),
    pytest.param(make_universal, id='universal'),
]


# Materialize tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_materialize_equals_original(make_fsa):
    """Materialized lazy FSA should equal the original."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    assert lazy.materialize().equal(m)


@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_materialize_preserves_language(make_fsa):
    """Materialized FSA should recognize same language."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    assert lazy.materialize().min().equal(m.min())


# EpsilonRemove tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_epsremove_preserves_language(make_fsa):
    """Epsilon removal should preserve language."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    eps_removed = lazy.epsremove().materialize()
    assert eps_removed.min().equal(m.min())


def test_epsilon_to_final_makes_source_final():
    """Epsilon path to final state makes source final."""
    m = FSA()
    m.add_start(1)
    m.add_stop(2)
    m.add_arc(1, EPSILON, 2)

    lazy = LazyWrapper(m)
    eps_removed = lazy.epsremove()
    assert eps_removed.is_final(1)


def test_closure_is_cached():
    """Verify closure is cached."""
    m = make_epsilon()
    lazy = LazyWrapper(m)
    eps_removed = EpsilonRemove(lazy)

    closure1 = eps_removed._closure(1)
    closure2 = eps_removed._closure(1)
    assert closure1 is closure2


def test_arcs_x_skips_epsilon():
    """arcs_x should skip epsilon arcs."""
    m = FSA()
    m.add_start(1)
    m.add_stop(2)
    m.add_arc(1, EPSILON, 2)
    m.add_arc(1, 'a', 2)

    lazy = LazyWrapper(m)
    eps_removed = lazy.epsremove()

    assert list(eps_removed.arcs_x(1, EPSILON)) == []
    assert len(list(eps_removed.arcs_x(1, 'a'))) > 0


# Determinize tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_det_preserves_language(make_fsa):
    """Determinization should preserve language."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    det = lazy.det().materialize()
    assert det.min().equal(m.min())


@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_det_produces_frozenset_states(make_fsa):
    """DFA states should be frozensets."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    det = lazy.det()

    for state in det.start():
        assert isinstance(state, frozenset)


def test_det_is_final_if_any_member_final():
    """Powerset state is final if any member is final."""
    m = make_nondeterministic()
    lazy = LazyWrapper(m)
    det = lazy.det()

    for state in det.start():
        for _, next_state in det.arcs(state):
            assert det.is_final(next_state)


# StartAt tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_start_at_changes_start(make_fsa):
    """start_at should change the start state."""
    m = make_fsa()
    lazy = LazyWrapper(m)

    # Pick a non-start state if available
    non_start = [s for s in m.states if s not in m.start]
    if non_start:
        new_start = non_start[0]
        from_new = lazy.start_at(new_start)
        starts = list(from_new.start())
        assert starts == [new_start]


def test_start_at_preserves_arcs():
    m = make_simple()
    lazy = LazyWrapper(m)
    from_2 = lazy.start_at(2)

    arcs = list(from_2.arcs(2))
    assert ('b', 3) in arcs


# Renumber tests

def test_renumber_to_integers():
    m = FSA()
    m.add_start('start')
    m.add_stop('end')
    m.add_arc('start', 'a', 'middle')
    m.add_arc('middle', 'b', 'end')

    lazy = LazyWrapper(m)
    renumbered = lazy.renumber().materialize()

    for state in renumbered.states:
        assert isinstance(state, int)
    assert renumbered.min().equal(m.min())


# accepts_universal tests

def test_universal_accepts_all():
    m = make_universal()
    lazy = LazyWrapper(m)
    assert lazy.accepts_universal(1, {'a', 'b'})


def test_not_universal_missing_arc():
    m = FSA()
    m.add_start(1)
    m.add_stop(1)
    m.add_arc(1, 'a', 1)

    lazy = LazyWrapper(m)
    assert not lazy.accepts_universal(1, {'a', 'b'})


def test_not_universal_non_final_reachable():
    m = FSA()
    m.add_start(1)
    m.add_stop(1)
    m.add_arc(1, 'a', 1)
    m.add_arc(1, 'b', 2)  # 2 is not final

    lazy = LazyWrapper(m)
    assert not lazy.accepts_universal(1, {'a', 'b'})


# Integration tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_epsremove_det_materialize(make_fsa):
    """Full pipeline: epsremove -> det -> materialize."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    result = lazy.epsremove().det().materialize()
    assert result.min().equal(m.min())


def test_max_steps_limits_exploration():
    """materialize(max_steps=N) should limit states explored."""
    m = make_epsilon()
    lazy = LazyWrapper(m)
    E = lazy.epsremove()
    assert E.materialize(max_steps=2).states == {1, 2}


def test_foo():
    m = FSA()
    m.add_start(1)
    m.add_stop(3)
    m.add_arc(1, '', 2)
    m.add_arc(2, 'x', 3)
    m.add_arc(3, '', 2)

    lazy = LazyWrapper(m)

    # sanity check for lazy wrapper
    assert lazy.materialize().equal(m)

    E = lazy.epsremove()

    assert E.materialize().equal(m)

    assert E.materialize(max_steps=2).states == {1, 2}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
