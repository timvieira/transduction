"""
Tests for the lazy automaton framework.
"""
import pytest
from transduction.fsa import FSA, EPSILON
from transduction.lazy import (
    Cached, EpsilonRemove, LazyWrapper,
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


# Cache tests

@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_cache_preserves_language(make_fsa):
    """Cached lazy FSA should recognize the same language."""
    m = make_fsa()
    lazy = LazyWrapper(m)
    cached = lazy.det().cache()
    assert cached.materialize().min().equal(m.min())


@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_cache_arcs_returns_same_object(make_fsa):
    """Repeated arcs() calls should return the same cached list."""
    m = make_fsa()
    det = LazyWrapper(m).det().cache()
    for state in det.start():
        first = det.arcs(state)
        second = det.arcs(state)
        assert first is second


@pytest.mark.parametrize('make_fsa', ALL_MACHINES)
def test_cache_is_final_returns_same_value(make_fsa):
    """is_final() should return consistent cached results."""
    m = make_fsa()
    det = LazyWrapper(m).det()
    cached = det.cache()
    for state in det.start():
        assert cached.is_final(state) == det.is_final(state)


def test_cache_on_nfa():
    """Cache should work on an NFA (before determinization)."""
    m = make_nondeterministic()
    lazy = LazyWrapper(m)
    cached = lazy.cache()
    assert cached.materialize().min().equal(m.min())


def test_cache_on_epsremove():
    """Cache should work after epsilon removal."""
    m = make_epsilon()
    lazy = LazyWrapper(m)
    cached = lazy.epsremove().cache()
    assert cached.materialize().min().equal(m.min())


def test_cache_chained():
    """epsremove -> det -> cache should preserve language."""
    m = make_complex()
    lazy = LazyWrapper(m)
    cached = lazy.epsremove().det().cache()
    assert cached.materialize().min().equal(m.min())


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


# ── Coverage: Lazy.arcs_x default implementation (lines 64-67) ───────────────

def test_lazy_base_arcs_x_default():
    """Lazy.arcs_x default filters arcs() by label (with deprecation warning)."""
    import warnings
    from transduction.lazy import Lazy

    class MinimalLazy(Lazy):
        """Only implements arcs/start/is_final — arcs_x falls through to default."""
        def __init__(self, fsa):
            self.fsa = fsa
        def start(self):
            return iter(self.fsa.start)
        def is_final(self, i):
            return self.fsa.is_final(i)
        def arcs(self, i):
            return self.fsa.arcs(i)

    m = make_simple()
    lazy = MinimalLazy(m)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Match: covers the yield branch (X == x is True)
        assert list(lazy.arcs_x(1, 'a')) == [2]
        # No match: covers the filter-skip branch (X == x is False)
        assert list(lazy.arcs_x(1, 'z')) == []


# ── Coverage: Lazy.min (line 88) ─────────────────────────────────────────────

def test_lazy_min():
    """Lazy.min() materializes then minimizes."""
    m = make_simple()
    lazy = LazyWrapper(m)
    minimized = lazy.min()
    assert minimized.equal(m.min())


# ── Coverage: LazyDeterminize.arcs_x match (line 206, branch 205->206) ──────

def test_det_arcs_x_match():
    """LazyDeterminize.arcs_x yields result when input matches."""
    m = make_simple()
    det = LazyWrapper(m).det()
    [start] = list(det.start())
    results = list(det.arcs_x(start, 'a'))
    assert len(results) == 1
    assert isinstance(results[0], frozenset)


def test_det_arcs_x_no_match():
    """LazyDeterminize.arcs_x returns nothing for unrecognized input."""
    m = make_simple()
    det = LazyWrapper(m).det()
    [start] = list(det.start())
    assert list(det.arcs_x(start, 'z')) == []


# ── Coverage: Renumber.arcs_x (lines 269-270) ───────────────────────────────

def test_renumber_arcs_x():
    """Renumber.arcs_x yields renumbered destinations."""
    m = make_simple()
    lazy = LazyWrapper(m)
    renum = lazy.renumber()
    [start] = list(renum.start())
    dests = list(renum.arcs_x(start, 'a'))
    assert len(dests) == 1
    assert isinstance(dests[0], int)


# ── Coverage: Cached.is_final cache hit (branch 226->228) ───────────────────

def test_cached_is_final_cache_hit():
    """Second is_final call returns cached result."""
    m = make_simple()
    cached = LazyWrapper(m).cache()
    # First call populates cache
    result1 = cached.is_final(1)
    # Second call hits cache
    result2 = cached.is_final(1)
    assert result1 == result2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
