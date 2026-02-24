"""Shared test helpers for decomposition algorithm tests."""

from transduction import Precover, EPSILON
from transduction.fsa import FSA


def assert_equal_decomp(have, want):
    """Compare two DecompositionResults, handling both set and FSA quotient/remainder."""
    hq, hr = have.quotient, have.remainder
    wq, wr = want.quotient, want.remainder
    if isinstance(hq, (set, frozenset)):
        hq = FSA.from_strings(hq)
    if isinstance(wq, (set, frozenset)):
        wq = FSA.from_strings(wq)
    if isinstance(hr, (set, frozenset)):
        hr = FSA.from_strings(hr)
    if isinstance(wr, (set, frozenset)):
        wr = FSA.from_strings(wr)
    assert hq.equal(wq), [hq.min(), wq.min()]
    assert hr.equal(wr), [hr.min(), wr.min()]


def assert_equal_decomp_map(have, want):
    """Assert two {symbol: DecompositionResult} dicts are equal."""
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


def run_decompose_next_test(cls, fst, target, depth, verbosity=0):
    """Recursively check decompose_next() against the reference Precover."""
    reference = Precover.factory(fst)
    target_alphabet = fst.B - {EPSILON}
    target = tuple(target)

    def recurse(target, depth, state):
        if depth == 0:
            return
        want = {y: reference(target + (y,)) for y in target_alphabet}
        have = state.decompose_next()
        assert_equal_decomp_map(have, want)
        for y in want:
            if verbosity > 0:
                print('>', repr(target + (y,)))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                recurse(target + (y,), depth - 1, have[y])

    recurse(target, depth, cls(fst, target))


def run_factory_test(impl, fst, depth):
    """Compare a factory-style impl against Precover for all target prefixes up to depth."""
    factory = impl(fst)
    reference = Precover.factory(fst)
    target_alphabet = fst.B - {EPSILON}

    def recurse(target, depth):
        if depth == 0:
            return
        assert_equal_decomp(factory(target), reference(target))
        for y in target_alphabet:
            ref_child = reference(target + y)
            q, r = ref_child.quotient, ref_child.remainder
            if isinstance(q, set): q = FSA.from_strings(q)
            if isinstance(r, set): r = FSA.from_strings(r)
            if q.trim().states or r.trim().states:
                recurse(target + y, depth - 1)

    recurse('', depth)
