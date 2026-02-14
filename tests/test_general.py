"""
Tests for general-case decomposition algorithms against the reference Precover.

This test suite exercises FSTs whose quotient/remainder languages may be infinite.
Only algorithms that handle the general case are included here — i.e., those
that operate over automata states and use target-buffer truncation to guarantee
termination.

Algorithms excluded (finite-language only — they lack truncation and diverge
on infinite quotients):
  - LazyIncremental: enumerates source strings; universality check diverges.
"""

import pytest
from transduction import examples, EPSILON, Precover
from transduction.fst import FST
from transduction.dfa_decomp_nonrecursive import NonrecursiveDFADecomp
from transduction.dfa_decomp_incremental_truncated import TruncatedIncrementalDFADecomp
from transduction.token_decompose import TokenDecompose
from transduction.universality import check_all_input_universal
from transduction.peekaboo_nonrecursive import Peekaboo as PeekabooNonrecursive
from transduction.peekaboo_incremental import PeekabooState
from transduction.peekaboo_dirty import DirtyPeekaboo

try:
    from transduction.rust_bridge import RustDecomp, RustDirtyState, RustDirtyPeekaboo
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def _token_decompose_or_skip(fst, target=''):
    if not check_all_input_universal(fst):
        pytest.skip("TokenDecompose requires all_input_universal")
    return TokenDecompose(fst, target)


def run_test(cls, fst, target, depth, verbosity=0):
    """Unified test runner: recursively checks decompose_next() against reference."""
    reference = Precover.factory(fst)
    target_alphabet = fst.B - {EPSILON}

    def recurse(target, depth, state):
        if depth == 0:
            return
        want = {y: reference(target + y) for y in target_alphabet}
        have = state.decompose_next()
        assert_equal_decomp_map(have, want)
        for y in want:
            if verbosity > 0:
                print('>', repr(target + y))
            q = want[y].quotient.trim()
            r = want[y].remainder.trim()
            if q.states or r.states:
                recurse(target + y, depth - 1, have[y])

    recurse(target, depth, cls(fst, target))


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


IMPLEMENTATIONS = [
    pytest.param(TruncatedIncrementalDFADecomp, id="truncated_incremental_dfa_decomp"),
    pytest.param(NonrecursiveDFADecomp, id="nonrecursive_dfa_decomp"),
    pytest.param(PeekabooState, id="peekaboo_incremental"),
    pytest.param(PeekabooNonrecursive, id="peekaboo_nonrecursive"),
    pytest.param(_token_decompose_or_skip, id="token_decompose"),
    pytest.param(DirtyPeekaboo, id="dirty_peekaboo"),
]

if HAS_RUST:
    IMPLEMENTATIONS.append(
        pytest.param(RustDecomp, id="rust_decomp"),
    )
    IMPLEMENTATIONS.append(
        pytest.param(RustDirtyState, id="rust_dirty_state"),
    )
    IMPLEMENTATIONS.append(
        pytest.param(RustDirtyPeekaboo, id="rust_dirty_peekaboo"),
    )


@pytest.fixture(params=IMPLEMENTATIONS)
def impl(request):
    return request.param


def test_abc(impl):
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    run_test(impl, fst, '', depth=4)


def test_delete_b(impl):
    fst = examples.delete_b()
    run_test(impl, fst, '', depth=10)


def test_samuel(impl):
    fst = examples.samuel_example()
    run_test(impl, fst, '', depth=5)


def test_small(impl):
    fst = examples.small()
    run_test(impl, fst, '', depth=5)


def test_sdd1(impl):
    fst = examples.sdd1_fst()
    run_test(impl, fst, '', depth=5)


def test_duplicate(impl):
    fst = examples.duplicate(set('12345'))
    run_test(impl, fst, '', depth=5)


def test_number_comma_separator(impl):
    fst = examples.number_comma_separator({'a', ',', ' ', '0'}, Digit={'0'})
    run_test(impl, fst, '', depth=4, verbosity=0)
    run_test(impl, fst, '0,| 0,', depth=1, verbosity=0)
    run_test(impl, fst, '0,| 0,|', depth=1, verbosity=0)


def test_newspeak2(impl):
    fst = examples.newspeak2()
    run_test(impl, fst, '', depth=1)
    run_test(impl, fst, 'ba', depth=1)
    run_test(impl, fst, 'bad', depth=1)


def test_lookahead(impl):
    fst = examples.lookahead()
    run_test(impl, fst, '', depth=6, verbosity=0)


def test_weird_copy(impl):
    fst = examples.weird_copy()
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_triplets_of_doom(impl):
    from arsenal import timelimit
    fst = examples.triplets_of_doom()
    with timelimit(5):
        run_test(impl, fst, '', depth=13, verbosity=0)


def test_infinite_quotient(impl):
    fst = examples.infinite_quotient()
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_parity(impl):
    fst = examples.parity({'a', 'b'})
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_gated_universal(impl):
    fst = examples.gated_universal()
    run_test(impl, fst, '', depth=5)


def test_complementary_halves(impl):
    fst = examples.complementary_halves()
    run_test(impl, fst, '', depth=5)


def test_shrinking_nonuniversal(impl):
    fst = examples.shrinking_nonuniversal()
    run_test(impl, fst, '', depth=5)


def test_scaled_newspeak(impl):
    fst = examples.scaled_newspeak(n_patterns=3, alpha_size=6)
    run_test(impl, fst, '', depth=3)


def test_layered_witnesses(impl):
    fst = examples.layered_witnesses()
    run_test(impl, fst, '', depth=5)


def test_doom_k5(impl):
    from arsenal import timelimit
    fst = examples.doom({'a', 'b'}, K=5)
    with timelimit(5):
        run_test(impl, fst, '', depth=10, verbosity=0)


def test_mystery2(impl):
    fst = examples.mystery2()
    run_test(impl, fst, '', depth=7, verbosity=0)


def test_infinite_quotient2(impl):
    fst = examples.infinite_quotient2()
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_rshift_chain(impl):
    """>> chain must produce same Q/R as decompose_next() chain."""
    from transduction.base import IncrementalDecomposition
    fst = examples.small()
    state = impl(fst, '')
    if not isinstance(state, IncrementalDecomposition):
        pytest.skip("non-incremental implementation")
    # >> path
    via_rshift = state >> 'x'
    # decompose_next path (fresh instance)
    via_decompose = type(state)(fst, '')
    via_dn = via_decompose.decompose_next()['x']
    assert via_rshift.quotient.equal(via_dn.quotient)
    assert via_rshift.remainder.equal(via_dn.remainder)


def test_multiple_start_states(impl):
    fst = FST()
    fst.add_start(0)
    fst.add_start(1)
    fst.add_stop(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(1, 'b', 'x', 1)
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_cross_validate_reference(impl):
    """Cross-validate: Precover.check_decomposition confirms Q/R correctness."""
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])
    reference = Precover.factory(fst)
    state = impl(fst, '')
    children = state.decompose_next()
    for y in (fst.B - {EPSILON}):
        Q = children[y].quotient.trim()
        R = children[y].remainder.trim()
        if Q.states or R.states:
            assert reference(y).check_decomposition(
                set(Q.language()), set(R.language()), throw=True
            )


def test_unreachable_target_symbol(impl):
    """Target symbol in alphabet but unreachable produces empty Q and R."""
    fst = examples.replace([('1', 'a')])
    fst.B.add('z')  # in alphabet but no arc produces it
    state = impl(fst, '')
    result = state.decompose_next()
    assert result['z'].quotient.trim().states == set()
    assert result['z'].remainder.trim().states == set()


def test_trivial_fst(impl):
    """FST with start=stop, no arcs: accepts only empty string."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.B.update({'x', 'y'})  # need target alphabet
    state = impl(fst, '')
    result = state.decompose_next()
    for y in result:
        assert result[y].quotient.trim().states == set()
        assert result[y].remainder.trim().states == set()


def test_anbn(impl):
    fst = examples.anbn()
    run_test(impl, fst, '', depth=5)


def test_backticks_to_quote(impl):
    fst = examples.backticks_to_quote()
    run_test(impl, fst, '', depth=5)


def test_parity_copy(impl):
    fst = examples.parity_copy()
    run_test(impl, fst, '', depth=5)


def test_consume_raises_on_double_use():
    """TruncatedIncrementalDFADecomp: double decompose_next() or >> raises RuntimeError."""
    fst = examples.replace([('1', 'a'), ('2', 'b')])
    # decompose_next then decompose_next
    s1 = TruncatedIncrementalDFADecomp(fst, '')
    s1.decompose_next()
    with pytest.raises(RuntimeError):
        s1.decompose_next()
    # >> then >>
    s2 = TruncatedIncrementalDFADecomp(fst, '')
    s2 >> 'a'
    with pytest.raises(RuntimeError):
        s2 >> 'b'
    # decompose_next then >>
    s3 = TruncatedIncrementalDFADecomp(fst, '')
    s3.decompose_next()
    with pytest.raises(RuntimeError):
        s3 >> 'a'
