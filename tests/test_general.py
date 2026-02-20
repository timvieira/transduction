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
from transduction.peekaboo_nonrecursive import Peekaboo as PeekabooNonrecursive
from transduction.peekaboo_incremental import PeekabooState
from transduction.peekaboo_dirty import DirtyPeekaboo

try:
    from transduction.rust_bridge import RustDecomp, RustDirtyState, RustDirtyPeekaboo
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from transduction.pynini_ops import PyniniNonrecursiveDecomp


def run_test(cls, fst, target, depth, verbosity=0):
    """Unified test runner: recursively checks decompose_next() against reference."""
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


def assert_equal_decomp_map(have, want):
    for y in have | want:
        assert have[y].quotient.equal(want[y].quotient)
        assert have[y].remainder.equal(want[y].remainder)


IMPLEMENTATIONS = [
    pytest.param(TruncatedIncrementalDFADecomp, id="truncated_incremental_dfa_decomp"),
    pytest.param(NonrecursiveDFADecomp, id="nonrecursive_dfa_decomp"),
    pytest.param(PeekabooState, id="peekaboo_incremental"),
    pytest.param(PeekabooNonrecursive, id="peekaboo_nonrecursive"),
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

IMPLEMENTATIONS.append(
    pytest.param(PyniniNonrecursiveDecomp, id="pynini_nonrecursive"),
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
    from transduction.util import timelimit
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


def test_scaled_newspeak_non_aui(impl):
    fst = examples.scaled_newspeak(n_patterns=3, alpha_size=6, n_partial=2)
    run_test(impl, fst, '', depth=3)


def test_layered_witnesses(impl):
    fst = examples.layered_witnesses()
    run_test(impl, fst, '', depth=5)


def test_doom_k5(impl):
    from transduction.util import timelimit
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
    fst = examples.small()
    state = impl(fst, '')
    # >> path
    via_rshift = state >> 'x'
    # decompose_next path (fresh instance)
    via_decompose = type(state)(fst, '')
    via_dn = via_decompose.decompose_next()['x']
    assert via_rshift.quotient.equal(via_dn.quotient)
    assert via_rshift.remainder.equal(via_dn.remainder)


def test_multiple_start_states(impl):
    """Multiple start states with disjoint inputs (infinite remainders)."""
    fst = FST()
    fst.add_start(0)
    fst.add_start(1)
    fst.add_stop(0)
    fst.add_stop(1)
    fst.add_arc(0, 'a', 'x', 0)
    fst.add_arc(1, 'b', 'x', 1)
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_multiple_start_states_shared_sink(impl):
    """Multiple start states merging into a shared sink (finite quotients)."""
    fst = FST()
    fst.add_start(0)
    fst.add_start(1)
    fst.add_stop(2)
    fst.add_arc(0, 'a', 'x', 2)   # start 0: 'a' -> shared sink
    fst.add_arc(1, 'b', 'x', 2)   # start 1: 'b' -> shared sink
    fst.add_arc(2, 'a', 'y', 2)   # sink accepts full source alphabet
    fst.add_arc(2, 'b', 'y', 2)
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


def test_togglecase(impl):
    fst = examples.togglecase()
    run_test(impl, fst, '', depth=2)


def test_lowercase(impl):
    fst = examples.lowercase()
    run_test(impl, fst, '', depth=2)


def test_mystery1(impl):
    fst = examples.mystery1()
    run_test(impl, fst, '', depth=7)


def test_mystery3(impl):
    fst = examples.mystery3()
    run_test(impl, fst, '', depth=5)


def test_mystery4(impl):
    fst = examples.mystery4()
    run_test(impl, fst, '', depth=5)


def test_mystery5(impl):
    fst = examples.mystery5()
    run_test(impl, fst, '', depth=5)


def test_mystery7(impl):
    fst = examples.mystery7()
    run_test(impl, fst, '', depth=7)


def test_mystery8(impl):
    fst = examples.mystery8()
    run_test(impl, fst, '', depth=7)


def test_productive_eps_chain(impl):
    """Functional FST with a long productive input-epsilon chain."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', '', 1)     # consume 'a', no output yet
    fst.add_arc(1, '', 'x', 2)     # eps chain: output 'x'
    fst.add_arc(2, '', 'y', 3)     # eps chain: output 'y'
    fst.add_arc(3, '', 'z', 4)     # eps chain: output 'z'
    fst.add_arc(4, 'b', 'w', 0)    # consume 'b', output 'w'
    # Functional: (ab)* -> (xyzw)*
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_nonproductive_eps_cycle(impl):
    """Functional FST with a nonproductive input-epsilon cycle (eps/eps self-loop)."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', 'x', 1)
    fst.add_arc(1, '', '', 1)      # nonproductive eps-input cycle (eps/eps)
    fst.add_arc(1, 'b', 'y', 0)
    # Functional: (ab)* -> (xy)*; the eps cycle doesn't change the output
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_delayed_output_cycle(impl):
    """Functional FST with a:eps followed by eps:b cycle — no net I/O delay."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', '', 1)     # consume input, no output
    fst.add_arc(1, '', 'b', 0)     # no input, produce output
    # Functional: a* -> b*; each cycle consumes one 'a' and emits one 'b'
    run_test(impl, fst, '', depth=5, verbosity=0)


def test_multichar_output_symbols(impl):
    """FST with multi-character output symbols — tuples must keep symbols distinct."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    # source 'a' produces multi-char output 'ab', source 'b' produces 'cd'
    fst.add_arc(0, 'a', 'ab', 0)
    fst.add_arc(0, 'b', 'cd', 0)
    # With string buffers, 'ab'+'cd' == 'a'+'bcd' — decomposition would be wrong.
    # With tuple buffers, ('ab','cd') != ('a','bcd') — symbols are kept distinct.
    run_test(impl, fst, (), depth=3, verbosity=0)


def test_oov_target_symbol(impl):
    """Constructing with a target symbol not in fst.B raises ValueError."""
    fst = examples.replace([('1', 'a'), ('2', 'b')])
    with pytest.raises(ValueError, match="Out of vocabulary"):
        impl(fst, 'z')


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
