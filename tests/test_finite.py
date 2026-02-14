import pytest
from transduction import (
    DecompositionResult, LazyNonrecursive, LazyIncremental, LazyPrecoverNFA,
    EagerNonrecursive, examples, Precover, FSA, EPSILON, PrioritizedLazyIncremental
)
from transduction.fst import FST


def assert_equal(have, want):
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


def _precover_factory(fst, **kwargs):
    """Precover.factory, ignoring kwargs it doesn't support (e.g., max_steps)."""
    return Precover.factory(fst)


IMPLEMENTATIONS = [
    pytest.param(EagerNonrecursive, id="eager_nonrecursive"),
    pytest.param(LazyNonrecursive, id="lazy_nonrecursive"),
    pytest.param(LazyIncremental, id="lazy_incremental"),
    pytest.param(_precover_factory, id="precover"),
    pytest.param(PrioritizedLazyIncremental, id="prioritized_lazy_incremental"),
]


@pytest.fixture(params=IMPLEMENTATIONS)
def impl(request):
    return request.param


def test_sdd1(impl):
    fst = examples.sdd1_fst()
    tmp = impl(fst)
    assert_equal(tmp(''), DecompositionResult({'a'}, set()))
    assert_equal(tmp('a'), DecompositionResult({'a'}, set()))
    assert_equal(tmp('aa'), DecompositionResult({'aa'}, set()))


def test_simple(impl):
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])
    tmp = impl(fst)
    assert_equal(tmp(''), DecompositionResult({''}, set()))
    assert_equal(tmp('a'), DecompositionResult({'1'}, set()))
    assert_equal(tmp('ab'), DecompositionResult({'12'}, set()))
    assert_equal(tmp('abc'), DecompositionResult({'123'}, set()))


def test_duplicate(impl):
    fst = examples.duplicate(set('12345'))
    tmp = impl(fst)
    assert_equal(tmp(''), DecompositionResult({''}, set()))
    assert_equal(tmp('1'), DecompositionResult({'1'}, set()))
    assert_equal(tmp('11'), DecompositionResult({'1'}, set()))
    assert_equal(tmp('1155'), DecompositionResult({'15'}, set()))
    assert_equal(tmp('115'), DecompositionResult({'15'}, set()))


def test_newspeak2(impl):
    n = examples.newspeak2()
    ba = DecompositionResult(
        {'bar', 'bax', 'baq', 'ban', 'bap', 'bau', 'bao', 'bay', 'bag', 'bae', 'bah',
         'bak', 'bai', 'bav', 'bac', 'bal', 'bam', 'bab', 'baz', 'baa', 'baf', 'bat',
         'bas', 'baj', 'baw'},
        {'ba'},
    )
    empty = DecompositionResult({''}, set())
    bad = DecompositionResult(set(), set())
    ungood = DecompositionResult({'bad', 'ungood'}, set())

    tmp = impl(n)
    assert_equal(tmp(''), empty)
    assert_equal(tmp('bad'), bad)
    assert_equal(tmp('ba'), ba)
    assert_equal(tmp('ungood'), ungood)


def test_samuel_example(impl):
    fst = examples.samuel_example()
    tmp = impl(fst)
    assert_equal(tmp('c'), DecompositionResult({'a'}, set()))


def test_delete_b(impl):
    fst = examples.delete_b()
    tmp = impl(fst, max_steps=30)
    assert_equal(tmp(''), DecompositionResult({''}, set()))
    assert_equal(tmp('b'), DecompositionResult(set(), set()))


def test_number_comma_separator(impl):
    import string
    fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))
    source_alphabet = fst.A - {EPSILON}
    tmp = impl(fst)
    assert_equal(tmp('1,| 2,| and 3'), DecompositionResult({'1, 2, and 3'}, set()))
    have = tmp('1,| 2,|')
    want = DecompositionResult(
        {'1, 2,' + x for x in source_alphabet if x not in '1234567890'},
        set(),
    )
    assert_equal(have, want)


def test_anbn(impl):
    fst = examples.anbn()
    tmp = impl(fst)
    assert_equal(tmp('b'), DecompositionResult({'aaa'}, {'a'}))
    assert_equal(tmp('c'), DecompositionResult(set(), {'aa'}))
    assert_equal(tmp('bb'), DecompositionResult({'aaa'}, set()))


def test_backticks_to_quote(impl):
    fst = examples.backticks_to_quote()
    tmp = impl(fst)
    assert_equal(tmp('b'), DecompositionResult({'a'}, set()))
    assert_equal(tmp('`'), DecompositionResult({'`a'}, {'`'}))
    assert_equal(tmp('"'), DecompositionResult({'``'}, set()))
    assert_equal(tmp('`b'), DecompositionResult({'`a'}, set()))


def run_test_finite(impl, fst, depth):
    """Compare impl against Precover reference for all target prefixes up to depth."""
    factory = impl(fst)
    reference = Precover.factory(fst)
    target_alphabet = fst.B - {EPSILON}

    def recurse(target, depth):
        if depth == 0:
            return
        assert_equal(factory(target), reference(target))
        for y in target_alphabet:
            ref_child = reference(target + y)
            q, r = ref_child.quotient, ref_child.remainder
            if isinstance(q, set): q = FSA.from_strings(q)
            if isinstance(r, set): r = FSA.from_strings(r)
            if q.trim().states or r.trim().states:
                recurse(target + y, depth - 1)

    recurse('', depth)


def test_trivial_fst(impl):
    """FST with start=stop, no arcs: accepts only empty string."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.B.update({'x', 'y'})
    factory = impl(fst)
    result = factory('')
    assert_equal(result, DecompositionResult({''}, set()))
    for y in fst.B - {EPSILON}:
        assert_equal(factory(y), DecompositionResult(set(), set()))


def test_unreachable_target_symbol(impl):
    """Target symbol in alphabet but unreachable produces empty Q and R."""
    fst = examples.replace([('1', 'a')])
    fst.B.add('z')
    factory = impl(fst)
    assert_equal(factory('z'), DecompositionResult(set(), set()))


def test_delayed_output_cycle(impl):
    """Functional FST with a:eps followed by eps:b cycle — no net I/O delay."""
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'a', '', 1)
    fst.add_arc(1, '', 'b', 0)
    run_test_finite(impl, fst, depth=5)


# ── Standalone tests (algorithm-specific) ─────────────────────────────────────

def test_lazy_precover_nfa():
    fst = examples.replace([('a', 'A'), ('b', 'B')])
    c = LazyPrecoverNFA(fst, 'AB')
    assert set(c.arcs((0, ''))) == {('a', (0, 'A'))}
    assert set(c.arcs((0, 'A'))) == {('b', (0, 'AB'))}
    assert set(c.arcs((0, 'AB'))) == {('a', (0, 'AB')), ('b', (0, 'AB'))}
    assert c.is_final((0, 'AB'))
    assert not c.is_final((0, 'A'))
    assert set(c.start()) == {(0, '')}


def test_delete_b_infinite_quotient():
    """Exact FSA comparison for delete_b 'AAA' target (infinite quotient)."""
    fst = examples.delete_b()
    a = FSA.lift('a')
    b = FSA.lift('b')
    bs = b.star()
    tmp = Precover.factory(fst)
    have = tmp('AAA')
    want = (bs * a * bs * a * bs * a).min()
    assert_equal(have, DecompositionResult(want, set()))


def test_delete_b_check_decomposition():
    """Enumeration algorithms on delete_b 'AAA': valid decomposition with max_steps."""
    fst = examples.delete_b()
    for alg in [EagerNonrecursive(fst, max_steps=30), LazyNonrecursive(fst, max_steps=30)]:
        have = alg('AAA')
        assert have.remainder == set()
        p = Precover(fst, 'AAA')
        p.check_decomposition(*have, skip_validity=True)


def test_prioritized_custom_heuristic():
    """Custom >> heuristic produces the same results (exhaustive search, just different order)."""
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c')])

    class ReverseAlpha:
        """Explore higher ord(symbol) first (lower = explored first)."""
        def __init__(self, score=0):
            self._score = score
        def __rshift__(self, symbol):
            return ReverseAlpha(-ord(symbol))
        def __lt__(self, other):
            return self._score < other._score

    tmp = PrioritizedLazyIncremental(fst, heuristic=ReverseAlpha())
    assert_equal(tmp(''), DecompositionResult({''}, set()))
    assert_equal(tmp('a'), DecompositionResult({'1'}, set()))
    assert_equal(tmp('ab'), DecompositionResult({'12'}, set()))
    assert_equal(tmp('abc'), DecompositionResult({'123'}, set()))


def test_prioritized_max_steps():
    """max_steps limits exploration, producing a subset of the full result."""
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])

    tmp_full = LazyIncremental(fst)
    tmp_limited = PrioritizedLazyIncremental(fst, max_steps=3)

    assert_equal(tmp_limited(''), DecompositionResult({''}, set()))

    have_limited = tmp_limited('a')
    have_full = tmp_full('a')
    assert have_limited.quotient <= have_full.quotient
    assert have_limited.remainder <= have_full.remainder


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
