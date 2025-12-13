from transduction import (
    PrecoverDecomp, LazyNonrecursive, BuggyLazyRecursive, LazyRecursive, LazyPrecoverNFA,
    EagerNonrecursive, examples, Precover, FSA
)


def assert_equal(have, want):
    assert have.quotient.equal(want.quotient), [have.quotient.min(), want.quotient]
    assert have.remainder.equal(want.remainder), [have.remainder.min(), want.remainder]


def test_sdd1():
    fst = examples.sdd1_fst()

    tmp = BuggyLazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({'a'}, set())
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = EagerNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({'a'}, set())
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = LazyNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({'a'}, set())
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = LazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({'a'}, set())
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = lambda target: Precover(fst, target)
    assert_equal(tmp(''), PrecoverDecomp({'a'}, set()))
    assert_equal(tmp('a'), PrecoverDecomp({'a'}, set()))
    assert_equal(tmp('aa'), PrecoverDecomp({'aa'}, set()))


def test_delete_b():
    # this example has an infinite quotient for non-empty target strings, but
    # always an empty remainder
    fst = examples.delete_b()

    a = FSA.lift('a')
    b = FSA.lift('b')
    bs = b.star()

    tmp = lambda target: Precover(fst, target)
    assert_equal(tmp(''), PrecoverDecomp({''}, set()))
    assert_equal(tmp('b'), PrecoverDecomp(set(), set()))
    have = tmp('AAA')
    want = (bs * a * bs * a * bs * a).min()
    assert_equal(have, PrecoverDecomp(want, set()))

    algs = [
        BuggyLazyRecursive(fst, max_steps=30),
        EagerNonrecursive(fst, max_steps=30),
        LazyNonrecursive(fst, max_steps=30),
        LazyRecursive(fst, max_steps=30),
    ]

    for tmp in algs:
        assert tmp('') == PrecoverDecomp({''}, set())
        assert tmp('b') == PrecoverDecomp(set(), set())

        target = 'AAA'
        have = tmp(target)
        assert have.remainder == set()
        p = Precover(fst, target)
        p.check_decomposition(*have, skip_validity=True)


def test_simple():
    fst = examples.replace([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e')])

    tmp = LazyNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('a') == PrecoverDecomp({'1'}, set())
    assert tmp('ab') == PrecoverDecomp({'12'}, set())
    assert tmp('abc') == PrecoverDecomp({'123'}, set())

    tmp = BuggyLazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('a') == PrecoverDecomp({'1'}, set())
    assert tmp('ab') == PrecoverDecomp({'12'}, set())
    assert tmp('abc') == PrecoverDecomp({'123'}, set())

    tmp = EagerNonrecursive(fst, max_steps=50)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('a') == PrecoverDecomp({'1'}, set())
    assert tmp('ab') == PrecoverDecomp({'12'}, set())
    assert tmp('abc') == PrecoverDecomp({'123'}, set())

    tmp = LazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('a') == PrecoverDecomp({'1'}, set())
    assert tmp('ab') == PrecoverDecomp({'12'}, set())
    assert tmp('abc') == PrecoverDecomp({'123'}, set())

    tmp = lambda target: Precover(fst, target)
    assert_equal(tmp(''), PrecoverDecomp({''}, set()))
    assert_equal(tmp('a'), PrecoverDecomp({'1'}, set()))
    assert_equal(tmp('ab'), PrecoverDecomp({'12'}, set()))
    assert_equal(tmp('abc'), PrecoverDecomp({'123'}, set()))


def test_lazy_precover_nfa():
    fst = examples.replace([('a', 'A'), ('b', 'B')])

    c = LazyPrecoverNFA(fst, 'AB')

    assert set(c.arcs((0, ''))) == {('a', (0, 'A'))}
    assert set(c.arcs((0, 'A'))) == {('b', (0, 'AB'))}
    assert set(c.arcs((0, 'AB'))) == {('a', (0, 'AB')), ('b', (0, 'AB'))}

    assert c.is_final((0, 'AB'))
    assert not c.is_final((0, 'A'))
    assert set(c.start()) == {(0, '')}


def test_duplicate():
    fst = examples.duplicate(set('12345'))

    tmp = BuggyLazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('1') == PrecoverDecomp({'1'}, set())
    assert tmp('11') == PrecoverDecomp({'1'}, set())
    assert tmp('1155') == PrecoverDecomp({'15'}, set())
    assert tmp('115') == PrecoverDecomp({'15'}, set())

    tmp = EagerNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('1') == PrecoverDecomp({'1'}, set())
    assert tmp('11') == PrecoverDecomp({'1'}, set())
    assert tmp('1155') == PrecoverDecomp({'15'}, set())
    assert tmp('115') == PrecoverDecomp({'15'}, set())

    tmp = LazyNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('1') == PrecoverDecomp({'1'}, set())
    assert tmp('11') == PrecoverDecomp({'1'}, set())
    assert tmp('1155') == PrecoverDecomp({'15'}, set())
    assert tmp('115') == PrecoverDecomp({'15'}, set())

    tmp = LazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())
    assert tmp('1') == PrecoverDecomp({'1'}, set())
    assert tmp('11') == PrecoverDecomp({'1'}, set())
    assert tmp('1155') == PrecoverDecomp({'15'}, set())
    assert tmp('115') == PrecoverDecomp({'15'}, set())

    tmp = lambda target: Precover(fst, target)
    assert_equal(tmp(''), PrecoverDecomp({''}, set()))
    assert_equal(tmp('1'), PrecoverDecomp({'1'}, set()))
    assert_equal(tmp('11'), PrecoverDecomp({'1'}, set()))
    assert_equal(tmp('1155'), PrecoverDecomp({'15'}, set()))
    assert_equal(tmp('115'), PrecoverDecomp({'15'}, set()))


def test_newspeak2():

    n = examples.newspeak2()

    ba = PrecoverDecomp(
        {'bar', 'bax', 'baq', 'ban', 'bap', 'bau', 'bao', 'bay', 'bag', 'bae', 'bah',
         'bak', 'bai', 'bav', 'bac', 'bal', 'bam', 'bab', 'baz', 'baa', 'baf', 'bat',
         'bas', 'baj', 'baw'},
        {'ba'},
    )
    empty = PrecoverDecomp({''}, set())
    bad = PrecoverDecomp(set(), set())
    ungood = PrecoverDecomp({'bad', 'ungood'}, set())

    tmp = BuggyLazyRecursive(n)
    assert tmp('') == empty
    assert tmp('bad') == bad
    assert tmp('ba') == ba
    assert tmp('ungood') == ungood, tmp('ungood')

    tmp = EagerNonrecursive(n)
    assert tmp('') == empty
    assert tmp('bad') ==  bad
    assert tmp('ba') == ba
    assert tmp('ungood') == ungood, tmp('ungood')

    tmp = LazyNonrecursive(n)
    assert tmp('') == empty
    assert tmp('bad') == bad
    assert tmp('ba') == ba
    assert tmp('ungood') == ungood, tmp('ungood')

    tmp = LazyRecursive(n)
    assert tmp('') == empty
    assert tmp('bad') == bad
    assert tmp('ba') == ba
    assert tmp('ungood') == ungood, tmp('ungood')

    tmp = lambda target: Precover(n, target)
    assert_equal(tmp(''), empty)
    assert_equal(tmp('bad'), bad)
    assert_equal(tmp('ba'), ba)
    assert_equal(tmp('ungood'), ungood)


def test_samuel_example():
    fst = examples.samuel_example()
    target = 'c'
    tmp = LazyNonrecursive(fst)
    have = tmp(target)
    assert have == ({'a'}, set())

    tmp = EagerNonrecursive(fst)
    have = tmp(target)
    assert have == ({'a'}, set()), have

    # this algorithm has an expected failure
    tmp = BuggyLazyRecursive(fst)
    have = tmp(target)
    assert have == ({'ab', 'aa'}, {'a'}), have
    Precover(fst, target).check_decomposition(*have, throw=False)

    # this algorithm fixes BuggyLazyRecursive's expected failure
    tmp = LazyRecursive(fst)
    have = tmp(target)
    assert have == ({'a'}, set()), have

    tmp = lambda target: Precover(fst, target)
    have = tmp(target)
    assert_equal(have, PrecoverDecomp({'a'}, set()))


def test_number_comma_separator():
    import string
    digits = {str(i) for i in range(10)}
    fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))

    tmp = LazyNonrecursive(fst)
    assert tmp('1,| 2,| and 3') == ({'1, 2, and 3'}, set())
    have = tmp('1,| 2,|')
    want = ({'1, 2,' + x for x in tmp.source_alphabet if x not in '1234567890'}, set())
    assert have == want

    #target = '1,| 2,'
    #for y in tmp.target_alphabet:
    #    Q,R = tmp(target+y)
    #    assert len(R) == 0
    #    if y == '|' and target.endswith(','):
    #        assert len(Q) == len(tmp.source_alphabet) - len(digits), [target, y, Q]
    #    else:
    #        assert len(Q) <= 1, [target, y, Q]
    #    if len(Q) > 0:
    #        print(repr(y), Q)

    tmp = BuggyLazyRecursive(fst, max_steps=100)
    assert tmp('1,| 2,| and 3') == ({'1, 2, and 3'}, set())
    have = tmp('1,| 2,|')
    want = ({'1, 2,' + x for x in tmp.source_alphabet if x not in '1234567890'}, set())
    assert have == want

    #target = '1,| 2,'
    #for y in tmp.target_alphabet:
    #    Q,R = tmp(target+y)
    #    assert len(R) == 0
    #    if y == '|' and target.endswith(','):
    #        assert len(Q) == len(tmp.source_alphabet) - len(digits), [target, y, Q]
    #    else:
    #        assert len(Q) <= 1, [target, y, Q]
    #    if len(Q) > 0:
    #        print(repr(y), Q)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
