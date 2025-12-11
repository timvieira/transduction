from transduction import (
    PrecoverDecomp, LazyNonrecursive, BuggyLazyRecursive, LazyRecursive, LazyPrecoverNFA,
    EagerNonrecursive, examples, Precover
)


def test_sdd1():
    fst = examples.sdd1_fst()

    tmp = BuggyLazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())    # this is a weird case and it differs from the other ones!
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = EagerNonrecursive(fst)
    print(tmp(''))
    assert tmp('') == PrecoverDecomp({'a'}, set())   # this is a weird case but this is correct
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = LazyNonrecursive(fst)
    assert tmp('') == PrecoverDecomp({'a'}, set())   # this is a weird case but this is correct
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())

    tmp = LazyRecursive(fst)
    assert tmp('') == PrecoverDecomp({''}, set())   # this is a weird case and this is less correct
    assert tmp('a') == PrecoverDecomp({'a'}, set())
    assert tmp('aa') == PrecoverDecomp({'aa'}, set())


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


def test_samuel_example():
    fst = examples.samuel_example()
    target = 'c'
    tmp = LazyNonrecursive(fst)
    have = tmp(target)
    print('decomp:', have)
    print(have)
    assert have == ({'a'}, set())

    # this algorithm has an expected failure
    tmp = EagerNonrecursive(fst)
    have = tmp(target)
    print('decomp:', have)
    print(have)
    assert have == ({'a'}, set()), have

    # this algorithm has an expected failure
    tmp = BuggyLazyRecursive(fst)
    have = tmp(target)
    print('decomp:', have)
    print(have)
    assert have == ({'ab', 'aa'}, {'a'}), have

    # this algorithm fixes BuggyLazyRecursive's expected failure
    tmp = LazyRecursive(fst)
    have = tmp(target)
    print('decomp:', have)
    print(have)
    assert have == ({'a'}, set()), have


def test_number_comma_separator():
    import string
    digits = {str(i) for i in range(10)}
    fst = examples.number_comma_separator(set(string.printable) - set('\t\n\r\x0b\x0c'))

    tmp = LazyNonrecursive(fst)
    assert tmp('1,| 2,| and 3') == ({'1, 2, and 3'}, set())
    have = tmp('1,| 2,|')
    want = ({'1, 2,' + x for x in tmp.source_alphabet if x not in '1234567890'}, set())
    assert have == want

    target = '1,| 2,'
    for y in tmp.target_alphabet:
        Q,R = tmp(target+y)
        assert len(R) == 0
        if y == '|' and target.endswith(','):
            assert len(Q) == len(tmp.source_alphabet) - len(digits), [target, y, Q]
        else:
            assert len(Q) <= 1, [target, y, Q]
        if len(Q) > 0:
            print(repr(y), Q)

    tmp = BuggyLazyRecursive(fst, max_steps=100)
    assert tmp('1,| 2,| and 3') == ({'1, 2, and 3'}, set())
    have = tmp('1,| 2,|')
    want = ({'1, 2,' + x for x in tmp.source_alphabet if x not in '1234567890'}, set())
    assert have == want

    target = '1,| 2,'
    for y in tmp.target_alphabet:
        Q,R = tmp(target+y)
        assert len(R) == 0
        if y == '|' and target.endswith(','):
            assert len(Q) == len(tmp.source_alphabet) - len(digits), [target, y, Q]
        else:
            assert len(Q) <= 1, [target, y, Q]
        if len(Q) > 0:
            print(repr(y), Q)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
